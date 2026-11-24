"""``WebFetchTool``: read a URL and return its readable, safe text.

The most common web need for a coding agent is "fetch this URL as plain
text" — documentation, config, API responses — without spinning up a whole
browser. This tool does exactly that over plain ``urllib`` (no extra
dependency), with several guardrails baked in:

* **SSRF defence (fail closed)** — by default a private / loopback / link-
  local / reserved destination is rejected (resolved IP addresses are
  checked; an explicit :attr:`allowed_hosts` allowlist short-circuits the IP
  check for trusted hosts). Only ``http`` / ``https`` are accepted.
* **Bounded body** — reads at most ``max_bytes`` and returns at most
  ``max_chars`` (tail-biased truncation), so a huge page never floods the
  agent's context.
* **Text extraction** — HTML is stripped (scripts/styles/tags, entity-decode)
  into readable prose; plain text / JSON is returned as-is.
* **Secret redaction** — everything returned is scrubbed through
  ``redact_secrets`` before reaching the conversation.

Only GET is supported (no request bodies), redirects are capped, and the
default ``fetcher`` is injectable so tests never touch the network.
"""

from __future__ import annotations

import html
import ipaddress
import re
import socket
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlsplit

from kinetic_sdk.security.redact import redact_secrets
from kinetic_sdk.tool.base import Tool, ToolResult

DEFAULT_MAX_BYTES = 1_000_000
DEFAULT_MAX_CHARS = 30_000
DEFAULT_TIMEOUT = 15.0


class WebFetchError(ValueError):
    """The URL is unsafe or the fetch failed (surfaced as a ToolResult error)."""


@dataclass
class WebFetchResponse:
    """What the (possibly injected) fetcher returns for one GET."""

    status: int
    text: str
    final_url: str
    content_type: str | None = None


#: ``fetcher(url, max_bytes, timeout) -> WebFetchResponse``. Injectable so
#: tests can substitute a fake without any network.
WebFetcher = Callable[[str, int, float], WebFetchResponse]


def _default_fetcher(url: str, max_bytes: int, timeout: float) -> WebFetchResponse:
    """Minimal GET via ``urllib``: capped body + timeout (redirects handled)."""
    request = urllib.request.Request(url, headers={"User-Agent": "kinetic-sdk/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - caller already ran assert_public_settable
        status = int(response.status)
        content_type = response.headers.get("Content-Type")
        raw = response.read(max_bytes + 1)
        payload = raw[:max_bytes]
        text = payload.decode(response.charset or "utf-8", errors="replace")
    return WebFetchResponse(
        status=status,
        text=text,
        final_url=str(request.full_url),
        content_type=content_type,
    )


#: Private/reserved networks an SSRF guard must never reach.
_PRIVATE_NETWORKS = (
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
)


def _is_unsafe_ip(ip: str) -> bool:
    try:
        address = ipaddress.ip_address(ip)
    except ValueError:
        return True  # unparseable -> fail closed
    if isinstance(address, ipaddress.IPv6Address) and (mapped := address.ipv4_mapped):
        address = mapped
    if address.version == 4 and any(address in network for network in _PRIVATE_NETWORKS):
        return True
    return address.is_private or address.is_loopback or address.is_link_local or address.is_reserved


def assert_public_settable(url: str, allowed_hosts: tuple[str, ...]) -> None:
    """Reject URLs that are not http(s) or resolve to a private/reserved host.

    Raises :class:`WebFetchError` when the destination is unsafe or cannot be
    resolved. An explicit *allowed_hosts* suffix short-circuits the DNS/IP
    check for hosts callers explicitly trust.
    """
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise WebFetchError(f"only http(s) URLs are allowed, got {parts.scheme!r}")
    host = parts.hostname
    if not host:
        raise WebFetchError(f"URL has no host: {url!r}")
    if allowed_hosts and any(
        host == allowed or host.endswith("." + allowed) for allowed in allowed_hosts
    ):
        return
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise WebFetchError(f"could not resolve host {host!r}: {exc}") from exc
    for info in infos:
        ip = str(info[4][0])
        if _is_unsafe_ip(ip):
            raise WebFetchError(
                f"refusing to fetch {url!r}: resolves to private/reserved "
                f"address {ip!r} (add it to WebFetchTool(allowed_hosts=...) "
                "to trust it explicitly)"
            )


def _strip_html(text: str) -> str:
    """Cheap HTML -> readable text (no dependency): drop scripts/styles/tags."""
    if "<" not in text:
        return text
    text = re.sub(r"<script\b.*?</script>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style\b.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"[ \t\r\f\v\n]+", " ", text).strip()


class WebFetchTool(Tool):
    """Fetch a URL and return it as readable, redacted, size-capped text.

    Args:
        allowed_hosts: Host suffixes trusted without the DNS/IP SSRF check
            (e.g. ``("raw.githubusercontent.com",)``). Empty (default) relies
            on the private/reserved-IP rejection only.
        max_bytes: Hard cap on the downloaded body.
        max_chars: Cap on returned output characters (tail-biased).
        timeout: Per-request timeout in seconds.
        fetcher: Injectable GET implementation (tests). Defaults to
            :func:`_default_fetcher`.
    """

    name: str = "web_fetch"
    description: str = (
        "Fetch a single http(s) URL and return its readable plain text "
        "(HTML is stripped, output is size-capped). Use to read docs, config, "
        "API responses or repository files without opening a browser. "
        "Private/loopback destinations are refused unless allowed."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The http(s) URL to fetch."},
            "max_chars": {
                "type": "integer",
                "description": "Optional per-call output cap (default: tool max_chars).",
            },
        },
        "required": ["url"],
    }

    thread_safe = True

    def __init__(
        self,
        *,
        allowed_hosts: tuple[str, ...] = (),
        max_bytes: int = DEFAULT_MAX_BYTES,
        max_chars: int = DEFAULT_MAX_CHARS,
        timeout: float = DEFAULT_TIMEOUT,
        fetcher: WebFetcher | None = None,
    ) -> None:
        if max_bytes <= 0 or max_chars < 200 or timeout <= 0:
            raise ValueError("max_bytes/timeout must be positive; max_chars >= 200")
        for host in allowed_hosts:
            if not isinstance(host, str) or not host or "/" in host:
                raise ValueError(f"invalid allowed_host entry: {host!r}")
        self._allowed_hosts = tuple(allowed_hosts)
        self._max_bytes = max_bytes
        self._max_chars = max_chars
        self._timeout = timeout
        self._fetcher: WebFetcher = fetcher if fetcher is not None else _default_fetcher

    def execute(self, url: str, max_chars: int | None = None, **_: Any) -> ToolResult:  # type: ignore[override]
        """Fetch *url* and return its readable text (SSRF-guarded)."""
        if not isinstance(url, str) or not url.strip():
            return ToolResult(error="url must be a non-empty string")
        try:
            assert_public_settable(url, self._allowed_hosts)
        except WebFetchError as exc:
            return ToolResult(error=f"web_fetch refused: {exc}")
        limit = self._max_chars if max_chars is None else max(200, int(max_chars))
        started = time.monotonic()
        try:
            response = self._fetcher(url, self._max_bytes, self._timeout)
        except Exception as exc:  # noqa: BLE001 - network errors become ToolResult
            return ToolResult(
                error=f"web_fetch failed: {type(exc).__name__}: {exc}",
                metadata={"url": url},
            )
        text = _strip_html(response.text)
        text = redact_secrets(text)
        truncated = len(text) > limit
        if truncated:
            text = text[-limit:]
        meta = {
            "url": url,
            "fetched_url": response.final_url or url,
            "status": response.status,
            "content_type": response.content_type,
            "truncated": truncated,
            "duration_seconds": round(time.monotonic() - started, 3),
        }
        if response.status >= 400:
            return ToolResult(error=f"HTTP {response.status}", output=text, metadata=meta)
        return ToolResult(output=text or "(empty body)", metadata=meta)
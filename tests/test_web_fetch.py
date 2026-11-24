"""Tests for WebFetchTool (SSRF guard + text extraction + redaction + caps)."""

from __future__ import annotations

from urllib.request import Request

import pytest

from kinetic_sdk.web import WebFetchResponse, WebFetchTool
from kinetic_sdk.web.tool import WebFetchError, _SSRFRedirectHandler


def _fake_fetcher(payload: str, status: int = 200, content_type: str = "text/html"):
    def fetch(url: str, max_bytes: int, timeout: float) -> WebFetchResponse:
        return WebFetchResponse(status=status, text=payload, final_url=url, content_type=content_type)

    return fetch


def test_strips_html_and_redacts_secrets() -> None:
    secret = "sk-" + "x" * 30
    body = (
        "<html><head><script>alert('x')</script><style>p{color:red}</style></head>"
        f"<body><h1>Docs</h1><p>api token = {secret} plain prose</p></body></html>"
    )
    tool = WebFetchTool(
        allowed_hosts=("docs.example.com",),
        fetcher=_fake_fetcher(body),
    )
    result = tool.execute(url="https://docs.example.com/page")
    assert not result.is_error
    out = result.output
    assert "Docs" in out
    assert "plain prose" in out
    assert "script" not in out and "alert" not in out
    assert secret not in out
    assert "[REDACTED]" in out
    assert result.metadata["status"] == 200


def test_tail_biased_truncation() -> None:
    tool = WebFetchTool(
        allowed_hosts=("x.example",),
        fetcher=_fake_fetcher("A" * 1000),
        max_chars=200,
    )
    result = tool.execute(url="http://x.example/p")
    assert result.metadata["truncated"] is True
    assert len(result.output) == 200


def test_rejects_private_and_loopback_by_default() -> None:
    tool = WebFetchTool()
    r = tool.execute(url="http://127.0.0.1/admin")
    assert r.is_error and "refusing" in r.error
    r2 = tool.execute(url="http://localhost/x")
    assert r2.is_error and "refusing" in r2.error


def test_rejects_non_http_scheme() -> None:
    tool = WebFetchTool()
    r = tool.execute(url="file:///etc/passwd")
    assert r.is_error and "only http(s)" in r.error
    r2 = tool.execute(url="ftp://example.com/x")
    assert r2.is_error


def test_redirect_handler_rejects_private_target() -> None:
    # A public page redirecting to a private/cloud-metadata host must be cut.
    handler = _SSRFRedirectHandler(())
    with pytest.raises(WebFetchError, match="redirect refused"):
        handler.redirect_request(
            Request("http://example.com/page"), None, 302, "Found", {}, "http://169.254.169.254/latest/meta-data"
        )


def test_allowed_host_short_circuits_ssrf_check() -> None:
    # No DNS needed: the allowlist trusts the host explicitly.
    tool = WebFetchTool(allowed_hosts=("raw.github.com",), fetcher=_fake_fetcher("ok:" + "a" * 400))
    r = tool.execute(url="https://raw.github.com/f/x.txt")
    assert not r.is_error
    assert r.output.startswith("ok:")


def test_http_error_status_is_error() -> None:
    tool = WebFetchTool(allowed_hosts=("x.example",), fetcher=_fake_fetcher("gone", status=404))
    r = tool.execute(url="http://x.example/404")
    assert r.is_error and "404" in r.error


def test_fetcher_exception_maps_to_error() -> None:
    def boom(url, max_bytes, timeout):
        raise TimeoutError("slow")

    tool = WebFetchTool(allowed_hosts=("x.example",), fetcher=boom)
    r = tool.execute(url="http://x.example/p")
    assert r.is_error and "TimeoutError" in r.error
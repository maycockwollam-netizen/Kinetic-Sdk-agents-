"""Secret redaction for logs and event payloads.

Tool inputs/outputs flow through the audit log and the event bus, and a model
may innocently embed a credential in either. :func:`redact_secrets` scrubs
GitHub, OpenAI-style, and AWS access tokens; Bearer tokens and JWTs; Slack
tokens; Google API keys; PEM private-key blocks; credentials embedded in URLs;
and common secret environment-variable and assignment forms before anything is
persisted or published. The goal is catching these common cases, not perfect
detection — defence in depth, not a guarantee that every secret is recognised.
"""

from __future__ import annotations

import re
from typing import Any

#: Placeholder substituted for anything recognised as a secret.
REDACTED = "[REDACTED]"

#: Well-known credential prefixes: GitHub tokens (ghp_/ghu_/gho_/ghs_/ghr_/
#: github_pat_), sk-* style API keys, AWS access key IDs.
_KNOWN_TOKEN_RE = re.compile(
    r"\b(?:ghp|ghu|gho|ghs|ghr)_[A-Za-z0-9]{20,}"
    r"|github_pat_[A-Za-z0-9_]{20,}"
    r"|\bsk-[A-Za-z0-9_\-]{20,}"
    r"|\bAKIA[0-9A-Z]{16}\b"
)

#: Heuristic: a sensitive keyword followed by a long unbroken token, e.g.
#: ``api_key = "..."`` or ``token: abc123...``. Longer keywords come first so
#: ``api_key`` wins over a bare ``key`` match.
_KEYWORD_TOKEN_RE = re.compile(
    r"(?i)\b(api[_-]?key|password|passwd|secret|token|key)"
    r"(\s*[:=]\s*|\s+)"
    r"(['\"]?)"
    r"[A-Za-z0-9_\-/+]{20,}"
    r"(['\"]?)"
)

#: Bearer tokens and standalone three-part JWTs. Requiring ten characters in
#: every JWT part avoids treating ordinary version strings (for example 1.2.3)
#: as credentials.
_BEARER_JWT_RE = re.compile(
    r"\bBearer\s+[A-Za-z0-9_\-\.]{20,}"
    r"|\b[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"
)

#: Slack bot, user, app, configuration, refresh, and service tokens.
_SLACK_TOKEN_RE = re.compile(r"\bxox[bpaors]-[A-Za-z0-9\-]{10,}")

#: Google API keys begin with ``AIza`` followed by exactly 35 base64url chars.
_GOOGLE_API_KEY_RE = re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")

#: PEM private-key blocks may span multiple lines. Non-greedy matching keeps
#: adjacent blocks separate.
_PEM_KEY_RE = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
    re.DOTALL,
)

#: Credentials in URLs; only the password (group 2) is sensitive.
_URL_CREDENTIAL_RE = re.compile(r"(://[^:/\s]+:)([^@\s]+)(@)")

#: Common secret-bearing environment variables. Values are redacted regardless
#: of length because environment assignment syntax is explicit.
_ENV_SECRET_RE = re.compile(
    r"(?i)\b((?:AWS|GCP|AZURE)?_?(?:SECRET|ACCESS_KEY|PRIVATE_KEY|TOKEN|"
    r"PASSWORD)[A-Z_]*)\s*=\s*(\S+)"
)

#: Explicit short secret assignments. Unlike the broad keyword heuristic above,
#: this requires ``:`` or ``=`` and intentionally excludes ambiguous ``key``
#: and ``token`` labels.
_SHORT_ASSIGNED_SECRET_RE = re.compile(
    r"(?i)\b(password|passwd|secret|api[_-]?key)"
    r"(\s*[:=]\s*)"
    r"(['\"]?)"
    r"([^\s'\"]{4,19})"
    r"(['\"]?)"
)


def redact_secrets(text: str) -> str:
    """Return *text* with recognised secrets replaced by ``[REDACTED]``."""
    if not text:
        return text

    def _keyword_sub(match: re.Match[str]) -> str:
        return f"{match.group(1)}{match.group(2)}{match.group(3)}{REDACTED}{match.group(4)}"

    def _url_credential_sub(match: re.Match[str]) -> str:
        return f"{match.group(1)}{REDACTED}{match.group(3)}"

    def _env_secret_sub(match: re.Match[str]) -> str:
        return f"{match.group(1)}={REDACTED}"

    def _short_assigned_secret_sub(match: re.Match[str]) -> str:
        return f"{match.group(1)}{match.group(2)}{match.group(3)}{REDACTED}{match.group(5)}"

    text = _PEM_KEY_RE.sub(REDACTED, text)
    text = _BEARER_JWT_RE.sub(REDACTED, text)
    text = _SLACK_TOKEN_RE.sub(REDACTED, text)
    text = _GOOGLE_API_KEY_RE.sub(REDACTED, text)
    text = _KNOWN_TOKEN_RE.sub(REDACTED, text)
    text = _URL_CREDENTIAL_RE.sub(_url_credential_sub, text)
    text = _KEYWORD_TOKEN_RE.sub(_keyword_sub, text)
    text = _ENV_SECRET_RE.sub(_env_secret_sub, text)
    return _SHORT_ASSIGNED_SECRET_RE.sub(_short_assigned_secret_sub, text)


def redact_value(value: Any) -> Any:
    """Recursively redact secrets inside a JSON-like structure.

    Strings are scrubbed with :func:`redact_secrets`; dicts and lists are
    walked (dict keys are left intact — they are field names, not values);
    everything else is returned unchanged.
    """
    if isinstance(value, str):
        return redact_secrets(value)
    if isinstance(value, dict):
        return {k: redact_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_value(v) for v in value]
    return value

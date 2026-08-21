"""Secret redaction for logs and event payloads.

Tool inputs/outputs flow through the audit log and the event bus, and a model
may innocently embed a credential in either. :func:`redact_secrets` scrubs the
common shapes before anything is persisted or published. The goal is catching
the common cases, not perfect detection — defence in depth, not a guarantee.
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


def redact_secrets(text: str) -> str:
    """Return *text* with recognised secrets replaced by ``[REDACTED]``."""
    if not text:
        return text

    def _keyword_sub(match: re.Match[str]) -> str:
        return f"{match.group(1)}{match.group(2)}{match.group(3)}{REDACTED}{match.group(4)}"

    return _KEYWORD_TOKEN_RE.sub(_keyword_sub, _KNOWN_TOKEN_RE.sub(REDACTED, text))


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

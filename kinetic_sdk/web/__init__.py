"""Web access tools: safe, bounded fetching of http(s) URLs."""

from __future__ import annotations

from kinetic_sdk.web.tool import (
    WebFetchError,
    WebFetchResponse,
    WebFetchTool,
    assert_public_settable,
)

__all__ = [
    "WebFetchTool",
    "WebFetchError",
    "WebFetchResponse",
    "assert_public_settable",
]
"""Tool package: abstract tool interface used by the agent loop."""

from kinetic_sdk.tool.base import (
    Tool,
    ToolCapability,
    ToolExecutionPolicy,
    ToolFailureCategory,
    ToolResult,
    ToolRiskLevel,
)
from kinetic_sdk.tool.cache import CachedTool, ToolCache, tool_cache_key
from kinetic_sdk.tool.validation import validate_tool_input

__all__ = [
    "Tool",
    "ToolCapability",
    "ToolExecutionPolicy",
    "ToolFailureCategory",
    "ToolResult",
    "ToolRiskLevel",
    "CachedTool",
    "ToolCache",
    "tool_cache_key",
    "validate_tool_input",
]

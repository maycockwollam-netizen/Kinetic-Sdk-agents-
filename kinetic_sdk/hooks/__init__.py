"""Hooks package: lifecycle interception points for the agent loop."""

from kinetic_sdk.hooks.base import Hook, HookContext, HookPoint, HookResult
from kinetic_sdk.hooks.registry import HookRegistry

__all__ = [
    "Hook",
    "HookContext",
    "HookPoint",
    "HookRegistry",
    "HookResult",
]

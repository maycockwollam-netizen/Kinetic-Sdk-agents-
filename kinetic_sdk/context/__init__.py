"""Context package: context-window management (Stage 2)."""

from kinetic_sdk.context.injection import InjectionGuard
from kinetic_sdk.context.manager import (
    ContextBudget,
    ContextManager,
    ContextSummarizer,
    LLMContextSummarizer,
    NoopContextManager,
    ProgressiveSummarizer,
    SimpleTruncateContextManager,
    StructuredSummary,
    SummarizingContextManager,
    ToolOutputCompressor,
    estimate_tokens,
)
from kinetic_sdk.context.tokens import ProviderTokenCounter, TiktokenCounter

__all__ = [
    "ContextManager",
    "ContextBudget",
    "ContextSummarizer",
    "LLMContextSummarizer",
    "NoopContextManager",
    "ProgressiveSummarizer",
    "StructuredSummary",
    "SimpleTruncateContextManager",
    "SummarizingContextManager",
    "ProviderTokenCounter",
    "TiktokenCounter",
    "ToolOutputCompressor",
    "InjectionGuard",
    "estimate_tokens",
]

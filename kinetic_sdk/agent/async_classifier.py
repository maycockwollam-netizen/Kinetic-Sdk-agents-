"""Async task classifier for FLASH/MAX routing in the async agent loop.

This mirrors :mod:`kinetic_sdk.agent.classifier` with awaitable
``classify`` methods so :class:`~kinetic_sdk.agent.async_agent.AsyncAgent`
never blocks the event loop on the classification call. The same rules
hold: the classifier uses a cheap, separate model; the real provider/model
name never leaks — only the alias ``kinetic-classifier-v1`` surfaces; any
failure falls back to COMPLEX (route to MAX, the safe side).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from kinetic_sdk.agent.classifier import (
    Classification,
    LiteLLMClassifier,
    TaskComplexity,
)
from kinetic_sdk.agent.modes import AgentMode
from kinetic_sdk.secret.registry import SecretRegistry
from kinetic_sdk.secret.value import SecretValue

logger = logging.getLogger(__name__)


class AsyncTaskClassifier(ABC):
    """Async interface for task-complexity classifiers.

    Same contract as :class:`~kinetic_sdk.agent.classifier.TaskClassifier`:
    implementations MUST use a model distinct from the main task LLM and
    MUST NOT expose the real provider/model name publicly.
    """

    #: Internal, provider-agnostic alias for the classifier model.
    alias: str = "kinetic-classifier-v1"

    @abstractmethod
    async def classify(self, task: str) -> Classification:
        """Classify *task* into a recommended :class:`AgentMode`.

        Implementations should favour MAX when unsure — running a complex
        task in FLASH mode is the worse failure mode.
        """


class AsyncDefaultClassifier(AsyncTaskClassifier):
    """Deterministic async placeholder: always routes to MAX.

    Conservative on purpose, same as the sync
    :class:`~kinetic_sdk.agent.classifier.DefaultClassifier`; used as the
    offline fallback and in tests that don't care about real routing.
    """

    async def classify(self, task: str) -> Classification:
        return Classification(
            complexity=TaskComplexity.COMPLEX,
            mode=AgentMode.MAX,
            confidence=1.0,
            rationale="default",
        )


class AsyncLiteLLMClassifier(AsyncTaskClassifier):
    """Async model-backed classifier using a cheap LiteLLM-reachable model.

    The async twin of
    :class:`~kinetic_sdk.agent.classifier.LiteLLMClassifier`: same private
    provider config (never logged, never in events or rationale), same
    one-word SIMPLE/COMPLEX prompt, same COMPLEX fallback on any failure.
    The only difference is that the underlying client is an
    :class:`~kinetic_sdk.llm.client.AsyncLLMClient` whose ``chat`` is
    awaited instead of called.
    """

    #: Real provider model string. Used only in the private API call path.
    _MODEL = LiteLLMClassifier._MODEL
    #: Real provider base URL. Used only in the private API call path.
    _API_BASE = LiteLLMClassifier._API_BASE
    #: Environment variable read for the API key.
    _API_KEY_ENV = LiteLLMClassifier._API_KEY_ENV
    #: Small token budget: the model only needs to emit one word.
    _MAX_TOKENS = LiteLLMClassifier._MAX_TOKENS

    def __init__(
        self,
        client: Any | None = None,
        api_key: str | SecretValue | None = None,
        summary: str | None = None,
        secrets: SecretRegistry | None = None,
    ) -> None:
        """Create the async classifier.

        Args:
            client: Optional pre-built
                :class:`~kinetic_sdk.llm.client.AsyncLLMClient` (used by
                tests to inject a mock). When omitted a real
                :class:`~kinetic_sdk.llm.async_client.AsyncLiteLLMClient` is
                built from the private provider config and the resolved key.
            api_key: API key as a plain string (wrapped automatically) or a
                :class:`SecretValue`. ``None`` resolves it from ``secrets``
                (default: the ``OPENHANDS_API_KEY`` environment variable).
            summary: Optional short summary of the conversation history
                forwarded to the classifier for context.
            secrets: Optional :class:`SecretRegistry` for key resolution.
        """
        self._summary = summary
        if api_key is None:
            registry = secrets if secrets is not None else SecretRegistry()
            self._api_key = registry.resolve(self._API_KEY_ENV, required=False)
        elif isinstance(api_key, SecretValue):
            self._api_key = api_key
        else:
            self._api_key = SecretValue(api_key)
        if client is not None:
            self._client = client
        else:
            # Lazy import so importing this module never forces litellm.
            from kinetic_sdk.llm.async_client import AsyncLiteLLMClient

            self._client = AsyncLiteLLMClient(
                model=self._MODEL,
                api_key=self._api_key,
                api_base=self._API_BASE,
                max_tokens=self._MAX_TOKENS,
            )

    @property
    def model(self) -> str:
        """Public model name. Always the alias, never the real provider name."""
        return self.alias

    async def classify(self, task: str) -> Classification:
        """Classify *task*, falling back to COMPLEX on any failure."""
        prompt = self._build_prompt(task)
        try:
            response = await self._client.chat(
                messages=prompt, max_tokens=self._MAX_TOKENS
            )
            answer = (response.content or "").strip().upper()
        except Exception as exc:  # noqa: BLE001 - safe-side fallback
            logger.warning(
                "%s classification call failed (falling back to COMPLEX): %s",
                self.alias,
                type(exc).__name__,
            )
            return self._fallback(reason="classifier_error")

        complexity = LiteLLMClassifier._parse(answer)
        if complexity is TaskComplexity.SIMPLE:
            return Classification(
                complexity=complexity,
                mode=complexity.to_mode(),
                confidence=0.8,
                rationale="model:SIMPLE",
            )
        return Classification(
            complexity=complexity,
            mode=complexity.to_mode(),
            confidence=0.9,
            rationale="model:COMPLEX",
        )

    def _build_prompt(self, task: str) -> list[dict[str, Any]]:
        """Build the one-shot user/system messages sent to the classifier.

        Same prompt as the sync classifier's (kept in sync deliberately —
        the wording is part of the routing behaviour).
        """
        instructions = (
            "You are a task-complexity classifier. Read the user's task and "
            "decide if it is SIMPLE or COMPLEX. Reply with exactly ONE word, "
            "either SIMPLE or COMPLEX, and nothing else. "
            "SIMPLE = a single trivial step (e.g. a short factual answer or one "
            "small edit). COMPLEX = multi-step reasoning, multiple tools, "
            "debugging, refactoring, or planning. When unsure, reply COMPLEX."
        )
        user_body = f"Task:\n{task}"
        if self._summary:
            user_body += f"\n\nConversation so far (summary):\n{self._summary}"
        return [
            {"role": "system", "content": instructions},
            {"role": "user", "content": user_body},
        ]

    def _fallback(self, reason: str) -> Classification:
        return Classification(
            complexity=TaskComplexity.COMPLEX,
            mode=AgentMode.MAX,
            confidence=0.0,
            rationale=reason,
        )

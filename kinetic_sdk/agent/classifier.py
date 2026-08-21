"""Task classifier for FLASH/MAX routing.

The classifier decides whether a task is SIMPLE (-> FLASH mode) or COMPLEX
(-> MAX mode) *before* the agent loop starts. Importantly it uses a cheap,
separate model - never the main task model - and the underlying provider/model
name is hidden behind an internal alias (e.g. ``kinetic-classifier-v1``) so it
does not leak into logs or public configuration.

Two implementations ship:

* :class:`DefaultClassifier` - a deterministic stub that always routes to MAX
  (kept as the conservative fallback for tests / offline use).
* :class:`LiteLLMClassifier` - the real, model-backed classifier. It drives a
  cheap model through :class:`kinetic_sdk.llm.client.LiteLLMClient`, asking it
  to reply with a single token (``SIMPLE`` or ``COMPLEX``). The real provider/
  model name and ``api_base`` are resolved privately at construction time and
  never surface in logs, events, or the :class:`Classification` rationale - only
  the :attr:`TaskClassifier.alias` (``kinetic-classifier-v1``) is ever shown.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

from kinetic_sdk.agent.modes import AgentMode
from kinetic_sdk.secret.registry import SecretRegistry
from kinetic_sdk.secret.value import SecretValue

logger = logging.getLogger(__name__)


class TaskComplexity(str, Enum):
    """Coarse complexity bucket assigned by a classifier.

    Subclassing ``str`` so the value serialises naturally and compares equal to
    its string value (``TaskComplexity.COMPLEX == "complex"``).
    """

    SIMPLE = "simple"
    COMPLEX = "complex"

    def to_mode(self) -> AgentMode:
        """Map this complexity bucket to the recommended :class:`AgentMode`."""
        return AgentMode.FLASH if self is TaskComplexity.SIMPLE else AgentMode.MAX


@dataclass(frozen=True)
class Classification:
    """Result of classifying a task.

    Attributes:
        complexity: The assigned :class:`TaskComplexity` bucket.
        mode: The recommended :class:`AgentMode` to run the task in (derived
            from :attr:`complexity`).
        confidence: Heuristic confidence in the classification, in ``[0, 1]``.
            Low confidence can be a signal to default to MAX.
        rationale: Short human-readable reason, useful for debugging. Must not
            leak the underlying classifier model name.
    """

    complexity: TaskComplexity
    mode: AgentMode
    confidence: float
    rationale: str = ""


class TaskClassifier(ABC):
    """Interface for task-complexity classifiers.

    Implementations MUST use a model distinct from the main task LLM and MUST
    NOT expose the real provider/model name in logs or configuration. Refer to
    the configured model only by its internal alias (see :attr:`alias`).
    """

    #: Internal, provider-agnostic alias for the classifier model. Real model
    #: names are resolved privately and never surfaced publicly.
    alias: str = "kinetic-classifier-v1"

    @abstractmethod
    def classify(self, task: str) -> Classification:
        """Classify *task* into a recommended :class:`AgentMode`.

        Args:
            task: The user's task description (the first user message).

        Returns:
            A :class:`Classification`. Implementations should favour MAX when
            unsure, since running a complex task in FLASH mode is the worse
            failure mode.
        """


class DefaultClassifier(TaskClassifier):
    """Deterministic placeholder: always routes to MAX.

    Kept conservative on purpose - the cost of a wrong FLASH choice (a complex
    task running with too few capabilities) is higher than running a simple
    task in MAX. Used as the offline fallback and in tests that don't care
    about real routing.
    """

    def classify(self, task: str) -> Classification:
        return Classification(
            complexity=TaskComplexity.COMPLEX,
            mode=AgentMode.MAX,
            confidence=1.0,
            rationale="default",
        )


class LiteLLMClassifier(TaskClassifier):
    """Model-backed classifier using a cheap LiteLLM-reachable model.

    The underlying model (``openai/openhands/glm-5.2``) and ``api_base``
    (``https://llm-proxy.app.all-hands.dev``) are the real provider config used
    only when calling the API. They are NEVER logged, never put into events or
    :class:`Classification` rationale, and never exposed through :attr:`model`
    publicly - the only name that surfaces anywhere outside the actual API call
    is :attr:`alias` (``kinetic-classifier-v1``).

    The classifier asks the model to reply with exactly one word - ``SIMPLE``
    or ``COMPLEX`` - given the user's task (plus, optionally, a short summary of
    the conversation history). ``max_tokens`` is kept very low (~10) because a
    single token is all that is needed.

    Any failure (network error, timeout, unparseable response, missing api key)
    is swallowed and falls back to :attr:`TaskComplexity.COMPLEX` (route to
    MAX). This is the safe side: it is better to run a simple task in MAX than
    to run a complex task in FLASH and skip needed steps.
    """

    #: Real provider model string. Used only in the private API call path.
    _MODEL = "openai/openhands/glm-5.2"
    #: Real provider base URL. Used only in the private API call path.
    _API_BASE = "https://llm-proxy.app.all-hands.dev"
    #: Environment variable read for the API key.
    _API_KEY_ENV = "OPENHANDS_API_KEY"
    #: Small token budget: the model only needs to emit one word.
    _MAX_TOKENS = 10

    def __init__(
        self,
        client: Any | None = None,
        api_key: str | SecretValue | None = None,
        summary: str | None = None,
        secrets: SecretRegistry | None = None,
    ) -> None:
        """Create the classifier.

        Args:
            client: Optional pre-built :class:`LiteLLMClient` (used by tests to
                inject a mock). When omitted a real :class:`LiteLLMClient` is
                built lazily from :attr:`_MODEL` / :attr:`_API_BASE` and the
                ``api_key`` (resolved from ``secrets`` if not given).
            api_key: API key for the classifier endpoint, as a plain string
                (wrapped automatically, backward compatible) or a
                :class:`SecretValue`. When ``None`` the key is resolved from
                the ``secrets`` registry (default: the ``OPENHANDS_API_KEY``
                environment variable). Never stored in a way that surfaces in
                logs.
            summary: Optional short summary of the existing conversation history
                forwarded to the classifier for context. ``None`` disables it.
            secrets: Optional :class:`SecretRegistry` used to resolve the API
                key when ``api_key`` is not given. Defaults to a registry that
                reads environment variables.
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
            from kinetic_sdk.llm.client import LiteLLMClient

            self._client = LiteLLMClient(
                model=self._MODEL,
                api_key=self._api_key,
                api_base=self._API_BASE,
                max_tokens=self._MAX_TOKENS,
            )

    @property
    def model(self) -> str:
        """Public model name. Always the alias, never the real provider name."""
        return self.alias

    def classify(self, task: str) -> Classification:
        """Classify *task*, falling back to COMPLEX on any failure."""
        prompt = self._build_prompt(task)
        try:
            response = self._client.chat(messages=prompt, max_tokens=self._MAX_TOKENS)
            answer = (response.content or "").strip().upper()
        except Exception as exc:  # noqa: BLE001 - safe-side fallback
            logger.warning(
                "%s classification call failed (falling back to COMPLEX): %s",
                self.alias,
                type(exc).__name__,
            )
            return self._fallback(reason="classifier_error")

        complexity = self._parse(answer)
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

    # --- internals ----------------------------------------------------

    def _build_prompt(self, task: str) -> list[dict[str, Any]]:
        """Build the one-shot user/system messages sent to the classifier."""
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

    @staticmethod
    def _parse(answer: str) -> TaskComplexity:
        """Parse the model's one-word reply. Default to COMPLEX when unclear."""
        if "COMPLEX" in answer:
            return TaskComplexity.COMPLEX
        if "SIMPLE" in answer:
            return TaskComplexity.SIMPLE
        return TaskComplexity.COMPLEX

    def _fallback(self, reason: str) -> Classification:
        return Classification(
            complexity=TaskComplexity.COMPLEX,
            mode=AgentMode.MAX,
            confidence=0.0,
            rationale=reason,
        )

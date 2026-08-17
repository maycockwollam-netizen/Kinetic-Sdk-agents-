"""Task classifier for FLASH/MAX routing (Stage 2 - interface stub).

The classifier decides whether a task is SIMPLE (-> FLASH mode) or COMPLEX
(-> MAX mode) *before* the agent loop starts. Importantly it uses a cheap,
separate model - never the main task model - and the underlying provider/model
name is hidden behind an internal alias (e.g. ``kinetic-classifier-v1``) so it
does not leak into logs or public configuration.

Stage 1 ships only the interface and a trivial default implementation that
routes everything to MAX. Real classification lands in Stage 2 once the core
loop is validated end-to-end.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from kinetic_sdk.agent.modes import AgentMode


@dataclass(frozen=True)
class Classification:
    """Result of classifying a task.

    Attributes:
        mode: The recommended :class:`AgentMode` to run the task in.
        confidence: Heuristic confidence in the classification, in ``[0, 1]``.
            Low confidence can be a signal to default to MAX.
        rationale: Short human-readable reason, useful for debugging. Must not
            leak the underlying classifier model name.
    """

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
    """Stage 1 placeholder: always routes to MAX.

    Kept conservative on purpose - the cost of a wrong FLASH choice (a complex
    task running with too few capabilities) is higher than running a simple
    task in MAX. Replaced by a model-backed classifier in Stage 2.
    """

    def classify(self, task: str) -> Classification:
        return Classification(mode=AgentMode.MAX, confidence=1.0, rationale="default")

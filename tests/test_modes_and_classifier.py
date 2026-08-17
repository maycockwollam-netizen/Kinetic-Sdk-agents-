"""Unit tests for AgentMode and the classifier stubs."""

from __future__ import annotations

from kinetic_sdk.agent.classifier import Classification, DefaultClassifier
from kinetic_sdk.agent.modes import AgentMode


def test_mode_values():
    assert AgentMode.FLASH.value == "flash"
    assert AgentMode.MAX.value == "max"
    # str subclass: compares equal to its string value (useful for JSON).
    assert AgentMode.FLASH == "flash"


def test_escalation_flash_to_max_allowed():
    assert AgentMode.escalates_to(AgentMode.FLASH) is AgentMode.MAX


def test_escalation_max_to_flash_not_allowed():
    assert AgentMode.escalates_to(AgentMode.MAX) is None


def test_default_classifier_routes_to_max():
    clf = DefaultClassifier()
    result = clf.classify("any task")
    assert isinstance(result, Classification)
    assert result.mode is AgentMode.MAX
    assert result.confidence == 1.0


def test_classifier_alias_is_internal_and_stable():
    assert DefaultClassifier.alias == "kinetic-classifier-v1"

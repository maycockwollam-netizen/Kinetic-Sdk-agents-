"""Tests for subagent/tool.py — DelegateTool."""

from __future__ import annotations

import pytest

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.security.audit import InMemoryAuditLogger
from kinetic_sdk.security.policy import AllowListPolicy, PermissivePolicy
from kinetic_sdk.subagent import (
    DELEGATE_TOOL_NAME,
    DelegateTool,
    SpawnBudget,
    SubagentSpec,
)
from kinetic_sdk.testing import MockLLMClient, text_response, tool_response

from ._helpers import EchoTool, LoopLLM

SPEC = SubagentSpec(
    name="worker", system_prompt="You are the worker.", description="Does work."
)


def make_root(
    root_llm: MockLLMClient,
    delegate: DelegateTool,
    policy: object = None,
    audit: InMemoryAuditLogger | None = None,
) -> tuple[Agent, InMemoryAuditLogger]:
    audit = audit if audit is not None else InMemoryAuditLogger()
    root = Agent(
        llm=root_llm,
        tools=[EchoTool(), delegate],
        permission_policy=policy or PermissivePolicy(),
        audit_logger=audit,
    )
    delegate.bind(root)
    return root, audit


class TestRegistry:
    def test_accepts_iterable_or_mapping(self) -> None:
        from_iterable = DelegateTool([SPEC])
        from_mapping = DelegateTool({SPEC.name: SPEC})
        assert from_iterable.subagent_specs == from_mapping.subagent_specs

    def test_non_spec_values_rejected(self) -> None:
        with pytest.raises(TypeError, match="SubagentSpec"):
            DelegateTool({"worker": object()})  # type: ignore[dict-item]

    def test_key_must_match_spec_name(self) -> None:
        with pytest.raises(ValueError, match="does not match"):
            DelegateTool({"other-key": SPEC})

    def test_schema_lists_registered_specs(self) -> None:
        delegate = DelegateTool([SPEC])
        schema = delegate.to_schema()
        assert schema["name"] == DELEGATE_TOOL_NAME
        props = schema["input_schema"]["properties"]
        assert props["subagent_name"]["enum"] == ["worker"]
        assert "worker: Does work." in schema["description"]

    def test_budget_defaults_when_omitted(self) -> None:
        delegate = DelegateTool([SPEC])
        assert isinstance(delegate.budget, SpawnBudget)
        assert delegate.budget.used == 0


class TestExecute:
    def test_unbound_tool_returns_error(self) -> None:
        delegate = DelegateTool([SPEC])
        result = delegate.execute(subagent_name="worker", task_prompt="go")
        assert result.is_error
        assert "not bound" in result.error

    def test_unknown_subagent_returns_error(self) -> None:
        root_llm = MockLLMClient([])
        delegate = DelegateTool([SPEC])
        make_root(root_llm, delegate)
        result = delegate.execute(subagent_name="ghost", task_prompt="go")
        assert result.is_error
        assert "Unknown sub-agent 'ghost'" in result.error
        assert "worker" in result.error  # lists what IS registered

    def test_success_returns_only_redacted_final_message(self) -> None:
        secret = "sk-" + "x" * 30
        child_llm = MockLLMClient([text_response(f"token={secret} done")])
        delegate = DelegateTool(
            [SubagentSpec(
                name="worker", system_prompt="p", description="d", model="child-model"
            )],
            llm_factory=lambda model: child_llm,
        )
        make_root(MockLLMClient([]), delegate)
        result = delegate.execute(subagent_name="worker", task_prompt="go")
        assert not result.is_error
        assert result.output["subagent"] == "worker"
        assert result.output["agent_id"]
        # The secret the sub-agent quoted is scrubbed before reaching the
        # parent's context; the transcript never appears either.
        assert result.output["final_message"] == "token=[REDACTED] done"

    def test_budget_exceeded_maps_to_tool_result_error(self) -> None:
        loop_llm = LoopLLM("echo", {"message": "again"})
        delegate = DelegateTool(
            [SubagentSpec(name="worker", system_prompt="p", model="loop-model")],
            budget=SpawnBudget(3),
            llm_factory=lambda model: loop_llm,
        )
        _, audit = make_root(MockLLMClient([]), delegate)
        result = delegate.execute(subagent_name="worker", task_prompt="loop")
        assert result.is_error
        assert result.error.startswith("BudgetExceededError")
        finishes = [e for e in audit.entries if e["event"] == "subagent_finished"]
        assert [e["outcome"] for e in finishes] == ["budget_exceeded"]

    def test_repetition_limit_maps_to_tool_result_error(self) -> None:
        loop_llm = LoopLLM("echo", {"message": "same"})
        delegate = DelegateTool(
            [SubagentSpec(name="worker", system_prompt="p", model="loop-model")],
            max_consecutive_repeats=3,
            llm_factory=lambda model: loop_llm,
        )
        _, audit = make_root(MockLLMClient([]), delegate)
        result = delegate.execute(subagent_name="worker", task_prompt="loop")
        assert result.is_error
        assert result.error.startswith("RepetitionLimitError")
        finishes = [e for e in audit.entries if e["event"] == "subagent_finished"]
        assert [e["outcome"] for e in finishes] == ["repetition_limit"]

    def test_subagent_crash_maps_to_tool_result_error(self) -> None:
        class ExplodingLLM(LoopLLM):
            def chat(self, messages, tools=None, system=None, **kwargs):  # noqa: ANN001
                raise RuntimeError("provider exploded")

        delegate = DelegateTool(
            [SubagentSpec(name="worker", system_prompt="p", model="loop-model")],
            llm_factory=lambda model: ExplodingLLM("echo"),
        )
        _, audit = make_root(MockLLMClient([]), delegate)
        result = delegate.execute(subagent_name="worker", task_prompt="boom")
        assert result.is_error
        assert "RuntimeError: provider exploded" in result.error
        finishes = [e for e in audit.entries if e["event"] == "subagent_finished"]
        assert [e["outcome"] for e in finishes] == ["error"]


class TestPermissionGating:
    def test_delegate_call_passes_through_permission_policy(self) -> None:
        factory_calls: list[str] = []
        root_llm = MockLLMClient(
            [
                tool_response(
                    "d1", "delegate",
                    {"subagent_name": "worker", "task_prompt": "go"},
                ),
                text_response("root recovered"),
            ]
        )
        def factory(model: str) -> MockLLMClient:
            factory_calls.append(model)
            return MockLLMClient([text_response("child")])

        delegate = DelegateTool(
            [SubagentSpec(name="worker", system_prompt="p", model="child-model")],
            llm_factory=factory,
        )
        audit = InMemoryAuditLogger()
        root, _ = make_root(
            root_llm, delegate,
            policy=AllowListPolicy(always_allow=["echo"]),  # no "delegate"
            audit=audit,
        )
        final = root.run("delegate something")
        assert final == "root recovered"
        # The sub-agent was never even built: the policy denied the call
        # before DelegateTool.execute ran.
        assert factory_calls == []
        assert not any(e["event"] == "subagent_spawn" for e in audit.entries)
        denials = [e for e in audit.entries if e["event"] == "permission_denied"]
        assert len(denials) == 1
        assert denials[0]["tool_name"] == "delegate"

    def test_delegate_call_allowed_when_policy_permits(self) -> None:
        root_llm = MockLLMClient(
            [
                tool_response(
                    "d1", "delegate",
                    {"subagent_name": "worker", "task_prompt": "go"},
                ),
                text_response("root done"),
            ]
        )
        delegate = DelegateTool(
            [SubagentSpec(name="worker", system_prompt="p", model="child-model")],
            llm_factory=lambda model: MockLLMClient([text_response("child done")]),
        )
        audit = InMemoryAuditLogger()
        root, _ = make_root(
            root_llm, delegate,
            policy=AllowListPolicy(always_allow=["echo", "delegate"]),
            audit=audit,
        )
        assert root.run("delegate something") == "root done"
        outcomes = [
            e.get("outcome")
            for e in audit.entries
            if e["event"] == "subagent_finished"
        ]
        assert outcomes == ["completed"]

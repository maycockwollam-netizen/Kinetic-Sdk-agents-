"""End-to-end: a real Agent.run() driving tools from a real MCP subprocess.

The MCP server here is the fake stdio server script (no Unity/Roblox
needed), but everything between the agent and it is production code:
registry -> StdioTransport -> MCPClient handshake -> MCPToolAdapter ->
agent loop (permission policy, audit log, observability, redaction).
"""

from __future__ import annotations

import sys

import pytest

from kinetic_sdk.agent import Agent
from kinetic_sdk.mcp import MCPServerConfig, MCPServerRegistry
from kinetic_sdk.observability import InMemoryObservabilityLogger, RunTrace
from kinetic_sdk.security import AllowListPolicy, InMemoryAuditLogger
from kinetic_sdk.testing import (
    MockLLMClient,
    assert_no_permission_denied,
    assert_tool_called,
    text_response,
    tool_response,
)

from ._mcp_fakes import FAKE_MCP_SERVER_SCRIPT


@pytest.fixture()
def fake_server_path(tmp_path):
    path = tmp_path / "fake_mcp_server.py"
    path.write_text(FAKE_MCP_SERVER_SCRIPT)
    return str(path)


@pytest.fixture()
def registry(fake_server_path):
    with MCPServerRegistry() as reg:
        reg.register(
            "fake",
            MCPServerConfig.stdio(command=sys.executable, args=[fake_server_path]),
        )
        yield reg


def _make_agent(registry, llm, policy, audit, obs):
    tools = registry.get_tools_as_kinetic_tools("fake")
    return Agent(
        llm=llm,
        tools=tools,
        permission_policy=policy,
        audit_logger=audit,
        observability_logger=obs,
    )


class TestAgentWithMCPTools:
    def test_agent_calls_mcp_tool_end_to_end(self, registry) -> None:
        llm = MockLLMClient(
            [
                tool_response("c1", "fake.echo", {"text": "hello from agent"}),
                text_response("Xong — server trả lời: hello from agent"),
            ]
        )
        audit = InMemoryAuditLogger()
        obs = InMemoryObservabilityLogger()
        agent = _make_agent(
            registry, llm, AllowListPolicy(always_allow=["fake.echo"]), audit, obs
        )

        answer = agent.run("Hãy gọi tool echo trên MCP server")

        assert answer == "Xong — server trả lời: hello from agent"
        trace = RunTrace.collect(obs.entries, agent.run_id)
        assert_tool_called(trace, "fake.echo", times=1)
        assert_no_permission_denied(trace)
        events = [e["event"] for e in audit.entries]
        assert events == ["tool_call", "tool_result"]
        assert audit.entries[0]["tool_name"] == "fake.echo"
        # The MCP server's answer really travelled back into the conversation.
        tool_results = [
            block
            for msg in agent.state.messages
            if msg["role"] == "user"
            for block in (msg["content"] if isinstance(msg["content"], list) else [])
            if isinstance(block, dict) and block.get("type") == "tool_result"
        ]
        assert any("hello from agent" in str(b.get("content")) for b in tool_results)

    def test_permission_denial_applies_to_mcp_tools_like_local_ones(
        self, registry
    ) -> None:
        llm = MockLLMClient(
            [
                tool_response("c1", "fake.echo", {"text": "hi"}),
                text_response("Tool bị chặn, dừng lại."),
            ]
        )
        audit = InMemoryAuditLogger()
        obs = InMemoryObservabilityLogger()
        # Policy allows NOTHING — the MCP tool is gated exactly like a local one.
        agent = _make_agent(registry, llm, AllowListPolicy(), audit, obs)

        answer = agent.run("Gọi echo đi")

        assert answer == "Tool bị chặn, dừng lại."
        events = [e["event"] for e in audit.entries]
        assert events == ["tool_call", "permission_denied"]
        trace = RunTrace.collect(obs.entries, agent.run_id)
        assert trace.to_summary()["permission_denied"] is True

    def test_mcp_server_crash_becomes_tool_error_not_agent_crash(self, registry) -> None:
        llm = MockLLMClient(
            [
                tool_response("c1", "fake.crash", {}),
                text_response("Server chết giữa chừng, báo người dùng."),
            ]
        )
        audit = InMemoryAuditLogger()
        obs = InMemoryObservabilityLogger()
        agent = _make_agent(
            registry, llm, AllowListPolicy(always_allow=["fake.crash"]), audit, obs
        )

        answer = agent.run("Gọi tool crash")

        assert answer == "Server chết giữa chừng, báo người dùng."
        result_entry = next(e for e in audit.entries if e["event"] == "tool_result")
        assert result_entry["result"]["is_error"] is True
        assert "code 3" in result_entry["result"]["error"]

    def test_secret_from_mcp_server_is_redacted_everywhere(self, registry) -> None:
        token = "ghp_" + "a1b2c3d4e5f6" * 3
        llm = MockLLMClient(
            [
                tool_response("c1", "fake.echo", {"text": token}),
                text_response("Đã xử lý xong."),
            ]
        )
        audit = InMemoryAuditLogger()
        obs = InMemoryObservabilityLogger()
        agent = _make_agent(
            registry, llm, AllowListPolicy(always_allow=["fake.echo"]), audit, obs
        )

        agent.run("Echo cái token này")

        # The fake server echoes the token back; the adapter must scrub it
        # before it reaches the audit log or the conversation history.
        result_entry = next(e for e in audit.entries if e["event"] == "tool_result")
        assert token not in str(result_entry)
        assert "[REDACTED]" in str(result_entry)
        for entry in obs.entries:
            assert token not in str(entry["payload"])


class TestKineticAsServerForAnotherAgent:
    """Direction B end-to-end: Kinetic serves tools, another agent's MCP
    client consumes them — full circle through real subprocess pipes."""

    def test_agent_consumes_tools_served_by_kinetic_server(self, tmp_path) -> None:
        server_script = tmp_path / "kinetic_as_server.py"
        server_script.write_text(
            "from kinetic_sdk.mcp import MCPServer\n"
            "from kinetic_sdk.security import AllowListPolicy\n"
            "from kinetic_sdk.testing import MockTool\n"
            "from kinetic_sdk.tool.base import ToolResult\n"
            "\n"
            "tools = [MockTool(name='greet', description='Greet in Vietnamese',\n"
            "         parameters={'type': 'object', 'properties': {'name': {'type': 'string'}}},\n"
            "         handler=lambda name='': ToolResult(output=f'Xin chào {name}'))]\n"
            "MCPServer.serve_stdio(tools, permission_policy=AllowListPolicy(always_allow=['greet']))\n"
        )
        with MCPServerRegistry() as registry:
            registry.register(
                "kinetic",
                MCPServerConfig.stdio(command=sys.executable, args=[str(server_script)]),
            )
            tools = registry.get_tools_as_kinetic_tools("kinetic")
            assert [t.name for t in tools] == ["kinetic.greet"]

            llm = MockLLMClient(
                [
                    tool_response("c1", "kinetic.greet", {"name": "Kinetic"}),
                    text_response("Server Kinetic chào: Xin chào Kinetic"),
                ]
            )
            agent = Agent(
                llm=llm,
                tools=tools,
                permission_policy=AllowListPolicy(always_allow=["kinetic.greet"]),
            )
            answer = agent.run("Chào Kinetic qua MCP")
            assert answer == "Server Kinetic chào: Xin chào Kinetic"

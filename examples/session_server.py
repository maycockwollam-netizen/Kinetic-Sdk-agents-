"""Run with ``python examples/session_server.py`` to see AgentServer session continuity."""

from kinetic_sdk.agent import Agent
from kinetic_sdk.security import PermissivePolicy
from kinetic_sdk.server import AgentServer
from kinetic_sdk.testing import MockLLMClient, text_response


def factory() -> Agent:
    return Agent(MockLLMClient([text_response("first reply"), text_response("second reply")]), permission_policy=PermissivePolicy())


if __name__ == "__main__":
    server = AgentServer(factory, port=0)
    server.start_in_thread()
    try:
        print(server._run_agent("hello", False, None, "demo-session")["final"])
        print(server._run_agent("continue", False, None, "demo-session")["final"])
    finally:
        server.shutdown()

"""Offline demo: an agent keeps a focused plan with todo tools."""

from kinetic_sdk.agent import Agent
from kinetic_sdk.security.policy import AllowListPolicy
from kinetic_sdk.testing import MockLLMClient, text_response, tool_response
from kinetic_sdk.todo import InMemoryTodoStore, TodoReadTool, TodoWriteTool

store = InMemoryTodoStore()
agent = Agent(
    llm=MockLLMClient(
        [
            tool_response(
                "write-plan",
                "todo_write",
                {
                    "todos": [
                        {"content": "Inspect the request", "status": "completed"},
                        {"content": "Implement the change", "status": "in_progress"},
                        {"content": "Run tests", "status": "pending"},
                    ]
                },
            ),
            tool_response("read-plan", "todo_read", {}),
            text_response("The plan is recorded; I am implementing the next step."),
        ]
    ),
    tools=[TodoWriteTool(store), TodoReadTool(store)],
    permission_policy=AllowListPolicy(always_allow=["todo_write", "todo_read"]),
)

print(agent.run("Plan this multi-step change and use the todo tools as you work."))

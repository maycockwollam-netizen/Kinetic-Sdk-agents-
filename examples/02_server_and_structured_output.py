"""Structured output + the AgentServer REST API (offline demo).

Run it: python examples/02_server_and_structured_output.py
Then:  curl -X POST http://127.0.0.1:<port>/runs \
        -H 'Content-Type: application/json' \
        -d '{"message": "rate this patch", "output_schema": {"type": "object"}}'
"""

import json
import urllib.request

from kinetic_sdk.agent.agent import Agent
from kinetic_sdk.security.policy import AllowListPolicy
from kinetic_sdk.server import AgentServer
from kinetic_sdk.testing import MockLLMClient, text_response

SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer"},
        "verdict": {"type": "string"},
    },
    "required": ["score"],
}


def factory() -> Agent:
    # One fresh agent per HTTP run (conversations are per-request here).
    return Agent(
        llm=MockLLMClient([text_response('{"score": 7, "verdict": "ok"}')]),
        permission_policy=AllowListPolicy(),
    )


if __name__ == "__main__":
    server = AgentServer(factory, port=0, token=None)
    server.start_in_thread()
    try:
        url = f"http://127.0.0.1:{server.port}/runs"
        req = urllib.request.Request(
            url,
            data=json.dumps({"message": "review", "output_schema": SCHEMA}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            print(json.dumps(json.load(resp), indent=2))
    finally:
        server.shutdown()

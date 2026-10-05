from kinetic_sdk.tool.validation import validate_value
from kinetic_sdk.verify import VerificationEvidence, verification_schema


def test_verification_schema_accepts_expected_evidence() -> None:
    evidence = VerificationEvidence(
        changed_files=["src/app.py"],
        commands_run=["python -m pytest -q"],
        test_results={"pytest": "passed"},
        remaining_risks=[],
    )
    problems = validate_value(verification_schema(), evidence.to_dict(), path="$")
    assert problems == []


def test_verification_requires_evidence_before_complete() -> None:
    assert not VerificationEvidence().is_verified()
    assert not VerificationEvidence(changed_files=["a.py"], commands_run=["pytest"]).is_verified()
    assert VerificationEvidence(
        changed_files=["a.py"],
        commands_run=["pytest"],
        test_results={"pytest": "passed"},
    ).is_verified()


def test_agent_sets_verification_evidence_when_required() -> None:
    import json

    from kinetic_sdk.agent.agent import Agent
    from kinetic_sdk.testing import MockLLMClient, text_response

    payload = {
        "changed_files": ["src/app.py"],
        "commands_run": ["python -m pytest -q"],
        "test_results": {"pytest": "passed"},
        "remaining_risks": ["none"],
    }
    llm = MockLLMClient([text_response(json.dumps(payload))])
    agent = Agent(llm=llm, verification_required=True)
    answer = agent.run("do something")
    assert not agent.verification_evidence is None
    assert agent.verification_evidence.changed_files == ["src/app.py"]
    assert agent.verification_evidence.is_verified()
    # Return value stays raw final text for existing callers.
    assert answer == json.dumps(payload)


def test_agent_settings_can_opt_into_verification() -> None:
    from kinetic_sdk.agent.settings import AgentSettings

    settings = AgentSettings(llm_profile="mock", verification_required=True)
    data = settings.to_dict()
    assert data["verification_required"] is True
    assert AgentSettings.from_dict(data).verification_required is True


def test_agent_retries_until_verification_evidence_is_complete() -> None:
    import json

    from kinetic_sdk.agent.agent import Agent
    from kinetic_sdk.testing import MockLLMClient, text_response

    weak = {
        "changed_files": [],
        "commands_run": ["pytest"],
        "test_results": {"pytest": "failed"},
        "remaining_risks": ["not done"],
    }
    strong = {
        "changed_files": ["src/app.py"],
        "commands_run": ["python -m pytest -q"],
        "test_results": {"pytest": "passed"},
        "remaining_risks": [],
    }
    llm = MockLLMClient([text_response(json.dumps(weak)), text_response(json.dumps(strong))])
    agent = Agent(llm=llm, verification_required=True)
    answer = agent.run("fix the bug")
    assert agent.verification_evidence is not None
    assert agent.verification_evidence.is_verified()
    assert answer == json.dumps(strong)

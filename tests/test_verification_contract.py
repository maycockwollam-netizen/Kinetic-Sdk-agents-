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

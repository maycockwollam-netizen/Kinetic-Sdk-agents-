import pytest

from kinetic_sdk.subagent import SubagentSpecError, load_subagent_spec


def test_load_subagent_spec_from_markdown(tmp_path):
    path = tmp_path / "agent.md"
    path.write_text("---\nname: code-reviewer\ndescription: Reviews PRs\nmodel: model-x\n---\nReview correctness.", encoding="utf-8")
    assert load_subagent_spec(path).system_prompt == "Review correctness."
    assert load_subagent_spec(path).model == "model-x"


def test_load_subagent_spec_requires_name(tmp_path):
    path = tmp_path / "agent.md"
    path.write_text("---\ndescription: desc\n---\nprompt", encoding="utf-8")
    with pytest.raises(SubagentSpecError):
        load_subagent_spec(path)


def test_load_subagent_spec_uses_existing_empty_prompt_validation(tmp_path):
    path = tmp_path / "agent.md"
    path.write_text("---\nname: reviewer\n---\n", encoding="utf-8")
    with pytest.raises(SubagentSpecError, match="non-empty system_prompt"):
        load_subagent_spec(path)

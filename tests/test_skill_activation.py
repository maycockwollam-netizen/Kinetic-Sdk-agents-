from kinetic_sdk.agent import Agent
from kinetic_sdk.conversation.state import ConversationState
from kinetic_sdk.hooks import HookPoint, HookRegistry
from kinetic_sdk.skills import Skill, SkillActivationHook, select_active_skills
from kinetic_sdk.testing import MockLLMClient, text_response


def skill(name: str, kind: str = "repo", triggers: tuple[str, ...] = ()) -> Skill:
    return Skill(name, f"{name} description", "local", "/tmp/" + name, type=kind, triggers=triggers)


def test_repo_skills_are_always_selected_and_knowledge_skills_need_trigger():
    repo = skill("repository")
    knowledge = skill("containers", "knowledge", ("docker", "container"))
    assert select_active_skills([repo, knowledge], "edit Python") == [repo]
    assert select_active_skills([repo, knowledge], "Build a DOCKER image") == [repo, knowledge]


def test_legacy_skill_defaults_to_repo(tmp_path):
    directory = tmp_path / "legacy"
    directory.mkdir()
    (directory / "SKILL.md").write_text("---\nname: legacy\ndescription: old\n---\nbody", encoding="utf-8")
    assert Skill.from_directory(directory, "local").type == "repo"


def test_knowledge_skill_parses_comma_separated_triggers(tmp_path):
    directory = tmp_path / "containers"
    directory.mkdir()
    (directory / "SKILL.md").write_text(
        "---\nname: containers\ndescription: container knowledge\n"
        "type: knowledge\ntriggers: docker, container\n---\nbody",
        encoding="utf-8",
    )
    loaded = Skill.from_directory(directory, "local")
    assert loaded.type == "knowledge"
    assert loaded.triggers == ("docker", "container")


def test_activation_appends_catalog_without_mutating_base_system_prompt():
    repo = skill("repository")
    registry = HookRegistry()
    registry.register(HookPoint.BEFORE_LLM_CALL, SkillActivationHook([repo]))
    llm = MockLLMClient([text_response("done")])
    state = ConversationState(system_prompt="Original persona")
    assert Agent(llm, state=state, hooks=registry).run("hello") == "done"
    assert llm.calls[0]["system"] == "Original persona\n\n- repository: repository description"
    assert state.system_prompt == "Original persona"

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


# --- ranked activation --------------------------------------------------

def test_rank_skills_orders_repo_then_knowledge_by_trigger_hits():
    from kinetic_sdk.skills import rank_skills

    a = skill("a", "knowledge", ("docker", "container"))
    b = skill("b", "knowledge", ("docker",))
    repo = skill("repo")
    ranked = rank_skills([a, b, repo], "use docker and container tooling")
    assert ranked == [repo, a, b]  # repo first; a (2 hits) before b (1)


def test_rank_skills_ties_break_by_name():
    from kinetic_sdk.skills import rank_skills

    beta = skill("beta", "knowledge", ("k8s",))
    alpha = skill("alpha", "knowledge", ("k8s",))
    assert rank_skills([beta, alpha], "k8s crd") == [alpha, beta]


def test_rank_skills_include_description_fallback_and_limit():
    from kinetic_sdk.skills import rank_skills
    from kinetic_sdk.skills.skill import Skill

    db = Skill(
        "db", "querying postgres tables", "local", "/tmp/db",
        type="knowledge", triggers=("postgres",),
    )
    assert rank_skills([db], "how to query tables") == []  # no trigger
    assert rank_skills([db], "how to query tables", include_description=True) == [db]

    a = skill("a", "knowledge", ("x",))
    b = skill("b", "knowledge", ("x",))
    repo = skill("repo")
    assert rank_skills([a, b, repo], "x y", limit=2) == [repo, a]
    assert rank_skills([repo], "z", limit=0) == []


def test_skill_matches_and_trigger_hits():
    from kinetic_sdk.skills.skill import Skill

    s = Skill("s", "d", "local", "/tmp/s", type="knowledge", triggers=("docker", "container"))
    assert s.matches("build a DOCKER image")
    assert s.trigger_hits("docker and container") == 2
    assert not s.matches("nothing here")


def test_activation_hook_max_active_and_description():
    from kinetic_sdk.hooks.base import HookContext, HookPoint
    from kinetic_sdk.skills import SkillActivationHook
    from kinetic_sdk.skills.skill import Skill

    kw = Skill("kw", "build kubernetes deployments", "local", "/tmp/kw", type="knowledge", triggers=("k8s", "deploy"))
    kd = Skill("kd", "deploy workloads", "local", "/tmp/kd", type="knowledge", triggers=("deploy",))
    hook = SkillActivationHook([kw, kd], include_description=True, max_active=1)
    result = hook(HookContext(point=HookPoint.BEFORE_LLM_CALL, user_message="k8s deploy"))

    assert result is not None and result.modified_context
    prompt = result.modified_context["system_prompt"]
    assert "kw" in prompt  # 2 hits outranks kd's 1, and max_active caps at 1
    assert "kd" not in prompt

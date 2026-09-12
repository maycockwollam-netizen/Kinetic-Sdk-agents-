"""Run with ``python examples/skill_activation.py`` to inspect trigger-based skill catalogs."""

from kinetic_sdk.skills import Skill, select_active_skills

if __name__ == "__main__":
    skills = [
        Skill("repository", "Repository conventions", "example", ".", type="repo"),
        Skill("docker", "Docker knowledge", "example", ".", type="knowledge", triggers=("container", "docker")),
    ]
    for question in ("edit a Python file", "build a Docker container"):
        print(question, "->", [skill.name for skill in select_active_skills(skills, question)])

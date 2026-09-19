"""Regression coverage for Docker run wrapper hardening."""

import pytest

from kinetic_sdk.terminal.docker import SandboxWrapperError, docker_run_wrapper


def test_docker_run_wrapper_builds_all_hardening_options_in_order() -> None:
    assert docker_run_wrapper(
        "python:3.12",
        workdir="/app",
        volumes=["/workspace:/app:ro"],
        env={"HOME": "/tmp/home", "MODE": "test"},
        network="none",
        read_only=True,
        tmpfs=["/tmp:rw,noexec", "/var/tmp"],
        drop_all_capabilities=True,
        no_new_privileges=True,
        memory_limit="512M",
        cpu_limit=1.5,
        pids_limit=64,
        user="1000:1000",
    ) == [
        "docker",
        "run",
        "--rm",
        "-i",
        "--network",
        "none",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,noexec",
        "--tmpfs",
        "/var/tmp",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges:true",
        "--memory",
        "512M",
        "--cpus",
        "1.5",
        "--pids-limit",
        "64",
        "--user",
        "1000:1000",
        "-v",
        "/workspace:/app:ro",
        "-e",
        "HOME",
        "-e",
        "MODE",
        "-w",
        "/app",
        "python:3.12",
    ]


def test_read_only_defaults_to_tmpfs_and_empty_list_disables_it() -> None:
    argv = docker_run_wrapper("img")
    assert ["--tmpfs", "/tmp"] == argv[argv.index("--tmpfs") : argv.index("--tmpfs") + 2]
    assert "--tmpfs" not in docker_run_wrapper("img", tmpfs=[])


@pytest.mark.parametrize("volume", ["/var/run/docker.sock:/socket", "/:/host"])
def test_docker_run_wrapper_rejects_unsafe_mounts(volume: str) -> None:
    with pytest.raises(SandboxWrapperError, match="unsafe mount source"):
        docker_run_wrapper("img", volumes=[volume])


def test_unsafe_mounts_can_be_explicitly_allowed() -> None:
    argv = docker_run_wrapper("img", volumes=["/:/host"], allow_unsafe_mounts=True)
    assert ["-v", "/:/host"] == argv[argv.index("-v") : argv.index("-v") + 2]


def test_docker_run_wrapper_rejects_option_like_image() -> None:
    with pytest.raises(SandboxWrapperError):
        docker_run_wrapper("--privileged")


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"workdir": "relative"}, "workdir"),
        ({"user": "bad user"}, "user"),
        ({"memory_limit": "1tb"}, "memory"),
        ({"cpu_limit": 0}, "cpu_limit"),
        ({"pids_limit": 0}, "pids_limit"),
        ({"env": {"NOT-VALID": "x"}}, "environment"),
    ],
)
def test_docker_run_wrapper_validates_option_values(
    kwargs: dict[str, object], error: str
) -> None:
    with pytest.raises(SandboxWrapperError, match=error):
        docker_run_wrapper("img", **kwargs)  # type: ignore[arg-type]

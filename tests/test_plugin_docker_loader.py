"""Offline regression tests for DockerPluginLoader; no Docker daemon needed."""
from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from kinetic_sdk import __version__
from kinetic_sdk.plugin import DockerPluginLoader, PluginLoadError, PluginManifest


def _manifest(tmp_path: Path, **kwargs: object) -> PluginManifest:
    values: dict[str, object] = dict(
        name="demo", version="1", entry_point="plugin:build", source="directory",
        directory=str(tmp_path), isolation="docker",
    )
    values.update(kwargs)
    return PluginManifest(**values)  # type: ignore[arg-type]


def test_docker_loader_build_failure_has_exit_and_stderr(tmp_path: Path) -> None:
    def runner(*_args: object, **_kwargs: object) -> object:
        return SimpleNamespace(returncode=17, stderr="x" * 600)

    with pytest.raises(PluginLoadError, match=r"build failed \(exit 17\)"):
        DockerPluginLoader(runner=runner).load(_manifest(tmp_path))


def test_docker_loader_uses_least_privilege_run_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[object] = []
    dockerfile: list[str] = []
    timeouts: list[object] = []

    class FakeTransport:
        def __init__(self, command: str, args: list[str], **_kwargs: object) -> None:
            seen.extend([command, args])
        def close(self) -> None: pass
        def send(self, _message: object) -> None: pass

    class FakeClient:
        def __init__(self, _transport: object) -> None: pass
        def initialize(self) -> dict[str, object]: return {"protocolVersion": "x"}
        def list_tools(self) -> list[dict[str, object]]: return []
        def close(self) -> None: pass

    monkeypatch.setattr("kinetic_sdk.plugin.docker_loader.StdioTransport", FakeTransport)
    monkeypatch.setattr("kinetic_sdk.plugin.docker_loader.MCPClient", FakeClient)
    def runner(argv: list[str], **kwargs: object) -> object:
        timeouts.append(kwargs["timeout"])
        if "build" in argv:
            dockerfile.append(Path(argv[argv.index("-f") + 1]).read_text())
        return SimpleNamespace(returncode=0, stderr="")

    loader = DockerPluginLoader(runner=runner)
    assert loader.load(_manifest(tmp_path)) == []
    args = seen[1]
    assert isinstance(args, list)
    assert ["--network", "none"] == args[args.index("--network"):args.index("--network") + 2]
    assert "--read-only" in args and ["--memory", "512m"] == args[args.index("--memory"):args.index("--memory") + 2]
    assert ["--pids-limit", "128"] == args[args.index("--pids-limit"):args.index("--pids-limit") + 2]
    assert "--cap-drop=ALL" in args
    assert "--security-opt=no-new-privileges:true" in args
    assert f"kinetic-agent-sdk=={__version__}" in dockerfile[0]
    assert "USER plugin" in dockerfile[0]
    assert timeouts == [300.0]
    loader.close()


def test_docker_loader_converts_build_timeout_to_plugin_load_error(tmp_path: Path) -> None:
    def runner(*_args: object, **_kwargs: object) -> object:
        raise subprocess.TimeoutExpired("docker build", 300)

    with pytest.raises(PluginLoadError, match="build timed out"):
        DockerPluginLoader(runner=runner).load(_manifest(tmp_path))


def test_docker_isolation_rejects_entry_point_plugin() -> None:
    with pytest.raises(Exception, match="requires a directory"):
        PluginManifest(name="demo", version=None, entry_point="demo:build", source="entry_point", isolation="docker")

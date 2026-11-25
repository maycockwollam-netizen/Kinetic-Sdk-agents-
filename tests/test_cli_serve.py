"""Tests for the ``kinetic serve`` web UI command."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from kinetic_sdk.cli.main import DEFAULT_HOST, DEFAULT_PORT, _ui_dir

REPO_ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def test_ui_dir_resolves_to_packaged_ui() -> None:
    ui_dir = _ui_dir()
    assert ui_dir is not None
    assert (ui_dir / "index.html").is_file()
    assert (ui_dir / "app.js").is_file()
    assert (ui_dir / "style.css").is_file()


def test_serve_defaults_are_sane() -> None:
    assert DEFAULT_HOST == "127.0.0.1"
    assert isinstance(DEFAULT_PORT, int) and 0 < DEFAULT_PORT < 65536


def test_serve_end_to_end(tmp_path) -> None:
    """Boot ``python -m kinetic_sdk.cli serve`` and chat with the UI/API."""
    port = _free_port()
    # Isolate the child from any host config/keys so it falls back to the
    # mock LLM (no network).
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "PYTHONPATH": str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    for key in (
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY",
        "GROQ_API_KEY", "OPENHANDS_API_KEY", "KINETIC_MODEL",
    ):
        env.pop(key, None)

    proc = subprocess.Popen(
        [sys.executable, "-m", "kinetic_sdk.cli", "serve", "--port", str(port), "--no-open"],
        cwd=str(tmp_path),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        root = f"http://127.0.0.1:{port}"
        deadline = time.time() + 20
        health = None
        while time.time() < deadline:
            if proc.poll() is not None:
                (output, _) = proc.communicate()
                raise AssertionError(f"serve exited early:\n{output}")
            try:
                with urllib.request.urlopen(f"{root}/health", timeout=1) as resp:
                    if resp.status == 200:
                        health = json.loads(resp.read())
                        break
            except Exception:
                time.sleep(0.3)
        assert health is not None, "serve did not become healthy in time"

        with urllib.request.urlopen(f"{root}/", timeout=5) as resp:
            html = resp.read().decode("utf-8")
        assert "<title>Kinetic" in html
        assert "/static/app.js" in html

        # Settings API is wired for the UI's Settings → LLM group.
        with urllib.request.urlopen(f"{root}/settings", timeout=5) as resp:
            settings_body = json.loads(resp.read())
        assert "model" in settings_body
        assert "api_key" not in settings_body  # secrets never returned

        # A full chat turn through the API (mock LLM answers).
        request = urllib.request.Request(
            f"{root}/runs",
            data=json.dumps({"message": "ping", "session_id": "e2e"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as resp:
            record = json.loads(resp.read())
        assert record["status"] == "completed"
        assert isinstance(record["final"], str)

        # Feedback endpoint records a rating for that run.
        fb = urllib.request.Request(
            f"{root}/runs/{record['run_id']}/feedback",
            data=json.dumps({"rating": "up"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(fb, timeout=10) as resp:
            fb_body = json.loads(resp.read())
        assert fb_body["ok"] is True
        assert fb_body["feedback"]["rating"] == "up"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
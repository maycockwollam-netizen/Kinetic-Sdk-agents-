"""Tests for named conversation sessions (conversation/sessions.py + CLI)."""

from __future__ import annotations

import argparse

import pytest

from kinetic_sdk.conversation.state import ConversationState


def _state(*, system: str = "sys", messages=None) -> ConversationState:
    return ConversationState(system_prompt=system, messages=messages or [])


def test_session_store_roundtrip_list_delete(tmp_path) -> None:
    from kinetic_sdk.conversation.sessions import SessionStore

    store = SessionStore(tmp_path)
    assert store.list() == []

    store.open("alpha").save(_state(messages=[{"role": "user", "content": "hi"}]))
    store.open("beta").save(_state(messages=[]))

    infos = store.list()
    assert {info.id for info in infos} == {"alpha", "beta"}
    alpha = next(info for info in infos if info.id == "alpha")
    assert alpha.message_count == 1

    loaded = store.load("alpha")
    assert loaded is not None
    assert loaded.messages == [{"role": "user", "content": "hi"}]

    assert store.delete("alpha") is True
    assert store.load("alpha") is None
    assert store.delete("alpha") is False
    assert {info.id for info in store.list()} == {"beta"}


def test_session_store_invalid_ids_rejected(tmp_path) -> None:
    from kinetic_sdk.conversation.sessions import SessionStore

    store = SessionStore(tmp_path)
    for bad in ("Bad", "UPPER", "", "with space", "a" * 80):
        with pytest.raises(ValueError):
            store.open(bad)
    # valid ids accepted
    store.open("my-session_1").save(_state())


def test_cmd_session_list(monkeypatch, tmp_path, capsys) -> None:
    import kinetic_sdk.cli.main as main
    from kinetic_sdk.conversation.sessions import SessionStore

    proj = tmp_path / ".kinetic"
    monkeypatch.setattr(main, "PROJECT_DIR", proj)
    store = SessionStore(proj / "sessions")
    store.open("alpha").save(_state(messages=[{"role": "user", "content": "hello"}]))

    args = argparse.Namespace(session_action="list", session_id=None)
    assert main.cmd_session(args) == 0
    assert "alpha" in capsys.readouterr().out


def test_cmd_session_show_and_delete(monkeypatch, tmp_path, capsys) -> None:
    import kinetic_sdk.cli.main as main
    from kinetic_sdk.conversation.sessions import SessionStore

    proj = tmp_path / ".kinetic"
    monkeypatch.setattr(main, "PROJECT_DIR", proj)
    store = SessionStore(proj / "sessions")
    store.open("work").save(_state(messages=[{"role": "user", "content": "hello"}]))

    args = argparse.Namespace(session_action="show", session_id="work")
    assert main.cmd_session(args) == 0
    assert "hello" in capsys.readouterr().out

    args = argparse.Namespace(session_action="show", session_id="missing")
    assert main.cmd_session(args) == 1

    args = argparse.Namespace(session_action="delete", session_id="work")
    assert main.cmd_session(args) == 0
    assert store.load("work") is None
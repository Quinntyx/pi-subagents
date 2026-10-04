"""Stable tmux session placement, including a deterministic user rename."""

from __future__ import annotations

import shlex

import pytest

from pi_subagents import envcheck, tmuxenv


@pytest.mark.parametrize("cached_name", ["2", "parent session"])
def test_spawn_survives_user_renaming_cached_tmux_session(tmp_path, monkeypatch, cached_name):
    placement = envcheck.TmuxPlacement(cached_name, "$2", "@3", "%4", "socket")
    monkeypatch.setattr(tmuxenv, "require_environment", lambda: placement)
    monkeypatch.setattr(tmuxenv, "resolve_agent_dir", lambda _: tmp_path)
    monkeypatch.setattr(tmuxenv, "socket_dir", lambda: tmp_path)
    monkeypatch.setattr(tmuxenv.shutil, "which", lambda _: "pi")
    targets = []
    current_name = cached_name

    def fake_tmux(*args):
        assert args[0] == "new-window"
        target = args[args.index("-t") + 1]
        targets.append(target)
        # Model tmux's exact regression: an old name no longer identifies the
        # session, but its ID is unchanged and resolves before/after rename.
        if target not in (f"{current_name}:", "$2:"):
            raise RuntimeError(f"can't find session: {cached_name}")
        command = shlex.split(args[-1])
        if len(targets) == 3:
            assert command[command.index("--session") + 1] == str(session_file)
            assert "third prompt is socket-only" not in command
        return f"@{len(targets) + 10}"

    monkeypatch.setattr(tmuxenv, "tmux", fake_tmux)
    kwargs = dict(name="worker", cwd=str(tmp_path), window_name="worker", model=None,
                  thinking=None, socket_name="worker", depth=1)
    first = tmuxenv.spawn_pi_window("first", **kwargs)
    current_name = "renamed-by-user"
    second = tmuxenv.spawn_pi_window("second", **kwargs)
    session_file = tmp_path / "retained session's transcript.jsonl"
    third = tmuxenv.spawn_pi_window("third prompt is socket-only", session_file=str(session_file), **kwargs)
    assert placement.session_name == cached_name, "the stale cache must remain stale in this regression"
    assert targets == ["$2:"] * 3
    assert len({first.window_id, second.window_id, third.window_id}) == 3

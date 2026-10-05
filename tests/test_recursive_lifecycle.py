"""Recursive window admission and owned-tree cleanup without a live tmux server."""

from __future__ import annotations

import concurrent.futures
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from pi_subagents import envcheck, handle as live, tmuxenv
from pi_subagents.errors import PiSubagentsTimeoutError


class Budget:
    def __init__(self):
        self.events = []
        self.records = []
        self.remaining = 10.0
        self.exhausted = False
        self.attach_error = False

    def reserve(self, agent_id, depth):
        self.events.append(("reserve", agent_id, depth))
        if self.exhausted:
            raise RuntimeError("root live budget exhausted")
        return "owned-token"

    def attach(self, token, window_id):
        self.events.append(("attach", token, window_id))
        if self.attach_error:
            raise RuntimeError("subtree closing")

    def child_env(self, token):
        return {"PI_SUBAGENTS_ROOT_STATE": "/private/root.json",
                "PI_SUBAGENTS_ROOT_ID": "root-id",
                "PI_SUBAGENTS_PARENT_TOKEN": token,
                "PI_SUBAGENTS_MAX_DEPTH": "3",
                "PI_SUBAGENTS_MAX_CONCURRENT": "8",
                "PI_SUBAGENTS_ROOT_MAX_CONCURRENT": "8",
                "PI_SUBAGENTS_ROOT_MAX_TASKS": "512",
                "PI_SUBAGENTS_ROOT_TIMEOUT": "1800"}

    def begin_close(self, token):
        self.events.append(("closing", token))

    def descendants(self, token):
        self.events.append(("descendants", token))
        return list(self.records)

    def release(self, token):
        self.events.append(("release", token))
        self.records = [r for r in self.records if r["token"] != token]

    def resume_admission(self, token):
        self.events.append(("resume", token))

    def remaining_seconds(self):
        return self.remaining


@pytest.fixture
def setup(monkeypatch, tmp_path):
    placement = envcheck.TmuxPlacement("s", "$0", "@1", "%1", "socket")
    monkeypatch.setattr(tmuxenv, "require_environment", lambda: placement)
    monkeypatch.setattr(live, "require_environment", lambda: placement)
    monkeypatch.setattr(tmuxenv, "ensure_can_spawn", lambda: 1)
    monkeypatch.setattr(live, "ensure_can_spawn", lambda: 1)
    monkeypatch.setattr(live, "current_depth", lambda: 0)
    monkeypatch.setattr(tmuxenv, "resolve_agent_dir", lambda _: tmp_path)
    budget = Budget()
    monkeypatch.setattr(tmuxenv.RootBudget, "for_environment", lambda: budget)
    return budget


def spawn(tmp_path):
    return tmuxenv.spawn_pi_window("prompt", name="worker", cwd=str(tmp_path),
                                  window_name="worker", model=None, thinking=None,
                                  socket_name="worker", depth=1)


def make_handle(tmp_path, budget):
    h = live.AgentHandle("prompt", name="worker", cwd=str(tmp_path), window_name="worker",
                         model=None, thinking=None, schema=None)
    h._bind(SimpleNamespace(window_id="@2", socket_path=str(tmp_path / "worker.sock"),
                            budget=budget, admission_token="owned-token"))
    return h


def record(token, depth, parent, window):
    return dict(token=token, agent_id=token, depth=depth, parent_token=parent, window_id=window)


def test_recursive_spawn_attaches_and_explicitly_pins_runtime(setup, tmp_path, monkeypatch):
    args = []
    monkeypatch.setenv("PTC_SUBAGENTS_SOURCE", "/stale/checkout")
    monkeypatch.setenv("PTC_PYTHON_EXECUTABLE", "/wrong/python")
    monkeypatch.setattr(tmuxenv, "tmux", lambda *tokens: args.extend(tokens) or "@2")
    ref = spawn(tmp_path)
    assert ref.admission_token == "owned-token" and ref.budget is setup
    for key, value in setup.child_env("owned-token").items():
        assert f"{key}={value}" in args
    assert f"PTC_PYTHON_EXECUTABLE={sys.executable}" in args
    source = Path(tmuxenv.__file__).resolve().parents[2]
    assert f"PTC_SUBAGENTS_SOURCE={source}" in args
    assert "PTC_SUBAGENTS_SOURCE=/stale/checkout" not in args
    assert setup.events == [("reserve", "worker", 1), ("attach", "owned-token", "@2")]
    assert args[-1].startswith("exec env ")
    for key, value in setup.child_env("owned-token").items():
        assert f"{key}={value}" in args[-1]
    assert f"PTC_PYTHON_EXECUTABLE={sys.executable}" in args[-1]


def test_flat_children_reuse_runtime_without_recursive_admission(setup, tmp_path, monkeypatch):
    args = []
    monkeypatch.setattr(tmuxenv.RootBudget, "for_environment", lambda: None)
    monkeypatch.setenv("PI_SUBAGENTS_MAX_DEPTH", "1")
    monkeypatch.setenv("PI_SUBAGENTS_MAX_CONCURRENT", "3")
    monkeypatch.setattr(tmuxenv, "tmux", lambda *tokens: args.extend(tokens) or "@2")
    ref = spawn(tmp_path)
    assert ref.budget is None and ref.admission_token is None
    assert f"PTC_PYTHON_EXECUTABLE={sys.executable}" in args
    assert f"PTC_SUBAGENTS_SOURCE={Path(tmuxenv.__file__).resolve().parents[2]}" in args
    assert "PI_SUBAGENTS_MAX_DEPTH=1" in args
    assert "PI_SUBAGENTS_MAX_CONCURRENT=3" in args
    assert not any("PI_SUBAGENTS_PARENT_TOKEN=" in arg for arg in args)
    assert not setup.events



def test_exhausted_admission_fails_before_tmux_without_waiting(setup, tmp_path, monkeypatch):
    setup.exhausted = True
    monkeypatch.setattr(tmuxenv, "tmux", lambda *args: pytest.fail("must fail before tmux"))
    with pytest.raises(RuntimeError, match="exhausted"):
        spawn(tmp_path)


def test_low_level_spawn_cannot_override_policy_depth(setup, tmp_path):
    with pytest.raises(ValueError, match="policy-approved"):
        tmuxenv.spawn_pi_window("p", name="x", cwd=str(tmp_path), window_name="x", model=None,
                                thinking=None, socket_name="x", depth=-1)
    assert not setup.events


@pytest.mark.parametrize("query_fails", [False, True])
def test_failed_new_window_releases_only_on_confirmed_absence(setup, tmp_path, monkeypatch, query_fails):
    def command(*args):
        if args[0] == "new-window" or query_fails:
            raise RuntimeError("tmux unavailable")
        assert args[0] == "list-panes"
        return ""
    monkeypatch.setattr(tmuxenv, "tmux", command)
    with pytest.raises(RuntimeError):
        spawn(tmp_path)
    assert (("release", "owned-token") in setup.events) is not query_fails


def test_attach_race_terminates_confirmed_own_window_then_releases(setup, tmp_path, monkeypatch):
    setup.attach_error = True
    events = setup.events
    def command(*args):
        if args[0] == "new-window":
            return "@2"
        assert args[0] == "list-panes"
        return "@2\texec env PI_SUBAGENTS_WINDOW_OWNER=owned-token pi\n@99\tpi"
    monkeypatch.setattr(tmuxenv, "tmux", command)
    monkeypatch.setattr(tmuxenv, "kill_window", lambda wid: events.append(("kill", wid)) or True)
    monkeypatch.setattr(tmuxenv, "window_absent", lambda wid: True)
    with pytest.raises(RuntimeError, match="closing"):
        spawn(tmp_path)
    assert events[-2:] == [("kill", "@2"), ("release", "owned-token")]
    assert ("kill", "@99") not in events


def test_cleanup_barrier_and_deepest_first_release(setup, tmp_path, monkeypatch):
    h = make_handle(tmp_path, setup)
    setup.records = [record("grandchild", 3, "child", "@4"),
                     record("child", 2, "owned-token", "@3")]
    monkeypatch.setattr(live, "unload_window", lambda wid: setup.events.append(("unload", wid)) or True)
    assert h._unload_owned_tree()
    assert setup.events[0:2] == [("closing", "owned-token"), ("descendants", "owned-token")]
    assert [e for e in setup.events if e[0] in ("unload", "release")] == [
        ("unload", "@4"), ("release", "grandchild"),
        ("unload", "@3"), ("release", "child"),
        ("unload", "@2"), ("release", "owned-token")]
    assert h._admission_token is None


@pytest.mark.parametrize("wid", [None, "@1", "@2", "@99;kill-server"])
def test_ambiguous_or_unsafe_descendants_never_kill_parent_or_release(setup, tmp_path, monkeypatch, wid):
    h = make_handle(tmp_path, setup)
    setup.records = [record("child", 2, "owned-token", wid)]
    monkeypatch.setattr(live, "unload_window", lambda wid: pytest.fail("unsafe teardown"))
    assert not h._unload_owned_tree()
    assert not any(event[0] == "release" for event in setup.events)


def test_failed_descendant_cleanup_keeps_all_remaining_permits(setup, tmp_path, monkeypatch):
    h = make_handle(tmp_path, setup)
    setup.records = [record("child", 2, "owned-token", "@3")]
    monkeypatch.setattr(live, "unload_window", lambda wid: False)
    assert not h._unload_owned_tree()
    assert h._admission_token == "owned-token"
    assert not any(event[0] == "release" for event in setup.events)


def test_leaf_self_release_during_ancestor_snapshot_cleanup_is_harmless(setup, tmp_path, monkeypatch):
    h = make_handle(tmp_path, setup)
    setup.records = [record("leaf", 2, "owned-token", "@3")]
    def unload(wid):
        if wid == "@3":
            # Leaf hibernated after the ancestor acquired its snapshot.
            setup.release("leaf")
        return True
    monkeypatch.setattr(live, "unload_window", unload)
    assert h._unload_owned_tree()
    assert setup.events.count(("release", "leaf")) == 2
    assert ("release", "owned-token") in setup.events


def test_successful_hibernate_failed_cleanup_is_visible_and_retryable(setup, tmp_path, monkeypatch):
    h = make_handle(tmp_path, setup)
    h.status, h.last_outcome = "settled", "ok"
    path = tmp_path / "session.jsonl"
    path.write_text("")
    session = SimpleNamespace(session_file=str(path), _load=lambda: None)
    monkeypatch.setattr(live.AgentHandle, "session", property(lambda self: session))
    monkeypatch.setattr(h, "state", lambda: {"isIdle": True})
    monkeypatch.setattr(live, "_last_assistant_outcome", lambda _: ("stop", "done"))
    monkeypatch.setattr(live, "unload_window", lambda wid: False)
    assert not h.hibernate()
    assert h.status == "settled" and not h.dormant
    assert h.agent_state()["cleanupError"] and h.phase == "cleanup"
    assert h._admission_token == "owned-token"
    monkeypatch.setattr(live, "unload_window", lambda wid: True)
    assert h.hibernate()
    assert h.dormant and h.agent_state()["cleanupError"] is None


def test_abort_closes_descendants_but_retains_parent_admission(setup, tmp_path, monkeypatch):
    h = make_handle(tmp_path, setup)
    setup.records = [record("child", 2, "owned-token", "@3")]
    monkeypatch.setattr(live, "unload_window", lambda wid: setup.events.append(("unload", wid)) or True)
    monkeypatch.setattr(h._sync, "abort", lambda: {"aborted": True})
    h.abort()
    assert ("unload", "@3") in setup.events
    assert ("release", "child") in setup.events
    assert ("release", "owned-token") not in setup.events




def test_aborted_retained_parent_reopens_admission_before_followup(setup, tmp_path, monkeypatch):
    h = make_handle(tmp_path, setup)
    setup.records = [record("child", 2, "owned-token", "@3")]
    monkeypatch.setattr(live, "unload_window", lambda wid: setup.events.append(("unload", wid)) or True)
    monkeypatch.setattr(h._sync, "abort", lambda: {"aborted": True})
    monkeypatch.setattr(h._sync, "message", lambda: None)
    monkeypatch.setattr(h._sync, "send", lambda text, **kw: setup.events.append(("send", text)))
    h.abort()
    assert h.resume("continue") is h
    assert h.status == "running" and h._admission_token == "owned-token"
    assert setup.events.index(("resume", "owned-token")) < setup.events.index(("send", "continue"))
    assert ("release", "owned-token") not in setup.events


def test_retained_resume_rejects_closing_ancestor_before_prompt(setup, tmp_path, monkeypatch):
    h = make_handle(tmp_path, setup)
    h.status = "failed"
    sent = []
    monkeypatch.setattr(h._sync, "send", lambda *a, **kw: sent.append(a))
    monkeypatch.setattr(setup, "resume_admission", lambda token: (_ for _ in ()).throw(
        RuntimeError("ancestor subtree is closing")))
    with pytest.raises(RuntimeError, match="ancestor subtree is closing"):
        h.resume("continue")
    assert not sent and h.status == "failed"


def test_reopen_rechecks_policy_and_takes_fresh_admission(setup, tmp_path, monkeypatch):
    h = make_handle(tmp_path, setup)
    h._admission_token = None
    h._dormant = True
    path = tmp_path / "session.jsonl"
    path.write_text("")
    h.session_file = str(path)
    commands = []
    monkeypatch.setattr(tmuxenv, "tmux", lambda *args: commands.append(args) or "@5")
    monkeypatch.setattr(live.threading.Thread, "start", lambda _: None)
    h._reopen("continue")
    assert h.window_id == "@5" and h._admission_token == "owned-token"
    assert setup.events[0][0] == "reserve"
    h._admission_token = None
    monkeypatch.setattr(live, "ensure_can_spawn", lambda: (_ for _ in ()).throw(RuntimeError("depth boundary")))
    with pytest.raises(RuntimeError, match="depth boundary"):
        h._reopen("continue")
    assert len(commands) == 1


def test_root_remaining_deadline_bounds_startup_and_settlement(setup, tmp_path, monkeypatch):
    h = make_handle(tmp_path, setup)
    setup.remaining = 0.01
    assert h._bounded_timeout(90) == 0.01
    h._startup_started = True
    h._ready = concurrent.futures.Future()
    setup.remaining = 0
    h._run_startup()
    assert isinstance(h._startup_error, PiSubagentsTimeoutError)
    assert h.status == "failed" and h._admission_token == "owned-token"
    with pytest.raises(PiSubagentsTimeoutError, match="root deadline"):
        h._bounded_timeout(1800)


def test_unknown_tmux_query_is_not_proof_of_termination(setup, monkeypatch):
    monkeypatch.setattr(tmuxenv, "tmux", lambda *args: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(live, "kill_window", lambda wid: False)
    assert not tmuxenv.window_absent("@2")
    assert tmuxenv.window_alive("@2")
    assert not live.unload_window("@2")


@pytest.mark.parametrize("wid", ["@1", "@2;kill-server", "caller", ""])
def test_kill_boundary_never_targets_caller_or_expressions(setup, monkeypatch, wid):
    monkeypatch.setattr(tmuxenv, "tmux", lambda *args: pytest.fail("unsafe kill"))
    assert not tmuxenv.kill_window(wid)

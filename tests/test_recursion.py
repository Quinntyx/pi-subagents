"""Broker-only tests: no real tmux windows, process killing, or Pi launches."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import stat
import time
from pathlib import Path

import pytest

from pi_subagents import recursion as rec


@pytest.fixture
def root_environment(monkeypatch, tmp_path):
    for name in tuple(os.environ):
        if name.startswith("PI_SUBAGENTS_") or name == "PI_SUBAGENT_DEPTH":
            monkeypatch.delenv(name)
    runtime = tmp_path / "runtime"
    runtime.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    monkeypatch.setenv("PI_SUBAGENTS_MAX_DEPTH", "4")
    identity = (os.getpid(), rec._process_info(os.getpid())[1])
    # Unit-test stand-in for the actual Pi ancestor; forked workers retain the
    # same captured identity, rather than accidentally selecting their own PID.
    monkeypatch.setattr(rec, "_find_root_pi", lambda: identity)
    return identity


def _root():
    budget = rec.RootBudget.for_environment()
    assert budget is not None
    return budget


def _child(monkeypatch, env):
    with monkeypatch.context() as child_env:
        for name, value in env.items():
            child_env.setenv(name, value)
        return _root()


def _state(budget):
    return json.loads(budget.state_path.read_text())


def _change_state(budget, **changes):
    state = _state(budget)
    state.update(changes)
    budget.state_path.write_text(json.dumps(state))


def _join(process):
    process.join(8)
    assert not process.is_alive(), "broker worker blocked instead of failing promptly"
    assert process.exitcode == 0


def _admission_worker(index, start, results):
    start.wait(5)
    try:
        budget = rec.RootBudget.for_environment()
        token = budget.reserve(f"worker-{index}", 1)
        budget.attach(token, f"@{index + 10}")
        results.put(("ok", token, str(budget.state_path)))
    except rec.RootBudgetError as error:
        results.put(("refused", str(error)))


def _pending_worker(results):
    budget = rec.RootBudget.for_environment()
    token = budget.reserve("pending", 1)
    results.put(token)
    # Deliberately exit without releasing: launcher death is not confirmation
    # that no window was created.


def _nested_worker(env, results):
    os.environ.update(env)
    started = time.monotonic()
    try:
        rec.RootBudget.for_environment().reserve("nested", 2)
        results.put(("unexpected admission", time.monotonic() - started))
    except rec.RootBudgetError as error:
        results.put((str(error), time.monotonic() - started))


def _closing_race_worker(env, ready, proceed, results):
    os.environ.update(env)
    budget = rec.RootBudget.for_environment()
    token = budget.reserve("in-flight", 2)
    ready.put(token)
    proceed.wait(5)
    try:
        budget.attach(token, "@77")
        results.put("unexpected attachment")
    except rec.RootBudgetError as error:
        results.put(str(error))
    # This reserving launcher alone can confirm its fake startup made no window.
    budget.release(token)


def _hold_process(stop):
    stop.wait(8)


def _duplicate_release_worker(token, start, results):
    budget = rec.RootBudget.for_environment()
    start.wait(5)
    try:
        budget.release(token)
        results.put("released")
    except rec.RootBudgetError as error:
        results.put(str(error))


@pytest.mark.parametrize("value", ["-1", "", "bad", "1.0", "+1", " 1", "1 ", "٠"])
def test_depth_is_strict(root_environment, monkeypatch, value):
    monkeypatch.setenv("PI_SUBAGENT_DEPTH", value)
    with pytest.raises(rec.RecursionPolicyError, match="PI_SUBAGENT_DEPTH"):
        rec.current_depth()
    with pytest.raises(rec.RecursionPolicyError):
        rec.RootBudget.for_environment()


@pytest.mark.parametrize("value", ["-1", "0", "", "bad", "1.5", "+2", " 2"])
def test_max_depth_is_strict(root_environment, monkeypatch, value):
    monkeypatch.setenv("PI_SUBAGENTS_MAX_DEPTH", value)
    with pytest.raises(rec.RecursionPolicyError, match="PI_SUBAGENTS_MAX_DEPTH"):
        rec.max_depth()


def test_flat_default_and_spawn_boundary(root_environment, monkeypatch):
    monkeypatch.delenv("PI_SUBAGENTS_MAX_DEPTH")
    assert rec.current_depth() == 0
    assert rec.max_depth() == 1
    assert rec.RootBudget.for_environment() is None
    assert rec.ensure_can_spawn() == 1
    monkeypatch.setenv("PI_SUBAGENT_DEPTH", "1")
    # Module use/import is legal, only spawning is rejected.
    assert rec.current_depth() == 1
    with pytest.raises(NotImplementedError, match="cannot spawn at depth 1"):
        rec.ensure_can_spawn()
    monkeypatch.setenv("PI_SUBAGENTS_MAX_DEPTH", "2")
    assert rec.ensure_can_spawn() == 2
    monkeypatch.setenv("PI_SUBAGENT_DEPTH", "2")
    with pytest.raises(NotImplementedError):
        rec.ensure_can_spawn()


@pytest.mark.parametrize("name,value", [
    ("PI_SUBAGENTS_MAX_CONCURRENT", "0"),
    ("PI_SUBAGENTS_ROOT_MAX_CONCURRENT", "-2"),
    ("PI_SUBAGENTS_ROOT_MAX_TASKS", "bad"),
    ("PI_SUBAGENTS_ROOT_TIMEOUT", "0"),
    ("PI_SUBAGENTS_ROOT_TIMEOUT", "-1"),
    ("PI_SUBAGENTS_ROOT_TIMEOUT", "nan"),
    ("PI_SUBAGENTS_ROOT_TIMEOUT", "inf"),
    ("PI_SUBAGENTS_ROOT_TIMEOUT", "1e3"),
])
def test_invalid_recursive_limits_fail(root_environment, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(rec.RecursionPolicyError):
        _root()


def test_shared_multiprocessing_cap_and_unique_records(root_environment, monkeypatch):
    monkeypatch.setenv("PI_SUBAGENTS_ROOT_MAX_CONCURRENT", "3")
    context = mp.get_context("fork")
    start, results = context.Event(), context.Queue()
    workers = [context.Process(target=_admission_worker, args=(i, start, results)) for i in range(12)]
    # Also exercise simultaneous *first* root initialization.
    for worker in workers:
        worker.start()
    start.set()
    outcomes = [results.get(timeout=8) for _ in workers]
    for worker in workers:
        _join(worker)
    admitted = [outcome for outcome in outcomes if outcome[0] == "ok"]
    refused = [outcome for outcome in outcomes if outcome[0] == "refused"]
    assert len(admitted) == 3
    assert len(refused) == 9
    assert all("live-window budget exhausted" in outcome[1] for outcome in refused)
    assert len({outcome[1] for outcome in admitted}) == 3
    assert len({outcome[2] for outcome in admitted}) == 1
    budget = _root()
    first = _state(budget)
    assert first["tasks"] == 3
    assert len(first["records"]) == 3
    assert first["root_pid"] == root_environment[0]
    assert first["root_starttime"] == root_environment[1]
    assert _state(_root()) == first
    # Attached records can be released after the original Python launcher exits;
    # their permits were NOT automatically reclaimed by that exit.
    for _, token, _ in admitted:
        budget.release(token)
    assert _state(budget)["tasks"] == 3
    assert _state(budget)["records"] == {}


def test_parent_awaiting_child_exhaustion_is_prompt(root_environment, monkeypatch):
    monkeypatch.setenv("PI_SUBAGENTS_ROOT_MAX_CONCURRENT", "1")
    budget = _root()
    parent = budget.reserve("parent", 1)
    budget.attach(parent, "@1")
    env = budget.child_env(parent)
    context = mp.get_context("fork")
    results = context.Queue()
    worker = context.Process(target=_nested_worker, args=(env, results))
    worker.start()
    message, elapsed = results.get(timeout=8)
    _join(worker)
    assert "live-window budget exhausted" in message
    assert elapsed < 2
    assert len(_state(budget)["records"]) == 1


def test_child_env_is_complete_immutable_and_parent_checked(root_environment, monkeypatch):
    monkeypatch.setenv("PI_SUBAGENTS_MAX_CONCURRENT", "3")
    monkeypatch.setenv("PI_SUBAGENTS_ROOT_MAX_TASKS", "9")
    monkeypatch.setenv("PI_SUBAGENTS_ROOT_TIMEOUT", "12.5")
    budget = _root()
    token = budget.reserve("parent", 1)
    env = budget.child_env(token)
    assert Path(env["PI_SUBAGENTS_ROOT_STATE"]).is_absolute()
    assert env["PI_SUBAGENTS_ROOT_MAX_CONCURRENT"] == "3"
    assert env["PI_SUBAGENTS_MAX_CONCURRENT"] == "3"
    assert env["PI_SUBAGENTS_MAX_DEPTH"] == "4"
    assert env["PI_SUBAGENTS_ROOT_MAX_TASKS"] == "9"
    assert env["PI_SUBAGENTS_ROOT_TIMEOUT"] == "12.5"
    assert env["PI_SUBAGENT_DEPTH"] == "1"
    assert env["PI_SUBAGENTS_PARENT_TOKEN"] == token
    assert _child(monkeypatch, env).state_path == budget.state_path
    for name in budget._policy:
        altered = {**env, name: "99"}
        with pytest.raises(rec.RootBudgetError, match="immutable"):
            _child(monkeypatch, altered)
    with pytest.raises(rec.RootBudgetError, match="wrong depth"):
        _child(monkeypatch, {**env, "PI_SUBAGENT_DEPTH": "2"})
    with pytest.raises(rec.RootBudgetError, match="parent token is missing"):
        _child(monkeypatch, {**env, "PI_SUBAGENTS_PARENT_TOKEN": "f" * 32})
    with pytest.raises(rec.RootBudgetError, match="generation"):
        _child(monkeypatch, {**env, "PI_SUBAGENTS_ROOT_ID": "f" * 32})


@pytest.mark.parametrize("present", [
    {"PI_SUBAGENTS_ROOT_STATE"}, {"PI_SUBAGENTS_ROOT_ID"},
    {"PI_SUBAGENTS_PARENT_TOKEN"},
    {"PI_SUBAGENTS_ROOT_STATE", "PI_SUBAGENTS_ROOT_ID"},
])
def test_partial_inheritance_cannot_bypass_flat(root_environment, monkeypatch, present):
    monkeypatch.setenv("PI_SUBAGENTS_MAX_DEPTH", "1")
    for name in present:
        monkeypatch.setenv(name, "anything")
    with pytest.raises(rec.RootBudgetError, match="incomplete"):
        rec.RootBudget.for_environment()


def test_full_inheritance_cannot_bypass_flat_or_reset_depth(root_environment, monkeypatch):
    budget = _root()
    token = budget.reserve("parent", 1)
    env = budget.child_env(token)
    for altered in ({**env, "PI_SUBAGENTS_MAX_DEPTH": "1"}, {**env, "PI_SUBAGENT_DEPTH": "0"}):
        with pytest.raises(rec.RootBudgetError, match="requires recursive mode and child depth"):
            _child(monkeypatch, altered)
    monkeypatch.setenv("PI_SUBAGENT_DEPTH", "1")
    with pytest.raises(rec.RootBudgetError, match="missing inherited"):
        rec.RootBudget.for_environment()


@pytest.mark.parametrize("bad", ["{", "[]", "null", '{"version":1,"version":1}', '{"version":true}'])
def test_malformed_state_is_not_recreated(root_environment, bad):
    budget = _root()
    budget.state_path.write_text(bad)
    with pytest.raises(rec.RootBudgetError):
        _root()
    assert budget.state_path.read_text() == bad


def test_missing_inherited_state_is_not_recreated(root_environment, monkeypatch):
    budget = _root()
    token = budget.reserve("parent", 1)
    env = budget.child_env(token)
    budget.state_path.rename(budget.state_path.with_suffix(".saved"))
    with pytest.raises(rec.RootBudgetError, match="missing"):
        _child(monkeypatch, env)
    assert not budget.state_path.exists()
    # A root kernel must not reset its task count/deadline/generation either.
    with pytest.raises(rec.RootBudgetError, match="missing"):
        _root()
    assert not budget.state_path.exists()


def test_permissions_and_state_symlink_fail_closed(root_environment):
    budget = _root()
    assert stat.S_IMODE(budget.state_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(budget.state_path.parent.stat().st_mode) == 0o700
    lock = budget.state_path.with_suffix(".json.lock")
    assert stat.S_IMODE(lock.stat().st_mode) == 0o600
    lock_inode = lock.stat().st_ino
    token = budget.reserve("one", 1)
    budget.release(token)
    assert lock.stat().st_ino == lock_inode
    budget.state_path.chmod(0o644)
    with pytest.raises(rec.RootBudgetError, match="mode 0600"):
        budget.reserve("bad-permissions", 1)
    budget.state_path.chmod(0o600)
    saved = budget.state_path.with_suffix(".saved")
    budget.state_path.rename(saved)
    budget.state_path.symlink_to(saved)
    with pytest.raises(rec.RootBudgetError, match="cannot read"):
        budget.reserve("symlink", 1)


def test_lock_symlink_and_directory_permissions_fail_closed(root_environment):
    budget = _root()
    lock = budget.state_path.with_suffix(".json.lock")
    saved = lock.with_suffix(".saved")
    lock.rename(saved)
    lock.symlink_to(saved)
    with pytest.raises(rec.RootBudgetError, match="cannot open"):
        budget.reserve("symlink", 1)
    budget.state_path.parent.chmod(0o755)
    with pytest.raises(rec.RootBudgetError, match="mode 0700"):
        budget.reserve("public-directory", 1)


def test_stale_pid_identity_refuses_spawn_but_allows_owned_cleanup(root_environment):
    budget = _root()
    token = budget.reserve("one", 1)
    budget.attach(token, "@1")
    _change_state(budget, root_starttime="0")
    with pytest.raises(rec.RootBudgetError, match="identity does not match"):
        budget.reserve("stale", 1)
    # Mutated state must not become cleanup authority either.
    with pytest.raises(rec.RootBudgetError, match="identity does not match"):
        budget.release(token)


def test_root_death_allows_authorized_cleanup_not_admission(root_environment, monkeypatch):
    context = mp.get_context("fork")
    stop = context.Event()
    process = context.Process(target=_hold_process, args=(stop,))
    process.start()
    identity = (process.pid, rec._process_info(process.pid)[1])
    monkeypatch.setattr(rec, "_find_root_pi", lambda: identity)
    budget = _root()
    parent = budget.reserve("parent", 1)
    budget.attach(parent, "@1")
    env = budget.child_env(parent)
    child = _child(monkeypatch, env)
    nested = child.reserve("nested", 2)
    child.attach(nested, "@2")
    stop.set()
    _join(process)
    with pytest.raises(rec.RootBudgetError, match="missing or unreadable"):
        budget.reserve("dead", 1)
    # A restarted child kernel can still obtain its existing cleanup authority.
    restarted = _child(monkeypatch, env)
    assert budget.begin_close(parent) == budget.descendants(parent)
    restarted.release(nested)
    budget.release(parent)
    assert _state(budget)["records"] == {}


def test_actual_pi_ancestor_not_python_pid(monkeypatch):
    monkeypatch.setattr(os, "getppid", lambda: 400)
    chain = {400: (300, "python-start", "python"), 300: (200, "pi-start", "pi")}
    monkeypatch.setattr(rec, "_process_info", lambda pid: chain[pid])
    assert rec._find_root_pi() == (300, "pi-start")


def test_unrelated_roots_are_isolated(root_environment, monkeypatch):
    first = _root()
    token = first.reserve("first", 1)
    context = mp.get_context("fork")
    stop = context.Event()
    process = context.Process(target=_hold_process, args=(stop,))
    process.start()
    identity = (process.pid, rec._process_info(process.pid)[1])
    monkeypatch.setattr(rec, "_find_root_pi", lambda: identity)
    second = _root()
    assert first.state_path != second.state_path
    assert _state(first)["root_id"] != _state(second)["root_id"]
    assert _state(second)["records"] == {}
    with pytest.raises(rec.RootBudgetError, match="unknown"):
        second.release(token)
    second_token = second.reserve("second", 1)
    second.release(second_token)
    assert token in _state(first)["records"]
    stop.set()
    _join(process)


def test_descendants_are_stable_deepest_first_and_release_is_ordered(root_environment, monkeypatch):
    budget = _root()
    parent = budget.reserve("parent", 1)
    sibling = budget.reserve("unrelated-sibling", 1)
    child = _child(monkeypatch, budget.child_env(parent))
    nested = child.reserve("nested", 2)
    grandchild = _child(monkeypatch, child.child_env(nested))
    leaf = grandchild.reserve("leaf", 3)
    snapshot = budget.descendants(parent)
    assert [record["token"] for record in snapshot] == [leaf, nested]
    assert all(set(record) == {"token", "agent_id", "depth", "parent_token", "window_id"} for record in snapshot)
    snapshot[0]["agent_id"] = "cannot mutate shared state"
    assert budget.descendants(parent)[0]["agent_id"] == "leaf"
    with pytest.raises(rec.RootBudgetError, match="live descendants"):
        budget.release(parent)
    with pytest.raises(rec.RootBudgetError, match="not an owned descendant"):
        child.release(sibling)
    with pytest.raises(rec.RootBudgetError, match="not an owned descendant"):
        child.release(parent)
    for record in budget.begin_close(parent):
        budget.release(record["token"])
    budget.release(parent)
    # A child whose parent was closed can still acknowledge its already-released
    # owned leaf; ancestry survives in bounded tombstones.
    grandchild.release(leaf)
    child.release(nested)
    budget.release(parent)
    assert budget.begin_close(parent) == []
    assert set(_state(budget)["records"]) == {sibling}


def test_begin_close_atomic_barrier_blocks_reserve_and_attach(root_environment, monkeypatch):
    budget = _root()
    parent = budget.reserve("parent", 1)
    env = budget.child_env(parent)
    child = _child(monkeypatch, env)
    pending = child.reserve("pending", 2)
    assert [record["token"] for record in budget.begin_close(parent)] == [pending]
    assert budget.begin_close(parent) == budget.descendants(parent)
    with pytest.raises(rec.RootBudgetError, match="closing parent"):
        child.reserve("too-late", 2)
    with pytest.raises(rec.RootBudgetError, match="closing"):
        child.attach(pending, "@3")
    with pytest.raises(rec.RootBudgetError, match="closing"):
        budget.child_env(parent)
    child.release(pending)
    budget.release(parent)

def test_resume_admission_requires_resolved_cleanup_and_open_ancestors(root_environment, monkeypatch):
    budget = _root()
    parent = budget.reserve("parent", 1)
    budget.attach(parent, "@2")
    child = _child(monkeypatch, budget.child_env(parent))
    pending = child.reserve("pending", 2)
    budget.begin_close(parent)
    with pytest.raises(rec.RootBudgetError, match="unresolved"):
        budget.resume_admission(parent)
    child.release(pending)
    budget.resume_admission(parent)
    assert parent not in _state(budget)["closing"]
    leaf = child.reserve("after-resume", 2)
    child.attach(leaf, "@3")
    budget.begin_close(parent)
    with pytest.raises(rec.RootBudgetError, match="closing ancestor"):
        child.resume_admission(leaf)
    child.release(leaf)
    budget.resume_admission(parent)
    budget.resume_admission(parent)  # safe idempotent continuation
    budget.release(parent)



def test_multiprocess_close_vs_inflight_attach(root_environment):
    budget = _root()
    parent = budget.reserve("parent", 1)
    env = budget.child_env(parent)
    context = mp.get_context("fork")
    ready, results, proceed = context.Queue(), context.Queue(), context.Event()
    worker = context.Process(target=_closing_race_worker, args=(env, ready, proceed, results))
    worker.start()
    pending = ready.get(timeout=8)
    assert budget.begin_close(parent)[0]["token"] == pending
    with pytest.raises(rec.RootBudgetError, match="pending startup may be in flight"):
        budget.release(pending)
    proceed.set()
    assert "reservation is closing" in results.get(timeout=8)
    _join(worker)
    assert budget.descendants(parent) == []
    budget.release(parent)


def test_pending_launcher_exit_does_not_reclaim_or_authorize_release(root_environment):
    budget = _root()
    context = mp.get_context("fork")
    results = context.Queue()
    worker = context.Process(target=_pending_worker, args=(results,))
    worker.start()
    pending = results.get(timeout=8)
    _join(worker)
    assert pending in _state(budget)["records"]
    budget.begin_close(pending)
    with pytest.raises(rec.RootBudgetError, match="only its reserving launcher"):
        budget.release(pending)
    assert pending in _state(budget)["records"]


def test_concurrent_duplicate_release_is_safe_and_does_not_reset_budget(root_environment):
    budget = _root()
    parent = budget.reserve("parent", 1)
    budget.attach(parent, "@1")
    context = mp.get_context("fork")
    start, results = context.Event(), context.Queue()
    workers = [context.Process(target=_duplicate_release_worker, args=(parent, start, results)) for _ in range(2)]
    for worker in workers:
        worker.start()
    start.set()
    assert [results.get(timeout=8) for _ in workers] == ["released", "released"]
    for worker in workers:
        _join(worker)
    budget.release(parent)
    state = _state(budget)
    assert state["records"] == {}
    assert state["tasks"] == 1
    assert set(state["released"]) == {parent}
    root_id, deadline = state["root_id"], state["deadline"]
    reloaded = _state(_root())
    assert reloaded["root_id"] == root_id
    assert reloaded["deadline"] == deadline
    assert reloaded["tasks"] == 1
    with pytest.raises(rec.RootBudgetError, match="unknown"):
        budget.release("e" * 32)


def test_pid_reuse_denies_admission_but_retains_cleanup_authority(root_environment, monkeypatch):
    budget = _root()
    token = budget.reserve("one", 1)
    budget.attach(token, "@1")
    original_info = rec._process_info
    def reused_info(pid):
        parent, started, comm = original_info(pid)
        return parent, "999999999" if pid == root_environment[0] else started, comm
    monkeypatch.setattr(rec, "_process_info", reused_info)
    with pytest.raises(rec.RootBudgetError, match="PID was reused"):
        budget.reserve("reused", 1)
    budget.begin_close(token)
    budget.release(token)


def test_task_budget_counts_admissions_including_reopens(root_environment, monkeypatch):
    monkeypatch.setenv("PI_SUBAGENTS_ROOT_MAX_TASKS", "2")
    budget = _root()
    for _ in range(2):
        token = budget.reserve("same-session-reopened", 1)
        budget.release(token)
    assert _state(budget)["records"] == {}
    assert _state(budget)["tasks"] == 2
    with pytest.raises(rec.RootBudgetError, match="task budget exhausted"):
        _root().reserve("third", 1)


def test_deadline_remaining_and_cleanup_after_expiry(root_environment, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(rec.time, "monotonic", lambda: now[0])
    monkeypatch.setenv("PI_SUBAGENTS_ROOT_TIMEOUT", "5")
    budget = _root()
    token = budget.reserve("one", 1)
    assert budget.remaining_seconds() == 5.0
    now[0] = 103.5
    assert _root().remaining_seconds() == 1.5
    now[0] = 106.0
    assert budget.remaining_seconds() == 0.0
    with pytest.raises(rec.RootBudgetError, match="deadline exhausted"):
        budget.reserve("late", 1)
    assert budget.begin_close(token) == []
    budget.release(token)


def test_root_deadline_excludes_quota_wait_union_and_historical_baseline(root_environment, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(rec.time, "monotonic", lambda: now[0])
    monkeypatch.setenv("PI_SUBAGENTS_ROOT_TIMEOUT", "1")
    budget = _root()
    a = budget.reserve("first-clock", 1)
    b = budget.reserve("different-clock", 1)
    # Independent child counters (including historical waits) aren't inputs.
    now[0] = 100.2
    budget.observe_quota_wait(a, True)
    now[0] = 105.0
    assert budget.remaining_seconds() == pytest.approx(0.8)
    budget.observe_quota_wait(b, True)
    budget.observe_quota_wait(a, True)  # repeated reports are idempotent
    now[0] = 106.0
    budget.observe_quota_wait(a, False)
    assert budget.remaining_seconds() == pytest.approx(0.8)
    now[0] = 107.0
    budget.observe_quota_wait(b, False)
    assert _state(budget)["deadline"] == pytest.approx(107.8)
    now[0] = 107.5
    assert budget.remaining_seconds() == pytest.approx(0.3)
    token = budget.reserve("after-quota", 1)
    now[0] = 107.9
    assert budget.remaining_seconds() == 0.0
    with pytest.raises(rec.RootBudgetError, match="deadline exhausted"):
        budget.reserve("late-after-resume", 1)
    budget.release(token)
    budget.release(a)
    budget.release(b)


def test_release_ends_root_quota_pause_on_cancellation(root_environment, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(rec.time, "monotonic", lambda: now[0])
    monkeypatch.setenv("PI_SUBAGENTS_ROOT_TIMEOUT", "1")
    budget = _root()
    token = budget.reserve("cancelled-wait", 1)
    now[0] = 100.1
    budget.observe_quota_wait(token, True)
    now[0] = 105.1
    budget.release(token)
    assert _state(budget)["deadline"] == pytest.approx(106.0)
    assert _state(budget)["quota_wait_sources"] == {}
    assert _state(budget)["quota_wait_started_at"] is None
    budget.observe_quota_wait(token, False)
    now[0] = 106.1
    assert budget.remaining_seconds() == 0.0
    with pytest.raises(rec.RootBudgetError, match="live root reservation"):
        budget.observe_quota_wait(token, True)


@pytest.mark.parametrize("window", ["", "1", "@-1", "@1.2", "@1;kill", "%1", "@١"])
def test_attach_window_validation(root_environment, window):
    budget = _root()
    token = budget.reserve("one", 1)
    with pytest.raises(rec.RootBudgetError, match="@digits"):
        budget.attach(token, window)
    assert _state(budget)["records"][token]["window_id"] is None


def test_attachment_is_immutable_and_window_unique(root_environment):
    budget = _root()
    one, two = budget.reserve("one", 1), budget.reserve("two", 1)
    budget.attach(one, "@10")
    budget.attach(one, "@10")
    with pytest.raises(rec.RootBudgetError, match="different window"):
        budget.attach(one, "@11")
    with pytest.raises(rec.RootBudgetError, match="another reservation"):
        budget.attach(two, "@10")
    with pytest.raises(rec.RootBudgetError, match="unknown"):
        budget.release("f" * 32)


def test_reservation_cannot_skip_or_bypass_depth(root_environment, monkeypatch):
    monkeypatch.setenv("PI_SUBAGENTS_MAX_DEPTH", "2")
    budget = _root()
    for invalid in (-1, 0, 2, 1.0, True):
        with pytest.raises(rec.RootBudgetError, match="child depth"):
            budget.reserve("bad", invalid)
    parent = budget.reserve("parent", 1)
    child = _child(monkeypatch, budget.child_env(parent))
    nested = child.reserve("nested", 2)
    deepest = _child(monkeypatch, child.child_env(nested))
    with pytest.raises(rec.RootBudgetError, match="max 2"):
        deepest.reserve("beyond-boundary", 3)

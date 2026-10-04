"""Focused lifecycle regressions using real handles and fake process boundaries.

No pi processes or tmux windows are created. The socket protocol and JSONL
transcripts are real; only spawn/window liveness and polling speed are replaced.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time
from types import SimpleNamespace

import pytest

from fake_server import FakePiSockServer
from pi_subagents import AgentPool, AgentPoolFailureError, SessionReuseError, Task
from pi_subagents import handle as live_mod
from pi_subagents import pool as pool_mod
from pi_subagents.client import PiSockSessionEnded, PiSockTurnFailed
from pi_subagents.schema import SchemaValidationError


def eventually(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    assert predicate(), "fake runtime did not reach the expected state"


def pop(pool):
    return asyncio.run(pool.pop(timeout=3))


class FakeRuntime:
    def __init__(self, tmp_path):
        self.root = tmp_path
        self.spawns = []
        self.killed = []
        self.windows = {}
        self.sessions = {}
        self.gates = {}
        self.commands = []
        self.lock = threading.Lock()

    def spawn(self, prompt, **kwargs):
        with self.lock:
            # Never manufacture an ID that could address a real user's window.
            window_id = f"@fake-runtime-{len(self.spawns) + 1}"
            self.spawns.append((prompt, dict(kwargs)))
            socket_name = kwargs["socket_name"]
            # Reopen may allocate a new socket name, but must use the old JSONL.
            resumed_path = kwargs.get("session_file") or kwargs.get("session_path")
            session = next((s for s in self.sessions.values()
                            if str(s.path) == str(resumed_path)), None)
            if session is None:
                session = self.sessions.get(socket_name)
            if session is None:
                session = SimpleNamespace(
                    path=self.root / f"session-{len(self.sessions)}.jsonl",
                    id=f"session-{len(self.sessions)}", text=None,
                    timestamp=0, stop=None, gate=None,
                )
                session.path.write_text(json.dumps({"type": "session", "id": session.id}) + "\n")
                self.sessions[socket_name] = session
            server = FakePiSockServer(str(self.root), f"window-{len(self.spawns)}")
            self.windows[window_id] = SimpleNamespace(alive=True, server=server, session=session)

        def state():
            return {"isIdle": session.gate is None or session.gate.is_set(),
                    "hasPendingMessages": False, "sessionFile": str(session.path),
                    "sessionId": session.id}

        server.behaviors["state"] = state
        server.behaviors["message"] = lambda: (
            {"content": session.text, "timestamp": session.timestamp,
             **({"stopReason": session.stop} if session.stop else {})}
            if session.text is not None else None
        )
        server.behaviors["activity"] = {
            "available": True, "activity": {"phase": "done", "label": "cached inspection",
            "run": {"toolCalls": 2, "thinkingMs": 17}},
        }
        original_reply = server._reply

        def reply(command):
            self.commands.append((window_id, dict(command)))
            if command.get("type") == "send":
                text = command["text"]
                if text == "startup-error":
                    return {"type": "response", "command": "send", "success": False,
                            "error": "startup delivery failed"}
                # Classify delivery before updating the fake turn, like Pi's
                # idle/direct versus active/steer boundary. The shared server's
                # generic default receipt cannot distinguish these states.
                was_idle = session.gate is None or session.gate.is_set()
                if was_idle or command.get("mode") == "follow_up":
                    self.turn(session, text)
                response = original_reply(command)
                if not callable(server.behaviors.get("send")):
                    response["data"]["mode"] = "direct" if was_idle else (command.get("mode") or "steer")
                return response
            elif command.get("type") == "abort" and session.gate is not None:
                session.gate.set()
                self.turn(session, "aborted")
            return original_reply(command)

        server._reply = reply
        return SimpleNamespace(window_id=window_id, socket_path=server.sock_path,
                               name=kwargs["name"])

    def turn(self, session, prompt):
        text = "not JSON" if prompt == "schema-invalid" else f"reply:{prompt}"
        stop = {"provider-error": "error", "aborted": "aborted",
                "unknown-outcome": None}.get(prompt, "stop")
        self.record_reply(session, text, stop=stop, gate=self.gates.get(prompt))

    def record_reply(self, session, text, *, stop="stop", gate=None):
        """Keep wire replies and durable outcomes consistent, including repairs."""
        session.timestamp += 1
        session.text = text
        session.stop = stop
        session.gate = gate
        entry = {"type": "message", "message": {
            "role": "assistant", "timestamp": session.timestamp,
            "content": [{"type": "thinking", "thinking": "cached thought"},
                        {"type": "text", "text": session.text}],
            **({"stopReason": session.stop} if session.stop else {}),
            **({"errorMessage": "quota exhausted"} if session.stop == "error" else {}),
        }}
        with session.path.open("a") as stream:
            stream.write(json.dumps(entry) + "\n")

    def alive(self, window_id):
        window = self.windows.get(window_id)
        return bool(window and window.alive)

    def kill(self, window_id):
        self.killed.append(window_id)
        self.windows[window_id].alive = False
        self.windows[window_id].server.close()

    def unload(self, window_id):
        self.kill(window_id)
        return True

    def close(self):
        for gate in self.gates.values():
            gate.set()
        for window in self.windows.values():
            window.server.close()


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    runtime = FakeRuntime(tmp_path)
    monkeypatch.setattr(pool_mod, "ensure_environment", lambda: None)
    monkeypatch.setattr(live_mod, "require_environment", lambda: None)
    monkeypatch.setattr(live_mod, "spawn_pi_window", runtime.spawn)
    monkeypatch.setattr(live_mod, "window_alive", runtime.alive)
    monkeypatch.setattr(live_mod, "kill_window", runtime.kill)
    monkeypatch.setattr(live_mod, "unload_window", runtime.unload)
    monkeypatch.setenv("PI_SUBAGENTS_SCHEMA_RETRIES", "0")
    original_wait = live_mod.AgentHandle.wait

    def fast_wait(self, timeout=None, poll=1):
        if self.prompt == "python-interrupt":
            self._await_ready()
            raise KeyboardInterrupt("interrupted worker")
        if self.prompt == "crashed":
            self._await_ready()
            raise PiSockSessionEnded("pi process crashed")
        return original_wait(self, timeout=timeout, poll=0.005)

    monkeypatch.setattr(live_mod.AgentHandle, "wait", fast_wait)
    yield runtime
    # Close pools before undoing the fake process boundary, including failures.
    from pi_subagents.registry import REGISTRY
    for pool in REGISTRY.pools():
        if not pool.closed:
            pool.close()
    runtime.close()


def test_validated_success_unloads_but_keeps_transcript_and_inspection(runtime):
    with AgentPool(concurrency=1) as pool:
        handle = pool.stage("work", slots=1).submit(Task("success", name="owner"))
        result = pop(pool)
        assert result.status == "settled"
        assert not runtime.alive(handle._live.window_id), "validated success must unload its process"
        assert len(runtime.killed) == 1
        assert handle._live.last_outcome == "ok"
        assert handle._live.dormant
        assert result.session.available
        path = result.session.session_file
        assert path and result.session.prose == "reply:success"
        assert result.session.thinking == "cached thought"
        assert handle.get_session().session_file == path
        before = len(runtime.commands)
        assert handle.state()["sessionFile"] == path
        assert handle.activity()["activity"]["label"] == "cached inspection"
        assert asyncio.run(handle.state_async())["sessionFile"] == path
        assert asyncio.run(handle.activity_async())["activity"]["label"] == "cached inspection"
        assert len(runtime.commands) == before, "dormant inspection must not reconnect"
        assert len(runtime.spawns) == 1
        assert pop(pool) is None


def test_schema_validated_success_unloads(runtime):
    with AgentPool(concurrency=1) as pool:
        stage = pool.stage("work", slots=1)
        # Use a valid JSON reply without bypassing real response validation.
        def json_turn(session, prompt):
            runtime.record_reply(session, '{"ok": true}')
        runtime.turn = json_turn
        handle = stage.submit(Task("success", schema={"type": "object", "required": ["ok"]}))
        assert pop(pool).body == {"ok": True}
        assert not runtime.alive(handle._live.window_id)


@pytest.mark.parametrize("prompt,expected", [
    ("provider-error", PiSockTurnFailed), ("aborted", PiSockSessionEnded),
    ("schema-invalid", SchemaValidationError), ("timeout", TimeoutError),
    ("crashed", PiSockSessionEnded), ("python-interrupt", KeyboardInterrupt),
])
def test_non_success_keeps_window_and_releases_capacity(runtime, prompt, expected):
    if prompt == "timeout":
        runtime.gates[prompt] = threading.Event()
    with AgentPool(concurrency=1) as pool:
        stage = pool.stage("work", slots=1)
        handle = stage.submit(Task(prompt, timeout=0.02,
            schema={"type": "object"} if prompt == "schema-invalid" else None))
        with pytest.raises(AgentPoolFailureError) as failure:
            pop(pool)
        assert isinstance(failure.value.result.error, expected)
        assert failure.value.result.handle is handle
        assert runtime.alive(handle._live.window_id)
        assert runtime.killed == [], "failed/interrupted sessions are inspection artifacts"
        assert pop(pool) is None, "a failure is delivered exactly once"
        assert pool.snapshot()["running"] == 0
        assert stage.snapshot()["failed"] == 1
        stage.submit(Task("healthy replacement"))
        assert pop(pool).status == "settled", "a retained failure must not consume a worker slot"
        assert runtime.alive(handle._live.window_id)


def test_unknown_outcome_is_not_automatically_unloaded(runtime):
    with AgentPool(concurrency=1) as pool:
        handle = pool.stage("work", slots=1).submit(Task("unknown-outcome"))
        assert pop(pool).status == "settled"
        assert runtime.alive(handle._live.window_id), "unload requires an explicitly successful outcome"
        assert handle._live.last_outcome != "ok", "absence of a terminal outcome is not explicit success"
        assert runtime.killed == []


def test_cancel_running_preserves_window_and_accounts_once(runtime):
    runtime.gates["blocked"] = threading.Event()
    with AgentPool(concurrency=1) as pool:
        stage = pool.stage("work", slots=1)
        handle = stage.submit(Task("blocked"))
        eventually(lambda: handle.status == "running" and any(
            c.get("text") == "blocked" for _, c in runtime.commands))
        assert handle.cancel()
        try:
            result = pop(pool)
        except AgentPoolFailureError as failure:
            result = failure.result
        assert result.status == "cancelled"
        assert runtime.alive(handle._live.window_id)
        assert runtime.killed == []
        assert not handle.cancel()
        assert pop(pool) is None
        assert stage.snapshot()["cancelled"] == 1
        assert pool.snapshot()["running"] == 0


@pytest.mark.parametrize("asynchronous", [False, True])
def test_active_send_steers_without_scheduling_a_new_task(runtime, asynchronous):
    runtime.gates["blocked"] = threading.Event()
    with AgentPool(concurrency=1) as pool:
        stage = pool.stage("work", slots=1)
        handle = stage.submit(Task("blocked"))
        eventually(lambda: handle.status == "running" and any(
            c.get("text") == "blocked" for _, c in runtime.commands))
        response = (asyncio.run(handle.send_async("steer")) if asynchronous else handle.send("steer"))
        assert response["delivered"] and response["mode"] == "steer"
        assert stage.snapshot()["submitted"] == 1
        assert len(runtime.spawns) == 1
        assert any(c.get("text") == "steer" and c.get("mode") == "steer"
                   for _, c in runtime.commands)
        runtime.gates["blocked"].set()
        assert pop(pool).handle is handle
        assert pop(pool) is None


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("initial", ["success", "provider-error", "cancelled"])
def test_terminal_send_schedules_followup(runtime, asynchronous, initial):
    if initial == "cancelled":
        runtime.gates[initial] = threading.Event()
    with AgentPool(concurrency=1) as pool:
        stage = pool.stage("work", slots=1)
        handle = stage.submit(Task(initial, metadata={"feature": "auth", "rounds": 2}))
        if initial == "cancelled":
            eventually(lambda: handle.status == "running" and any(
                c.get("text") == initial for _, c in runtime.commands))
            assert handle.cancel()
        try:
            first = pop(pool)
        except AgentPoolFailureError as failure:
            first = failure.result
        first_status = handle.status
        initial_window = handle._live.window_id
        receipt = (asyncio.run(handle.send_async("continue")) if asynchronous
                   else handle.send("continue"))
        assert receipt["scheduled"] is True and receipt["status"] == "queued"
        followup = receipt["handle"]
        assert receipt["handleId"] == followup.id
        assert followup is not handle and followup.stage is stage
        waited = asyncio.run(followup.wait_async(timeout=3))
        second = pop(pool)
        assert second is waited and second.handle is followup
        assert second.status == "settled" and second.body == "reply:continue"
        assert second.handle is not handle
        assert second.handle._live is handle._live
        assert second.parent is first
        assert second.task.metadata["feature"] == "auth"
        assert second.task.metadata["rounds"] == first.task.metadata["rounds"]
        assert first.handle.status == first_status and first.handle.result is first
        assert second.sequence == first.sequence + 1
        assert stage.snapshot()["submitted"] == 2
        assert len(runtime.spawns) == (2 if initial == "success" else 1)
        if initial != "success":
            assert second.handle._live.window_id == initial_window, "live terminal sessions must not respawn"
        assert pop(pool) is None


def test_dormant_reuse_waits_for_capacity_and_reserves_all_aliases(runtime):
    runtime.gates["blocker"] = threading.Event()
    runtime.gates["continue"] = threading.Event()
    with AgentPool(concurrency=1) as pool:
        stage = pool.stage("work", slots=1)
        source = stage.submit(Task("success", metadata={"rounds": 3}))
        first = pop(pool)
        assert not runtime.alive(source._live.window_id)
        blocker = stage.submit(Task("blocker"))
        eventually(lambda: blocker.status == "running" and len(runtime.spawns) == 2)
        followup = stage.submit(Task("continue"), parent=first, session_handle=source)
        assert followup.status == "queued"
        assert len(runtime.spawns) == 2, "reopen must wait for a scheduler slot"
        with pytest.raises(SessionReuseError):
            stage.submit(Task("duplicate"), session_handle=source)
        with pytest.raises(SessionReuseError):
            source.send("duplicate via send")
        runtime.gates["blocker"].set()
        assert pop(pool).handle is blocker
        eventually(lambda: followup.status == "running" and len(runtime.spawns) == 3)
        with pytest.raises(SessionReuseError):
            source.send("duplicate while running")
        runtime.gates["continue"].set()
        second = pop(pool)
        assert second.handle is followup and second.handle._live is source._live
        assert second.task.metadata["rounds"] == 3
        assert first.body == "reply:success" and first.status == "settled"
        assert first.session.session_file == second.session.session_file
        assert first.session.prose == "reply:success", "historical inspection caches must not be invalidated"
        assert second.session.prose.endswith("reply:continue")
        assert runtime.spawns[-1][1]["session_file"] == first.session.session_file
        assert source._live.session_id == runtime.windows[source._live.window_id].session.id
        assert pop(pool) is None


def test_queued_roster_is_not_active_concurrency(runtime):
    runtime.gates["blocked"] = threading.Event()
    with AgentPool(concurrency=2) as pool:
        stage = pool.stage("work", slots=2)
        handles = stage.submit_all(Task("blocked", name=f"job-{i}") for i in range(6))
        eventually(lambda: len(runtime.spawns) == 2 and all(h.status == "running" for h in handles[:2]))
        snapshot = pool.snapshot()
        assert snapshot["running"] == 2 and snapshot["queued"] == 4
        assert stage.snapshot()["submitted"] == 6
        assert handles[-1].cancel()
        assert pop(pool).handle is handles[-1]
        assert len(runtime.spawns) == 2
        runtime.gates["blocked"].set()
        results = [pop(pool) for _ in range(5)]
        assert len({result.sequence for result in results}) == 5
        assert all(result.status == "settled" for result in results)
        assert len(runtime.spawns) == 5
        assert pool.snapshot()["running"] == pool.snapshot()["queued"] == 0
        assert pop(pool) is None
        summary = pool.close()
        assert (summary.submitted, summary.settled, summary.cancelled) == (6, 5, 1)


def test_startup_failure_keeps_its_created_window(runtime):
    with AgentPool(concurrency=1) as pool:
        handle = pool.stage("work", slots=1).submit(Task("startup-error"))
        with pytest.raises(AgentPoolFailureError, match="startup delivery failed"):
            pop(pool)
        assert handle.status == "failed"
        assert runtime.alive(handle._live.window_id)
        assert runtime.killed == []
        assert pool.snapshot()["running"] == 0
        assert pop(pool) is None


def test_cancelled_queued_continuation_releases_its_session_reservation(runtime):
    runtime.gates["blocker"] = threading.Event()
    with AgentPool(concurrency=1) as pool:
        stage = pool.stage("work", slots=1)
        source = stage.submit(Task("success"))
        first = pop(pool)
        blocker = stage.submit(Task("blocker"))
        eventually(lambda: blocker.status == "running" and len(runtime.spawns) == 2)
        cancelled = stage.submit(Task("discard"), session_handle=source)
        assert cancelled.cancel()
        assert pop(pool).handle is cancelled
        assert len(runtime.spawns) == 2
        source.send("continue")
        assert len(runtime.spawns) == 2, "cancel/requeue must not reopen before capacity is available"
        runtime.gates["blocker"].set()
        assert pop(pool).handle is blocker
        result = pop(pool)
        assert result.body == "reply:continue" and result.parent is first
        assert result.handle._live is source._live
        assert len(runtime.spawns) == 3
        assert stage.snapshot()["cancelled"] == 1
        assert stage.snapshot()["submitted"] == 4
        assert pop(pool) is None


def test_dormant_reopen_waits_for_process_wide_capacity(runtime, monkeypatch):
    monkeypatch.setenv("PI_SUBAGENTS_MAX_CONCURRENT", "1")
    runtime.gates["blocker"] = threading.Event()
    with AgentPool(concurrency=1) as owner, AgentPool(concurrency=1) as other:
        stage = owner.stage("work", slots=1)
        source = stage.submit(Task("success"))
        first = pop(owner)
        blocker = other.stage("busy", slots=1).submit(Task("blocker"))
        eventually(lambda: blocker.status == "running" and len(runtime.spawns) == 2)
        source.send("continue")
        # Acquire the global condition to inspect capacity atomically; the
        # scheduler may already classify the continuation as 'starting'.
        with pool_mod._GLOBAL_CAPACITY._condition:
            assert pool_mod._GLOBAL_CAPACITY._active == 1
            assert len(runtime.spawns) == 2
        runtime.gates["blocker"].set()
        assert pop(other).handle is blocker
        result = pop(owner)
        assert result.parent is first and result.body == "reply:continue"
        assert len(runtime.spawns) == 3
        assert pop(owner) is None and pop(other) is None

"""Deterministic get_state/send race with mocked process and socket boundaries."""

import asyncio
import json
import threading

import pytest

from pi_subagents import AgentPool, Task
from pi_subagents import handle as live_mod
from pi_subagents import pool as pool_mod
from test_terminal_correlation import append_assistant


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("instant", [False, True])
def test_direct_receipt_revalidates_before_publication_and_capacity_release(
        tmp_path, monkeypatch, asynchronous, structured, instant):
    capacity = pool_mod._GlobalCapacity()
    monkeypatch.setattr(pool_mod, "_GLOBAL_CAPACITY", capacity)
    monkeypatch.setattr(pool_mod, "max_concurrent", lambda: 1)
    monkeypatch.setattr(pool_mod, "ensure_environment", lambda: None)
    monkeypatch.setenv("PI_SUBAGENTS_SCHEMA_RETRIES", "0")
    selected, release_old, rewaiting = threading.Event(), threading.Event(), threading.Event()
    spawned, hibernated = [], []
    old_text = '{"ok": true}' if structured else "old response"
    new_text = '{"ok": false}' if structured else "new response"
    schema = {"type": "object", "required": ["ok"], "properties": {"ok": {"type": "boolean"}}} if structured else None
    state = {"isIdle": False, "hasPendingMessages": False}
    wire = [{"content": old_text, "timestamp": 2, "stopReason": "stop"}]
    path = tmp_path / "direct.jsonl"
    append_assistant(path, 1, [{"type": "text", "text": "commentary"}], "toolUse")
    state["sessionFile"] = str(path)

    def spawn(pool, job):
        live = live_mod.AgentHandle(job.task.prompt, name=job.handle.name, cwd=str(tmp_path),
                                   window_name=job.handle.name, model=None, thinking=None, schema=job.task.schema)
        live.window_id = f"@fake-runtime-direct-{len(spawned)}"
        live.status = "running"
        live._sync.sock_path = ""
        spawned.append(live)
        monkeypatch.setattr(live, "kill", lambda: None)
        monkeypatch.setattr(live, "abort", lambda: None)
        monkeypatch.setattr(live, "hibernate", lambda: hibernated.append(live))
        if job.task.prompt == "peer":
            peer_path = tmp_path / "peer.jsonl"
            append_assistant(peer_path, 1, [{"type": "text", "text": "peer"}])
            peer_state = {"isIdle": True, "hasPendingMessages": False, "sessionFile": str(peer_path)}
            monkeypatch.setattr(live._sync, "state", lambda: dict(peer_state))
            monkeypatch.setattr(live._sync, "message", lambda: {"content": "peer", "timestamp": 1})
            live._remember_state(peer_state)
        else:
            monkeypatch.setattr(live._sync, "state", lambda: dict(state))
            monkeypatch.setattr(live._sync, "message", lambda: wire[0])
            live._remember_state(state)
            original_wait = live._sync.wait_settled
            first = True

            def wait(*args, **kwargs):
                nonlocal first
                if first:
                    first = False
                    # Freeze the original selected reply before validation. RPC
                    # acceptance happens while this poll is in flight.
                    old = {**state, "isIdle": True, "lastAssistant": dict(wire[0])}
                    selected.set()
                    assert release_old.wait(3), "original waiter was not released"
                    return old
                rewaiting.set()
                kwargs["poll"] = 0.001
                return original_wait(*args, **kwargs)

            def send(text, mode="steer"):
                assert mode == "steer" and not state["isIdle"]
                # The original turn ends AFTER the inspected busy state and
                # pre-RPC file offset, but BEFORE send is accepted as direct.
                append_assistant(path, 2, [{"type": "text", "text": old_text}])
                with path.open("a") as stream:
                    stream.write(json.dumps({"type": "custom_message", "customType": "session-message", "content": text}) + "\n")
                state["isIdle"] = True
                if instant:
                    append_assistant(path, 3, [{"type": "text", "text": new_text}])
                    wire[0] = {"content": new_text, "timestamp": 3, "stopReason": "stop"}
                return {"delivered": True, "mode": "direct"}

            monkeypatch.setattr(live._sync, "wait_settled", wait)
            monkeypatch.setattr(live._sync, "send", send)
        return live, {"cwd": str(tmp_path), "model": None, "thinking": None, "agentDir": None}

    monkeypatch.setattr(pool_mod.AgentPool, "_spawn_new", spawn)
    pool = AgentPool(concurrency=1)
    try:
        stage = pool.stage("work", slots=1)
        source = stage.submit(Task("first", schema=schema, timeout=3))
        assert selected.wait(3), "original turn did not start"
        peer = stage.submit(Task("peer", timeout=3))
        receipt = asyncio.run(source.send_async("next")) if asynchronous else source.send("next")
        assert receipt == {"delivered": True, "mode": "direct"}
        release_old.set()
        assert rewaiting.wait(3), "accepted direct turn was not correlated"
        if not instant:
            assert not source._future.done(), "old response was published"
            assert not peer._future.done() and len(spawned) == 1
            assert not hibernated, "old success was hibernated"
            with capacity._condition:
                assert capacity._active == 1, "new turn lost its capacity admission"
            append_assistant(path, 3, [{"type": "text", "text": new_text}])
            wire[0] = {"content": new_text, "timestamp": 3, "stopReason": "stop"}
        result = source.wait(timeout=3)
        assert result.body == ({"ok": False} if structured else new_text)
        assert source._live.settled_data["lastAssistant"]["timestamp"] == 3
        assert source._live.last_outcome == "ok"
        assert peer.wait(timeout=3).body == "peer"
        assert stage.snapshot()["submitted"] == 2, "a raced direct delivery stays admitted"
    finally:
        release_old.set()
        pool.close()


def test_pre_readiness_cancel_aborts_after_delivery_before_capacity_release(tmp_path, monkeypatch):
    capacity = pool_mod._GlobalCapacity()
    monkeypatch.setattr(pool_mod, "_GLOBAL_CAPACITY", capacity)
    monkeypatch.setattr(pool_mod, "max_concurrent", lambda: 1)
    monkeypatch.setattr(pool_mod, "ensure_environment", lambda: None)
    entered = threading.Event()
    aborted, destroyed = [], []
    path = tmp_path / "starting.jsonl"
    append_assistant(path, 1, [{"type": "text", "text": "partial"}], "toolUse")
    state = {"isIdle": False, "hasPendingMessages": False, "sessionFile": str(path)}
    live = live_mod.AgentHandle("first", name="starting", cwd=str(tmp_path), window_name="starting",
                               model=None, thinking=None, schema=None)
    live.window_id = "@fake-runtime-pre-ready"
    live._startup_started = True
    live._sync.sock_path = ""
    live._remember_state(state)
    ready = live._await_ready

    def await_ready(*args, **kwargs):
        entered.set()
        return ready(*args, **kwargs)

    def abort():
        aborted.append("abort")
        append_assistant(path, 2, [], "aborted")
        state["isIdle"] = True
        return {"aborted": True}

    monkeypatch.setattr(live, "_await_ready", await_ready)
    monkeypatch.setattr(live._sync, "state", lambda: dict(state))
    monkeypatch.setattr(live._sync, "message", lambda: None)
    monkeypatch.setattr(live._sync, "abort", abort)
    monkeypatch.setattr(live, "kill", lambda: destroyed.append("kill"))
    monkeypatch.setattr(live, "hibernate", lambda: destroyed.append("hibernate"))
    monkeypatch.setattr(pool_mod.AgentPool, "_spawn_new", lambda pool, job: (live, {}))
    pool = AgentPool(concurrency=1)
    try:
        handle = pool.stage("work", slots=1).submit(Task("first", timeout=3))
        assert entered.wait(3), "worker did not await initial delivery"
        assert handle.cancel()
        assert not aborted and not handle._future.done()
        with capacity._condition:
            assert capacity._active == 1
        # The initial prompt can finish delivery after the cancellation request.
        # Its readiness callback must abort before the slot can be reused.
        live.status = "running"
        live._ready.set_result(None)
        result = handle._future.result(timeout=3)
        assert result.status == "cancelled" and aborted
        assert not destroyed and live.closed and live.last_outcome == "cancelled"
        assert result.session.session_file == str(path)
    finally:
        if not live._ready.done():
            live._ready.set_result(None)
        pool.close()

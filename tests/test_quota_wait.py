"""Provisioned-quota waits do not consume execution timeouts."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import pi_subagents.handle as handle_mod  # noqa: E402
from fake_server import FakePiSockServer  # noqa: E402
from pi_subagents import AgentPool, Task  # noqa: E402
from pi_subagents.client import AsyncSockClient, PiSockSessionEnded, SockClient, _QuotaWaitExcluder  # noqa: E402
from pi_subagents.handle import AgentHandle as LiveHandle  # noqa: E402


def _blank_assistant(session_path: str, *, stop_reason: str = "aborted", error: str = "stopped") -> None:
	with open(session_path, "a", encoding="utf-8") as stream:
		stream.write(json.dumps({"type": "session", "id": "quota", "version": 3}) + "\n")
		stream.write(json.dumps({
			"type": "message",
			"id": "blank-terminal",
			"message": {
				"role": "assistant",
				"content": [],
				"stopReason": stop_reason,
				"errorMessage": error,
			},
		}) + "\n")


def _quota_state(server: FakePiSockServer, active: threading.Event, started: float) -> dict:
	return {
		"isIdle": True,
		"hasPendingMessages": False,
		"sessionFile": server.session_path,
		"sessionId": server.name,
		"quotaWait": {
			"active": active.is_set(),
			"elapsedMs": int(max(0.0, time.monotonic() - started) * 1000) if active.is_set() else int(max(0.0, time.monotonic() - started) * 1000),
		},
	}


def test_sync_wait_ignores_idle_terminal_message_until_quota_wait_resumes(tmp_path):
	server = FakePiSockServer(str(tmp_path), "sync-quota")
	_blank_assistant(server.session_path, stop_reason="aborted", error="weekly limit")
	active = threading.Event()
	active.set()
	started = time.monotonic()
	server.behaviors["state"] = lambda: _quota_state(server, active, started)
	server.behaviors["message"] = {"content": "", "stopReason": "aborted", "errorMessage": "weekly limit"}
	client = SockClient(server.sock_path)
	outcome: list[object] = []

	def waiter():
		try:
			outcome.append(client.wait_settled(0.05, poll=0.01))
		except BaseException as error:  # noqa: BLE001 - assert exact type below
			outcome.append(error)

	thread = threading.Thread(target=waiter)
	thread.start()
	time.sleep(0.15)
	assert outcome == [], "active quotaWait must suppress settle/abort past the tiny timeout"
	active.clear()
	thread.join(2)
	assert len(outcome) == 1
	assert isinstance(outcome[0], PiSockSessionEnded)
	assert "weekly limit" in str(outcome[0])


def test_async_wait_excludes_quota_pause_and_then_settles(tmp_path):
	async def scenario():
		server = FakePiSockServer(str(tmp_path), "async-quota")
		server.complete_turn("done")
		active = threading.Event()
		active.set()
		started = time.monotonic()
		server.behaviors["state"] = lambda: _quota_state(server, active, started)
		client = AsyncSockClient(server.sock_path)
		task = asyncio.create_task(client.wait_settled(0.05, poll=0.01))
		await asyncio.sleep(0.15)
		assert not task.done(), "async waiter must also remain pending during quotaWait.active"
		active.clear()
		settle = await asyncio.wait_for(task, 2)
		assert settle is not None
		assert settle["lastAssistant"]["content"] == "done"

	asyncio.run(scenario())


def test_agent_pool_task_timeout_pauses_for_quota_wait_and_succeeds(tmp_path, monkeypatch):
	active = threading.Event()
	active.set()
	started = time.monotonic()
	servers: list[FakePiSockServer] = []

	def spawn(prompt, *, name=None, cwd=None, window_name=None, model=None, thinking=None,
	          schema=None, agentDir=None, group=None, session_name=None, register=False):
		server = FakePiSockServer(str(tmp_path), name or "pool-quota")
		server.complete_turn("pool-ok")
		server.behaviors["state"] = lambda server=server: _quota_state(server, active, started)
		live = LiveHandle(prompt, name=name or server.name, cwd=str(tmp_path), window_name=name or server.name,
		                  model=model, thinking=thinking, schema=schema)
		ref = type("Ref", (), {"window_id": "@fake-quota", "socket_path": server.sock_path, "name": live.name})()
		live._bind(ref)
		live._startup_started = True
		live._ready.set_result(None)
		live.status = "running"
		live.group = group
		live.kill = lambda window_id=None: None  # type: ignore[method-assign]
		servers.append(server)
		return live

	monkeypatch.setattr(handle_mod, "spawn_pi_window_handle", spawn)
	monkeypatch.setattr(handle_mod, "unload_window", lambda window_id: False)
	with AgentPool(concurrency=1) as pool:
		stage = pool.stage("quota", slots=1)
		handle = stage.submit(Task("work", name="quota-job", timeout=0.05))
		deadline = time.monotonic() + 2
		while time.monotonic() < deadline and handle.status != "running":
			time.sleep(0.01)
		time.sleep(0.15)
		assert not handle._future.done(), "pool Future must stay pending beyond task timeout during quota pause"
		active.clear()
		result = asyncio.run(pool.pop(timeout=2))
		assert result is not None and result.status == "settled"
		assert str(result.body) == "pool-ok"


def test_historical_waits_dont_credit_future_timeout_and_active_never_expires(monkeypatch):
    import pi_subagents.client as client_mod
    now = [100.0]
    monkeypatch.setattr(client_mod.time, "monotonic", lambda: now[0])
    clock = _QuotaWaitExcluder(1)
    clock.observe({"quotaWait": {"active": False, "elapsedMs": 900_000}})
    assert clock.deadline == 101.0
    now[0] = 100.2
    clock.observe({"quotaWait": {"active": True, "elapsedMs": 900_000}})
    now[0] = 200.2
    assert not clock.expired(), "a coarse/unchanged quota counter cannot expire an active pause"
    clock.observe({"quotaWait": {"active": False, "elapsedMs": 1_000_000}})
    assert clock.deadline == 201.0
    now[0] = 201.1
    assert clock.expired(), "normal execution deadline resumes after the pause"


def test_client_enforces_live_root_guard_without_freezing_initial_allowance(tmp_path):
    server = FakePiSockServer(str(tmp_path), "live-root-guard")
    server.behaviors["state"] = {"isIdle": False, "hasPendingMessages": False}
    remaining = [1.0]
    client = SockClient(server.sock_path)
    outcome = []
    thread = threading.Thread(target=lambda: outcome.append(client.wait_settled(
        2, poll=0.01, root_remaining=lambda: remaining[0])))
    thread.start()
    time.sleep(0.1)
    assert not outcome
    remaining[0] = 0
    thread.join(2)
    assert outcome == [None], "a live root deadline still terminates normal work"


def test_live_wait_clears_root_pause_when_socket_dies(tmp_path):
    server = FakePiSockServer(str(tmp_path), "dead-quota-socket")
    live = LiveHandle("work", name="dead-quota-socket", cwd=str(tmp_path),
                      window_name="unused", model=None, thinking=None, schema=None)
    ref = type("Ref", (), {"window_id": "@fake", "socket_path": server.sock_path, "name": live.name})()
    live._bind(ref)
    live._ready.set_result(None)
    observations = []
    class Budget:
        def remaining_seconds(self): return 1.0
        def observe_quota_wait(self, token, active): observations.append((token, active))
    live._budget = Budget()
    live._admission_token = "owned-token"
    def dead_wait(*args, **kwargs):
        kwargs["on_quota_observed"](100, 0, 100, True)
        raise PiSockSessionEnded("socket died")
    live._sync.wait_settled = dead_wait
    with pytest.raises(PiSockSessionEnded, match="socket died"):
        live.wait(timeout=1)
    assert observations == [("owned-token", True), ("owned-token", False)]

"""Tests for the AgentPool / AgentStage / Task / AgentResult workflow API."""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import pi_subagents.handle as handle_mod  # noqa: E402
from fake_server import FakePiSockServer  # noqa: E402
import pi_subagents  # noqa: E402
from pi_subagents import (  # noqa: E402
	AgentPoolTimeoutError,
	AgentStrResponse,
	PoolClosedError,
	SessionReuseError,
	Task,

	AgentPoolFailureError,)
from pi_subagents.handle import AgentHandle as LiveHandle  # noqa: E402
from pi_subagents.pool import AgentPool, AgentStage  # noqa: E402
from pi_subagents.registry import REGISTRY  # noqa: E402
from pi_subagents.response import AgentDictResponse  # noqa: E402


class FakeLiveFactory:
	"""Builds real live AgentHandles bound to fake pi-sock servers.

	``gate`` (a threading.Event, when given) holds each run busy until set, so
	tests can observe concurrency and scheduling without real processes.
	"""

	def __init__(self, tmp_path, gate: threading.Event | None = None, reply: str = "hello"):
		self.tmp_path = str(tmp_path)
		self.gate = gate
		self.reply = reply
		self.servers: list[FakePiSockServer] = []
		self.spawned: list[LiveHandle] = []
		self.killed: list[str] = []
		self.renamed: list[tuple[str, str]] = []
		self.resumed: list[tuple[str, str]] = []
		self.counter = 0
		self.lock = threading.Lock()

	def make(self, prompt: str, **kwargs) -> LiveHandle:
		with self.lock:
			self.counter += 1
			name = f"fake{self.counter}"
		server = FakePiSockServer(self.tmp_path, name)
		# Initial prompt is already accepted/completed by this fake process.
		# Repeated reads are stable; only a new turn advances the outcome marker.
		reply_text = "{}" if kwargs.get("schema") is not None and self.reply == "hello" else self.reply
		server.complete_turn(reply_text)

		def send(command):
			busy = self.gate is not None and not self.gate.is_set()
			if command.get("mode") == "steer" and busy:
				return {"delivered": True, "mode": "steer"}
			server.complete_turn(reply_text)
			return {"delivered": True, "mode": "follow_up" if busy else "direct"}

		server.behaviors["send"] = send
		live = LiveHandle(prompt, name=kwargs.get("name") or name, cwd=kwargs.get("cwd") or self.tmp_path,
			window_name=kwargs.get("name") or name, model=kwargs.get("model"), thinking=None, schema=kwargs.get("schema"))
		ref = type("Ref", (), {"window_id": f"@fake-pool-{name}", "socket_path": server.sock_path, "name": live.name})()
		live._bind(ref)
		live._startup_started = True
		live._ready.set_result(None)
		live.status = "running"
		factory_renamed = self.renamed
		factory_resumed = self.resumed
		original_rename = live.set_session_name

		def track_rename(session_name):
			factory_renamed.append((live.name, session_name))
			return original_rename(session_name)
		live.set_session_name = track_rename  # type: ignore[method-assign]
		original_resume = live.resume

		def track_resume(prompt=None):
			factory_resumed.append((live.name, prompt or ""))
			return original_resume(prompt)
		live.resume = track_resume  # type: ignore[method-assign]

		original_reply = server._reply

		def reply_with_gate(command):
			response = original_reply(command)
			if command.get("type") == "get_state" and self.gate is not None:
				# stay busy until the gate opens
				response["data"] = dict(response["data"])
				if not self.gate.is_set():
					response["data"]["isIdle"] = False
			return response

		if self.gate is not None:
			server._reply = reply_with_gate

		def track_kill(window_id=None):
			self.killed.append(window_id or live.window_id)
			live.status = "dead"

		live.kill = track_kill  # type: ignore[method-assign]
		self.servers.append(server)
		self.spawned.append(live)
		return live


@pytest.fixture(autouse=True)
def _retain_fake_windows(monkeypatch):
	# These compatibility tests exercise retained-window reuse, not unloading.
	# The dedicated lifecycle runtime owns success-only hibernation coverage.
	monkeypatch.setattr(handle_mod, "unload_window", lambda window_id: False)


@pytest.fixture()
def instant_factory(tmp_path, monkeypatch):
	"""Fake spawn wired into the handle module for the duration of the test."""
	factory = FakeLiveFactory(tmp_path)

	def spawn(prompt, *, name=None, cwd=None, window_name=None, model=None, thinking=None,
	          schema=None, agentDir=None, group=None, session_name=None, register=False):
		live = factory.make(prompt, name=name, cwd=cwd, model=model, schema=schema)
		live.group = group
		factory.renamed.append((name or "", session_name or ""))
		return live

	monkeypatch.setattr(handle_mod, "spawn_pi_window_handle", spawn)
	return factory, spawn


def run(coro):
	return asyncio.run(coro)


def test_fake_protocol_distinguishes_active_steer_from_new_turn(tmp_path):
	from pi_subagents.client import _assistant_outcome_marker

	gate = threading.Event()
	factory = FakeLiveFactory(tmp_path, gate=gate)
	live = factory.make("first")
	server = factory.servers[0]
	try:
		baseline = live._sync.message()
		marker = _assistant_outcome_marker(server.session_path)
		assert live._sync.message() == baseline
		assert live.send("steer active turn")["mode"] == "steer"
		assert live._sync.message() == baseline
		assert _assistant_outcome_marker(server.session_path) == marker
		assert live.state()["isIdle"] is False

		assert live.send("next turn", mode="follow_up")["delivered"] is True
		assert live._sync.message() != baseline
		assert _assistant_outcome_marker(server.session_path) != marker
		gate.set()
		assert live.state()["isIdle"] is True
	finally:
		gate.set()
		server.close()


def test_submit_pop_and_result_roundtrip(tmp_path, instant_factory):
	with AgentPool(concurrency=2) as pool:
		build = pool.stage("build", slots=2)
		handle = build.submit(Task("do it", name="w1"))
		assert handle.status in ("queued", "starting", "running")
		result = run(pool.pop())
		assert result is not None
		assert (result.status == "settled")
		assert result.sequence == 1
		assert result.stage is build
		assert result.handle is handle
		assert isinstance(result.body, AgentStrResponse)
		assert result.body == "hello"
		assert handle.status == "settled"
		assert result.duration_ms is not None
		assert run(pool.pop()) is None  # quiescent


def test_pop_on_empty_pool_is_immediately_none():
	with AgentPool(concurrency=1) as pool:
		pool._workers = []  # no workers needed for this check
		assert run(pool.pop()) is None


def test_pop_timeout_raises_and_pool_survives(tmp_path):
	gate = threading.Event()
	factory = FakeLiveFactory(tmp_path, gate=gate)

	def spawn(prompt, **kwargs):
		return factory.make(prompt, name=kwargs.get("name"))

	monkey_spawn = spawn
	original = handle_mod.spawn_pi_window_handle
	handle_mod.spawn_pi_window_handle = monkey_spawn
	try:
		with AgentPool(concurrency=1) as pool:
			stage = pool.stage("work", slots=1)
			stage.submit(Task("slow", name="slowpoke", timeout=30))
			with pytest.raises(AgentPoolTimeoutError) as caught:
				run(pool.pop(timeout=0.3))
			assert caught.value.pool is pool
			assert any(h.name == "slowpoke" for h in caught.value.active_handles)
			assert pool.snapshot()["running"] == 1, "pool must survive the timeout"
			gate.set()
			result = run(pool.pop(timeout=10))
			assert result is not None and (result.status == "settled")
	finally:
		gate.set()
		handle_mod.spawn_pi_window_handle = original


def test_work_stealing_borrows_idle_slots(tmp_path):
	gate = threading.Event()
	factory = FakeLiveFactory(tmp_path, gate=gate)

	def spawn(prompt, **kwargs):
		return factory.make(prompt, name=kwargs.get("name"))

	original = handle_mod.spawn_pi_window_handle
	handle_mod.spawn_pi_window_handle = spawn
	try:
		with AgentPool(concurrency=2) as pool:
			solo = pool.stage("solo", slots=1)
			solo.submit_all([Task("a", name="j1"), Task("b", name="j2")])
			deadline = time.monotonic() + 5
			running = []
			while time.monotonic() < deadline:
				running = [h for h in pool.handles() if h.status in ("running", "starting")]
				if len(running) >= 2:
					break  # second job stole the idle slot
				time.sleep(0.02)
			assert len(running) >= 2, "idle slots from other stages must be borrowed"
			gate.set()
			results = []
			while True:
				result = run(pool.pop(timeout=10))
				if result is None:
					break
				results.append(result)
			assert {r.task.name for r in results} == {"j1", "j2"}
	finally:
		gate.set()
		handle_mod.spawn_pi_window_handle = original


def test_under_slot_stages_are_prioritized(tmp_path, instant_factory):
	with AgentPool(concurrency=2) as pool:
		a = pool.stage("a", slots=1)
		b = pool.stage("b", slots=1)
		# simulate: a has one running (at its slot cap) and one queued; b queued only.
		# Hold the scheduler lock throughout so live workers cannot pop the dummies.
		class FakeJob:
			def __init__(self, sequence):
				self.sequence = sequence
				handle = None
		with pool._condition:
			a._running = 1
			a._queue.append(FakeJob(1))
			b._queue.append(FakeJob(2))
			assert pool._select_stage_locked() is b
			b._queue.clear()
			b._running = 1
			assert pool._select_stage_locked() is a  # steal: no under-slot stage
			a._queue.clear()


def test_slot_sum_cannot_exceed_concurrency():
	with AgentPool(concurrency=2) as pool:
		pool.stage("a", slots=1)
		with pytest.raises(ValueError, match="slots"):
			pool.stage("b", slots=2)
		with pytest.raises(ValueError):
			pool.stage("a", slots=1)  # duplicate name


def test_pool_concurrency_capped_by_global_limit(monkeypatch):
	monkeypatch.setenv("PI_SUBAGENTS_MAX_CONCURRENT", "4")
	with pytest.raises(ValueError, match="exceeds"):
		AgentPool(concurrency=5)
	monkeypatch.setenv("PI_SUBAGENTS_MAX_CONCURRENT", "bogus")
	with AgentPool() as pool:
		assert pool.concurrency == 8


def test_close_invalidates_handles_and_kills_windows(tmp_path, instant_factory):
	factory, _ = instant_factory
	gate = threading.Event()
	factory.gate = gate
	pool = AgentPool(concurrency=1)
	try:
		stage = pool.stage("w", slots=1)
		handle = stage.submit(Task("p", name="victim"))
		deadline = time.monotonic() + 5
		while time.monotonic() < deadline and handle.status != "running":
			time.sleep(0.02)
		assert handle.status == "running", "the task must be dispatched before close"
		pool.close()
		assert handle.status == "closed"
		assert factory.killed, "close must destroy the retained tmux window"
		with pytest.raises(PoolClosedError):
			run(pool.pop())
		with pytest.raises(PoolClosedError):
			stage.submit(Task("again"))
		with pytest.raises(PoolClosedError):
			handle.wait(timeout=1)
		pool.close()  # idempotent
	finally:
		gate.set()


def test_cancel_queued_task_produces_cancelled_result(tmp_path):
	gate = threading.Event()
	factory = FakeLiveFactory(tmp_path, gate=gate)

	def spawn(prompt, **kwargs):
		return factory.make(prompt, name=kwargs.get("name"))

	original = handle_mod.spawn_pi_window_handle
	handle_mod.spawn_pi_window_handle = spawn
	try:
		with AgentPool(concurrency=1) as pool:
			stage = pool.stage("w", slots=1)
			first = stage.submit(Task("one", name="one"))
			second = stage.submit(Task("two", name="two"))
			deadline = time.monotonic() + 5
			while time.monotonic() < deadline and second.status != "queued":
				time.sleep(0.02)
			assert second.cancel() is True
			result = run(pool.pop(timeout=5))
			assert result is not None and result.status == "cancelled"
			assert result.handle is second
			assert not (result.status == "settled")
			assert result.body is None
			gate.set()
			other = run(pool.pop(timeout=5))
			assert other is not None and other.handle is first and (other.status == "settled")
			assert run(pool.pop(timeout=5)) is None
	finally:
		gate.set()
		handle_mod.spawn_pi_window_handle = original


def test_failed_task_raises_from_pop_and_handle(tmp_path, monkeypatch):
	from pi_subagents.errors import AgentPoolFailureError
	def spawn(prompt, **kwargs):
		live = FakeLiveFactory(tmp_path).make(prompt, name=kwargs.get("name"))
		def fail(timeout=None, poll=1.0):
			raise TimeoutError("pi died")
		live.wait = fail
		return live
	monkeypatch.setattr(handle_mod, "spawn_pi_window_handle", spawn)
	with AgentPool(concurrency=1) as pool:
		stage = pool.stage("w", slots=1)
		handle = stage.submit(Task("p", name="doomed"))
		with pytest.raises(AgentPoolFailureError, match="pi died") as failure:
			run(pool.pop(timeout=10))
		assert failure.value.result.status == "failed"
		assert isinstance(failure.value.__cause__, TimeoutError)
		with pytest.raises(TimeoutError, match="pi died"):
			handle.wait(timeout=1)
		with pytest.raises(TimeoutError, match="pi died"):
			run(handle.wait_async(timeout=1))
		assert run(pool.pop(timeout=1)) is None


def test_schema_task_returns_dict_response(tmp_path, monkeypatch):
	factory = FakeLiveFactory(tmp_path, reply='{"ok": true, "score": 3}')

	def spawn(prompt, **kwargs):
		return factory.make(prompt, name=kwargs.get("name"), schema=kwargs.get("schema"))

	monkeypatch.setattr(handle_mod, "spawn_pi_window_handle", spawn)
	with AgentPool(concurrency=1) as pool:
		stage = pool.stage("w", slots=1)
		stage.submit(Task("p", name="graded", schema={"type": "object", "required": ["ok"], "properties": {"ok": {"type": "boolean"}}}))
		result = run(pool.pop(timeout=10))
		assert isinstance(result.body, AgentDictResponse)
		assert result.body["ok"] is True


def test_session_reuse_runs_followup_on_same_window(tmp_path, instant_factory):
	factory, _ = instant_factory
	with AgentPool(concurrency=1) as pool:
		stage = pool.stage("chat", slots=1)
		first = stage.submit(Task("first prompt", name="session-owner"))
		first_result = run(pool.pop(timeout=10))
		assert first_result is not None and (first_result.status == "settled")
		source_live = first._live
		server = factory.servers[0]
		baseline = source_live._sync.message()
		from pi_subagents.client import _assistant_outcome_marker
		outcome_baseline = _assistant_outcome_marker(server.session_path)
		assert source_live._sync.message() == baseline, "reads must not invent turns"
		followup = stage.submit(
			Task("second prompt", name="followup"),
			parent=first_result,
			session_handle=first,
			session_name="renamed-owner",
		)
		assert followup is not first
		second_result = run(pool.pop(timeout=10))
		assert second_result is not None and (second_result.status == "settled")
		assert second_result.handle is followup
		assert followup._live is source_live, "the same pi session must be reused"
		assert len(factory.spawned) == 1
		assert source_live._sync.message() != baseline
		assert _assistant_outcome_marker(server.session_path) != outcome_baseline
		latest_send = next(command for command in reversed(server.sent) if command.get("type") == "send")
		assert latest_send["mode"] == "follow_up"
		assert any(r[1] == "renamed-owner" for r in factory.renamed)
		assert factory.resumed and factory.resumed[-1][1] == "second prompt"
		# original handle/result remain settled and untouched
		assert first.status == "settled" and (first_result.status == "settled")


def test_session_reuse_validates_config_and_reservation(tmp_path, instant_factory):
	factory, _ = instant_factory
	with AgentPool(concurrency=1) as pool:
		stage = pool.stage("chat", slots=1)
		handle = stage.submit(Task("first", name="owner", model="provider/a"))
		result = run(pool.pop(timeout=10))
		assert result is not None
		with pytest.raises(SessionReuseError, match="model"):
			stage.submit(Task("again", model="provider/b"), session_handle=handle)
		# omitted fields inherit the retained session's configuration
		ok = stage.submit(Task("again"), session_handle=handle)
		assert run(pool.pop(timeout=10)) is not None
		# while a follow-up is queued, the session cannot be double-booked
		queued_followup = stage.submit(Task("fourth"), session_handle=handle)
		with pytest.raises(SessionReuseError, match="queued or running"):
			stage.submit(Task("third"), session_handle=handle)
		assert run(pool.pop(timeout=10)) is not None
		_ = ok, queued_followup
		assert len(factory.spawned) == 1
		assert [command["text"] for command in factory.servers[0].sent
		        if command["type"] == "send"] == ["again", "fourth"]
		# a foreign pool must not be reservable; it raised mid-block, so this
		# one uses an explicit close (with-statements keep pools alive on
		# exceptions by design).
		other = AgentPool(concurrency=1)
		try:
			with pytest.raises(SessionReuseError, match="another pool"):
				other.stage("s", slots=1).submit(Task("x"), session_handle=handle)
		finally:
			other.close()


def test_parent_metadata_inheritance_and_rounds(tmp_path, instant_factory):
	with AgentPool(concurrency=1) as pool:
		build = pool.stage("build", slots=1)
		root = build.submit(Task("implement", name="b1", metadata={"feature": "auth"}))
		root_result = run(pool.pop(timeout=10))
		assert root_result is not None
		assert root_result.task.metadata["rounds"] == 0
		review = build.submit(
			Task("review", name="r1", schema={"type": "object"}),
			parent=root_result,
		)
		review_result = run(pool.pop(timeout=10))
		assert review_result is not None
		# inherited feature, automatic rounds carried through
		assert review_result.task.metadata["feature"] == "auth"
		assert review_result.task.metadata["rounds"] == 0
		assert review_result.parent is root_result
		fix = build.submit(
			Task("fix", name="f1", metadata={"rounds": 1}),
			parent=review_result,
		)
		fix_result = run(pool.pop(timeout=10))
		assert fix_result is not None
		assert fix_result.task.metadata["rounds"] == 1
		assert fix_result.task.metadata["feature"] == "auth"
		# historical metadata is immutable
		assert review_result.task.metadata["rounds"] == 0
		assert root_result.task.metadata["rounds"] == 0


def test_completion_order_not_submission_order(tmp_path, monkeypatch):
	class Staggered(FakeLiveFactory):
		def make(self, prompt, **kwargs):
			live = super().make(prompt, **kwargs)
			original_wait = live.wait
			delay = 0.25 if kwargs.get("name") == "slow" else 0.02

			def wait(timeout=None, poll=1.0):
				time.sleep(delay)
				return original_wait(timeout=timeout)
			live.wait = wait  # type: ignore[method-assign]
			return live

	factory = Staggered(tmp_path)

	def spawn(prompt, **kwargs):
		return factory.make(prompt, name=kwargs.get("name"))

	monkeypatch.setattr(handle_mod, "spawn_pi_window_handle", spawn)
	with AgentPool(concurrency=2) as pool:
		stage = pool.stage("w", slots=2)
		stage.submit(Task("p", name="slow"))
		stage.submit(Task("p", name="fast"))
		names = []
		while True:
			result = run(pool.pop(timeout=10))
			if result is None:
				break
			names.append(result.task.name)
		assert names.index("fast") < names.index("slow")


def test_stage_snapshot_tracks_only_busy_periods():
	with AgentPool(concurrency=1) as pool:
		stage = pool.stage("timed", slots=1)
		stage._started_running(1_000.0)
		active = stage.snapshot()
		assert active["busyMs"] == 0
		assert active["activeSince"] == 1_000.0

		stage._stopped_running(3_500.0)
		idle = stage.snapshot()
		assert idle["busyMs"] == 2_500
		assert idle["activeSince"] is None

		stage._started_running(8_000.0)
		stage._stopped_running(9_250.0)
		assert stage.snapshot()["busyMs"] == 3_750


def test_snapshot_reports_pools_and_stages(tmp_path, instant_factory):
	with AgentPool(concurrency=2, name="wf") as pool:
		build = pool.stage("build", slots=1)
		review = pool.stage("review", slots=1)
		build.submit(Task("p", name="b1"))
		snap = pool.snapshot()
		assert snap["name"] == "wf"
		assert snap["concurrency"] == 2
		assert {s["name"] for s in snap["stages"]} == {"build", "review"}
		full = REGISTRY.snapshot()
		assert any(p["name"] == "wf" for p in full["pools"])
		assert full["groups"].keys() >= {"build", "review"}


def test_handle_requires_dispatch_for_live_operations(tmp_path, instant_factory):
	with AgentPool(concurrency=1) as pool:
		stage = pool.stage("w", slots=1)
		handle = stage.submit(Task("p", name="later"))
		with pytest.raises(ValueError, match="dispatched"):
			handle.send("steer")
		with pytest.raises(ValueError, match="dispatched"):
			handle.activity()


def test_direct_agent_api_is_gone():
	assert not hasattr(pi_subagents, "agent")
	assert not hasattr(pi_subagents, "spawn")
	assert not hasattr(pi_subagents, "phase")
	assert not hasattr(pi_subagents, "wait_all")
	assert not hasattr(pi_subagents, "wait_all_async")
	assert not hasattr(pi_subagents, "set_phase")


def test_close_returns_pool_summary(tmp_path, instant_factory):
	with AgentPool(concurrency=1) as pool:
		stage = pool.stage("w", slots=1)
		stage.submit(Task("p", name="job-one"))
		result = run(pool.pop(timeout=10))
		assert result is not None and (result.status == "settled")
		summary = pool.close()
		assert summary.name.startswith("pool-")
		assert summary.submitted == 1
		assert summary.settled == 1
		assert summary.failed == 0
		assert summary.tool_calls >= 0
		assert summary.wall_ms >= 0
		assert "w" in summary.stages
		text = str(summary)
		assert "PoolSummary ✓" in text and "1 tasks" in text
		assert "1/1 settled" in text
		# close() also stores the summary for with-statement users
		assert pool.last_summary is summary
		# AgentResult echoes readably: status + timing + the actual body
		text = str(result)
		assert text.startswith("✓ job-one (w)")
		assert "hello" in text, "the echo shows the actual body, not a summary"


def test_finish_closes_every_pool(tmp_path, instant_factory):
	p1 = AgentPool(concurrency=1)
	p2 = AgentPool(concurrency=1)
	p1.stage("s", slots=1)
	p2.stage("s", slots=1)
	assert pi_subagents.finish() == 2
	assert p1.closed and p2.closed
	assert pi_subagents.finish() == 0


def test_with_statement_closes_pool_on_clean_exit(tmp_path, instant_factory):
	with AgentPool(concurrency=1, name="withy") as pool:
		stage = pool.stage("w", slots=1)
		stage.submit(Task("p", name="wither"))
		result = run(pool.pop(timeout=10))
		assert result is not None and (result.status == "settled")
	# clean exit: the pool closed itself and recorded the workflow report
	assert pool.closed
	assert pool.last_summary is not None
	assert pool.last_summary.name == "withy"
	assert pool.last_summary.settled == 1
	with pytest.raises(PoolClosedError):
		pool.stage("again", slots=1)


def test_with_statement_keeps_pool_alive_on_exception(tmp_path, instant_factory):
	holder: list[AgentPool] = []
	with pytest.raises(RuntimeError, match="boom"):
		with AgentPool(concurrency=2) as pool:
			holder.append(pool)
			stage = pool.stage("w", slots=1)
			stage.submit(Task("p", name="survivor"))
			result = run(pool.pop(timeout=10))
			assert result is not None and (result.status == "settled")
			raise RuntimeError("boom")

	# the exception must NOT have closed the pool: windows, results, and the
	# scheduler stay alive so a follow-up cell can continue the run
	pool = holder[0]
	assert not pool.closed
	assert pool.last_summary is None
	stage = pool.stage("continue", slots=1)
	stage.submit(Task("p2", name="continued"))
	continued = run(pool.pop(timeout=10))
	assert continued is not None and (continued.status == "settled")
	assert continued.task.name == "continued"
	# the orchestrator closes it explicitly once done
	summary = pool.close()
	assert summary.settled >= 1
	assert pool.last_summary is summary

def test_fail_fast_pool_raises_on_failed_job(tmp_path, monkeypatch):
	class Boom(FakeLiveFactory):
		def make(self, prompt, **kwargs):
			live = super().make(prompt, **kwargs)
			original_wait = live.wait

			def wait(timeout=None, poll=1.0):
				raise TimeoutError("pi process died before settling")
			live.wait = wait  # type: ignore[method-assign]
			return live

	factory = Boom(tmp_path)

	def spawn(prompt, **kwargs):
		return factory.make(prompt, name=kwargs.get("name"))

	monkeypatch.setattr(handle_mod, "spawn_pi_window_handle", spawn)
	with AgentPool(concurrency=1) as pool:
		stage = pool.stage("w", slots=1)
		stage.submit(Task("p", name="doomed"))
		with pytest.raises(AgentPoolFailureError) as excinfo:
			run(pool.pop(timeout=10))
		result = excinfo.value.result
		assert result is not None and not (result.status == "settled")
		assert result.status == "failed"
		assert "pi process died before settling" in str(excinfo.value)
		# one-shot: the next pop drains normally instead of re-raising
		assert run(pool.pop(timeout=5)) is None


def test_default_pool_raises_for_failed_tasks(tmp_path, monkeypatch):
	class Boom(FakeLiveFactory):
		def make(self, prompt, **kwargs):
			live = super().make(prompt, **kwargs)
			original_wait = live.wait

			def wait(timeout=None, poll=1.0):
				raise TimeoutError("pi died")
			live.wait = wait  # type: ignore[method-assign]
			return live

	factory = Boom(tmp_path)

	def spawn(prompt, **kwargs):
		return factory.make(prompt, name=kwargs.get("name"))

	monkeypatch.setattr(handle_mod, "spawn_pi_window_handle", spawn)
	with AgentPool(concurrency=1) as pool:
		stage = pool.stage("w", slots=1)
		stage.submit(Task("p", name="doomed"))
		from pi_subagents.errors import AgentPoolFailureError
		with pytest.raises(AgentPoolFailureError, match="pi died"):
			run(pool.pop(timeout=10))

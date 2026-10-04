"""Async client + async handle flows (the PTC path)."""

from __future__ import annotations

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from fake_server import FakePiSockServer  # noqa: E402
from pi_subagents.handle import AgentHandle  # noqa: E402
from pi_subagents.response import AgentStrResponse  # noqa: E402


def bind_handle(h: AgentHandle, tmp_path) -> FakePiSockServer:
	ref = type("Ref", (), {
		"window_id": "@fake-async-" + h.id,
		"socket_path": os.path.join(str(tmp_path), f"{h.id}.sock"),
		"name": h.name,
	})()
	server = FakePiSockServer(str(tmp_path), h.id)
	h._bind(ref)
	# The fake process has already accepted its initial prompt.
	h._startup_started = True
	h._ready.set_result(None)
	return server


def test_wait_async_and_send_async(tmp_path):
	async def run():
		h = AgentHandle("p", name="a1", cwd=str(tmp_path), window_name="a1", model=None, thinking=None, schema=None)
		server = bind_handle(h, tmp_path)
		server.behaviors["state"] = {"isIdle": True, "hasPendingMessages": False}
		server.behaviors["message"] = {"content": "async answer", "timestamp": 1}
		try:
			resp = await h
			assert isinstance(resp, AgentStrResponse)
			assert resp == "async answer"
			delivered = await h.send_async("follow-up question")
			assert delivered["delivered"] is True
		finally:
			server.close()

	asyncio.run(run())


def test_wait_async_state_polling_flow(tmp_path):
	async def run():
		h = AgentHandle("p", name="a2", cwd=str(tmp_path), window_name="a2", model=None, thinking=None, schema=None)
		server = bind_handle(h, tmp_path)
		states = iter([
			{"isIdle": False, "hasPendingMessages": False},
			{"isIdle": True, "hasPendingMessages": False},
		])

		def _state():
			try:
				return next(states)
			except StopIteration:
				return {"isIdle": True, "hasPendingMessages": False}

		server.behaviors["state"] = _state
		server.behaviors["message"] = {"content": "settled", "timestamp": 2}
		try:
			resp = await h.wait_async(timeout=5)
			assert resp == "settled"
		finally:
			server.close()

	asyncio.run(run())


def test_wait_async_after_abort_raises(tmp_path):
	async def run():
		h = AgentHandle("p", name="a3", cwd=str(tmp_path), window_name="a3", model=None, thinking=None, schema=None)
		bind_handle(h, tmp_path)
		await h.abort_async()
		with pytest.raises(ValueError, match="await on closed handle"):
			await h

	asyncio.run(run())


def test_resume_async_reopens(tmp_path):
	async def run():
		h = AgentHandle("p", name="a4", cwd=str(tmp_path), window_name="a4", model=None, thinking=None, schema=None)
		server = bind_handle(h, tmp_path)
		server.complete_turn("old answer", timestamp=3)
		def followup(command):
			assert command["mode"] == "follow_up"
			server.complete_turn("continued", timestamp=4)
			return {"delivered": True, "mode": "direct"}
		server.behaviors["send"] = followup
		try:
			await h.wait_async(timeout=5)
			from pi_subagents.client import _assistant_outcome_marker
			baseline = _assistant_outcome_marker(server.session_path)
			h2 = await h.resume_async("continue please")
			assert h2 is h and h.closed is False and h.status == "running"
			# Both the wire reply and persisted outcome belong to the new turn.
			assert _assistant_outcome_marker(server.session_path) != baseline
			resp = await h.wait_async(timeout=5)
			assert resp == "continued"
			assert server.sent[-1]["text"] == "continue please"
		finally:
			server.close()

	asyncio.run(run())


def test_wait_paths_emit_bridge_frames_each_tick(tmp_path, monkeypatch):
	"""Both wait paths must push a snapshot every poll tick.

	The PTC host re-arms its idle timeout on those frames, so a wait that stops
	emitting looks like a hung session and gets killed.
	"""
	import builtins

	frames = []
	monkeypatch.setattr(builtins, "PTC_STATE_EMIT", lambda snap: frames.append(snap), raising=False)

	async def run_async():
		h = AgentHandle("p", name="tick-a", cwd=str(tmp_path), window_name="tick-a", model=None, thinking=None, schema=None)
		server = bind_handle(h, tmp_path)
		states = iter([
			{"isIdle": False, "hasPendingMessages": False},
			{"isIdle": False, "hasPendingMessages": False},
			{"isIdle": False, "hasPendingMessages": False},
			{"isIdle": False, "hasPendingMessages": False},
			{"isIdle": False, "hasPendingMessages": False},
		])

		def _state():
			try:
				return next(states)
			except StopIteration:
				return {"isIdle": True, "hasPendingMessages": False}

		server.behaviors["state"] = _state
		server.behaviors["message"] = {"content": "done", "timestamp": 1}
		try:
			before = len(frames)
			resp = await h.wait_async(timeout=8, poll=0.05)
			assert str(resp) == "done"
			# One frame per busy tick (plus the settle frame). A wait that only
			# emits on settle would report 1 and starve the host's idle timer.
			assert len(frames) - before >= 4, f"expected per-tick bridge frames, got {len(frames) - before}"
		finally:
			server.close()

	asyncio.run(run_async())

	frames.clear()
	h = AgentHandle("p", name="tick-s", cwd=str(tmp_path), window_name="tick-s", model=None, thinking=None, schema=None)
	server = bind_handle(h, tmp_path)
	states = iter([
		{"isIdle": False, "hasPendingMessages": False},
		{"isIdle": False, "hasPendingMessages": False},
		{"isIdle": False, "hasPendingMessages": False},
		{"isIdle": False, "hasPendingMessages": False},
		{"isIdle": False, "hasPendingMessages": False},
	])

	def _state():
		try:
			return next(states)
		except StopIteration:
			return {"isIdle": True, "hasPendingMessages": False}

	server.behaviors["state"] = _state
	server.behaviors["message"] = {"content": "done", "timestamp": 2}
	try:
		before = len(frames)
		assert str(h.wait(timeout=8, poll=0.05)) == "done"
		assert len(frames) - before >= 4, f"expected per-tick bridge frames, got {len(frames) - before}"
	finally:
		server.close()

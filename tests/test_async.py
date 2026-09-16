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
		"window_id": "win1",
		"socket_path": os.path.join(str(tmp_path), f"{h.id}.sock"),
		"name": h.name,
	})()
	server = FakePiSockServer(str(tmp_path), h.id)
	h._bind(ref)
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
		server.behaviors["state"] = {"isIdle": True, "hasPendingMessages": False}
		server.behaviors["message"] = {"content": "continued", "timestamp": 3}
		try:
			await h
			h2 = await h.resume_async("continue please")
			assert h2 is h and h.closed is False and h.status == "running"
			resp = await h.wait_async(timeout=5)
			assert resp == "continued"
		finally:
			server.close()

	asyncio.run(run())

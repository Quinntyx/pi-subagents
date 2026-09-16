"""Tests for the pi-sock client against a fake server."""

from __future__ import annotations

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from fake_server import FakePiSockServer  # noqa: E402
from pi_subagents.client import PiSockError, PiSockUnavailable, SockClient  # noqa: E402


@pytest.fixture()
def sock_env(tmp_path):
	return str(tmp_path)


def test_send_state_message_roundtrip(sock_env):
	server = FakePiSockServer(sock_env, "subagent-a")
	server.behaviors["state"] = {"isIdle": False, "hasPendingMessages": False, "sessionFile": "/tmp/x.jsonl"}
	server.behaviors["message"] = {"content": "hello", "timestamp": 1}
	client = SockClient(server.sock_path)
	try:
		assert client.send("hello")["delivered"] is True
		state = client.state()
		assert state["isIdle"] is False
		assert state["sessionFile"] == "/tmp/x.jsonl"
		assert client.message()["content"] == "hello"
	finally:
		server.close()


def test_missing_socket_raises_unavailable(sock_env):
	client = SockClient(os.path.join(sock_env, "subagent-none.sock"))
	with pytest.raises(PiSockUnavailable):
		client.state()


def test_command_error_surfaces(sock_env):
	server = FakePiSockServer(sock_env, "subagent-b")
	try:
		with pytest.raises(PiSockError):
			SockClient(server.sock_path).request({"type": "bogus"})
	finally:
		server.close()


def test_wait_settled_flow(sock_env):
	server = FakePiSockServer(sock_env, "subagent-c")
	# busy first, then idle with a message
	states = iter([
		{"isIdle": False, "hasPendingMessages": False},
		{"isIdle": False, "hasPendingMessages": False},
		{"isIdle": True, "hasPendingMessages": False},
	])
	def _state_data():
		try:
			return next(states)
		except StopIteration:
			return {"isIdle": True, "hasPendingMessages": False}

	server.behaviors["state"] = _state_data
	server.behaviors["message"] = {"content": "done!", "timestamp": 2}
	client = SockClient(server.sock_path)
	try:
		result = client.wait_settled(timeout=5, poll=0.05)
		assert result["lastAssistant"]["content"] == "done!"
		assert result["isIdle"] is True
	finally:
		server.close()


def test_wait_settled_timeout_returns_none(sock_env):
	server = FakePiSockServer(sock_env, "subagent-d")
	server.behaviors["state"] = {"isIdle": False, "hasPendingMessages": False}
	try:
		start = time.monotonic()
		assert SockClient(server.sock_path).wait_settled(timeout=0.4, poll=0.05) is None
		assert time.monotonic() - start >= 0.35
	finally:
		server.close()


def test_wait_settled_dead_socket_returns_none(sock_env):
	# never-created socket: poll treats missing pi as dead
	result = SockClient(os.path.join(sock_env, "subagent-gone.sock")).wait_settled(timeout=2, poll=0.05)
	assert result is None

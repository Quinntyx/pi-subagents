"""Tests for the pi-sock client against a fake server."""

from __future__ import annotations

import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from fake_server import FakePiSockServer  # noqa: E402
from pi_subagents.client import (
	PiSockError,
	PiSockSessionEnded,
	PiSockTurnFailed,
	PiSockUnavailable,
	SockClient,
)  # noqa: E402


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


def test_wait_settled_dead_socket_raises_session_ended(sock_env):
	# never-created socket: the session never came up mid-wait → explicit error
	# (handle.wait awaits the socket before polling, so reaching this state means
	# the session ended)
	client = SockClient(os.path.join(sock_env, "subagent-gone.sock"))
	with pytest.raises(PiSockSessionEnded):
		client.wait_settled(timeout=2, poll=0.05)


def test_wait_settled_socket_lost_midwait_raises(sock_env):
	server = FakePiSockServer(sock_env, "subagent-dies")
	server.behaviors["state"] = {"isIdle": False, "hasPendingMessages": False}
	client = SockClient(server.sock_path)
	try:
		# the fake server stops answering mid-wait; the socket file lingers, so
		# connection-refused is the signal that the session ended
		threading.Timer(0.3, server.close).start()
		with pytest.raises(PiSockSessionEnded):
			client.wait_settled(timeout=5, poll=0.05)
	finally:
		server.close()


def test_wait_settled_interrupted_turn_raises(sock_env, tmp_path):
	# last assistant message in the session file has stopReason "aborted": the
	# user interrupted the agent; that must NOT settle as a clean result
	session_file = tmp_path / "session-abort.jsonl"
	session_file.write_text(
		'{"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"partial"}],"stopReason":"aborted"}}\n'
	)
	server = FakePiSockServer(sock_env, "subagent-abort")
	server.behaviors["state"] = {
		"isIdle": True, "hasPendingMessages": False, "sessionFile": str(session_file),
	}
	server.behaviors["message"] = {"content": "partial", "timestamp": 2}
	client = SockClient(server.sock_path)
	try:
		with pytest.raises(PiSockSessionEnded, match="interrupted"):
			client.wait_settled(timeout=5, poll=0.05)
	finally:
		server.close()


def test_wait_settled_provider_error_raises_even_without_text(sock_env, tmp_path):
	# quota-exhausted-class failure: the run errored before producing any text,
	# so get_message filters the entry out and the settle condition can never
	# fire — the outcome check must still raise, immediately
	session_file = tmp_path / "session-error.jsonl"
	session_file.write_text(
		'{"type":"message","message":{"role":"assistant","content":[],"stopReason":"error",'
		'"errorMessage":"quota exhausted: usage limit reached"}}\n'
	)
	server = FakePiSockServer(sock_env, "subagent-err")
	server.behaviors["state"] = {
		"isIdle": True, "hasPendingMessages": False, "sessionFile": str(session_file),
	}
	server.behaviors["message"] = None  # textless error entry is filtered out
	client = SockClient(server.sock_path)
	try:
		with pytest.raises(PiSockTurnFailed, match="quota exhausted"):
			client.wait_settled(timeout=5, poll=0.05)
	finally:
		server.close()


def test_wait_settled_length_stop_still_settles(sock_env, tmp_path):
	# "length" is a truncated-but-usable response: settles, does not raise
	session_file = tmp_path / "session-length.jsonl"
	session_file.write_text(
		'{"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"truncated…"}],"stopReason":"length"}}\n'
	)
	server = FakePiSockServer(sock_env, "subagent-len")
	server.behaviors["state"] = {
		"isIdle": True, "hasPendingMessages": False, "sessionFile": str(session_file),
	}
	server.behaviors["message"] = {"content": "truncated…", "timestamp": 3}
	client = SockClient(server.sock_path)
	try:
		result = client.wait_settled(timeout=5, poll=0.05)
		assert result is not None and result["isIdle"] is True
	finally:
		server.close()

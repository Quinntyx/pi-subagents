"""Tests for AgentHandle/AgentSession semantics against fake servers."""

from __future__ import annotations

import builtins
import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from fake_server import FakePiSockServer  # noqa: E402
from pi_subagents import handle as handle_mod  # noqa: E402
from pi_subagents.handle import AgentHandle  # noqa: E402
from pi_subagents.response import AgentDictResponse, AgentStrResponse  # noqa: E402
from pi_subagents.session_file import AgentSession  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_registry():
	from pi_subagents.registry import REGISTRY

	REGISTRY._handles.clear()
	yield
	REGISTRY._handles.clear()


def bind_handle(h: AgentHandle, tmp_path) -> FakePiSockServer:
	ref = type("Ref", (), {
		"window_id": h.name + "-win",
		"socket_path": os.path.join(str(tmp_path), f"{h.id}.sock"),
		"name": h.name,
	})()
	server = FakePiSockServer(str(tmp_path), h.id)
	h._bind(ref)
	return server


def test_wait_returns_str_response(tmp_path):
	h = AgentHandle("p", name="t1", cwd=str(tmp_path), window_name="t1", model=None, thinking=None, schema=None)
	server = bind_handle(h, tmp_path)
	server.behaviors["state"] = {"isIdle": True, "hasPendingMessages": False}
	server.behaviors["message"] = {"content": "all done", "timestamp": 3}
	try:
		resp = h.wait(timeout=5)
		assert isinstance(resp, AgentStrResponse)
		assert isinstance(resp, str)
		assert resp == "all done"
		assert resp.upper() == "ALL DONE"
		assert resp.get_session() is h.session
		assert h.status == "settled"
	finally:
		server.close()


def test_await_after_abort_raises(tmp_path):
	h = AgentHandle("p", name="t2", cwd=str(tmp_path), window_name="t2", model=None, thinking=None, schema=None)
	bind_handle(h, tmp_path)
	h.abort()
	assert h.closed
	with pytest.raises(ValueError, match="await on closed handle"):
		h.wait(timeout=1)
	with pytest.raises(ValueError, match="await on closed handle"):
		h.wait_async(timeout=1).__await__().__next__()


def test_resume_reopens_handle(tmp_path):
	h = AgentHandle("p", name="t3", cwd=str(tmp_path), window_name="t3", model=None, thinking=None, schema=None)
	server = bind_handle(h, tmp_path)
	server.behaviors["state"] = {"isIdle": False, "hasPendingMessages": False}
	try:
		h.abort()
		assert h.status == "stopped"
		h.resume("keep going")
		assert h.closed is False
		assert h.status == "running"
		with pytest.raises(ValueError):
			h.resume("again")
	finally:
		server.close()


def test_resume_on_running_handle_raises(tmp_path):
	h = AgentHandle("p", name="t4", cwd=str(tmp_path), window_name="t4", model=None, thinking=None, schema=None)
	server = bind_handle(h, tmp_path)
	server.behaviors["state"] = {"isIdle": False, "hasPendingMessages": False}
	try:
		with pytest.raises(ValueError, match="still running"):
			h.resume("x")
	finally:
		server.close()


def test_dict_response_with_schema(tmp_path):
	h = AgentHandle("p", name="t5", cwd=str(tmp_path), window_name="t5", model=None, thinking=None,
					schema={"type": "object", "required": ["root_causes"], "properties": {"root_causes": {"type": "array", "items": {"type": "string"}}}})
	server = bind_handle(h, tmp_path)
	server.behaviors["state"] = {"isIdle": True, "hasPendingMessages": False}
	server.behaviors["message"] = {"content": '```json\n{"root_causes": ["flaky test"]}\n```', "timestamp": 5}
	try:
		resp = h.wait(timeout=5)
		assert isinstance(resp, AgentDictResponse)
		assert resp["root_causes"] == ["flaky test"]
		assert resp.valid is True
		assert resp.get_session() is h.session
	finally:
		server.close()


def test_dict_response_retry_and_failure(tmp_path, monkeypatch):
	monkeypatch.setenv("PI_SUBAGENTS_SCHEMA_RETRIES", "1")
	h = AgentHandle("p", name="t6", cwd=str(tmp_path), window_name="t6", model=None, thinking=None,
					schema={"type": "object", "required": ["x"], "properties": {"x": {"type": "string"}}})
	server = bind_handle(h, tmp_path)
	server.behaviors["state"] = {"isIdle": True, "hasPendingMessages": False}
	server.behaviors["message"] = {"content": "not json at all", "timestamp": 6}
	try:
		resp = h.wait(timeout=5)
		assert resp.valid is False
		assert resp["raw"] == "not json at all"
	finally:
		server.close()


def test_registry_emits_via_ptc_bridge(tmp_path):
	seen = []
	def bridge(snapshot):
		seen.append(snapshot)
	builtins.PTC_STATE_EMIT = bridge
	try:
		h = AgentHandle("p", name="t7", cwd=str(tmp_path), window_name="t7", model=None, thinking=None, schema=None)
		server = bind_handle(h, tmp_path)
		server.behaviors["state"] = {"isIdle": True, "hasPendingMessages": False}
		server.behaviors["message"] = {"content": "x", "timestamp": 9}
		try:
			from pi_subagents.registry import REGISTRY
			REGISTRY.register(h)
			REGISTRY.emit()
			assert seen and seen[0]["agents"][0]["name"] == "t7"
		finally:
			server.close()
			del builtins.PTC_STATE_EMIT
	finally:
		pass


def test_registry_prints_status_line_without_bridge(tmp_path, capsys):
	from pi_subagents.registry import REGISTRY
	assert getattr(builtins, "PTC_STATE_EMIT", None) is None
	h = AgentHandle("p", name="t8", cwd=str(tmp_path), window_name="t8", model=None, thinking=None, schema=None)
	server = bind_handle(h, tmp_path)
	server.behaviors["state"] = {"isIdle": False, "hasPendingMessages": False}
	try:
		REGISTRY.register(h)
		REGISTRY.emit()
		out = capsys.readouterr().out
		assert "subagents:" in out and "running" in out and "t8" in out
	finally:
		server.close()


def test_activity_absorbed_into_handle_state(tmp_path):
	h = AgentHandle("p", name="t9", cwd=str(tmp_path), window_name="t9", model=None, thinking=None, schema=None)
	server = bind_handle(h, tmp_path)
	server.behaviors["state"] = {"isIdle": True, "hasPendingMessages": False}
	server.behaviors["message"] = {"content": "ok", "timestamp": 10}
	server.behaviors["activity"] = {
		"available": True,
		"activity": {
			"phase": "tool",
			"label": "reading tests",
			"run": {"toolCalls": 4, "thinkingMs": 6200, "elapsedMs": 12000},
		},
	}
	try:
		h.wait(timeout=5)
		snap = h.agent_state()
		assert snap["phase"] == "tool"
		assert snap["label"] == "reading tests"
		assert snap["toolCalls"] == 4
		assert snap["thinkingMs"] == 6200
	finally:
		server.close()


def test_concurrent_limit(tmp_path, monkeypatch):
	monkeypatch.setenv("PI_SUBAGENTS_MAX_CONCURRENT", "1")
	from pi_subagents.registry import REGISTRY
	h = AgentHandle("p", name="t10", cwd=str(tmp_path), window_name="t10", model=None, thinking=None, schema=None)
	server = bind_handle(h, tmp_path)
	server.behaviors["state"] = {"isIdle": False, "hasPendingMessages": False}
	REGISTRY.register(h)
	try:
		with pytest.raises(Exception, match="limit reached"):
			handle_mod.spawn_pi_window_handle("p2", name="t11", cwd=str(tmp_path))
	finally:
		server.close()


def test_session_jsonl_trajectory(tmp_path):
	h = AgentHandle("p", name="t12", cwd=str(tmp_path), window_name="t12", model=None, thinking=None, schema=None)
	server = bind_handle(h, tmp_path)
	session_path = os.path.join(str(tmp_path), "session.jsonl")
	entries = [
		{"type": "session", "version": 3},
		{"type": "message", "id": "e1", "parentId": None, "timestamp": "2026-09-15T18:00:00.000Z", "message": {"role": "user", "content": "go", "timestamp": "2026-09-15T18:00:00.000Z"}},
		{"type": "message", "id": "e2", "parentId": "e1", "timestamp": "2026-09-15T18:00:02.000Z", "message": {"role": "assistant", "content": [
			{"type": "thinking", "thinking": "hmm"},
			{"type": "toolCall", "id": "c1", "name": "read", "arguments": {"path": "x.py"}},
		], "timestamp": "2026-09-15T18:00:02.000Z"}},
		{"type": "message", "id": "e3", "parentId": "e2", "timestamp": "2026-09-15T18:00:05.000Z", "message": {"role": "toolResult", "toolCallId": "c1", "toolName": "read", "content": [{"type": "text", "text": "..."}], "isError": False, "timestamp": "2026-09-15T18:00:05.000Z"}},
		{"type": "message", "id": "e4", "parentId": "e3", "timestamp": "2026-09-15T18:00:08.000Z", "message": {"role": "assistant", "content": [{"type": "text", "text": "Root cause: X"}], "timestamp": "2026-09-15T18:00:08.000Z"}},
	]
	with open(session_path, "w") as fh:
		for e in entries:
			fh.write(json.dumps(e) + "\n")
	server.behaviors["state"] = {"isIdle": True, "hasPendingMessages": False, "sessionFile": session_path}
	server.behaviors["message"] = {"content": "Root cause: X", "timestamp": 1726425608000}
	try:
		resp = h.wait(timeout=5)
		session = resp.get_session()
		assert session.available
		assert session.turns == 2
		assert session.prose.endswith("Root cause: X")
		assert session.thinking == "hmm"
		calls = session.tool_calls
		assert len(calls) == 1 and calls[0]["tool"] == "read"
		assert calls[0]["durationMs"] == 3000
		assert calls[0]["isError"] is False
		assert session.duration_ms == 8000
		kinds = {e["kind"] for e in session.trajectory()}
		assert kinds == {"prose", "thinking", "tool_call", "tool_result"}
		session.invalidate()
	finally:
		server.close()


def test_agent_handle_str_shows_result_or_status(tmp_path):
	h = AgentHandle("p", name="st", cwd=str(tmp_path), window_name="st", model=None, thinking=None, schema=None)
	server = bind_handle(h, tmp_path)
	server.behaviors["state"] = {"isIdle": False, "hasPendingMessages": False}
	try:
		assert "status=starting" in str(h)
		h.status = "running"
		assert "status=running" in str(h)
		assert "AgentHandle object at" not in repr(h)
		h.settled_data = {"lastAssistant": {"content": "the reply", "timestamp": 1}}
		assert str(h) == "the reply"
	finally:
		server.close()


def test_str_response_plain_string_semantics(tmp_path):
	h = AgentHandle("p", name="t13", cwd=str(tmp_path), window_name="t13", model=None, thinking=None, schema=None)
	server = bind_handle(h, tmp_path)
	server.behaviors["state"] = {"isIdle": True, "hasPendingMessages": False}
	server.behaviors["message"] = {"content": "answer", "timestamp": 11}
	try:
		resp = h.wait(timeout=5)
		assert f"got {resp}" == "got answer"
		assert resp in ("answer",)
		assert len(resp) == 6
	finally:
		server.close()


def test_finish_kills_windows_and_prunes_sockets(tmp_path, monkeypatch):
	"""After finish(), tmux windows are gone and sockets are unlinked."""
	from pi_subagents.registry import REGISTRY
	import pi_subagents.handle as handle_mod

	killed = []
	monkeypatch.setattr(handle_mod, "kill_window", lambda window_id: killed.append(window_id))
	# window_alive is consulted by handle.is_window_alive; keep it permissive.
	monkeypatch.setattr(handle_mod, "window_alive", lambda window_id: False)

	h1 = AgentHandle("p", name="f1", cwd=str(tmp_path), window_name="f1", model=None, thinking=None, schema=None)
	server1 = bind_handle(h1, tmp_path)
	server1.behaviors["state"] = {"isIdle": True, "hasPendingMessages": False}
	server1.behaviors["message"] = {"content": "x", "timestamp": 1}
	h2 = AgentHandle("p", name="f2", cwd=str(tmp_path), window_name="f2", model=None, thinking=None, schema=None)
	server2 = bind_handle(h2, tmp_path)
	REGISTRY.register(h1)
	REGISTRY.register(h2)
	try:
		str(h1.wait(timeout=5))
		closed = __import__("pi_subagents").finish([h1, h2])
		assert closed == 2
		assert killed == [h1.window_id, h2.window_id]
		assert h1.status == "dead" and h2.status == "dead"
		assert not os.path.exists(h1.socket_path), "socket should be unlinked after kill"
		assert REGISTRY.handles()[0].status == "dead"
	finally:
		server1.close()
		server2.close()


def test_finish_defaults_to_every_spawned_handle(tmp_path, monkeypatch):
	from pi_subagents import finish
	from pi_subagents.registry import REGISTRY
	import pi_subagents.handle as handle_mod

	killed = []
	monkeypatch.setattr(handle_mod, "kill_window", lambda window_id: killed.append(window_id))

	handles = []
	servers = []
	for name in ("g1", "g2", "g3"):
		h = AgentHandle("p", name=name, cwd=str(tmp_path), window_name=name, model=None, thinking=None, schema=None)
		server = bind_handle(h, tmp_path)
		server.behaviors["state"] = {"isIdle": True, "hasPendingMessages": False}
		server.behaviors["message"] = {"content": "x", "timestamp": 1}
		REGISTRY.register(h)
		handles.append(h)
		servers.append(server)
	try:
		str(h.wait(timeout=5))
		assert __import__("pi_subagents").finish() == 3
		assert killed == [h.window_id for h in handles]
		assert all(h.status == "dead" for h in handles)
	finally:
		for server in servers:
			server.close()


def test_startup_delivers_the_initial_prompt_over_pisock(tmp_path, monkeypatch):
	"""The initial prompt travels over pi-sock, not through tmux argv."""
	from pi_subagents.handle import AgentHandle as AH

	monkeypatch.setenv("PI_SUBAGENTS_STARTUP_TIMEOUT", "10")
	monkeypatch.setattr("pi_subagents.handle.window_alive", lambda window_id: True)

	prompt = "it's a test: don't \"fail\" on 'quotes' " * 50
	h = AH(prompt, name="boot", cwd=str(tmp_path), window_name="boot", model=None, thinking=None, schema=None)
	server = bind_handle(h, tmp_path)
	h._startup_started = True
	try:
		h._run_startup()
		assert h.status == "running"
		assert h._startup_error is None
		assert len(server.sent) == 1
		assert server.sent[0]["type"] == "send"
		assert server.sent[0]["text"] == prompt, "the prompt must arrive byte-for-byte"
	finally:
		server.close()


def test_startup_failure_marks_the_handle_failed(tmp_path, monkeypatch):
	from pi_subagents.handle import AgentHandle as AH
	from pi_subagents import PiSubagentsError  # noqa: F401  (re-exported)

	monkeypatch.setenv("PI_SUBAGENTS_STARTUP_TIMEOUT", "5")
	# No server, and the window is gone: startup must fail loudly, not hang.
	monkeypatch.setattr("pi_subagents.handle.window_alive", lambda window_id: False)

	h = AH("hello", name="boot-fail", cwd=str(tmp_path), window_name="boot-fail", model=None, thinking=None, schema=None)
	h._startup_started = True
	h.window_id = "gone"
	h.socket_path = os.path.join(str(tmp_path), "never.sock")
	h._run_startup()

	assert h.status == "failed"
	assert h._startup_error is not None
	with pytest.raises(PiSubagentsError, match="startup failed"):
		h.wait(timeout=1)

"""AgentHandle — a currently executing subagent pi instance.

Wraps a pi-sock unix socket and its tmux window. All control (send/abort/
resume/kill) lives here; inspection of the resulting conversation lives on the
AgentSession it exposes (the pi session JSONL on disk).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

from .client import AsyncSockClient, PiSockError, PiSockUnavailable, SockClient
from .envcheck import require_environment
from .registry import REGISTRY
from .response import AgentDictResponse, AgentStrResponse
from .schema import SchemaValidationError, validate_schema, validate_with_schema
from .tmuxenv import kill_window, max_concurrent, spawn_pi_window, unique_socket_name, window_alive

DEFAULT_SETTLE_TIMEOUT = 30 * 60.0


def _settle_timeout_default() -> float:
	try:
		return max(1.0, float(os.environ.get("PI_SUBAGENTS_SETTLE_TIMEOUT", str(DEFAULT_SETTLE_TIMEOUT))))
	except ValueError:
		return DEFAULT_SETTLE_TIMEOUT


class AgentHandle:
	"""Live subagent. Await it for the settled response; steer it while running."""

	def __init__(
		self,
		prompt: str,
		*,
		name: str,
		cwd: str,
		window_name: str,
		model: str | None,
		thinking: str | None,
		schema: dict | None,
	):
		self.id = unique_socket_name(name or "subagent")
		self.name = name or self.id
		self.prompt = prompt
		self.cwd = cwd
		self.window_name = window_name or self.name
		self.model = model
		self.thinking = thinking
		self.schema = schema
		self.started_at = time.time() * 1000
		self.status = "starting"
		self.closed = False
		self.settled_data: dict | None = None
		self.window_id: str | None = None
		self.socket_path: str | None = None
		# latest pi-tool-tree activity (filled by activity polling)
		self.phase: str | None = None
		self.label: str | None = None
		self.tool_calls = 0
		self.thinking_ms = 0
		self._session: AgentSession | None = None
		self._sync = SockClient(f"<unbound:{self.id}>")
		self._async = AsyncSockClient(f"<unbound:{self.id}>")

	def _bind(self, window_ref) -> None:
		self.window_id = window_ref.window_id
		self.socket_path = window_ref.socket_path
		self._sync = SockClient(window_ref.socket_path)
		self._async = AsyncSockClient(window_ref.socket_path)
		self.status = "running"

	def agent_state(self) -> dict:
		"""Compact snapshot row for the registry."""
		elapsed = (time.time() * 1000) - self.started_at
		return {
			"id": self.id,
			"name": self.name,
			"status": self.status,
			"startedAt": self.started_at,
			"elapsedMs": round(elapsed),
			"socketPath": self.socket_path,
			"windowId": self.window_id,
			"toolCalls": self.tool_calls,
			"thinkingMs": self.thinking_ms,
			"phase": self.phase,
			"label": self.label,
			"depth": int(os.environ.get("PI_SUBAGENTS_MAX_DEPTH", "0") or 0) or None,
		}

	# -- control ------------------------------------------------------------

	def send(self, text: str, mode: str = "steer") -> dict:
		"""Steer a running agent (mode="steer") or queue a follow-up."""
		return self._sync.send(text, mode=mode)

	async def send_async(self, text: str, mode: str = "steer") -> dict:
		return await self._async.send(text, mode)

	def abort(self) -> dict:
		"""Stop the current run. The handle closes; resume with resume_async."""
		try:
			result = self._sync.abort()
		except PiSockUnavailable:
			self._mark_dead()
			return {"aborted": False, "dead": True}
		self.status = "stopped"
		self.closed = True
		REGISTRY.emit()
		return result

	async def abort_async(self) -> dict:
		try:
			result = await self._async.abort()
		except PiSockUnavailable:
			self._mark_dead()
			return {"aborted": False, "dead": True}
		self.status = "stopped"
		self.closed = True
		REGISTRY.emit()
		return result

	def resume(self, prompt: str | None = None) -> "AgentHandle":
		"""Un-close the handle and send a new prompt on the same pi session.

		Re-sending a prompt continues the session from where the agent left off
		(pi sessions persist on disk). This is the intended "resume".
		"""
		if not self.closed and self.status not in ("stopped", "failed", "dead", "settled"):
			raise ValueError("resume on a handle that is still running; abort it first")
		self.closed = False
		self.settled_data = None
		self.status = "running"
		REGISTRY.emit()
		text = prompt if prompt is not None else "Continue."
		try:
			self._sync.send(text, mode="follow_up")
		except PiSockUnavailable:
			self._mark_dead()
		return self

	async def resume_async(self, prompt: str | None = None) -> "AgentHandle":
		if not self.closed and self.status not in ("stopped", "failed", "dead", "settled"):
			raise ValueError("resume on a handle that is still running; abort it first")
		self.closed = False
		self.settled_data = None
		self.status = "running"
		REGISTRY.emit()
		text = prompt if prompt is not None else "Continue."
		try:
			await self._async.send(text, mode="follow_up")
		except PiSockUnavailable:
			self._mark_dead()
		return self

	def kill(self) -> None:
		"""Abort and tear down the tmux window."""
		try:
			self.abort()
		except Exception:
			pass
		if self.window_id:
			kill_window(self.window_id)
		self.status = "dead"
		REGISTRY.emit()

	# -- observation --------------------------------------------------------

	def state(self) -> dict:
		return self._sync.state()

	async def state_async(self) -> dict:
		return await self._async.state()

	def activity(self) -> dict | None:
		"""Latest pi-tool-tree snapshot (None when the relay is unavailable)."""
		snap = self._sync.activity()
		if snap and snap.get("available"):
			self._absorb_activity(snap)
			return snap
		return None

	async def activity_async(self) -> dict | None:
		snap = await self._async.activity()
		if snap and snap.get("available"):
			self._absorb_activity(snap)
			return snap
		return None

	def _absorb_activity(self, snap: dict) -> None:
		activity = snap.get("activity") or snap
		self.phase = activity.get("phase")
		self.label = activity.get("label")
		run = activity.get("run") or {}
		self.tool_calls = run.get("toolCalls", self.tool_calls)
		self.thinking_ms = run.get("thinkingMs", self.thinking_ms)

	def _mark_dead(self) -> None:
		self.status = "dead"
		self.closed = True
		REGISTRY.emit()

	def is_window_alive(self) -> bool:
		return bool(self.window_id and window_alive(self.window_id))

	# -- waiting ------------------------------------------------------------

	def wait(self, timeout: float | None = None) -> Any:
		"""Block until the agent settles; returns AgentStrResponse/AgentDictResponse."""
		if self.closed and self.status != "settled":
			raise ValueError("await on closed handle (it was aborted; call resume() first)")
		if self.settled_data is not None:
			return self._response()
		if self.status == "starting":
			# window spawned; pi may still be booting — the socket appears when
			# the session starts. wait_settled treats missing sockets as dead,
			# so poll for the socket to appear first.
			if not self._await_socket(self.socket_path, timeout):
				self.status = "failed"
				REGISTRY.emit()
				raise TimeoutError(f"subagent {self.name}: pi-sock socket never appeared")
		settle = self._sync.wait_settled(
			timeout if timeout is not None else _settle_timeout_default(),
			on_tick=self._absorb_from_state,
		)
		return self._finish_wait(settle)

	async def wait_async(self, timeout: float | None = None) -> Any:
		if self.closed and self.status != "settled":
			raise ValueError("await on closed handle (it was aborted; call resume_async() first)")
		if self.settled_data is not None:
			return self._response()
		if self.status == "starting":
			if not await self._await_socket_async():
				self.status = "failed"
				REGISTRY.emit()
				raise TimeoutError(f"subagent {self.name}: pi-sock socket never appeared")
		try:
			settle = await self._async.wait_settled(
				timeout if timeout is not None else _settle_timeout_default(),
				on_tick=self._absorb_from_state_async,
			)
		except PiSockUnavailable:
			self._mark_dead()
			raise
		return self._finish_wait(settle)

	def __await__(self):
		return self.wait_async().__await__()

	def _await_socket(self, path: str | None, timeout: float | None) -> bool:
		deadline = (time.monotonic() + (timeout or 60.0)) if timeout is not None else time.monotonic() + 60.0
		while time.monotonic() < deadline:
			if path and os.path.exists(path):
				try:
					if self._sync.is_alive():
						return True
				except Exception:
					pass
			if not self.is_window_alive():
				return False
			time.sleep(0.25)
		return False

	async def _await_socket_async(self) -> bool:
		deadline = time.monotonic() + 60.0
		while time.monotonic() < deadline:
			if self.socket_path and os.path.exists(self.socket_path):
				try:
					await self._async.state()
					return True
				except Exception:
					pass
			if not self.is_window_alive():
				return False
			await asyncio.sleep(0.25)
		return False

	def _absorb_from_state(self, _state: dict) -> None:
		try:
			self.activity()
		except Exception:
			pass

	async def _absorb_from_state_async(self, _state: dict) -> None:
		try:
			await self.activity_async()
		except Exception:
			pass

	def _finish_wait(self, settle: dict | None) -> Any:
		if settle is None:
			if not self.is_window_alive():
				self._mark_dead()
				raise TimeoutError(f"subagent {self.name}: pi process died before settling")
			self.status = "failed"
			REGISTRY.emit()
			raise TimeoutError(f"subagent {self.name}: settle timeout exceeded")
		self._absorb_from_state(settle)
		self.settled_data = settle
		self.status = "settled"
		REGISTRY.emit()
		return self._response()

	def _response(self) -> Any:
		text = ""
		settle = self.settled_data or {}
		last = settle.get("lastAssistant") or None
		if isinstance(last, dict):
			text = last.get("content") or ""
		if self.schema is not None:
			return _dict_response(self, text, self.session)
		return AgentStrResponse(text, self.session)

	@property
	def session(self) -> "AgentSession":
		if self._session is None:
			from .session_file import AgentSession

			self._session = AgentSession(self)
		return self._session


def _dict_response(handle: AgentHandle, text: str, session) -> "AgentDictResponse":
	attempt_error: str | None = None
	retries = _schema_retries()
	parsed: dict = {}
	for attempt in range(max(1, retries)):
		parsed, error = _parse_json(text)
		if error is None:
			try:
				validate_with_schema(handle.schema, parsed)
				return AgentDictResponse(parsed, session)
			except SchemaValidationError as schema_error:
				attempt_error = f"schema: {schema_error}"
		else:
			attempt_error = error
		if attempt < max(1, retries) - 1:
			# ask the agent to emit valid JSON matching the schema
			try:
				handle.send(_schema_retry_prompt(handle.schema, attempt_error))
			except PiSockError:
				break
			settle = handle._sync.wait_settled(_settle_timeout_default())
			if settle is None:
				break
			handle.settled_data = settle
			last = settle.get("lastAssistant") or {}
			text = last.get("content") or ""
			session.invalidate()
	return AgentDictResponse(
		{"valid": False, "error": attempt_error or "no reply", "raw": text},
		session,
	)


def _parse_json(text: str) -> tuple[dict | None, str | None]:
	clean = text.strip()
	if clean.startswith("```"):
		clean = clean.strip("`")
		if clean.startswith("json"):
			clean = clean[4:]
	try:
		value = json.loads(clean)
	except json.JSONDecodeError as error:
		match = None
		depth = 0
		start = -1
		for index, char in enumerate(clean):
			if char == "{":
				if depth == 0:
					start = index
				depth += 1
			elif char == "}":
				depth -= 1
				if depth == 0 and start >= 0:
					match = clean[start : index + 1]
					break
		if match:
			try:
				return json.loads(match), None
			except json.JSONDecodeError:
				pass
		return None, f"invalid JSON: {error}"
	if not isinstance(value, dict):
		return None, "reply was JSON but not an object"
	return value, None


def _schema_retry_prompt(schema: dict, error: str | None) -> str:
	spec = json.dumps(schema)
	suffix = f" Your previous reply failed validation: {error}." if error else ""
	return (
		"Reply again with ONLY a JSON object (no prose, no code fence) matching "
		f"this schema: {spec}.{suffix}"
	)


def _schema_retries() -> int:
	try:
		return max(1, int(os.environ.get("PI_SUBAGENTS_SCHEMA_RETRIES", "3")))
	except ValueError:
		return 3


def spawn_pi_window_handle(
	prompt: str,
	*,
	name: str | None = None,
	cwd: str | None = None,
	window_name: str | None = None,
	model: str | None = None,
	thinking: str | None = None,
	schema: dict | None = None,
) -> AgentHandle:
	"""Spawn a pi window and return its AgentHandle (used by subagents.agent)."""
	depth = int(os.environ.get("PI_SUBAGENT_DEPTH", "0") or 0) + 1
	running = sum(1 for state in REGISTRY.snapshot()["agents"] if state["status"] in ("starting", "running"))
	if running >= max_concurrent():
		raise PiSockError(
			f"concurrent subagent limit reached ({max_concurrent()}); abort or wait for one to settle"
		)
	handle = AgentHandle(
		prompt,
		name=name or "subagent",
		cwd=cwd or os.getcwd(),
		window_name=window_name or name or "subagent",
		model=model,
		thinking=thinking,
		schema=schema,
	)
	window_ref = spawn_pi_window(
		prompt,
		name=handle.name,
		cwd=handle.cwd,
		window_name=handle.window_name,
		model=handle.model,
		thinking=handle.thinking,
		socket_name=handle.id,
		depth=depth,
	)
	handle._bind(window_ref)
	REGISTRY.register(handle)
	return handle

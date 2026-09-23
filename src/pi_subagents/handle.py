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
		# Frozen at the moment the agent stops running, so a settled row shows its
		# actual runtime instead of growing forever with the age of the handle.
		runtime_ms: int | None = None
		self.runtime_ms = runtime_ms
		self.status = "starting"
		self.closed = False
		self.group = None
		# Exec this agent was spawned in (the session runtime stamps the current
		# exec id into builtins.PTC_EXEC_SCOPE); the host's viewer scopes by it.
		from .registry import current_exec_scope

		self.exec_scope = current_exec_scope()
		self.settled_data: dict | None = None
		self.window_id: str | None = None
		self.socket_path: str | None = None
		# latest pi-tool-tree activity (filled by activity polling)
		self.phase: str | None = None
		self.label: str | None = None
		self.label_elapsed_ms: int | None = None
		self.label_calls: int | None = None
		self.live_tool: str | None = None
		self.tool_calls = 0
		self.thinking_ms = 0
		# viewer fields: is an await pending on this handle right now?
		self.awaited = False
		# context usage from pi-sock get_state (tokens / limit / percent)
		self.ctx_tokens: int | None = None
		self.ctx_limit: int | None = None
		self.ctx_percent: float | None = None
		self._session: AgentSession | None = None
		self._sync = SockClient(f"<unbound:{self.id}>")
		self._async = AsyncSockClient(f"<unbound:{self.id}>")

	def _bind(self, window_ref) -> None:
		self.window_id = window_ref.window_id
		self.socket_path = window_ref.socket_path
		self._sync = SockClient(window_ref.socket_path)
		self._async = AsyncSockClient(window_ref.socket_path)
		self.status = "running"

	def _stamp_runtime(self) -> None:
		"""Freeze the runtime the moment the agent stops running."""
		if self.runtime_ms is None:
			self.runtime_ms = round((time.time() * 1000) - self.started_at)

	def agent_state(self) -> dict:
		"""Compact snapshot row for the registry."""
		elapsed = self.runtime_ms
		if elapsed is None:
			elapsed = round((time.time() * 1000) - self.started_at)
		return {
			"id": self.id,
			"name": self.name,
			"group": self.group,
			"status": self.status,
			"execScope": self.exec_scope,
			"startedAt": self.started_at,
			"elapsedMs": round(elapsed),
			"socketPath": self.socket_path,
			"windowId": self.window_id,
			"toolCalls": self.tool_calls,
			"thinkingMs": self.thinking_ms,
			"phase": self.phase,
			"label": self.label,
			"labelElapsedMs": self.label_elapsed_ms,
			"labelCalls": self.label_calls,
			"liveTool": self.live_tool,
			"awaited": self.awaited,
			"ctx": {
				"tokens": self.ctx_tokens,
				"limit": self.ctx_limit,
				"percent": self.ctx_percent,
			} if self.ctx_tokens is not None else None,
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
		self._stamp_runtime()
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
		self._stamp_runtime()
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
		"""Close the subagent permanently: abort the run and destroy its tmux window.

		Call once the work is done and there is no need to resume, steer, or read the
		session history again — this is what keeps long sessions from accumulating
		orphaned tmux windows. The handle's status becomes "dead" and the socket file
		is cleaned up best-effort.
		"""
		try:
			self.abort()
		except Exception:
			pass
		if self.window_id:
			kill_window(self.window_id)
		self.status = "dead"
		self._stamp_runtime()
		try:
			if self.socket_path and os.path.exists(self.socket_path):
				os.unlink(self.socket_path)
		except OSError:
			pass
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
		self.label_elapsed_ms = activity.get("labelElapsedMs", self.label_elapsed_ms)
		self.label_calls = activity.get("labelCalls", self.label_calls)
		run = activity.get("run") or {}
		self.tool_calls = run.get("toolCalls", self.tool_calls)
		self.thinking_ms = run.get("thinkingMs", self.thinking_ms)
		calls = activity.get("calls") or []
		if isinstance(calls, list) and calls:
			call = calls[0]
			tool = call.get("toolName") if isinstance(call, dict) else None
			preview = call.get("argPreview") if isinstance(call, dict) else None
			self.live_tool = (f"{tool} {preview}" if preview else tool) if tool else self.live_tool
		else:
			self.live_tool = None

	def _mark_dead(self) -> None:
		self.status = "dead"
		self._stamp_runtime()
		self.closed = True
		REGISTRY.emit()

	def is_window_alive(self) -> bool:
		return bool(self.window_id and window_alive(self.window_id))

	# -- waiting ------------------------------------------------------------

	def wait(self, timeout: float | None = None, poll: float = 1.0) -> Any:
		"""Block until the agent settles; returns AgentStrResponse/AgentDictResponse."""
		if self.closed and self.status != "settled":
			raise ValueError("await on closed handle (it was aborted; call resume() first)")
		if self.settled_data is not None:
			return self._response()
		if self.socket_path and not os.path.exists(self.socket_path):
			# pi may still be booting — the socket appears when the session
			# starts. wait_settled treats missing sockets as dead, so wait for
			# the socket to appear first.
			if not self._await_socket(self.socket_path, timeout):
				self._mark_dead()
				raise TimeoutError(f"subagent {self.name}: pi-sock socket never appeared")
		self.awaited = True
		REGISTRY.emit()
		try:
			settle = self._sync.wait_settled(
				timeout if timeout is not None else _settle_timeout_default(),
				poll=poll,
				on_tick=self._tick_sync,
			)
		finally:
			self.awaited = False
			REGISTRY.emit()
		return self._finish_wait(settle)

	async def wait_async(self, timeout: float | None = None, poll: float = 1.0) -> Any:
		if self.closed and self.status != "settled":
			raise ValueError("await on closed handle (it was aborted; call resume_async() first)")
		if self.settled_data is not None:
			return self._response()
		if self.socket_path and not os.path.exists(self.socket_path):
			# pi may still be booting — the socket appears when the session
			# starts. wait_settled treats missing sockets as dead, so wait for
			# the socket to appear first.
			if not self._await_socket(self.socket_path, timeout):
				self._mark_dead()
				raise TimeoutError(f"subagent {self.name}: pi-sock socket never appeared")
		self.awaited = True
		REGISTRY.emit()
		try:
			settle = await self._async.wait_settled(
				timeout if timeout is not None else _settle_timeout_default(),
				poll=poll,
				on_tick=self._tick_async,
			)
		except PiSockUnavailable:
			self._mark_dead()
			raise
		finally:
			self.awaited = False
			REGISTRY.emit()
		return self._finish_wait(settle)

	def __str__(self) -> str:
		"""The settled response text once available, otherwise a status line."""
		if self.settled_data is not None:
			last = (self.settled_data or {}).get("lastAssistant") or {}
			text = last.get("content") if isinstance(last, dict) else None
			if text:
				return str(text)
		return f"<AgentHandle {self.name!r} status={self.status}>"

	def __repr__(self) -> str:
		return self.__str__()

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

	def _absorb_from_state(self, state: dict | None = None) -> None:
		self._absorb_ctx(state)
		try:
			self.activity()
		except Exception:
			pass
			pass

	async def _absorb_from_state_async(self, state: dict | None = None) -> None:
		self._absorb_ctx(state)
		try:
			await self.activity_async()
		except Exception:
			pass
			pass

	def _tick_sync(self, state: dict) -> None:
		self._absorb_from_state(state)
		REGISTRY.emit()

	async def _tick_async(self, state: dict) -> None:
		await self._absorb_from_state_async(state)
		REGISTRY.emit()

	def _absorb_ctx(self, state: dict | None) -> None:
		context = (state or {}).get("context")
		if isinstance(context, dict):
			tokens = context.get("tokens")
			if tokens is not None:
				self.ctx_tokens = tokens
				self.ctx_limit = context.get("limit") or context.get("contextWindow")
				self.ctx_percent = context.get("percent")

	def _finish_wait(self, settle: dict | None) -> Any:
		if settle is None:
			if not self.is_window_alive():
				self._mark_dead()
				raise TimeoutError(f"subagent {self.name}: pi process died before settling")
			self.status = "failed"
			self._stamp_runtime()
			REGISTRY.emit()
			raise TimeoutError(f"subagent {self.name}: settle timeout exceeded")
		self._absorb_from_state(settle)
		self.settled_data = settle
		self.status = "settled"
		self._stamp_runtime()
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
	profile: str | os.PathLike[str] | None = None,
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
		profile=profile,
	)
	handle.group = current_phase()
	handle._bind(window_ref)
	REGISTRY.register(handle)
	return handle


# Grouping (viewer support): the orchestrating agent labels phases before
# spawning; agents spawned under a phase are grouped under it in the PTC viewer.
# Phase state (label + start time) lives on the registry.


def current_phase() -> str | None:
	return REGISTRY.current_phase()


def set_phase(label: str | None) -> None:
	REGISTRY.set_phase(label)

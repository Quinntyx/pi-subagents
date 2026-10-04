"""AgentHandle — a currently executing subagent pi instance.

Wraps a pi-sock unix socket and its tmux window. All control (send/abort/
resume/kill) lives here; inspection of the resulting conversation lives on the
AgentSession it exposes (the pi session JSONL on disk).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import json
import os
import time
from typing import Any

from .client import (
	AsyncSockClient, PiSockError, PiSockSessionEnded, PiSockUnavailable, SockClient,
	_last_assistant_outcome, _assistant_outcome_marker, raise_for_failed_outcome,
)
from .errors import PiSubagentsError, PiSubagentsTimeoutError
from .envcheck import require_environment
from .registry import REGISTRY
from .response import AgentDictResponse, AgentStrResponse
from .schema import SchemaValidationError, validate_schema, validate_with_schema
from .tmuxenv import (
	kill_window, resolve_agent_dir, spawn_pi_window, unique_socket_name, window_alive,
)

DEFAULT_SETTLE_TIMEOUT = 30 * 60.0
# How long to wait for a freshly spawned pi to bring its pi-sock socket up and
# accept the initial prompt over the socket.
DEFAULT_STARTUP_TIMEOUT = 90.0


def _settle_timeout_default() -> float:
	try:
		return max(1.0, float(os.environ.get("PI_SUBAGENTS_SETTLE_TIMEOUT", str(DEFAULT_SETTLE_TIMEOUT))))
	except ValueError:
		return DEFAULT_SETTLE_TIMEOUT


def _startup_timeout() -> float:
	try:
		return max(5.0, float(os.environ.get("PI_SUBAGENTS_STARTUP_TIMEOUT", str(DEFAULT_STARTUP_TIMEOUT))))
	except ValueError:
		return DEFAULT_STARTUP_TIMEOUT


def unload_window(window_id: str) -> bool:
	"""Confirm cleanup without mistaking an unreachable tmux server for success.

	Keep kill_window as the underlying boundary for existing wrappers, including
	older wrappers returning None; the built-in reports explicit command failure.
	"""
	killed = kill_window(window_id)
	return killed is not False and not window_alive(window_id)


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
		# Execution time excludes observed between-turn idle periods. The wall
		# runtime above remains the task/result duration; these fields drive the
		# live viewer's non-ticking idle state.
		self._busy_ms = 0.0
		self._busy_since: float | None = self.started_at
		self._idle = False
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
		self.session_file: str | None = None
		self.session_id: str | None = None
		self.last_outcome: str | None = None
		self._dormant = False
		self._cached_state: dict = {}
		self._cached_activity: dict | None = None
		self._agent_dir: str | os.PathLike[str] | None = None
		self._depth = int(os.environ.get("PI_SUBAGENT_DEPTH", "0") or 0) + 1
		# Latest pi-activity API snapshot (filled by activity polling).
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
		# Message present immediately before a follow-up is sent. Settlement must
		# produce a different message so a retained session cannot return stale data.
		self._wait_baseline: dict | None = None
		self._settlement_revision = 0
		# Readiness/delivery of the initial prompt (see _run_startup).
		self._ready: concurrent.futures.Future = concurrent.futures.Future()
		self._startup_error: Exception | None = None
		self._killed = False
		# True only for instances this library started (and therefore must deliver a
		# first prompt to). Manually-constructed handles have nothing to wait for.
		self._startup_started = False
		self._sync = SockClient(f"<unbound:{self.id}>")
		self._async = AsyncSockClient(f"<unbound:{self.id}>")

	def _bind(self, window_ref) -> None:
		self.window_id = window_ref.window_id
		self.socket_path = window_ref.socket_path
		self._sync = SockClient(window_ref.socket_path)
		self._async = AsyncSockClient(window_ref.socket_path)
		# Still "starting" until the initial prompt has been delivered over pi-sock.
		self.status = "starting"

	# -- startup: readiness + first prompt over pi-sock -------------------------

	def _await_ready(self, timeout: float | None = None) -> None:
		"""Block until the initial prompt has been delivered (or startup failed)."""
		if not self._startup_started:
			return
		try:
			self._ready.result(timeout=timeout if timeout is not None else _startup_timeout() + 5.0)
		except Exception as error:  # concurrent.futures.TimeoutError
			raise PiSubagentsError(
				f"subagent {self.name}: the initial prompt was not delivered within "
				f"{_startup_timeout():.0f}s (is the subagent pi still starting?)"
			) from error
		if self._startup_error is not None:
			raise PiSubagentsError(
				f"subagent {self.name}: startup failed: {self._startup_error}"
			) from self._startup_error

	async def _await_ready_async(self, timeout: float | None = None) -> None:
		import asyncio

		if not self._startup_started:
			return
		await asyncio.wait_for(
			asyncio.shield(asyncio.wrap_future(self._ready)),
			timeout=timeout if timeout is not None else _startup_timeout() + 5.0,
		)
		if self._startup_error is not None:
			raise PiSubagentsError(
				f"subagent {self.name}: startup failed: {self._startup_error}"
			) from self._startup_error

	def _run_startup(self) -> None:
		"""Wait for pi-sock, then deliver the initial prompt as the first turn.

		Runs in a daemon thread so spawn() stays instant. The handle's control
		methods wait on `_ready`, so a follow-up steer can never overtake the
		initial prompt. Startup failures retain their windows for inspection.
		"""
		try:
			deadline = time.monotonic() + _startup_timeout()
			while True:
				if self._killed:
					return
				try:
					self._remember_state(self._sync.state())
					break  # pi-sock answers: session started, UI is coming up
				except PiSockUnavailable:
					if not self.is_window_alive():
						raise RuntimeError("the subagent's pi process exited before becoming ready")
					if time.monotonic() >= deadline:
						raise TimeoutError(f"pi-sock did not answer within {_startup_timeout():.0f}s")
					time.sleep(0.2)
			if self._killed:
				return
			# pi is idle here, so a plain send triggers the first turn immediately.
			self._sync.send(self.prompt, mode="steer")
			self.status = "running"
		except Exception as error:
			self._startup_error = error
			self.status = "failed"
			self.last_outcome = "failed"
			self._stamp_runtime()
		finally:
			if not self._ready.done():
				self._ready.set_result(None)

	def _stamp_runtime(self) -> None:
		"""Freeze the runtime the moment the agent stops running."""
		now = time.time() * 1000
		if self.runtime_ms is None:
			self.runtime_ms = round(now - self.started_at)
		if self._busy_since is not None:
			self._busy_ms += max(0, now - self._busy_since)
			self._busy_since = None

	def _absorb_execution_state(self, state: dict | None) -> None:
		"""Track whether pi is executing or idle between orchestrator turns."""
		if not isinstance(state, dict) or "isIdle" not in state:
			return
		# hasPendingMessages means a delivered turn is waiting to start; it is not
		# the retained-session, between-turn idle state shown by the viewer.
		idle = bool(state.get("isIdle")) and not bool(state.get("hasPendingMessages"))
		now = time.time() * 1000
		if idle and not self._idle:
			if self._busy_since is not None:
				self._busy_ms += max(0, now - self._busy_since)
				self._busy_since = None
		elif not idle and self._idle and self.status in ("starting", "running"):
			self._busy_since = now
		self._idle = idle

	def _busy_elapsed_ms(self) -> int:
		busy = self._busy_ms
		if self._busy_since is not None:
			busy += max(0, time.time() * 1000 - self._busy_since)
		return round(busy)

	def agent_state(self) -> dict:
		"""Compact snapshot row for the registry."""
		busy_ms = self._busy_elapsed_ms()
		elapsed = self.runtime_ms if self.runtime_ms is not None else busy_ms
		return {
			"id": self.id,
			"name": self.name,
			"group": self.group,
			"status": self.status,
			"idle": self._idle and self.status == "running",
			"execScope": self.exec_scope,
			"startedAt": self.started_at,
			"elapsedMs": round(elapsed),
			"busyMs": busy_ms,
			"socketPath": self.socket_path,
			"windowId": self.window_id,
			"sessionFile": self.session_file,
			"sessionId": self.session_id,
			"dormant": self._dormant,
			"lastOutcome": self.last_outcome,
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
		self._await_ready()
		return self._sync.send(text, mode=mode)

	async def send_async(self, text: str, mode: str = "steer") -> dict:
		await self._await_ready_async()
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
		"""Start a pool-scheduled follow-up, reopening only a dormant session."""
		if not self.closed and self.status not in ("stopped", "failed", "dead", "settled"):
			raise ValueError("resume on a handle that is still running; abort it first")
		text = prompt if prompt is not None else "Continue."
		if self._dormant:
			return self._reopen(text)
		# A failed first delivery is a past turn's error, not permanent poison
		# for an otherwise live session. Never respawn it: retry normal control
		# only when the scheduler has admitted this explicitly requested turn.
		if self._startup_started:
			self._ready.result(timeout=_startup_timeout() + 5)
		self._startup_error = None
		self._await_ready()
		self._capture_outcome_baseline()
		try:
			self._wait_baseline = self._sync.message()
		except PiSockError:
			self._wait_baseline = None
		self._reset_turn()
		self.prompt = text
		self.status = "running"
		REGISTRY.emit()
		try:
			self._sync.send(text, mode="follow_up")
		except PiSockUnavailable:
			self._mark_dead()
			raise
		return self

	async def resume_async(self, prompt: str | None = None) -> "AgentHandle":
		return await asyncio.to_thread(self.resume, prompt)

	def _reset_turn(self) -> None:
		self.closed = False
		self.settled_data = None
		self.runtime_ms = None
		self.started_at = time.time() * 1000
		self._busy_ms = 0.0
		self._busy_since = self.started_at
		self._idle = False
		self.last_outcome = None
		# Freeze historical inspection before the continuation appends new turns.
		if self._session is not None:
			self._session._load()
		self._session = None

	def _prepare_accepted_deliveries(self, deliveries: tuple) -> None:
		"""Re-wait accepted direct turns without respawning or releasing capacity."""
		self.settled_data = None
		self._wait_baseline = None
		self.last_outcome = None
		self.status = "running"
		self.runtime_ms = None
		self.session.invalidate()
		for client in (self._sync, self._async):
			client._after_outcome = None
			client._after_deliveries = deliveries

	def _capture_outcome_baseline(self) -> None:
		marker = _assistant_outcome_marker(self.session_file)
		self._sync._after_outcome = marker
		self._async._after_outcome = marker
		self._sync._after_deliveries = None
		self._async._after_deliveries = None

	def _reopen(self, prompt: str) -> "AgentHandle":
		if not self.session_file or not os.path.isfile(self.session_file):
			raise PiSubagentsError(f"subagent {self.name}: durable session file is unavailable")
		baseline = (self.settled_data or {}).get("lastAssistant")
		window_ref = spawn_pi_window(
			prompt, name=self.name, cwd=self.cwd, window_name=self.window_name,
			model=self._cached_state.get("model") or self.model,
			thinking=self._cached_state.get("thinkingLevel") or self.thinking,
			socket_name=unique_socket_name(self.name), depth=self._depth,
			agentDir=self._agent_dir, session_file=self.session_file,
		)
		self.prompt = prompt
		self._reset_turn()
		self._wait_baseline = baseline
		self._dormant = False
		self._ready = concurrent.futures.Future()
		self._startup_error = None
		self._killed = False
		self._bind(window_ref)
		self._capture_outcome_baseline()
		self._startup_started = True
		threading.Thread(target=self._run_startup, name=f"subagent-startup-{self.id}", daemon=True).start()
		return self

	@property
	def dormant(self) -> bool:
		return self._dormant

	def hibernate(self) -> bool:
		"""Unload only an idle, validated success with a durable terminal outcome."""
		if self._dormant:
			return True
		if self.status != "settled" or self.last_outcome != "ok" or self._killed or self.closed:
			return False
		try:
			state = self.state()
			if not state.get("isIdle") or state.get("hasPendingMessages"):
				return False
			path = self.session.session_file
			if not path or not os.path.isfile(path):
				return False
			outcome = _last_assistant_outcome(path)
			if outcome is None or outcome[0] not in ("stop", "length"):
				return False
			self.session._load()
			if not self.window_id:
				return False
			if not unload_window(self.window_id):
				return False
		except Exception:
			# Inspection/cleanup failures never turn a successful task into a failure.
			return False
		self._dormant = True
		return True

	def set_session_name(self, name: str) -> dict:
		"""Rename the live pi session; pi updates its terminal title immediately."""
		self._await_ready()
		return self._sync.set_session_name(name)

	def kill(self) -> None:
		"""Close the subagent permanently: abort the run and destroy its tmux window.

		Call once the work is done and there is no need to resume, steer, or read the
		session history again — this is what keeps long sessions from accumulating
		orphaned tmux windows. The handle's status becomes "dead" and the socket file
		is cleaned up best-effort.
		"""
		self._killed = True
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

	def _remember_state(self, state: dict) -> dict:
		self._cached_state = dict(state)
		path = state.get("sessionFile")
		if isinstance(path, str) and path:
			self.session_file = path
		identity = state.get("sessionId")
		if isinstance(identity, str) and identity:
			self.session_id = identity
		return state

	def state(self) -> dict:
		if self._dormant:
			return {**self._cached_state, "dormant": True}
		self._await_ready()
		return self._remember_state(self._sync.state())

	async def state_async(self) -> dict:
		if self._dormant:
			return {**self._cached_state, "dormant": True}
		await self._await_ready_async()
		return self._remember_state(await self._async.state())

	def activity(self) -> dict | None:
		"""Latest pi-activity API snapshot (cached while dormant)."""
		if self._dormant:
			return self._cached_activity
		self._await_ready()
		snap = self._sync.activity()
		if snap and snap.get("available"):
			self._absorb_activity(snap)
			return snap
		return None

	async def activity_async(self) -> dict | None:
		if self._dormant:
			return self._cached_activity
		await self._await_ready_async()
		snap = await self._async.activity()
		if snap and snap.get("available"):
			self._absorb_activity(snap)
			return snap
		return None

	def _absorb_activity(self, snap: dict) -> None:
		self._cached_activity = dict(snap)
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
		return not self._dormant and bool(self.window_id and window_alive(self.window_id))

	# -- waiting ------------------------------------------------------------

	def wait(self, timeout: float | None = None, poll: float = 1.0) -> Any:
		"""Block until the agent settles; returns AgentStrResponse/AgentDictResponse."""
		# Reject interruption before touching readiness, preserving startup errors.
		if self.closed and self.status != "settled" and self._startup_error is None:
			raise ValueError("await on closed handle (it was aborted; call resume() first)")
		self._await_ready()
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
		revision = self._settlement_revision
		try:
			settle = self._sync.wait_settled(
				timeout if timeout is not None else _settle_timeout_default(),
				poll=poll,
				on_tick=self._tick_sync,
				after_message=self._wait_baseline,
			)
		finally:
			self.awaited = False
			REGISTRY.emit()
		if revision != self._settlement_revision:
			raise PiSockError("settlement superseded by an accepted direct turn")
		return self._finish_wait(settle)

	async def wait_async(self, timeout: float | None = None, poll: float = 1.0) -> Any:
		# This guard must also work when a coroutine is advanced without a loop.
		if self.closed and self.status != "settled" and self._startup_error is None:
			raise ValueError("await on closed handle (it was aborted; call resume_async() first)")
		await self._await_ready_async()
		if self.closed and self.status != "settled":
			raise ValueError("await on closed handle (it was aborted; call resume_async() first)")
		if self.settled_data is not None:
			return await asyncio.to_thread(self._response)
		if self.socket_path and not os.path.exists(self.socket_path):
			# pi may still be booting — the socket appears when the session
			# starts. wait_settled treats missing sockets as dead, so wait for
			# the socket to appear first.
			if not await self._await_socket_async(timeout):
				self._mark_dead()
				raise TimeoutError(f"subagent {self.name}: pi-sock socket never appeared")
		self.awaited = True
		REGISTRY.emit()
		revision = self._settlement_revision
		try:
			settle = await self._async.wait_settled(
				timeout if timeout is not None else _settle_timeout_default(),
				poll=poll,
				on_tick=self._tick_async,
				after_message=self._wait_baseline,
			)
		except PiSockUnavailable:
			self._mark_dead()
			raise
		finally:
			self.awaited = False
			REGISTRY.emit()
		if revision != self._settlement_revision:
			raise PiSockError("settlement superseded by an accepted direct turn")
		# Validation/repair and disk inspection use synchronous clients; keep them
		# off the event loop just like scheduled send/resume control.
		return await asyncio.to_thread(self._finish_wait, settle)

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

	async def _await_socket_async(self, timeout: float | None = None) -> bool:
		deadline = time.monotonic() + (timeout or 60.0)
		while time.monotonic() < deadline:
			if self.socket_path and os.path.exists(self.socket_path):
				try:
					await self._async.state()
					return True
				except Exception:
					pass
			if not await asyncio.to_thread(self.is_window_alive):
				return False
			await asyncio.sleep(0.25)
		return False

	def _absorb_from_state(self, state: dict | None = None) -> None:
		if state is not None and "sessionFile" in state:
			self._remember_state(state)
		self._absorb_execution_state(state)
		self._absorb_ctx(state)
		try:
			self.activity()
		except Exception:
			pass

	async def _absorb_from_state_async(self, state: dict | None = None) -> None:
		if state is not None and "sessionFile" in state:
			self._remember_state(state)
		self._absorb_execution_state(state)
		self._absorb_ctx(state)
		try:
			await self.activity_async()
		except Exception:
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
		self._wait_baseline = None
		# Validation and bounded repair follow-ups are part of the active turn.
		# Do not expose a terminal state (or freeze its runtime) prematurely.
		self.status = "running"
		REGISTRY.emit()
		try:
			response = self._response()
			raise_for_failed_outcome(
				self.session.session_file,
				last_message=(self.settled_data or {}).get("lastAssistant"),
			)
			if self.closed:
				raise PiSockSessionEnded(f"subagent {self.name}: interrupted during validation")
			outcome = _last_assistant_outcome(self.session.session_file)
			last_message = (self.settled_data or {}).get("lastAssistant") or {}
			stop = outcome[0] if outcome is not None else last_message.get("stopReason")
			self.last_outcome = "ok" if stop in ("stop", "length") else "unknown"
			self.status = "settled"
			self._stamp_runtime()
			REGISTRY.emit()
			return response
		except Exception:
			self.last_outcome = "failed"
			self.status = "failed"
			self._stamp_runtime()
			REGISTRY.emit()
			raise

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


def _raise_for_failed_schema_turn(handle: AgentHandle, session, *, check_outcome: bool = True) -> None:
	# abort() closes the handle before the aborted session entry is necessarily
	# visible. Never revive that run by sending an automatic repair prompt.
	if getattr(handle, "closed", False):
		raise PiSockSessionEnded(f"subagent {handle.name}: interrupted or closed; schema repair stopped")
	if check_outcome:
		raise_for_failed_outcome(
			getattr(session, "session_file", None),
			last_message=(handle.settled_data or {}).get("lastAssistant"),
		)


def _dict_response(handle: AgentHandle, text: str, session) -> "AgentDictResponse":
	retries = _schema_retries()
	for attempt in range(retries + 1):
		_raise_for_failed_schema_turn(handle, session)
		parsed, error = _parse_json(text)
		if error is None:
			try:
				validate_with_schema(handle.schema, parsed)
				return AgentDictResponse(parsed, session)
			except SchemaValidationError as invalid:
				error = str(invalid)
		if attempt == retries:
			raise SchemaValidationError(
				f"subagent {handle.name}: response invalid after {attempt + 1} attempts: {error}"
			)
		# Require a newer assistant message; an idle socket may still expose the
		# rejected reply briefly after the repair prompt has been accepted.
		baseline = (handle.settled_data or {}).get("lastAssistant")
		_raise_for_failed_schema_turn(handle, session)
		# Real handles correlate both the durable outcome and the wire reply;
		# a repair can finish at the provider without producing new text.
		capture = getattr(handle, "_capture_outcome_baseline", None)
		if capture is not None:
			capture()
		handle.send(_schema_retry_prompt(handle.schema, error), mode="follow_up")
		settle = handle._sync.wait_settled(
			_settle_timeout_default(), after_message=baseline,
			on_tick=lambda state: _raise_for_failed_schema_turn(
				handle, session,
				# pi can retry provider failures internally while still running.
				check_outcome=bool(state.get("isIdle") and not state.get("hasPendingMessages")),
			),
		)
		if settle is None:
			raise PiSubagentsTimeoutError(f"subagent {handle.name}: schema repair timed out")
		handle.settled_data = settle
		last = settle.get("lastAssistant") or {}
		text = last.get("content") or ""
		session.invalidate()
	raise AssertionError("unreachable response validation state")


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
	"""At most three repair follow-ups, in addition to the initial response."""
	try:
		return max(0, min(3, int(os.environ.get("PI_SUBAGENTS_SCHEMA_RETRIES", "3"))))
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
	agentDir: str | os.PathLike[str] | None = None,
	group: str | None = None,
	session_name: str | None = None,
	register: bool = True,
) -> AgentHandle:
	"""Spawn one live pi session for the pool scheduler."""
	depth = int(os.environ.get("PI_SUBAGENT_DEPTH", "0") or 0) + 1
	handle = AgentHandle(
		prompt,
		name=name or "subagent",
		cwd=cwd or os.getcwd(),
		window_name=window_name or name or "subagent",
		model=model,
		thinking=thinking,
		schema=schema,
	)
	handle._agent_dir = str(resolve_agent_dir(agentDir))
	handle._depth = depth
	window_ref = spawn_pi_window(
		prompt,
		name=handle.name,
		cwd=handle.cwd,
		window_name=handle.window_name,
		model=handle.model,
		thinking=handle.thinking,
		socket_name=handle.id,
		depth=depth,
		agentDir=handle._agent_dir,
		session_name=session_name,
	)
	handle.group = group
	handle._bind(window_ref)
	if register:
		REGISTRY.register(handle)
	# Deliver the initial prompt over pi-sock once the instance is ready.
	handle._startup_started = True
	threading.Thread(target=handle._run_startup, name=f"subagent-startup-{handle.id}", daemon=True).start()
	REGISTRY.emit()
	return handle

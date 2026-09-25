"""Pool-first subagent workflow scheduler."""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

from .envcheck import ensure_environment
from .errors import AgentPoolTimeoutError, PoolClosedError, SessionReuseError
from .registry import REGISTRY, current_exec_scope
from .result import AgentResult
from .task import Task
from .tmuxenv import max_concurrent

if TYPE_CHECKING:  # pragma: no cover
	from .handle import AgentHandle as _LiveSession


class _GlobalCapacity:
	"""Process-wide active-turn ceiling shared by every pool."""

	def __init__(self) -> None:
		self._condition = threading.Condition()
		self._active = 0

	def acquire(self, pool: "AgentPool") -> bool:
		with self._condition:
			while self._active >= max_concurrent():
				if pool.closed:
					return False
				self._condition.wait(0.25)
			self._active += 1
			return True

	def release(self) -> None:
		with self._condition:
			self._active = max(0, self._active - 1)
			self._condition.notify_all()

	def wake(self) -> None:
		with self._condition:
			self._condition.notify_all()


_GLOBAL_CAPACITY = _GlobalCapacity()


@dataclass(slots=True)
class _Job:
	sequence: int
	task: Task
	stage: "AgentStage"
	handle: "AgentHandle"
	parent: AgentResult | None
	session_handle: "AgentHandle | None"
	session_name: str | None
	dispatched: bool = False


class AgentHandle:
	"""Awaitable handle for one queued or running task submission.

	A follow-up submission receives a new handle even when it reuses the same pi
	session. Previous handles and results therefore remain immutable.
	"""

	def __init__(self, pool: "AgentPool", stage: "AgentStage", task: Task):
		self.id = uuid.uuid4().hex
		self.pool = pool
		self.stage = stage
		self.task = task
		self.name = task.name or f"{stage.name}-{self.id[:6]}"
		self.exec_scope = current_exec_scope()
		self.started_at: float | None = None
		self.completed_at: float | None = None
		self.runtime_ms: int | None = None
		self._status = "queued"
		self._future: concurrent.futures.Future[AgentResult] = concurrent.futures.Future()
		self._live: _LiveSession | None = None
		self._session_config: dict[str, Any] | None = None
		self._invalidated = False
		self._awaited = False
		self._cancel_requested = False

	@property
	def status(self) -> str:
		return "closed" if self._invalidated else self._status

	@property
	def closed(self) -> bool:
		return self._invalidated or self.pool.closed

	@property
	def result(self) -> AgentResult | None:
		if not self._future.done() or self._future.cancelled():
			return None
		try:
			return self._future.result()
		except Exception:
			return None

	def wait(self, timeout: float | None = None) -> AgentResult:
		self._ensure_valid()
		self._awaited = True
		REGISTRY.emit()
		try:
			result = self._future.result(timeout=timeout)
		finally:
			self._awaited = False
			REGISTRY.emit()
		self._ensure_valid()
		return result

	async def wait_async(self, timeout: float | None = None) -> AgentResult:
		self._ensure_valid()
		self._awaited = True
		REGISTRY.emit()
		wrapped = asyncio.wrap_future(self._future)
		try:
			if timeout is None:
				result = await asyncio.shield(wrapped)
			else:
				result = await asyncio.wait_for(asyncio.shield(wrapped), timeout)
		finally:
			self._awaited = False
			REGISTRY.emit()
		self._ensure_valid()
		return result

	def __await__(self):
		return self.wait_async().__await__()

	def send(self, text: str, mode: str = "steer") -> dict:
		"""Steer the currently running turn; follow-ups must use stage.submit()."""
		live = self._require_live("send")
		if self._status not in ("starting", "running"):
			raise ValueError("send is only valid while this task is running")
		return live.send(text, mode=mode)

	async def send_async(self, text: str, mode: str = "steer") -> dict:
		live = self._require_live("send")
		if self._status not in ("starting", "running"):
			raise ValueError("send is only valid while this task is running")
		return await live.send_async(text, mode=mode)

	def cancel(self) -> bool:
		"""Cancel queued work or abort the currently running turn."""
		return self.pool._cancel(self)

	abort = cancel

	def state(self) -> dict:
		if self._live is None:
			return {"status": self.status, "queued": self._status == "queued"}
		return self._require_live("state").state()

	async def state_async(self) -> dict:
		if self._live is None:
			return {"status": self.status, "queued": self._status == "queued"}
		return await self._require_live("state").state_async()

	def activity(self) -> dict | None:
		return self._require_live("activity").activity()

	async def activity_async(self) -> dict | None:
		return await self._require_live("activity").activity_async()

	def get_session(self):
		result = self.result
		return result.session if result is not None else None

	def agent_state(self) -> dict:
		now = time.time() * 1000
		elapsed = self.runtime_ms
		if elapsed is None and self.started_at is not None:
			elapsed = max(0, now - self.started_at)
		base = {
			"id": self.id,
			"name": self.name,
			"group": self.stage.name,
			"status": self.status,
			"execScope": self.exec_scope,
			"startedAt": self.started_at or self.pool.started_at,
			"elapsedMs": round(elapsed or 0),
			"socketPath": None,
			"windowId": None,
			"toolCalls": 0,
			"thinkingMs": 0,
			"phase": None,
			"label": "waiting for a pool slot" if self._status == "queued" else None,
			"labelElapsedMs": None,
			"labelCalls": None,
			"liveTool": None,
			"awaited": self._awaited,
			"ctx": None,
			"depth": None,
		}
		if self._live is not None:
			live = self._live.agent_state()
			for key in (
				"socketPath", "windowId", "toolCalls", "thinkingMs", "phase", "label",
				"labelElapsedMs", "labelCalls", "liveTool", "ctx", "depth",
			):
				base[key] = live.get(key)
		return base

	def _bind_live(self, live: "_LiveSession", config: dict[str, Any]) -> None:
		self._live = live
		self._session_config = config

	def _ensure_valid(self) -> None:
		if self.closed:
			raise PoolClosedError(f"handle {self.name!r} was invalidated when its pool closed")

	def _require_live(self, operation: str) -> "_LiveSession":
		self._ensure_valid()
		if self._live is None:
			raise ValueError(f"cannot {operation}: task {self.name!r} has not been dispatched")
		return self._live

	def _invalidate(self) -> None:
		self._invalidated = True
		if not self._future.done():
			self._future.set_exception(PoolClosedError(f"pool closed before {self.name!r} completed"))

	def __repr__(self) -> str:
		return f"<AgentHandle {self.name!r} status={self.status}>"


class AgentStage:
	"""Pool-owned FIFO queue with a soft priority-slot allocation."""

	def __init__(self, pool: "AgentPool", name: str, slots: int):
		self.pool = pool
		self.name = name
		self.id = uuid.uuid4().hex
		self.slots = slots
		self.created_at = time.time() * 1000
		self._queue: deque[_Job] = deque()
		self._running = 0
		self._submitted = 0
		self._settled = 0
		self._failed = 0
		self._cancelled = 0

	def submit(
		self,
		task: Task,
		*,
		parent: AgentResult | None = None,
		session_handle: AgentHandle | None = None,
		session_name: str | None = None,
	) -> AgentHandle:
		return self.pool._submit(
			self,
			task,
			parent=parent,
			session_handle=session_handle,
			session_name=session_name,
		)

	def submit_all(
		self,
		tasks: Iterable[Task],
		*,
		parent: AgentResult | None = None,
	) -> list[AgentHandle]:
		return [self.submit(task, parent=parent) for task in tasks]

	def snapshot(self) -> dict:
		return {
			"id": self.id,
			"name": self.name,
			"slots": self.slots,
			"queued": len(self._queue),
			"running": self._running,
			"submitted": self._submitted,
			"settled": self._settled,
			"failed": self._failed,
			"cancelled": self._cancelled,
			"startedAt": self.created_at,
		}

	def __repr__(self) -> str:
		return f"<AgentStage {self.name!r} slots={self.slots}>"


@dataclass(frozen=True)
class PoolSummary:
	"""Report returned by pool.close() — auto-echoed as the workflow's final line.

	Wall time is what the user waited; agent time is what the fan-out consumed
	(Σ agent runtimes) — their ratio is the parallelism win.
	"""

	name: str
	submitted: int
	settled: int
	failed: int
	cancelled: int
	wall_ms: int
	agent_ms: int
	tool_calls: int
	stages: dict[str, dict[str, int]]
	failures: tuple[str, ...]

	def __str__(self) -> str:
		problems = self.failed + self.cancelled
		glyph = "✗" if problems else "✓"
		head = (
			f"PoolSummary {glyph} {self.name}: {self.submitted} tasks · "
			f"{self.settled} settled · {problems} stopped/failed · "
			f"wall {self.wall_ms / 1000:.1f}s · agent-time {self.agent_ms / 1000:.1f}s · "
			f"{self.tool_calls} tool calls"
		)
		stage_lines = [
			f"  {name}: {counts.get('settled', 0)}/{counts.get('submitted', 0)} settled"
			+ (f" · {counts.get('failed', 0)} failed" if counts.get("failed") else "")
			+ (f" · {counts.get('cancelled', 0)} cancelled" if counts.get("cancelled") else "")
			for name, counts in self.stages.items()
		]
		if self.failures:
			stage_lines.append("  failures: " + "; ".join(self.failures[:5]) + (" …" if len(self.failures) > 5 else ""))
		return "\n".join([head] + stage_lines)


class AgentPool:
	"""Autonomous bounded scheduler for multi-stage subagent workflows."""

	def __init__(self, concurrency: int | None = None, *, name: str | None = None):
		ensure_environment()
		limit = max_concurrent()
		self.concurrency = limit if concurrency is None else concurrency
		if isinstance(self.concurrency, bool) or not isinstance(self.concurrency, int):
			raise TypeError("AgentPool.concurrency must be an integer")
		if self.concurrency < 1:
			raise ValueError("AgentPool.concurrency must be at least 1")
		if self.concurrency > limit:
			raise ValueError(
				f"AgentPool.concurrency={self.concurrency} exceeds PI_SUBAGENTS_MAX_CONCURRENT={limit}"
			)
		self.id = uuid.uuid4().hex
		self.name = name or f"pool-{self.id[:6]}"
		self.started_at = time.time() * 1000
		self._condition = threading.Condition(threading.RLock())
		self._stages: list[AgentStage] = []
		self._stages_by_name: dict[str, AgentStage] = {}
		self._handles: dict[str, AgentHandle] = {}
		self._jobs_by_handle: dict[str, _Job] = {}
		self._results: deque[AgentResult] = deque()
		self._reserved_sessions: dict[int, str] = {}
		self._agent_ms = 0
		self._tool_calls = 0
		self._submit_sequence = 0
		self._result_sequence = 0
		self._running = 0
		self._closed = False
		self._pop_waiter: tuple[asyncio.AbstractEventLoop, asyncio.Future] | None = None
		self._workers = [
			threading.Thread(target=self._worker, name=f"subagent-pool-{self.id[:6]}-{i}", daemon=True)
			for i in range(self.concurrency)
		]
		REGISTRY.register_pool(self)
		for worker in self._workers:
			worker.start()
		REGISTRY.emit()

	@property
	def closed(self) -> bool:
		return self._closed

	def stage(self, name: str, *, slots: int) -> AgentStage:
		if not isinstance(name, str) or not name.strip():
			raise ValueError("stage name must be a non-empty string")
		if isinstance(slots, bool) or not isinstance(slots, int) or slots < 0:
			raise ValueError("stage slots must be a non-negative integer")
		with self._condition:
			self._ensure_open()
			if name in self._stages_by_name:
				raise ValueError(f"stage {name!r} already exists")
			if sum(stage.slots for stage in self._stages) + slots > self.concurrency:
				raise ValueError("sum of stage slots cannot exceed pool concurrency")
			stage = AgentStage(self, name, slots)
			self._stages.append(stage)
			self._stages_by_name[name] = stage
			self._condition.notify_all()
		REGISTRY.emit()
		return stage

	async def pop(self, timeout: float | None = None) -> AgentResult | None:
		"""Return the next completion, None on quiescence, or raise on timeout."""
		if timeout is not None and timeout <= 0:
			raise ValueError("pop timeout must be positive or None")
		loop = asyncio.get_running_loop()
		started = time.monotonic()
		while True:
			with self._condition:
				self._ensure_open()
				self._reattach_exec_scope_locked()
				if self._results:
					result = self._results.popleft()
					waiter = None
				elif self._is_quiescent_locked():
					return None
				else:
					if self._pop_waiter is not None:
						stale_loop, stale_future = self._pop_waiter
						# A chunk interrupted between turns can leave a waiter whose
						# loop is gone; such a waiter can never be woken again.
						if stale_future.done() or stale_loop.is_closed():
							self._pop_waiter = None
					if self._pop_waiter is not None:
						raise RuntimeError("AgentPool supports only one concurrent pop() consumer")
					result = None
					waiter = loop.create_future()
					self._pop_waiter = (loop, waiter)
			if waiter is None:
				REGISTRY.emit()
				return result
			remaining = None if timeout is None else timeout - (time.monotonic() - started)
			if remaining is not None and remaining <= 0:
				with self._condition:
					if self._pop_waiter and self._pop_waiter[1] is waiter:
						self._pop_waiter = None
				raise self._timeout_error(timeout)
			try:
				if remaining is None:
					await asyncio.shield(waiter)
				else:
					await asyncio.wait_for(asyncio.shield(waiter), remaining)
			except asyncio.TimeoutError as error:
				with self._condition:
					if self._pop_waiter and self._pop_waiter[1] is waiter:
						self._pop_waiter = None
				raise self._timeout_error(timeout) from error
			finally:
				with self._condition:
					if self._pop_waiter and self._pop_waiter[1] is waiter:
						self._pop_waiter = None

	def handles(self, status: set[str] | None = None) -> list[AgentHandle]:
		with self._condition:
			values = list(self._handles.values())
		if status is None:
			return values
		return [handle for handle in values if handle.status in status]

	def snapshot(self) -> dict:
		with self._condition:
			stages = [stage.snapshot() for stage in self._stages]
			return {
				"id": self.id,
				"name": self.name,
				"status": "closed" if self._closed else "open",
				"concurrency": self.concurrency,
				"running": self._running,
				"queued": sum(len(stage._queue) for stage in self._stages),
				"results": len(self._results),
				"stages": stages,
				"startedAt": self.started_at,
			}

	def close(self) -> PoolSummary:
		"""Invalidate handles, stop work, destroy every retained window, and close.

		Returns the workflow report (PoolSummary) — echoed when close() is the
		cell's last line, so ending a fan-out produces its own stats.
		"""
		with self._condition:
			if self._closed:
				return PoolSummary(
					name=self.name,
					submitted=self._submit_sequence,
					settled=0,
					failed=0,
					cancelled=0,
					wall_ms=max(0, round(time.time() * 1000 - self.started_at)),
					agent_ms=self._agent_ms,
					tool_calls=self._tool_calls,
					stages={},
					failures=(),
				)
			self._closed = True
			handles = list(self._handles.values())
			live_sessions = {id(handle._live): handle._live for handle in handles if handle._live}
			stages_summary = {
				stage.name: {
					"submitted": stage._submitted,
					"settled": stage._settled,
					"failed": stage._failed,
					"cancelled": stage._cancelled,
				}
				for stage in self._stages
			}
			failures = tuple(
				f"{result.task.name or result.handle.name} ({type(result.error).__name__ if result.error else result.status})"
				for result in self._results
				if result.status != "settled"
			)
			for stage in self._stages:
				stage._queue.clear()
			self._results.clear()
			for handle in handles:
				handle._invalidate()
			self._wake_pop_locked()
			self._condition.notify_all()
		_GLOBAL_CAPACITY.wake()
		for live in live_sessions.values():
			try:
				live.kill()
			except Exception:
				pass
		for worker in self._workers:
			if worker is not threading.current_thread():
				worker.join(timeout=2.0)
		# Handles stay registered as "closed": the completed chunk's snapshot keeps
		# the workflow visible (notification + panel), and later execs drop these
		# rows naturally via exec-scope filtering.
		REGISTRY.unregister_pool(self.id)
		REGISTRY.emit()

		return PoolSummary(
			name=self.name,
			submitted=self._submit_sequence,
			settled=sum(stage._settled for stage in self._stages),
			failed=sum(stage._failed for stage in self._stages),
			cancelled=sum(stage._cancelled for stage in self._stages),
			wall_ms=max(0, round(time.time() * 1000 - self.started_at)),
			agent_ms=self._agent_ms,
			tool_calls=self._tool_calls,
			stages=stages_summary,
			failures=failures,
		)

	def __enter__(self) -> "AgentPool":
		return self

	def __exit__(self, exc_type, exc, tb) -> None:
		self.close()

	def _submit(
		self,
		stage: AgentStage,
		task: Task,
		*,
		parent: AgentResult | None,
		session_handle: AgentHandle | None,
		session_name: str | None,
	) -> AgentHandle:
		if not isinstance(task, Task):
			raise TypeError("stage.submit() requires a Task")
		if parent is not None:
			if parent.handle.pool is not self:
				raise ValueError("parent result belongs to another pool")
			task = task.with_parent_metadata(parent.task.metadata)
		with self._condition:
			self._ensure_open()
			if stage.pool is not self:
				raise ValueError("stage belongs to another pool")
			if session_handle is not None:
				self._reserve_session_locked(session_handle, task)
			self._submit_sequence += 1
			handle = AgentHandle(self, stage, task)
			job = _Job(
				self._submit_sequence,
				task,
				stage,
				handle,
				parent,
				session_handle,
				session_name,
			)
			self._handles[handle.id] = handle
			self._jobs_by_handle[handle.id] = job
			stage._queue.append(job)
			stage._submitted += 1
			REGISTRY.register(handle)
			self._condition.notify_all()
		REGISTRY.emit()
		return handle

	def _reserve_session_locked(self, source: AgentHandle, task: Task) -> None:
		if source.pool is not self:
			raise SessionReuseError("session handle belongs to another pool")
		if source.closed or source._live is None:
			raise SessionReuseError("session handle is invalid or has not started")
		if source.status != "settled":
			raise SessionReuseError("session reuse requires a settled handle")
		key = id(source._live)
		if key in self._reserved_sessions:
			raise SessionReuseError("this session already has a queued or running follow-up")
		config = source._session_config or {}
		checks = {
			"model": task.model,
			"thinking": task.thinking,
			"cwd": os.path.abspath(task.cwd) if task.cwd else None,
			"profile": os.path.abspath(task.profile) if task.profile else None,
		}
		for field, requested in checks.items():
			if requested is not None and requested != config.get(field):
				raise SessionReuseError(
					f"follow-up {field}={requested!r} conflicts with session {field}={config.get(field)!r}"
				)
		self._reserved_sessions[key] = "reserved"

	def _cancel(self, handle: AgentHandle) -> bool:
		with self._condition:
			self._ensure_open()
			if handle.pool is not self:
				return False
			job = self._jobs_by_handle.get(handle.id)
			if job is None or handle._future.done():
				return False
			if handle._status == "queued":
				try:
					job.stage._queue.remove(job)
				except ValueError:
					return False
				handle._cancel_requested = True
				self._finish_job_locked(job, body=None, error=None, status="cancelled")
				self._condition.notify_all()
				wake = True
			else:
				handle._cancel_requested = True
				live = handle._live
				wake = False
		if wake:
			REGISTRY.emit()
			return True
		if live is not None:
			try:
				live.abort()
			except Exception:
				pass
		return True

	def _worker(self) -> None:
		while True:
			job = self._take_job()
			if job is None:
				return
			if not _GLOBAL_CAPACITY.acquire(self):
				return
			try:
				self._execute(job)
			finally:
				_GLOBAL_CAPACITY.release()

	def _take_job(self) -> _Job | None:
		with self._condition:
			while True:
				if self._closed:
					return None
				stage = self._select_stage_locked()
				if stage is not None:
					job = stage._queue.popleft()
					job.dispatched = True
					stage._running += 1
					self._running += 1
					job.handle._status = "starting"
					job.handle.started_at = time.time() * 1000
					return job
				self._condition.wait()

	def _select_stage_locked(self) -> AgentStage | None:
		available = [stage for stage in self._stages if stage._queue]
		if not available:
			return None
		under = [stage for stage in available if stage._running < stage.slots]
		candidates = under or available
		return min(candidates, key=lambda stage: stage._queue[0].sequence)

	def _execute(self, job: _Job) -> None:
		handle = job.handle
		try:
			if job.session_handle is None:
				live, config = self._spawn_new(job)
			else:
				live, config = self._reuse_session(job)
			with self._condition:
				closed_during_startup = self._closed
			if closed_during_startup:
				# close() ran while this spawn was in flight; the window exists
				# now, so destroy it here instead of leaking it.
				try:
					live.kill()
				except Exception:
					pass
				return
			handle._bind_live(live, config)
			handle._status = "running"
			REGISTRY.emit()
			body = live.wait(timeout=job.task.timeout)
			status = "cancelled" if handle._cancel_requested else "settled"
			error = None
		except BaseException as caught:
			body = None
			error = caught
			status = "cancelled" if handle._cancel_requested else "failed"
			if handle._live is not None and isinstance(caught, TimeoutError):
				try:
					handle._live.abort()
				except Exception:
					pass
		with self._condition:
			if not self._closed:
				self._finish_job_locked(job, body=body, error=error, status=status)
			elif handle._live is not None:
				try:
					handle._live.kill()
				except Exception:
					pass
			self._condition.notify_all()
		REGISTRY.emit()

	def _spawn_new(self, job: _Job) -> tuple["_LiveSession", dict[str, Any]]:
		from .handle import spawn_pi_window_handle

		task = job.task
		cwd = os.path.abspath(task.cwd or os.getcwd())
		profile = os.path.abspath(task.profile) if task.profile else None
		live = spawn_pi_window_handle(
			task.prompt,
			name=job.handle.name,
			cwd=cwd,
			window_name=job.handle.name,
			model=task.model,
			thinking=task.thinking,
			schema=task.schema,
			profile=profile,
			group=job.stage.name,
			session_name=job.session_name,
			register=False,
		)
		config = {
			"model": task.model,
			"thinking": task.thinking,
			"cwd": cwd,
			"profile": profile,
		}
		return live, config

	def _reuse_session(self, job: _Job) -> tuple["_LiveSession", dict[str, Any]]:
		source = job.session_handle
		assert source is not None and source._live is not None
		live = source._live
		if job.session_name is not None:
			live.set_session_name(job.session_name)
		live.schema = job.task.schema
		live.resume(job.task.prompt)
		return live, dict(source._session_config or {})

	def _finish_job_locked(self, job: _Job, *, body, error, status: str) -> AgentResult:
		now = time.time() * 1000
		handle = job.handle
		handle.completed_at = now
		handle.runtime_ms = round(now - handle.started_at) if handle.started_at else None
		handle._status = status
		if job.dispatched:
			job.stage._running = max(0, job.stage._running - 1)
			self._running = max(0, self._running - 1)
		if status == "settled":
			job.stage._settled += 1
		elif status == "cancelled":
			job.stage._cancelled += 1
		else:
			job.stage._failed += 1
		self._result_sequence += 1
		result = AgentResult(
			sequence=self._result_sequence,
			task=job.task,
			stage=job.stage,
			handle=handle,
			body=body,
			error=error,
			status=status,
			started_at=handle.started_at,
			completed_at=now,
			duration_ms=handle.runtime_ms,
			parent=job.parent,
		)
		self._results.append(result)
		self._jobs_by_handle.pop(handle.id, None)
		if handle._live is not None:
			self._reserved_sessions.pop(id(handle._live), None)
			try:
				self._tool_calls += int(getattr(handle._live, "tool_calls", 0) or 0)
			except Exception:
				pass
		if handle.runtime_ms:
			self._agent_ms += handle.runtime_ms
		if job.session_handle is not None and job.session_handle._live is not None:
			self._reserved_sessions.pop(id(job.session_handle._live), None)
		if not handle._future.done():
			handle._future.set_result(result)
		self._wake_pop_locked()
		return result

	def _is_quiescent_locked(self) -> bool:
		return self._running == 0 and not any(stage._queue for stage in self._stages) and not self._results

	def _wake_pop_locked(self) -> None:
		waiter = self._pop_waiter
		if waiter is None:
			return
		loop, future = waiter
		if future.done():
			return
		try:
			loop.call_soon_threadsafe(_resolve_waiter, future)
		except RuntimeError:
			pass

	def _timeout_error(self, timeout: float | None) -> AgentPoolTimeoutError:
		return AgentPoolTimeoutError(
			f"pool {self.name!r} produced no result within {timeout:g}s",
			pool=self,
			snapshot=self.snapshot(),
		)

	def _reattach_exec_scope_locked(self) -> None:
		scope = current_exec_scope()
		if scope:
			for handle in self._handles.values():
				if handle.status in ("queued", "starting", "running"):
					handle.exec_scope = scope

	def _ensure_open(self) -> None:
		if self._closed:
			raise PoolClosedError(f"pool {self.name!r} is closed")

	def __repr__(self) -> str:
		return f"<AgentPool {self.name!r} concurrency={self.concurrency} status={'closed' if self.closed else 'open'}>"


def _resolve_waiter(future: asyncio.Future) -> None:
	if not future.done():
		future.set_result(None)

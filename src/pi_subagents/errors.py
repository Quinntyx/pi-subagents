"""Exception types for pi_subagents (kept separate to avoid import cycles)."""

from __future__ import annotations

from typing import Any


class PiSubagentsError(Exception):
	"""Base class for pi-subagents failures."""


class PiSubagentsTimeoutError(PiSubagentsError, TimeoutError):
	"""An agent turn did not settle within its allotted timeout."""


class AgentPoolTimeoutError(PiSubagentsError, TimeoutError):
	"""No task completed before an AgentPool.pop() deadline.

	The pool remains open and all work continues. ``snapshot`` is captured at the
	time of the timeout so callers can report or inspect the stalled workflow.
	"""

	def __init__(self, message: str, *, pool: Any, snapshot: dict):
		super().__init__(message)
		self.pool = pool
		self.snapshot = snapshot

	@property
	def active_handles(self) -> tuple:
		return tuple(self.pool.handles(status={"queued", "starting", "running"}))


class PoolClosedError(PiSubagentsError):
	"""Operation attempted through a closed pool or invalidated handle."""


class SessionReuseError(PiSubagentsError):
	"""A submitted task cannot reuse the requested retained pi session."""

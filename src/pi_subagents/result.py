"""Terminal outcome values produced by an AgentPool."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # pragma: no cover
	from .pool import AgentHandle, AgentStage
	from .response import AgentDictResponse, AgentStrResponse
	from .task import Task

ResultStatus = Literal["settled", "failed", "cancelled"]


@dataclass(frozen=True, slots=True)
class AgentResult:
	"""One terminal task outcome, emitted in pool-observed completion order."""

	sequence: int
	task: "Task"
	stage: "AgentStage"
	handle: "AgentHandle"
	body: "AgentStrResponse | AgentDictResponse | None"
	error: BaseException | None
	status: ResultStatus
	started_at: float | None
	completed_at: float
	duration_ms: int | None
	parent: "AgentResult | None" = None

	@property
	def session(self):
		"""Parsed persisted session transcript, when a response exists."""
		return self.body.get_session() if self.body is not None else None

	def __str__(self) -> str:
		"""Readable form for kernel auto-echo (Out[n]); repr stays constructor-shaped.

		The actual body is shown, subject to the caller's output truncation —
		this is not a summary preview.
		"""
		glyph = {"settled": "✓", "failed": "✗", "cancelled": "■"}.get(self.status, "●")
		head = f"{glyph} {self.task.name or self.handle.id} ({self.stage.name})"
		if self.error is not None:
			return f"{head} · {type(self.error).__name__}: {self.error}"
		if self.body is None:
			return f"{head} · no response"
		seconds = (self.duration_ms or 0) / 1000
		return f"{head} · {seconds:.1f}s\n{self.body}"

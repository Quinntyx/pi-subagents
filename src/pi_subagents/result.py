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
	def ok(self) -> bool:
		return self.status == "settled" and self.error is None

	def unwrap(self) -> Any:
		"""Return the response body or raise the task's terminal error."""
		if self.error is not None:
			raise self.error
		if self.body is None:
			raise RuntimeError(f"task {self.task.name or self.handle.id} produced no response")
		return self.body

	@property
	def session(self):
		"""Parsed persisted session transcript, when a response exists."""
		return self.body.get_session() if self.body is not None else None

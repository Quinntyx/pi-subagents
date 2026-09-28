"""Immutable task specifications submitted to an AgentStage."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class Task:
	"""Description of one scheduled subagent turn.

	Metadata is copied and exposed read-only. Every task has a non-negative integer
	``rounds`` value; roots default to zero. AgentStage.submit() merges parent
	metadata before the task's explicitly supplied metadata.
	"""

	prompt: str
	name: str | None = None
	model: str | None = None
	thinking: str | None = None
	schema: dict | None = None
	cwd: str | os.PathLike[str] | None = None
	# Agent dir for spawned instances: bare name under the pi-profiles root,
	# or a path used verbatim. None → PI_CODING_SUBAGENT_DIR or the
	# orchestrator's own agent dir.
	agentDir: str | os.PathLike[str] | None = None
	timeout: float | None = None
	metadata: Mapping[str, Any] = field(default_factory=dict)
	_explicit_metadata: frozenset[str] = field(init=False, repr=False, compare=False)

	def __post_init__(self) -> None:
		if not isinstance(self.prompt, str) or not self.prompt.strip():
			raise ValueError("Task.prompt must be a non-empty string")
		if self.name is not None and (not isinstance(self.name, str) or not self.name.strip()):
			raise ValueError("Task.name must be a non-empty string when provided")
		if self.timeout is not None and self.timeout <= 0:
			raise ValueError("Task.timeout must be positive or None")
		metadata = dict(self.metadata)
		explicit = frozenset(metadata)
		metadata.setdefault("rounds", 0)
		rounds = metadata["rounds"]
		if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 0:
			raise ValueError("Task.metadata['rounds'] must be a non-negative integer")
		object.__setattr__(self, "metadata", MappingProxyType(metadata))
		object.__setattr__(self, "_explicit_metadata", explicit)
		if self.cwd is not None:
			object.__setattr__(self, "cwd", os.fspath(self.cwd))
		if self.agentDir is not None:
			object.__setattr__(self, "agentDir", os.fspath(self.agentDir))

	def with_parent_metadata(self, parent: Mapping[str, Any]) -> "Task":
		"""Return a task whose metadata inherits from a parent result.

		Only keys explicitly supplied to this task override the parent. The automatic
		root ``rounds=0`` default therefore does not reset a parent's round counter.
		"""
		merged = dict(parent)
		for key in self._explicit_metadata:
			merged[key] = self.metadata[key]
		return Task(
			self.prompt,
			name=self.name,
			model=self.model,
			thinking=self.thinking,
			schema=self.schema,
			cwd=self.cwd,
			agentDir=self.agentDir,
			timeout=self.timeout,
			metadata=merged,
		)

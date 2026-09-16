"""Registry of spawned subagents + state emission.

Every mutation emits a compact snapshot. When running inside a PTC session the
runtime has injected ``builtins.PTC_STATE_EMIT`` and the snapshot rides the PTC
RPC pipe as a ``subagent_state`` frame (transparent UI); otherwise a one-line
status is printed to stdout so standalone tmux scripts still show progress.
"""

from __future__ import annotations

import builtins
import os
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
	from .handle import AgentHandle


def _ptc_bridge():
	bridge = getattr(builtins, "PTC_STATE_EMIT", None)
	return bridge if callable(bridge) else None


class Registry:
	def __init__(self) -> None:
		self._handles: dict[str, "AgentHandle"] = {}
		self._lock = threading.Lock()
		self._phase: str | None = None
		self._phase_started_at: float | None = None
		self._last_fingerprint: str | None = None

	def register(self, handle: "AgentHandle") -> None:
		with self._lock:
			self._handles[handle.id] = handle

	def handles(self) -> list["AgentHandle"]:
		with self._lock:
			return list(self._handles.values())

	def get(self, handle_id: str) -> "AgentHandle | None":
		with self._lock:
			return self._handles.get(handle_id)

	def set_phase(self, label: str | None) -> None:
		"""Label the current orchestration phase (viewer grouping)."""
		with self._lock:
			self._phase = label
			self._phase_started_at = time.time() * 1000 if label else None
		self.emit()

	def current_phase(self) -> str | None:
		with self._lock:
			return self._phase

	def snapshot(self) -> dict:
		with self._lock:
			agents = [handle.agent_state() for handle in self._handles.values()]
		running = sum(1 for a in agents if a["status"] in ("starting", "running"))
		settled = sum(1 for a in agents if a["status"] == "settled")
		failed = sum(1 for a in agents if a["status"] in ("failed", "dead", "stopped"))
		return {
			"pid": os.getpid(),
			"depth": _depth_int(),
			"agents": agents,
			"totals": {"running": running, "settled": settled, "failed": failed},
			"groups": {self._phase: self._phase_started_at} if self._phase else {},
			"timestamp": time.time() * 1000,
		}

	def emit(self, force: bool = False) -> None:
		snapshot = self.snapshot()
		fingerprint = repr(sorted((a["id"], a["status"], a["toolCalls"], a["thinkingMs"], a["label"], a["awaited"], a["ctx"], a["group"]) for a in snapshot["agents"]))
		if not force and fingerprint == self._last_fingerprint:
			return  # nothing changed; skip standalone print and bridge push
		self._last_fingerprint = fingerprint
		bridge = _ptc_bridge()
		if bridge is not None:
			try:
				bridge(snapshot)
				return
			except Exception:
				pass  # never let UI plumbing break agent control flow
		emit_status_line(snapshot)


def _depth_int() -> int:
	try:
		return int(os.environ.get("PI_SUBAGENT_DEPTH", "0") or 0)
	except ValueError:
		return 0


def emit_status_line(snapshot: dict) -> None:
	"""Compact human-readable one-liner for standalone runs."""
	totals = snapshot.get("totals", {})
	parts = []
	if totals.get("running"):
		parts.append(f"● {totals['running']} running")
	if totals.get("settled"):
		parts.append(f"✓ {totals['settled']} done")
	if totals.get("failed"):
		parts.append(f"! {totals['failed']} failed/stopped")
	if not parts:
		return
	details = []
	for agent in snapshot.get("agents", []):
		if agent["status"] in ("starting", "running"):
			detail = f"{agent['name']}: {agent.get('toolCalls') or 0} calls"
			label = agent.get("label") or agent.get("phase")
			if label:
				detail += f" · {label}"
			details.append(detail)
	line = "subagents: " + " · ".join(parts)
	if details:
		line += " — " + "; ".join(details)
	print(line, flush=True)


REGISTRY = Registry()

"""Registry of pool-owned handles plus PTC state emission."""

from __future__ import annotations

import builtins
import os
import threading
import time
from typing import Any


def current_exec_scope() -> str | None:
	"""Exec token of the PTC chunk that is currently running."""
	token = getattr(builtins, "PTC_EXEC_SCOPE", None)
	return token if isinstance(token, str) and token else None


def _ptc_bridge():
	bridge = getattr(builtins, "PTC_STATE_EMIT", None)
	return bridge if callable(bridge) else None


class Registry:
	def __init__(self) -> None:
		self._handles: dict[str, Any] = {}
		self._pools: dict[str, Any] = {}
		self._lock = threading.Lock()
		self._last_fingerprint: str | None = None

	def register(self, handle: Any) -> None:
		with self._lock:
			self._handles[handle.id] = handle

	def unregister(self, handle_id: str) -> None:
		with self._lock:
			self._handles.pop(handle_id, None)

	def handles(self) -> list[Any]:
		with self._lock:
			return list(self._handles.values())

	def get(self, handle_id: str):
		with self._lock:
			return self._handles.get(handle_id)

	def register_pool(self, pool: Any) -> None:
		with self._lock:
			self._pools[pool.id] = pool

	def unregister_pool(self, pool_id: str) -> None:
		with self._lock:
			self._pools.pop(pool_id, None)

	def pools(self) -> list[Any]:
		with self._lock:
			return list(self._pools.values())

	def snapshot(self) -> dict:
		# Never call user/pool methods while holding the registry lock. Pool workers
		# emit concurrently, and the inverse lock order would deadlock.
		with self._lock:
			handles = list(self._handles.values())
			pools = list(self._pools.values())
		agents = [handle.agent_state() for handle in handles]
		pool_states = [pool.snapshot() for pool in pools]
		running = sum(1 for agent in agents if agent["status"] in ("starting", "running"))
		queued = sum(1 for agent in agents if agent["status"] == "queued")
		settled = sum(1 for agent in agents if agent["status"] == "settled")
		failed = sum(1 for agent in agents if agent["status"] in ("failed", "dead", "stopped", "cancelled"))
		groups = {
			stage["name"]: stage.get("startedAt")
			for pool in pool_states
			for stage in pool.get("stages", [])
		}
		return {
			"pid": os.getpid(),
			"depth": _depth_int(),
			"agents": agents,
			"pools": pool_states,
			"totals": {
				"queued": queued,
				"running": running,
				"settled": settled,
				"failed": failed,
			},
			"groups": groups,
			"timestamp": time.time() * 1000,
		}

	def emit(self, force: bool = False) -> None:
		snapshot = self.snapshot()
		bridge = _ptc_bridge()
		if bridge is not None:
			try:
				bridge(snapshot)
				return
			except Exception:
				pass
		fingerprint = repr(
			(
				sorted(
					(
						agent["id"], agent["status"], agent["toolCalls"], agent["thinkingMs"],
						agent["label"], agent["awaited"], agent["ctx"], agent["group"],
					)
					for agent in snapshot["agents"]
				),
				[
					(
						pool["id"], pool["running"], pool["queued"], pool["results"],
						tuple((stage["id"], stage["queued"], stage["running"], stage["settled"], stage["failed"])
						      for stage in pool["stages"]),
					)
					for pool in snapshot["pools"]
				],
			)
		)
		if not force and fingerprint == self._last_fingerprint:
			return
		self._last_fingerprint = fingerprint
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
	if totals.get("queued"):
		parts.append(f"… {totals['queued']} queued")
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
		if agent["status"] in ("queued", "starting", "running"):
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

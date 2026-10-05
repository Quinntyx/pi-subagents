"""pi-subagents — pool-based orchestration of standalone pi instances.

Subagents are real, interactive pi processes running in their own tmux windows
(running under the orchestrator's agent dir or `PI_CODING_SUBAGENT_DIR`), controlled over the pi-sock unix
socket. The library works both inside a PTC session (state is forwarded to the
PTC runtime as `subagent_state` frames for live UI) and from plain scripts run
directly in a tmux pane (state is printed as one-line status updates).

The public orchestration API is the AgentPool:

    pool = subagents.AgentPool(concurrency=8)
    build = pool.stage("build", slots=4)
    review = pool.stage("review", slots=4)

    handle = build.submit(subagents.Task("Implement X", name="build-x"))
    result = await handle                    # or while r := await pool.pop(): ...

Environment contract (checked at import):

- Imports are legal at every depth. Spawning is flat by default; recursion is
  opt-in via ``PI_SUBAGENTS_MAX_DEPTH > 1`` and bounded by a shared root budget.
  ``PI_SUBAGENTS_ROOT_MAX_TASKS`` counts admitted window creations, including
  reopens; retained failure windows remain charged until confirmed cleanup.
- Not inside tmux (``$TMUX`` missing/unreachable) → the module imports with a
  warning; every API call raises ``NotImplementedError``.
"""

from .catalog import (
	best_model_match,
	capabilities,
	list_models,
	model_slugs,
	agent_dir_defaults,
	resolve_models,
	scoped_models,
	thinking_levels,
)
from .envcheck import ensure_environment
from .client import PiSockSessionEnded
from .errors import (
	AgentPoolFailureError,
	AgentPoolTimeoutError,
	PiSubagentsError,
	PiSubagentsTimeoutError,
	PoolClosedError,
	SessionReuseError,
)
from .tmuxenv import subagent_session_name
import os
from .registry import REGISTRY, emit_status_line
from .handle import AgentHandle as _LiveSessionHandle
from .pool import AgentHandle, AgentPool, AgentStage, PoolSummary
from .result import AgentResult
from .schema import SchemaValidationError
from .session_file import AgentSession
from .response import AgentDictResponse, AgentStrResponse
from .task import Task

__all__ = [
	"AgentPool",
	"AgentStage",
	"AgentHandle",
	"AgentResult",
	"PoolSummary",
	"Task",
	"capabilities",
	"list_models",
	"model_slugs",
	"resolve_models",
	"best_model_match",
	"thinking_levels",
	"agent_dir_defaults",
	"scoped_models",
	"AgentSession",
	"AgentStrResponse",
	"AgentDictResponse",
	"SchemaValidationError",
	"subagent_session_name",
	"AgentPoolFailureError",
	"AgentPoolTimeoutError",
	"PiSockSessionEnded",
	"PoolClosedError",
	"SessionReuseError",
	"PiSubagentsError",
	"PiSubagentsTimeoutError",
	"NotImplementedError",
]

__version__ = "0.2.0"


def _depth() -> int:
	from .recursion import current_depth

	return current_depth()


# Warn when tmux is absent; capture placement without imposing spawn policy.
ensure_environment()


def list_agents() -> list[dict]:
	"""Snapshot rows for every handle owned by live pools."""
	return REGISTRY.snapshot()["agents"]


def stop_all() -> None:
	"""Cancel every queued or running task in every live pool (pools stay open)."""
	for pool in REGISTRY.pools():
		if pool.closed:
			continue
		for handle in pool.handles(status={"queued", "starting", "running"}):
			try:
				handle.cancel()
			except Exception:
				pass


def finish(pools: list | None = None) -> int:
	"""Close pools permanently: kill their tmux windows and invalidate handles.

	With `pools=None` (the default), every live pool is closed. Returns the number
	of pools closed. Safe to call twice. Settled sessions a caller still wants to
	continue must be kept in an explicitly open pool.
	"""
	targets = pools if pools is not None else [p for p in REGISTRY.pools() if not p.closed]
	closed = 0
	for pool in targets:
		try:
			pool.close()
			closed += 1
		except Exception:
			pass
	return closed


def emit_status_line() -> None:  # re-exported for the runtime bridge tests
	REGISTRY.emit()

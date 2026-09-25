"""pi-subagents — pool-based orchestration of standalone pi instances.

Subagents are real, interactive pi processes running in their own tmux windows
(under the dedicated `subagents` pi profile), controlled over the pi-sock unix
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

- ``PI_SUBAGENT_DEPTH`` set → ``import pi_subagents`` raises ``NotImplementedError``.
  Spawned agents cannot spawn further agents; the orchestration tree stays height 2.
- Not inside tmux (``$TMUX`` missing/unreachable) → the module imports with a
  warning; every API call raises ``NotImplementedError``.
"""

from .catalog import (
	best_model_match,
	capabilities,
	list_models,
	model_slugs,
	profile_defaults,
	resolve_models,
	scoped_models,
	thinking_levels,
)
from .envcheck import ensure_environment
from .errors import (
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
	"profile_defaults",
	"scoped_models",
	"AgentSession",
	"AgentStrResponse",
	"AgentDictResponse",
	"SchemaValidationError",
	"subagent_session_name",
	"AgentPoolTimeoutError",
	"PoolClosedError",
	"SessionReuseError",
	"PiSubagentsError",
	"PiSubagentsTimeoutError",
	"NotImplementedError",
]

__version__ = "0.2.0"


def _depth() -> int:
	try:
		return int(__import__("os").environ.get("PI_SUBAGENT_DEPTH", "0"))
	except ValueError:
		return 0


# Runs at import: raises NotImplementedError when PI_SUBAGENT_DEPTH is set,
# warns when tmux is absent, and otherwise captures the tmux placement context.
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

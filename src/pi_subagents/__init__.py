"""pi-subagents — spawn, monitor, and steer standalone pi instances.

Subagents are real, interactive pi processes running in their own tmux windows
(under the dedicated `subagents` pi profile), controlled over the pi-sock unix
socket. The library works both inside a PTC session (state is forwarded to the
PTC runtime as `subagent_state` frames for live UI) and from plain scripts run
directly in a tmux pane (state is printed as one-line status updates).

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
from .registry import REGISTRY, emit_status_line
from .handle import AgentHandle
from .schema import SchemaValidationError
from .session_file import AgentSession
from .response import AgentDictResponse, AgentStrResponse

__all__ = [
	"agent",
	"capabilities",
	"list_models",
	"model_slugs",
	"resolve_models",
	"best_model_match",
	"thinking_levels",
	"profile_defaults",
	"scoped_models",
	"AgentHandle",
	"AgentSession",
	"AgentStrResponse",
	"AgentDictResponse",
	"SchemaValidationError",
	"list_agents",
	"phase",
	"wait_all",
	"wait_all_async",
	"stop_all",
	"PiSubagentsError",
	"PiSubagentsTimeoutError",
	"NotImplementedError",
]

__version__ = "0.1.0"


class PiSubagentsError(Exception):
	"""Base class for pi-subagents failures."""


class PiSubagentsTimeoutError(PiSubagentsError, TimeoutError):
	"""An agent did not settle within the allotted timeout."""


def _depth() -> int:
	try:
		return int(__import__("os").environ.get("PI_SUBAGENT_DEPTH", "0"))
	except ValueError:
		return 0


# Runs at import: raises NotImplementedError when PI_SUBAGENT_DEPTH is set,
# warns when tmux is absent, and otherwise captures the tmux placement context.
ensure_environment()


def agent(
	prompt: str,
	*,
	name: str | None = None,
	cwd: str | None = None,
	window_name: str | None = None,
	model: str | None = None,
	thinking: str | None = None,
	schema: dict | None = None,
) -> AgentHandle:
	"""Spawn a subagent pi in a new tmux window and return its handle immediately.

	The handle can be awaited (``resp = await handle``) to get an
	AgentStrResponse/AgentDictResponse, or driven interactively with
	``send``/``abort``/``resume`` while it runs.
	"""
	from .handle import spawn_pi_window_handle

	ensure_environment()
	handle = spawn_pi_window_handle(
		prompt,
		name=name,
		cwd=cwd,
		window_name=window_name,
		model=model,
		thinking=thinking,
		schema=schema,
	)
	REGISTRY.emit()
	return handle


# Alias for familiarity with Claude-style APIs.
spawn = agent


def phase(label: str | None) -> None:
	"""Label the current orchestration phase. Subagents spawned after this call
	are grouped under `label` in the PTC live viewer. Pass None to close the
	current phase (subsequent spawns get no group)."""
	from .handle import set_phase

	set_phase(label)


def list_agents() -> list[dict]:
	"""Snapshot of every subagent spawned by this process."""
	return REGISTRY.snapshot()["agents"]


def wait_all(handles: list[AgentHandle], timeout: float | None = None) -> list:
	"""Wait for every handle to settle; returns responses in input order.

	Sync context: blocking. Inside a running event loop use ``await wait_all_async``.
	"""
	if not handles:
		return []
	import asyncio

	try:
		asyncio.get_running_loop()
	except RuntimeError:
		return [h.wait(timeout=timeout) for h in handles]
	raise PiSubagentsError(
		"wait_all() called from a running event loop; use await wait_all_async() instead"
	)


async def wait_all_async(handles: list, timeout: float | None = None) -> list:
	"""Async variant of wait_all for use inside PTC sessions."""
	import asyncio

	if not handles:
		return []
	return list(await asyncio.gather(*[h.wait_async(timeout=timeout) for h in handles]))


def stop_all() -> None:
	"""Abort every subagent spawned by this process."""
	for handle in REGISTRY.handles():
		try:
			handle.abort()
		except Exception:
			pass


def emit_status_line() -> None:  # re-exported for the runtime bridge tests
	REGISTRY.emit()

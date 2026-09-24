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
from .errors import PiSubagentsError, PiSubagentsTimeoutError
from .tmuxenv import subagent_session_name
import os
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
	"subagent_session_name",
	"wait_all",
	"wait_all_async",
	"stop_all",
	"finish",
	"PiSubagentsError",
	"PiSubagentsTimeoutError",
	"NotImplementedError",
]

__version__ = "0.1.0"




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
	profile: str | os.PathLike[str] | None = None,
) -> AgentHandle:
	"""Spawn a subagent pi in a new tmux window and return its handle immediately.

	The handle can be awaited (``resp = await handle``) to get an
	AgentStrResponse/AgentDictResponse, or driven interactively with
	``send``/``abort``/``resume`` while it runs.

	``profile`` selects the pi profile the agent runs under: a profile name
	under ``~/.config/pi/profiles`` (e.g. ``"design-subagents"``), a path to a
	profile directory, or None for the default ``subagents`` profile. This is
	how special-purpose subagent profiles (extra extensions, system prompts,
	skills) are selected per spawn without env-var juggling.
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
		profile=profile,
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
	"""Abort every subagent spawned by this process (runs stop; windows stay)."""
	for handle in REGISTRY.handles():
		try:
			handle.abort()
		except Exception:
			pass


def finish(handles: list | None = None) -> int:
	"""Close subagents permanently: kill their tmux windows and clean their sockets.

	Call this once a fan-out's results are in and the agents do not need to be
	resumed, steered, or inspected further — leaving them running is what produces
	orphaned tmux windows.

	With `handles=None` (the default), every subagent spawned by this process is
	closed. Returns the number of handles closed. Safe to call twice.
	"""
	targets = handles if handles is not None else REGISTRY.handles()
	closed = 0
	for handle in targets:
		try:
			handle.kill()
			closed += 1
		except Exception:
			pass
	return closed


def emit_status_line() -> None:  # re-exported for the runtime bridge tests
	REGISTRY.emit()

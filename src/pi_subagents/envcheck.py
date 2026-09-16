"""Environment probing for pi-subagents.

Runs at ``import pi_subagents``. The contract:

1. ``PI_SUBAGENT_DEPTH`` is set → raise ``NotImplementedError`` (spawned agents
   cannot spawn agents; the orchestration tree stays height 2 until the
   PI_PTC_PRIMARY reporting mesh exists).
2. ``$TMUX`` is missing or the tmux server is unreachable → warn once, set
   ``ENV_OK = False``; every API call then raises ``NotImplementedError``.
3. Otherwise capture the current tmux session so spawned windows land next to
   the caller, and set ``ENV_OK = True``.
"""

from __future__ import annotations

import os
import sys

ENV_OK = False
WARNING_SHOWN = False

TMUX_ENV_MISSING = (
	"pi_subagents: not running inside a tmux session ($TMUX is unset). "
	"Subagent spawning is nonfunctional here — any call to the subagent API "
	"will raise NotImplementedError. The rest of this script still runs."
)
TMUX_UNREACHABLE = (
	"pi_subagents: $TMUX is set but the tmux server did not answer "
	"({detail}). Subagent spawning is nonfunctional here — any call to the "
	"subagent API will raise NotImplementedError. The rest of this script "
	"still runs."
)
DEPTH_LOCKED = (
	"pi_subagents: subagent spawning is disabled in this environment "
	"(PI_SUBAGENT_DEPTH={depth} is set — spawned agents cannot spawn further "
	"agents). Use provision_python_session/python_exec for computation instead."
)


def _warn(message: str) -> None:
	global WARNING_SHOWN
	if WARNING_SHOWN:
		return
	WARNING_SHOWN = True
	print(f"WARNING: {message}", file=sys_stderr())


def sys_stderr():
	import sys

	return sys.stderr


class TmuxPlacement:
	"""Where in tmux the current process lives (used for window placement)."""

	__slots__ = ("session_name", "session_id", "window_id", "pane_id", "tmux_socket")

	def __init__(self, session_name: str, session_id: str, window_id: str, pane_id: str, tmux_socket: str):
		self.session_name = session_name
		self.session_id = session_id
		self.window_id = window_id
		self.pane_id = pane_id
		self.tmux_socket = tmux_socket


placement: TmuxPlacement | None = None


def ensure_environment() -> None:
	"""Idempotent import-time probe. Raises NotImplementedError on depth lock."""
	depth = os.environ.get("PI_SUBAGENT_DEPTH", "")
	if depth not in ("", "0"):
		try:
			parsed = int(depth)
		except ValueError:
			parsed = 1
		if parsed > 0:
			raise NotImplementedError(DEPTH_LOCKED.format(depth=depth))

	global placement, ENV_OK
	if placement is not None:
		ENV_OK = True
		return

	if not os.environ.get("TMUX"):
		_warn(TMUX_ENV_MISSING)
		ENV_OK = False
		return

	try:
		info = _tmux_display()
	except Exception as error:
		_warn(TMUX_UNREACHABLE.format(detail=error))
		ENV_OK = False
		return

	session_name, session_id, window_id, pane_id, tmux_socket = info
	placement = TmuxPlacement(session_name, session_id, window_id, pane_id, tmux_socket)
	ENV_OK = True


def _tmux_display() -> tuple[str, str, str, str, str]:
	"""Query tmux for the current pane's placement context."""
	out = tmux(
		"display-message",
		"-p",
		"-t",
		os.environ.get("TMUX_PANE", ""),
		"#{session_name}\t#{session_id}\t#{window_id}\t#{pane_id}\t#{socket_path}",
	)
	parts = out.rstrip("\n").split("\t")
	if len(parts) != 5:
		raise RuntimeError(f"unexpected tmux display output: {out!r}")
	return tuple(parts)  # type: ignore[return-value]


def tmux(*args: str) -> str:
	"""Run a tmux command against the server this process belongs to."""
	import subprocess

	# tmux picks the server from $TMUX; the caller's env is already correct,
	# but be explicit in case the env var is a socket path plus metadata.
	return subprocess.run(
		["tmux", *args],
		capture_output=True,
		text=True,
		check=True,
		env={**os.environ},
	).stdout


def require_environment() -> TmuxPlacement:
	"""For API entry points: raise unless spawning is functional here."""
	ensure_environment()
	if not ENV_OK or placement is None:
		raise NotImplementedError(
			"pi_subagents: not running inside a tmux session — subagent spawning "
			"is unavailable here. Run this script inside tmux to spawn subagents."
		)
	return placement

"""tmux + pi process management for subagents."""

from __future__ import annotations

import os
import re
import secrets
import shlex
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from .envcheck import require_environment, tmux

SOCK_DIR = Path.home() / ".pi" / "pi-sock"
PROFILES_ROOT = Path.home() / ".config" / "pi" / "profiles"


def parent_agent_dir() -> Path:
	"""The agent dir this orchestrating pi instance runs under.

	PI_CODING_AGENT_DIR if set, else pi's default ~/.pi/agent.
	"""
	env = os.environ.get("PI_CODING_AGENT_DIR", "").strip()
	return Path(env).expanduser() if env else Path.home() / ".pi" / "agent"


def subagent_agent_dir() -> Path:
	"""Agent dir spawned subagent pi instances run under.

	PI_CODING_SUBAGENT_DIR wins when set (any directory with a pi config —
	pi-profiles-managed or a hand-rolled copy). Default: the parent's own
	agent dir, so installing pi-subagents works with zero setup — subagents
	share the orchestrator's config, extensions (pi-sock!), and auth.
	"""
	env = os.environ.get("PI_CODING_SUBAGENT_DIR", "").strip()
	return Path(env).expanduser() if env else parent_agent_dir()


def resolve_agent_dir(agent_dir: str | os.PathLike[str] | None) -> Path:
	"""Resolve the agent dir a spawned subagent runs under.

	None → PI_CODING_SUBAGENT_DIR or the parent's agent dir (see
	subagent_agent_dir). A bare name → <pi-profiles-root>/<name> (pi-profiles
	layout, e.g. ~/.config/pi/profiles/subagents); an absolute/relative path →
	that directory, used verbatim as the agent dir. Raises when the resolved
	directory does not exist.
	"""
	explicit = agent_dir is not None or "PI_CODING_SUBAGENT_DIR" in os.environ
	if agent_dir is None:
		candidate = subagent_agent_dir()
	else:
		candidate = Path(agent_dir).expanduser()
		if not candidate.is_absolute() and "/" not in str(agent_dir) and "\\" not in str(agent_dir):
			candidate = PROFILES_ROOT / agent_dir
	# Only explicitly configured dirs are validated: the default is the
	# parent's own agent dir, which exists by construction (pi is running
	# out of it), and a missing-dir error there would be pure noise.
	if explicit and not candidate.is_dir():
		raise RuntimeError(f"subagent agent dir not found: {candidate}")
	return candidate


def socket_dir() -> Path:
	return Path(os.environ.get("PI_SOCK_DIR", str(SOCK_DIR)))


def _slug(name: str) -> str:
	slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip().lower()).strip("-")
	return slug[:28] or "subagent"


def unique_socket_name(name: str) -> str:
	"""Allocate a collision-resistant PI_SOCK_NAME independent of socket timing."""
	base = f"subagent-{_slug(name)}"
	# The old base-name-first strategy raced when two windows were submitted before
	# either socket bound. Every allocation now receives an opaque suffix.
	for _ in range(64):
		candidate = f"{base}-{secrets.token_hex(4)}"
		if not (socket_dir() / f"{candidate}.sock").exists():
			return candidate
	raise RuntimeError("could not allocate a unique pi-sock socket name")


def max_concurrent() -> int:
	try:
		return max(1, int(os.environ.get("PI_SUBAGENTS_MAX_CONCURRENT", "8")))
	except ValueError:
		return 8


def spawn_pi_window(
	prompt: str,
	*,
	name: str,
	cwd: str,
	window_name: str,
	model: str | None,
	thinking: str | None,
	socket_name: str,
	depth: int,
	agentDir: str | os.PathLike[str] | None = None,
	session_name: str | None = None,
) -> "WindowRef":
	"""Create a tmux window running an interactive pi instance with a prompt.

	Returns a WindowRef (window id + socket path). Raises on tmux failure.
	`agentDir` selects the agent dir subagents run under: a bare name under
	the pi-profiles root, or a path used verbatim; None → PI_CODING_SUBAGENT_DIR
	or the parent's own agent dir.
	"""
	placement = require_environment()
	sock_path = socket_dir() / f"{socket_name}.sock"
	agent_dir_resolved = resolve_agent_dir(agentDir)

	pi_cmd = _build_pi_command(
		prompt,
		model=model,
		thinking=thinking,
		session_name=subagent_session_name(session_name or name),
	)
	args = [
		"new-window",
		"-d",
		"-P",
		"-F",
		"#{window_id}",
		"-t",
		f"{placement.session_name}:",
		"-n",
		window_name,
		"-e",
		f"PI_CODING_AGENT_DIR={agent_dir_resolved}",
		"-e",
		f"PI_SOCK_NAME={socket_name}",
		"-e",
		f"PI_SUBAGENT_DEPTH={depth}",
		"-e",
		"PTC_USE_DOCKER=false",
		"-e",
		"PTC_ALLOW_UNSANDBOXED_SUBPROCESS=true",
		"-c",
		cwd,
	]
	primary = os.environ.get("PI_PTC_PRIMARY")
	if primary:
		args += ["-e", f"PI_PTC_PRIMARY={primary}"]
	args.append(pi_cmd)

	window_id = tmux(*args).strip()
	if not window_id:
		raise RuntimeError("tmux new-window returned no window id")
	return WindowRef(window_id=window_id, socket_path=str(sock_path), name=name)


def subagent_session_name(name: str) -> str:
	"""Session display name for a subagent.

	pi composes its terminal/pane title as "π - [<session name> - ]<cwd>", and the
	tmux overview surfaces exactly that. Passing this as --name makes every
	subagent identifiable at a glance in a wall of pi windows (and names the
	session in pi's own session list / --resume picker).
	"""
	label = (name or "subagent").strip()
	return f"(subagent) {label}"


def _build_pi_command(
	prompt: str,
	*,
	model: str | None,
	thinking: str | None,
	session_name: str | None = None,
) -> str:
	"""Command for the tmux window.

	The prompt is deliberately NOT part of this command: it is delivered over
	pi-sock after the instance reports ready. Passing it as a tmux argument
	worked only up to tmux/OS argv limits — beyond that it was silently
	truncated mid-string (which surfaced as a bizarre quoting error, since the
	closing quote never arrived).

	`session_name` becomes pi's --name, which drives the pane title.
	"""
	pi = shutil.which("pi") or "pi"
	parts = [shlex.quote(pi)]
	if session_name:
		parts += ["--name", shlex.quote(session_name)]
	if model:
		parts += ["--model", shlex.quote(model)]
	if thinking:
		parts += ["--thinking", shlex.quote(thinking)]
	return " ".join(parts)


def _quote_for_tmux_shell(text: str) -> str:
	"""Quote a string for the shell tmux uses to run a window command."""
	return shlex.quote(text)


def window_alive(window_id: str) -> bool:
	try:
		windows = tmux("list-windows", "-a", "-F", "#{window_id}").splitlines()
	except Exception:
		return False
	return window_id in windows


def kill_window(window_id: str) -> None:
	try:
		tmux("kill-window", "-t", window_id)
	except Exception:
		pass


@dataclass
class WindowRef:
	window_id: str
	socket_path: str
	name: str
	created_at: float = field(default_factory=time.time)

"""tmux + pi process management for subagents."""

from __future__ import annotations

import os
import re
import secrets
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from .envcheck import require_environment, tmux

SOCK_DIR = Path.home() / ".pi" / "pi-sock"
DEFAULT_PROFILE = Path.home() / ".config" / "pi" / "profiles" / "subagents"


def subagent_profile_dir() -> Path:
	return Path(os.environ.get("PI_SUBAGENTS_PROFILE", str(DEFAULT_PROFILE)))


def socket_dir() -> Path:
	return Path(os.environ.get("PI_SOCK_DIR", str(SOCK_DIR)))


def _slug(name: str) -> str:
	slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip().lower()).strip("-")
	return slug[:28] or "subagent"


def unique_socket_name(name: str) -> str:
	"""Pick a PI_SOCK_NAME that does not collide with an existing socket."""
	base = f"subagent-{_slug(name)}"
	sdir = socket_dir()
	if not (sdir / f"{base}.sock").exists():
		return base
	for _ in range(64):
		candidate = f"{base}-{secrets.token_hex(2)}"
		if not (sdir / f"{candidate}.sock").exists():
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
) -> "WindowRef":
	"""Create a tmux window running an interactive pi instance with a prompt.

	Returns a WindowRef (window id + socket path). Raises on tmux failure.
	"""
	placement = require_environment()
	sock_path = socket_dir() / f"{socket_name}.sock"
	profile = subagent_profile_dir()
	if not profile.is_dir():
		raise RuntimeError(f"subagent pi profile not found: {profile}")

	pi_cmd = _build_pi_command(prompt, model=model, thinking=thinking)
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
		f"PI_CODING_AGENT_DIR={profile}",
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


def _build_pi_command(prompt: str, *, model: str | None, thinking: str | None) -> str:
	pi = shutil.which("pi") or "pi"
	parts = [pi]
	if model:
		parts += ["--model", model]
	if thinking:
		parts += ["--thinking", thinking]
	parts.append(_quote_for_tmux_shell(prompt))
	return " ".join(parts)


def _quote_for_tmux_shell(text: str) -> str:
	"""tmux runs the command via sh; single quotes are the safe envelope."""
	return "'" + text.replace("'", "'\\''") + "'"


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

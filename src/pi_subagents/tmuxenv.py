"""tmux + pi process management for subagents."""

from __future__ import annotations

import os
import re
import secrets
import shlex
import shutil
# Keep the established process-boundary alias used by external wrappers.
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .envcheck import require_environment, tmux
from .recursion import RootBudget, ensure_can_spawn

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
	session_file: str | None = None,
) -> "WindowRef":
	"""Create a tmux window running an interactive pi instance with a prompt.

	Returns a WindowRef (window id + socket path). Raises on tmux failure.
	`agentDir` selects the agent dir subagents run under: a bare name under
	the pi-profiles root, or a path used verbatim; None → PI_CODING_SUBAGENT_DIR
	or the parent's own agent dir.
	"""
	child_depth = ensure_can_spawn()
	if type(depth) is not int or depth != child_depth:
		raise ValueError("subagent depth must equal the policy-approved child depth")
	placement = require_environment()
	sock_path = socket_dir() / f"{socket_name}.sock"
	agent_dir_resolved = resolve_agent_dir(agentDir)

	pi_cmd = _build_pi_command(
		prompt,
		model=model,
		thinking=thinking,
		# Reopening inherits the persisted display name unless explicitly renamed.
		session_name=(subagent_session_name(session_name or name)
		              if session_file is None or session_name is not None else None),
		session_file=session_file,
	)
	args = [
		"new-window",
		"-d",
		"-P",
		"-F",
		"#{window_id}",
		"-t",
		f"{placement.session_id}:",
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
	budget = RootBudget.for_environment()
	token = budget.reserve(socket_name, child_depth) if budget is not None else None
	window_id = None
	attempted = False
	# A unique marker in pane_start_command lets failed new-window calls be
	# reconciled without assuming an exception means no window was created.
	marker = f"PI_SUBAGENTS_WINDOW_OWNER={token}"
	try:
		if budget is not None:
			child_environment = {**runtime_child_env(), **budget.child_env(token)}
			for key, value in child_environment.items():
				args += ["-e", f"{key}={value}"]
			pi_cmd = f"exec env {shlex.quote(marker)} {pi_cmd}"
		args.append(pi_cmd)
		attempted = True
		window_id = tmux(*args).strip()
		if not re.fullmatch(r"@[0-9]+", window_id):
			raise RuntimeError(f"tmux new-window returned invalid window id: {window_id!r}")
		if budget is not None:
			budget.attach(token, window_id)
		return WindowRef(window_id=window_id, socket_path=str(sock_path), name=name,
		                 budget=budget, admission_token=token)
	except BaseException:
		if budget is not None:
			confirmed = not attempted
			if attempted:
				confirmed = _cleanup_failed_spawn(marker, window_id, placement.window_id)
			if confirmed:
				budget.release(token)
		raise


def runtime_child_env() -> dict[str, str]:
	"""Pin children to the interpreter and source actually imported here.

	Do not trust a stale PTC_SUBAGENTS_SOURCE or tmux's server environment.
	Preserve the venv executable's symlink (resolving it loses the venv).
	"""
	package_parent = Path(__file__).resolve().parent.parent
	source = package_parent.parent if package_parent.name == "src" else package_parent
	pythonpath = str(package_parent)
	if os.environ.get("PYTHONPATH"):
		pythonpath += os.pathsep + os.environ["PYTHONPATH"]
	env = {
		"PTC_PYTHON_EXECUTABLE": os.path.abspath(sys.executable),
		"PTC_SUBAGENTS_SOURCE": str(source),
		"PYTHONPATH": pythonpath,
	}
	for key in ("PI_SOCK_DIR", "PI_CODING_SUBAGENT_DIR"):
		if key in os.environ:
			env[key] = os.environ[key]
	return env


def _cleanup_failed_spawn(marker: str, window_id: str | None, caller_window: str) -> bool:
	"""Release admission only after an authoritative owned-window reconciliation."""
	try:
		rows = tmux("list-panes", "-a", "-F", "#{window_id}\t#{pane_start_command}")
		owned = set()
		for row in rows.splitlines():
			wid, _, command = row.partition("\t")
			if marker in shlex.split(command):
				owned.add(wid)
		if window_id and re.fullmatch(r"@[0-9]+", window_id):
			# Never kill a returned ID whose ownership cannot be reconciled.
			listed = {row.partition("\t")[0] for row in rows.splitlines()}
			if window_id in listed and window_id not in owned:
				return False
		for wid in owned:
			if wid == caller_window:
				return False
			kill_window(wid)
			if not window_absent(wid):
				return False
		return True
	except Exception:
		return False


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
	session_file: str | None = None,
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
	if session_file:
		parts += ["--session", shlex.quote(session_file)]
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


def window_absent(window_id: str) -> bool:
	"""True only when tmux authoritatively reports an exact window ID absent."""
	if not re.fullmatch(r"@[0-9]+", window_id):
		return False
	try:
		windows = tmux("list-windows", "-a", "-F", "#{window_id}").splitlines()
	except Exception:
		return False
	return window_id not in windows


def window_alive(window_id: str) -> bool:
	try:
		windows = tmux("list-windows", "-a", "-F", "#{window_id}").splitlines()
	except Exception:
		# Unknown is not confirmed gone: admission must remain charged.
		return True
	return window_id in windows


def kill_window(window_id: str) -> bool:
	"""Kill an exact window ID, never a caller window or a tmux target expression."""
	try:
		if not re.fullmatch(r"@[0-9]+", window_id):
			return False
		if window_id == require_environment().window_id:
			return False
		tmux("kill-window", "-t", window_id)
	except Exception:
		return False
	return True


@dataclass
class WindowRef:
	window_id: str
	socket_path: str
	name: str
	created_at: float = field(default_factory=time.time)
	budget: RootBudget | None = field(default=None, repr=False)
	admission_token: str | None = field(default=None, repr=False)

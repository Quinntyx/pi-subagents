"""python -m pi_subagents — list, abort, or prune subagent pi instances."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .client import PiSockError, PiSockUnavailable, SockClient
from .tmuxenv import socket_dir


def _all_subagent_sockets() -> list[Path]:
	sdir = socket_dir()
	if not sdir.is_dir():
		return []
	return sorted(p for p in sdir.glob("subagent-*.sock"))


def cmd_list() -> int:
	socks = _all_subagent_sockets()
	if not socks:
		print("no subagent sockets")
		return 0
	for path in socks:
		client = SockClient(str(path), timeout=2.0)
		name = path.stem
		try:
			state = client.state()
			idle = "idle" if state.get("isIdle") else "busy"
			print(f"{name:44s} {idle:5s} {state.get('sessionFile', '')}")
		except PiSockUnavailable:
			print(f"{name:44s} DEAD  {path} (stale socket — try: pi-subagents prune {name})")
		except Exception as error:
			print(f"{name:44s} ERR   {error}")
	return 0


def cmd_abort(name: str) -> int:
	path = socket_dir() / f"{name}.sock"
	if not name.startswith("subagent-"):
		candidates = [p for p in _all_subagent_sockets() if name in p.stem]
		if len(candidates) == 1:
			path = candidates[0]
		elif not candidates:
			print(f"no subagent socket matching {name!r}")
			return 1
		else:
			print(f"ambiguous: {', '.join(p.stem for p in candidates)}")
			return 1
	try:
		SockClient(str(path)).abort()
		print(f"aborted {path.stem}")
		return 0
	except PiSockUnavailable:
		print(f"no live pi at {path}")
		return 1


def cmd_prune(name: str | None = None) -> int:
	targets = [socket_dir() / f"{name}.sock"] if name else _all_subagent_sockets()
	pruned = 0
	for path in targets:
		if not path.exists():
			continue
		client = SockClient(str(path), timeout=2.0)
		try:
			client.state()
			continue  # live — leave it alone
		except PiSockUnavailable:
			pass
		except Exception:
			pass
		os.unlink(path)
		print(f"unlinked stale socket {path}")
		pruned += 1
	print(f"pruned {pruned} socket(s)")
	return 0


def main(argv: list[str] | None = None) -> int:
	argv = list(sys.argv[1:] if argv is None else argv)
	if not argv or argv[0] in ("-h", "--help"):
		print(__doc__.strip())
		print("\nusage: python -m pi_subagents [list|abort <name>|prune [name]]")
		return 0
	command = argv[0]
	if command == "list":
		return cmd_list()
	if command == "abort":
		if len(argv) < 2:
			print("abort requires a subagent name")
			return 1
		return cmd_abort(argv[1])
	if command == "prune":
		return cmd_prune(argv[1] if len(argv) > 1 else None)
	print(f"unknown command {command!r}")
	return 1


if __name__ == "__main__":
	raise SystemExit(main())

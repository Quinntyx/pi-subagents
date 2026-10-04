"""Client for the pi-sock unix-socket JSONL RPC (sync + asyncio).

Commands used: send, get_state, get_message, get_activity (pi-sock extension
with the pi-activity relay), subscribe, abort. Protocol per pi-sock README:
newline-delimited JSON; responses carry {type:"response",command,success,data?,
error?,id?}; subscribed events arrive as {type:"event",event,data?}.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import socket
import threading
import time
from typing import Any, Callable

SOCK_DIR = os.path.expanduser("~/.pi/pi-sock")

DEFAULT_POLL_INTERVAL = 1.0
CONNECT_TIMEOUT = 5.0


class PiSockError(Exception):
	pass


class PiSockUnavailable(PiSockError, ConnectionError):
	"""Socket missing, refused, or the pi process is gone."""


class PiSockSessionEnded(PiSockError):
	"""The subagent pi session ended before its turn settled.

	Raised the moment the session's socket disappears or refuses connections
	mid-wait (the pi process exited or was terminated), or when the session's
	last assistant message carries ``stopReason: "aborted"`` (the turn was
	interrupted). Deliberately distinct from a settle timeout: the workflow
	gets the failure immediately instead of waiting out the timeout, and the
	resulting failed AgentResult carries this error.
	"""


class PiSockTurnFailed(PiSockError):
	"""The agent's last run failed at the provider (quota exhausted, rate
	limit, API error — session entry with ``stopReason: "error"``).

	The turn settles (pi goes idle), but treating it as a clean result lets a
	workflow silently drain on a dead agent. Raised instead, carrying the
	session's ``errorMessage`` when present.
	"""


def socket_path_for(name: str) -> str:
	return os.path.join(os.environ.get("PI_SOCK_DIR", SOCK_DIR), f"{name}.sock")


def _last_assistant_outcome(session_file: str | None) -> tuple[str, str | None] | None:
	"""Best-effort (stopReason, errorMessage) of the session's last assistant entry.

	Reads a small tail of the session JSONL (same machine as the subagent).
	Returns None when the file is unavailable or no stopReason is present, so
	older pi versions simply keep the old settle semantics.
	"""
	if not session_file or not os.path.exists(session_file):
		return None
	try:
		with open(session_file, "rb") as fh:
			fh.seek(0, os.SEEK_END)
			size = fh.tell()
			fh.seek(max(0, size - 16384))
			tail = fh.read().decode("utf-8", "replace")
	except OSError:
		return None
	for line in reversed(tail.splitlines()):
		if '"role":"assistant"' not in line and '"role": "assistant"' not in line:
			continue
		stop_match = re.search(r'"stopReason"\s*:\s*"([a-z]+)"', line)
		if not stop_match:
			return None
		error_match = re.search(r'"errorMessage"\s*:\s*"((?:[^"\\]|\\.)*)"', line)
		error_message = None
		if error_match:
			try:
				error_message = json.loads('"' + error_match.group(1) + '"')
			except Exception:
				error_message = error_match.group(1)
		return stop_match.group(1), error_message
	return None


def raise_for_failed_outcome(session_file: str | None, *, last_message: dict | None = None) -> None:
	"""Raise when the session's last assistant entry ended abnormally.

	- stopReason "aborted": the user interrupted the turn.
	- Any other non-success terminal reason ("error", ...): the run failed at
	  the provider — quota exhausted, rate limit, API error.

	Use wire-message metadata as a fallback when the session file is unavailable.
	Both bubble immediately as errors instead of settling cleanly, so a
	workflow cannot silently drain on a dead agent. "stop", "toolUse",
	"length", "pending" and "deferred" are treated as successful settles.
	"""
	outcome = _last_assistant_outcome(session_file)
	if outcome is None and isinstance(last_message, dict):
		stop = last_message.get("stopReason")
		if isinstance(stop, str):
			outcome = (stop, last_message.get("errorMessage"))
	if outcome is None:
		return
	stop, error_message = outcome
	detail = f": {error_message}" if error_message else ""
	if stop == "aborted":
		raise PiSockSessionEnded(
			f"pi session's last turn was interrupted (stopReason=aborted){detail}"
		)
	if stop not in ("stop", "toolUse", "length", "pending", "deferred"):
		raise PiSockTurnFailed(
			f"agent run failed at the provider (stopReason={stop}{detail})"
		)


# ---------------------------------------------------------------------------
# sync client
# ---------------------------------------------------------------------------

class SockClient:
	"""One-shot-command sync client. Each command opens its own connection."""

	def __init__(self, sock_path: str, timeout: float = CONNECT_TIMEOUT):
		self.sock_path = sock_path
		self.timeout = timeout

	def _connect(self) -> socket.socket:
		sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
		sock.settimeout(self.timeout)
		try:
			sock.connect(self.sock_path)
		except FileNotFoundError as error:
			raise PiSockUnavailable(f"no pi-sock socket at {self.sock_path}") from error
		except (ConnectionRefusedError, OSError) as error:
			raise PiSockUnavailable(f"pi-sock socket at {self.sock_path} refused: {error}") from error
		return sock

	def _request(self, payload: dict) -> dict:
		sock = self._connect()
		try:
			with sock.makefile("rwb") as stream:
				stream.write((json.dumps(payload) + "\n").encode())
				stream.flush()
				while True:
					line = stream.readline()
					if not line:
						raise PiSockError(f"pi-sock closed the connection ({self.sock_path})")
					try:
						message = json.loads(line.decode())
					except json.JSONDecodeError:
						continue
					if message.get("type") == "response":
						return message
		except socket.timeout as error:
			raise PiSockError(f"pi-sock command timed out ({self.sock_path})") from error

	def request(self, command: dict) -> dict:
		response = self._request({"id": f"c{next(_SEQ)}", **command})
		if not response.get("success"):
			raise PiSockError(response.get("error") or f"{command.get('type')} failed")
		return response.get("data") or {}

	def send(self, text: str, mode: str = "steer") -> dict:
		return self.request({"type": "send", "text": text, "mode": mode})

	def state(self) -> dict:
		return self.request({"type": "get_state"})

	def message(self) -> dict | None:
		return (self.request({"type": "get_message"}) or {}).get("message")

	def activity(self) -> dict | None:
		try:
			return self.request({"type": "get_activity"})
		except PiSockError:
			return None

	def abort(self) -> dict:
		return self.request({"type": "abort"})

	def set_session_name(self, name: str) -> dict:
		return self.request({"type": "set_session_name", "name": name})

	def is_alive(self) -> bool:
		try:
			self.state()
			return True
		except PiSockUnavailable:
			return False

	def wait_settled(
		self,
		timeout: float | None,
		poll: float = DEFAULT_POLL_INTERVAL,
		on_tick: Callable[[dict], None] | None = None,
		after_message: dict | None = None,
	) -> dict | None:
		"""Wait until the agent settles.

		Returns the agent_settled-style payload ({"lastAssistant", "ranMs", ...})
		or None on timeout. Correctness notes:
		- while streaming: idle=False; poll until idle=True
		- prompt accepted but not started: idle=True, hasPendingMessages=True
		- finished before we started watching: idle=True, no pending, last message exists
		Raises:
		- PiSockSessionEnded: the pi session ended mid-wait (process exited or was
		  terminated — socket gone or refusing), or the last turn was interrupted
		  (last assistant message has stopReason "aborted"). Callers surface this
		  as a failed result instead of a silent drain.
		"""
		deadline = (time.monotonic() + timeout) if timeout is not None else None
		saw_running = False
		saw_settle = False
		outcome_checked = False
		while True:
			if self.sock_path and not os.path.exists(self.sock_path):
				raise PiSockSessionEnded(
					f"pi session ended before settling (socket {self.sock_path} is gone)"
				)
			try:
				state = self.state()
			except PiSockUnavailable as error:
				raise PiSockSessionEnded(
					f"pi session ended before settling ({error})"
				) from error
			if not state.get("isIdle"):
				saw_running = True
				outcome_checked = False
			elif not state.get("hasPendingMessages"):
				# Idle with nothing queued: the run has ended — completed, errored,
				# or interrupted. Check the terminal outcome ONCE per run before the
				# settle test: a provider failure that produced no text content is
				# filtered out of get_message, so the settle condition below may
				# never fire and the wait would silently run out its timeout.
				if not outcome_checked:
					outcome_checked = True
					raise_for_failed_outcome(state.get("sessionFile"))
				last = self.message()
				if last is not None and (saw_running or last != after_message):
					saw_settle = True
			if saw_settle:
				return {"lastAssistant": last, "isIdle": True}
			if on_tick is not None:
				on_tick(state)
			if deadline is not None and time.monotonic() >= deadline:
				return None
			time.sleep(poll)

	def wait_settled_threaded(
		self,
		timeout: float | None,
		poll: float = DEFAULT_POLL_INTERVAL,
		on_tick: Callable[[dict], None] | None = None,
	) -> threading.Thread:
		"""Fire-and-forget watcher thread that keeps the registry fresh."""
		def _run():
			try:
				self.wait_settled(timeout, poll=poll, on_tick=on_tick)
			except Exception:
				pass

		thread = threading.Thread(target=_run, daemon=True)
		thread.start()
		return thread


_SEQ = iter(range(1, 10**9))


# ---------------------------------------------------------------------------
# asyncio client
# ---------------------------------------------------------------------------

class AsyncSockClient:
	def __init__(self, sock_path: str, timeout: float = CONNECT_TIMEOUT):
		self.sock_path = sock_path
		self.timeout = timeout

	async def _connect(self):
		try:
			return await asyncio.wait_for(asyncio.open_unix_connection(self.sock_path), self.timeout)
		except FileNotFoundError as error:
			raise PiSockUnavailable(f"no pi-sock socket at {self.sock_path}") from error
		except (ConnectionRefusedError, OSError) as error:
			raise PiSockUnavailable(f"pi-sock socket at {self.sock_path} refused: {error}") from error
		except asyncio.TimeoutError as error:
			raise PiSockError(f"pi-sock connect timed out ({self.sock_path})") from error

	async def request(self, command: dict) -> dict:
		reader, writer = await self._connect()
		try:
			writer.write((json.dumps({"id": f"a{next(_SEQ)}", **command}) + "\n").encode())
			await writer.drain()
			deadline = time.monotonic() + self.timeout
			while True:
				remaining = deadline - time.monotonic()
				if remaining <= 0:
					raise PiSockError(f"pi-sock command timed out ({self.sock_path})")
				try:
					line = await asyncio.wait_for(reader.readline(), remaining)
				except asyncio.TimeoutError as error:
					raise PiSockError(f"pi-sock command timed out ({self.sock_path})") from error
				if not line:
					raise PiSockError(f"pi-sock closed the connection ({self.sock_path})")
				try:
					message = json.loads(line.decode())
				except json.JSONDecodeError:
					continue
				if message.get("type") == "response":
					if not message.get("success"):
						raise PiSockError(message.get("error") or f"{command.get('type')} failed")
					return message.get("data") or {}
		finally:
			writer.close()
			try:
				await writer.wait_closed()
			except Exception:
				pass

	async def send(self, text: str, mode: str = "steer") -> dict:
		return await self.request({"type": "send", "text": text, "mode": mode})

	async def state(self) -> dict:
		return await self.request({"type": "get_state"})

	async def message(self) -> dict | None:
		return (await self.request({"type": "get_message"}) or {}).get("message")

	async def activity(self) -> dict | None:
		try:
			return await self.request({"type": "get_activity"})
		except PiSockError:
			return None

	async def abort(self) -> dict:
		return await self.request({"type": "abort"})

	async def set_session_name(self, name: str) -> dict:
		return await self.request({"type": "set_session_name", "name": name})

	async def wait_settled(
		self,
		timeout: float | None,
		poll: float = DEFAULT_POLL_INTERVAL,
		on_tick: Callable[[dict], Any] | None = None,
		after_message: dict | None = None,
	) -> dict | None:
		deadline = (time.monotonic() + timeout) if timeout is not None else None
		saw_running = False
		saw_settle = False
		outcome_checked = False
		while True:
			if self.sock_path and not os.path.exists(self.sock_path):
				raise PiSockSessionEnded(
					f"pi session ended before settling (socket {self.sock_path} is gone)"
				)
			try:
				state = await self.state()
			except PiSockUnavailable as error:
				raise PiSockSessionEnded(
					f"pi session ended before settling ({error})"
				) from error
			if not state.get("isIdle"):
				saw_running = True
				outcome_checked = False
			elif not state.get("hasPendingMessages"):
				# Idle with nothing queued: the run has ended — completed, errored,
				# or interrupted. Check the terminal outcome ONCE per run before the
				# settle test: a provider failure that produced no text content is
				# filtered out of get_message, so the settle condition below may
				# never fire and the wait would silently run out its timeout.
				if not outcome_checked:
					outcome_checked = True
					raise_for_failed_outcome(state.get("sessionFile"))
				last = await self.message()
				if last is not None and (saw_running or last != after_message):
					saw_settle = True
			if saw_settle:
				return {"lastAssistant": last, "isIdle": True}
			if on_tick is not None:
				result = on_tick(state)
				if asyncio.iscoroutine(result):
					await result
			if deadline is not None and time.monotonic() >= deadline:
				return None
			await asyncio.sleep(poll)

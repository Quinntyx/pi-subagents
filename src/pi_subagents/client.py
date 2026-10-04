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


def _assistant_outcome_marker(session_file: str | None) -> str | None:
	"""Latest assistant entry identity, including content-free/large replies.

	Read backwards in chunks, retaining complete JSONL entries. Fixed-size tail
	reads can miss the role/stopReason of a large successful or failed message.
	"""
	if not session_file:
		return None
	try:
		with open(session_file, "rb") as fh:
			fh.seek(0, os.SEEK_END)
			position = fh.tell()
			prefix = b""
			while position > 0:
				size = min(position, 16384)
				position -= size
				fh.seek(position)
				lines = (fh.read(size) + prefix).split(b"\n")
				prefix = lines.pop(0) if position > 0 else b""
				for raw in reversed(lines):
					try:
						entry = json.loads(raw)
					except (ValueError, UnicodeDecodeError):
						continue
					if isinstance(entry, dict) and entry.get("type") == "message":
						message = entry.get("message")
						if isinstance(message, dict) and message.get("role") == "assistant":
							return raw.decode("utf-8", "replace")
	except OSError:
		return None
	return None


def _last_assistant_outcome(session_file: str | None) -> tuple[str, str | None] | None:
	"""Best-effort terminal outcome of the latest persisted assistant entry."""
	marker = _assistant_outcome_marker(session_file)
	if marker is None:
		return None
	message = json.loads(marker).get("message") or {}
	stop = message.get("stopReason")
	if not isinstance(stop, str):
		return None
	error = message.get("errorMessage")
	return stop, error if isinstance(error, str) else None


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


def _delivered_assistant_marker(session_file: str | None, deliveries: tuple) -> str | None:
	"""Latest assistant after all accepted direct sends' existing log entries.

	pi-sock sends custom session-message entries. Pre-RPC offsets distinguish
	repeated text. The old turn may finish between get_state and send, but its
	assistant precedes the delivery boundary and cannot satisfy the new turn.
	"""
	if not session_file or not deliveries or any(offset is None for offset, _ in deliveries):
		return None
	try:
		with open(session_file, "rb") as fh:
			fh.seek(deliveries[0][0])
			matched = 0
			marker = None
			while True:
				position = fh.tell()
				raw = fh.readline()
				if not raw or not raw.endswith(b"\n"):
					break  # ignore an entry the writer has not finished
				try:
					entry = json.loads(raw)
				except (ValueError, UnicodeDecodeError):
					continue
				if not isinstance(entry, dict):
					continue
				if matched < len(deliveries):
					offset, text = deliveries[matched]
					if (position >= offset and entry.get("type") == "custom_message"
						and entry.get("customType") == "session-message" and entry.get("content") == text):
						matched += 1
					continue
				message = entry.get("message")
				if entry.get("type") == "message" and isinstance(message, dict) and message.get("role") == "assistant":
					marker = raw.decode("utf-8", "replace")
			return marker if matched == len(deliveries) else None
	except OSError:
		return None


def _settled_assistant(marker: str | None, wire: dict | None, *, after_outcome: str | None,
                       after_message: dict | None, deliveries: tuple | None = None) -> dict | None:
	"""Correlate a reply with the actual durable terminal, including empty text."""
	if deliveries is not None and marker is None:
		return None
	if after_outcome is not None and (marker is None or marker == after_outcome):
		return None
	if marker is not None:
		message = json.loads(marker)["message"]
		raise_for_failed_outcome(None, last_message=message)
		stop = message.get("stopReason")
		if stop in ("toolUse", "pending", "deferred"):
			return None
		content = message.get("content", [])
		text = content if isinstance(content, str) else "\n".join(
			part.get("text", "") for part in content
			if isinstance(part, dict) and part.get("type") == "text"
		)
		last = {"content": text, "timestamp": message.get("timestamp")}
		if "timestamp" not in message and wire is not None and wire.get("content") == text:
			last["timestamp"] = wire.get("timestamp")
		if isinstance(stop, str):
			last["stopReason"] = stop
		if after_outcome is not None or deliveries is not None or last != after_message:
			return last
		return None
	if wire is not None and wire != after_message:
		raise_for_failed_outcome(None, last_message=wire)
		return wire
	return None


# ---------------------------------------------------------------------------
# sync client
# ---------------------------------------------------------------------------

class SockClient:
	"""One-shot-command sync client. Each command opens its own connection."""

	def __init__(self, sock_path: str, timeout: float = CONNECT_TIMEOUT):
		self.sock_path = sock_path
		self.timeout = timeout
		self._after_outcome: str | None = None
		self._after_deliveries: tuple | None = None

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
			if state.get("isIdle") and not state.get("hasPendingMessages"):
				marker = (_delivered_assistant_marker(state.get("sessionFile"), self._after_deliveries)
				          if self._after_deliveries is not None else
				          _assistant_outcome_marker(state.get("sessionFile")))
				last = _settled_assistant(marker, self.message(), after_outcome=self._after_outcome,
				                          after_message=after_message, deliveries=self._after_deliveries)
				if last is not None:
					self._after_outcome = None
					self._after_deliveries = None
					return {**state, "lastAssistant": last, "isIdle": True}
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
		self._after_outcome: str | None = None
		self._after_deliveries: tuple | None = None

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
			if state.get("isIdle") and not state.get("hasPendingMessages"):
				if self._after_deliveries is not None:
					marker = await asyncio.to_thread(_delivered_assistant_marker, state.get("sessionFile"), self._after_deliveries)
				else:
					marker = await asyncio.to_thread(_assistant_outcome_marker, state.get("sessionFile"))
				last = _settled_assistant(marker, await self.message(), after_outcome=self._after_outcome,
				                          after_message=after_message, deliveries=self._after_deliveries)
				if last is not None:
					self._after_outcome = None
					self._after_deliveries = None
					return {**state, "lastAssistant": last, "isIdle": True}
			if on_tick is not None:
				result = on_tick(state)
				if asyncio.iscoroutine(result):
					await result
			if deadline is not None and time.monotonic() >= deadline:
				return None
			await asyncio.sleep(poll)

"""Client for the pi-sock unix-socket JSONL RPC (sync + asyncio).

Commands used: send, get_state, get_message, get_activity (pi-sock extension
with the pi-tool-tree relay), subscribe, abort. Protocol per pi-sock README:
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


def socket_path_for(name: str) -> str:
	return os.path.join(os.environ.get("PI_SOCK_DIR", SOCK_DIR), f"{name}.sock")


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
	) -> dict | None:
		"""Wait until the agent settles.

		Returns the agent_settled-style payload ({"lastAssistant", "ranMs", ...})
		or None on timeout. Correctness notes:
		- while streaming: idle=False; poll until idle=True
		- prompt accepted but not started: idle=True, hasPendingMessages=True
		- finished before we started watching: idle=True, no pending, last message exists
		"""
		deadline = (time.monotonic() + timeout) if timeout is not None else None
		saw_running = False
		saw_settle = False
		while True:
			try:
				state = self.state()
			except PiSockUnavailable:
				return None  # pi is gone; treat as dead rather than hanging
			if not state.get("isIdle"):
				saw_running = True
			elif saw_running:
				saw_settle = True
			elif not state.get("hasPendingMessages") and self.message() is not None:
				# finished (or never started and already replied) — settle
				saw_settle = True
			if saw_settle:
				last = self.message()
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

	async def wait_settled(
		self,
		timeout: float | None,
		poll: float = DEFAULT_POLL_INTERVAL,
		on_tick: Callable[[dict], Any] | None = None,
	) -> dict | None:
		deadline = (time.monotonic() + timeout) if timeout is not None else None
		saw_running = False
		saw_settle = False
		while True:
			try:
				state = await self.state()
			except PiSockUnavailable:
				return None
			if not state.get("isIdle"):
				saw_running = True
			elif saw_running:
				saw_settle = True
			elif not state.get("hasPendingMessages") and (await self.message()) is not None:
				saw_settle = True
			if saw_settle:
				last = await self.message()
				return {"lastAssistant": last, "isIdle": True}
			if on_tick is not None:
				result = on_tick(state)
				if asyncio.iscoroutine(result):
					await result
			if deadline is not None and time.monotonic() >= deadline:
				return None
			await asyncio.sleep(poll)

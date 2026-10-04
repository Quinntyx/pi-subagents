"""Shared helpers: fake pi-sock server for client/handle tests.

wait_settled() is poll-based, so tests drive settle behavior purely through the
behavior map — no server-push events needed. Opt-in completed turns and send
hooks model retained-session continuation without changing custom runtimes.
"""

from __future__ import annotations

import json
import os
import socket
import threading


class FakePiSockServer:
	"""Minimal pi-sock server: accepts command lines, replies per a behavior map."""

	def __init__(self, sock_dir: str, name: str):
		self.sock_path = os.path.join(sock_dir, f"{name}.sock")
		self.name = name
		self.behaviors: dict[str, object] = {}
		self.sent: list[dict] = []
		self.session_path = os.path.join(sock_dir, f"{name}.jsonl")
		self._turn = 0
		self._turn_lock = threading.Lock()
		self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
		self._server.bind(self.sock_path)
		os.chmod(self.sock_path, 0o600)
		self._server.listen(16)
		self._running = True
		self._thread = threading.Thread(target=self._serve, daemon=True)
		self._thread.start()

	def complete_turn(self, content: str, *, timestamp: int | None = None,
	                  stop_reason: str = "stop") -> dict:
		"""Publish one completed turn, with matching wire and durable identities.

		Opt-in: lifecycle/race fixtures own their protocol and session files.
		Reads never manufacture a new turn; a send hook must explicitly finish it.
		"""
		with self._turn_lock:
			self._turn += 1
			stamp = timestamp if timestamp is not None else self._turn
			message = {"content": content, "timestamp": stamp, "stopReason": stop_reason}
			with open(self.session_path, "a", encoding="utf-8") as stream:
				if self._turn == 1:
					stream.write(json.dumps({"type": "session", "id": self.name, "version": 3}) + "\n")
				stream.write(json.dumps({"type": "message", "id": f"reply-{self._turn}",
					"message": {**message, "role": "assistant",
						"content": [{"type": "text", "text": content}]}}) + "\n")
			self.behaviors["message"] = message
			self.behaviors["state"] = {
				"isIdle": True, "hasPendingMessages": False,
				"sessionFile": self.session_path, "sessionId": self.name,
			}
			return dict(message)

	def close(self) -> None:
		self._running = False
		try:
			self._server.close()
		except Exception:
			pass

	def _serve(self) -> None:
		while self._running:
			try:
				conn, _ = self._server.accept()
			except OSError:
				return
			threading.Thread(target=self._handle_conn, args=(conn,), daemon=True).start()

	def _handle_conn(self, conn: socket.socket) -> None:
		with conn:
			buf = b""
			while True:
				try:
					chunk = conn.recv(4096)
				except OSError:
					return
				if not chunk:
					return
				buf += chunk
				while b"\n" in buf:
					line, buf = buf.split(b"\n", 1)
					if not line.strip():
						continue
					try:
						command = json.loads(line.decode())
					except json.JSONDecodeError:
						continue
					conn.sendall((json.dumps(self._reply(command)) + "\n").encode())

	def _reply(self, command: dict) -> dict:
		ctype = command.get("type")

		def _behavior(key: str, default):
			value = self.behaviors.get(key, default)
			return value() if callable(value) else value

		if ctype == "get_state":
			return {"type": "response", "command": ctype, "success": True, "data": _behavior("state", {"isIdle": False, "hasPendingMessages": False})}
		if ctype == "get_message":
			return {"type": "response", "command": ctype, "success": True, "data": {"message": _behavior("message", None)}}
		if ctype == "get_activity":
			return {"type": "response", "command": ctype, "success": True, "data": _behavior("activity", {"available": False})}
		if ctype == "send":
			self.sent.append(command)
			hook = self.behaviors.get("send")
			data = hook(command) if callable(hook) else {"delivered": True, "mode": "direct"}
			return {"type": "response", "command": ctype, "success": True, "data": data}
		if ctype in ("abort", "subscribe"):
			return {"type": "response", "command": ctype, "success": True, "data": {}}
		if ctype == "set_session_name":
			self.sent.append(command)
			return {"type": "response", "command": ctype, "success": True, "data": {"renamed": command.get("name")}}
		return {"type": "response", "command": ctype, "success": False, "error": "unsupported"}

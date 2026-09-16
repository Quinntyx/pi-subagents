"""AgentSession — a pi session on disk (JSONL), for trajectory inspection.

This is deliberately separate from AgentHandle (the currently-executing agent /
pi-sock socket). The session is the durable artifact: tool call trajectory,
thinking traces, prose, and resume all resolve against it.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
	from .handle import AgentHandle


class AgentSession:
	"""Read-only analysis of a subagent's pi session JSONL file."""

	def __init__(self, handle: "AgentHandle"):
		self._handle = handle
		self._session_file: str | None = None
		self._parsed: dict[str, Any] | None = None

	@property
	def handle(self) -> "AgentHandle":
		return self._handle

	def get_handle(self) -> "AgentHandle":
		return self._handle

	def _resolve_session_file(self) -> str | None:
		if self._session_file:
			return self._session_file
		if not self._handle.socket_path:
			return None
		from .client import PiSockUnavailable, SockClient

		client = SockClient(self._handle.socket_path)
		try:
			state = client.state()
		except Exception:
			return None
		path = state.get("sessionFile")
		if isinstance(path, str) and os.path.exists(path):
			self._session_file = path
		return self._session_file

	@property
	def session_file(self) -> str | None:
		return self._resolve_session_file()

	def _load(self) -> dict[str, Any]:
		if self._parsed is not None:
			return self._parsed
		path = self._resolve_session_file()
		result: dict[str, Any] = {
			"prose": "",
			"thinking": "",
			"tool_calls": [],
			"turns": 0,
			"duration_ms": None,
			"events": [],
			"available": False,
		}
		if not path:
			self._parsed = result
			return result
		prose_parts: list[str] = []
		thinking_parts: list[str] = []
		events: list[dict[str, Any]] = []
		pending_calls: dict[str, dict[str, Any]] = {}
		assistant_count = 0
		first_ts: int | None = None
		last_ts: int | None = None
		try:
			with open(path, encoding="utf-8") as fh:
				for line in fh:
					line = line.strip()
					if not line:
						continue
					try:
						entry = json.loads(line)
					except json.JSONDecodeError:
						continue
					if entry.get("type") != "message":
						continue
					message = entry.get("message") or {}
					role = message.get("role")
					timestamp = _to_ms(message.get("timestamp"))
					if timestamp:
						first_ts = first_ts or timestamp
						last_ts = max(last_ts or 0, timestamp)
					if role == "assistant":
						assistant_count += 1
						for block in message.get("content") or []:
							btype = block.get("type")
							if btype == "text":
								text = block.get("text") or ""
								prose_parts.append(text)
								events.append({"kind": "prose", "text": text, "timestamp": timestamp})
							elif btype == "thinking":
								think = block.get("thinking") or ""
								thinking_parts.append(think)
								events.append({"kind": "thinking", "text": think, "timestamp": timestamp})
							elif btype == "toolCall":
								call = {
									"id": block.get("id"),
									"tool": block.get("name"),
									"arguments": block.get("arguments") or {},
									"startedAt": timestamp,
									"durationMs": None,
									"isError": None,
									"result_preview": None,
								}
								pending_calls[call["id"]] = call
								events.append({"kind": "tool_call", **call})
					elif role == "toolResult":
						call = pending_calls.get(message.get("toolCallId"))
						if call is not None:
							call["isError"] = bool(message.get("isError"))
							if timestamp and call.get("startedAt"):
								call["durationMs"] = max(0, timestamp - call["startedAt"])
							texts = [
								b.get("text") or ""
								for b in (message.get("content") or [])
								if isinstance(b, dict) and b.get("type") == "text"
							]
							preview = "\n".join(texts)
							call["result_preview"] = preview[:2000]
							events.append(
								{
									"kind": "tool_result",
									"toolCallId": message.get("toolCallId"),
									"tool": message.get("toolName"),
									"isError": call["isError"],
									"durationMs": call["durationMs"],
									"timestamp": timestamp,
								}
							)
		except OSError:
			pass
		tool_calls = [
			call
			for call in pending_calls.values()
			if call.get("isError") is not None  # only settled calls
		]
		result.update(
			prose="\n".join(p for p in prose_parts if p),
			thinking="\n".join(t for t in thinking_parts if t),
			tool_calls=tool_calls,
			turns=assistant_count,
			duration_ms=(last_ts - first_ts) if first_ts and last_ts and last_ts >= first_ts else None,
			events=events,
			available=True,
		)
		self._parsed = result
		return result

	@property
	def available(self) -> bool:
		return bool(self._load()["available"])

	@property
	def prose(self) -> str:
		return self._load()["prose"]

	@property
	def thinking(self) -> str:
		return self._load()["thinking"]

	@property
	def tool_calls(self) -> list[dict[str, Any]]:
		return self._load()["tool_calls"]

	@property
	def turns(self) -> int:
		return self._load()["turns"]

	@property
	def duration_ms(self) -> int | None:
		return self._load()["duration_ms"]

	def trajectory(self) -> list[dict[str, Any]]:
		return self._load()["events"]

	def invalidate(self) -> None:
		"""Drop the parsed cache (e.g. after resume() added new turns)."""
		self._parsed = None
		self._session_file = None


def _to_ms(value: Any) -> int | None:
	if isinstance(value, (int, float)):
		return int(value) if value < 10**12 else int(value)
	if isinstance(value, str):
		try:
			from datetime import datetime

			return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)
		except Exception:
			return None
	return None

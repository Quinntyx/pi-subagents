"""Response value objects.

AgentStrResponse subclasses str and AgentDictResponse subclasses dict; both
delegate every inspection to the AgentSession returned by .get_session(), so
string/dict behavior stays native and no logic is duplicated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
	from .handle import AgentHandle
	from .session_file import AgentSession


def _session_delegator(name: str):
	def _method(self, *args, **kwargs):
		session = self.get_session()
		attr = getattr(session, name)
		return attr(*args, **kwargs) if callable(attr) else attr

	_method.__name__ = name
	return _method


class _ResponseMixin:
	_session: "AgentSession"

	def get_session(self) -> "AgentSession":
		return self._session

	@property
	def session(self) -> "AgentSession":
		return self._session

	@property
	def tool_calls(self) -> list[dict[str, Any]]:
		return self.get_session().tool_calls

	@property
	def thinking(self) -> str:
		return self.get_session().thinking

	@property
	def prose(self) -> str:
		return self.get_session().prose

	@property
	def turns(self) -> int:
		return self.get_session().turns

	@property
	def duration_ms(self) -> int | None:
		return self.get_session().duration_ms

	@property
	def session_file(self) -> str | None:
		return self.get_session().session_file

	def trajectory(self) -> list[dict[str, Any]]:
		return self.get_session().trajectory()

	def handle(self) -> "AgentHandle":
		return self.get_session().get_handle()


class AgentStrResponse(str, _ResponseMixin):
	"""Settled text response of a subagent; behaves exactly like str."""

	def __new__(cls, text: str, session: "AgentSession"):
		obj = super().__new__(cls, text)
		obj._session = session
		return obj


class AgentDictResponse(dict, _ResponseMixin):
	"""Settled structured response of a subagent; behaves exactly like dict."""

	def __new__(cls, value: dict, session: "AgentSession"):
		obj = super().__new__(cls, value)
		obj._session = session
		return obj

	def __init__(self, value: dict, session: "AgentSession"):  # noqa: D107
		super().__init__(value)

	@property
	def valid(self) -> bool:
		return self.get("valid") is not False

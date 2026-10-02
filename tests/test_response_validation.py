"""Schema replies are validated dictionaries, never invalid/raw response values."""
from types import SimpleNamespace

import pytest

from pi_subagents.errors import PiSubagentsTimeoutError
from pi_subagents.handle import _dict_response, _schema_retries
from pi_subagents.response import AgentDictResponse
from pi_subagents.schema import SchemaValidationError

SCHEMA = {
    "type": "object",
    "required": ["summary"],
    "properties": {"summary": {"type": "string"}},
    "additionalProperties": False,
}


def fake_handle(replies):
    prompts, baselines = [], []
    replies = iter(replies)
    session = SimpleNamespace(invalidate=lambda: None)
    initial = {"content": "initial reply", "timestamp": 0}

    def send(prompt, mode):
        prompts.append((prompt, mode))

    def wait_settled(timeout, after_message=None):
        baselines.append(after_message)
        reply = next(replies)
        if reply is None:
            return None
        return {"lastAssistant": {"content": reply, "timestamp": len(baselines)}}

    handle = SimpleNamespace(
        name="schema-job", schema=SCHEMA, session=session,
        settled_data={"lastAssistant": initial}, send=send,
        _sync=SimpleNamespace(wait_settled=wait_settled),
    )
    return handle, session, prompts, baselines


def test_valid_reply_is_immediately_a_dict(monkeypatch):
    monkeypatch.delenv("PI_SUBAGENTS_SCHEMA_RETRIES", raising=False)
    handle, session, prompts, _ = fake_handle([])
    body = _dict_response(handle, '{"summary": "done"}', session)
    assert isinstance(body, dict)
    assert isinstance(body, AgentDictResponse)
    assert body["summary"] == "done"
    assert not hasattr(body, "valid")
    assert not hasattr(body, "raw")
    assert not prompts


def test_parse_and_schema_errors_are_sent_back_until_a_valid_dict(monkeypatch):
    monkeypatch.delenv("PI_SUBAGENTS_SCHEMA_RETRIES", raising=False)
    handle, session, prompts, baselines = fake_handle([
        '{"summary": 7}', '{}', '{"summary": "repaired"}',
    ])
    body = _dict_response(handle, "not JSON", session)
    assert body == {"summary": "repaired"}
    assert len(prompts) == 3
    assert all(mode == "follow_up" for _, mode in prompts)
    assert "invalid JSON" in prompts[0][0]
    assert "summary" in prompts[1][0] and "string" in prompts[1][0]
    assert "summary" in prompts[2][0]
    assert [message["timestamp"] for message in baselines] == [0, 1, 2]


def test_three_failed_repairs_raise_instead_of_exposing_raw_json(monkeypatch):
    monkeypatch.delenv("PI_SUBAGENTS_SCHEMA_RETRIES", raising=False)
    handle, session, prompts, _ = fake_handle(["bad 1", "bad 2", "bad 3"])
    with pytest.raises(SchemaValidationError, match="invalid after 4 attempts.*invalid JSON"):
        _dict_response(handle, "bad initial", session)
    assert len(prompts) == 3


def test_repair_timeout_raises(monkeypatch):
    monkeypatch.setenv("PI_SUBAGENTS_SCHEMA_RETRIES", "3")
    handle, session, prompts, _ = fake_handle([None])
    with pytest.raises(PiSubagentsTimeoutError, match="schema repair timed out"):
        _dict_response(handle, "bad", session)
    assert len(prompts) == 1


@pytest.mark.parametrize("setting, expected", [(None, 3), ("bad", 3), ("9", 3), ("-1", 0), ("0", 0)])
def test_repair_count_is_bounded(monkeypatch, setting, expected):
    if setting is None:
        monkeypatch.delenv("PI_SUBAGENTS_SCHEMA_RETRIES", raising=False)
    else:
        monkeypatch.setenv("PI_SUBAGENTS_SCHEMA_RETRIES", setting)
    assert _schema_retries() == expected


def test_legacy_result_and_pool_flags_are_removed():
    from pi_subagents import AgentPool, AgentResult
    import inspect
    assert "fail_fast" not in inspect.signature(AgentPool).parameters
    assert not hasattr(AgentResult, "ok")
    assert not hasattr(AgentResult, "unwrap")
    assert "valid" not in inspect.signature(AgentDictResponse).parameters

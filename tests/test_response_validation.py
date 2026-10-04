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

    def wait_settled(timeout, after_message=None, on_tick=None):
        baselines.append(after_message)
        reply = next(replies)
        if reply is None:
            return None
        message = reply if isinstance(reply, dict) else {"content": reply}
        return {"lastAssistant": {**message, "timestamp": len(baselines)}}

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


@pytest.mark.parametrize("content", ["partial reply", '{"summary": "done"}'])
@pytest.mark.parametrize("retries", ["0", "3"])
def test_aborted_reply_never_requests_schema_repair(monkeypatch, content, retries):
    from pi_subagents.client import PiSockSessionEnded
    monkeypatch.setenv("PI_SUBAGENTS_SCHEMA_RETRIES", retries)
    handle, session, prompts, _ = fake_handle([])
    handle.settled_data["lastAssistant"].update(
        content=content, stopReason="aborted", errorMessage="Operation aborted",
    )
    with pytest.raises(PiSockSessionEnded, match="interrupted.*Operation aborted"):
        _dict_response(handle, content, session)
    assert not prompts


def test_aborted_session_file_overrides_cached_success_metadata(tmp_path):
    import json
    from pi_subagents.client import PiSockSessionEnded
    handle, session, prompts, _ = fake_handle([])
    session_file = tmp_path / "aborted.jsonl"
    session_file.write_text(json.dumps({
        "type": "message", "message": {
            "role": "assistant", "content": [], "stopReason": "aborted",
        },
    }) + "\n")
    session.session_file = str(session_file)
    handle.settled_data["lastAssistant"]["stopReason"] = "stop"
    with pytest.raises(PiSockSessionEnded, match="interrupted"):
        _dict_response(handle, "partial", session)
    assert not prompts


def test_aborted_repair_reply_does_not_trigger_another_followup(monkeypatch):
    from pi_subagents.client import PiSockSessionEnded
    monkeypatch.setenv("PI_SUBAGENTS_SCHEMA_RETRIES", "3")
    handle, session, prompts, _ = fake_handle([
        {"content": "partial repair", "stopReason": "aborted"},
    ])
    with pytest.raises(PiSockSessionEnded, match="interrupted"):
        _dict_response(handle, "bad initial", session)
    assert len(prompts) == 1


def test_local_abort_during_validation_prevents_followup(monkeypatch):
    from pi_subagents.client import PiSockSessionEnded
    from pi_subagents import handle as handle_mod
    handle, session, prompts, _ = fake_handle([])
    def interrupted_validation(schema, value):
        handle.closed = True
        raise SchemaValidationError("invalid summary")
    monkeypatch.setattr(handle_mod, "validate_with_schema", interrupted_validation)
    with pytest.raises(PiSockSessionEnded, match="schema repair stopped"):
        _dict_response(handle, '{"summary": 7}', session)
    assert not prompts


def test_local_abort_while_waiting_for_repair_stops_wait_immediately():
    from pi_subagents.client import PiSockSessionEnded
    handle, session, prompts, _ = fake_handle([])
    def interrupted_wait(timeout, after_message=None, on_tick=None):
        handle.closed = True
        on_tick({"isIdle": True})
        raise AssertionError("interrupt must stop the wait")
    handle._sync.wait_settled = interrupted_wait
    with pytest.raises(PiSockSessionEnded, match="schema repair stopped"):
        _dict_response(handle, "bad initial", session)
    assert len(prompts) == 1


def test_provider_failure_is_not_a_schema_repair_case():
    from pi_subagents.client import PiSockTurnFailed
    handle, session, prompts, _ = fake_handle([])
    handle.settled_data["lastAssistant"].update(
        stopReason="error", errorMessage="quota exhausted",
    )
    with pytest.raises(PiSockTurnFailed, match="quota exhausted"):
        _dict_response(handle, "", session)
    assert not prompts


@pytest.mark.parametrize("state", [
    {"isIdle": False},
    {"isIdle": True, "hasPendingMessages": True},
])
def test_schema_repair_allows_provider_retry_while_work_is_pending(tmp_path, state):
    import json
    handle, session, prompts, _ = fake_handle([])
    session_file = tmp_path / "provider-retry.jsonl"
    session.session_file = str(session_file)
    def wait_during_retry(timeout, after_message=None, on_tick=None):
        def record(stop):
            session_file.write_text(json.dumps({
                "type": "message", "message": {"role": "assistant", "stopReason": stop},
            }) + "\n")
        record("error")
        on_tick(state)
        record("stop")
        return {"lastAssistant": {"content": '{"summary": "recovered"}', "timestamp": 1}}
    handle._sync.wait_settled = wait_during_retry
    assert _dict_response(handle, "bad initial", session) == {"summary": "recovered"}
    assert len(prompts) == 1

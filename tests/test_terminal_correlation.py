"""Terminal correlation with mocked socket I/O; no Pi or tmux processes."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from pi_subagents.client import AsyncSockClient, SockClient, _assistant_outcome_marker
from pi_subagents.handle import AgentHandle
from pi_subagents.schema import SchemaValidationError


def append_assistant(path, timestamp, content, stop="stop"):
    with path.open("a") as stream:
        stream.write(json.dumps({"type": "message", "message": {
            "role": "assistant", "timestamp": timestamp,
            "content": content, "stopReason": stop,
        }}) + "\n")


def terminal_clients(path, monkeypatch, wire):
    state = {"isIdle": True, "hasPendingMessages": False, "sessionFile": str(path)}
    sync, asynchronous = SockClient(""), AsyncSockClient("")
    monkeypatch.setattr(sync, "state", lambda: dict(state))
    monkeypatch.setattr(sync, "message", lambda: wire[0])
    monkeypatch.setattr(asynchronous, "state", AsyncMock(side_effect=lambda: dict(state)))
    monkeypatch.setattr(asynchronous, "message", AsyncMock(side_effect=lambda: wire[0]))
    return sync, asynchronous, state


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("filtered_wire", [False, True])
def test_textless_terminal_never_returns_intermediate_commentary(tmp_path, monkeypatch,
                                                                 asynchronous, filtered_wire):
    path = tmp_path / "terminal.jsonl"
    append_assistant(path, 1, [{"type": "text", "text": "old"}])
    baseline = _assistant_outcome_marker(str(path))
    commentary = {"content": '{"ok": true}', "timestamp": 2}
    append_assistant(path, 2, [{"type": "text", "text": commentary["content"]}], "toolUse")
    append_assistant(path, 3, [])
    wire = [None if filtered_wire else commentary]
    sync, async_client, _ = terminal_clients(path, monkeypatch, wire)
    client = async_client if asynchronous else sync
    client._after_outcome = baseline
    wait = client.wait_settled(timeout=0.1, poll=0, after_message={"content": "old", "timestamp": 1})
    result = asyncio.run(wait) if asynchronous else wait
    assert result["lastAssistant"] == {"content": "", "timestamp": 3, "stopReason": "stop"}
    assert client._after_outcome is None


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("repairs", [0, 1])
def test_textless_schema_terminal_fails_or_repairs_not_commentary(tmp_path, monkeypatch,
                                                                 asynchronous, repairs):
    path = tmp_path / "schema.jsonl"
    append_assistant(path, 2, [{"type": "text", "text": '{"ok": true}'}], "toolUse")
    append_assistant(path, 3, [])
    wire = [{"content": '{"ok": true}', "timestamp": 2}]
    sync, async_client, state = terminal_clients(path, monkeypatch, wire)
    schema = {"type": "object", "required": ["ok"], "properties": {"ok": {"type": "boolean"}}}
    handle = AgentHandle("p", name="textless", cwd=str(tmp_path), window_name="textless",
                         model=None, thinking=None, schema=schema)
    handle._sync, handle._async = sync, async_client
    handle._remember_state(state)
    sent = []

    def repair(text, mode="steer"):
        sent.append((text, mode))
        append_assistant(path, 4, [{"type": "text", "text": '{"ok": false}'}])
        wire[0] = {"content": '{"ok": false}', "timestamp": 4}
        return {"delivered": True, "mode": "direct"}

    monkeypatch.setattr(sync, "send", repair)
    monkeypatch.setenv("PI_SUBAGENTS_SCHEMA_RETRIES", str(repairs))

    def wait():
        return asyncio.run(handle.wait_async(timeout=0.2, poll=0)) if asynchronous else handle.wait(timeout=0.2, poll=0)

    if repairs:
        assert wait() == {"ok": False}
        assert len(sent) == 1 and sent[0][1] == "follow_up"
        assert handle.settled_data["lastAssistant"]["timestamp"] == 4
    else:
        with pytest.raises(SchemaValidationError):
            wait()
        assert not sent
        assert handle.last_outcome == "failed"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_direct_delivery_boundary_excludes_old_terminal(tmp_path, monkeypatch, asynchronous):
    path = tmp_path / "direct.jsonl"
    append_assistant(path, 1, [{"type": "text", "text": "previous"}])
    offset = path.stat().st_size
    # Original turn finishes after the pre-send snapshot but before acceptance.
    append_assistant(path, 2, [{"type": "text", "text": "old response"}])
    wire = [{"content": "old response", "timestamp": 2}]
    sync, async_client, _ = terminal_clients(path, monkeypatch, wire)
    client = async_client if asynchronous else sync
    client._after_deliveries = ((offset, "next"),)

    def wait():
        call = client.wait_settled(timeout=0, poll=0)
        return asyncio.run(call) if asynchronous else call

    assert wait() is None
    with path.open("a") as stream:
        stream.write(json.dumps({"type": "custom_message", "customType": "session-message", "content": "next"}) + "\n")
    assert wait() is None, "acceptance alone cannot settle the new turn"
    append_assistant(path, 3, [])
    assert wait()["lastAssistant"] == {"content": "", "timestamp": 3, "stopReason": "stop"}
    assert client._after_deliveries is None

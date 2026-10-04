"""Event-gated lifecycle coverage; never starts pi or tmux processes.

Import the fake-runtime fixture explicitly so the process boundary is shared
without depending on pytest collection order. Readiness tests use bare handles.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeout
import threading

import pytest

from pi_subagents import AgentPool, AgentPoolFailureError, SessionReuseError, Task
from pi_subagents import handle as live_mod
from pi_subagents import pool as pool_mod
from pi_subagents.client import PiSockSessionEnded, PiSockTurnFailed
from pi_subagents.schema import SchemaValidationError
from test_runtime_lifecycle import eventually, pop, runtime  # noqa: F401


@pytest.fixture
def single_turn_capacity(runtime, monkeypatch):
    """Use an isolated one-turn ceiling, not the parent's active-turn budget."""
    capacity = pool_mod._GlobalCapacity()
    monkeypatch.setattr(pool_mod, "_GLOBAL_CAPACITY", capacity)
    monkeypatch.setattr(pool_mod, "max_concurrent", lambda: 1)
    return capacity


def background_call(call):
    """Keep a broken scheduling lock from hanging executor shutdown forever."""
    future = Future()

    def run():
        try:
            future.set_result(call())
        except BaseException as error:
            future.set_exception(error)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return future, thread


def completion(pool):
    try:
        return pop(pool)
    except AgentPoolFailureError as failure:
        return failure.result


def repair_commands(runtime):
    return [command for _, command in runtime.commands
            if command.get("type") == "send" and command.get("mode") == "follow_up"]


@pytest.mark.parametrize("repair,expected,count", [
    ("valid", None, 1), ("invalid", SchemaValidationError, 2),
    ("error", PiSockTurnFailed, 1), ("aborted", PiSockSessionEnded, 1),
])
def test_repaired_or_unsuccessful_schema_turn_controls_unloading(runtime, monkeypatch, repair, expected, count):
    monkeypatch.setenv("PI_SUBAGENTS_SCHEMA_RETRIES", "2")
    original_turn = runtime.turn

    def schema_turn(session, prompt):
        if prompt == "schema-invalid":
            original_turn(session, prompt)
        else:
            text = '{"ok": true}' if repair == "valid" else "invalid repair"
            stop = repair if repair in ("error", "aborted") else "stop"
            runtime.record_reply(session, text, stop=stop)

    monkeypatch.setattr(runtime, "turn", schema_turn)
    with AgentPool(concurrency=1) as pool:
        stage = pool.stage("schema", slots=1)
        handle = stage.submit(Task("schema-invalid", schema={
            "type": "object", "required": ["ok"],
            "properties": {"ok": {"type": "boolean"}},
        }))
        result = completion(pool)
        assert len(repair_commands(runtime)) == count
        assert stage.snapshot()["submitted"] == 1, "repairs are not new scheduled tasks"
        if expected is None:
            assert result.body == {"ok": True} and result.error is None
            assert result.status == "settled" and handle._live.last_outcome == "ok"
            assert handle._live.dormant and not runtime.alive(handle._live.window_id)
            assert len(runtime.killed) == 1
            assert result.session.prose.endswith('{"ok": true}')
        else:
            assert result.status == "failed" and isinstance(result.error, expected)
            assert handle._live.last_outcome != "ok"
            assert runtime.alive(handle._live.window_id) and runtime.killed == []
        assert pool.snapshot()["running"] == 0
        assert pop(pool) is None, "repair outcome must be accounted for only once"


def test_cancel_during_schema_repair_retains_window_without_extra_repairs(runtime, monkeypatch):
    monkeypatch.setenv("PI_SUBAGENTS_SCHEMA_RETRIES", "3")
    started = threading.Event()
    runtime.gates["repair"] = threading.Event()
    original_turn = runtime.turn

    def schema_turn(session, prompt):
        if prompt in ("schema-invalid", "aborted"):
            original_turn(session, prompt)
        else:
            runtime.record_reply(session, "partial repair", gate=runtime.gates["repair"])
            started.set()

    monkeypatch.setattr(runtime, "turn", schema_turn)
    with AgentPool(concurrency=1) as pool:
        stage = pool.stage("schema", slots=1)
        handle = stage.submit(Task("schema-invalid", schema={"type": "object"}))
        assert started.wait(3), "expected exactly one in-flight repair"
        assert handle.cancel()
        result = completion(pool)
        assert result.status == "cancelled" and result.handle is handle
        assert len(repair_commands(runtime)) == 1
        assert runtime.alive(handle._live.window_id) and runtime.killed == []
        assert stage.snapshot()["cancelled"] == 1
        assert pool.snapshot()["running"] == 0 and pop(pool) is None


@pytest.mark.parametrize("initial", ["provider-error", "aborted"])
def test_recovery_ignores_stale_failure_until_delayed_turn_starts(runtime, monkeypatch, initial):
    with AgentPool(concurrency=1) as pool:
        stage = pool.stage("work", slots=1)
        source = stage.submit(Task(initial, metadata={"rounds": 2}))
        first = completion(pool)
        assert first.status == "failed"
        live = source._live
        window = runtime.windows[live.window_id]
        original_turn = runtime.turn
        idle_seen, start_turn, busy_seen, finish_turn = (threading.Event() for _ in range(4))
        wait_settled = live._sync.wait_settled

        def delayed_turn(session, prompt):
            if prompt != "continue":
                original_turn(session, prompt)
            # pi accepted the continuation but still reports the old failure.

        def traced_wait(*args, **kwargs):
            tick = kwargs.get("on_tick")

            def on_tick(state):
                if state.get("isIdle"):
                    idle_seen.set()
                    assert start_turn.wait(3), "test did not release delayed start"
                    window.session.gate = threading.Event()
                else:
                    busy_seen.set()
                    assert finish_turn.wait(3), "test did not release delayed finish"
                if tick is not None:
                    tick(state)

            kwargs["on_tick"] = on_tick
            return wait_settled(*args, **kwargs)

        monkeypatch.setattr(runtime, "turn", delayed_turn)
        monkeypatch.setattr(live._sync, "wait_settled", traced_wait)
        receipt = source.send("continue")
        followup = receipt["handle"]
        try:
            assert idle_seen.wait(3), "stale idle failure must not fail/settle the continuation"
            assert followup.result is None and source.result is first
            assert runtime.alive(live.window_id) and len(runtime.spawns) == 1
            start_turn.set()
            assert busy_seen.wait(3), "new turn must be allowed to start over a stale failure"
            assert followup.result is None
            original_turn(window.session, "continue")
        finally:
            start_turn.set()
            finish_turn.set()
        second = pop(pool)
        assert second.handle is followup and second.parent is first
        assert second.body == "reply:continue" and second.status == "settled"
        assert source.result is first and first.status == "failed"
        assert first.session.session_file == second.session.session_file
        assert second.task.metadata["rounds"] == 2
        assert live.last_outcome == "ok" and live.dormant
        assert len(runtime.spawns) == 1 and len(runtime.killed) == 1
        assert pop(pool) is None


@pytest.mark.parametrize("phase", ["before-bind", "after-validation"])
def test_cancel_at_completion_boundaries_never_unloads_or_double_accounts(runtime, monkeypatch, phase):
    reached, release = threading.Event(), threading.Event()
    if phase == "before-bind":
        spawn = runtime.spawn

        def blocked_spawn(*args, **kwargs):
            ref = spawn(*args, **kwargs)
            reached.set()
            assert release.wait(3), "test did not release startup"
            return ref

        monkeypatch.setattr(live_mod, "spawn_pi_window", blocked_spawn)
    else:
        wait = live_mod.AgentHandle.wait

        def blocked_wait(self, *args, **kwargs):
            body = wait(self, *args, **kwargs)
            reached.set()
            assert release.wait(3), "test did not release validated response"
            return body

        monkeypatch.setattr(live_mod.AgentHandle, "wait", blocked_wait)
    with AgentPool(concurrency=1) as pool, ThreadPoolExecutor(max_workers=1) as callers:
        stage = pool.stage("work", slots=1)
        handle = stage.submit(Task("success"))
        try:
            assert reached.wait(3)
            assert callers.submit(handle.cancel).result(timeout=2)
        finally:
            release.set()
        result = completion(pool)
        assert result.handle is handle and result.status == "cancelled"
        assert not handle.cancel()
        assert runtime.alive(handle._live.window_id) and runtime.killed == []
        assert stage.snapshot()["cancelled"] == 1
        assert stage.snapshot()["settled"] == stage.snapshot()["failed"] == 0
        assert pool.snapshot()["running"] == 0 and pop(pool) is None


@pytest.mark.parametrize("asynchronous", [False, True])
def test_slow_active_send_does_not_lock_out_unrelated_scheduling(runtime, monkeypatch, asynchronous):
    runtime.gates["blocked"] = threading.Event()
    entered, release, unrelated_done = (threading.Event() for _ in range(3))
    with AgentPool(concurrency=2) as pool, ThreadPoolExecutor(max_workers=2) as callers:
        stage = pool.stage("work", slots=2)
        source = stage.submit(Task("blocked"))
        eventually(lambda: source.status == "running" and any(
            command.get("text") == "blocked" for _, command in runtime.commands))
        send = source._live.send

        def slow_send(*args, **kwargs):
            entered.set()
            assert release.wait(5), "test did not release slow socket send"
            return send(*args, **kwargs)

        monkeypatch.setattr(source._live, "send", slow_send)

        def steer():
            return asyncio.run(source.send_async("steer")) if asynchronous else source.send("steer")

        def unrelated():
            handle = stage.submit(Task("unrelated"))
            result = handle.wait(timeout=3)
            unrelated_done.set()
            return result

        sender = callers.submit(steer)
        try:
            assert entered.wait(3)
            other = callers.submit(unrelated)
            assert unrelated_done.wait(3), "socket I/O must not hold the pool scheduler lock"
            assert not sender.done(), "unrelated completion must occur before slow send returns"
        finally:
            release.set()
        try:
            assert sender.result(timeout=3)["delivered"]
        finally:
            # Keep this turn active until the steer returns; otherwise the
            # fake could unload it first and manufacture a send/settle race.
            runtime.gates["blocked"].set()
        other_result = other.result(timeout=3)
        results = [pop(pool), pop(pool)]
        assert any(result is other_result for result in results)
        assert {result.handle for result in results} == {source, other_result.handle}
        assert stage.snapshot()["submitted"] == 2 and pop(pool) is None


def test_send_after_validated_wait_owns_a_continuation(runtime, monkeypatch, single_turn_capacity):
    validated, publish, sending = (threading.Event() for _ in range(3))
    runtime.gates["boundary-followup"] = threading.Event()
    wait = live_mod.AgentHandle.wait

    def before_publication(self, *args, **kwargs):
        body = wait(self, *args, **kwargs)
        if self.prompt == "boundary-first":
            validated.set()
            assert publish.wait(3), "test did not release terminal publication"
        return body

    monkeypatch.setattr(live_mod.AgentHandle, "wait", before_publication)
    with AgentPool(concurrency=1) as pool:
        stage = pool.stage("work", slots=1)
        source = stage.submit(Task("boundary-first", metadata={"rounds": 2}))
        require_live = source._require_live

        def observed_require_live(operation):
            live = require_live(operation)
            if operation == "send":
                sending.set()
            return live

        monkeypatch.setattr(source, "_require_live", observed_require_live)

        def send_at_boundary():
            return source.send("boundary-followup")

        try:
            assert validated.wait(3), "old response must be validated before send"
            assert source.result is None and source._live.last_outcome == "ok"
            sender, sender_thread = background_call(send_at_boundary)
            assert sending.wait(3)
            # A repaired implementation may wait for publication. An unsafe
            # active-branch acknowledgement returns here before we release it.
            try:
                sender.result(timeout=1)
            except FutureTimeout:
                pass
        finally:
            publish.set()
        try:
            receipt = sender.result(timeout=3)
            sender_thread.join(timeout=3)
            assert not sender_thread.is_alive()
            assert receipt.get("scheduled"), "an idle Pi turn needs a new owning handle"
            followup = receipt["handle"]
            assert receipt["handleId"] == followup.id and followup is not source
            first = pop(pool)
            assert first.handle is source and first.body == "reply:boundary-first"
            eventually(lambda: followup.status == "running" and any(
                command.get("text") == "boundary-followup" for _, command in runtime.commands))
            assert followup._live is source._live and followup.result is None
            assert pool.snapshot()["running"] == 1 and stage.snapshot()["submitted"] == 2
            with single_turn_capacity._condition:
                assert single_turn_capacity._active == 1, "the new turn must retain capacity"
        finally:
            runtime.gates["boundary-followup"].set()
        second = pop(pool)
        assert second.handle is followup and second.parent is first
        assert second.body == "reply:boundary-followup" and second.status == "settled"
        assert source.result is first and first.body == "reply:boundary-first"
        assert second.task.metadata["rounds"] == 2
        assert first.session.session_file == second.session.session_file
        assert stage.snapshot()["settled"] == 2 and pop(pool) is None
        eventually(lambda: single_turn_capacity._active == 0)


@pytest.mark.parametrize("continuation", [False, True], ids=["root", "continuation"])
def test_cancel_while_awaiting_global_capacity_never_delivers(runtime, monkeypatch, single_turn_capacity, continuation):
    capacity = single_turn_capacity
    waiting = threading.Event()
    runtime.gates["capacity-holder"] = threading.Event()
    with AgentPool(concurrency=1) as owner, AgentPool(concurrency=1) as other:
        stage = owner.stage("work", slots=1)
        source = None
        if continuation:
            source = stage.submit(Task("success"))
            first = pop(owner)
        holder = other.stage("busy", slots=1).submit(Task("capacity-holder"))
        eventually(lambda: holder.status == "running" and any(
            command.get("text") == "capacity-holder" for _, command in runtime.commands))
        condition_wait = capacity._condition.wait

        def observe_capacity_wait(timeout=None):
            if threading.current_thread() in owner._workers:
                waiting.set()
            return condition_wait(timeout)

        monkeypatch.setattr(capacity._condition, "wait", observe_capacity_wait)
        discarded = (source.send("discarded-prompt")["handle"] if continuation
                     else stage.submit(Task("discarded-prompt")))
        replacement_error = None
        try:
            assert waiting.wait(3), "target must reach the full global ceiling"
            # Taking the condition proves the worker released it to wait.
            with capacity._condition:
                assert capacity._active == 1 and discarded.status == "starting"
            assert discarded.result is None
            spawns_before_cancel = len(runtime.spawns)
            assert discarded.cancel()
            immediate_result = discarded.result
            with owner._condition:
                reservation_released = source is None or id(source._live) not in owner._reserved_sessions
            if continuation:
                try:
                    replacement = source.send("replacement")["handle"]
                except SessionReuseError as error:
                    # Drain the worker even on the buggy reservation path, so
                    # late prompt delivery is observed rather than hidden.
                    replacement_error = error
                    replacement = stage.submit(Task("replacement"))
            else:
                replacement = stage.submit(Task("replacement"))
            if continuation and replacement_error is None:
                with pytest.raises(SessionReuseError):
                    source.send("overlapping-replacement")
                assert source._live.dormant, "cancellation must not wake the reserved session"
            assert len(runtime.spawns) == spawns_before_cancel
        finally:
            runtime.gates["capacity-holder"].set()
        assert pop(other).handle is holder
        results = [completion(owner), completion(owner)]
        cancelled = next(result for result in results if result.handle is discarded)
        healthy = next(result for result in results if result.handle is replacement)
        assert not any(command.get("text") == "discarded-prompt"
                       for _, command in runtime.commands), "cancelled work must never deliver"
        assert not any(prompt == "discarded-prompt" for prompt, _ in runtime.spawns)
        assert cancelled is immediate_result and cancelled.status == "cancelled"
        assert reservation_released and replacement_error is None
        assert healthy.body == "reply:replacement" and healthy.status == "settled"
        if continuation:
            assert healthy.parent is first and replacement._live is source._live
            assert source.result is first and first.body == "reply:success"
        assert stage.snapshot()["cancelled"] == 1
        assert owner.snapshot()["running"] == 0 and pop(owner) is None and pop(other) is None
        eventually(lambda: capacity._active == 0)


def test_closed_pool_cannot_acquire_free_global_capacity(runtime, monkeypatch, single_turn_capacity):
    capacity = single_turn_capacity
    taken, admit, finished = (threading.Event() for _ in range(3))
    observations = []
    acquire = capacity.acquire

    def before_admission(pool, handle):
        taken.set()
        assert admit.wait(3), "test did not release capacity admission"
        admitted = acquire(pool, handle)
        with capacity._condition:
            observations.append((admitted, capacity._active))
        finished.set()
        return admitted

    monkeypatch.setattr(capacity, "acquire", before_admission)
    pool = AgentPool(concurrency=1)
    try:
        handle = pool.stage("work", slots=1).submit(Task("never-start"))
        assert taken.wait(3) and handle.status == "starting"
        with capacity._condition:
            assert capacity._active == 0, "this is the free-capacity admission path"
        closer, closer_thread = background_call(pool.close)
        eventually(lambda: pool.closed)
    finally:
        admit.set()
    assert finished.wait(3)
    closer.result(timeout=3)
    closer_thread.join(timeout=3)
    assert not closer_thread.is_alive()
    assert observations == [(False, 0)], "closure must be checked before acquiring free capacity"
    assert all(not worker.is_alive() for worker in pool._workers)
    assert runtime.spawns == [] and runtime.commands == [] and runtime.killed == []


@pytest.mark.parametrize("interruption", ["cancel", "timeout"])
def test_shared_readiness_survives_cancelled_or_timedout_waiter(tmp_path, interruption):
    handle = live_mod.AgentHandle("pending", name="ready", cwd=str(tmp_path),
                                 window_name="ready", model=None, thinking=None, schema=None)
    handle._startup_started = True

    async def run():
        survivor = asyncio.create_task(handle._await_ready_async(timeout=3))
        await asyncio.sleep(0)  # register the survivor on the shared future
        if interruption == "cancel":
            waiter = asyncio.create_task(handle._await_ready_async(timeout=3))
            await asyncio.sleep(0)  # register before cancelling just this waiter
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
        else:
            with pytest.raises(asyncio.TimeoutError):
                await handle._await_ready_async(timeout=0)
        await asyncio.sleep(0)  # let cancellation propagation callbacks run
        assert not handle._ready.cancelled(), "one observer must not cancel shared startup"
        assert not survivor.done()
        handle._ready.set_result(None)
        await survivor
        await handle._await_ready_async(timeout=1)

    asyncio.run(run())
    handle._await_ready(timeout=1)


@pytest.mark.parametrize("interruption", ["cancel", "timeout"])
def test_task_completion_future_survives_interrupted_async_waiter(runtime, interruption):
    runtime.gates["blocked"] = threading.Event()
    with AgentPool(concurrency=1) as pool:
        source = pool.stage("work", slots=1).submit(Task("blocked"))
        eventually(lambda: source.status == "running" and any(
            command.get("text") == "blocked" for _, command in runtime.commands))

        async def run():
            if interruption == "cancel":
                waiter = asyncio.create_task(source.wait_async(timeout=3))
                await asyncio.sleep(0)
                waiter.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await waiter
            else:
                with pytest.raises(asyncio.TimeoutError):
                    await source.wait_async(timeout=0)
            assert not source._future.cancelled() and source.result is None
            runtime.gates["blocked"].set()
            return await source.wait_async(timeout=3)

        waited = asyncio.run(run())
        assert waited is pop(pool) and waited.status == "settled"
        assert pop(pool) is None, "observer interruption must not lose or duplicate completion"

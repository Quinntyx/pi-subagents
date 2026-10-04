# pi-subagents

> [!NOTE]
> **Upstream:** the canonical, community-facing home of this project is [github.com/Quinntyx/pi-subagents](https://github.com/Quinntyx/pi-subagents). This git.quinntyx.dev copy is the author's development fork — day-to-day churn lands here and is PR'd to GitHub on release. Install instructions below point at GitHub.

Pool-based orchestration of standalone pi instances from Python.

Subagents are real, interactive pi processes running in their own tmux windows
(running under your own agent dir by default — see `PI_CODING_SUBAGENT_DIR`
below), controlled over the pi-sock unix
socket. The library works both inside a PTC session (state is forwarded to the
PTC runtime as `subagent_state` frames for live UI) and from plain scripts run
directly in a tmux pane (state is printed as one-line status updates).

## The model

Everything runs through an `AgentPool`: a bounded scheduler that owns stage
queues, worker slots, and every pi process it starts.

```python
import pi_subagents as subagents

pool = subagents.AgentPool(concurrency=8, name="features")
build = pool.stage("build", slots=4)
review = pool.stage("review", slots=4)

build.submit_all(
    subagents.Task(f"Implement {feature}", name=f"build-{feature}",
                   metadata={"feature": feature})
    for feature in FEATURES
)

while (result := await pool.pop()) is not None:
    if result.stage is build:
        review.submit(
            subagents.Task(f"Review this implementation:\n\n{result.body}",
                           schema=REVIEW_SCHEMA),
            parent=result,
        )
    elif result.stage is review:
        verdict = result.body
        rounds = result.task.metadata["rounds"]
        if not verdict["passed"] and rounds < 5:
            build.submit(
                subagents.Task(f"Address this feedback:\n\n{verdict['feedback']}",
                               name=f"fix-{result.task.metadata['feature']}",
                               metadata={"rounds": rounds + 1}),
                parent=result,
            )

pool.close()  # destroy retained windows and invalidate handles
```

## API

### `Task(prompt, *, name=None, model=None, thinking=None, schema=None, cwd=None, profile=None, timeout=None, metadata=None)`

Immutable description of one scheduled subagent turn. `metadata` is copied and
exposed read-only; every task carries a non-negative integer `metadata["rounds"]`
(roots default to 0). `timeout` bounds the turn's settle wait.

### `AgentPool(concurrency=None, *, name=None)`

Hard cap on concurrently running subagents (default and ceiling:
`PI_SUBAGENTS_MAX_CONCURRENT`, 8). The pool runs its own worker threads, so work
keeps flowing even when the orchestrating PTC chunk ends between turns.
- `pool.stage(name, *, slots)` — create a queue (names are unique; the sum of
  slots may not exceed the pool's concurrency).
- `await pool.pop(timeout=None)` — next completion in pool-observed completion
  order; `None` means *currently quiescent* (queues and running work empty) and
  the pool stays usable. A deadline raises `AgentPoolTimeoutError` (carrying
  `pool` and `snapshot`) without touching the work.
- `pool.handles(status=None)` — snapshot of submitted handles.
- `pool.snapshot()` — counters plus per-stage rows.
- `pool.close()` — the only teardown: rejects submissions, cancels queued work,
  aborts running turns, kills every tmux window (settled sessions included),
  invalidates all handles, and wakes blocked `pop()` calls with
  `PoolClosedError`. Idempotent.

### `AgentStage`

`stage.submit(task, *, parent=None, session_handle=None, session_name=None)` —
enqueue and return an `AgentHandle` immediately; spawning happens when a worker
slot frees. `submit_all(tasks)` fans out. Each stage has `slots` **soft**
priority reservations: idle slots are borrowed by other stages (non-preemptive —
borrowed runs finish before slots revert). With no stages defined the pool
still schedules fairly across submissions to any stage.

### `AgentHandle`

One submission; valid while queued (steer/inspect only after dispatch).
`await handle` (or `handle.wait(timeout)`) resolves to that task's `AgentResult`.
`handle.cancel()` cancels queued work or aborts the running turn.
`send(text, mode="steer")` steers the currently running turn; new turns go
through `stage.submit()`.

### `AgentResult`

Successful completion: `task`, `stage`, `handle`, `body`, `status`,
`duration_ms`, and `parent`. Text-task bodies are native strings; schema-task
bodies are validated Python dictionaries (`AgentDictResponse`), not JSON text.
Use `result.body` directly; there is no `result.ok` or `.unwrap()` step.
`result.session` exposes the parsed transcript.

Failures raise by default, with no `fail_fast` option. Handle waits raise the
underlying exception; `pool.pop()` raises `AgentPoolFailureError`, whose
`.result` identifies the failed task and whose cause preserves the original
exception. Neither closes the pool: inspect it and continue from a later
notebook cell. Catch expected failures around individual agents/completions;
avoid a blanket handler around the entire workflow.

Invalid schema output is parsed/validated internally and repaired using the
actual error in up to three follow-up prompts. Aborted turns and terminal
provider failures stop schema repair instead of requesting another JSON reply.
Exhaustion raises `SchemaValidationError`; a repair timeout raises
`PiSubagentsTimeoutError`.
The library never returns an invalid/raw JSON envelope as a successful schema
body. Pop timeouts raise `AgentPoolTimeoutError` with a pool snapshot. Explicit
cancellation is a separate `cancelled` outcome, not a successful response.

### Session reuse

`session_handle=result.handle` re-runs a settled pi session as a follow-up turn
in the same tmux window instead of spawning a new one. The reused turn gets a
new handle and result; the old ones stay settled and immutable. Omitted
model/thinking/cwd/profile inherit the session's configuration; explicit
conflicts raise `SessionReuseError`. A session processes one turn at a time
(double-booking raises). `session_name="new-name"` renames the live pi session
(pi updates its terminal title immediately). Settlement is correlated against
the pre-follow-up message, so a reused session can never return a stale reply.

### Cyclic workflows

Reuse `parent=result` to propagate workflow identity. Only override what
changes (typically `rounds`); everything else is inherited. Gate every cycle on
a round limit — `pop()` returns `None` only when nothing is queued or running,
so an ungated build→review→fix cycle runs forever.

## Choosing a model or effort level

Each subagent runs with the agent dir resolved above, so it can reach **every** model its pi install knows, so **never
guess a slug and never refuse a named model** — look it up:

```python
caps = subagents.capabilities()          # counts, providers, defaults, caps
subagents.best_model_match("flash")      # one pick: .slug, .context, .thinking, .images
subagents.model_slugs("astra")           # every matching slug
subagents.resolve_models("opus")         # full rows (context, thinking, images)
```

`best_model_match` prefers an exact slug, then the profile's default provider,
then first-party entries over proxied ones; it returns `None` when nothing
genuinely matches. Pass a full `provider/model` slug when a specific provider's
variant matters. The catalog is re-read every couple of minutes
(`PI_SUBAGENTS_CATALOG_TTL`), a lookup that misses is re-checked live, and
`list_models(refresh=True)` bypasses the cache.

Prompts have no size limit: they are delivered to the subagent over pi-sock,
not through the tmux command.

## Identifying subagents in the tmux overview

Every subagent window is launched with `pi --name "(subagent) <name>"`, so the
terminal title tmux's overview shows reads `π - (subagent) build-auth - Vault`
against a main agent's `π - Vault`. The tmux window keeps the short agent name.

## Environment knobs

- `PI_SUBAGENTS_MAX_CONCURRENT` (8) — global ceiling shared by all pools.
- `PI_SUBAGENTS_SETTLE_TIMEOUT` (30 min) — default per-turn settle wait.
- `PI_SUBAGENTS_STARTUP_TIMEOUT` (90 s) — pi-sock readiness + first delivery.
- `PI_SUBAGENTS_SCHEMA_RETRIES` (3) — repair follow-ups after invalid structured output (clamped to 0–3); exhaustion raises `SchemaValidationError`.
- `PI_SUBAGENTS_CATALOG_TTL` (120 s) — model-catalog cache lifetime.
- `PI_CODING_SUBAGENT_DIR` — the agent dir spawned subagents run under. Unset, subagents share the orchestrator's own agent dir (`PI_CODING_AGENT_DIR` or `~/.pi/agent`) — zero setup. Point it at any directory with a pi config (including a pi-profiles-managed profile) for a separate subagent environment.

## Rules

- Top-level `await` is available inside `python_exec` chunks — `await handle`,
  never `asyncio.run(...)`.
- Spawned agents cannot spawn agents (depth 1; `import pi_subagents` raises
  there).
- Statuses: `queued → starting → running → settled`, or `failed`
  (startup/provider error), `cancelled`, `closed` (pool closed). An errored
  provider turn may keep the turn waiting until `Task.timeout` fires — set
  explicit timeouts for tasks whose models can fail.
- The idle timeout of a PTC chunk (`PTC_EXECUTION_TIMEOUT_MS`) is re-armed by
  every registry tick, so a pool workflow may run for hours; interrupting the
  chunk (Esc / `/ptc interrupt`) stops the chunk, not the pool — re-`pop()` in
  the next chunk to resume consuming.
- `subagents.finish()` closes every live pool; `subagents.stop_all()` cancels
  queued/running work but leaves pools open.

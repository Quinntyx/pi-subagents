# pi-subagents

Spawn, monitor, and steer **standalone, interactive pi instances** in tmux windows — the Python
half of pi's dynamic subagent workflows. Subagents are real pi sessions in their own tmux
windows (under the dedicated `subagents` pi profile), controlled over the
[pi-sock](https://git.quinntyx.dev/quinntyx/pi-sock) unix socket, so you can watch them work
in the TUI and intervene at any time.

## Install

```sh
# systemwide (the PTC venv — makes `import pi_subagents` available to PTC sessions)
uv pip install --editable ~/docs/src/pi-subagents/main
```

## Environment contract (checked at import)

- `PI_SUBAGENT_DEPTH` set → `import pi_subagents` raises `NotImplementedError`.
  Spawned agents cannot spawn further agents (the tree stays height 2 until the
  `PI_PTC_PRIMARY` reporting mesh lands).
- Not inside tmux (`$TMUX` missing or server unreachable) → the module imports with a
  warning; every API call raises `NotImplementedError`. A script run outside tmux still
  runs to completion — it just can't spawn agents.

## API

```python
import pi_subagents as subagents

# spawn — returns a live AgentHandle immediately
handle = subagents.agent("Analyze the failing tests in tests/ and report root causes",
                         name="test-digger")

# per-subagent model / thinking level (passed to the spawned pi as --model/--thinking)
reviewer = subagents.agent("Review the diff for correctness", name="reviewer",
                           model="openai-codex/gpt-6-astra", thinking="high")
cheap = subagents.agent("Count TODO comments in src/", name="counter",
                        model="deepseek-router/deepseek-v4.1-flash", thinking="low")

# the agent is a real pi instance in a tmux window: watch it there, or steer it
await handle.send_async("focus on the auth module first", mode="steer")
await handle.abort_async()          # "pause": stops the current run

# wait for the settled result (str when no schema, dict when schema given)
resp = await handle                                  # AgentStrResponse(str)
resp = await subagents.agent("classify tests", name="c",
                             schema={"type": "object",
                                     "required": ["root_causes"],
                                     "properties": {"root_causes": {"type": "array",
                                                                    "items": {"type": "string"}}}})
resp["root_causes"]                                  # AgentDictResponse(dict)

# both response types carry the AgentSession (the pi session JSONL on disk)
session = resp.get_session()
session.tool_calls    # [{tool, arguments, durationMs, isError, result_preview}]
session.thinking      # concatenated thinking blocks
session.prose         # assistant text
session.trajectory()  # ordered event stream
session.session_file  # path to the pi session JSONL

# resume after an abort: re-send a prompt on the same pi session — it continues
# from where it left off (pi sessions persist on disk)
handle.resume("continue")            # sync — re-opens the handle, returns it
await handle                          # wait again

# fan-out helpers
responses = await subagents.wait_all_async([h1, h2, h3], timeout=600)
subagents.stop_all()
```

### Choosing a model (`model=` / `thinking=`)

```python
caps = subagents.capabilities()        # defaults, providers, thinking levels, caps; no catalog dump
subagents.thinking_levels()            # ["off", "minimal", "low", "medium", "high", "xhigh", "max"]
subagents.scoped_models()              # the subagent profile's picker set (not a limit on agent(model=))
subagents.model_slugs("astra")         # every matching `provider/model` slug
subagents.resolve_models("opus")       # matching ModelInfo rows (context, max output, thinking, images)
subagents.best_model_match("astra")    # one pick: .slug / .context / .thinking / .images
```

The catalog is read from `pi --list-models` as seen by the *subagent* profile, so
it is exactly what a spawned instance can resolve (610 models across 9 providers
on the author's machine). `best_model_match` prefers an exact slug, then the
profile's own default provider, then first-party entries over proxied ones;
it returns `None` when nothing genuinely matches (the CLI search is fuzzy and
returns unrelated neighbours, which the helper filters out).

### Sync vs async

Plain scripts use the sync surface: `handle.wait(timeout=...)`, `handle.send(text)`,
`handle.resume(prompt)`, `subagents.wait_all(handles)`. Inside PTC sessions (async
user code) prefer `wait_async` / `send_async` / `resume_async` / `wait_all_async`.
Calling a blocking `wait*` from a running event loop raises `ValueError("await on
closed handle")`-style guards appropriately — actually it raises
`PiSubagentsError` directing you to the async variant.

### Semantics

- **`await handle` never implicitly resumes an aborted handle** — it raises
  `ValueError: await on closed handle`. Resuming is explicit: `resume()`/`resume_async()`.
- **Settle detection** subscribes to pi-sock's `agent_settled` semantics via polling
  (`get_state` + `get_message`), so retries/compaction/queued follow-ups are drained
  before the response resolves.
- **State emission**: inside a PTC session, every mutation emits a `subagent_state`
  snapshot through the PTC bridge (`builtins.PTC_STATE_EMIT`), which the
  [pi-ptc-next](https://github.com/Quinntyx/pi-ptc-next) extension renders as a live
  subagent panel. Standalone scripts print one-line status updates to stdout.
  Activity detail (tool calls, thinking time, phase/label) comes from pi-sock's
  `get_activity` command when pi-tool-tree is installed in the subagent's pi session.
- **Caps**: `PI_SUBAGENTS_MAX_CONCURRENT` (default 8) refuses spawns beyond the limit.

### Environment variables

| Variable | Meaning |
| --- | --- |
| `PI_SUBAGENTS_PROFILE` | pi profile directory subagents launch with (default `~/.config/pi/profiles/subagents`) |
| `PI_SUBAGENT_DEPTH` | nesting depth (set by the spawner on children; locks spawning) |
| `PI_SUBAGENTS_MAX_CONCURRENT` | concurrent-subagent cap (default 8) |
| `PI_SUBAGENTS_SETTLE_TIMEOUT` | default settle timeout in seconds (default 1800) |
| `PI_SUBAGENTS_SCHEMA_RETRIES` | structured-output retry attempts (default 3) |
| `PI_PTC_PRIMARY` | reserved for the future primary-session usage reporting mesh |

## CLI

```sh
python -m pi_subagents list            # all subagent sockets + state
python -m pi_subagents abort <name>    # abort a running subagent
python -m pi_subagents prune [name]    # unlink stale sockets from crashed pi processes
```

## Example standalone script

```python
#!/usr/bin/env python3
"""Run from a tmux pane (not from pi) to see subagents work in real time."""
import pi_subagents as subagents

handles = [
    subagents.agent("Audit every file under src/ for TODO comments; return a list.", name="todo-sweep"),
    subagents.agent("Run the test suite and summarize any failures.", name="test-runner"),
]
for h, resp in zip(handles, subagents.wait_all(handles, timeout=900)):
    print(f"=== {h.name} ({h.session.duration_ms} ms, {h.session.turns} turns)")
    print(resp)
```

## Development

```sh
python3 -m pytest tests/ -q
```

"""Strict opt-in recursion policy and a root-wide, fail-fast admission broker.

Reservations count *live windows*, including startup and retained failures. This
module never probes or terminates windows and never reclaims a reservation based
on a Python launcher's lifetime. Root max tasks counts window admissions (including
reopens), not turns performed in an already admitted window. ``begin_close`` is
the atomic barrier required before enumerating and terminating descendants. The lifecycle owner must confirm termination
(or confirm that startup created no window) before calling ``release``.

The persistent lock inode is separate from the atomically replaced JSON file.
Neither is removed when a budget becomes empty: all kernels beneath the same
actual Pi process must continue to share its task count and deadline.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import re
import stat
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class RecursionPolicyError(ValueError):
    """An invalid or inconsistent recursion environment."""


class RootBudgetError(RuntimeError):
    """Admission was refused, or the shared root identity/state is unsafe."""


_DEPTH = "PI_SUBAGENT_DEPTH"
_MAX_DEPTH = "PI_SUBAGENTS_MAX_DEPTH"
_STATE = "PI_SUBAGENTS_ROOT_STATE"
_ROOT_ID = "PI_SUBAGENTS_ROOT_ID"
_PARENT = "PI_SUBAGENTS_PARENT_TOKEN"
_POLICY_DEFAULTS = {
    _MAX_DEPTH: 1,
    "PI_SUBAGENTS_MAX_CONCURRENT": 8,
    "PI_SUBAGENTS_ROOT_MAX_TASKS": 512,
}
_UINT = re.compile(r"[0-9]+\Z")
_WINDOW = re.compile(r"@[0-9]+\Z")
_TOKEN = re.compile(r"[0-9a-f]{32}\Z")
_TIMEOUT = re.compile(r"[0-9]+(?:\.[0-9]+)?\Z")


def _integer(name: str, default: int, *, minimum: int) -> int:
    value = os.environ.get(name, str(default))
    if not _UINT.fullmatch(value):
        raise RecursionPolicyError(f"{name} must be an integer >= {minimum}; got {value!r}")
    try:
        parsed = int(value)
    except ValueError as error:
        raise RecursionPolicyError(f"{name} is not a supported integer") from error
    if parsed < minimum:
        raise RecursionPolicyError(f"{name} must be >= {minimum}; got {value!r}")
    return parsed


def current_depth() -> int:
    """Root = 0; children = 1. Reject malformed and negative values."""
    return _integer(_DEPTH, 0, minimum=0)


def max_depth() -> int:
    """Maximum child depth; 1 (the default) preserves flat spawning."""
    return _integer(_MAX_DEPTH, 1, minimum=1)


def ensure_can_spawn() -> int:
    """Return the prospective child's depth, or fail at the spawn boundary.

    Merely importing this module is always legal, including at maximum depth.
    """
    depth, limit = current_depth(), max_depth()
    if depth >= limit:
        raise NotImplementedError(
            f"pi_subagents: cannot spawn at depth {depth}: {_MAX_DEPTH}={limit} "
            "(recursion is opt-in with PI_SUBAGENTS_MAX_DEPTH>1)"
        )
    return depth + 1


def _policy() -> dict[str, int | float]:
    result: dict[str, int | float] = {
        name: _integer(name, default, minimum=1)
        for name, default in _POLICY_DEFAULTS.items()
    }
    result["PI_SUBAGENTS_ROOT_MAX_CONCURRENT"] = _integer(
        "PI_SUBAGENTS_ROOT_MAX_CONCURRENT",
        int(result["PI_SUBAGENTS_MAX_CONCURRENT"]), minimum=1,
    )
    value = os.environ.get("PI_SUBAGENTS_ROOT_TIMEOUT", "1800")
    if not _TIMEOUT.fullmatch(value):
        raise RecursionPolicyError("PI_SUBAGENTS_ROOT_TIMEOUT must be finite and positive")
    timeout = float(value)
    if not math.isfinite(timeout) or timeout <= 0:
        raise RecursionPolicyError("PI_SUBAGENTS_ROOT_TIMEOUT must be finite and positive")
    result["PI_SUBAGENTS_ROOT_TIMEOUT"] = timeout
    return result


def _process_info(pid: int) -> tuple[int, str, str]:
    """Read Linux parent PID, start ticks and comm without trusting PID alone."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        left, right = text.index("("), text.rindex(")")
        fields = text[right + 2:].split()
        if fields[0] in {"Z", "X", "x"}:
            raise ValueError("process is dead")
        return int(fields[1]), fields[19], text[left + 1:right]
    except (OSError, ValueError, IndexError) as error:
        raise RootBudgetError(f"root Pi process {pid} is missing or unreadable") from error


def _find_root_pi() -> tuple[int, str]:
    """Use the nearest actual Pi ancestor, not the transient Python kernel PID."""
    pid = os.getppid()
    seen: set[int] = set()
    while pid > 1 and pid not in seen:
        seen.add(pid)
        parent, started, comm = _process_info(pid)
        is_pi = comm in {"pi", "pi-coding-agent"}
        if not is_pi:
            try:
                argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
                # Node may retain its generic comm; match the Pi entrypoint,
                # never a prompt or an arbitrary argument mentioning Pi.
                entry = os.fsdecode(argv[1]) if len(argv) > 1 else ""
                is_pi = (
                    "/pi-coding-agent/" in entry
                    and Path(entry).name in {"cli.js", "cli.mjs", "pi"}
                ) or entry.endswith("/bin/pi")
            except OSError:
                pass
        if is_pi:
            return pid, started
        pid = parent
    raise RootBudgetError("cannot identify an actual parent Pi process for recursive admission")


def _private_directory(path: Path, *, create: bool) -> None:
    if not path.is_absolute() or ".." in path.parts:
        raise RootBudgetError("root budget directory must be an absolute, non-traversing path")
    try:
        # O_NOFOLLOW protects the final file; also reject symlinked directories
        # so inherited metadata cannot redirect reads/writes through an alias.
        for ancestor in path.parents:
            if stat.S_ISLNK(ancestor.lstat().st_mode):
                raise RootBudgetError(f"symlinked root budget directory: {ancestor}")
        if create:
            path.mkdir(mode=0o700, exist_ok=True)
        info = path.lstat()
    except OSError as error:
        raise RootBudgetError(f"cannot access private root budget directory: {path}") from error
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise RootBudgetError(f"root budget directory must be owned by this user and mode 0700: {path}")


def _state_directory() -> Path:
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if runtime:
        base = Path(runtime)
        if not base.is_absolute():
            raise RootBudgetError("XDG_RUNTIME_DIR must be absolute")
        _private_directory(base, create=False)
        directory = base / "pi-subagents"
    else:
        directory = Path(tempfile.gettempdir()) / f"pi-subagents-{os.getuid()}"
    _private_directory(directory, create=True)
    return directory


def _check_file(fd: int, path: Path) -> None:
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1:
        raise RootBudgetError(f"root budget file must be private, regular and mode 0600: {path}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _positive_int(value: object) -> bool:
    return type(value) is int and value > 0


class RootBudget:
    """Cross-process admission serialized by flock; no waiting for capacity.

    ``for_environment`` returns None only in flat mode. Inherited state is never
    recreated, and every operation rechecks the root PID's start time. Environment
    policy must match the frozen root policy; changing it cannot enlarge a root.
    """

    def __init__(self, state_path: Path, root_id: str | None, parent_token: str | None,
                 depth: int, policy: dict[str, int | float], *, identity: tuple[int, str] | None = None):
        self.state_path = state_path
        self._root_id = root_id
        self._parent_token = parent_token
        self._depth = depth
        self._policy = dict(policy)
        self._identity = identity

    @classmethod
    def for_environment(cls) -> RootBudget | None:
        depth, limit = current_depth(), max_depth()
        inherited = [name in os.environ for name in (_STATE, _ROOT_ID, _PARENT)]
        if any(inherited):
            if not all(inherited):
                raise RootBudgetError("incomplete inherited root state/id/parent token")
            if limit <= 1 or depth == 0:
                raise RootBudgetError("inherited root admission requires recursive mode and child depth")
            raw_path, root_id, parent = (os.environ[name] for name in (_STATE, _ROOT_ID, _PARENT))
            path = Path(raw_path)
            if not path.is_absolute() or not _TOKEN.fullmatch(root_id) or not _TOKEN.fullmatch(parent):
                raise RootBudgetError("invalid inherited root state path/id/parent token")
            _private_directory(path.parent, create=False)
            budget = cls(path, root_id, parent, depth, _policy())
            with budget._locked() as state:
                budget._validate(state, allow_dead=True)
                budget._identity = (state["root_pid"], state["root_starttime"])
            return budget
        if limit == 1:
            return None
        if depth != 0:
            raise RootBudgetError("recursive child is missing inherited root admission state")
        identity = _find_root_pi()
        pid, started = identity
        path = _state_directory() / f"root-{pid}-{started}.json"
        budget = cls(path, None, None, depth, _policy(), identity=identity)
        with budget._locked(create=True) as state:
            if state is None:
                now = time.monotonic()
                state = {
                    "version": 1, "root_id": uuid.uuid4().hex,
                    "root_pid": pid, "root_starttime": started,
                    "policy": budget._policy, "deadline": now + budget._policy["PI_SUBAGENTS_ROOT_TIMEOUT"],
                    "tasks": 0, "records": {}, "closing": [], "launchers": {}, "released": {},
                }
                budget._validate(state)
                budget._write(state)
            else:
                budget._validate(state)
            budget._root_id = state["root_id"]
        return budget

    @contextmanager
    def _locked(self, *, create: bool = False) -> Iterator[dict | None]:
        _private_directory(self.state_path.parent, create=False)
        lock_path = self.state_path.with_suffix(self.state_path.suffix + ".lock")
        flags = os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW
        try:
            fd = os.open(lock_path, flags | (os.O_CREAT if create else 0), 0o600)
        except OSError as error:
            raise RootBudgetError(f"cannot open root admission lock: {lock_path}") from error
        try:
            _check_file(fd, lock_path)
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                state_fd = os.open(self.state_path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            except FileNotFoundError as error:
                if create and os.fstat(fd).st_size == 0:
                    yield None
                    # A persistent initialization marker prevents a missing JSON
                    # file from resetting this Pi root's generation or budgets.
                    os.write(fd, b"initialized\n")
                    os.fsync(fd)
                    return
                raise RootBudgetError("root state is missing; refusing to recreate an initialized budget") from error
            except OSError as error:
                raise RootBudgetError("cannot read root admission state") from error
            with os.fdopen(state_fd, "r", encoding="utf-8") as stream:
                _check_file(stream.fileno(), self.state_path)
                try:
                    state = json.load(stream, object_pairs_hook=_unique_object)
                except (ValueError, UnicodeError, RecursionError) as error:
                    raise RootBudgetError("malformed root admission state") from error
            yield state
            if create and os.fstat(fd).st_size == 0:
                os.write(fd, b"initialized\n")
                os.fsync(fd)
        finally:
            # Closing unlocks without ever unlinking the shared lock inode.
            os.close(fd)

    def _validate(self, state: object, *, allow_dead: bool = False) -> None:
        if not isinstance(state, dict):
            raise RootBudgetError("malformed root admission state")
        try:
            if type(state["version"]) is not int or state["version"] != 1:
                raise ValueError("unsupported version")
            root_id, pid, started = state["root_id"], state["root_pid"], state["root_starttime"]
            if not isinstance(root_id, str) or not _TOKEN.fullmatch(root_id) or not _positive_int(pid):
                raise ValueError("invalid root identity")
            if not isinstance(started, str) or not _UINT.fullmatch(started):
                raise ValueError("invalid process start time")
            if self._root_id is not None and root_id != self._root_id:
                raise ValueError("root generation does not match inherited identity")
            if self._identity is not None and (pid, started) != self._identity:
                raise ValueError("root Pi identity does not match")
            if not allow_dead:
                if _process_info(pid)[1] != started:
                    raise ValueError("root Pi PID was reused (start time mismatch)")
            policy = state["policy"]
            if not isinstance(policy, dict) or set(policy) != set(self._policy):
                raise ValueError("invalid frozen policy")
            for name, value in policy.items():
                if name == "PI_SUBAGENTS_ROOT_TIMEOUT":
                    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                        raise ValueError("invalid timeout")
                elif not _positive_int(value):
                    raise ValueError("invalid limit")
            if policy != self._policy:
                raise ValueError("environment differs from immutable root policy")
            if type(state["deadline"]) not in (int, float) or not math.isfinite(state["deadline"]) or state["deadline"] <= 0:
                raise ValueError("invalid deadline")
            records, released, tasks = state["records"], state["released"], state["tasks"]
            if not isinstance(records, dict) or not isinstance(released, dict) or set(records) & set(released):
                raise ValueError("invalid reservation/tombstone maps")
            if type(tasks) is not int or tasks != len(records) + len(released) or not 0 <= tasks <= policy["PI_SUBAGENTS_ROOT_MAX_TASKS"]:
                raise ValueError("invalid task count")
            all_records = {**released, **records}
            if len(records) > policy["PI_SUBAGENTS_ROOT_MAX_CONCURRENT"]:
                raise ValueError("invalid live reservation count")
            closing, launchers = state["closing"], state["launchers"]
            if not isinstance(closing, list) or any(not isinstance(token, str) for token in closing) or len(set(closing)) != len(closing) or not set(closing) <= set(records):
                raise ValueError("invalid closing barrier")
            if not isinstance(launchers, dict) or set(launchers) != set(records):
                raise ValueError("invalid launcher ownership")
            for owner in launchers.values():
                if not isinstance(owner, dict) or set(owner) != {"pid", "starttime"} or not _positive_int(owner["pid"]) or not isinstance(owner["starttime"], str) or not _UINT.fullmatch(owner["starttime"]):
                    raise ValueError("invalid launcher identity")
            windows: set[str] = set()
            for token, record in all_records.items():
                if not isinstance(token, str) or not _TOKEN.fullmatch(token) or not isinstance(record, dict):
                    raise ValueError("invalid reservation")
                if set(record) != {"token", "agent_id", "depth", "parent_token", "window_id"} or record["token"] != token:
                    raise ValueError("invalid reservation fields")
                if not isinstance(record["agent_id"], str) or not record["agent_id"].strip():
                    raise ValueError("invalid agent id")
                if not _positive_int(record["depth"]) or record["depth"] > policy[_MAX_DEPTH]:
                    raise ValueError("invalid reservation depth")
                parent = record["parent_token"]
                if token in records and isinstance(parent, str) and parent in closing and token not in closing:
                    raise ValueError("closing barrier does not include a descendant")
                if record["depth"] == 1:
                    if parent is not None:
                        raise ValueError("root reservation has a parent token")
                else:
                    parents = records if token in records else all_records
                    if not isinstance(parent, str) or parent not in parents or parents[parent]["depth"] != record["depth"] - 1:
                        raise ValueError("missing or inconsistent reservation parent")
                window = record["window_id"]
                if window is not None:
                    if not isinstance(window, str) or not _WINDOW.fullmatch(window) or (token in records and window in windows):
                        raise ValueError("invalid or duplicate window id")
                    if token in records:
                        windows.add(window)
            if self._parent_token is not None:
                parent = (all_records if allow_dead else records).get(self._parent_token)
                if parent is None or parent["depth"] != self._depth:
                    raise ValueError("inherited parent token is missing or has the wrong depth")
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise RootBudgetError(f"unsafe root admission state: {error}") from error

    def _write(self, state: dict) -> None:
        try:
            fd, temporary = tempfile.mkstemp(prefix=self.state_path.name + ".", dir=self.state_path.parent)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                os.fchmod(stream.fileno(), 0o600)
                json.dump(state, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.state_path)
            directory_fd = os.open(self.state_path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except (OSError, ValueError) as error:
            raise RootBudgetError("could not persist root admission state; reservation may remain charged") from error

    def _record(self, state: dict, token: str, *, include_released: bool = False) -> dict:
        records = {**state["released"], **state["records"]}
        eligible = records if include_released else state["records"]
        if not isinstance(token, str) or not _TOKEN.fullmatch(token) or token not in eligible:
            raise RootBudgetError("unknown root reservation token")
        record = eligible[token]
        if self._parent_token is not None:
            cursor = record
            while cursor["parent_token"] is not None:
                if cursor["parent_token"] == self._parent_token:
                    break
                cursor = records[cursor["parent_token"]]
            else:
                raise RootBudgetError("reservation is not an owned descendant of this caller")
        return record

    def _admit(self, state: dict) -> None:
        if time.monotonic() >= state["deadline"]:
            raise RootBudgetError("root admission deadline exhausted; close existing descendants")
        if state["tasks"] >= self._policy["PI_SUBAGENTS_ROOT_MAX_TASKS"]:
            raise RootBudgetError("root admission task budget exhausted")
        if len(state["records"]) >= self._policy["PI_SUBAGENTS_ROOT_MAX_CONCURRENT"]:
            raise RootBudgetError("root live-window budget exhausted; close an existing window before spawning (admission never waits)")

    def reserve(self, agent_id: str, depth: int) -> str:
        """Charge one stable, unique token immediately, before window startup."""
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise RootBudgetError("agent_id must be a nonempty string")
        if type(depth) is not int or depth != self._depth + 1 or depth > self._policy[_MAX_DEPTH]:
            raise RootBudgetError(f"cannot reserve child depth {depth!r} from depth {self._depth} (max {self._policy[_MAX_DEPTH]})")
        with self._locked() as state:
            self._validate(state)
            if self._parent_token in state["closing"]:
                raise RootBudgetError("cannot reserve beneath a closing parent")
            self._admit(state)
            token = uuid.uuid4().hex
            while token in state["records"] or token in state["released"]:
                token = uuid.uuid4().hex
            state["records"][token] = {
                "token": token, "agent_id": agent_id, "depth": depth,
                "parent_token": self._parent_token, "window_id": None,
            }
            state["launchers"][token] = {"pid": os.getpid(), "starttime": _process_info(os.getpid())[1]}
            state["tasks"] += 1
            self._write(state)
            return token

    def attach(self, token: str, window_id: str) -> None:
        """Attach immutable, validated tmux metadata (never probe the window)."""
        if not isinstance(window_id, str) or not _WINDOW.fullmatch(window_id):
            raise RootBudgetError("window_id must be a tmux @digits identifier")
        with self._locked() as state:
            self._validate(state)
            record = self._record(state, token)
            if token in state["closing"]:
                raise RootBudgetError("reservation is closing; terminate the newly created owned window before release")
            if state["launchers"][token] != {"pid": os.getpid(), "starttime": _process_info(os.getpid())[1]}:
                raise RootBudgetError("only the reserving launcher may attach its window")
            if record["window_id"] is not None and record["window_id"] != window_id:
                raise RootBudgetError("reservation is already attached to a different window")
            if any(other["window_id"] == window_id and other["token"] != token for other in state["records"].values()):
                raise RootBudgetError("window is already attached to another reservation")
            record["window_id"] = window_id
            self._write(state)

    def child_env(self, token: str) -> dict[str, str]:
        """Explicit child environment for tmux, including all immutable limits."""
        with self._locked() as state:
            self._validate(state)
            record = self._record(state, token)
            if token in state["closing"]:
                raise RootBudgetError("cannot bootstrap a closing reservation")
            return {
                _STATE: str(self.state_path), _ROOT_ID: state["root_id"], _PARENT: token,
                _DEPTH: str(record["depth"]),
                **{name: str(value) for name, value in state["policy"].items()},
            }

    @staticmethod
    def _descendants(state: dict, token: str) -> list[dict]:
        result: list[dict] = []
        for record in state["records"].values():
            cursor = record
            while cursor["parent_token"] is not None:
                if cursor["parent_token"] == token:
                    result.append(dict(record))
                    break
                cursor = state["records"][cursor["parent_token"]]
        return sorted(result, key=lambda record: (-record["depth"], record["token"]))

    def descendants(self, token: str) -> list[dict]:
        """Snapshot descendants, excluding token; use begin_close for cleanup."""
        with self._locked() as state:
            self._validate(state, allow_dead=True)
            self._record(state, token, include_released=True)
            return self._descendants(state, token)

    def begin_close(self, token: str) -> list[dict]:
        """Atomically bar spawn/attach in this subtree; return deepest-first children.

        Pending window=None records must be settled by their reserving launcher,
        never treated as proof that no window exists by another process.
        """
        with self._locked() as state:
            self._validate(state, allow_dead=True)
            self._record(state, token, include_released=True)
            if token in state["released"]:
                return []
            descendants = self._descendants(state, token)
            state["closing"] = sorted(set(state["closing"]) | {token} | {record["token"] for record in descendants})
            self._write(state)
            return descendants

    def remaining_seconds(self) -> float:
        """Root time remaining; lifecycle must cap startup and settlement waits.

        Cleanup remains legal when this reaches zero, but new admissions do not.
        """
        with self._locked() as state:
            self._validate(state)
            return max(0.0, state["deadline"] - time.monotonic())

    def release(self, token: str) -> None:
        """Release AFTER confirmed termination; children must be released first.

        This is intentionally not recursive: bulk deletion could reclaim permits
        for descendants whose window termination has not been confirmed. Authorized
        duplicate releases are harmless. Tombstones retain identity/ancestry and
        are bounded by the immutable root task budget, never reset on empty.
        """
        with self._locked() as state:
            self._validate(state, allow_dead=True)
            record = self._record(state, token, include_released=True)
            if token in state["released"]:
                return
            if record["window_id"] is None and state["launchers"][token] != {"pid": os.getpid(), "starttime": _process_info(os.getpid())[1]}:
                raise RootBudgetError("pending startup may be in flight; only its reserving launcher can confirm no window and release")
            if any(other["parent_token"] == token for other in state["records"].values()):
                raise RootBudgetError("cannot release a reservation with live descendants; close deepest first")
            state["released"][token] = dict(record)
            del state["records"][token]
            del state["launchers"][token]
            if token in state["closing"]:
                state["closing"].remove(token)
            self._write(state)

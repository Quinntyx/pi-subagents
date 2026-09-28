"""Model/thinking catalog for subagents.

Answers "what can a subagent run on?" without guessing slugs. The catalog comes
from the pi CLI as seen by a *spawned subagent* (`pi --list-models` under the subagent agent dir), so it
reflects exactly what a spawned instance can resolve, and the agent dir's settings
supply the defaults and the picker-scoped model list.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Any

from .envcheck import require_environment
from .tmuxenv import subagent_agent_dir

__all__ = [
    "ModelInfo",
    "list_models",
    "model_slugs",
    "resolve_models",
    "best_model_match",
    "thinking_levels",
    "agent_dir_defaults",
    "scoped_models",
    "capabilities",
]

# The canonical effort ladder pi accepts for --thinking.
THINKING_LEVELS = ["off", "minimal", "low", "medium", "high", "xhigh", "max"]

DEFAULT_CATALOG_TIMEOUT = 60.0
# The interpreter is long-lived (that is the point of a persistent session), so an
# unbounded cache froze the catalog for the life of the session: a model added
# later never appeared, even though spawning it with --model worked. Entries now
# expire; PI_SUBAGENTS_CATALOG_TTL tunes the window (0 disables caching).
DEFAULT_CATALOG_TTL = 120.0
_CACHE: dict[str | None, tuple[float, list["ModelInfo"]]] = {}
# Context sizes as printed by `pi --list-models`: 200K, 1M, 131.1K, 104.9K, ...
_SIZE_COLUMN = re.compile(r"^\d+(?:\.\d+)?[KMB]?$")


@dataclass(frozen=True)
class ModelInfo:
    provider: str
    model: str
    context: str | None = None
    max_output: str | None = None
    thinking: bool | None = None
    images: bool | None = None

    @property
    def slug(self) -> str:
        return f"{self.provider}/{self.model}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "provider": self.provider,
            "model": self.model,
            "context": self.context,
            "max_output": self.max_output,
            "thinking": self.thinking,
            "images": self.images,
        }


def _catalog_timeout() -> float:
    try:
        return max(1.0, float(os.environ.get("PI_SUBAGENTS_CATALOG_TIMEOUT", DEFAULT_CATALOG_TIMEOUT)))
    except ValueError:
        return DEFAULT_CATALOG_TIMEOUT


def _catalog_ttl() -> float:
    try:
        return max(0.0, float(os.environ.get("PI_SUBAGENTS_CATALOG_TTL", DEFAULT_CATALOG_TTL)))
    except ValueError:
        return DEFAULT_CATALOG_TTL


def _pi_command() -> str:
    return shutil.which("pi") or "pi"


def _parse_table(output: str) -> list[ModelInfo]:
    rows: list[ModelInfo] = []
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("provider"):
            continue
        # Non-table output ("No models matching \"x\"") must not become a row, and
        # the context column is always a size like 200K / 1M / 131.1K.
        if line.lower().startswith("no models") or not _SIZE_COLUMN.match(line.split()[2] if len(line.split()) > 2 else ""):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        provider, model = parts[0], parts[1]
        rest = parts[2:]

        def flag(index: int) -> bool | None:
            if index >= len(rest):
                return None
            value = rest[index].lower()
            if value in ("yes", "no"):
                return value == "yes"
            return None

        rows.append(
            ModelInfo(
                provider=provider,
                model=model,
                context=rest[0] if len(rest) > 0 else None,
                max_output=rest[1] if len(rest) > 1 else None,
                thinking=flag(2),
                images=flag(3),
            )
        )
    return rows


def list_models(search: str | None = None, *, refresh: bool = False) -> list[ModelInfo]:
    """Models a subagent can run on, optionally filtered by a fuzzy search.

    `search` is passed to `pi --list-models <search>`, which matches loosely (for
    example "astra" returns every provider's Astra variant), so it doubles as the
    lookup for "the user asked for model X, what is the slug?".

    Results are cached per search term for `PI_SUBAGENTS_CATALOG_TTL` seconds
    (default 120) so a long-lived session still sees newly added models; pass
    `refresh=True` to bypass the cache.
    """
    key = search or ""
    if not refresh:
        cached = _CACHE.get(key)
        if cached is not None and (time.monotonic() - cached[0]) < _catalog_ttl():
            return list(cached[1])

    env = {**os.environ, "PI_CODING_AGENT_DIR": str(subagent_agent_dir())}
    command = [_pi_command(), "--list-models"]
    if search:
        command.append(search)
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=_catalog_timeout(),
            env=env,
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            "pi CLI not found on PATH; cannot list subagent models (pass model= explicitly)"
        ) from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"pi --list-models timed out after {_catalog_timeout():.0f}s") from error

    output = completed.stdout or ""
    if completed.returncode != 0 and not output.strip():
        detail = (completed.stderr or "").strip()[:400]
        raise RuntimeError(f"pi --list-models failed ({completed.returncode}): {detail}")

    rows = _parse_table(output)
    _CACHE[key] = (time.monotonic(), rows)
    return list(rows)


def model_slugs(search: str | None = None, *, refresh: bool = False) -> list[str]:
    """`provider/model` slugs, ready to pass as `agent(model=...)`."""
    return [row.slug for row in list_models(search, refresh=refresh)]


def resolve_models(query: str, *, refresh: bool = False) -> list[ModelInfo]:
    """Every catalog entry a loose query really matches (e.g. "astra", "opus").

    The CLI's search is fuzzy and returns unrelated neighbours (asking for "opus"
    also lists gemini models), so results are narrowed to rows whose slug actually
    contains the query. An exact slug always wins even if the search is odd.
    """
    needle = query.strip().lower()
    if not needle:
        return []
    exact = [row for row in list_models(None, refresh=refresh) if row.slug.lower() == needle]
    if exact:
        return exact
    matches = [row for row in list_models(query, refresh=refresh) if needle in row.slug.lower()]
    if matches or refresh:
        return matches
    # A miss is never served blindly from a possibly stale cache: ask the CLI again.
    return [row for row in list_models(query, refresh=True) if needle in row.slug.lower()]


def best_model_match(query: str, *, refresh: bool = False) -> ModelInfo | None:
    """Single best match for a loose model name, or None when nothing matches.

    Ordering prefers: an exact slug, then the agent dir's own default provider, then
    first-party entries (no `~` proxy prefix), then catalog order — so "astra"
    resolves to `openai-codex/gpt-6-astra` rather than a proxied variant.
    """
    matches = resolve_models(query, refresh=refresh)
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]

    needle = query.strip().lower()
    exact = [m for m in matches if m.slug.lower() == needle]
    if exact:
        return exact[0]

    default_provider = agent_dir_defaults().get("provider")

    def rank(model: ModelInfo) -> tuple[int, int, str]:
        scored = 0
        if default_provider and model.provider == default_provider:
            scored -= 2
        if not model.model.startswith("~") and not model.provider.startswith("~"):
            scored -= 1
        return (scored, matches.index(model), model.slug)

    return sorted(matches, key=rank)[0]


def thinking_levels() -> list[str]:
    """Effort levels accepted by `agent(thinking=...)`."""
    return list(THINKING_LEVELS)


def _agent_dir_settings() -> dict[str, Any]:
    path = subagent_agent_dir() / "settings.json"
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def agent_dir_defaults() -> dict[str, Any]:
    """What a spawned subagent uses when no model/thinking is passed."""
    settings = _agent_dir_settings()
    provider = settings.get("defaultProvider")
    model = settings.get("defaultModel")
    default_slug = f"{provider}/{model}" if provider and model else (model or None)
    return {
        "agentDir": str(subagent_agent_dir()),
        "provider": provider,
        "model": model,
        "slug": default_slug,
        "thinking": settings.get("defaultThinkingLevel"),
    }


def scoped_models() -> list[str]:
    """The agent dir's picker/enabled model list (`enabledModels`).

    This is what the picker cycles through interactively; `agent(model=...)` may
    still use any slug from `list_models()` because spawns pass `--model`.
    """
    scoped = _agent_dir_settings().get("enabledModels")
    return [str(entry) for entry in scoped] if isinstance(scoped, list) else []


def capabilities(*, include_models: bool = False) -> dict[str, Any]:
    """One call that answers "what can subagents run with?".

    Combines the live catalog summary, the agent dir's defaults, the picker-scoped
    list, and the runtime caps. The full catalog is ~600 entries, so it is only
    returned when `include_models=True`; use `list_models(search)` for specifics.
    """
    defaults = agent_dir_defaults()
    models = list_models()
    result: dict[str, Any] = {
        "agentDir": defaults["agentDir"],
        "default_model": defaults["slug"],
        "default_thinking": defaults["thinking"],
        "thinking_levels": thinking_levels(),
        "scoped_models": scoped_models(),
        "model_count": len(models),
        "providers": sorted({row.provider for row in models}),
        "max_concurrent": _int_env("PI_SUBAGENTS_MAX_CONCURRENT", 8),
        "settle_timeout_seconds": _float_env("PI_SUBAGENTS_SETTLE_TIMEOUT", 1800.0),
        "depth": _int_env("PI_SUBAGENT_DEPTH", 0),
    }
    if include_models:
        result["models"] = [row.as_dict() for row in models]
    return result


def _int_env(name: str, fallback: int) -> int:
    try:
        return int(os.environ.get(name, str(fallback)))
    except ValueError:
        return fallback


def _float_env(name: str, fallback: float) -> float:
    try:
        return float(os.environ.get(name, str(fallback)))
    except ValueError:
        return fallback


# Imported here to keep the module import-time surface small; require_environment
# is used by callers that only import the helpers above.
_ = require_environment

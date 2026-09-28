"""Catalog tests with a fake `pi` CLI (no network, no real catalog)."""

from __future__ import annotations

import json
import os
import stat
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pi_subagents.catalog as catalog  # noqa: E402

FAKE_TABLE = """\
provider         model                                   context  max-out  thinking  images
deepseek-router  deepseek-v4.1-flash                     200K     128K     yes       yes
openai-codex     gpt-6-astra                             272K     128K     yes       yes
opencode         gpt-6-astra                             1.1M     128K     yes       yes
opencode         claude-opus-4-5                         200K     64K      yes       yes
openrouter       ~openai/gpt-astra-latest                1.1M     128K     yes       yes
google           gemini-2.5-computer-use-preview        131.1K   65.5K    yes       yes
"""


@pytest.fixture()
def fake_pi(tmp_path, monkeypatch):
    """A `pi` on PATH that prints FAKE_TABLE, plus an agent dir with settings."""
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "settings.json").write_text(
        json.dumps(
            {
                "defaultProvider": "deepseek-router",
                "defaultModel": "deepseek-v4.1-flash",
                "defaultThinkingLevel": "high",
                "enabledModels": ["deepseek-router/deepseek-v4.1-flash"],
            }
        )
    )

    script = tmp_path / "pi"
    script.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "$1" = "--list-models" ]; then\n'
        '  if [ -n "$2" ]; then\n'
        '    case "$2" in\n'
        '      zzz-nope*) echo "No models matching \\"$2\\""; exit 0;;\n'
        '      astra) printf "%s\\n" "provider model context max-out thinking images" '
        '"openai-codex gpt-6-astra 272K 128K yes yes" "opencode gpt-6-astra 1.1M 128K yes yes" '
        '"openrouter ~openai/gpt-astra-latest 1.1M 128K yes yes"; exit 0;;\n'
        '      opus) printf "%s\\n" "provider model context max-out thinking images" '
        '"opencode claude-opus-4-5 200K 64K yes yes" "google gemini-2.5-computer-use-preview 131.1K 65.5K yes yes"; exit 0;;\n'
        "    esac\n"
        "  fi\n"
        f"  cat <<'TABLE'\n{FAKE_TABLE}TABLE\n"
        "fi\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)

    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setenv("PI_CODING_SUBAGENT_DIR", str(profile))
    catalog._CACHE.clear()
    yield
    catalog._CACHE.clear()


def test_list_models_parses_the_table(fake_pi):
    rows = catalog.list_models()
    assert len(rows) == 6
    first = rows[0]
    assert first.slug == "deepseek-router/deepseek-v4.1-flash"
    assert first.context == "200K"
    assert first.max_output == "128K"
    assert first.thinking is True
    assert first.images is True


def test_no_models_message_is_not_a_row(fake_pi):
    assert catalog.list_models("zzz-nope-999") == []
    assert catalog.best_model_match("zzz-nope-999") is None


def test_resolve_filters_cli_fuzzy_false_positives(fake_pi):
    # The fake CLI returns a gemini row for "opus"; only real opus rows survive.
    assert [m.slug for m in catalog.resolve_models("opus")] == ["opencode/claude-opus-4-5"]
    assert catalog.best_model_match("opus").slug == "opencode/claude-opus-4-5"


def test_best_match_prefers_default_provider_then_first_party(fake_pi):
    # "flash" is only in the default provider's row set.
    assert catalog.best_model_match("flash").slug == "deepseek-router/deepseek-v4.1-flash"
    # "astra" has no default-provider match, so first-party order decides.
    assert catalog.best_model_match("astra").slug == "openai-codex/gpt-6-astra"
    # An exact slug always wins.
    assert catalog.best_model_match("opencode/gpt-6-astra").slug == "opencode/gpt-6-astra"


def test_thinking_levels_and_agent_dir_scoping(fake_pi):
    assert catalog.thinking_levels()[0] == "off"
    assert "xhigh" in catalog.thinking_levels()
    assert catalog.scoped_models() == ["deepseek-router/deepseek-v4.1-flash"]
    defaults = catalog.agent_dir_defaults()
    assert defaults["slug"] == "deepseek-router/deepseek-v4.1-flash"
    assert defaults["thinking"] == "high"


def test_capabilities_summarizes_without_dumping_the_catalog(fake_pi):
    caps = catalog.capabilities()
    assert caps["model_count"] == 6
    assert caps["default_model"] == "deepseek-router/deepseek-v4.1-flash"
    assert "openai-codex" in caps["providers"]
    assert caps["thinking_levels"] == catalog.thinking_levels()
    assert "models" not in caps
    assert len(catalog.capabilities(include_models=True)["models"]) == 6


def test_catalog_caches_until_refresh(fake_pi, tmp_path):
    catalog.list_models()
    script = tmp_path / "pi"
    script.write_text("#!/usr/bin/env bash\necho 'No models matching' >&2\nexit 1\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    # Cached: still the first result.
    assert len(catalog.list_models()) == 6
    with pytest.raises(RuntimeError):
        catalog.list_models(refresh=True)


def test_a_stale_miss_is_retried_against_the_live_catalog(fake_pi, tmp_path, monkeypatch):
    """A model added after the cache was filled must still be found.

    This is the failure that motivated the TTL + retry: a long-lived PTC
    interpreter cached the morning's catalog, so a model added later resolved to
    None even though the CLI (and --model) knew it.
    """
    # 1. warm the cache with a lookup that misses
    assert catalog.best_model_match("space-bunny-free") is None

    # 2. the model shows up in the provider catalog (fake CLI now returns it)
    script = tmp_path / "pi"
    script.write_text(
        "#!/usr/bin/env bash\n"
        "if [ \"$1\" = \"--list-models\" ]; then\n"
        "  if [ -n \"$2\" ]; then\n"
        "    printf '%s\\n' 'provider model context max-out thinking images' "
        "'opencode space-bunny-free 1.0M 524.3K yes yes' "
        "'opencode-go space-bunny-free 1.0M 524.3K yes yes'; exit 0\n"
        "  fi\n"
        f"  cat <<'TABLE'\n{FAKE_TABLE}TABLE\n"
        "fi\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)

    # 3. still found, because a miss is never served from a stale cache
    match = catalog.best_model_match("space-bunny-free")
    assert match is not None
    assert match.slug in ("opencode/space-bunny-free", "opencode-go/space-bunny-free")
    assert catalog.model_slugs("space-bunny-free")


def test_cache_entries_expire(fake_pi, monkeypatch):
    monkeypatch.setenv("PI_SUBAGENTS_CATALOG_TTL", "0")
    assert catalog._catalog_ttl() == 0.0
    # TTL 0 disables caching: every call hits the CLI (and cannot go stale).
    assert catalog.list_models()
    first = catalog.list_models()
    assert catalog.list_models() == first  # same data, fresh call

    monkeypatch.setenv("PI_SUBAGENTS_CATALOG_TTL", "not-a-number")
    assert catalog._catalog_ttl() == catalog.DEFAULT_CATALOG_TTL

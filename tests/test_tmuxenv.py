"""Tests for tmux command construction and name allocation (no live tmux)."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pi_subagents.tmuxenv as tmuxenv  # noqa: E402


def test_unique_socket_name_prefix_and_fallback(tmp_path, monkeypatch):
	monkeypatch.setenv("PI_SOCK_DIR", str(tmp_path))
	name = tmuxenv.unique_socket_name("Test Digger!")
	assert name.startswith("subagent-test-digger")
	# collision path: create the base socket file, expect a suffixed name
	(tmp_path / f"{name}.sock").touch()
	second = tmuxenv.unique_socket_name("Test Digger!")
	assert second != name and second.startswith("subagent-test-digger-")


def test_quote_for_tmux_shell(monkeypatch):
	monkeypatch.setattr(tmuxenv.shutil, "which", lambda name: "pi")
	quoted = tmuxenv._quote_for_tmux_shell("it's a 'quoted' string")
	assert quoted == "'it'\\''s a '\\''quoted'\\'' string'"
	command = tmuxenv._build_pi_command("hello world", model="deepseek-router/deepseek-v4.1-flash", thinking="high")
	assert command.startswith("pi --model deepseek-router/deepseek-v4.1-flash --thinking high")
	assert command.endswith("'hello world'")


def test_max_concurrent_env(monkeypatch):
	monkeypatch.setenv("PI_SUBAGENTS_MAX_CONCURRENT", "3")
	assert tmuxenv.max_concurrent() == 3
	monkeypatch.setenv("PI_SUBAGENTS_MAX_CONCURRENT", "bogus")
	assert tmuxenv.max_concurrent() == 8


def test_spawn_requires_environment(monkeypatch, tmp_path):
	monkeypatch.delenv("TMUX", raising=False)
	import pi_subagents.envcheck as envcheck

	envcheck.placement = None
	envcheck.ENV_OK = False
	envcheck.WARNING_SHOWN = False
	with pytest.raises(NotImplementedError):
		tmuxenv.spawn_pi_window(
			"prompt", name="x", cwd=str(tmp_path), window_name="x",
			model=None, thinking=None, socket_name="subagent-x", depth=1,
		)


def test_missing_profile_raises(monkeypatch, tmp_path):
	# a tmux placement stub plus a profile dir that does not exist
	import pi_subagents.envcheck as envcheck

	envcheck.placement = envcheck.TmuxPlacement("sess", "$0", "@1", "%0", "sock")
	envcheck.ENV_OK = True
	monkeypatch.setenv("PI_SUBAGENTS_PROFILE", str(tmp_path / "nope"))
	with pytest.raises(RuntimeError, match="profile not found"):
		tmuxenv.spawn_pi_window(
			"prompt", name="x", cwd=str(tmp_path), window_name="x",
			model=None, thinking=None, socket_name="subagent-x", depth=1,
		)

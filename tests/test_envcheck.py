"""Tests for env probing (tmux guard, depth lock) and schema validation."""

from __future__ import annotations

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pi_subagents.envcheck as envcheck  # noqa: E402
from pi_subagents.schema import SchemaValidationError, validate_with_schema  # noqa: E402


@pytest.fixture()
def reset_env(monkeypatch):
	monkeypatch.delenv("TMUX", raising=False)
	monkeypatch.delenv("TMUX_PANE", raising=False)
	monkeypatch.delenv("PI_SUBAGENT_DEPTH", raising=False)
	envcheck.placement = None
	envcheck.ENV_OK = False
	envcheck.WARNING_SHOWN = False
	yield
	envcheck.placement = None
	envcheck.ENV_OK = False
	envcheck.WARNING_SHOWN = False


def test_no_tmux_warns_and_locks(reset_env, capsys):
	with pytest.warns(UserWarning) if False else _noop():
		envcheck.ensure_environment()
	err = capsys.readouterr().err
	assert "not running inside a tmux session" in err
	assert envcheck.ENV_OK is False
	assert envcheck.WARNING_SHOWN is True


def _noop():
	from contextlib import nullcontext

	return nullcontext()


def test_depth_lock_raises_not_implemented(reset_env, monkeypatch):
	monkeypatch.setenv("PI_SUBAGENT_DEPTH", "1")
	with pytest.raises(NotImplementedError, match="PI_SUBAGENT_DEPTH"):
		envcheck.ensure_environment()


def test_require_environment_raises_without_tmux(reset_env):
	with pytest.raises(NotImplementedError, match="subagent spawning"):
		envcheck.require_environment()


def test_tmux_unreachable_warns(reset_env, monkeypatch):
	monkeypatch.setenv("TMUX", "/tmp/nonexistent-sock,0,0")
	monkeypatch.setenv("TMUX_PANE", "%0")
	envcheck.ensure_environment()
	assert envcheck.ENV_OK is False
	assert envcheck.placement is None


def test_schema_validation_subset():
	schema = {
		"type": "object",
		"required": ["a"],
		"additionalProperties": False,
		"properties": {"a": {"type": "integer"}, "b": {"type": "array", "items": {"type": "string"}}},
	}
	validate_with_schema(schema, {"a": 1, "b": ["x"]})
	with pytest.raises(SchemaValidationError):
		validate_with_schema(schema, {})
	with pytest.raises(SchemaValidationError):
		validate_with_schema(schema, {"a": 1, "c": 2})
	with pytest.raises(SchemaValidationError):
		validate_with_schema(schema, {"a": "not-int"})


def test_schema_fallback_validator_handles_all_types(monkeypatch):
	"""Without jsonschema installed, the built-in subset validator must work
	for every supported type (a boolean schema value once crashed it).""
	for every supported type (a boolean schema value once crashed it)."""

	schema = {
		"type": "object",
		"required": ["flag", "n", "s", "items", "nothing"],
		"additionalProperties": False,
		"properties": {
			"flag": {"type": "boolean"},
			"n": {"type": "integer"},
			"s": {"type": "string"},
			"items": {"type": "array", "items": {"type": "string"}},
			"nothing": {"type": "null"},
		},
	}
	value = {"flag": True, "n": 2, "s": "x", "items": ["a"], "nothing": None}
	import builtins
	real_import = builtins.__import__

	def no_jsonschema(name, *args, **kwargs):
		if name == "jsonschema":
			raise ImportError("hidden for the fallback test")
		return real_import(name, *args, **kwargs)

	monkeypatch.setattr(builtins, "__import__", no_jsonschema)
	validate_with_schema(schema, value)
	with pytest.raises(SchemaValidationError):
		validate_with_schema(schema, {**value, "flag": "yes"})
	with pytest.raises(SchemaValidationError):
		validate_with_schema(schema, {**value, "extra": 1})

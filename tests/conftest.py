"""Shared pytest fixtures for the pi-subagents test suite.

Two autouse guards make window/pool leaks structurally impossible rather than
a matter of test discipline:

- ``_clean_registry`` closes any pool still open at teardown.
- ``_kill_spawned_windows`` records every *real* tmux window spawned during a
  test and kills any survivor on teardown — even if the test crashed mid-way
  or a pool was abandoned without close().

Tests should still prefer ``with AgentPool(...) as pool:`` for normal
lifecycle (clean exit closes the pool; an exception deliberately leaves it
alive for continuation in a follow-up cell), reserving explicit close() for
tests that assert close/finish semantics themselves.
"""

from __future__ import annotations

import pytest

import pi_subagents.handle as handle_mod
from pi_subagents import tmuxenv
from pi_subagents.registry import REGISTRY


@pytest.fixture(autouse=True)
def _clean_registry():
	REGISTRY._handles.clear()
	REGISTRY._pools.clear()
	yield
	for pool in REGISTRY.pools():
		if not pool.closed:
			try:
				pool.close()
			except Exception:
				pass
	REGISTRY._handles.clear()
	REGISTRY._pools.clear()


@pytest.fixture(autouse=True)
def _kill_spawned_windows():
	"""Kill any tmux window a test actually spawned, leaked or not."""
	spawned: list[str] = []
	original = handle_mod.spawn_pi_window

	def tracking_spawn(*args, **kwargs):
		ref = original(*args, **kwargs)
		spawned.append(ref.window_id)
		return ref

	handle_mod.spawn_pi_window = tracking_spawn
	yield
	handle_mod.spawn_pi_window = original
	for window_id in spawned:
		try:
			if tmuxenv.window_alive(window_id):
				tmuxenv.kill_window(window_id)
		except Exception:
			pass

"""Exception types for pi_subagents (kept separate to avoid import cycles)."""


class PiSubagentsError(Exception):
    """Base class for pi-subagents failures."""


class PiSubagentsTimeoutError(PiSubagentsError, TimeoutError):
    """An agent did not settle within the allotted timeout."""

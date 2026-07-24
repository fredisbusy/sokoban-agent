"""Bounded reference-solver result contracts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ReferenceResult:
    """One bounded A* result, without claiming mathematical optimality."""

    solved: bool
    action_count: int | None = None
    action_sequence: tuple[str, ...] = ()
    push_count: int | None = None
    expanded_states: int = 0
    elapsed_seconds: float = 0.0

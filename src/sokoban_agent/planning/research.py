"""Research-only metric helpers shared by guarded planners."""

from __future__ import annotations

from typing import TypedDict

from sokoban_agent.planning.base import SearchResult


class GuardReferenceFields(TypedDict):
    """Fields comparing an adopted prefix with a bounded search reference."""

    guard_suffix_expanded_states: int
    guard_reference_calls: int
    guard_reference_action_count: int
    guard_reference_expanded_states: int
    guard_reference_elapsed_seconds: float
    guard_expansions_saved: int


def guard_reference_fields(
    reference: SearchResult | None,
    *,
    reference_called: bool,
    suffix_expanded_states: int,
    contribution: bool = True,
    diagnostic_elapsed: bool = True,
) -> GuardReferenceFields:
    """Build contribution fields without mixing diagnostic and policy time."""

    reference_expanded = reference.expanded_states if reference else 0
    return {
        "guard_suffix_expanded_states": suffix_expanded_states,
        "guard_reference_calls": int(reference_called),
        "guard_reference_action_count": (
            len(reference.actions) if reference else 0
        ),
        "guard_reference_expanded_states": reference_expanded,
        "guard_reference_elapsed_seconds": (
            reference.elapsed_seconds
            if reference is not None and diagnostic_elapsed
            else 0.0
        ),
        "guard_expansions_saved": (
            reference_expanded - suffix_expanded_states
            if reference is not None and contribution
            else 0
        ),
    }

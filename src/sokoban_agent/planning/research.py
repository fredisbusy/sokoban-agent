"""Research-only metric helpers shared by guarded planners."""

from __future__ import annotations

from sokoban_agent.planning.base import (
    GuardDisposition,
    GuardPlanningMetrics,
    SearchResult,
)


def guard_metrics(
    reference: SearchResult | None,
    *,
    disposition: GuardDisposition,
    proposed_actions: int,
    legal_prefix_actions: int,
    adopted_actions: int,
    reference_called: bool,
    suffix_expanded_states: int,
    contribution: bool = True,
    diagnostic_elapsed: bool = True,
) -> GuardPlanningMetrics:
    """Build one cohesive guard measurement."""

    reference_expanded = reference.expanded_states if reference else 0
    return GuardPlanningMetrics(
        disposition=disposition,
        proposed_actions=proposed_actions,
        legal_prefix_actions=legal_prefix_actions,
        adopted_actions=adopted_actions,
        suffix_expanded_states=suffix_expanded_states,
        reference_calls=int(reference_called),
        reference_action_count=len(reference.actions) if reference else 0,
        reference_expanded_states=reference_expanded,
        reference_elapsed_seconds=(
            reference.elapsed_seconds
            if reference is not None and diagnostic_elapsed
            else 0.0
        ),
        expansions_saved=(
            reference_expanded - suffix_expanded_states
            if reference is not None and contribution
            else 0
        ),
    )

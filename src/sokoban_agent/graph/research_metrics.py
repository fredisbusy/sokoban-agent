"""Stable keys used by episode-level research diagnostics."""

from __future__ import annotations

from hashlib import sha256

from sokoban_agent.env import Action
from sokoban_agent.graph.state import SokobanGraphState
from sokoban_agent.planning import Observation, PlanningOutcome


def observation_key(observation: Observation) -> str:
    """Hash one complete board state without retaining its raw bytes."""

    digest = sha256()
    digest.update(str(observation.shape).encode())
    digest.update(observation.tobytes())
    return digest.hexdigest()


def proposal_key(
    observation: Observation,
    actions: tuple[Action, ...],
) -> str:
    """Hash a non-empty proposal together with the state that produced it."""

    digest = sha256()
    digest.update(observation_key(observation).encode())
    digest.update(",".join(action.name for action in actions).encode())
    return digest.hexdigest()


def planning_research_update(
    state: SokobanGraphState,
    outcome: PlanningOutcome,
) -> dict[str, object]:
    """Accumulate proposal, guard, and repeated-plan diagnostics."""

    proposed = outcome.proposed_actions or outcome.actions
    key = proposal_key(state["observation"], proposed) if proposed else None
    repeated = key is not None and key in state["seen_plan_keys"]
    seen = state["seen_plan_keys"]
    if key is not None and not repeated:
        seen = (*seen, key)
    disposition = outcome.guard_disposition
    return {
        "seen_plan_keys": seen,
        "repeated_plans": state["repeated_plans"] + int(repeated),
        "guard_accepted": state["guard_accepted"]
        + int(disposition == "accepted"),
        "guard_suffix_added": state["guard_suffix_added"]
        + int(disposition == "suffix_added"),
        "guard_replaced": state["guard_replaced"]
        + int(disposition == "replaced"),
        "guard_failed": state["guard_failed"] + int(disposition == "failed"),
        "guard_proposed_actions": (
            state["guard_proposed_actions"] + outcome.guard_proposed_actions
        ),
        "guard_legal_prefix_actions": (
            state["guard_legal_prefix_actions"]
            + outcome.guard_legal_prefix_actions
        ),
        "guard_adopted_actions": (
            state["guard_adopted_actions"] + outcome.guard_adopted_actions
        ),
        "guard_suffix_expanded_states": (
            state["guard_suffix_expanded_states"]
            + outcome.guard_suffix_expanded_states
        ),
        "guard_reference_calls": (
            state["guard_reference_calls"] + outcome.guard_reference_calls
        ),
        "guard_reference_action_count": (
            state["guard_reference_action_count"]
            + outcome.guard_reference_action_count
        ),
        "guard_reference_expanded_states": (
            state["guard_reference_expanded_states"]
            + outcome.guard_reference_expanded_states
        ),
        "guard_reference_elapsed_seconds": (
            state["guard_reference_elapsed_seconds"]
            + outcome.guard_reference_elapsed_seconds
        ),
        "guard_expansions_saved": (
            state["guard_expansions_saved"] + outcome.guard_expansions_saved
        ),
    }


def execution_research_update(
    state: SokobanGraphState,
    observation: Observation,
    *,
    pushed: bool,
) -> dict[str, object]:
    """Accumulate board revisit and push diagnostics after one action."""

    key = observation_key(observation)
    revisited = key in state["visited_state_keys"]
    visited = state["visited_state_keys"]
    if not revisited:
        visited = (*visited, key)
    return {
        "visited_state_keys": visited,
        "push_count": state["push_count"] + int(pushed),
        "revisited_states": state["revisited_states"] + int(revisited),
    }

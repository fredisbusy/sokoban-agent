"""Bounded A* reference measurements for research comparisons."""

from __future__ import annotations

from dataclasses import dataclass

from sokoban_agent.env import SokobanEnv
from sokoban_agent.env.rules import (
    apply_action,
    initial_state,
    observation_for,
)
from sokoban_agent.planning import (
    NoSolutionError,
    SearchLimitError,
    solve_astar_result,
)


@dataclass(frozen=True, slots=True)
class ReferenceResult:
    """One bounded A* result, without claiming mathematical optimality."""

    solved: bool
    action_count: int | None = None
    action_sequence: tuple[str, ...] = ()
    push_count: int | None = None
    expanded_states: int = 0
    elapsed_seconds: float = 0.0


def measure_bounded_astar_reference(
    env: SokobanEnv,
    level_id: str,
    *,
    max_expanded_states: int = 100_000,
) -> ReferenceResult:
    """Measure a fixed level without mutating the environment episode."""

    level = env.level_provider.get(level_id)
    state = initial_state(level)
    observation = observation_for(level, state)
    try:
        result = solve_astar_result(
            observation,
            max_expanded_states=max_expanded_states,
        )
    except (NoSolutionError, SearchLimitError):
        return ReferenceResult(solved=False)

    pushes = 0
    for action in result.actions:
        move = apply_action(level, state, action)
        if move.invalid_move:
            raise RuntimeError("bounded A* reference returned an invalid plan")
        pushes += int(move.pushed)
        state = move.state
    return ReferenceResult(
        solved=True,
        action_count=len(result.actions),
        action_sequence=tuple(action.name for action in result.actions),
        push_count=pushes,
        expanded_states=result.expanded_states,
        elapsed_seconds=result.elapsed_seconds,
    )

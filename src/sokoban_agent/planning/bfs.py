"""Breadth-first search planning node for small Sokoban boards."""

from __future__ import annotations

from collections import deque
from time import perf_counter

from sokoban_agent.env import Action, SokobanState
from sokoban_agent.env.rules import (
    apply_action,
    decode_observation,
    has_static_corner_deadlock,
    is_success,
)
from sokoban_agent.planning.base import (
    NoSolutionError,
    Observation,
    PlanningContext,
    PlanningOutcome,
    SearchLimitError,
)

_Parent = tuple[SokobanState, Action] | None


def solve_bfs(
    observation: Observation,
    *,
    max_expanded_states: int = 100_000,
) -> tuple[Action, ...]:
    """Find a shortest primitive-action plan for a small board."""

    if max_expanded_states <= 0:
        raise ValueError("max_expanded_states must be positive")

    level, initial = decode_observation(observation)
    if is_success(level, initial):
        return ()
    if has_static_corner_deadlock(level, initial):
        raise NoSolutionError("initial state has a static corner deadlock")

    frontier = deque([initial])
    parents: dict[SokobanState, _Parent] = {initial: None}
    expanded_states = 0

    while frontier:
        if expanded_states >= max_expanded_states:
            raise SearchLimitError(
                f"BFS reached the {max_expanded_states} state expansion limit"
            )
        state = frontier.popleft()
        expanded_states += 1

        for action in Action:
            move = apply_action(level, state, action)
            next_state = move.state
            if move.invalid_move or next_state in parents:
                continue
            if has_static_corner_deadlock(level, next_state):
                continue

            parents[next_state] = (state, action)
            if is_success(level, next_state):
                return _restore_plan(parents, next_state)
            frontier.append(next_state)

    raise NoSolutionError("BFS exhausted every reachable state")


def _restore_plan(
    parents: dict[SokobanState, _Parent],
    goal: SokobanState,
) -> tuple[Action, ...]:
    actions: list[Action] = []
    state = goal
    while parents[state] is not None:
        parent = parents[state]
        if parent is None:
            break
        state, action = parent
        actions.append(action)
    actions.reverse()
    return tuple(actions)


class BFSPlanner:
    """Produce a complete shortest plan for the graph to validate and execute."""

    def __init__(self, *, max_expanded_states: int = 100_000) -> None:
        if max_expanded_states <= 0:
            raise ValueError("max_expanded_states must be positive")
        self.max_expanded_states = max_expanded_states

    @property
    def name(self) -> str:
        """Return the experiment name."""

        return "graph:bfs"

    def reset(self, *, seed: int | None = None) -> None:
        """Reset the stateless algorithm node."""

        del seed

    def plan(self, context: PlanningContext) -> PlanningOutcome:
        """Run BFS from the graph's current observation."""

        started_at = perf_counter()
        try:
            actions = solve_bfs(
                context.observation,
                max_expanded_states=self.max_expanded_states,
            )
        except (NoSolutionError, SearchLimitError) as error:
            return PlanningOutcome(
                error=str(error),
                error_kind="search",
                elapsed_seconds=perf_counter() - started_at,
            )
        return PlanningOutcome(
            actions=actions,
            elapsed_seconds=perf_counter() - started_at,
        )

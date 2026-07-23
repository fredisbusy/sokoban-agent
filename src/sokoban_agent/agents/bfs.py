"""Deterministic breadth-first search baseline for small Sokoban levels."""

from __future__ import annotations

from collections import deque

from sokoban_agent.agents.base import (
    AgentInfo,
    AgentStopped,
    NoSolutionError,
    Observation,
    SearchLimitError,
)
from sokoban_agent.env import Action, SokobanState
from sokoban_agent.env.rules import (
    apply_action,
    decode_observation,
    has_static_corner_deadlock,
    is_success,
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


class BFSAgent:
    """Replay a shortest plan found from the episode's initial observation."""

    def __init__(self, *, max_expanded_states: int = 100_000) -> None:
        if max_expanded_states <= 0:
            raise ValueError("max_expanded_states must be positive")
        self.max_expanded_states = max_expanded_states
        self._plan: deque[Action] = deque()

    @property
    def name(self) -> str:
        """Return the experiment name."""

        return "bfs"

    def reset(
        self,
        observation: Observation,
        info: AgentInfo,
        *,
        seed: int | None = None,
    ) -> None:
        """Plan a shortest path for the new episode."""

        del info, seed
        self._plan = deque(
            solve_bfs(
                observation,
                max_expanded_states=self.max_expanded_states,
            )
        )

    def act(
        self,
        observation: Observation,
        info: AgentInfo,
    ) -> Action:
        """Return the next planned action."""

        del observation, info
        if not self._plan:
            raise AgentStopped("BFS plan has no remaining actions")
        return self._plan.popleft()

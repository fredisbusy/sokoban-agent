"""Hybrid planner that guards model proposals with deterministic search."""

from __future__ import annotations

from dataclasses import replace

from sokoban_agent.env.rules import (
    apply_action,
    decode_observation,
    has_static_corner_deadlock,
    observation_for,
)
from sokoban_agent.planning.base import (
    NoSolutionError,
    Planner,
    PlanningContext,
    PlanningOutcome,
    SearchLimitError,
)
from sokoban_agent.planning.bfs import solve_bfs


class SearchGuardPlanner:
    """Use BFS to accept a proposal or replace it with a safe plan."""

    def __init__(
        self,
        primary: Planner,
        *,
        max_expanded_states: int = 100_000,
    ) -> None:
        if max_expanded_states <= 0:
            raise ValueError("max_expanded_states must be positive")
        self.primary = primary
        self.max_expanded_states = max_expanded_states

    @property
    def name(self) -> str:
        """Return a planner name that makes the hybrid policy explicit."""

        return f"graph:hybrid:{self.primary.name}+bfs"

    def reset(self, *, seed: int | None = None) -> None:
        """Reset the primary planner; BFS has no episode-local state."""

        self.primary.reset(seed=seed)

    def plan(self, context: PlanningContext) -> PlanningOutcome:
        """Guard the primary proposal and fall back to BFS when needed."""

        primary = self.primary.plan(context)
        if not primary.actions:
            return self._fallback(context, primary)

        level, state = decode_observation(context.observation)
        move = apply_action(level, state, primary.actions[0])
        if move.invalid_move or has_static_corner_deadlock(level, move.state):
            return self._fallback(context, primary)

        next_observation = observation_for(level, move.state)
        try:
            solve_bfs(
                next_observation,
                max_expanded_states=self.max_expanded_states,
            )
        except (NoSolutionError, SearchLimitError):
            return self._fallback(context, primary)
        return replace(
            primary,
            algorithm_calls=primary.algorithm_calls + 1,
        )

    def _fallback(
        self,
        context: PlanningContext,
        primary: PlanningOutcome,
    ) -> PlanningOutcome:
        try:
            actions = solve_bfs(
                context.observation,
                max_expanded_states=self.max_expanded_states,
            )
        except (NoSolutionError, SearchLimitError) as error:
            return replace(
                primary,
                actions=(),
                error=str(error),
                error_kind="search",
                algorithm_calls=primary.algorithm_calls + 1,
                algorithm_fallbacks=primary.algorithm_fallbacks + 1,
            )
        return replace(
            primary,
            actions=actions,
            error=None,
            error_kind=None,
            algorithm_calls=primary.algorithm_calls + 1,
            algorithm_fallbacks=primary.algorithm_fallbacks + 1,
        )

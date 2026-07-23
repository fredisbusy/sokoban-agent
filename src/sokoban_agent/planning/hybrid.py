"""Hybrid planner that grounds model plans with cached search."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Literal

from sokoban_agent.env import Action
from sokoban_agent.env.rules import (
    apply_action,
    decode_observation,
    has_static_corner_deadlock,
    is_success,
    observation_for,
)
from sokoban_agent.planning.astar import solve_astar_result
from sokoban_agent.planning.base import (
    NoSolutionError,
    Observation,
    Planner,
    PlanningContext,
    PlanningOutcome,
    SearchLimitError,
    SearchResult,
)
from sokoban_agent.planning.bfs import solve_bfs_result

SearchAlgorithm = Literal["astar", "bfs"]
_CacheKey = tuple[tuple[int, ...], bytes]


class SearchGuardPlanner:
    """Validate a primary plan and append a cached grounded search suffix."""

    def __init__(
        self,
        primary: Planner,
        *,
        algorithm: SearchAlgorithm = "astar",
        max_expanded_states: int = 100_000,
    ) -> None:
        if max_expanded_states <= 0:
            raise ValueError("max_expanded_states must be positive")
        self.primary = primary
        self.algorithm = algorithm
        self.max_expanded_states = max_expanded_states
        self._cache: dict[_CacheKey, SearchResult] = {}

    @property
    def name(self) -> str:
        """Return a planner name that makes the hybrid policy explicit."""

        return f"graph:hybrid:{self.primary.name}+{self.algorithm}"

    def reset(self, *, seed: int | None = None) -> None:
        """Reset the primary planner and episode-local search cache."""

        self.primary.reset(seed=seed)
        self._cache.clear()

    def plan(self, context: PlanningContext) -> PlanningOutcome:
        """Ground the complete proposal and reuse the safe search suffix."""

        started_at = perf_counter()
        primary = self.primary.plan(context)
        if not primary.actions:
            return self._fallback(
                context,
                primary,
                started_at=started_at,
                reason=primary.error or "LLM이 행동 계획을 제안하지 못했습니다",
            )

        level, state = decode_observation(context.observation)
        accepted: list[Action] = []
        for index, action in enumerate(primary.actions):
            move = apply_action(level, state, action)
            if move.invalid_move:
                return self._fallback(
                    context,
                    primary,
                    started_at=started_at,
                    reason=(
                        f"LLM 계획의 {index + 1}번째 행동 "
                        f"{action.name}이 막혀 있습니다"
                    ),
                )
            if has_static_corner_deadlock(level, move.state):
                return self._fallback(
                    context,
                    primary,
                    started_at=started_at,
                    reason=(
                        f"LLM 계획의 {index + 1}번째 행동 "
                        f"{action.name}이 데드락을 만듭니다"
                    ),
                )
            state = move.state
            accepted.append(action)
            if is_success(level, state):
                return replace(
                    primary,
                    actions=tuple(accepted),
                    guard_summary=(
                        "LLM 계획만으로 해결 가능하여 A* 보강이 필요 없습니다"
                    ),
                    elapsed_seconds=perf_counter() - started_at,
                )

        next_observation = observation_for(level, state)
        search_started = perf_counter()
        try:
            suffix, cache_hit = self._search(next_observation)
        except (NoSolutionError, SearchLimitError):
            return self._fallback(
                context,
                primary,
                started_at=started_at,
                reason="LLM 계획 이후 상태에서 A*가 해답을 찾지 못했습니다",
                prior_algorithm_calls=1,
                prior_algorithm_elapsed=perf_counter() - search_started,
            )
        return replace(
            primary,
            actions=(*accepted, *suffix.actions),
            guard_summary=(
                "LLM 계획이 안전합니다. "
                f"{self.algorithm.upper()}가 후속 행동 "
                f"{len(suffix.actions)}개를 보강했습니다"
            ),
            algorithm_calls=primary.algorithm_calls + int(not cache_hit),
            algorithm_expanded_states=(
                primary.algorithm_expanded_states
                + (0 if cache_hit else suffix.expanded_states)
            ),
            algorithm_elapsed_seconds=(
                primary.algorithm_elapsed_seconds
                + (0.0 if cache_hit else suffix.elapsed_seconds)
            ),
            elapsed_seconds=perf_counter() - started_at,
        )

    def _fallback(
        self,
        context: PlanningContext,
        primary: PlanningOutcome,
        *,
        started_at: float,
        reason: str,
        prior_algorithm_calls: int = 0,
        prior_algorithm_elapsed: float = 0.0,
    ) -> PlanningOutcome:
        search_started = perf_counter()
        try:
            result, cache_hit = self._search(context.observation)
        except (NoSolutionError, SearchLimitError) as error:
            search_elapsed = perf_counter() - search_started
            return replace(
                primary,
                actions=(),
                error=str(error),
                error_kind="search",
                guard_summary=(
                    f"{reason}. 현재 상태의 {self.algorithm.upper()} "
                    f"대체 계획도 실패했습니다: {error}"
                ),
                algorithm_calls=(
                    primary.algorithm_calls + prior_algorithm_calls + 1
                ),
                algorithm_fallbacks=primary.algorithm_fallbacks + 1,
                algorithm_elapsed_seconds=(
                    primary.algorithm_elapsed_seconds
                    + prior_algorithm_elapsed
                    + search_elapsed
                ),
                elapsed_seconds=perf_counter() - started_at,
            )
        return replace(
            primary,
            actions=result.actions,
            error=None,
            error_kind=None,
            guard_summary=(
                f"{reason}. {self.algorithm.upper()}가 안전한 전체 계획 "
                f"{len(result.actions)}개 행동으로 대체했습니다"
            ),
            algorithm_calls=(
                primary.algorithm_calls
                + prior_algorithm_calls
                + int(not cache_hit)
            ),
            algorithm_fallbacks=primary.algorithm_fallbacks + 1,
            algorithm_expanded_states=(
                primary.algorithm_expanded_states
                + (0 if cache_hit else result.expanded_states)
            ),
            algorithm_elapsed_seconds=(
                primary.algorithm_elapsed_seconds
                + prior_algorithm_elapsed
                + (0.0 if cache_hit else result.elapsed_seconds)
            ),
            elapsed_seconds=perf_counter() - started_at,
        )

    def _search(
        self,
        observation: Observation,
    ) -> tuple[SearchResult, bool]:
        key = (observation.shape, observation.tobytes())
        cached = self._cache.get(key)
        if cached is not None:
            return cached, True
        if self.algorithm == "astar":
            result = solve_astar_result(
                observation,
                max_expanded_states=self.max_expanded_states,
            )
        else:
            result = solve_bfs_result(
                observation,
                max_expanded_states=self.max_expanded_states,
            )
        self._cache[key] = result
        return result, False

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
from sokoban_agent.planning.contracts import (
    NoSolutionError,
    Observation,
    Planner,
    PlanningContext,
    PlanningFailure,
    PlanningOutcome,
    SearchLimitError,
    SearchResult,
)
from sokoban_agent.planning.guards.metrics import guard_metrics
from sokoban_agent.planning.search.astar import solve_astar_result
from sokoban_agent.planning.search.bfs import solve_bfs_result

SearchAlgorithm = Literal["astar", "bfs"]
FallbackPolicy = Literal["none", "full", "always"]
_CacheKey = tuple[tuple[int, ...], bytes]


class SearchGuardPlanner:
    """Validate a primary plan and append a cached grounded search suffix."""

    def __init__(
        self,
        primary: Planner,
        *,
        algorithm: SearchAlgorithm = "astar",
        max_expanded_states: int = 100_000,
        fallback_policy: FallbackPolicy = "full",
        measure_contribution: bool = False,
    ) -> None:
        if max_expanded_states <= 0:
            raise ValueError("max_expanded_states must be positive")
        if fallback_policy not in {"none", "full", "always"}:
            raise ValueError(f"unsupported fallback policy: {fallback_policy}")
        self.primary = primary
        self.algorithm = algorithm
        self.max_expanded_states = max_expanded_states
        self.fallback_policy = fallback_policy
        self.measure_contribution = measure_contribution
        self._cache: dict[_CacheKey, SearchResult] = {}

    @property
    def name(self) -> str:
        """Return a planner name that makes the hybrid policy explicit."""

        suffix = (
            self.algorithm
            if self.fallback_policy == "full"
            else f"{self.algorithm}-{self.fallback_policy}"
        )
        return f"graph:hybrid:{self.primary.name}+{suffix}"

    def reset(self, *, seed: int | None = None) -> None:
        """Reset the primary planner and episode-local search cache."""

        self.primary.reset(seed=seed)
        self._cache.clear()

    def plan(self, context: PlanningContext) -> PlanningOutcome:
        """Ground the complete proposal and reuse the safe search suffix."""

        started_at = perf_counter()
        primary = self.primary.plan(context)
        proposed = primary.proposed_actions or primary.actions
        proposed_count = len(proposed)
        if not primary.actions:
            return self._reject_or_fallback(
                context,
                primary,
                started_at=started_at,
                reason=primary.error or "LLM이 행동 계획을 제안하지 못했습니다",
                proposed_count=proposed_count,
                legal_prefix=0,
            )

        level, state = decode_observation(context.observation)
        accepted: list[Action] = []
        proposal_solves = False
        for index, action in enumerate(primary.actions):
            move = apply_action(level, state, action)
            if move.invalid_move:
                return self._reject_or_fallback(
                    context,
                    primary,
                    started_at=started_at,
                    reason=(
                        f"LLM 계획의 {index + 1}번째 행동 "
                        f"{action.name}이 막혀 있습니다"
                    ),
                    proposed_count=proposed_count,
                    legal_prefix=len(accepted),
                )
            if has_static_corner_deadlock(level, move.state):
                return self._reject_or_fallback(
                    context,
                    primary,
                    started_at=started_at,
                    reason=(
                        f"LLM 계획의 {index + 1}번째 행동 "
                        f"{action.name}이 데드락을 만듭니다"
                    ),
                    proposed_count=proposed_count,
                    legal_prefix=len(accepted),
                )
            state = move.state
            accepted.append(action)
            if is_success(level, state):
                proposal_solves = True
                break

        if self.fallback_policy == "always":
            return self._fallback(
                context,
                primary,
                started_at=started_at,
                reason="진단 대조군 정책이 LLM 계획을 사용하지 않습니다",
                proposed_count=proposed_count,
                legal_prefix=len(accepted),
            )

        if proposal_solves:
            reference, reference_called = self._measure_reference(context)
            return replace(
                primary,
                actions=tuple(accepted),
                narrative=replace(
                    primary.narrative,
                    guard_summary=(
                        "LLM 계획만으로 해결 가능하여 "
                        "A* 보강이 필요 없습니다"
                    ),
                ),
                guard=guard_metrics(
                    reference,
                    disposition="accepted",
                    proposed_actions=proposed_count,
                    legal_prefix_actions=len(accepted),
                    adopted_actions=len(accepted),
                    reference_called=reference_called,
                    suffix_expanded_states=0,
                ),
                elapsed_seconds=perf_counter() - started_at,
            )

        next_observation = observation_for(level, state)
        search_started = perf_counter()
        try:
            suffix, cache_hit = self._search(next_observation)
        except (NoSolutionError, SearchLimitError):
            return self._reject_or_fallback(
                context,
                primary,
                started_at=started_at,
                reason="LLM 계획 이후 상태에서 A*가 해답을 찾지 못했습니다",
                proposed_count=proposed_count,
                legal_prefix=len(accepted),
                prior_algorithm_calls=1,
                prior_algorithm_requests=1,
                prior_algorithm_failures=1,
                prior_algorithm_elapsed=perf_counter() - search_started,
            )
        reference, reference_called = self._measure_reference(context)
        return replace(
            primary,
            actions=(*accepted, *suffix.actions),
            narrative=replace(
                primary.narrative,
                guard_summary=(
                    "LLM 계획이 안전합니다. "
                    f"{self.algorithm.upper()}가 후속 행동 "
                    f"{len(suffix.actions)}개를 보강했습니다"
                ),
            ),
            algorithm=primary.algorithm.plus(
                calls=int(not cache_hit),
                requests=1,
                cache_hits=int(cache_hit),
                expanded_states=0 if cache_hit else suffix.expanded_states,
                elapsed_seconds=0.0 if cache_hit else suffix.elapsed_seconds,
            ),
            guard=guard_metrics(
                reference,
                disposition="suffix_added",
                proposed_actions=proposed_count,
                legal_prefix_actions=len(accepted),
                adopted_actions=len(accepted),
                reference_called=reference_called,
                suffix_expanded_states=suffix.expanded_states,
            ),
            elapsed_seconds=perf_counter() - started_at,
        )

    def _reject_or_fallback(
        self,
        context: PlanningContext,
        primary: PlanningOutcome,
        *,
        started_at: float,
        reason: str,
        proposed_count: int,
        legal_prefix: int,
        prior_algorithm_calls: int = 0,
        prior_algorithm_requests: int = 0,
        prior_algorithm_failures: int = 0,
        prior_algorithm_elapsed: float = 0.0,
    ) -> PlanningOutcome:
        if self.fallback_policy != "none":
            return self._fallback(
                context,
                primary,
                started_at=started_at,
                reason=reason,
                proposed_count=proposed_count,
                legal_prefix=legal_prefix,
                prior_algorithm_calls=prior_algorithm_calls,
                prior_algorithm_requests=prior_algorithm_requests,
                prior_algorithm_failures=prior_algorithm_failures,
                prior_algorithm_elapsed=prior_algorithm_elapsed,
            )
        return replace(
            primary,
            actions=(),
            failure=PlanningFailure(reason, "search"),
            narrative=replace(
                primary.narrative,
                guard_summary=(
                    f"{reason}. suffix-only 정책은 "
                    "전체 대체를 하지 않습니다"
                ),
            ),
            guard=guard_metrics(
                None,
                disposition="failed",
                proposed_actions=proposed_count,
                legal_prefix_actions=legal_prefix,
                adopted_actions=0,
                reference_called=False,
                suffix_expanded_states=0,
            ),
            algorithm=primary.algorithm.plus(
                calls=prior_algorithm_calls,
                requests=prior_algorithm_requests,
                failures=prior_algorithm_failures,
                elapsed_seconds=prior_algorithm_elapsed,
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
        proposed_count: int,
        legal_prefix: int,
        prior_algorithm_calls: int = 0,
        prior_algorithm_requests: int = 0,
        prior_algorithm_failures: int = 0,
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
                failure=PlanningFailure(str(error), "search"),
                narrative=replace(
                    primary.narrative,
                    guard_summary=(
                        f"{reason}. 현재 상태의 {self.algorithm.upper()} "
                        f"대체 계획도 실패했습니다: {error}"
                    ),
                ),
                guard=guard_metrics(
                    None,
                    disposition="failed",
                    proposed_actions=proposed_count,
                    legal_prefix_actions=legal_prefix,
                    adopted_actions=0,
                    reference_called=False,
                    suffix_expanded_states=0,
                ),
                algorithm=primary.algorithm.plus(
                    calls=prior_algorithm_calls + 1,
                    requests=prior_algorithm_requests + 1,
                    failures=prior_algorithm_failures + 1,
                    fallbacks=1,
                    elapsed_seconds=prior_algorithm_elapsed + search_elapsed,
                ),
                elapsed_seconds=perf_counter() - started_at,
            )
        return replace(
            primary,
            actions=result.actions,
            failure=None,
            narrative=replace(
                primary.narrative,
                guard_summary=(
                    f"{reason}. {self.algorithm.upper()}가 안전한 전체 계획 "
                    f"{len(result.actions)}개 행동으로 대체했습니다"
                ),
            ),
            algorithm=primary.algorithm.plus(
                calls=prior_algorithm_calls + int(not cache_hit),
                requests=prior_algorithm_requests + 1,
                cache_hits=int(cache_hit),
                failures=prior_algorithm_failures,
                fallbacks=1,
                expanded_states=0 if cache_hit else result.expanded_states,
                elapsed_seconds=(
                    prior_algorithm_elapsed
                    + (0.0 if cache_hit else result.elapsed_seconds)
                ),
            ),
            guard=guard_metrics(
                result if self.measure_contribution else None,
                disposition="replaced",
                proposed_actions=proposed_count,
                legal_prefix_actions=legal_prefix,
                adopted_actions=0,
                reference_called=self.measure_contribution,
                suffix_expanded_states=0,
                contribution=False,
                diagnostic_elapsed=False,
            ),
            elapsed_seconds=perf_counter() - started_at,
        )

    def _measure_reference(
        self,
        context: PlanningContext,
    ) -> tuple[SearchResult | None, bool]:
        if not self.measure_contribution:
            return None, False
        try:
            return self._solve(context.observation), True
        except (NoSolutionError, SearchLimitError):
            return None, True

    def _search(
        self,
        observation: Observation,
    ) -> tuple[SearchResult, bool]:
        key = (observation.shape, observation.tobytes())
        cached = self._cache.get(key)
        if cached is not None:
            return cached, True
        result = self._solve(observation)
        self._cache[key] = result
        return result, False

    def _solve(self, observation: Observation) -> SearchResult:
        if self.algorithm == "astar":
            return solve_astar_result(
                observation,
                max_expanded_states=self.max_expanded_states,
            )
        return solve_bfs_result(
            observation,
            max_expanded_states=self.max_expanded_states,
        )

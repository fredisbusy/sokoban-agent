"""Immutable experiment results and dependency-free summary metrics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean, median


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Raw measurements from one agent, level, and seed combination."""

    planner_name: str
    level_id: str
    seed: int | None
    success: bool
    deadlock: bool
    truncated: bool
    action_count: int
    invalid_moves: int
    total_reward: float
    elapsed_seconds: float
    failure_reason: str | None = None
    planning_calls: int = 0
    planning_retries: int = 0
    planning_errors: int = 0
    planning_elapsed_seconds: float = 0.0
    algorithm_calls: int = 0
    algorithm_requests: int = 0
    algorithm_cache_hits: int = 0
    algorithm_failures: int = 0
    algorithm_fallbacks: int = 0
    algorithm_expanded_states: int = 0
    algorithm_elapsed_seconds: float = 0.0
    llm_calls: int = 0
    llm_retries: int = 0
    llm_client_errors: int = 0
    llm_format_errors: int = 0
    llm_invalid_actions: int = 0
    llm_elapsed_seconds: float = 0.0
    llm_load_seconds: float = 0.0
    llm_prompt_eval_seconds: float = 0.0
    llm_eval_seconds: float = 0.0
    llm_prompt_tokens: int = 0
    llm_output_tokens: int = 0
    push_count: int = 0
    revisited_states: int = 0
    repeated_plans: int = 0
    guard_accepted: int = 0
    guard_suffix_added: int = 0
    guard_replaced: int = 0
    guard_failed: int = 0
    guard_proposed_actions: int = 0
    guard_legal_prefix_actions: int = 0
    guard_adopted_actions: int = 0
    guard_suffix_expanded_states: int = 0
    guard_reference_calls: int = 0
    guard_reference_action_count: int = 0
    guard_reference_expanded_states: int = 0
    guard_reference_elapsed_seconds: float = 0.0
    guard_expansions_saved: int = 0
    reference_solved: bool = False
    reference_action_count: int | None = None
    reference_push_count: int | None = None
    reference_expanded_states: int = 0
    reference_elapsed_seconds: float = 0.0
    action_overhead_vs_reference: int | None = None
    push_overhead_vs_reference: int | None = None
    policy_elapsed_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class PlannerSummary:
    """Aggregate metrics for one agent."""

    planner_name: str
    episode_count: int
    success_count: int
    success_rate: float
    deadlock_count: int
    deadlock_rate: float
    truncated_count: int
    mean_actions: float
    mean_actions_on_success: float | None
    mean_invalid_moves: float
    mean_elapsed_seconds: float
    p50_elapsed_seconds: float
    p95_elapsed_seconds: float
    total_planning_calls: int
    total_planning_retries: int
    total_planning_errors: int
    mean_planning_elapsed_seconds: float
    total_algorithm_calls: int
    total_algorithm_requests: int
    total_algorithm_cache_hits: int
    total_algorithm_failures: int
    total_algorithm_fallbacks: int
    total_algorithm_expanded_states: int
    mean_algorithm_elapsed_seconds: float
    total_llm_calls: int
    total_llm_retries: int
    total_llm_client_errors: int
    total_llm_format_errors: int
    total_llm_invalid_actions: int
    mean_llm_elapsed_seconds: float
    p50_llm_elapsed_seconds: float
    p95_llm_elapsed_seconds: float
    total_llm_prompt_tokens: int
    total_llm_output_tokens: int
    llm_output_tokens_per_second: float | None
    mean_pushes_on_success: float | None
    total_revisited_states: int
    total_repeated_plans: int
    total_guard_accepted: int
    total_guard_suffix_added: int
    total_guard_replaced: int
    total_guard_failed: int
    total_guard_proposed_actions: int
    total_guard_legal_prefix_actions: int
    total_guard_adopted_actions: int
    guard_adoption_rate: float | None
    total_guard_suffix_expanded_states: int
    total_guard_reference_calls: int
    total_guard_reference_expanded_states: int
    total_guard_expansions_saved: int
    reference_solved_count: int
    mean_action_overhead_vs_reference: float | None
    mean_push_overhead_vs_reference: float | None
    mean_policy_elapsed_seconds: float


def summarize_by_planner(
    results: Sequence[EpisodeResult],
) -> list[PlannerSummary]:
    """Aggregate episode results while preserving agent encounter order."""

    grouped: dict[str, list[EpisodeResult]] = {}
    for result in results:
        grouped.setdefault(result.planner_name, []).append(result)

    summaries: list[PlannerSummary] = []
    for planner_name, episodes in grouped.items():
        successes = [episode for episode in episodes if episode.success]
        episode_count = len(episodes)
        success_count = len(successes)
        deadlock_count = sum(episode.deadlock for episode in episodes)
        truncated_count = sum(episode.truncated for episode in episodes)
        summaries.append(
            PlannerSummary(
                planner_name=planner_name,
                episode_count=episode_count,
                success_count=success_count,
                success_rate=success_count / episode_count,
                deadlock_count=deadlock_count,
                deadlock_rate=deadlock_count / episode_count,
                truncated_count=truncated_count,
                mean_actions=fmean(
                    episode.action_count for episode in episodes
                ),
                mean_actions_on_success=(
                    fmean(episode.action_count for episode in successes)
                    if successes
                    else None
                ),
                mean_invalid_moves=fmean(
                    episode.invalid_moves for episode in episodes
                ),
                mean_elapsed_seconds=fmean(
                    episode.elapsed_seconds for episode in episodes
                ),
                p50_elapsed_seconds=median(
                    episode.elapsed_seconds for episode in episodes
                ),
                p95_elapsed_seconds=_percentile(
                    [episode.elapsed_seconds for episode in episodes],
                    0.95,
                ),
                total_planning_calls=sum(
                    episode.planning_calls for episode in episodes
                ),
                total_planning_retries=sum(
                    episode.planning_retries for episode in episodes
                ),
                total_planning_errors=sum(
                    episode.planning_errors for episode in episodes
                ),
                mean_planning_elapsed_seconds=fmean(
                    episode.planning_elapsed_seconds for episode in episodes
                ),
                total_algorithm_calls=sum(
                    episode.algorithm_calls for episode in episodes
                ),
                total_algorithm_requests=sum(
                    episode.algorithm_requests for episode in episodes
                ),
                total_algorithm_cache_hits=sum(
                    episode.algorithm_cache_hits for episode in episodes
                ),
                total_algorithm_failures=sum(
                    episode.algorithm_failures for episode in episodes
                ),
                total_algorithm_fallbacks=sum(
                    episode.algorithm_fallbacks for episode in episodes
                ),
                total_algorithm_expanded_states=sum(
                    episode.algorithm_expanded_states for episode in episodes
                ),
                mean_algorithm_elapsed_seconds=fmean(
                    episode.algorithm_elapsed_seconds for episode in episodes
                ),
                total_llm_calls=sum(
                    episode.llm_calls for episode in episodes
                ),
                total_llm_retries=sum(
                    episode.llm_retries for episode in episodes
                ),
                total_llm_client_errors=sum(
                    episode.llm_client_errors for episode in episodes
                ),
                total_llm_format_errors=sum(
                    episode.llm_format_errors for episode in episodes
                ),
                total_llm_invalid_actions=sum(
                    episode.llm_invalid_actions for episode in episodes
                ),
                mean_llm_elapsed_seconds=fmean(
                    episode.llm_elapsed_seconds for episode in episodes
                ),
                p50_llm_elapsed_seconds=median(
                    episode.llm_elapsed_seconds for episode in episodes
                ),
                p95_llm_elapsed_seconds=_percentile(
                    [episode.llm_elapsed_seconds for episode in episodes],
                    0.95,
                ),
                total_llm_prompt_tokens=sum(
                    episode.llm_prompt_tokens for episode in episodes
                ),
                total_llm_output_tokens=sum(
                    episode.llm_output_tokens for episode in episodes
                ),
                llm_output_tokens_per_second=(
                    sum(episode.llm_output_tokens for episode in episodes)
                    / sum(episode.llm_eval_seconds for episode in episodes)
                    if sum(
                        episode.llm_eval_seconds for episode in episodes
                    )
                    > 0
                    else None
                ),
                mean_pushes_on_success=(
                    fmean(episode.push_count for episode in successes)
                    if successes
                    else None
                ),
                total_revisited_states=sum(
                    episode.revisited_states for episode in episodes
                ),
                total_repeated_plans=sum(
                    episode.repeated_plans for episode in episodes
                ),
                total_guard_accepted=sum(
                    episode.guard_accepted for episode in episodes
                ),
                total_guard_suffix_added=sum(
                    episode.guard_suffix_added for episode in episodes
                ),
                total_guard_replaced=sum(
                    episode.guard_replaced for episode in episodes
                ),
                total_guard_failed=sum(
                    episode.guard_failed for episode in episodes
                ),
                total_guard_proposed_actions=sum(
                    episode.guard_proposed_actions for episode in episodes
                ),
                total_guard_legal_prefix_actions=sum(
                    episode.guard_legal_prefix_actions for episode in episodes
                ),
                total_guard_adopted_actions=sum(
                    episode.guard_adopted_actions for episode in episodes
                ),
                guard_adoption_rate=_ratio(
                    sum(
                        episode.guard_adopted_actions
                        for episode in episodes
                    ),
                    sum(
                        episode.guard_proposed_actions
                        for episode in episodes
                    ),
                ),
                total_guard_suffix_expanded_states=sum(
                    episode.guard_suffix_expanded_states
                    for episode in episodes
                ),
                total_guard_reference_calls=sum(
                    episode.guard_reference_calls for episode in episodes
                ),
                total_guard_reference_expanded_states=sum(
                    episode.guard_reference_expanded_states
                    for episode in episodes
                ),
                total_guard_expansions_saved=sum(
                    episode.guard_expansions_saved for episode in episodes
                ),
                reference_solved_count=sum(
                    episode.reference_solved for episode in episodes
                ),
                mean_action_overhead_vs_reference=_optional_mean(
                    [
                        episode.action_overhead_vs_reference
                        for episode in episodes
                    ]
                ),
                mean_push_overhead_vs_reference=_optional_mean(
                    [
                        episode.push_overhead_vs_reference
                        for episode in episodes
                    ]
                ),
                mean_policy_elapsed_seconds=fmean(
                    episode.policy_elapsed_seconds for episode in episodes
                ),
            )
        )
    return summaries


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _optional_mean(values: list[int | None]) -> float | None:
    present = [value for value in values if value is not None]
    return fmean(present) if present else None

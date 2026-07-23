"""Immutable experiment results and dependency-free summary metrics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean


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
    algorithm_fallbacks: int = 0
    llm_calls: int = 0
    llm_retries: int = 0
    llm_client_errors: int = 0
    llm_format_errors: int = 0
    llm_invalid_actions: int = 0
    llm_elapsed_seconds: float = 0.0


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
    total_planning_calls: int
    total_planning_retries: int
    total_planning_errors: int
    mean_planning_elapsed_seconds: float
    total_algorithm_calls: int
    total_algorithm_fallbacks: int
    total_llm_calls: int
    total_llm_retries: int
    total_llm_client_errors: int
    total_llm_format_errors: int
    total_llm_invalid_actions: int
    mean_llm_elapsed_seconds: float


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
                total_algorithm_fallbacks=sum(
                    episode.algorithm_fallbacks for episode in episodes
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
            )
        )
    return summaries

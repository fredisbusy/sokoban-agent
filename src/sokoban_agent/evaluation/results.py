"""Immutable experiment results and dependency-free summary metrics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Raw measurements from one agent, level, and seed combination."""

    agent_name: str
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
    llm_calls: int = 0
    llm_retries: int = 0
    llm_client_errors: int = 0
    llm_format_errors: int = 0
    llm_invalid_actions: int = 0
    llm_elapsed_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class AgentSummary:
    """Aggregate metrics for one agent."""

    agent_name: str
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
    total_llm_calls: int
    total_llm_retries: int
    total_llm_client_errors: int
    total_llm_format_errors: int
    total_llm_invalid_actions: int
    mean_llm_elapsed_seconds: float


def summarize_by_agent(
    results: Sequence[EpisodeResult],
) -> list[AgentSummary]:
    """Aggregate episode results while preserving agent encounter order."""

    grouped: dict[str, list[EpisodeResult]] = {}
    for result in results:
        grouped.setdefault(result.agent_name, []).append(result)

    summaries: list[AgentSummary] = []
    for agent_name, episodes in grouped.items():
        successes = [episode for episode in episodes if episode.success]
        episode_count = len(episodes)
        success_count = len(successes)
        deadlock_count = sum(episode.deadlock for episode in episodes)
        truncated_count = sum(episode.truncated for episode in episodes)
        summaries.append(
            AgentSummary(
                agent_name=agent_name,
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

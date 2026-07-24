"""Immutable baseline result aggregation by measurement responsibility."""

from __future__ import annotations

from collections.abc import Sequence
from statistics import fmean, median

from sokoban_agent.evaluation.schemas.baseline import EpisodeResult
from sokoban_agent.evaluation.schemas.baseline_summary import (
    ActionSummary,
    AlgorithmSummary,
    GuardSummary,
    LatencySummary,
    LLMSummary,
    OutcomeSummary,
    PlannerSummary,
    PlanningSummary,
    ReferenceSummary,
)


def summarize_by_planner(
    results: Sequence[EpisodeResult],
) -> list[PlannerSummary]:
    """Aggregate episode results while preserving planner encounter order."""

    grouped: dict[str, list[EpisodeResult]] = {}
    for result in results:
        grouped.setdefault(result.identity.planner_name, []).append(result)
    return [
        _summarize(planner_name, episodes)
        for planner_name, episodes in grouped.items()
    ]


def _summarize(
    planner_name: str,
    episodes: list[EpisodeResult],
) -> PlannerSummary:
    successes = [item for item in episodes if item.outcome.success]
    outcomes = [item.outcome for item in episodes]
    plannings = [item.planning for item in episodes]
    algorithms = [item.algorithm for item in episodes]
    llms = [item.llm for item in episodes]
    guards = [item.guard for item in episodes]
    elapsed = [item.timing.elapsed_seconds for item in episodes]
    return PlannerSummary(
        planner_name=planner_name,
        outcome=OutcomeSummary(
            episode_count=len(episodes),
            success_count=len(successes),
            deadlock_count=sum(item.deadlock for item in outcomes),
            truncated_count=sum(item.truncated for item in outcomes),
        ),
        actions=ActionSummary(
            mean_actions=fmean(item.action_count for item in outcomes),
            mean_actions_on_success=_mean(
                [item.outcome.action_count for item in successes]
            ),
            mean_invalid_moves=fmean(
                item.invalid_moves for item in outcomes
            ),
            mean_pushes_on_success=_mean(
                [item.outcome.push_count for item in successes]
            ),
            total_revisited_states=sum(
                item.revisited_states for item in outcomes
            ),
            total_repeated_plans=sum(
                item.repeated_plans for item in outcomes
            ),
        ),
        timing=LatencySummary(
            mean_elapsed_seconds=fmean(elapsed),
            p50_elapsed_seconds=median(elapsed),
            p95_elapsed_seconds=_percentile(elapsed, 0.95),
            mean_policy_elapsed_seconds=fmean(
                item.policy_elapsed_seconds for item in episodes
            ),
        ),
        planning=PlanningSummary(
            total_calls=sum(item.calls for item in plannings),
            total_retries=sum(item.retries for item in plannings),
            total_errors=sum(item.errors for item in plannings),
            mean_elapsed_seconds=fmean(
                item.elapsed_seconds for item in plannings
            ),
        ),
        algorithm=AlgorithmSummary(
            total_calls=sum(item.calls for item in algorithms),
            total_requests=sum(item.requests for item in algorithms),
            total_cache_hits=sum(item.cache_hits for item in algorithms),
            total_failures=sum(item.failures for item in algorithms),
            total_fallbacks=sum(item.fallbacks for item in algorithms),
            total_expanded_states=sum(
                item.expanded_states for item in algorithms
            ),
            mean_elapsed_seconds=fmean(
                item.elapsed_seconds for item in algorithms
            ),
        ),
        llm=LLMSummary(
            total_calls=sum(item.calls for item in llms),
            total_retries=sum(item.retries for item in llms),
            total_client_errors=sum(item.client_errors for item in llms),
            total_format_errors=sum(item.format_errors for item in llms),
            total_invalid_actions=sum(item.invalid_actions for item in llms),
            mean_elapsed_seconds=fmean(
                item.elapsed_seconds for item in llms
            ),
            p50_elapsed_seconds=median(
                item.elapsed_seconds for item in llms
            ),
            p95_elapsed_seconds=_percentile(
                [item.elapsed_seconds for item in llms], 0.95
            ),
            total_prompt_tokens=sum(item.prompt_tokens for item in llms),
            total_output_tokens=sum(item.output_tokens for item in llms),
            total_eval_seconds=sum(item.eval_seconds for item in llms),
        ),
        guard=GuardSummary(
            total_accepted=sum(item.accepted for item in guards),
            total_suffix_added=sum(item.suffix_added for item in guards),
            total_replaced=sum(item.replaced for item in guards),
            total_failed=sum(item.failed for item in guards),
            total_proposed_actions=sum(
                item.proposed_actions for item in guards
            ),
            total_legal_prefix_actions=sum(
                item.legal_prefix_actions for item in guards
            ),
            total_adopted_actions=sum(
                item.adopted_actions for item in guards
            ),
            total_suffix_expanded_states=sum(
                item.suffix_expanded_states for item in guards
            ),
            total_reference_calls=sum(
                item.reference_calls for item in guards
            ),
            total_reference_expanded_states=sum(
                item.reference_expanded_states for item in guards
            ),
            total_expansions_saved=sum(
                item.expansions_saved for item in guards
            ),
        ),
        reference=ReferenceSummary(
            solved_count=sum(item.reference.solved for item in episodes),
            mean_action_overhead=_optional_mean(
                [item.action_overhead_vs_reference for item in episodes]
            ),
            mean_push_overhead=_optional_mean(
                [item.push_overhead_vs_reference for item in episodes]
            ),
        ),
    )


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _mean(values: list[int]) -> float | None:
    return fmean(values) if values else None


def _optional_mean(values: list[int | None]) -> float | None:
    present = [value for value in values if value is not None]
    return fmean(present) if present else None


__all__ = ["EpisodeResult", "PlannerSummary", "summarize_by_planner"]

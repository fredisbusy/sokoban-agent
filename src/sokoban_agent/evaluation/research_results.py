"""Normalized records and aggregates for agentic research comparisons."""

from __future__ import annotations

from statistics import fmean

from sokoban_agent.evaluation.schemas.research import (
    RationaleIntervention as RationaleIntervention,
)
from sokoban_agent.evaluation.schemas.research import (
    ResearchEpisodeRecord as ResearchEpisodeRecord,
)
from sokoban_agent.evaluation.schemas.research import (
    ResearchExperiment as ResearchExperiment,
)
from sokoban_agent.evaluation.schemas.research import (
    ResearchPolicySummary as ResearchPolicySummary,
)

POLICY_NAMES = (
    "primitive-llm",
    "structured-llm",
    "structured-local-search",
    "structured-no-rationale",
    "current-full-guard",
    "astar-oracle",
)


def summarize_research(
    records: tuple[ResearchEpisodeRecord, ...],
) -> tuple[ResearchPolicySummary, ...]:
    """Aggregate every declared policy in stable comparison order."""

    summaries = []
    for policy_name in POLICY_NAMES:
        episodes = [r for r in records if r.policy_name == policy_name]
        strategies = [r.strategy for r in episodes if r.strategy is not None]
        attempts = sum(item.subgoal_attempts for item in strategies)
        effects = sum(
            item.effect_matches + item.effect_mismatches
            for item in strategies
        )
        summaries.append(
            ResearchPolicySummary(
                policy_name=policy_name,
                episode_count=len(episodes),
                success_rate=(
                    sum(r.outcome.success for r in episodes) / len(episodes)
                ),
                mean_actions=fmean(r.action_count for r in episodes),
                subgoal_success_rate=_ratio(
                    sum(r.subgoal_successes for r in episodes), attempts
                ),
                effect_match_rate=_ratio(
                    sum(item.effect_matches for item in strategies), effects
                ),
                total_protected_violations=sum(
                    item.protected_constraint_violations
                    for item in strategies
                ),
                total_llm_calls=sum(r.llm.calls for r in episodes if r.llm),
                total_llm_tokens=sum(
                    r.llm.prompt_tokens + r.llm.output_tokens
                    for r in episodes
                    if r.llm
                ),
                total_memory_requests=sum(
                    r.memory.requests for r in episodes if r.memory
                ),
                total_memory_hits=sum(
                    r.memory.hits for r in episodes if r.memory
                ),
                total_memory_writes=sum(
                    r.memory.writes for r in episodes if r.memory
                ),
                total_llm_calls_saved=sum(
                    r.memory.llm_calls_saved for r in episodes if r.memory
                ),
                total_rule_checks=sum(
                    r.rules.checks for r in episodes if r.rules
                ),
                total_reachability_calls=sum(
                    r.rules.reachability_calls for r in episodes if r.rules
                ),
                total_local_search_calls=sum(
                    r.local_search.calls
                    for r in episodes
                    if r.local_search
                ),
                total_local_expanded_states=sum(
                    r.local_search.expanded_states
                    for r in episodes
                    if r.local_search
                ),
                total_algorithm_calls=sum(
                    r.algorithm.calls for r in episodes if r.algorithm
                ),
                total_algorithm_expanded_states=sum(
                    r.algorithm.expanded_states
                    for r in episodes
                    if r.algorithm
                ),
                mean_policy_elapsed_seconds=fmean(
                    r.policy_elapsed_seconds for r in episodes
                ),
            )
        )
    return tuple(summaries)


def measure_rationale_intervention(
    records: tuple[ResearchEpisodeRecord, ...],
) -> RationaleIntervention:
    """Compare exact behavior with and without natural-language rationale."""

    with_rationale = {
        (r.level_id, r.seed): r
        for r in records
        if r.policy_name == "structured-local-search"
    }
    without_rationale = {
        (r.level_id, r.seed): r
        for r in records
        if r.policy_name == "structured-no-rationale"
    }
    keys = sorted(with_rationale.keys() & without_rationale.keys())
    return RationaleIntervention(
        compared_cases=len(keys),
        action_sequence_changes=sum(
            with_rationale[key].outcome.action_sequence
            != without_rationale[key].outcome.action_sequence
            for key in keys
        ),
        success_changes=sum(
            with_rationale[key].outcome.success
            != without_rationale[key].outcome.success
            for key in keys
        ),
        mean_action_count_delta=fmean(
            without_rationale[key].action_count
            - with_rationale[key].action_count
            for key in keys
        ),
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None

"""Checkpoint-safe metric updates for the baseline Sokoban graph."""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
from typing import cast

from sokoban_agent.graph.baseline.state import (
    AlgorithmMetricsState,
    BaselineMetrics,
    EpisodeMetricsState,
    GuardMetricsState,
    LLMMetricsState,
    PlanningMetricsState,
    SokobanGraphState,
)
from sokoban_agent.planning import Observation, PlanningOutcome


def initial_baseline_metrics() -> BaselineMetrics:
    """Return the complete nested metric state for one episode."""

    return {
        "episode": {
            "validation_rejections": 0,
            "total_reward": 0.0,
            "push_count": 0,
            "revisited_states": 0,
            "repeated_plans": 0,
        },
        "planning": {
            "calls": 0,
            "retries": 0,
            "errors": 0,
            "elapsed_seconds": 0.0,
        },
        "algorithm": {
            "calls": 0,
            "requests": 0,
            "cache_hits": 0,
            "failures": 0,
            "fallbacks": 0,
            "expanded_states": 0,
            "elapsed_seconds": 0.0,
        },
        "guard": {
            "accepted": 0,
            "suffix_added": 0,
            "replaced": 0,
            "failed": 0,
            "proposed_actions": 0,
            "legal_prefix_actions": 0,
            "adopted_actions": 0,
            "suffix_expanded_states": 0,
            "reference_calls": 0,
            "reference_action_count": 0,
            "reference_expanded_states": 0,
            "reference_elapsed_seconds": 0.0,
            "expansions_saved": 0,
        },
        "llm": {
            "calls": 0,
            "retries": 0,
            "client_errors": 0,
            "format_errors": 0,
            "invalid_actions": 0,
            "elapsed_seconds": 0.0,
            "load_seconds": 0.0,
            "prompt_eval_seconds": 0.0,
            "eval_seconds": 0.0,
            "prompt_tokens": 0,
            "output_tokens": 0,
        },
    }


def update_baseline_metrics(
    metrics: BaselineMetrics,
    *,
    episode: Mapping[str, int | float] | None = None,
    planning: Mapping[str, int | float] | None = None,
    algorithm: Mapping[str, int | float] | None = None,
    guard: Mapping[str, int | float] | None = None,
    llm: Mapping[str, int | float] | None = None,
) -> BaselineMetrics:
    """Replace selected absolute values while retaining every metric key."""

    return {
        "episode": cast(
            EpisodeMetricsState, {**metrics["episode"], **(episode or {})}
        ),
        "planning": cast(
            PlanningMetricsState, {**metrics["planning"], **(planning or {})}
        ),
        "algorithm": cast(
            AlgorithmMetricsState, {**metrics["algorithm"], **(algorithm or {})}
        ),
        "guard": cast(
            GuardMetricsState, {**metrics["guard"], **(guard or {})}
        ),
        "llm": cast(LLMMetricsState, {**metrics["llm"], **(llm or {})}),
    }


def observation_key(observation: Observation) -> str:
    """Hash one complete board state without retaining its raw bytes."""

    digest = sha256()
    digest.update(str(observation.shape).encode())
    digest.update(observation.tobytes())
    return digest.hexdigest()


def proposal_key(
    observation: Observation,
    actions: tuple[str, ...],
) -> str:
    """Hash a non-empty proposal together with the state that produced it."""

    digest = sha256()
    digest.update(observation_key(observation).encode())
    digest.update(",".join(actions).encode())
    return digest.hexdigest()


def planning_research_update(
    state: SokobanGraphState,
    observation: Observation,
    outcome: PlanningOutcome,
    *,
    elapsed_seconds: float,
    error: bool,
    retry: bool,
) -> dict[str, object]:
    """Accumulate planning, proposal, guard, search, and LLM measurements."""

    proposed = tuple(
        action.name for action in (outcome.proposed_actions or outcome.actions)
    )
    key = proposal_key(observation, proposed) if proposed else None
    repeated = key is not None and key in state["seen_plan_keys"]
    seen = state["seen_plan_keys"]
    if key is not None and not repeated:
        seen = [*seen, key]

    metrics = state["metrics"]
    episode = metrics["episode"]
    planning = metrics["planning"]
    algorithm = metrics["algorithm"]
    guard = metrics["guard"]
    llm = metrics["llm"]
    disposition = outcome.guard_disposition
    return {
        "seen_plan_keys": seen,
        "metrics": update_baseline_metrics(
            metrics,
            episode={
                "repeated_plans": episode["repeated_plans"] + int(repeated)
            },
            planning={
                "calls": planning["calls"] + 1,
                "retries": planning["retries"] + int(retry),
                "errors": planning["errors"] + int(error),
                "elapsed_seconds": planning["elapsed_seconds"] + elapsed_seconds,
            },
            algorithm={
                "calls": algorithm["calls"] + outcome.algorithm_calls,
                "requests": algorithm["requests"] + outcome.algorithm_requests,
                "cache_hits": (
                    algorithm["cache_hits"] + outcome.algorithm_cache_hits
                ),
                "failures": algorithm["failures"] + outcome.algorithm_failures,
                "fallbacks": (
                    algorithm["fallbacks"] + outcome.algorithm_fallbacks
                ),
                "expanded_states": (
                    algorithm["expanded_states"]
                    + outcome.algorithm_expanded_states
                ),
                "elapsed_seconds": (
                    algorithm["elapsed_seconds"]
                    + outcome.algorithm_elapsed_seconds
                ),
            },
            guard={
                "accepted": guard["accepted"]
                + int(disposition == "accepted"),
                "suffix_added": guard["suffix_added"]
                + int(disposition == "suffix_added"),
                "replaced": guard["replaced"]
                + int(disposition == "replaced"),
                "failed": guard["failed"] + int(disposition == "failed"),
                "proposed_actions": (
                    guard["proposed_actions"] + outcome.guard_proposed_actions
                ),
                "legal_prefix_actions": (
                    guard["legal_prefix_actions"]
                    + outcome.guard_legal_prefix_actions
                ),
                "adopted_actions": (
                    guard["adopted_actions"] + outcome.guard_adopted_actions
                ),
                "suffix_expanded_states": (
                    guard["suffix_expanded_states"]
                    + outcome.guard_suffix_expanded_states
                ),
                "reference_calls": (
                    guard["reference_calls"] + outcome.guard_reference_calls
                ),
                "reference_action_count": (
                    guard["reference_action_count"]
                    + outcome.guard_reference_action_count
                ),
                "reference_expanded_states": (
                    guard["reference_expanded_states"]
                    + outcome.guard_reference_expanded_states
                ),
                "reference_elapsed_seconds": (
                    guard["reference_elapsed_seconds"]
                    + outcome.guard_reference_elapsed_seconds
                ),
                "expansions_saved": (
                    guard["expansions_saved"] + outcome.guard_expansions_saved
                ),
            },
            llm={
                "calls": llm["calls"] + outcome.llm_calls,
                "retries": (
                    llm["retries"] + int(retry and outcome.llm_calls > 0)
                ),
                "client_errors": (
                    llm["client_errors"] + outcome.llm_client_errors
                ),
                "format_errors": (
                    llm["format_errors"] + outcome.llm_format_errors
                ),
                "elapsed_seconds": (
                    llm["elapsed_seconds"] + outcome.llm_elapsed_seconds
                ),
                "load_seconds": llm["load_seconds"] + outcome.llm_load_seconds,
                "prompt_eval_seconds": (
                    llm["prompt_eval_seconds"]
                    + outcome.llm_prompt_eval_seconds
                ),
                "eval_seconds": llm["eval_seconds"] + outcome.llm_eval_seconds,
                "prompt_tokens": (
                    llm["prompt_tokens"] + outcome.llm_prompt_tokens
                ),
                "output_tokens": (
                    llm["output_tokens"] + outcome.llm_output_tokens
                ),
            },
        ),
    }


def validation_research_update(
    state: SokobanGraphState,
    *,
    retry: bool,
) -> BaselineMetrics:
    """Record one rejected plan with source-aware LLM attribution."""

    metrics = state["metrics"]
    episode = metrics["episode"]
    planning = metrics["planning"]
    llm = metrics["llm"]
    proposal = state["proposal"]
    from_llm = proposal is not None and proposal["used_llm_actions"]
    return update_baseline_metrics(
        metrics,
        episode={
            "validation_rejections": episode["validation_rejections"] + 1
        },
        planning={"retries": planning["retries"] + int(retry)},
        llm={
            "retries": llm["retries"] + int(retry and from_llm),
            "invalid_actions": llm["invalid_actions"] + int(from_llm),
        },
    )


def execution_research_update(
    state: SokobanGraphState,
    observation: Observation,
    *,
    pushed: bool,
    reward: float,
) -> dict[str, object]:
    """Accumulate board revisit, push, and reward measurements."""

    key = observation_key(observation)
    revisited = key in state["visited_state_keys"]
    visited = state["visited_state_keys"]
    if not revisited:
        visited = [*visited, key]
    metrics = state["metrics"]
    episode = metrics["episode"]
    return {
        "visited_state_keys": visited,
        "metrics": update_baseline_metrics(
            metrics,
            episode={
                "push_count": episode["push_count"] + int(pushed),
                "revisited_states": (
                    episode["revisited_states"] + int(revisited)
                ),
                "total_reward": episode["total_reward"] + reward,
            },
        ),
    }

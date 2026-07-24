"""Evaluation helpers built on the LangGraph episode runtime."""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter

from sokoban_agent.env import SokobanEnv
from sokoban_agent.evaluation.research.reference import measure_bounded_astar_reference
from sokoban_agent.evaluation.schemas.baseline import (
    AlgorithmUsage,
    BaselineEpisodeOutcome,
    BaselineLLMUsage,
    EpisodeResult,
    EpisodeTiming,
    GuardUsage,
    PlannerEpisodeIdentity,
    PlanningUsage,
)
from sokoban_agent.evaluation.schemas.reference import ReferenceResult
from sokoban_agent.graph import SokobanGraph, StepObserver
from sokoban_agent.planning import Planner


def run_episode(
    env: SokobanEnv,
    planner: Planner,
    *,
    seed: int | None = None,
    level_id: str | None = None,
    max_planning_attempts: int = 3,
    step_observer: StepObserver | None = None,
    measure_reference: bool = False,
    reference_max_expanded_states: int = 100_000,
    reference_result: ReferenceResult | None = None,
) -> EpisodeResult:
    """Run one planner inside the checkpointed LangGraph state machine."""

    reference = reference_result or (
        measure_bounded_astar_reference(
            env,
            level_id,
            max_expanded_states=reference_max_expanded_states,
        )
        if measure_reference and level_id is not None
        else ReferenceResult(solved=False)
    )
    graph = SokobanGraph(
        env,
        planner,
        max_planning_attempts=max_planning_attempts,
    )
    started_at = perf_counter()
    state = graph.run(
        seed=seed,
        level_id=level_id,
        step_observer=step_observer,
    )
    elapsed_seconds = perf_counter() - started_at
    info = state["info"]
    metrics = state["metrics"]
    episode = metrics["episode"]
    planning = metrics["planning"]
    algorithm = metrics["algorithm"]
    guard = metrics["guard"]
    llm = metrics["llm"]
    action_count = len(state["action_history"])
    return EpisodeResult(
        identity=PlannerEpisodeIdentity(graph.name, state["level_id"], seed),
        outcome=BaselineEpisodeOutcome(
            success=bool(info["success"]),
            deadlock=bool(info["deadlock"]),
            truncated=state["truncated"],
            action_count=action_count,
            invalid_moves=episode["validation_rejections"],
            total_reward=episode["total_reward"],
            failure_reason=state["failure_reason"],
            push_count=episode["push_count"],
            revisited_states=episode["revisited_states"],
            repeated_plans=episode["repeated_plans"],
        ),
        planning=PlanningUsage(
            calls=planning["calls"],
            retries=planning["retries"],
            errors=planning["errors"],
            elapsed_seconds=planning["elapsed_seconds"],
        ),
        algorithm=AlgorithmUsage(
            calls=algorithm["calls"],
            requests=algorithm["requests"],
            cache_hits=algorithm["cache_hits"],
            failures=algorithm["failures"],
            fallbacks=algorithm["fallbacks"],
            expanded_states=algorithm["expanded_states"],
            elapsed_seconds=algorithm["elapsed_seconds"],
        ),
        llm=BaselineLLMUsage(
            calls=llm["calls"],
            retries=llm["retries"],
            client_errors=llm["client_errors"],
            format_errors=llm["format_errors"],
            invalid_actions=llm["invalid_actions"],
            elapsed_seconds=llm["elapsed_seconds"],
            load_seconds=llm["load_seconds"],
            prompt_eval_seconds=llm["prompt_eval_seconds"],
            eval_seconds=llm["eval_seconds"],
            prompt_tokens=llm["prompt_tokens"],
            output_tokens=llm["output_tokens"],
        ),
        guard=GuardUsage(
            accepted=guard["accepted"],
            suffix_added=guard["suffix_added"],
            replaced=guard["replaced"],
            failed=guard["failed"],
            proposed_actions=guard["proposed_actions"],
            legal_prefix_actions=guard["legal_prefix_actions"],
            adopted_actions=guard["adopted_actions"],
            suffix_expanded_states=guard["suffix_expanded_states"],
            reference_calls=guard["reference_calls"],
            reference_action_count=guard["reference_action_count"],
            reference_expanded_states=guard["reference_expanded_states"],
            reference_elapsed_seconds=guard["reference_elapsed_seconds"],
            expansions_saved=guard["expansions_saved"],
        ),
        reference=reference,
        timing=EpisodeTiming(elapsed_seconds),
    )


def run_benchmark(
    env: SokobanEnv,
    planners: Sequence[Planner],
    *,
    level_ids: Sequence[str],
    seeds: Sequence[int],
    max_planning_attempts: int = 3,
    measure_reference: bool = False,
    reference_max_expanded_states: int = 100_000,
) -> list[EpisodeResult]:
    """Run every planner through the same graph and case grid."""

    if not planners:
        raise ValueError("at least one planner is required")
    if not level_ids:
        raise ValueError("at least one level_id is required")
    if not seeds:
        raise ValueError("at least one seed is required")

    return [
        run_episode(
            env,
            planner,
            seed=seed,
            level_id=level_id,
            max_planning_attempts=max_planning_attempts,
            measure_reference=measure_reference,
            reference_max_expanded_states=reference_max_expanded_states,
        )
        for planner in planners
        for level_id in level_ids
        for seed in seeds
    ]

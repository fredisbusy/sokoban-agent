"""Evaluation helpers built on the LangGraph episode runtime."""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter

from sokoban_agent.env import SokobanEnv
from sokoban_agent.evaluation.research.reference import measure_bounded_astar_reference
from sokoban_agent.evaluation.schemas.episode import EpisodeResult
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
        planner_name=graph.name,
        level_id=state["level_id"],
        seed=seed,
        success=bool(info["success"]),
        deadlock=bool(info["deadlock"]),
        truncated=state["truncated"],
        action_count=action_count,
        invalid_moves=episode["validation_rejections"],
        total_reward=episode["total_reward"],
        elapsed_seconds=elapsed_seconds,
        failure_reason=state["failure_reason"],
        planning_calls=planning["calls"],
        planning_retries=planning["retries"],
        planning_errors=planning["errors"],
        planning_elapsed_seconds=planning["elapsed_seconds"],
        algorithm_calls=algorithm["calls"],
        algorithm_requests=algorithm["requests"],
        algorithm_cache_hits=algorithm["cache_hits"],
        algorithm_failures=algorithm["failures"],
        algorithm_fallbacks=algorithm["fallbacks"],
        algorithm_expanded_states=algorithm["expanded_states"],
        algorithm_elapsed_seconds=algorithm["elapsed_seconds"],
        llm_calls=llm["calls"],
        llm_retries=llm["retries"],
        llm_client_errors=llm["client_errors"],
        llm_format_errors=llm["format_errors"],
        llm_invalid_actions=llm["invalid_actions"],
        llm_elapsed_seconds=llm["elapsed_seconds"],
        llm_load_seconds=llm["load_seconds"],
        llm_prompt_eval_seconds=llm["prompt_eval_seconds"],
        llm_eval_seconds=llm["eval_seconds"],
        llm_prompt_tokens=llm["prompt_tokens"],
        llm_output_tokens=llm["output_tokens"],
        push_count=episode["push_count"],
        revisited_states=episode["revisited_states"],
        repeated_plans=episode["repeated_plans"],
        guard_accepted=guard["accepted"],
        guard_suffix_added=guard["suffix_added"],
        guard_replaced=guard["replaced"],
        guard_failed=guard["failed"],
        guard_proposed_actions=guard["proposed_actions"],
        guard_legal_prefix_actions=guard["legal_prefix_actions"],
        guard_adopted_actions=guard["adopted_actions"],
        guard_suffix_expanded_states=guard["suffix_expanded_states"],
        guard_reference_calls=guard["reference_calls"],
        guard_reference_action_count=guard["reference_action_count"],
        guard_reference_expanded_states=guard["reference_expanded_states"],
        guard_reference_elapsed_seconds=guard["reference_elapsed_seconds"],
        guard_expansions_saved=guard["expansions_saved"],
        reference_solved=reference.solved,
        reference_action_count=reference.action_count,
        reference_push_count=reference.push_count,
        reference_expanded_states=reference.expanded_states,
        reference_elapsed_seconds=reference.elapsed_seconds,
        action_overhead_vs_reference=(
            action_count - reference.action_count
            if bool(info["success"]) and reference.action_count is not None
            else None
        ),
        push_overhead_vs_reference=(
            episode["push_count"] - reference.push_count
            if bool(info["success"]) and reference.push_count is not None
            else None
        ),
        policy_elapsed_seconds=max(
            0.0,
            elapsed_seconds - guard["reference_elapsed_seconds"],
        ),
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

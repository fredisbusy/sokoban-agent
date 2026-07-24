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
    return EpisodeResult(
        planner_name=graph.name,
        level_id=state["level_id"],
        seed=seed,
        success=bool(info["success"]),
        deadlock=bool(info["deadlock"]),
        truncated=state["truncated"],
        action_count=state["action_count"],
        invalid_moves=state["invalid_moves"],
        total_reward=state["total_reward"],
        elapsed_seconds=elapsed_seconds,
        failure_reason=state["failure_reason"],
        planning_calls=state["planning_calls"],
        planning_retries=state["planning_retries"],
        planning_errors=state["planning_errors"],
        planning_elapsed_seconds=state["planning_elapsed_seconds"],
        algorithm_calls=state["algorithm_calls"],
        algorithm_requests=state["algorithm_requests"],
        algorithm_cache_hits=state["algorithm_cache_hits"],
        algorithm_failures=state["algorithm_failures"],
        algorithm_fallbacks=state["algorithm_fallbacks"],
        algorithm_expanded_states=state["algorithm_expanded_states"],
        algorithm_elapsed_seconds=state["algorithm_elapsed_seconds"],
        llm_calls=state["llm_calls"],
        llm_retries=state["planning_retries"] if state["llm_calls"] else 0,
        llm_client_errors=state["llm_client_errors"],
        llm_format_errors=state["llm_format_errors"],
        llm_invalid_actions=state["llm_invalid_actions"],
        llm_elapsed_seconds=state["llm_elapsed_seconds"],
        llm_load_seconds=state["llm_load_seconds"],
        llm_prompt_eval_seconds=state["llm_prompt_eval_seconds"],
        llm_eval_seconds=state["llm_eval_seconds"],
        llm_prompt_tokens=state["llm_prompt_tokens"],
        llm_output_tokens=state["llm_output_tokens"],
        push_count=state["push_count"],
        revisited_states=state["revisited_states"],
        repeated_plans=state["repeated_plans"],
        guard_accepted=state["guard_accepted"],
        guard_suffix_added=state["guard_suffix_added"],
        guard_replaced=state["guard_replaced"],
        guard_failed=state["guard_failed"],
        guard_proposed_actions=state["guard_proposed_actions"],
        guard_legal_prefix_actions=state["guard_legal_prefix_actions"],
        guard_adopted_actions=state["guard_adopted_actions"],
        guard_suffix_expanded_states=state["guard_suffix_expanded_states"],
        guard_reference_calls=state["guard_reference_calls"],
        guard_reference_action_count=state["guard_reference_action_count"],
        guard_reference_expanded_states=(
            state["guard_reference_expanded_states"]
        ),
        guard_reference_elapsed_seconds=(
            state["guard_reference_elapsed_seconds"]
        ),
        guard_expansions_saved=state["guard_expansions_saved"],
        reference_solved=reference.solved,
        reference_action_count=reference.action_count,
        reference_push_count=reference.push_count,
        reference_expanded_states=reference.expanded_states,
        reference_elapsed_seconds=reference.elapsed_seconds,
        action_overhead_vs_reference=(
            state["action_count"] - reference.action_count
            if bool(info["success"]) and reference.action_count is not None
            else None
        ),
        push_overhead_vs_reference=(
            state["push_count"] - reference.push_count
            if bool(info["success"]) and reference.push_count is not None
            else None
        ),
        policy_elapsed_seconds=max(
            0.0,
            elapsed_seconds - state["guard_reference_elapsed_seconds"],
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

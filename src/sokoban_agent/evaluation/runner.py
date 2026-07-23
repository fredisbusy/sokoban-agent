"""Evaluation helpers built on the LangGraph episode runtime."""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter

from sokoban_agent.env import SokobanEnv
from sokoban_agent.evaluation.results import EpisodeResult
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
) -> EpisodeResult:
    """Run one planner inside the checkpointed LangGraph state machine."""

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
        algorithm_fallbacks=state["algorithm_fallbacks"],
        llm_calls=state["llm_calls"],
        llm_retries=state["planning_retries"] if state["llm_calls"] else 0,
        llm_client_errors=state["llm_client_errors"],
        llm_format_errors=state["llm_format_errors"],
        llm_invalid_actions=state["llm_invalid_actions"],
        llm_elapsed_seconds=state["llm_elapsed_seconds"],
    )


def run_benchmark(
    env: SokobanEnv,
    planners: Sequence[Planner],
    *,
    level_ids: Sequence[str],
    seeds: Sequence[int],
    max_planning_attempts: int = 3,
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
        )
        for planner in planners
        for level_id in level_ids
        for seed in seeds
    ]

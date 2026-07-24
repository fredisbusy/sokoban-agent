"""Exact trajectories captured from LangGraph execute nodes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from sokoban_agent.env import Action, SokobanEnv
from sokoban_agent.evaluation.runner import run_episode
from sokoban_agent.evaluation.schemas.trace import EpisodeFrame as EpisodeFrame
from sokoban_agent.evaluation.schemas.trace import EpisodeTrace as EpisodeTrace
from sokoban_agent.planning import Observation, Planner


def run_episode_trace(
    env: SokobanEnv,
    planner: Planner,
    *,
    seed: int | None = None,
    level_id: str | None = None,
    max_planning_attempts: int = 3,
) -> EpisodeTrace:
    """Run one graph and retain every environment state it executes."""

    frames: list[EpisodeFrame] = []

    def record(
        observation: Observation,
        action: Action | None,
        info: Mapping[str, object],
    ) -> None:
        frames.append(
            EpisodeFrame(
                index=len(frames),
                observation=observation,
                action=action,
                invalid_move=bool(info["invalid_move"]),
                pushed=bool(info["pushed"]),
                success=bool(info["success"]),
                deadlock=bool(info["deadlock"]),
            )
        )

    result = run_episode(
        env,
        planner,
        seed=seed,
        level_id=level_id,
        max_planning_attempts=max_planning_attempts,
        step_observer=record,
    )
    return EpisodeTrace(result=result, frames=tuple(frames))


def run_benchmark_traces(
    env: SokobanEnv,
    planners: Sequence[Planner],
    *,
    level_ids: Sequence[str],
    seeds: Sequence[int],
    max_planning_attempts: int = 3,
) -> list[EpisodeTrace]:
    """Run identical graph cases while retaining exact trajectories."""

    if not planners:
        raise ValueError("at least one planner is required")
    if not level_ids:
        raise ValueError("at least one level_id is required")
    if not seeds:
        raise ValueError("at least one seed is required")

    return [
        run_episode_trace(
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

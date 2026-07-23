"""Exact state trajectories captured while benchmark episodes run."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from sokoban_agent.agents import Agent, Observation
from sokoban_agent.env import Action, SokobanEnv
from sokoban_agent.evaluation.results import EpisodeResult
from sokoban_agent.evaluation.runner import run_episode


@dataclass(frozen=True, slots=True)
class EpisodeFrame:
    """One observed board and the action that led to it."""

    index: int
    observation: Observation
    action: Action | None
    invalid_move: bool
    pushed: bool
    success: bool
    deadlock: bool


@dataclass(frozen=True, slots=True)
class EpisodeTrace:
    """Episode result paired with its exact observed state sequence."""

    result: EpisodeResult
    frames: tuple[EpisodeFrame, ...]


def run_episode_trace(
    env: SokobanEnv,
    agent: Agent,
    *,
    seed: int | None = None,
    level_id: str | None = None,
) -> EpisodeTrace:
    """Run one episode and retain every board used in the result."""

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
        agent,
        seed=seed,
        level_id=level_id,
        step_observer=record,
    )
    return EpisodeTrace(result=result, frames=tuple(frames))


def run_benchmark_traces(
    env: SokobanEnv,
    agents: Sequence[Agent],
    *,
    level_ids: Sequence[str],
    seeds: Sequence[int],
) -> list[EpisodeTrace]:
    """Run identical benchmark cases while retaining exact trajectories."""

    if not agents:
        raise ValueError("at least one agent is required")
    if not level_ids:
        raise ValueError("at least one level_id is required")
    if not seeds:
        raise ValueError("at least one seed is required")

    return [
        run_episode_trace(env, agent, seed=seed, level_id=level_id)
        for agent in agents
        for level_id in level_ids
        for seed in seeds
    ]

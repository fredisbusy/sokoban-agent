"""Episode and benchmark runners shared by every agent."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from time import perf_counter

from sokoban_agent.agents import Agent, AgentStopped
from sokoban_agent.env import SokobanEnv
from sokoban_agent.evaluation.results import EpisodeResult

Clock = Callable[[], float]


def run_episode(
    env: SokobanEnv,
    agent: Agent,
    *,
    seed: int | None = None,
    level_id: str | None = None,
    clock: Clock | None = None,
) -> EpisodeResult:
    """Run one episode and record planning, action, and environment metrics."""

    options = {"level_id": level_id} if level_id is not None else None
    observation, raw_info = env.reset(seed=seed, options=options)
    info: dict[str, object] = dict(raw_info)
    timer = clock or perf_counter
    started_at = timer()
    action_count = 0
    invalid_moves = 0
    total_reward = 0.0
    truncated = False
    failure_reason: str | None = None

    try:
        agent.reset(observation, info, seed=seed)
        while not bool(info["success"]) and not bool(info["deadlock"]):
            action = agent.act(observation, info)
            observation, reward, terminated, truncated, raw_info = env.step(
                action
            )
            info = dict(raw_info)
            action_count += 1
            invalid_moves += int(bool(info["invalid_move"]))
            total_reward += reward
            if terminated or truncated:
                break
    except AgentStopped as error:
        failure_reason = str(error)

    elapsed_seconds = timer() - started_at
    return EpisodeResult(
        agent_name=agent.name,
        level_id=str(info["level_id"]),
        seed=seed,
        success=bool(info["success"]),
        deadlock=bool(info["deadlock"]),
        truncated=truncated,
        action_count=action_count,
        invalid_moves=invalid_moves,
        total_reward=total_reward,
        elapsed_seconds=elapsed_seconds,
        failure_reason=failure_reason,
    )


def run_benchmark(
    env: SokobanEnv,
    agents: Sequence[Agent],
    *,
    level_ids: Sequence[str],
    seeds: Sequence[int],
) -> list[EpisodeResult]:
    """Run every agent on the same ordered Cartesian product of cases."""

    if not agents:
        raise ValueError("at least one agent is required")
    if not level_ids:
        raise ValueError("at least one level_id is required")
    if not seeds:
        raise ValueError("at least one seed is required")

    return [
        run_episode(env, agent, seed=seed, level_id=level_id)
        for agent in agents
        for level_id in level_ids
        for seed in seeds
    ]

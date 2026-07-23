from collections import deque
from collections.abc import Mapping

import pytest

from sokoban_agent.agents import (
    AgentStopped,
    BFSAgent,
    Observation,
    RandomAgent,
)
from sokoban_agent.env import (
    Action,
    FixedLevelProvider,
    SokobanEnv,
    parse_level,
)
from sokoban_agent.evaluation import (
    EpisodeResult,
    run_benchmark,
    run_episode,
    summarize_by_agent,
)


class ScriptedAgent:
    def __init__(self, actions: list[Action]) -> None:
        self._initial_actions = actions
        self._actions: deque[Action] = deque()

    @property
    def name(self) -> str:
        return "scripted"

    def reset(
        self,
        observation: Observation,
        info: Mapping[str, object],
        *,
        seed: int | None = None,
    ) -> None:
        del observation, info, seed
        self._actions = deque(self._initial_actions)

    def act(
        self,
        observation: Observation,
        info: Mapping[str, object],
    ) -> Action:
        del observation, info
        return self._actions.popleft()


class StoppedAgent(ScriptedAgent):
    def reset(
        self,
        observation: Observation,
        info: Mapping[str, object],
        *,
        seed: int | None = None,
    ) -> None:
        del observation, info, seed
        raise AgentStopped("no plan")


def test_run_episode_records_success_invalid_moves_and_time() -> None:
    env = SokobanEnv()
    times = iter([10.0, 10.25])

    result = run_episode(
        env,
        ScriptedAgent([Action.DOWN, Action.UP]),
        seed=7,
        level_id="tiny-push",
        clock=lambda: next(times),
    )

    assert result.level_id == "tiny-push"
    assert result.seed == 7
    assert result.success
    assert not result.deadlock
    assert not result.truncated
    assert result.action_count == 2
    assert result.invalid_moves == 1
    assert result.elapsed_seconds == pytest.approx(0.25)


def test_run_episode_records_deadlock() -> None:
    level = parse_level(
        "deadlock",
        ["#####", "## .#", "# $@#", "#   #", "#####"],
    )
    env = SokobanEnv(level_provider=FixedLevelProvider([level]))

    result = run_episode(env, ScriptedAgent([Action.LEFT]))

    assert not result.success
    assert result.deadlock
    assert not result.truncated
    assert result.action_count == 1


def test_run_episode_records_step_limit() -> None:
    env = SokobanEnv(max_steps=2)

    result = run_episode(
        env,
        ScriptedAgent([Action.RIGHT, Action.LEFT]),
        level_id="tiny-walk",
    )

    assert not result.success
    assert not result.deadlock
    assert result.truncated
    assert result.action_count == 2


def test_run_episode_records_expected_agent_stop() -> None:
    result = run_episode(
        SokobanEnv(),
        StoppedAgent([]),
        level_id="tiny-walk",
    )

    assert not result.success
    assert result.action_count == 0
    assert result.failure_reason == "no plan"


def test_summarize_by_agent_calculates_required_metrics() -> None:
    results = [
        EpisodeResult(
            "agent",
            "one",
            1,
            True,
            False,
            False,
            2,
            0,
            10.0,
            0.1,
        ),
        EpisodeResult(
            "agent",
            "two",
            2,
            True,
            False,
            False,
            4,
            1,
            9.0,
            0.2,
        ),
        EpisodeResult(
            "agent",
            "three",
            3,
            False,
            True,
            False,
            6,
            2,
            -10.0,
            0.3,
        ),
    ]

    summary = summarize_by_agent(results)[0]

    assert summary.episode_count == 3
    assert summary.success_count == 2
    assert summary.success_rate == pytest.approx(2 / 3)
    assert summary.deadlock_count == 1
    assert summary.deadlock_rate == pytest.approx(1 / 3)
    assert summary.mean_actions == 4
    assert summary.mean_actions_on_success == 3
    assert summary.mean_invalid_moves == 1
    assert summary.mean_elapsed_seconds == pytest.approx(0.2)
    assert summary.total_llm_calls == 0
    assert summary.total_llm_retries == 0


def test_summarize_by_agent_accepts_no_results() -> None:
    assert summarize_by_agent([]) == []


def test_benchmark_runs_identical_cases_for_both_baselines() -> None:
    env = SokobanEnv(max_steps=30)
    results = run_benchmark(
        env,
        [RandomAgent(), BFSAgent()],
        level_ids=["tiny-push", "tiny-walk"],
        seeds=[3, 5],
    )

    random_cases = {
        (result.level_id, result.seed)
        for result in results
        if result.agent_name == "random"
    }
    bfs_results = [
        result for result in results if result.agent_name == "bfs"
    ]
    bfs_cases = {
        (result.level_id, result.seed) for result in bfs_results
    }

    assert len(results) == 8
    assert random_cases == bfs_cases
    assert all(result.success for result in bfs_results)


def test_benchmark_requires_agents_levels_and_seeds() -> None:
    env = SokobanEnv()

    with pytest.raises(ValueError, match="agent"):
        run_benchmark(env, [], level_ids=["tiny-push"], seeds=[1])
    with pytest.raises(ValueError, match="level"):
        run_benchmark(env, [RandomAgent()], level_ids=[], seeds=[1])
    with pytest.raises(ValueError, match="seed"):
        run_benchmark(
            env,
            [RandomAgent()],
            level_ids=["tiny-push"],
            seeds=[],
        )

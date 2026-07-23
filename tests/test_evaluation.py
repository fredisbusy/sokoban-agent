from collections import deque

import numpy as np
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from sokoban_agent.env import (
    Action,
    FixedLevelProvider,
    SokobanEnv,
    parse_level,
)
from sokoban_agent.evaluation import (
    EpisodeResult,
    run_benchmark,
    run_benchmark_traces,
    run_episode,
    run_episode_trace,
    summarize_by_planner,
)
from sokoban_agent.graph import SokobanGraph
from sokoban_agent.planning import (
    BFSPlanner,
    PlanningContext,
    PlanningOutcome,
    RandomPlanner,
)


class ScriptedPlanner:
    def __init__(self, actions: list[Action]) -> None:
        self._initial_actions = actions
        self._actions: deque[Action] = deque()

    @property
    def name(self) -> str:
        return "graph:scripted"

    def reset(self, *, seed: int | None = None) -> None:
        del seed
        self._actions = deque(self._initial_actions)

    def plan(self, context: PlanningContext) -> PlanningOutcome:
        del context
        if not self._actions:
            return PlanningOutcome(error="no plan", error_kind="empty")
        return PlanningOutcome(actions=(self._actions.popleft(),))


class StoppedPlanner(ScriptedPlanner):
    def plan(self, context: PlanningContext) -> PlanningOutcome:
        del context
        return PlanningOutcome(error="no plan", error_kind="empty")


def test_run_episode_records_success_invalid_moves_and_time() -> None:
    env = SokobanEnv()

    result = run_episode(
        env,
        ScriptedPlanner([Action.DOWN, Action.UP]),
        seed=7,
        level_id="tiny-push",
    )

    assert result.level_id == "tiny-push"
    assert result.seed == 7
    assert result.success
    assert not result.deadlock
    assert not result.truncated
    assert result.action_count == 1
    assert result.invalid_moves == 1
    assert result.planning_retries == 1
    assert result.elapsed_seconds >= 0


def test_run_episode_rejects_a_plan_that_causes_deadlock() -> None:
    level = parse_level(
        "deadlock",
        ["#####", "## .#", "# $@#", "#   #", "#####"],
    )
    env = SokobanEnv(level_provider=FixedLevelProvider([level]))

    result = run_episode(env, ScriptedPlanner([Action.LEFT]))

    assert not result.success
    assert not result.deadlock
    assert not result.truncated
    assert result.action_count == 0
    assert result.invalid_moves == 1


def test_run_episode_records_step_limit() -> None:
    env = SokobanEnv(max_steps=2)

    result = run_episode(
        env,
        ScriptedPlanner([Action.RIGHT, Action.LEFT]),
        level_id="tiny-walk",
    )

    assert not result.success
    assert not result.deadlock
    assert result.truncated
    assert result.action_count == 2
    assert result.push_count == 0
    assert result.revisited_states == 1


def test_run_episode_counts_repeated_proposal_on_the_same_state() -> None:
    result = run_episode(
        SokobanEnv(),
        ScriptedPlanner([Action.DOWN, Action.DOWN]),
        level_id="tiny-push",
    )

    assert not result.success
    assert result.repeated_plans == 1


def test_run_episode_records_bounded_astar_reference() -> None:
    result = run_episode(
        SokobanEnv(),
        ScriptedPlanner([Action.UP]),
        level_id="tiny-push",
        measure_reference=True,
    )

    assert result.success
    assert result.reference_solved
    assert result.reference_action_count == 1
    assert result.reference_push_count == 1
    assert result.action_overhead_vs_reference == 0
    assert result.push_overhead_vs_reference == 0


def test_run_episode_records_expected_agent_stop() -> None:
    result = run_episode(
        SokobanEnv(),
        StoppedPlanner([]),
        level_id="tiny-walk",
    )

    assert not result.success
    assert result.action_count == 0
    assert result.failure_reason == "no plan"


def test_graph_checkpoints_episode_by_thread_id() -> None:
    checkpointer = InMemorySaver()
    graph = SokobanGraph(
        SokobanEnv(),
        BFSPlanner(),
        checkpointer=checkpointer,
    )

    state = graph.run(level_id="tiny-push", thread_id="checkpoint-test")
    checkpoint = checkpointer.get(
        {"configurable": {"thread_id": "checkpoint-test"}}
    )

    assert state["info"]["success"]
    assert checkpoint is not None


def test_episode_trace_retains_initial_and_executed_boards() -> None:
    env = SokobanEnv()

    trace = run_episode_trace(
        env,
        ScriptedPlanner([Action.UP]),
        seed=4,
        level_id="tiny-push",
    )

    assert trace.result.success
    assert len(trace.frames) == 2
    assert trace.frames[0].index == 0
    assert trace.frames[0].action is None
    assert not trace.frames[0].success
    assert trace.frames[1].action is Action.UP
    assert trace.frames[1].pushed
    assert trace.frames[1].success
    assert not np.array_equal(
        trace.frames[0].observation,
        trace.frames[1].observation,
    )


def test_trace_benchmark_runs_the_same_case_grid() -> None:
    traces = run_benchmark_traces(
        SokobanEnv(max_steps=10),
        [BFSPlanner()],
        level_ids=["tiny-push", "tiny-walk"],
        seeds=[1, 2],
    )

    assert len(traces) == 4
    assert {
        (trace.result.level_id, trace.result.seed) for trace in traces
    } == {
        ("tiny-push", 1),
        ("tiny-push", 2),
        ("tiny-walk", 1),
        ("tiny-walk", 2),
    }
    assert all(trace.frames for trace in traces)


def test_summarize_by_planner_calculates_required_metrics() -> None:
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
            guard_proposed_actions=2,
            guard_adopted_actions=1,
            guard_accepted=1,
            reference_solved=True,
            action_overhead_vs_reference=0,
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
            guard_proposed_actions=2,
            guard_adopted_actions=2,
            guard_suffix_added=1,
            reference_solved=True,
            action_overhead_vs_reference=1,
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

    summary = summarize_by_planner(results)[0]

    assert summary.episode_count == 3
    assert summary.success_count == 2
    assert summary.success_rate == pytest.approx(2 / 3)
    assert summary.deadlock_count == 1
    assert summary.deadlock_rate == pytest.approx(1 / 3)
    assert summary.mean_actions == 4
    assert summary.mean_actions_on_success == 3
    assert summary.mean_invalid_moves == 1
    assert summary.mean_elapsed_seconds == pytest.approx(0.2)
    assert summary.p50_elapsed_seconds == pytest.approx(0.2)
    assert summary.p95_elapsed_seconds == pytest.approx(0.29)
    assert summary.total_planning_calls == 0
    assert summary.total_llm_calls == 0
    assert summary.total_llm_retries == 0
    assert summary.total_guard_accepted == 1
    assert summary.total_guard_suffix_added == 1
    assert summary.guard_adoption_rate == pytest.approx(0.75)
    assert summary.reference_solved_count == 2
    assert summary.mean_action_overhead_vs_reference == pytest.approx(0.5)


def test_summarize_by_planner_accepts_no_results() -> None:
    assert summarize_by_planner([]) == []


def test_benchmark_runs_identical_cases_for_both_baselines() -> None:
    env = SokobanEnv(max_steps=30)
    results = run_benchmark(
        env,
        [RandomPlanner(), BFSPlanner()],
        level_ids=["tiny-push", "tiny-walk"],
        seeds=[3, 5],
    )

    random_cases = {
        (result.level_id, result.seed)
        for result in results
        if result.planner_name == "graph:random"
    }
    bfs_results = [
        result for result in results if result.planner_name == "graph:bfs"
    ]
    bfs_cases = {
        (result.level_id, result.seed) for result in bfs_results
    }

    assert len(results) == 8
    assert random_cases == bfs_cases
    assert all(result.success for result in bfs_results)


def test_benchmark_requires_planners_levels_and_seeds() -> None:
    env = SokobanEnv()

    with pytest.raises(ValueError, match="planner"):
        run_benchmark(env, [], level_ids=["tiny-push"], seeds=[1])
    with pytest.raises(ValueError, match="level"):
        run_benchmark(env, [RandomPlanner()], level_ids=[], seeds=[1])
    with pytest.raises(ValueError, match="seed"):
        run_benchmark(
            env,
            [RandomPlanner()],
            level_ids=["tiny-push"],
            seeds=[],
        )

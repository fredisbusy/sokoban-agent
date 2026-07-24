from __future__ import annotations

from collections import deque
from typing import cast

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from sokoban_agent.env import Action, SokobanEnv
from sokoban_agent.evaluation import run_episode
from sokoban_agent.graph import SokobanGraph
from sokoban_agent.planning import (
    BFSPlanner,
    GuardPlanningMetrics,
    LLMPlanningMetrics,
    Observation,
    PlanningContext,
    PlanningOutcome,
)


class OutcomePlanner:
    def __init__(self, outcomes: list[PlanningOutcome]) -> None:
        self._initial_outcomes = outcomes
        self._outcomes: deque[PlanningOutcome] = deque()

    @property
    def name(self) -> str:
        return "graph:outcomes"

    def reset(self, *, seed: int | None = None) -> None:
        del seed
        self._outcomes = deque(self._initial_outcomes)

    def plan(self, context: PlanningContext) -> PlanningOutcome:
        del context
        return self._outcomes.popleft()


def test_baseline_state_round_trips_with_strict_msgpack() -> None:
    checkpointer = InMemorySaver(
        serde=JsonPlusSerializer(allowed_msgpack_modules=None)
    )
    graph = SokobanGraph(
        SokobanEnv(),
        BFSPlanner(),
        checkpointer=checkpointer,
    )

    state = graph.run(level_id="tiny-push", thread_id="baseline-v2-checkpoint")
    checkpoint = checkpointer.get(
        {"configurable": {"thread_id": "baseline-v2-checkpoint"}}
    )

    assert state["info"]["success"]
    assert checkpoint is not None
    channel_values = cast(
        dict[str, object],
        checkpoint["channel_values"],
    )
    assert isinstance(channel_values["observation"], list)
    assert channel_values["action_history"] == ["UP"]
    assert "action_count" not in channel_values
    assert "llm_calls" not in channel_values
    assert "metrics" in channel_values


def test_execution_uses_checkpoint_state_not_mutable_environment() -> None:
    env = SokobanEnv()
    graph = SokobanGraph(env, OutcomePlanner([]))
    raw_observation, info = env.reset(options={"level_id": "tiny-push"})
    state = graph._initial_state(
        cast(Observation, raw_observation),
        dict(info),
        None,
    )
    state["plan"] = ["UP"]

    env.reset(options={"level_id": "tiny-walk"})
    update = graph._execute(state)
    next_info = cast(dict[str, object], update["info"])

    assert next_info["level_id"] == "tiny-push"
    assert next_info["success"] is True
    assert update["action_history"] == ["UP"]


def test_guard_replacement_failure_is_not_attributed_to_llm_actions() -> None:
    planner = OutcomePlanner(
        [
            PlanningOutcome(
                actions=(Action.DOWN,),
                llm=LLMPlanningMetrics(calls=1),
                guard=GuardPlanningMetrics(disposition="replaced"),
            ),
            PlanningOutcome(actions=(Action.UP,)),
        ]
    )

    result = run_episode(
        SokobanEnv(),
        planner,
        level_id="tiny-push",
    )

    assert result.success
    assert result.planning_retries == 1
    assert result.llm_retries == 0
    assert result.llm_invalid_actions == 0

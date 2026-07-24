import json
from collections.abc import Mapping
from typing import cast

from langgraph.checkpoint.memory import InMemorySaver

from sokoban_agent.evaluation import run_agentic_episode
from sokoban_agent.evaluation.agentic.runner import _strategy_usage
from sokoban_agent.graph import build_agentic_graph
from sokoban_agent.graph.agentic.metrics import initial_agentic_metrics
from sokoban_agent.graph.agentic.state import AgenticState
from sokoban_agent.planning.agentic.runtime import (
    PromptReferenceValue,
    RenderedStrategyPrompt,
)
from sokoban_agent.planning.llm.client import CompletionMetrics, TextCompletion


class FixturePromptSource:
    def resolve(self, name: str, selector: str) -> PromptReferenceValue:
        return PromptReferenceValue(name, selector)

    def render(
        self,
        reference: PromptReferenceValue,
        variables: Mapping[str, object],
    ) -> RenderedStrategyPrompt:
        del reference, variables
        return RenderedStrategyPrompt("system", "user")


def test_strategy_revision_metrics_follow_changed_fields() -> None:
    state = cast(
        AgenticState,
        {
            "metrics": initial_agentic_metrics(),
            "feedback": [],
            "plan_revisions": [
                {"changed_fields": ["subgoal", "expected_effect"]},
                {"changed_fields": ["hypothesis"]},
                {"changed_fields": ["assignments", "subgoal"]},
            ],
        },
    )

    usage = _strategy_usage(state)

    assert usage.plan_revisions == 3
    assert usage.assignment_revisions == 1
    assert usage.hypothesis_revisions == 1
    assert usage.subgoal_revisions == 2


class DirectPushGenerator:
    def __init__(self, *, column: int = 2) -> None:
        self.column = column

    def generate(
        self,
        prompt: RenderedStrategyPrompt,
        *,
        seed: int | None,
        response_schema: Mapping[str, object],
    ) -> TextCompletion:
        del prompt, seed, response_schema
        payload = _push_payload(
            direction="UP",
            from_position=(2, self.column),
            to_position=(1, self.column),
        )
        return TextCompletion(json.dumps(payload), CompletionMetrics())


class TinyWalkGenerator:
    def __init__(self) -> None:
        self.calls = 0

    def generate(
        self,
        prompt: RenderedStrategyPrompt,
        *,
        seed: int | None,
        response_schema: Mapping[str, object],
    ) -> TextCompletion:
        del prompt, seed, response_schema
        variants = (
            _push_payload(
                direction="UP",
                from_position=(2, 2),
                to_position=(1, 2),
            ),
            _push_payload(
                direction="RIGHT",
                from_position=(1, 2),
                to_position=(1, 3),
            ),
        )
        payload = variants[self.calls]
        self.calls += 1
        return TextCompletion(json.dumps(payload), CompletionMetrics())


def _push_payload(
    *,
    direction: str,
    from_position: tuple[int, int],
    to_position: tuple[int, int],
) -> dict[str, object]:
    return {
        "summary": "B1을 T1으로 한 칸 이동한다",
        "assignments": [
            {
                "box_id": "B1",
                "target_id": "T1",
                "reason": "현재 보드의 검증 가능한 push다",
            }
        ],
        "protected_constraints": [],
        "subgoal": {
            "kind": "push",
            "box_id": "B1",
            "target_id": "T1",
            "direction": direction,
            "destination": {
                "row": to_position[0],
                "col": to_position[1],
            },
        },
        "expected_effect": {
            "box_id": "B1",
            "from_position": {
                "row": from_position[0],
                "col": from_position[1],
            },
            "to_position": {
                "row": to_position[0],
                "col": to_position[1],
            },
        },
        "failure_conditions": [
            {
                "kind": "unexpected_state",
                "description": "상자가 예상 위치로 이동하지 않는다",
            }
        ],
    }


def test_agentic_evaluation_reads_metrics_from_shared_graph_state() -> None:
    result = run_agentic_episode(
        {
            "level_id": "held-out-one-push",
            "level_rows": [
                "#######",
                "#  .  #",
                "#  $  #",
                "#  @  #",
                "#######",
            ],
            "seed": 7,
            "max_steps": 15,
        },
        context={
            "prompt_name": "sokoban-strategy",
            "prompt_commit": "fixture-commit",
            "model_name": "fixture-model",
            "rationale_mode": "on",
            "grounding_mode": "local-search",
        },
        prompt_source=FixturePromptSource(),
        strategy_generator=DirectPushGenerator(column=3),
        thread_id="agentic-evaluation-contract",
    )

    assert result.success
    assert result.level_id == "held-out-one-push"
    assert result.action_count == 1
    assert result.push_count == 1
    assert result.strategy.proposals == 1
    assert result.llm.calls == 1
    assert result.local_search.calls == 1
    assert result.local_search.expanded_states > 0
    assert result.local_search.elapsed_seconds >= 0
    assert result.rules.checks >= 3
    assert result.rules.reachability_calls >= 2
    assert result.strategy.subgoal_attempts == 1
    assert result.subgoal_successes == 1
    assert result.subgoal_failures == 0
    assert result.strategy.effect_matches == 1
    assert result.prompt.commit == "fixture-commit"


def test_structured_llm_ablation_uses_direct_push_without_local_search() -> None:
    result = run_agentic_episode(
        {"level_id": "tiny-push", "seed": 7, "max_steps": 15},
        context={
            "prompt_name": "sokoban-strategy",
            "prompt_commit": "fixture-commit",
            "model_name": "fixture-model",
            "rationale_mode": "on",
            "grounding_mode": "direct",
        },
        prompt_source=FixturePromptSource(),
        strategy_generator=DirectPushGenerator(),
        thread_id="agentic-direct-grounding",
    )

    assert result.policy_name == "structured-llm"
    assert result.success
    assert result.local_search.calls == 0
    assert result.local_search.expanded_states == 0


def test_shared_graph_streams_each_action_before_terminal_state() -> None:
    checkpointer = InMemorySaver()
    graph = build_agentic_graph(
        checkpointer=checkpointer,
        prompt_source=FixturePromptSource(),
        strategy_generator=TinyWalkGenerator(),
    )
    config = {
        "configurable": {"thread_id": "viewer-stream-contract"},
        "recursion_limit": 230,
    }
    updates = list(
        graph.stream(
            {"level_id": "tiny-walk", "seed": 7, "max_steps": 15},
            config,
            context={
                "prompt_name": "sokoban-strategy",
                "prompt_commit": "fixture-commit",
                "model_name": "fixture-model",
                "rationale_mode": "on",
                "grounding_mode": "local-search",
            },
            stream_mode="updates",
        )
    )
    action_updates = [
        cast(dict[str, object], update["execute_until_push"])
        for update in updates
        if "execute_until_push" in update
    ]
    final_state = cast(AgenticState, graph.get_state(config).values)

    assert [
        cast(list[str], update["action_history"])[-1]
        for update in action_updates
    ] == ["RIGHT", "UP", "LEFT", "UP", "RIGHT"]
    assert len(
        {
            json.dumps(update["observation"])
            for update in action_updates
        }
    ) == len(action_updates)
    assert action_updates[-1]["observation"] == final_state["observation"]
    assert final_state["status"] == "success"
    assert next(
        index
        for index, update in enumerate(updates)
        if "execute_until_push" in update
    ) < len(updates) - 1

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from sokoban_agent.graph import AgenticRuntimeContext, build_agentic_graph
from sokoban_agent.planning.agentic.runtime import (
    PromptReferenceValue,
    RenderedStrategyPrompt,
)
from sokoban_agent.planning.llm.client import CompletionMetrics, TextCompletion


class StaticPromptSource:
    def resolve(self, name: str, selector: str) -> PromptReferenceValue:
        if selector == "unresolved":
            raise AssertionError("agent graph used an unresolved prompt selector")
        commit = "fixture-commit" if selector == "latest" else selector
        return PromptReferenceValue(name, commit)

    def render(
        self,
        reference: PromptReferenceValue,
        variables: Mapping[str, object],
    ) -> RenderedStrategyPrompt:
        return RenderedStrategyPrompt("system", "user")


class StaticStrategyGenerator:
    def generate(
        self,
        prompt: RenderedStrategyPrompt,
        *,
        seed: int | None,
        response_schema: Mapping[str, object],
    ) -> TextCompletion:
        response = {
            "summary": "B1을 T1으로 올린다",
            "assignments": [
                {
                    "box_id": "B1",
                    "target_id": "T1",
                    "reason": "한 칸 위 목표가 있다",
                }
            ],
            "protected_constraints": [],
            "subgoal": {
                "kind": "push",
                "box_id": "B1",
                "target_id": "T1",
                "direction": "UP",
                "destination": {"row": 1, "col": 2},
            },
            "expected_effect": {
                "box_id": "B1",
                "from_position": {"row": 2, "col": 2},
                "to_position": {"row": 1, "col": 2},
            },
            "failure_conditions": [
                {
                    "kind": "unexpected_state",
                    "description": "상자 위치가 다르다",
                }
            ],
        }
        return TextCompletion(json.dumps(response), CompletionMetrics())


def _build_test_graph(*, checkpointer: InMemorySaver | None = None) -> Any:
    return build_agentic_graph(
        checkpointer=checkpointer,
        prompt_source=StaticPromptSource(),
        strategy_generator=StaticStrategyGenerator(),
    )


def test_agentic_graph_initializes_json_safe_checkpoint_state() -> None:
    graph = _build_test_graph(checkpointer=InMemorySaver())
    context: AgenticRuntimeContext = {
        "prompt_name": "sokoban-strategy",
        "prompt_commit": "abc123",
        "model_name": "test-model",
    }
    config = {"configurable": {"thread_id": "agentic-contract"}}

    result = graph.invoke(
        {"level_id": "tiny-push", "seed": 7, "max_steps": 15},
        config,
        context=context,
    )

    assert result["level_id"] == "tiny-push"
    assert result["observation"] == [
        [1, 1, 1, 1, 1],
        [1, 0, 5, 0, 1],
        [1, 0, 4, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ]
    assert result["prompt"] == {
        "name": "sokoban-strategy",
        "commit": "abc123",
    }
    assert result["model_name"] == "test-model"
    assert result["board_analysis"]["boxes"] == [
        {"box_id": "B1", "position": {"row": 2, "col": 2}}
    ]
    assert result["strategy_hypothesis"]["subgoal"]["box_id"] == "B1"
    assert result["active_subgoal"]["direction"] == "UP"
    assert result["protected_constraints"] == []
    assert result["expected_effect"]["to_position"] == {"row": 1, "col": 2}
    assert len(result["failure_conditions"]) == 1
    assert result["plan_revisions"] == []
    assert result["feedback"] == []
    assert result["decision_events"] == [
        {
            "step": 0,
            "stage": "initialize",
            "summary": "tiny-push 레벨을 초기화했습니다",
        },
        {
            "step": 0,
            "stage": "analyze",
            "summary": "상자 1개와 가능한 push 4개를 분석했습니다",
        },
        {
            "step": 0,
            "stage": "resolve_prompt",
            "summary": "sokoban-strategy prompt를 abc123 commit으로 고정했습니다",
        },
        {
            "step": 0,
            "stage": "recall_failures",
            "summary": "현재 보드의 제외 push 0개를 불러왔습니다",
        },
        {
            "step": 0,
            "stage": "compose_strategy_input",
            "summary": "BoardAnalysis에서 전략 입력을 구성했습니다",
        },
        {
            "step": 0,
            "stage": "recall_strategy",
            "summary": "재사용할 검증 전략이 없습니다",
        },
        {
            "step": 0,
            "stage": "propose_strategy",
            "summary": "B1을 T1으로 올린다",
        },
        {
            "step": 0,
            "stage": "verify_strategy",
            "summary": "전략 가설과 단일 push 하위 목표를 승인했습니다",
        },
        {
            "step": 0,
            "stage": "detect_repetition",
            "summary": "새 보드와 하위 목표 조합을 승인했습니다",
        },
        {
            "step": 0,
            "stage": "recall_grounding",
            "summary": "재사용할 접지 경로가 없습니다",
        },
        {
            "step": 0,
            "stage": "ground_subgoal",
            "summary": "플레이어 이동 0회와 push 1회를 접지했습니다",
        },
        {
            "step": 1,
            "stage": "execute_until_push",
            "summary": "UP 행동을 실행했습니다",
            "action": "UP",
            "pushed": True,
        },
        {
            "step": 1,
            "stage": "reflect",
            "summary": "퍼즐을 해결했습니다",
        },
        {
            "step": 1,
            "stage": "remember_outcome",
            "summary": "성공 결과를 에피소드 메모리에 유지했습니다",
        },
    ]
    assert graph.get_state(config).values["board_analysis"] == result[
        "board_analysis"
    ]
    json.dumps(result)


def test_agentic_graph_reducer_accumulates_decision_events_per_thread() -> None:
    graph = _build_test_graph(checkpointer=InMemorySaver())
    context: AgenticRuntimeContext = {
        "prompt_name": "sokoban-strategy",
        "prompt_commit": "abc123",
        "model_name": "test-model",
    }
    config = {"configurable": {"thread_id": "agentic-events"}}
    graph_input = {"level_id": "tiny-push", "seed": 0, "max_steps": 15}

    graph.invoke(graph_input, config, context=context)
    result = graph.invoke(graph_input, config, context=context)

    assert [event["stage"] for event in result["decision_events"]] == [
        "initialize",
        "analyze",
        "resolve_prompt",
        "recall_failures",
        "compose_strategy_input",
        "recall_strategy",
        "propose_strategy",
        "verify_strategy",
        "detect_repetition",
        "recall_grounding",
        "ground_subgoal",
        "execute_until_push",
        "reflect",
        "remember_outcome",
        "initialize",
        "analyze",
        "resolve_prompt",
        "recall_failures",
        "compose_strategy_input",
        "recall_strategy",
        "propose_strategy",
        "verify_strategy",
        "detect_repetition",
        "recall_grounding",
        "ground_subgoal",
        "execute_until_push",
        "reflect",
        "remember_outcome",
    ]


def test_agentic_graph_excludes_global_search_oracle_nodes() -> None:
    graph = build_agentic_graph()

    node_names = set(graph.get_graph().nodes)

    assert "analyze" in node_names
    assert not any(
        "astar" in name or "oracle" in name or "search_guard" in name
        for name in node_names
    )


def test_agentic_graph_has_agent_server_defaults() -> None:
    graph = _build_test_graph()

    result = graph.invoke({"level_id": "tiny-push"})

    # The fixture resolves the default `latest` selector to this immutable commit.
    assert result["prompt"] == {
        "name": "sokoban-strategy",
        "commit": "fixture-commit",
    }
    assert result["model_name"] == "unconfigured"


def test_langgraph_config_loads_agentic_graph_directly() -> None:
    config = json.loads(Path("langgraph.json").read_text(encoding="utf-8"))

    assert config["graphs"] == {
        "sokoban_agent": (
            "./src/sokoban_agent/graph/agentic/builder.py:graph"
        )
    }

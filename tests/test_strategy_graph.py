import json
from collections.abc import Mapping
from typing import Literal, cast

from sokoban_agent.graph import build_agentic_graph
from sokoban_agent.graph.agentic_state import AgenticState
from sokoban_agent.planning.llm import CompletionMetrics, TextCompletion
from sokoban_agent.planning.strategy import StrategyHypothesis
from sokoban_agent.planning.strategy_runtime import (
    PromptReferenceValue,
    PromptSource,
    RenderedStrategyPrompt,
    StrategyGenerator,
    TransientAgenticError,
)


class FixedPromptSource(PromptSource):
    def __init__(self, *, fail_once: bool = False) -> None:
        self.fail_once = fail_once
        self.resolve_calls = 0
        self.rendered_variables: list[Mapping[str, object]] = []

    def resolve(self, name: str, selector: str) -> PromptReferenceValue:
        self.resolve_calls += 1
        if self.fail_once and self.resolve_calls == 1:
            raise TransientAgenticError("temporary prompt outage")
        assert name == "sokoban-strategy"
        assert selector == "fixture-commit"
        return PromptReferenceValue(name=name, commit="fixture-commit")

    def render(
        self,
        reference: PromptReferenceValue,
        variables: Mapping[str, object],
    ) -> RenderedStrategyPrompt:
        assert reference.commit == "fixture-commit"
        self.rendered_variables.append(variables)
        return RenderedStrategyPrompt(
            system_prompt="fixture system",
            user_prompt="fixture user",
        )


class SequenceStrategyGenerator(StrategyGenerator):
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.calls = 0
        self.schemas: list[Mapping[str, object]] = []

    def generate(
        self,
        prompt: RenderedStrategyPrompt,
        *,
        seed: int | None,
        response_schema: Mapping[str, object],
    ) -> TextCompletion:
        assert prompt.system_prompt == "fixture system"
        assert seed == 7
        self.schemas.append(response_schema)
        response = self.responses[self.calls]
        self.calls += 1
        return TextCompletion(response, CompletionMetrics())


def _strategy_json(
    *,
    target_id: str = "T1",
    direction: str = "UP",
    from_row: int = 2,
    from_col: int = 2,
    to_row: int = 1,
    to_col: int = 2,
) -> str:
    return StrategyHypothesis.model_validate(
        {
            "summary": "B1을 T1로 올리는 첫 push를 검증한다",
            "assignments": [
                {
                    "box_id": "B1",
                    "target_id": target_id,
                    "reason": "reverse-pull 거리가 1이다",
                }
            ],
            "protected_constraints": [],
            "subgoal": {
                "kind": "push",
                "box_id": "B1",
                "target_id": target_id,
                "direction": direction,
                "destination": {"row": to_row, "col": to_col},
            },
            "expected_effect": {
                "box_id": "B1",
                "from_position": {"row": from_row, "col": from_col},
                "to_position": {"row": to_row, "col": to_col},
            },
            "failure_conditions": [
                {
                    "kind": "unexpected_state",
                    "description": "상자가 예상 칸으로 이동하지 않음",
                }
            ],
        }
    ).model_dump_json()


def _decision_json() -> str:
    return json.dumps(
        {
            "summary": "B1을 T1로 올리는 안전한 push를 선택한다",
            "push_id": "B1:UP",
            "target_id": "T1",
            "protected_cells": [],
            "risk": "상자가 예상 위치와 다르면 전략을 수정한다",
        },
        ensure_ascii=False,
    )


def _invoke(
    prompt_source: FixedPromptSource,
    generator: SequenceStrategyGenerator,
    *,
    rationale_mode: Literal["on", "off"] = "on",
) -> AgenticState:
    graph = build_agentic_graph(
        prompt_source=prompt_source,
        strategy_generator=generator,
    )
    return cast(
        AgenticState,
        graph.invoke(
            {"level_id": "tiny-push", "seed": 7, "max_steps": 15},
            context={
                "prompt_name": "sokoban-strategy",
                "prompt_commit": "fixture-commit",
                "model_name": "fixture-model",
                "rationale_mode": rationale_mode,
            },
        ),
    )


def test_strategy_nodes_use_pinned_prompt_and_board_analysis() -> None:
    prompt_source = FixedPromptSource()
    generator = SequenceStrategyGenerator(_strategy_json())

    result = _invoke(prompt_source, generator)

    assert result["status"] == "success"
    assert result["prompt"] == {
        "name": "sokoban-strategy",
        "commit": "fixture-commit",
    }
    hypothesis = cast(dict[str, object], result["strategy_hypothesis"])
    subgoal = cast(dict[str, object], hypothesis["subgoal"])
    active_subgoal = cast(dict[str, object], result["active_subgoal"])
    assert subgoal["box_id"] == "B1"
    assert active_subgoal["direction"] == "UP"
    assert result["strategy_attempts"] == 0
    assert generator.calls == 1
    assert result["grounded_actions"] == ["UP"]
    assert result["grounding_failure"] is None
    variables = prompt_source.rendered_variables[0]
    assert set(variables) == {
        "level_id",
        "board_analysis_json",
        "feedback_json",
        "plan_revisions_json",
        "rationale_mode",
    }
    assert "observation" not in variables
    compact_analysis = json.loads(str(variables["board_analysis_json"]))
    assert set(compact_analysis) == {
        "boxes",
        "targets",
        "safe_push_options",
        "reverse_pull_distances",
        "recent_pushes",
    }
    assert "reachable_cells" not in compact_analysis
    assert "dead_squares" not in compact_analysis
    schema = generator.schemas[0]
    assert schema["title"] == "StrategyDecision"
    properties = cast(dict[str, object], schema["properties"])
    assert set(properties) == {
        "summary",
        "push_id",
        "target_id",
        "protected_cells",
        "risk",
    }
    assert "expected_effect" not in properties
    assert "failure_conditions" not in properties


def test_compact_decision_materializes_complete_strategy_artifact() -> None:
    result = _invoke(FixedPromptSource(), SequenceStrategyGenerator(_decision_json()))
    assert result["status"] == "success"
    hypothesis = cast(dict[str, object], result["strategy_hypothesis"])
    assert hypothesis["expected_effect"] == {
        "box_id": "B1",
        "from_position": {"row": 2, "col": 2},
        "to_position": {"row": 1, "col": 2},
    }
    assert hypothesis["failure_conditions"] == [
        {
            "kind": "unexpected_state",
            "description": "상자가 예상 위치와 다르면 전략을 수정한다",
        }
    ]


def test_schema_error_routes_back_to_strategy_proposal() -> None:
    prompt_source = FixedPromptSource()
    generator = SequenceStrategyGenerator("{}", _strategy_json())

    result = _invoke(prompt_source, generator)

    assert result["status"] == "success"
    assert result["strategy_attempts"] == 0
    assert generator.calls == 2
    assert any("스키마" in feedback for feedback in result["feedback"])
    assert len(prompt_source.rendered_variables) == 2
    retry_feedback = json.loads(
        str(prompt_source.rendered_variables[1]["feedback_json"])
    )
    assert any("path=summary" in item for item in retry_feedback)


def test_schema_error_exposes_bounded_validation_issues() -> None:
    prompt_source = FixedPromptSource()
    generator = SequenceStrategyGenerator("{}", "{}")

    result = _invoke(prompt_source, generator)

    assert result["status"] == "strategy_schema_error"
    issues = result["strategy_schema_issues"]
    assert len(issues) <= 8
    assert issues[0] == {
        "path": "summary",
        "code": "missing",
        "message": "Field required",
    }
    assert result["strategy_error"] == "strategy_schema_error"


def test_semantic_violation_routes_back_with_structured_feedback() -> None:
    prompt_source = FixedPromptSource()
    generator = SequenceStrategyGenerator(
        _strategy_json(target_id="T9"),
        _strategy_json(),
    )

    result = _invoke(prompt_source, generator)

    assert result["status"] == "success"
    assert result["strategy_attempts"] == 0
    assert any("unknown_target" in feedback for feedback in result["feedback"])


def test_langgraph_retry_policy_retries_transient_prompt_failure() -> None:
    prompt_source = FixedPromptSource(fail_once=True)
    generator = SequenceStrategyGenerator(_strategy_json())

    result = _invoke(prompt_source, generator)

    assert result["status"] == "success"
    assert prompt_source.resolve_calls == 2


def test_strategy_graph_exposes_prompt_lifecycle_nodes() -> None:
    graph = build_agentic_graph(
        prompt_source=FixedPromptSource(),
        strategy_generator=SequenceStrategyGenerator(_strategy_json()),
    )

    assert {
        "resolve_prompt",
        "compose_strategy_input",
        "propose_strategy",
        "verify_strategy",
        "ground_subgoal",
    } <= set(graph.get_graph().nodes)


def test_no_rationale_ablation_keeps_execution_fields() -> None:
    on_source = FixedPromptSource()
    off_source = FixedPromptSource()

    with_rationale = _invoke(
        on_source,
        SequenceStrategyGenerator(_strategy_json()),
        rationale_mode="on",
    )
    without_rationale = _invoke(
        off_source,
        SequenceStrategyGenerator(_strategy_json()),
        rationale_mode="off",
    )

    assert off_source.rendered_variables[0]["rationale_mode"] == "off"
    for field in (
        "active_subgoal",
        "protected_constraints",
        "expected_effect",
        "failure_conditions",
    ):
        assert without_rationale[field] == with_rationale[field]


def test_agentic_loop_replans_after_each_push_until_success() -> None:
    prompt_source = FixedPromptSource()
    generator = SequenceStrategyGenerator(
        _strategy_json(),
        _strategy_json(
            direction="RIGHT",
            from_row=1,
            from_col=2,
            to_row=1,
            to_col=3,
        ),
    )
    graph = build_agentic_graph(
        prompt_source=prompt_source,
        strategy_generator=generator,
    )

    result = cast(
        AgenticState,
        graph.invoke(
            {"level_id": "tiny-walk", "seed": 7, "max_steps": 15},
            context={
                "prompt_name": "sokoban-strategy",
                "prompt_commit": "fixture-commit",
                "model_name": "fixture-model",
            },
        ),
    )

    assert result["status"] == "success"
    assert result["info"]["success"] is True
    assert result["action_history"] == [
        "RIGHT",
        "UP",
        "LEFT",
        "UP",
        "RIGHT",
    ]
    assert len(result["completed_subgoals"]) == 2
    assert generator.calls == 2
    action_events = [
        event
        for event in result["decision_events"]
        if event["stage"] == "execute_until_push"
    ]
    assert [event.get("action") for event in action_events] == result[
        "action_history"
    ]
    assert [event.get("pushed") for event in action_events] == [
        False,
        True,
        False,
        False,
        True,
    ]
    next_context = json.loads(
        str(prompt_source.rendered_variables[1]["board_analysis_json"])
    )
    assert next_context["recent_pushes"] == ["B1:UP"]
    assert "B1:DOWN" not in json.dumps(next_context["safe_push_options"])


def test_agentic_loop_stops_when_step_limit_prevents_push() -> None:
    prompt_source = FixedPromptSource()
    generator = SequenceStrategyGenerator(_strategy_json())
    graph = build_agentic_graph(
        prompt_source=prompt_source,
        strategy_generator=generator,
    )

    result = cast(
        AgenticState,
        graph.invoke(
            {"level_id": "tiny-walk", "seed": 7, "max_steps": 1},
            context={
                "prompt_name": "sokoban-strategy",
                "prompt_commit": "fixture-commit",
                "model_name": "fixture-model",
            },
        ),
    )

    assert result["status"] == "step_limit"
    assert result["info"]["steps"] == 1
    assert result["action_history"] == ["RIGHT"]
    execution_result = cast(dict[str, object], result["execution_result"])
    assert execution_result["push_count"] == 0
    assert generator.calls == 1

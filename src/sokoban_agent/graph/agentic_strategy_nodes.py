"""LangGraph nodes for prompt resolution and structured strategy planning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from langgraph.runtime import Runtime
from langsmith import traceable
from pydantic import ValidationError

from sokoban_agent.graph.agentic_state import (
    AgenticRuntimeContext,
    AgenticState,
    StrategySchemaIssue,
)
from sokoban_agent.planning.llm import TextCompletion
from sokoban_agent.planning.strategy import (
    BoardAnalysis,
    StrategyHypothesis,
    validate_strategy,
)
from sokoban_agent.planning.strategy_runtime import (
    LangSmithPromptSource,
    LiteLLMStrategyGenerator,
    PromptReferenceValue,
    PromptSource,
    StrategyGenerator,
)

StrategyRoute = Literal[
    "compose_strategy_input",
    "verify_strategy",
    "detect_repetition",
    "__end__",
]


@dataclass(frozen=True, slots=True)
class StrategyNodes:
    """Dependency boundary for the prompt and model lifecycle nodes."""

    prompt_source: PromptSource | None = None
    strategy_generator: StrategyGenerator | None = None

    def resolve_prompt(
        self,
        state: AgenticState,
        runtime: Runtime[AgenticRuntimeContext],
    ) -> dict[str, object]:
        """Resolve an assistant prompt selector to an immutable commit."""

        context = runtime.context or {}
        current = state["prompt"]
        name = context.get("prompt_name", current["name"])
        selector = context.get("prompt_commit", current["commit"])
        reference = self._prompt_source().resolve(name, selector)
        return {
            "prompt": {"name": reference.name, "commit": reference.commit},
            "prompt_resolved": True,
            "status": "prompt_resolved",
            "decision_events": [
                {
                    "step": _step(state),
                    "stage": "resolve_prompt",
                    "summary": (
                        f"{reference.name} prompt를 "
                        f"{reference.commit[:8]} commit으로 고정했습니다"
                    ),
                }
            ],
        }

    def compose_strategy_input(
        self,
        state: AgenticState,
    ) -> dict[str, object]:
        """Compose safe prompt variables from BoardAnalysis and feedback."""

        variables: dict[str, object] = {
            "level_id": state["level_id"],
            "board_analysis_json": json.dumps(
                state["board_analysis"],
                ensure_ascii=False,
                sort_keys=True,
            ),
            "feedback_json": json.dumps(
                state["latest_strategy_feedback"],
                ensure_ascii=False,
            ),
            "plan_revisions_json": json.dumps(
                state["plan_revisions"],
                ensure_ascii=False,
            ),
            "rationale_mode": state["rationale_mode"],
        }
        return {
            "strategy_input": variables,
            "status": "strategy_input_composed",
            "decision_events": [
                {
                    "step": _step(state),
                    "stage": "compose_strategy_input",
                    "summary": "BoardAnalysis에서 전략 입력을 구성했습니다",
                }
            ],
        }

    def propose_strategy(self, state: AgenticState) -> dict[str, object]:
        """Render the pinned prompt and request one structured hypothesis."""

        prompt = state["prompt"]
        reference = PromptReferenceValue(
            name=prompt["name"],
            commit=prompt["commit"],
        )
        rendered = self._prompt_source().render(
            reference,
            state["strategy_input"],
        )
        completion = self._strategy_generator().generate(
            rendered,
            seed=state["seed"],
            response_schema=StrategyHypothesis.model_json_schema(),
        )
        attempt = state["strategy_attempts"] + 1
        try:
            hypothesis = _validate_strategy_response(completion.content)
        except ValidationError as error:
            issues = _schema_issues(error)
            feedback = [_schema_feedback(issue) for issue in issues]
            summary = (
                "strategy_schema_error: "
                f"{issues[0]['path']} {issues[0]['message']}"
            )
            return {
                "strategy_attempts": attempt,
                "strategy_proposals": state["strategy_proposals"] + 1,
                "strategy_schema_rejections": (
                    state["strategy_schema_rejections"] + 1
                ),
                **_completion_metrics(state, completion),
                "strategy_hypothesis": None,
                "strategy_error": "strategy_schema_error",
                "strategy_schema_issues": issues,
                "latest_strategy_feedback": feedback,
                "status": "strategy_schema_error",
                "feedback": feedback,
                "decision_events": [
                    {
                        "step": _step(state),
                        "stage": "propose_strategy",
                        "summary": summary,
                    }
                ],
            }
        return {
            "strategy_attempts": attempt,
            "strategy_proposals": state["strategy_proposals"] + 1,
            **_completion_metrics(state, completion),
            "strategy_hypothesis": hypothesis.model_dump(mode="json"),
            "strategy_error": None,
            "strategy_schema_issues": [],
            "latest_strategy_feedback": [],
            "strategy_violations": [],
            "status": "strategy_proposed",
            "decision_events": [
                {
                    "step": _step(state),
                    "stage": "propose_strategy",
                    "summary": hypothesis.summary,
                }
            ],
        }

    def verify_strategy(self, state: AgenticState) -> dict[str, object]:
        """Reject semantic conflicts before any environment transition."""

        analysis = BoardAnalysis.model_validate(state["board_analysis"])
        hypothesis = StrategyHypothesis.model_validate(
            state["strategy_hypothesis"]
        )
        violations = validate_strategy(analysis, hypothesis)
        if violations:
            payloads = [
                violation.model_dump(mode="json") for violation in violations
            ]
            feedback = [
                f"{violation.code}: {violation.message}"
                for violation in violations
            ]
            return {
                "strategy_violations": payloads,
                "rule_checks": state["rule_checks"] + 1,
                "strategy_semantic_rejections": (
                    state["strategy_semantic_rejections"] + len(violations)
                ),
                "strategy_error": "strategy_semantic_error",
                "status": "strategy_semantic_error",
                "latest_strategy_feedback": feedback,
                "feedback": feedback,
                "decision_events": [
                    {
                        "step": _step(state),
                        "stage": "verify_strategy",
                        "summary": f"전략 위반 {len(violations)}개를 거절했습니다",
                    }
                ],
            }
        return {
            "strategy_violations": [],
            "rule_checks": state["rule_checks"] + 1,
            "strategy_error": None,
            "latest_strategy_feedback": [],
            "active_subgoal": hypothesis.subgoal.model_dump(mode="json"),
            "protected_constraints": [
                item.model_dump(mode="json")
                for item in hypothesis.protected_constraints
            ],
            "expected_effect": hypothesis.expected_effect.model_dump(
                mode="json"
            ),
            "failure_conditions": [
                item.model_dump(mode="json")
                for item in hypothesis.failure_conditions
            ],
            "status": "strategy_ready",
            "decision_events": [
                {
                    "step": _step(state),
                    "stage": "verify_strategy",
                    "summary": "전략 가설과 단일 push 하위 목표를 승인했습니다",
                }
            ],
        }

    def _prompt_source(self) -> PromptSource:
        return self.prompt_source or LangSmithPromptSource()

    def _strategy_generator(self) -> StrategyGenerator:
        return self.strategy_generator or LiteLLMStrategyGenerator()


def route_after_strategy_proposal(state: AgenticState) -> StrategyRoute:
    """Route schema failures to bounded correction and valid output to verify."""

    if state.get("strategy_error") is None:
        return "verify_strategy"
    if state["strategy_attempts"] < 3:
        return "compose_strategy_input"
    return "__end__"


def route_after_strategy_verification(state: AgenticState) -> StrategyRoute:
    """Route semantic violations to bounded correction."""

    if not state["strategy_violations"]:
        return "detect_repetition"
    if state["strategy_attempts"] < 3:
        return "compose_strategy_input"
    return "__end__"


def _step(state: AgenticState) -> int:
    steps = state["info"].get("steps")
    if not isinstance(steps, int):
        raise TypeError("graph info steps must be an integer")
    return steps


@traceable(name="validate_strategy_response", run_type="parser")
def _validate_strategy_response(content: str) -> StrategyHypothesis:
    """Trace the exact structured response and any schema validation error."""

    return StrategyHypothesis.model_validate_json(content)


def _schema_issues(error: ValidationError) -> list[StrategySchemaIssue]:
    """Return bounded, JSON-safe issues suitable for an LLM correction turn."""

    issues: list[StrategySchemaIssue] = []
    for detail in error.errors(include_url=False)[:8]:
        location = detail.get("loc", ())
        path = ".".join(str(part) for part in location) or "$"
        issue_type = detail.get("type", "validation_error")
        message = detail.get("msg", "유효하지 않은 값입니다")
        issues.append(
            {
                "path": path,
                "code": str(issue_type),
                "message": str(message),
            }
        )
    return issues or [
        {
            "path": "$",
            "code": "validation_error",
            "message": "전략 JSON이 스키마와 일치하지 않습니다",
        }
    ]


def _schema_feedback(issue: StrategySchemaIssue) -> str:
    return (
        "strategy_schema_error: 스키마 오류; "
        f"path={issue['path']}; code={issue['code']}; "
        f"message={issue['message']}. 이 항목을 수정해 전체 JSON을 다시 제출하세요"
    )


def _completion_metrics(
    state: AgenticState,
    completion: TextCompletion,
) -> dict[str, object]:
    return {
        "llm_calls": state["llm_calls"] + 1,
        "llm_elapsed_seconds": (
            state["llm_elapsed_seconds"] + completion.metrics.total_seconds
        ),
        "llm_prompt_tokens": (
            state["llm_prompt_tokens"] + completion.metrics.prompt_tokens
        ),
        "llm_output_tokens": (
            state["llm_output_tokens"] + completion.metrics.output_tokens
        ),
    }

"""LangGraph nodes for prompt resolution and structured strategy planning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from langgraph.runtime import Runtime
from pydantic import ValidationError

from sokoban_agent.graph.agentic_state import (
    AgenticRuntimeContext,
    AgenticState,
)
from sokoban_agent.planning.strategy import (
    BoardAnalysis,
    StrategyHypothesis,
    validate_strategy,
)
from sokoban_agent.planning.strategy_runtime import (
    LangSmithPromptSource,
    OllamaStrategyGenerator,
    PromptReferenceValue,
    PromptSource,
    StrategyGenerator,
)

StrategyRoute = Literal["compose_strategy_input", "verify_strategy", "__end__"]


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
                state["feedback"],
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
            hypothesis = StrategyHypothesis.model_validate_json(
                completion.content
            )
        except ValidationError:
            feedback = "strategy_schema_error: 전략 응답이 스키마와 다릅니다"
            return {
                "strategy_attempts": attempt,
                "strategy_hypothesis": None,
                "strategy_error": feedback,
                "status": "strategy_schema_error",
                "feedback": [feedback],
                "decision_events": [
                    {
                        "step": _step(state),
                        "stage": "propose_strategy",
                        "summary": feedback,
                    }
                ],
            }
        return {
            "strategy_attempts": attempt,
            "strategy_hypothesis": hypothesis.model_dump(mode="json"),
            "strategy_error": None,
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
                "strategy_error": "strategy_semantic_error",
                "status": "strategy_semantic_error",
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
            "strategy_error": None,
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
        return self.strategy_generator or OllamaStrategyGenerator()


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
        return "__end__"
    if state["strategy_attempts"] < 3:
        return "compose_strategy_input"
    return "__end__"


def _step(state: AgenticState) -> int:
    steps = state["info"].get("steps")
    if not isinstance(steps, int):
        raise TypeError("graph info steps must be an integer")
    return steps

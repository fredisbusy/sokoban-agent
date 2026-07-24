"""LangGraph nodes for prompt resolution and structured strategy planning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from langgraph.runtime import Runtime
from langsmith import traceable
from pydantic import ValidationError

from sokoban_agent.graph.agentic.memory.nodes import filter_rejected_pushes
from sokoban_agent.graph.agentic.metrics import update_agentic_metrics
from sokoban_agent.graph.agentic.nodes.strategy_support import (
    completion_metrics,
    schema_feedback,
    schema_issues,
)
from sokoban_agent.graph.agentic.state import (
    AgenticRuntimeContext,
    AgenticState,
)
from sokoban_agent.planning.agentic.decision import (
    StrategyDecision,
    compact_board_analysis,
    materialize_strategy,
    without_immediate_reverse,
)
from sokoban_agent.planning.agentic.models import (
    BoardAnalysis,
    PushSubgoal,
    StrategyHypothesis,
    validate_strategy,
    validate_strategy_progress,
)
from sokoban_agent.planning.agentic.runtime import (
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
    "remember_failure",
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

        analysis = BoardAnalysis.model_validate(state["board_analysis"])
        model_context = compact_board_analysis(analysis)
        recent_pushes = [
            f"{subgoal['box_id']}:{subgoal['direction']}"
            for subgoal in state["completed_subgoals"][-4:]
        ]
        model_context = without_immediate_reverse(
            model_context,
            recent_pushes[-1] if recent_pushes else None,
        )
        model_context, filtered_count = filter_rejected_pushes(
            state,
            model_context,
        )
        metrics = state["metrics"]
        model_context["recent_pushes"] = recent_pushes
        variables: dict[str, object] = {
            "level_id": state["level_id"],
            "board_analysis_json": json.dumps(
                model_context,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "feedback_json": json.dumps(
                state["latest_strategy_feedback"],
                ensure_ascii=False,
            ),
            "plan_revisions_json": json.dumps(
                state["plan_revisions"][-3:],
                ensure_ascii=False,
            ),
            "rationale_mode": state["rationale_mode"],
        }
        return {
            "strategy_input": variables,
            "metrics": update_agentic_metrics(
                metrics,
                memory={
                    "rejected_pushes_filtered": (
                        metrics["memory"]["rejected_pushes_filtered"]
                        + filtered_count
                    )
                },
            ),
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
            response_schema=StrategyDecision.model_json_schema(),
        )
        attempt = state["strategy_attempts"] + 1
        try:
            response = _validate_strategy_response(completion.content)
            hypothesis = (
                response
                if isinstance(response, StrategyHypothesis)
                else materialize_strategy(
                    BoardAnalysis.model_validate(state["board_analysis"]),
                    response,
                )
            )
        except ValidationError as error:
            issues = schema_issues(error)
            feedback = [schema_feedback(issue) for issue in issues]
            summary = (
                "strategy_schema_error: "
                f"{issues[0]['path']} {issues[0]['message']}"
            )
            return {
                "strategy_attempts": attempt,
                **completion_metrics(
                    state,
                    completion,
                    strategy={
                        "proposals": (
                            state["metrics"]["strategy"]["proposals"] + 1
                        ),
                        "schema_rejections": (
                            state["metrics"]["strategy"][
                                "schema_rejections"
                            ]
                            + 1
                        ),
                    },
                ),
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
            **completion_metrics(
                state,
                completion,
                strategy={
                    "proposals": (
                        state["metrics"]["strategy"]["proposals"] + 1
                    )
                },
            ),
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
        previous = (
            PushSubgoal.model_validate(state["completed_subgoals"][-1])
            if state["completed_subgoals"]
            else None
        )
        violations.extend(validate_strategy_progress(previous, hypothesis))
        if violations:
            payloads = [
                violation.model_dump(mode="json") for violation in violations
            ]
            feedback = [
                f"{violation.code}: {violation.message}"
                for violation in violations
            ]
            metrics = state["metrics"]
            return {
                "strategy_violations": payloads,
                "metrics": update_agentic_metrics(
                    metrics,
                    rules={"checks": metrics["rules"]["checks"] + 1},
                    strategy={
                        "semantic_rejections": (
                            metrics["strategy"]["semantic_rejections"]
                            + len(violations)
                        )
                    },
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
        metrics = state["metrics"]
        return {
            "strategy_violations": [],
            "metrics": update_agentic_metrics(
                metrics,
                rules={"checks": metrics["rules"]["checks"] + 1},
            ),
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
    if state["strategy_attempts"] < 2:
        return "compose_strategy_input"
    return "__end__"


def route_after_strategy_verification(state: AgenticState) -> StrategyRoute:
    """Route semantic violations to bounded correction."""

    if not state["strategy_violations"]:
        return "detect_repetition"
    return "remember_failure"


def _step(state: AgenticState) -> int:
    steps = state["info"].get("steps")
    if not isinstance(steps, int):
        raise TypeError("graph info steps must be an integer")
    return steps


@traceable(name="validate_strategy_response", run_type="parser")
def _validate_strategy_response(
    content: str,
) -> StrategyDecision | StrategyHypothesis:
    """Parse the compact contract while accepting pinned legacy fixtures."""

    try:
        return StrategyDecision.model_validate_json(content)
    except ValidationError as decision_error:
            try:
                return StrategyHypothesis.model_validate_json(content)
            except ValidationError:
                raise decision_error from None

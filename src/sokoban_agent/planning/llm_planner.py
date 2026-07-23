"""한국어 구조화 응답을 사용하는 다중 행동 LLM Planner."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from sokoban_agent.env import Action, Tile
from sokoban_agent.env.rules import decode_observation
from sokoban_agent.planning.base import (
    Observation,
    PlanningContext,
    PlanningOutcome,
)
from sokoban_agent.planning.llm import TextCompletion

ActionName = Literal["UP", "RIGHT", "DOWN", "LEFT"]
_TILE_SYMBOLS: dict[Tile, str] = {
    Tile.FLOOR: " ",
    Tile.WALL: "#",
    Tile.TARGET: ".",
    Tile.BOX: "$",
    Tile.PLAYER: "@",
    Tile.BOX_ON_TARGET: "*",
    Tile.PLAYER_ON_TARGET: "+",
}
_SYSTEM_PROMPT = """당신은 소코반 행동 계획을 세우는 에이전트입니다.
반드시 제공된 JSON Schema와 정확히 일치하는 JSON 객체 하나만 반환하세요.
벽, 막힌 상자, 반복 상태와 데드락을 피하세요.
해결에 가까워지는 짧고 실행 가능한 계획을 우선하세요.
goal, decision_summary, risk는 짧고 명확한 한국어로 작성하세요.
마크다운이나 JSON 밖의 설명은 포함하지 마세요."""


class ActionPlanResponse(BaseModel):
    """Strict structured response accepted from the model."""

    model_config = ConfigDict(extra="forbid")
    goal: str = Field(
        min_length=1,
        max_length=120,
        description="이번 계획에서 달성하려는 목표를 한국어로 요약",
    )
    decision_summary: str = Field(
        min_length=1,
        max_length=240,
        description="현재 보드를 보고 이 행동을 선택한 이유를 한국어로 요약",
    )
    risk: str = Field(
        min_length=1,
        max_length=160,
        description="피해야 할 벽, 막힘 또는 데드락 위험을 한국어로 요약",
    )
    actions: list[ActionName] = Field(min_length=1, max_length=8)


@dataclass(frozen=True, slots=True)
class ActionPlan:
    """Parsed actions and the model's explicit Korean decision summary."""

    actions: tuple[Action, ...]
    goal: str
    decision_summary: str
    risk: str


class StructuredTextClient(Protocol):
    """Text generation boundary used by the planner and test doubles."""

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        seed: int | None = None,
        response_schema: Mapping[str, object] | None = None,
    ) -> TextCompletion:
        """Generate one response."""
        ...


def serialize_board(observation: Observation) -> str:
    """Serialize a validated observation using standard Sokoban symbols."""

    decode_observation(observation)
    return "\n".join(
        "".join(_TILE_SYMBOLS[Tile(int(tile))] for tile in row)
        for row in observation
    )


def parse_plan_response(response: str) -> ActionPlan:
    """Parse a strict JSON action plan and its visible decision summary."""

    if not response.strip():
        raise ValueError("모델이 빈 응답을 반환했습니다")
    try:
        parsed = ActionPlanResponse.model_validate_json(response)
    except ValidationError as error:
        raise ValueError(
            "모델 응답이 행동 계획 스키마와 일치하지 않습니다"
        ) from error
    return ActionPlan(
        actions=tuple(Action[name] for name in parsed.actions),
        goal=parsed.goal,
        decision_summary=parsed.decision_summary,
        risk=parsed.risk,
    )


class LLMPlanner:
    """Propose a short plan; LangGraph owns validation and retry routing."""

    def __init__(self, client: StructuredTextClient, *, model_name: str) -> None:
        if not model_name:
            raise ValueError("model_name must not be empty")
        self.client = client
        self.model_name = model_name
        self._seed: int | None = None

    @property
    def name(self) -> str:
        """Return a stable model-specific experiment name."""

        return f"graph:llm:{self.model_name}"

    def reset(self, *, seed: int | None = None) -> None:
        """Reset episode-local model configuration."""

        self._seed = seed

    def plan(self, context: PlanningContext) -> PlanningOutcome:
        """Request one structured action and report errors to graph routing."""

        prompt = self._build_prompt(context)
        started_at = perf_counter()
        try:
            completion = self.client.complete(
                prompt,
                system_prompt=_SYSTEM_PROMPT,
                seed=self._seed,
                response_schema=ActionPlanResponse.model_json_schema(),
            )
        except Exception as error:
            elapsed = perf_counter() - started_at
            return PlanningOutcome(
                error=f"모델 요청 실패: {type(error).__name__}",
                error_kind="client",
                llm_calls=1,
                llm_client_errors=1,
                llm_elapsed_seconds=elapsed,
                elapsed_seconds=elapsed,
            )

        try:
            plan = parse_plan_response(completion.content)
        except ValueError as error:
            elapsed = perf_counter() - started_at
            return PlanningOutcome(
                error=str(error),
                error_kind="format",
                llm_calls=1,
                llm_format_errors=1,
                llm_elapsed_seconds=elapsed,
                llm_load_seconds=completion.metrics.load_seconds,
                llm_prompt_eval_seconds=(
                    completion.metrics.prompt_eval_seconds
                ),
                llm_eval_seconds=completion.metrics.eval_seconds,
                llm_prompt_tokens=completion.metrics.prompt_tokens,
                llm_output_tokens=completion.metrics.output_tokens,
                elapsed_seconds=elapsed,
            )
        elapsed = perf_counter() - started_at
        return PlanningOutcome(
            actions=plan.actions,
            proposed_actions=plan.actions,
            goal=plan.goal,
            decision_summary=plan.decision_summary,
            risk=plan.risk,
            llm_calls=1,
            llm_elapsed_seconds=elapsed,
            llm_load_seconds=completion.metrics.load_seconds,
            llm_prompt_eval_seconds=completion.metrics.prompt_eval_seconds,
            llm_eval_seconds=completion.metrics.eval_seconds,
            llm_prompt_tokens=completion.metrics.prompt_tokens,
            llm_output_tokens=completion.metrics.output_tokens,
            elapsed_seconds=elapsed,
        )

    def _build_prompt(self, context: PlanningContext) -> str:
        recent_history = context.action_history[-12:]
        history = ", ".join(action.name for action in recent_history)
        feedback = "; ".join(context.feedback) or "없음"
        return (
            "기호: # 벽, . 목표, $ 상자, @ 플레이어, "
            "* 목표 위 상자, + 목표 위 플레이어.\n"
            f"레벨: {context.info.get('level_id', '알 수 없음')}\n"
            f"이미 실행한 행동: {history or '(없음)'}\n"
            f"이전에 거절된 계획: {feedback}\n"
            "현재 보드:\n"
            f"{serialize_board(context.observation)}\n"
            "합법적인 행동을 1개에서 8개까지 선택하세요. "
            "확신할 수 없는 이동 전에는 계획을 멈추세요. "
            '예: {"goal":"상자를 목표로 밀기",'
            '"decision_summary":"상자 아래에서 위로 민다",'
            '"risk":"오른쪽 벽을 피한다","actions":["UP","LEFT"]}.'
        )

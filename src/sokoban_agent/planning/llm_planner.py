"""Structured one-action LLM planning node."""

from __future__ import annotations

from collections.abc import Mapping
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
_SYSTEM_PROMPT = """You plan Sokoban actions.
Return exactly one JSON object matching the supplied schema.
Avoid walls, blocked boxes, repeated states, and deadlocks.
Prefer a short executable plan that makes progress toward a solved board.
Do not include markdown or explanatory text."""


class ActionPlanResponse(BaseModel):
    """Strict structured response accepted from the model."""

    model_config = ConfigDict(extra="forbid")
    actions: list[ActionName] = Field(min_length=1, max_length=8)


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


def parse_plan_response(response: str) -> tuple[Action, ...]:
    """Parse a strict JSON action plan into environment enums."""

    if not response.strip():
        raise ValueError("model returned an empty response")
    try:
        parsed = ActionPlanResponse.model_validate_json(response)
    except ValidationError as error:
        raise ValueError("model response does not match the plan schema") from error
    return tuple(Action[name] for name in parsed.actions)


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
                error=f"request failed: {type(error).__name__}",
                error_kind="client",
                llm_calls=1,
                llm_client_errors=1,
                llm_elapsed_seconds=elapsed,
                elapsed_seconds=elapsed,
            )

        try:
            actions = parse_plan_response(completion.content)
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
            actions=actions,
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
        feedback = "; ".join(context.feedback) or "none"
        return (
            "Legend: # wall, . target, $ box, @ player, * box on target, "
            "+ player on target.\n"
            f"Level: {context.info.get('level_id', 'unknown')}\n"
            f"Executed actions: {history or '(none)'}\n"
            f"Rejected proposals: {feedback}\n"
            "Current board:\n"
            f"{serialize_board(context.observation)}\n"
            "Choose 1 to 8 legal actions. Stop before any uncertain move. "
            'Example: {"actions":["UP","LEFT"]}.'
        )

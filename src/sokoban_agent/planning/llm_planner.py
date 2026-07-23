"""Structured one-action LLM planning node."""

from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from sokoban_agent.env import Action, Tile
from sokoban_agent.env.rules import decode_observation
from sokoban_agent.planning.base import (
    Observation,
    PlanningContext,
    PlanningOutcome,
)

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
Do not include markdown or explanatory text."""


class ActionResponse(BaseModel):
    """Strict structured response accepted from the model."""

    model_config = ConfigDict(extra="forbid")
    action: ActionName


class StructuredTextClient(Protocol):
    """Text generation boundary used by the planner and test doubles."""

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        seed: int | None = None,
        response_format: Mapping[str, object] | None = None,
    ) -> str:
        """Generate one response."""
        ...


def serialize_board(observation: Observation) -> str:
    """Serialize a validated observation using standard Sokoban symbols."""

    decode_observation(observation)
    return "\n".join(
        "".join(_TILE_SYMBOLS[Tile(int(tile))] for tile in row)
        for row in observation
    )


def parse_action_response(response: str) -> Action:
    """Parse a strict JSON action object into the environment enum."""

    if not response.strip():
        raise ValueError("model returned an empty response")
    try:
        parsed = ActionResponse.model_validate_json(response)
    except ValidationError as error:
        raise ValueError("model response does not match the action schema") from error
    return Action[parsed.action]


class LLMPlanner:
    """Propose one action; LangGraph owns validation and retry routing."""

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
            response = self.client.complete(
                prompt,
                system_prompt=_SYSTEM_PROMPT,
                seed=self._seed,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "sokoban_action",
                        "schema": ActionResponse.model_json_schema(),
                    },
                },
            )
        except Exception as error:
            return PlanningOutcome(
                error=f"request failed: {type(error).__name__}",
                error_kind="client",
                llm_calls=1,
                llm_client_errors=1,
                elapsed_seconds=perf_counter() - started_at,
            )

        try:
            action = parse_action_response(response)
        except ValueError as error:
            return PlanningOutcome(
                error=str(error),
                error_kind="format",
                llm_calls=1,
                llm_format_errors=1,
                elapsed_seconds=perf_counter() - started_at,
            )
        return PlanningOutcome(
            actions=(action,),
            llm_calls=1,
            elapsed_seconds=perf_counter() - started_at,
        )

    def _build_prompt(self, context: PlanningContext) -> str:
        history = ", ".join(action.name for action in context.action_history)
        feedback = "; ".join(context.feedback) or "none"
        return (
            "Legend: # wall, . target, $ box, @ player, * box on target, "
            "+ player on target.\n"
            f"Level: {context.info.get('level_id', 'unknown')}\n"
            f"Executed actions: {history or '(none)'}\n"
            f"Rejected proposals: {feedback}\n"
            "Current board:\n"
            f"{serialize_board(context.observation)}\n"
            'Choose one legal action as JSON, for example {"action":"UP"}.'
        )

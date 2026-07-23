"""Structured one-step LLM policy for Sokoban."""

from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from sokoban_agent.agents.base import (
    AgentDiagnostics,
    AgentInfo,
    AgentStopped,
    Observation,
)
from sokoban_agent.env import Action, Tile
from sokoban_agent.env.rules import apply_action, decode_observation

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
_SYSTEM_PROMPT = """You choose one primitive move in Sokoban.
Return exactly one JSON object matching the supplied schema.
Plan before choosing, avoid walls, blocked boxes, repeated states, and deadlocks.
Do not include markdown or explanatory text."""


class ActionResponse(BaseModel):
    """Strict structured response accepted from the model."""

    model_config = ConfigDict(extra="forbid")

    action: ActionName


class StructuredTextClient(Protocol):
    """Text generation boundary used by the LLM agent and test doubles."""

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


def _is_valid_move(observation: Observation, action: Action) -> bool:
    level, state = decode_observation(observation)
    return not apply_action(level, state, action).invalid_move


class LLMAgent:
    """Replan one validated primitive action from every observed state."""

    def __init__(
        self,
        client: StructuredTextClient,
        *,
        model_name: str,
        max_attempts: int = 3,
    ) -> None:
        if not model_name:
            raise ValueError("model_name must not be empty")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        self.client = client
        self.model_name = model_name
        self.max_attempts = max_attempts
        self._seed: int | None = None
        self._history: list[Action] = []
        self._diagnostics = AgentDiagnostics()

    @property
    def name(self) -> str:
        """Return a stable model-specific experiment name."""

        return f"llm:{self.model_name}"

    @property
    def diagnostics(self) -> AgentDiagnostics:
        """Return measurements collected during the current episode."""

        return self._diagnostics

    def reset(
        self,
        observation: Observation,
        info: AgentInfo,
        *,
        seed: int | None = None,
    ) -> None:
        """Reset history and measurements for a new episode."""

        del info
        serialize_board(observation)
        self._seed = seed
        self._history = []
        self._diagnostics = AgentDiagnostics()

    def act(self, observation: Observation, info: AgentInfo) -> Action:
        """Request, validate, and return one legal primitive action."""

        errors: list[str] = []
        for attempt in range(self.max_attempts):
            prompt = self._build_prompt(observation, info, errors)
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
                self._record_call(perf_counter() - started_at)
                self._record_error("client")
                errors.append(f"request failed: {type(error).__name__}")
            else:
                self._record_call(perf_counter() - started_at)
                try:
                    action = parse_action_response(response)
                except ValueError as error:
                    self._record_error("format")
                    errors.append(str(error))
                else:
                    if _is_valid_move(observation, action):
                        self._history.append(action)
                        return action
                    self._record_error("action")
                    errors.append(f"{action.name} is blocked in the current board")

            if attempt + 1 < self.max_attempts:
                self._record_retry()

        raise AgentStopped(
            f"LLM failed to return a legal action after {self.max_attempts} attempts"
        )

    def _build_prompt(
        self,
        observation: Observation,
        info: AgentInfo,
        errors: list[str],
    ) -> str:
        history = ", ".join(action.name for action in self._history) or "(none)"
        feedback: list[str] = []
        if bool(info.get("invalid_move")) and self._history:
            feedback.append(
                f"The previous move {self._history[-1].name} was invalid."
            )
        if errors:
            feedback.append(f"Rejected attempts: {'; '.join(errors)}.")
        feedback_text = "\n".join(feedback) or "No rejected attempt."
        return (
            "Legend: # wall, . target, $ box, @ player, * box on target, "
            "+ player on target.\n"
            f"Level: {info.get('level_id', 'unknown')}\n"
            f"Previous actions: {history}\n"
            f"{feedback_text}\n"
            "Current board:\n"
            f"{serialize_board(observation)}\n"
            'Choose one legal action as JSON, for example {"action":"UP"}.'
        )

    def _record_call(self, elapsed_seconds: float) -> None:
        current = self._diagnostics
        self._diagnostics = AgentDiagnostics(
            llm_calls=current.llm_calls + 1,
            llm_retries=current.llm_retries,
            llm_client_errors=current.llm_client_errors,
            llm_format_errors=current.llm_format_errors,
            llm_invalid_actions=current.llm_invalid_actions,
            llm_elapsed_seconds=current.llm_elapsed_seconds + elapsed_seconds,
        )

    def _record_retry(self) -> None:
        current = self._diagnostics
        self._diagnostics = AgentDiagnostics(
            llm_calls=current.llm_calls,
            llm_retries=current.llm_retries + 1,
            llm_client_errors=current.llm_client_errors,
            llm_format_errors=current.llm_format_errors,
            llm_invalid_actions=current.llm_invalid_actions,
            llm_elapsed_seconds=current.llm_elapsed_seconds,
        )

    def _record_error(self, kind: Literal["client", "format", "action"]) -> None:
        current = self._diagnostics
        self._diagnostics = AgentDiagnostics(
            llm_calls=current.llm_calls,
            llm_retries=current.llm_retries,
            llm_client_errors=current.llm_client_errors
            + int(kind == "client"),
            llm_format_errors=current.llm_format_errors
            + int(kind == "format"),
            llm_invalid_actions=current.llm_invalid_actions
            + int(kind == "action"),
            llm_elapsed_seconds=current.llm_elapsed_seconds,
        )

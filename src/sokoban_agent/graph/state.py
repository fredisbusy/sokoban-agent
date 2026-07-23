"""Typed state shared by every Sokoban LangGraph node."""

from __future__ import annotations

from typing_extensions import TypedDict

from sokoban_agent.env import Action
from sokoban_agent.planning import Observation


class SokobanGraphState(TypedDict):
    """Complete checkpointable state for one episode."""

    observation: Observation
    info: dict[str, object]
    seed: int | None
    level_id: str
    plan: tuple[Action, ...]
    action_history: tuple[Action, ...]
    feedback: tuple[str, ...]
    planning_attempts: int
    action_count: int
    invalid_moves: int
    total_reward: float
    truncated: bool
    failure_reason: str | None
    planning_calls: int
    planning_retries: int
    planning_errors: int
    planning_elapsed_seconds: float
    algorithm_calls: int
    algorithm_fallbacks: int
    llm_calls: int
    llm_client_errors: int
    llm_format_errors: int
    llm_invalid_actions: int
    llm_elapsed_seconds: float
    last_proposal_used_llm: bool

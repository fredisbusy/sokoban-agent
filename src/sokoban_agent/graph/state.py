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
    proposed_plan: tuple[Action, ...]
    planner_goal: str | None
    decision_summary: str | None
    risk: str | None
    guard_summary: str | None
    validation_summary: str | None
    execution_summary: str | None
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
    algorithm_expanded_states: int
    algorithm_elapsed_seconds: float
    llm_calls: int
    llm_client_errors: int
    llm_format_errors: int
    llm_invalid_actions: int
    llm_elapsed_seconds: float
    llm_load_seconds: float
    llm_prompt_eval_seconds: float
    llm_eval_seconds: float
    llm_prompt_tokens: int
    llm_output_tokens: int
    last_proposal_used_llm: bool

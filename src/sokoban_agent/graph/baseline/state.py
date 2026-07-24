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
    visited_state_keys: tuple[str, ...]
    seen_plan_keys: tuple[str, ...]
    feedback: tuple[str, ...]
    planning_attempts: int
    action_count: int
    push_count: int
    revisited_states: int
    repeated_plans: int
    invalid_moves: int
    total_reward: float
    truncated: bool
    failure_reason: str | None
    planning_calls: int
    planning_retries: int
    planning_errors: int
    planning_elapsed_seconds: float
    algorithm_calls: int
    algorithm_requests: int
    algorithm_cache_hits: int
    algorithm_failures: int
    algorithm_fallbacks: int
    algorithm_expanded_states: int
    algorithm_elapsed_seconds: float
    guard_accepted: int
    guard_suffix_added: int
    guard_replaced: int
    guard_failed: int
    guard_proposed_actions: int
    guard_legal_prefix_actions: int
    guard_adopted_actions: int
    guard_suffix_expanded_states: int
    guard_reference_calls: int
    guard_reference_action_count: int
    guard_reference_expanded_states: int
    guard_reference_elapsed_seconds: float
    guard_expansions_saved: int
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

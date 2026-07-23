"""Result contract for structured LangGraph Sokoban episodes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AgenticEpisodeResult:
    """One structured-policy episode measured from final graph state."""

    policy_name: str
    level_id: str
    seed: int | None
    status: str
    success: bool
    deadlock: bool
    truncated: bool
    cycle_detected: bool
    action_count: int
    action_sequence: tuple[str, ...]
    push_count: int
    strategy_proposals: int
    strategy_schema_rejections: int
    strategy_semantic_rejections: int
    plan_revision_count: int
    protected_constraint_violations: int
    effect_matches: int
    effect_mismatches: int
    llm_calls: int
    llm_elapsed_seconds: float
    llm_prompt_tokens: int
    llm_output_tokens: int
    local_search_calls: int
    local_expanded_states: int
    local_search_elapsed_seconds: float
    rule_checks: int
    reachability_calls: int
    subgoal_attempts: int
    subgoal_successes: int
    subgoal_failures: int
    assignment_revision_count: int
    hypothesis_revision_count: int
    actions_derived_from_subgoal: int
    algorithm_calls: int
    prompt_name: str
    prompt_commit: str
    model_name: str
    elapsed_seconds: float

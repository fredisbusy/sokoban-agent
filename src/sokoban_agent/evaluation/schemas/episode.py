"""Episode-level outcomes and policy aggregates."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Raw measurements from one agent, level, and seed combination."""

    planner_name: str
    level_id: str
    seed: int | None
    success: bool
    deadlock: bool
    truncated: bool
    action_count: int
    invalid_moves: int
    total_reward: float
    elapsed_seconds: float
    failure_reason: str | None = None
    planning_calls: int = 0
    planning_retries: int = 0
    planning_errors: int = 0
    planning_elapsed_seconds: float = 0.0
    algorithm_calls: int = 0
    algorithm_requests: int = 0
    algorithm_cache_hits: int = 0
    algorithm_failures: int = 0
    algorithm_fallbacks: int = 0
    algorithm_expanded_states: int = 0
    algorithm_elapsed_seconds: float = 0.0
    llm_calls: int = 0
    llm_retries: int = 0
    llm_client_errors: int = 0
    llm_format_errors: int = 0
    llm_invalid_actions: int = 0
    llm_elapsed_seconds: float = 0.0
    llm_load_seconds: float = 0.0
    llm_prompt_eval_seconds: float = 0.0
    llm_eval_seconds: float = 0.0
    llm_prompt_tokens: int = 0
    llm_output_tokens: int = 0
    push_count: int = 0
    revisited_states: int = 0
    repeated_plans: int = 0
    guard_accepted: int = 0
    guard_suffix_added: int = 0
    guard_replaced: int = 0
    guard_failed: int = 0
    guard_proposed_actions: int = 0
    guard_legal_prefix_actions: int = 0
    guard_adopted_actions: int = 0
    guard_suffix_expanded_states: int = 0
    guard_reference_calls: int = 0
    guard_reference_action_count: int = 0
    guard_reference_expanded_states: int = 0
    guard_reference_elapsed_seconds: float = 0.0
    guard_expansions_saved: int = 0
    reference_solved: bool = False
    reference_action_count: int | None = None
    reference_push_count: int | None = None
    reference_expanded_states: int = 0
    reference_elapsed_seconds: float = 0.0
    action_overhead_vs_reference: int | None = None
    push_overhead_vs_reference: int | None = None
    policy_elapsed_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class PlannerSummary:
    """Aggregate metrics for one agent."""

    planner_name: str
    episode_count: int
    success_count: int
    success_rate: float
    deadlock_count: int
    deadlock_rate: float
    truncated_count: int
    mean_actions: float
    mean_actions_on_success: float | None
    mean_invalid_moves: float
    mean_elapsed_seconds: float
    p50_elapsed_seconds: float
    p95_elapsed_seconds: float
    total_planning_calls: int
    total_planning_retries: int
    total_planning_errors: int
    mean_planning_elapsed_seconds: float
    total_algorithm_calls: int
    total_algorithm_requests: int
    total_algorithm_cache_hits: int
    total_algorithm_failures: int
    total_algorithm_fallbacks: int
    total_algorithm_expanded_states: int
    mean_algorithm_elapsed_seconds: float
    total_llm_calls: int
    total_llm_retries: int
    total_llm_client_errors: int
    total_llm_format_errors: int
    total_llm_invalid_actions: int
    mean_llm_elapsed_seconds: float
    p50_llm_elapsed_seconds: float
    p95_llm_elapsed_seconds: float
    total_llm_prompt_tokens: int
    total_llm_output_tokens: int
    llm_output_tokens_per_second: float | None
    mean_pushes_on_success: float | None
    total_revisited_states: int
    total_repeated_plans: int
    total_guard_accepted: int
    total_guard_suffix_added: int
    total_guard_replaced: int
    total_guard_failed: int
    total_guard_proposed_actions: int
    total_guard_legal_prefix_actions: int
    total_guard_adopted_actions: int
    guard_adoption_rate: float | None
    total_guard_suffix_expanded_states: int
    total_guard_reference_calls: int
    total_guard_reference_expanded_states: int
    total_guard_expansions_saved: int
    reference_solved_count: int
    mean_action_overhead_vs_reference: float | None
    mean_push_overhead_vs_reference: float | None
    mean_policy_elapsed_seconds: float


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
    memory_requests: int
    memory_hits: int
    memory_writes: int
    strategy_cache_hits: int
    grounding_cache_hits: int
    analysis_cache_hits: int
    llm_calls_saved: int
    rejected_pushes_filtered: int
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

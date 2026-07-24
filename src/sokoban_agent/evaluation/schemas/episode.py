"""Episode-level outcomes and policy aggregates."""

from __future__ import annotations

from dataclasses import dataclass

from sokoban_agent.evaluation.schemas.metrics import (
    LLMUsage,
    MemoryUsage,
    PromptIdentity,
    RuleUsage,
    SearchUsage,
    StrategyUsage,
)


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
class EpisodeIdentity:
    """Stable policy and case identity."""

    policy_name: str
    level_id: str
    seed: int | None


@dataclass(frozen=True, slots=True)
class EpisodeOutcome:
    """Terminal outcome and canonical action trajectory."""

    status: str
    success: bool
    deadlock: bool
    truncated: bool
    cycle_detected: bool
    action_sequence: tuple[str, ...]
    push_count: int

    @property
    def action_count(self) -> int:
        return len(self.action_sequence)


@dataclass(frozen=True, slots=True)
class AgenticEpisodeResult:
    """One structured-policy episode composed by measurement responsibility."""

    identity: EpisodeIdentity
    outcome: EpisodeOutcome
    strategy: StrategyUsage
    llm: LLMUsage
    memory: MemoryUsage
    local_search: SearchUsage
    rules: RuleUsage
    prompt: PromptIdentity
    elapsed_seconds: float

    @property
    def policy_name(self) -> str:
        return self.identity.policy_name

    @property
    def level_id(self) -> str:
        return self.identity.level_id

    @property
    def seed(self) -> int | None:
        return self.identity.seed

    @property
    def status(self) -> str:
        return self.outcome.status

    @property
    def success(self) -> bool:
        return self.outcome.success

    @property
    def deadlock(self) -> bool:
        return self.outcome.deadlock

    @property
    def truncated(self) -> bool:
        return self.outcome.truncated

    @property
    def cycle_detected(self) -> bool:
        return self.outcome.cycle_detected

    @property
    def action_count(self) -> int:
        return self.outcome.action_count

    @property
    def action_sequence(self) -> tuple[str, ...]:
        return self.outcome.action_sequence

    @property
    def push_count(self) -> int:
        return self.outcome.push_count

    @property
    def subgoal_successes(self) -> int:
        return self.strategy.subgoal_successes

    @property
    def subgoal_failures(self) -> int:
        return self.strategy.subgoal_failures

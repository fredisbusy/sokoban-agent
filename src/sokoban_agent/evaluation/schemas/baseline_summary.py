"""Composable baseline planner aggregate contracts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OutcomeSummary:
    episode_count: int
    success_count: int
    deadlock_count: int
    truncated_count: int

    @property
    def success_rate(self) -> float:
        return self.success_count / self.episode_count

    @property
    def deadlock_rate(self) -> float:
        return self.deadlock_count / self.episode_count


@dataclass(frozen=True, slots=True)
class ActionSummary:
    mean_actions: float
    mean_actions_on_success: float | None
    mean_invalid_moves: float
    mean_pushes_on_success: float | None
    total_revisited_states: int
    total_repeated_plans: int


@dataclass(frozen=True, slots=True)
class LatencySummary:
    mean_elapsed_seconds: float
    p50_elapsed_seconds: float
    p95_elapsed_seconds: float
    mean_policy_elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class PlanningSummary:
    total_calls: int
    total_retries: int
    total_errors: int
    mean_elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class AlgorithmSummary:
    total_calls: int
    total_requests: int
    total_cache_hits: int
    total_failures: int
    total_fallbacks: int
    total_expanded_states: int
    mean_elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class LLMSummary:
    total_calls: int
    total_retries: int
    total_client_errors: int
    total_format_errors: int
    total_invalid_actions: int
    mean_elapsed_seconds: float
    p50_elapsed_seconds: float
    p95_elapsed_seconds: float
    total_prompt_tokens: int
    total_output_tokens: int
    total_eval_seconds: float

    @property
    def output_tokens_per_second(self) -> float | None:
        if self.total_eval_seconds <= 0:
            return None
        return self.total_output_tokens / self.total_eval_seconds


@dataclass(frozen=True, slots=True)
class GuardSummary:
    total_accepted: int
    total_suffix_added: int
    total_replaced: int
    total_failed: int
    total_proposed_actions: int
    total_legal_prefix_actions: int
    total_adopted_actions: int
    total_suffix_expanded_states: int
    total_reference_calls: int
    total_reference_expanded_states: int
    total_expansions_saved: int

    @property
    def adoption_rate(self) -> float | None:
        if self.total_proposed_actions == 0:
            return None
        return self.total_adopted_actions / self.total_proposed_actions


@dataclass(frozen=True, slots=True)
class ReferenceSummary:
    solved_count: int
    mean_action_overhead: float | None
    mean_push_overhead: float | None


@dataclass(frozen=True, slots=True)
class PlannerSummary:
    """Aggregate baseline metrics composed by reporting responsibility."""

    planner_name: str
    outcome: OutcomeSummary
    actions: ActionSummary
    timing: LatencySummary
    planning: PlanningSummary
    algorithm: AlgorithmSummary
    llm: LLMSummary
    guard: GuardSummary
    reference: ReferenceSummary

    @property
    def episode_count(self) -> int:
        return self.outcome.episode_count

    @property
    def success_count(self) -> int:
        return self.outcome.success_count

    @property
    def success_rate(self) -> float:
        return self.outcome.success_rate

    @property
    def deadlock_count(self) -> int:
        return self.outcome.deadlock_count

    @property
    def deadlock_rate(self) -> float:
        return self.outcome.deadlock_rate

    @property
    def mean_actions(self) -> float:
        return self.actions.mean_actions

    @property
    def mean_actions_on_success(self) -> float | None:
        return self.actions.mean_actions_on_success

    @property
    def mean_invalid_moves(self) -> float:
        return self.actions.mean_invalid_moves

    @property
    def mean_elapsed_seconds(self) -> float:
        return self.timing.mean_elapsed_seconds

    @property
    def p50_elapsed_seconds(self) -> float:
        return self.timing.p50_elapsed_seconds

    @property
    def p95_elapsed_seconds(self) -> float:
        return self.timing.p95_elapsed_seconds

    @property
    def total_planning_calls(self) -> int:
        return self.planning.total_calls

    @property
    def total_llm_calls(self) -> int:
        return self.llm.total_calls

    @property
    def total_llm_retries(self) -> int:
        return self.llm.total_retries

    @property
    def total_guard_accepted(self) -> int:
        return self.guard.total_accepted

    @property
    def total_guard_suffix_added(self) -> int:
        return self.guard.total_suffix_added

    @property
    def guard_adoption_rate(self) -> float | None:
        return self.guard.adoption_rate

    @property
    def reference_solved_count(self) -> int:
        return self.reference.solved_count

    @property
    def mean_action_overhead_vs_reference(self) -> float | None:
        return self.reference.mean_action_overhead

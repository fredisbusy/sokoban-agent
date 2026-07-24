"""Composable baseline episode and aggregate result contracts."""

from __future__ import annotations

from dataclasses import dataclass

from sokoban_agent.evaluation.schemas.reference import ReferenceResult


@dataclass(frozen=True, slots=True)
class PlannerEpisodeIdentity:
    planner_name: str
    level_id: str
    seed: int | None


@dataclass(frozen=True, slots=True)
class BaselineEpisodeOutcome:
    success: bool
    deadlock: bool
    truncated: bool
    action_count: int
    invalid_moves: int
    total_reward: float
    failure_reason: str | None = None
    push_count: int = 0
    revisited_states: int = 0
    repeated_plans: int = 0


@dataclass(frozen=True, slots=True)
class PlanningUsage:
    calls: int = 0
    retries: int = 0
    errors: int = 0
    elapsed_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class AlgorithmUsage:
    calls: int = 0
    requests: int = 0
    cache_hits: int = 0
    failures: int = 0
    fallbacks: int = 0
    expanded_states: int = 0
    elapsed_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class BaselineLLMUsage:
    calls: int = 0
    retries: int = 0
    client_errors: int = 0
    format_errors: int = 0
    invalid_actions: int = 0
    elapsed_seconds: float = 0.0
    load_seconds: float = 0.0
    prompt_eval_seconds: float = 0.0
    eval_seconds: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True, slots=True)
class GuardUsage:
    accepted: int = 0
    suffix_added: int = 0
    replaced: int = 0
    failed: int = 0
    proposed_actions: int = 0
    legal_prefix_actions: int = 0
    adopted_actions: int = 0
    suffix_expanded_states: int = 0
    reference_calls: int = 0
    reference_action_count: int = 0
    reference_expanded_states: int = 0
    reference_elapsed_seconds: float = 0.0
    expansions_saved: int = 0


@dataclass(frozen=True, slots=True)
class EpisodeTiming:
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """One baseline episode composed by measurement responsibility."""

    identity: PlannerEpisodeIdentity
    outcome: BaselineEpisodeOutcome
    planning: PlanningUsage = PlanningUsage()
    algorithm: AlgorithmUsage = AlgorithmUsage()
    llm: BaselineLLMUsage = BaselineLLMUsage()
    guard: GuardUsage = GuardUsage()
    reference: ReferenceResult = ReferenceResult(solved=False)
    timing: EpisodeTiming = EpisodeTiming(0.0)

    @property
    def planner_name(self) -> str:
        return self.identity.planner_name

    @property
    def level_id(self) -> str:
        return self.identity.level_id

    @property
    def seed(self) -> int | None:
        return self.identity.seed

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
    def action_count(self) -> int:
        return self.outcome.action_count

    @property
    def invalid_moves(self) -> int:
        return self.outcome.invalid_moves

    @property
    def push_count(self) -> int:
        return self.outcome.push_count

    @property
    def revisited_states(self) -> int:
        return self.outcome.revisited_states

    @property
    def repeated_plans(self) -> int:
        return self.outcome.repeated_plans

    @property
    def failure_reason(self) -> str | None:
        return self.outcome.failure_reason

    @property
    def elapsed_seconds(self) -> float:
        return self.timing.elapsed_seconds

    @property
    def llm_calls(self) -> int:
        return self.llm.calls

    @property
    def planning_retries(self) -> int:
        return self.planning.retries

    @property
    def llm_retries(self) -> int:
        return self.llm.retries

    @property
    def llm_client_errors(self) -> int:
        return self.llm.client_errors

    @property
    def llm_format_errors(self) -> int:
        return self.llm.format_errors

    @property
    def llm_invalid_actions(self) -> int:
        return self.llm.invalid_actions

    @property
    def llm_prompt_tokens(self) -> int:
        return self.llm.prompt_tokens

    @property
    def llm_output_tokens(self) -> int:
        return self.llm.output_tokens

    @property
    def llm_elapsed_seconds(self) -> float:
        return self.llm.elapsed_seconds

    @property
    def algorithm_calls(self) -> int:
        return self.algorithm.calls

    @property
    def algorithm_expanded_states(self) -> int:
        return self.algorithm.expanded_states

    @property
    def algorithm_elapsed_seconds(self) -> float:
        return self.algorithm.elapsed_seconds

    @property
    def reference_solved(self) -> bool:
        return self.reference.solved

    @property
    def reference_action_count(self) -> int | None:
        return self.reference.action_count

    @property
    def reference_push_count(self) -> int | None:
        return self.reference.push_count

    @property
    def policy_elapsed_seconds(self) -> float:
        return max(
            0.0,
            self.timing.elapsed_seconds - self.guard.reference_elapsed_seconds,
        )

    @property
    def action_overhead_vs_reference(self) -> int | None:
        if not self.outcome.success or self.reference.action_count is None:
            return None
        return self.outcome.action_count - self.reference.action_count

    @property
    def push_overhead_vs_reference(self) -> int | None:
        if not self.outcome.success or self.reference.push_count is None:
            return None
        return self.outcome.push_count - self.reference.push_count

"""Structured-policy episode result contracts."""

from __future__ import annotations

from dataclasses import dataclass

from sokoban_agent.evaluation.schemas.baseline import (
    EpisodeResult as EpisodeResult,
)
from sokoban_agent.evaluation.schemas.baseline_summary import (
    PlannerSummary as PlannerSummary,
)
from sokoban_agent.evaluation.schemas.metrics import (
    LLMUsage,
    MemoryUsage,
    PromptIdentity,
    RuleUsage,
    SearchUsage,
    StrategyUsage,
)


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

"""Research experiment records and aggregate contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from sokoban_agent.evaluation.schemas.episode import (
    EpisodeIdentity,
    EpisodeOutcome,
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
class LevelProfile:
    """Held-out level dimensions and structural tags."""

    difficulty: str
    layout_family: str
    corridor_structure: str
    trap_types: tuple[str, ...]
    height: int
    width: int
    box_count: int


@dataclass(frozen=True, slots=True)
class OracleOverhead:
    """Action and push overhead against the bounded oracle."""

    action_overhead_vs_oracle: int | None = None
    push_overhead_vs_oracle: int | None = None


@dataclass(frozen=True, slots=True)
class ResearchEpisodeRecord:
    """Nested in-process record with an explicit flat artifact adapter."""

    identity: EpisodeIdentity
    level: LevelProfile
    outcome: EpisodeOutcome
    oracle: OracleOverhead = OracleOverhead()
    strategy: StrategyUsage | None = None
    llm: LLMUsage | None = None
    memory: MemoryUsage | None = None
    rules: RuleUsage | None = None
    local_search: SearchUsage | None = None
    algorithm: SearchUsage | None = None
    prompt: PromptIdentity | None = None
    policy_elapsed_seconds: float = 0.0

    @property
    def policy_name(self) -> str:
        return self.identity.policy_name

    @property
    def level_id(self) -> str:
        return self.identity.level_id

    @property
    def seed(self) -> int:
        if self.identity.seed is None:
            raise ValueError("research records require a seed")
        return self.identity.seed

    @property
    def action_count(self) -> int:
        return self.outcome.action_count

    @property
    def subgoal_successes(self) -> int:
        return self.strategy.subgoal_successes if self.strategy else 0

    def to_flat_dict(self) -> dict[str, object]:
        """Return stable scalar columns for JSON and pandas consumers."""

        strategy = self.strategy
        llm = self.llm
        memory = self.memory
        rules = self.rules
        local = self.local_search
        algorithm = self.algorithm
        prompt = self.prompt
        return {
            "policy_name": self.policy_name,
            "level_id": self.level_id,
            "difficulty": self.level.difficulty,
            "seed": self.seed,
            "layout_family": self.level.layout_family,
            "corridor_structure": self.level.corridor_structure,
            "trap_types": self.level.trap_types,
            "board_height": self.level.height,
            "board_width": self.level.width,
            "box_count": self.level.box_count,
            "status": self.outcome.status,
            "success": self.outcome.success,
            "deadlock": self.outcome.deadlock,
            "truncated": self.outcome.truncated,
            "cycle_detected": self.outcome.cycle_detected,
            "action_count": self.action_count,
            "action_sequence": self.outcome.action_sequence,
            "push_count": self.outcome.push_count,
            **asdict(self.oracle),
            "strategy_proposals": strategy.proposals if strategy else None,
            "strategy_schema_rejections": (
                strategy.schema_rejections if strategy else None
            ),
            "strategy_semantic_rejections": (
                strategy.semantic_rejections if strategy else None
            ),
            "subgoal_attempts": (
                strategy.subgoal_attempts if strategy else None
            ),
            "subgoal_successes": (
                strategy.subgoal_successes if strategy else None
            ),
            "subgoal_failures": (
                strategy.subgoal_failures if strategy else None
            ),
            "plan_revision_count": (
                strategy.plan_revisions if strategy else None
            ),
            "assignment_revision_count": (
                strategy.assignment_revisions if strategy else None
            ),
            "hypothesis_revision_count": (
                strategy.hypothesis_revisions if strategy else None
            ),
            "subgoal_revision_count": (
                strategy.subgoal_revisions if strategy else None
            ),
            "protected_constraint_violations": (
                strategy.protected_constraint_violations
                if strategy
                else None
            ),
            "effect_matches": strategy.effect_matches if strategy else None,
            "effect_mismatches": (
                strategy.effect_mismatches if strategy else None
            ),
            "llm_calls": llm.calls if llm else None,
            "llm_prompt_tokens": llm.prompt_tokens if llm else None,
            "llm_output_tokens": llm.output_tokens if llm else None,
            "llm_elapsed_seconds": llm.elapsed_seconds if llm else None,
            "memory_requests": memory.requests if memory else None,
            "memory_hits": memory.hits if memory else None,
            "memory_writes": memory.writes if memory else None,
            "strategy_cache_hits": (
                memory.strategy_cache_hits if memory else None
            ),
            "grounding_cache_hits": (
                memory.grounding_cache_hits if memory else None
            ),
            "analysis_cache_hits": (
                memory.analysis_cache_hits if memory else None
            ),
            "llm_calls_saved": memory.llm_calls_saved if memory else None,
            "rejected_pushes_filtered": (
                memory.rejected_pushes_filtered if memory else None
            ),
            "rule_checks": rules.checks if rules else None,
            "reachability_calls": (
                rules.reachability_calls if rules else None
            ),
            "local_search_calls": local.calls if local else None,
            "local_expanded_states": (
                local.expanded_states if local else None
            ),
            "local_search_elapsed_seconds": (
                local.elapsed_seconds if local else None
            ),
            "algorithm_calls": algorithm.calls if algorithm else None,
            "algorithm_expanded_states": (
                algorithm.expanded_states if algorithm else None
            ),
            "algorithm_elapsed_seconds": (
                algorithm.elapsed_seconds if algorithm else None
            ),
            "prompt_name": prompt.name if prompt else None,
            "prompt_commit": prompt.commit if prompt else None,
            "model_name": prompt.model_name if prompt else None,
            "policy_elapsed_seconds": self.policy_elapsed_seconds,
        }


@dataclass(frozen=True, slots=True)
class ResearchPolicySummary:
    """Compact policy-level success, reasoning, and cost aggregates."""

    policy_name: str
    episode_count: int
    success_rate: float
    mean_actions: float
    subgoal_success_rate: float | None
    effect_match_rate: float | None
    total_protected_violations: int
    total_llm_calls: int
    total_llm_tokens: int
    total_memory_requests: int
    total_memory_hits: int
    total_memory_writes: int
    total_llm_calls_saved: int
    total_rule_checks: int
    total_reachability_calls: int
    total_local_search_calls: int
    total_local_expanded_states: int
    total_algorithm_calls: int
    total_algorithm_expanded_states: int
    mean_policy_elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class RationaleIntervention:
    """Behavioral difference after removing natural-language rationale."""

    compared_cases: int
    action_sequence_changes: int
    success_changes: int
    mean_action_count_delta: float


@dataclass(frozen=True, slots=True)
class ResearchExperiment:
    """Serializable research handoff with records and causal summary."""

    run_manifest: dict[str, object]
    records: tuple[ResearchEpisodeRecord, ...]
    summaries: tuple[ResearchPolicySummary, ...]
    rationale_intervention: RationaleIntervention

    def to_json_dict(self) -> dict[str, object]:
        """Return a JSON-safe experiment payload."""

        return {
            "run_manifest": self.run_manifest,
            "record_schema_version": 2,
            "records": [record.to_flat_dict() for record in self.records],
            "summaries": [asdict(summary) for summary in self.summaries],
            "rationale_intervention": asdict(self.rationale_intervention),
        }

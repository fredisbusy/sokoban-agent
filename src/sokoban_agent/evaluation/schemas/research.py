"""Research experiment records and aggregate contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class ResearchEpisodeRecord:
    """Normalized outcome and separated costs for one policy case."""

    policy_name: str
    level_id: str
    difficulty: str
    seed: int
    layout_family: str
    corridor_structure: str
    trap_types: tuple[str, ...]
    board_height: int
    board_width: int
    box_count: int
    status: str
    success: bool
    deadlock: bool
    truncated: bool
    action_count: int
    action_sequence: tuple[str, ...]
    push_count: int
    action_overhead_vs_oracle: int | None = None
    push_overhead_vs_oracle: int | None = None
    strategy_proposals: int = 0
    subgoal_attempts: int = 0
    subgoal_successes: int = 0
    subgoal_failures: int = 0
    assignment_revision_count: int = 0
    hypothesis_revision_count: int = 0
    protected_constraint_violations: int = 0
    effect_matches: int = 0
    effect_mismatches: int = 0
    actions_derived_from_subgoal: int = 0
    llm_calls: int = 0
    llm_prompt_tokens: int = 0
    llm_output_tokens: int = 0
    llm_elapsed_seconds: float = 0.0
    memory_requests: int = 0
    memory_hits: int = 0
    memory_writes: int = 0
    llm_calls_saved: int = 0
    rule_checks: int = 0
    reachability_calls: int = 0
    local_search_calls: int = 0
    local_expanded_states: int = 0
    local_search_elapsed_seconds: float = 0.0
    algorithm_calls: int = 0
    algorithm_expanded_states: int = 0
    algorithm_elapsed_seconds: float = 0.0
    policy_elapsed_seconds: float = 0.0


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
            "records": [asdict(record) for record in self.records],
            "summaries": [asdict(summary) for summary in self.summaries],
            "rationale_intervention": asdict(self.rationale_intervention),
        }

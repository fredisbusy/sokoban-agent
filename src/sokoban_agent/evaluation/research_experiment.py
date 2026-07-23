"""Six-policy held-out comparison over shared LangGraph runtimes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

from sokoban_agent.env import FixedLevelProvider, SokobanEnv
from sokoban_agent.evaluation.agentic_cohort import (
    AgenticCohortManifest,
    AgenticLevelCase,
)
from sokoban_agent.evaluation.agentic_results import AgenticEpisodeResult
from sokoban_agent.evaluation.agentic_runner import run_agentic_episode
from sokoban_agent.evaluation.reference import (
    ReferenceResult,
    measure_bounded_astar_reference,
)
from sokoban_agent.evaluation.research_results import (
    POLICY_NAMES,
    ResearchEpisodeRecord,
    ResearchExperiment,
    measure_rationale_intervention,
    summarize_research,
)
from sokoban_agent.evaluation.results import EpisodeResult
from sokoban_agent.evaluation.traces import run_episode_trace
from sokoban_agent.planning import Planner
from sokoban_agent.planning.strategy_runtime import (
    PromptSource,
    StrategyGenerator,
)

_STRUCTURED_VARIANTS: tuple[
    tuple[
        str,
        Literal["on", "off"],
        Literal["direct", "local-search"],
    ],
    ...,
] = (
    ("structured-llm", "on", "direct"),
    ("structured-local-search", "on", "local-search"),
    ("structured-no-rationale", "off", "local-search"),
)

@dataclass(frozen=True, slots=True)
class ResearchRunConfig:
    """Reproducibility identity shared by every comparison record."""

    prompt_name: str
    prompt_commit: str
    graph_id: str
    graph_revision: str
    model_name: str
    seeds: tuple[int, ...]
    max_steps: int = 100
    max_planning_attempts: int = 3
    oracle_max_expanded_states: int = 100_000
    model_config: dict[str, object] = field(default_factory=dict)
    dirty_worktree: bool = False

    def __post_init__(self) -> None:
        if self.prompt_commit in {"", "unresolved", "latest"}:
            raise ValueError("research runs require an immutable prompt commit")
        if not self.seeds:
            raise ValueError("research runs require at least one seed")
        if self.max_steps < 1 or self.oracle_max_expanded_states < 1:
            raise ValueError("research limits must be positive")


def run_research_experiment(
    cohort: AgenticCohortManifest,
    config: ResearchRunConfig,
    *,
    primitive_planner: Planner,
    full_guard_planner: Planner,
    prompt_source: PromptSource | None = None,
    strategy_generator: StrategyGenerator | None = None,
) -> ResearchExperiment:
    """Run all six policies over the exact same level and seed grid."""

    records: list[ResearchEpisodeRecord] = []
    for case in cohort.levels:
        reference = _reference(case, config)
        for seed in config.seeds:
            records.append(
                _legacy_record(
                    case,
                    seed,
                    "primitive-llm",
                    primitive_planner,
                    reference,
                    config,
                )
            )
            for policy_name, rationale_mode, grounding_mode in (
                _STRUCTURED_VARIANTS
            ):
                result = run_agentic_episode(
                    case.graph_input(seed=seed, max_steps=config.max_steps),
                    context={
                        "prompt_name": config.prompt_name,
                        "prompt_commit": config.prompt_commit,
                        "model_name": config.model_name,
                        "rationale_mode": rationale_mode,
                        "grounding_mode": grounding_mode,
                        "memory_mode": "off",
                    },
                    prompt_source=prompt_source,
                    strategy_generator=strategy_generator,
                    thread_id=f"{policy_name}:{case.level_id}:{seed}",
                )
                records.append(
                    _agentic_record(case, seed, result, reference)
                )
            records.append(
                _legacy_record(
                    case,
                    seed,
                    "current-full-guard",
                    full_guard_planner,
                    reference,
                    config,
                )
            )
            records.append(_oracle_record(case, seed, reference))

    frozen_records = tuple(records)
    return ResearchExperiment(
        run_manifest=_run_manifest(cohort, config),
        records=frozen_records,
        summaries=summarize_research(frozen_records),
        rationale_intervention=measure_rationale_intervention(frozen_records),
    )


def _reference(
    case: AgenticLevelCase,
    config: ResearchRunConfig,
) -> ReferenceResult:
    env = _env(case, config.max_steps)
    try:
        return measure_bounded_astar_reference(
            env,
            case.level_id,
            max_expanded_states=config.oracle_max_expanded_states,
        )
    finally:
        env.close()


def _legacy_record(
    case: AgenticLevelCase,
    seed: int,
    policy_name: str,
    planner: Planner,
    reference: ReferenceResult,
    config: ResearchRunConfig,
) -> ResearchEpisodeRecord:
    env = _env(case, config.max_steps)
    try:
        trace = run_episode_trace(
            env,
            planner,
            seed=seed,
            level_id=case.level_id,
            max_planning_attempts=config.max_planning_attempts,
        )
    finally:
        env.close()
    result = trace.result
    actions = tuple(
        frame.action.name for frame in trace.frames if frame.action is not None
    )
    return ResearchEpisodeRecord(
        policy_name=policy_name,
        level_id=case.level_id,
        difficulty=case.difficulty,
        seed=seed,
        layout_family=case.layout_family,
        corridor_structure=case.corridor_structure,
        trap_types=case.trap_types,
        board_height=case.height,
        board_width=case.width,
        box_count=case.box_count,
        status=_legacy_status(result),
        success=result.success,
        deadlock=result.deadlock,
        truncated=result.truncated,
        action_count=result.action_count,
        action_sequence=actions,
        push_count=result.push_count,
        action_overhead_vs_oracle=_overhead(
            result.success, result.action_count, reference.action_count
        ),
        push_overhead_vs_oracle=_overhead(
            result.success, result.push_count, reference.push_count
        ),
        llm_calls=result.llm_calls,
        llm_prompt_tokens=result.llm_prompt_tokens,
        llm_output_tokens=result.llm_output_tokens,
        llm_elapsed_seconds=result.llm_elapsed_seconds,
        algorithm_calls=result.algorithm_calls,
        algorithm_expanded_states=result.algorithm_expanded_states,
        algorithm_elapsed_seconds=result.algorithm_elapsed_seconds,
        policy_elapsed_seconds=result.policy_elapsed_seconds,
    )


def _agentic_record(
    case: AgenticLevelCase,
    seed: int,
    result: AgenticEpisodeResult,
    reference: ReferenceResult,
) -> ResearchEpisodeRecord:
    return ResearchEpisodeRecord(
        policy_name=result.policy_name,
        level_id=case.level_id,
        difficulty=case.difficulty,
        seed=seed,
        layout_family=case.layout_family,
        corridor_structure=case.corridor_structure,
        trap_types=case.trap_types,
        board_height=case.height,
        board_width=case.width,
        box_count=case.box_count,
        status=result.status,
        success=result.success,
        deadlock=result.deadlock,
        truncated=result.truncated,
        action_count=result.action_count,
        action_sequence=result.action_sequence,
        push_count=result.push_count,
        action_overhead_vs_oracle=_overhead(
            result.success, result.action_count, reference.action_count
        ),
        push_overhead_vs_oracle=_overhead(
            result.success, result.push_count, reference.push_count
        ),
        strategy_proposals=result.strategy_proposals,
        subgoal_attempts=result.subgoal_attempts,
        subgoal_successes=result.subgoal_successes,
        subgoal_failures=result.subgoal_failures,
        assignment_revision_count=result.assignment_revision_count,
        hypothesis_revision_count=result.hypothesis_revision_count,
        protected_constraint_violations=(
            result.protected_constraint_violations
        ),
        effect_matches=result.effect_matches,
        effect_mismatches=result.effect_mismatches,
        actions_derived_from_subgoal=result.actions_derived_from_subgoal,
        llm_calls=result.llm_calls,
        llm_prompt_tokens=result.llm_prompt_tokens,
        llm_output_tokens=result.llm_output_tokens,
        llm_elapsed_seconds=result.llm_elapsed_seconds,
        memory_requests=result.memory_requests,
        memory_hits=result.memory_hits,
        memory_writes=result.memory_writes,
        llm_calls_saved=result.llm_calls_saved,
        rule_checks=result.rule_checks,
        reachability_calls=result.reachability_calls,
        local_search_calls=result.local_search_calls,
        local_expanded_states=result.local_expanded_states,
        local_search_elapsed_seconds=result.local_search_elapsed_seconds,
        policy_elapsed_seconds=result.elapsed_seconds,
    )


def _oracle_record(
    case: AgenticLevelCase,
    seed: int,
    result: ReferenceResult,
) -> ResearchEpisodeRecord:
    return ResearchEpisodeRecord(
        policy_name="astar-oracle",
        level_id=case.level_id,
        difficulty=case.difficulty,
        seed=seed,
        layout_family=case.layout_family,
        corridor_structure=case.corridor_structure,
        trap_types=case.trap_types,
        board_height=case.height,
        board_width=case.width,
        box_count=case.box_count,
        status="solved" if result.solved else "oracle-unresolved",
        success=result.solved,
        deadlock=False,
        truncated=False,
        action_count=result.action_count or 0,
        action_sequence=result.action_sequence,
        push_count=result.push_count or 0,
        algorithm_calls=1,
        algorithm_expanded_states=result.expanded_states,
        algorithm_elapsed_seconds=result.elapsed_seconds,
        policy_elapsed_seconds=result.elapsed_seconds,
    )

def _run_manifest(
    cohort: AgenticCohortManifest,
    config: ResearchRunConfig,
) -> dict[str, object]:
    return {
        "cohort_version": cohort.version,
        "cohort_sha256": cohort.cohort_sha256,
        "split": cohort.split,
        "level_ids": [case.level_id for case in cohort.levels],
        "policies": list(POLICY_NAMES),
        **asdict(config),
    }


def _env(case: AgenticLevelCase, max_steps: int) -> SokobanEnv:
    return SokobanEnv(
        level_provider=FixedLevelProvider([case.to_level()]),
        max_steps=max_steps,
    )


def _legacy_status(result: EpisodeResult) -> str:
    if result.success:
        return "success"
    if result.deadlock:
        return "deadlock"
    if result.truncated:
        return "step_limit"
    return result.failure_reason or "failed"


def _overhead(
    success: bool,
    value: int,
    reference: int | None,
) -> int | None:
    return value - reference if success and reference is not None else None

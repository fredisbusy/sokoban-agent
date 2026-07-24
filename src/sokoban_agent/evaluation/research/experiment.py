"""Six-policy held-out comparison over shared LangGraph runtimes."""

from __future__ import annotations

from dataclasses import asdict
from typing import Literal

from sokoban_agent.env import FixedLevelProvider, SokobanEnv
from sokoban_agent.evaluation.agentic.runner import run_agentic_episode
from sokoban_agent.evaluation.baseline.traces import run_episode_trace
from sokoban_agent.evaluation.research.cohort import (
    AgenticCohortManifest,
    AgenticLevelCase,
)
from sokoban_agent.evaluation.research.config import (
    ResearchRunConfig as ResearchRunConfig,
)
from sokoban_agent.evaluation.research.reference import (
    measure_bounded_astar_reference,
)
from sokoban_agent.evaluation.research.results import (
    POLICY_NAMES,
    measure_rationale_intervention,
    summarize_research,
)
from sokoban_agent.evaluation.schemas.episode import (
    AgenticEpisodeResult,
    EpisodeIdentity,
    EpisodeOutcome,
    EpisodeResult,
)
from sokoban_agent.evaluation.schemas.metrics import LLMUsage, SearchUsage
from sokoban_agent.evaluation.schemas.reference import ReferenceResult
from sokoban_agent.evaluation.schemas.research import (
    LevelProfile,
    OracleOverhead,
    ResearchEpisodeRecord,
    ResearchExperiment,
)
from sokoban_agent.graph.agentic.runtime import AgenticGraphRunner
from sokoban_agent.planning import Planner
from sokoban_agent.planning.agentic.runtime import (
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
    agentic_runner = AgenticGraphRunner(
        prompt_source=prompt_source,
        strategy_generator=strategy_generator,
    )
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
                    runner=agentic_runner,
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
        identity=EpisodeIdentity(policy_name, case.level_id, seed),
        level=_level_profile(case),
        outcome=EpisodeOutcome(
            status=_legacy_status(result),
            success=result.success,
            deadlock=result.deadlock,
            truncated=result.truncated,
            cycle_detected=False,
            action_sequence=actions,
            push_count=result.push_count,
        ),
        oracle=OracleOverhead(
            action_overhead_vs_oracle=_overhead(
                result.success, result.action_count, reference.action_count
            ),
            push_overhead_vs_oracle=_overhead(
                result.success, result.push_count, reference.push_count
            ),
        ),
        llm=LLMUsage(
            calls=result.llm_calls,
            elapsed_seconds=result.llm_elapsed_seconds,
            prompt_tokens=result.llm_prompt_tokens,
            output_tokens=result.llm_output_tokens,
        ),
        algorithm=SearchUsage(
            calls=result.algorithm_calls,
            expanded_states=result.algorithm_expanded_states,
            elapsed_seconds=result.algorithm_elapsed_seconds,
        ),
        policy_elapsed_seconds=result.policy_elapsed_seconds,
    )


def _agentic_record(
    case: AgenticLevelCase,
    seed: int,
    result: AgenticEpisodeResult,
    reference: ReferenceResult,
) -> ResearchEpisodeRecord:
    return ResearchEpisodeRecord(
        identity=EpisodeIdentity(result.policy_name, case.level_id, seed),
        level=_level_profile(case),
        outcome=result.outcome,
        oracle=OracleOverhead(
            action_overhead_vs_oracle=_overhead(
                result.success, result.action_count, reference.action_count
            ),
            push_overhead_vs_oracle=_overhead(
                result.success, result.push_count, reference.push_count
            ),
        ),
        strategy=result.strategy,
        llm=result.llm,
        memory=result.memory,
        rules=result.rules,
        local_search=result.local_search,
        prompt=result.prompt,
        policy_elapsed_seconds=result.elapsed_seconds,
    )


def _oracle_record(
    case: AgenticLevelCase,
    seed: int,
    result: ReferenceResult,
) -> ResearchEpisodeRecord:
    return ResearchEpisodeRecord(
        identity=EpisodeIdentity("astar-oracle", case.level_id, seed),
        level=_level_profile(case),
        outcome=EpisodeOutcome(
            status="solved" if result.solved else "oracle-unresolved",
            success=result.solved,
            deadlock=False,
            truncated=False,
            cycle_detected=False,
            action_sequence=result.action_sequence,
            push_count=result.push_count or 0,
        ),
        algorithm=SearchUsage(
            calls=1,
            expanded_states=result.expanded_states,
            elapsed_seconds=result.elapsed_seconds,
        ),
        policy_elapsed_seconds=result.elapsed_seconds,
    )


def _level_profile(case: AgenticLevelCase) -> LevelProfile:
    return LevelProfile(
        difficulty=case.difficulty,
        layout_family=case.layout_family,
        corridor_structure=case.corridor_structure,
        trap_types=case.trap_types,
        height=case.height,
        width=case.width,
        box_count=case.box_count,
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

"""Run the pinned six-policy held-out agentic experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

from sokoban_agent.evaluation import (
    AgenticCohortManifest,
    ResearchRunConfig,
    load_agentic_cohort_manifest,
    run_research_experiment,
)
from sokoban_agent.planning import LLMPlanner, SearchGuardPlanner
from sokoban_agent.planning.llm.client import LiteLLMClient, OllamaSettings


def main() -> None:
    """Run the comparison and persist one self-describing JSON artifact."""

    args = _parse_args()
    load_dotenv(".env", override=True)
    cohort = load_agentic_cohort_manifest(args.manifest)
    cohort = _select_cohort(
        cohort,
        difficulties=args.difficulties,
        cases_per_difficulty=args.cases_per_difficulty,
    )
    settings = OllamaSettings.from_env()
    git_commit = _git_output("rev-parse", "HEAD")
    dirty = bool(_git_output("status", "--short"))
    config = ResearchRunConfig(
        prompt_name=args.prompt_name,
        prompt_commit=args.prompt_commit,
        graph_id="sokoban_agent",
        graph_revision=git_commit,
        model_name=settings.model,
        seeds=tuple(args.seeds),
        max_steps=args.max_steps,
        max_planning_attempts=args.max_planning_attempts,
        oracle_max_expanded_states=args.max_expanded_states,
        model_config={
            "temperature": settings.temperature,
            "num_ctx": settings.num_ctx,
            "max_output_tokens": settings.max_output_tokens,
            "strategy_max_output_tokens": (
                settings.strategy_max_output_tokens
            ),
            "think": settings.think,
        },
        dirty_worktree=dirty,
    )
    primitive_client = LiteLLMClient(settings)
    guard_client = LiteLLMClient(settings)
    primitive = LLMPlanner(
        primitive_client,
        model_name=settings.model,
    )
    full_guard = SearchGuardPlanner(
        LLMPlanner(guard_client, model_name=settings.model),
        algorithm="astar",
        max_expanded_states=args.max_expanded_states,
        fallback_policy="full",
        measure_contribution=True,
    )
    experiment = run_research_experiment(
        cohort,
        config,
        primitive_planner=primitive,
        full_guard_planner=full_guard,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            experiment.to_json_dict(),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for summary in experiment.summaries:
        print(
            f"{summary.policy_name}: "
            f"success={summary.success_rate:.1%}, "
            f"episodes={summary.episode_count}"
        )
    print(f"artifact: {args.output}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmarks/boxoban_research_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("_workspace/benchmarks/agentic_research_v1.json"),
    )
    parser.add_argument("--prompt-name", default="sokoban-strategy")
    parser.add_argument(
        "--prompt-commit",
        required=True,
        help="LangSmith immutable prompt commit hash",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--difficulties",
        nargs="+",
        choices=["unfiltered", "medium", "hard"],
    )
    parser.add_argument("--cases-per-difficulty", type=int)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-planning-attempts", type=int, default=3)
    parser.add_argument("--max-expanded-states", type=int, default=100_000)
    args = parser.parse_args()
    if any(seed < 0 for seed in args.seeds):
        parser.error("--seeds must be non-negative")
    if (
        args.cases_per_difficulty is not None
        and args.cases_per_difficulty < 1
    ):
        parser.error("--cases-per-difficulty must be positive")
    return args


def _select_cohort(
    cohort: AgenticCohortManifest,
    *,
    difficulties: list[str] | None,
    cases_per_difficulty: int | None,
) -> AgenticCohortManifest:
    selected_difficulties = set(difficulties or ())
    levels = [
        case
        for case in cohort.levels
        if not selected_difficulties
        or case.difficulty in selected_difficulties
    ]
    if cases_per_difficulty is not None:
        counts: dict[str, int] = {}
        limited = []
        for case in levels:
            count = counts.get(case.difficulty, 0)
            if count >= cases_per_difficulty:
                continue
            limited.append(case)
            counts[case.difficulty] = count + 1
        levels = limited
    if not levels:
        raise ValueError("research selection must contain at least one level")
    return replace(cohort, levels=tuple(levels))


def _git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        text=True,
    ).strip()


if __name__ == "__main__":
    main()

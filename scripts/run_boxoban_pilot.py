"""Run the five-policy Boxoban contribution pilot with resumable JSONL."""

from __future__ import annotations

import argparse
import json
import subprocess
from hashlib import sha256
from pathlib import Path
from typing import Any

from sokoban_agent.env import SokobanEnv
from sokoban_agent.evaluation import (
    EpisodeResult,
    load_cohort_manifest,
    measure_bounded_astar_reference,
    run_episode,
    summarize_by_planner,
)
from sokoban_agent.evaluation.schemas.baseline_rows import (
    EPISODE_SCHEMA_VERSION,
    SUMMARY_SCHEMA_VERSION,
    BaselineEpisodeRowV1,
    planner_summary_to_flat_dict,
)
from sokoban_agent.planning import (
    AStarPlanner,
    LLMPlanner,
    Planner,
    PlanningContext,
    PlanningOutcome,
    SearchGuardPlanner,
)
from sokoban_agent.planning.llm.client import LiteLLMClient, OllamaSettings

VARIANTS = (
    "astar-only",
    "llm-common-validation",
    "llm-suffix-only",
    "llm-full-guard",
    "llm-always-replace",
)


class VariantPlanner:
    """Give one planner policy a stable experiment identifier."""

    def __init__(self, name: str, delegate: Planner) -> None:
        self._name = name
        self._delegate = delegate

    @property
    def name(self) -> str:
        return self._name

    def reset(self, *, seed: int | None = None) -> None:
        self._delegate.reset(seed=seed)

    def plan(self, context: PlanningContext) -> PlanningOutcome:
        return self._delegate.plan(context)


def main() -> None:
    args = _parse_args()
    manifest, provider = load_cohort_manifest(args.manifest, args.data_root)
    level_ids = manifest.level_ids[: args.cohort_size]
    settings = OllamaSettings.from_env()
    planners = _build_planners(settings, args.max_expanded_states)
    selected = {name: planners[name] for name in args.variants}
    config = _run_config(args, manifest.version, settings, level_ids)
    config_hash = _hash(config)
    completed, records = _load_existing(args.output, config_hash)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    reference_cache = {
        level_id: measure_bounded_astar_reference(
            SokobanEnv(level_provider=provider, max_steps=args.max_steps),
            level_id,
            max_expanded_states=args.max_expanded_states,
        )
        for level_id in level_ids
    }
    for repeat in range(args.repeats):
        for level_index, level_id in enumerate(level_ids):
            names = list(selected)
            shift = (level_index + repeat) % len(names)
            names = names[shift:] + names[:shift]
            for variant in names:
                key = (variant, level_id, repeat)
                if key in completed:
                    continue
                result = run_episode(
                    SokobanEnv(
                        level_provider=provider,
                        max_steps=args.max_steps,
                    ),
                    selected[variant],
                    seed=repeat,
                    level_id=level_id,
                    max_planning_attempts=args.max_planning_attempts,
                    reference_result=reference_cache[level_id],
                )
                record = {
                    "record_type": "episode",
                    "episode_schema_version": EPISODE_SCHEMA_VERSION,
                    "run_config_hash": config_hash,
                    "variant": variant,
                    "repeat": repeat,
                    "result": BaselineEpisodeRowV1.to_dict(result),
                }
                _append_json(args.output, record)
                records.append(record)
                completed.add(key)
                print(
                    f"{variant} {level_id} repeat={repeat} "
                    f"success={result.success} "
                    f"policy={result.policy_elapsed_seconds:.3f}s"
                )

    matching_results = [
        _decode_episode_record(record)
        for record in records
        if record.get("run_config_hash") == config_hash
    ]
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "config": config,
                "run_config_hash": config_hash,
                "summary_schema_version": SUMMARY_SCHEMA_VERSION,
                "summaries": [
                    planner_summary_to_flat_dict(summary)
                    for summary in summarize_by_planner(matching_results)
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"summary: {summary_path}")


def _decode_episode_record(record: dict[str, Any]) -> EpisodeResult:
    version = record.get("episode_schema_version", EPISODE_SCHEMA_VERSION)
    if version != EPISODE_SCHEMA_VERSION:
        raise ValueError(f"unsupported episode schema version: {version}")
    payload = record.get("result")
    if not isinstance(payload, dict):
        raise ValueError("episode record result must be an object")
    return BaselineEpisodeRowV1.from_dict(payload)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmarks/boxoban_pilot_v1.json"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/boxoban"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("_workspace/benchmarks/boxoban_pilot_v1.jsonl"),
    )
    parser.add_argument("--cohort-size", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-planning-attempts", type=int, default=3)
    parser.add_argument("--max-expanded-states", type=int, default=100_000)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=list(VARIANTS),
    )
    args = parser.parse_args()
    if not 1 <= args.cohort_size <= 50:
        parser.error("--cohort-size must be between 1 and 50")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    return args


def _build_planners(
    settings: OllamaSettings,
    max_expanded_states: int,
) -> dict[str, Planner]:
    def llm() -> LLMPlanner:
        client = LiteLLMClient(settings)
        return LLMPlanner(client, model_name=settings.model)

    return {
        "astar-only": VariantPlanner(
            "astar-only",
            AStarPlanner(max_expanded_states=max_expanded_states),
        ),
        "llm-common-validation": VariantPlanner(
            "llm-common-validation",
            llm(),
        ),
        "llm-suffix-only": VariantPlanner(
            "llm-suffix-only",
            SearchGuardPlanner(
                llm(),
                max_expanded_states=max_expanded_states,
                fallback_policy="none",
                measure_contribution=True,
            ),
        ),
        "llm-full-guard": VariantPlanner(
            "llm-full-guard",
            SearchGuardPlanner(
                llm(),
                max_expanded_states=max_expanded_states,
                fallback_policy="full",
                measure_contribution=True,
            ),
        ),
        "llm-always-replace": VariantPlanner(
            "llm-always-replace",
            SearchGuardPlanner(
                llm(),
                max_expanded_states=max_expanded_states,
                fallback_policy="always",
                measure_contribution=True,
            ),
        ),
    }


def _run_config(
    args: argparse.Namespace,
    cohort_version: str,
    settings: OllamaSettings,
    level_ids: tuple[str, ...],
) -> dict[str, object]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--short"],
            text=True,
        ).strip()
    )
    return {
        "cohort_version": cohort_version,
        "level_ids": list(level_ids),
        "independent_level_count": len(level_ids),
        "model_repeats": args.repeats,
        "variants": args.variants,
        "model": settings.model,
        "ollama_api_base": settings.api_base,
        "temperature": settings.temperature,
        "num_ctx": settings.num_ctx,
        "max_output_tokens": settings.max_output_tokens,
        "llm_timeout_seconds": settings.timeout_seconds,
        "think": settings.think,
        "prompt_version": "korean-plan-v1",
        "max_steps": args.max_steps,
        "max_planning_attempts": args.max_planning_attempts,
        "max_expanded_states": args.max_expanded_states,
        "git_commit": commit,
        "dirty_worktree": dirty,
    }


def _load_existing(
    path: Path,
    config_hash: str,
) -> tuple[set[tuple[str, str, int]], list[dict[str, object]]]:
    completed: set[tuple[str, str, int]] = set()
    records: list[dict[str, object]] = []
    if not path.is_file():
        return completed, records
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if not isinstance(record, dict):
            continue
        records.append(record)
        if record.get("run_config_hash") != config_hash:
            continue
        result = record.get("result")
        if not isinstance(result, dict):
            continue
        completed.add(
            (
                str(record["variant"]),
                str(result["level_id"]),
                int(record["repeat"]),
            )
        )
    return completed, records


def _append_json(path: Path, value: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, ensure_ascii=False) + "\n")


def _hash(value: object) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return sha256(serialized.encode()).hexdigest()


if __name__ == "__main__":
    main()

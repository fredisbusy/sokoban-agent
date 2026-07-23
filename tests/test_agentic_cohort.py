import json
from pathlib import Path

import pytest

from sokoban_agent.env import FixedLevelProvider, SokobanEnv
from sokoban_agent.evaluation.agentic_cohort import (
    load_agentic_cohort_manifest,
)
from sokoban_agent.planning import NoSolutionError
from sokoban_agent.planning.astar import solve_astar_result

MANIFEST = (
    Path(__file__).parents[1] / "benchmarks" / "agentic_heldout_v1.json"
)


def test_agentic_cohort_is_structurally_tagged_and_dev_isolated() -> None:
    manifest = load_agentic_cohort_manifest(MANIFEST)

    assert manifest.version == "agentic-heldout-v1"
    assert manifest.split == "test"
    assert len(manifest.levels) >= 6
    assert not (
        {case.level_id for case in manifest.levels}
        & set(manifest.development_level_ids)
    )
    assert {case.layout_family for case in manifest.levels} >= {
        "open-room",
        "straight-corridor",
        "turn",
        "multi-box",
    }
    assert {case.box_count for case in manifest.levels} >= {1, 2}
    assert any(case.trap_types for case in manifest.levels)

    for case in manifest.levels:
        level = case.to_level()
        assert level.shape == (case.height, case.width)
        assert len(level.boxes) == case.box_count
        assert len(level.targets) == case.box_count


def test_agentic_cohort_rejects_board_drift(tmp_path: Path) -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload["levels"][0]["rows"][1] = "# .   #"
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="checksum"):
        load_agentic_cohort_manifest(changed)


def test_agentic_cohort_rejects_development_overlap(tmp_path: Path) -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload["development_level_ids"].append(payload["levels"][0]["level_id"])
    changed = tmp_path / "overlap.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="development"):
        load_agentic_cohort_manifest(changed)


def test_agentic_cohort_oracle_labels_match_bounded_astar() -> None:
    manifest = load_agentic_cohort_manifest(MANIFEST)

    for case in manifest.levels:
        env = SokobanEnv(
            level_provider=FixedLevelProvider([case.to_level()]),
            max_steps=100,
        )
        observation, _ = env.reset()
        env.close()
        if case.oracle_expectation == "solved":
            assert solve_astar_result(observation).actions
        else:
            with pytest.raises(NoSolutionError):
                solve_astar_result(observation)

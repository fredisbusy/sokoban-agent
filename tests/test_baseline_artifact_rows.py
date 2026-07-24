from __future__ import annotations

import pytest

from sokoban_agent.evaluation.schemas.baseline_rows import (
    BaselineEpisodeRowV1,
)


def _minimal_row() -> dict[str, object]:
    return {
        "planner_name": "bfs",
        "level_id": "tiny-push",
        "seed": 0,
        "success": True,
        "deadlock": False,
        "truncated": False,
        "action_count": 2,
        "invalid_moves": 0,
        "total_reward": 10.0,
        "elapsed_seconds": 0.25,
    }


def test_v1_reader_adds_canonical_optional_columns() -> None:
    result = BaselineEpisodeRowV1.from_dict(_minimal_row())
    encoded = BaselineEpisodeRowV1.to_dict(result)

    assert tuple(encoded) == BaselineEpisodeRowV1.columns()
    assert len(encoded) == 57
    assert encoded["planning_calls"] == 0
    assert encoded["failure_reason"] is None
    assert encoded["policy_elapsed_seconds"] == 0.25


@pytest.mark.parametrize("value", ["false", 0, 1])
def test_v1_reader_rejects_non_boolean_success(value: object) -> None:
    row = {**_minimal_row(), "success": value}

    with pytest.raises(ValueError, match="success must be a boolean"):
        BaselineEpisodeRowV1.from_dict(row)


def test_v1_reader_rejects_unknown_and_inconsistent_derived_fields() -> None:
    with pytest.raises(ValueError, match="unknown keys"):
        BaselineEpisodeRowV1.from_dict(
            {**_minimal_row(), "future_metric": 1}
        )

    with pytest.raises(ValueError, match="policy_elapsed_seconds"):
        BaselineEpisodeRowV1.from_dict(
            {**_minimal_row(), "policy_elapsed_seconds": 9.0}
        )


def test_policy_elapsed_is_clamped_to_zero() -> None:
    row = {
        **_minimal_row(),
        "guard_reference_elapsed_seconds": 1.0,
        "policy_elapsed_seconds": 0.0,
    }

    result = BaselineEpisodeRowV1.from_dict(row)

    assert result.policy_elapsed_seconds == 0.0

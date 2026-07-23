import pytest
from pydantic import ValidationError

from sokoban_agent.planning import (
    BoardAnalysis,
    PlanRevision,
    StrategyHypothesis,
    validate_strategy,
)


def _board_analysis() -> BoardAnalysis:
    return BoardAnalysis.model_validate(
        {
            "boxes": [{"box_id": "B1", "position": {"row": 2, "col": 2}}],
            "targets": [
                {"target_id": "T1", "position": {"row": 1, "col": 2}}
            ],
            "dead_squares": [],
            "reachable_cells": [
                {"row": 2, "col": 1},
                {"row": 2, "col": 3},
                {"row": 3, "col": 2},
            ],
            "push_options": [
                {
                    "box_id": "B1",
                    "direction": "UP",
                    "support": {"row": 3, "col": 2},
                    "destination": {"row": 1, "col": 2},
                    "creates_static_deadlock": False,
                }
            ],
            "reverse_pull_distances": [
                {"box_id": "B1", "target_id": "T1", "distance": 1}
            ],
        }
    )


def _strategy_hypothesis() -> StrategyHypothesis:
    return StrategyHypothesis.model_validate_json(
        """
        {
          "summary": "B1을 T1으로 먼저 민다",
          "assignments": [
            {"box_id": "B1", "target_id": "T1", "reason": "직선으로 접근 가능"}
          ],
          "protected_constraints": [
            {
              "kind": "keep_clear",
              "cells": [{"row": 3, "col": 2}],
              "reason": "B1을 미는 지지 칸"
            }
          ],
          "subgoal": {
            "kind": "push",
            "box_id": "B1",
            "target_id": "T1",
            "direction": "UP",
            "destination": {"row": 1, "col": 2}
          },
          "expected_effect": {
            "box_id": "B1",
            "from_position": {"row": 2, "col": 2},
            "to_position": {"row": 1, "col": 2}
          },
          "failure_conditions": [
            {
              "kind": "support_unreachable",
              "description": "플레이어가 B1 아래에 도달할 수 없음"
            }
          ]
        }
        """
    )


def test_structured_strategy_round_trips_and_matches_board_analysis() -> None:
    strategy = _strategy_hypothesis()

    assert validate_strategy(_board_analysis(), strategy) == []
    assert StrategyHypothesis.model_validate_json(
        strategy.model_dump_json()
    ) == strategy


def test_structured_strategy_rejects_missing_expected_effect() -> None:
    with pytest.raises(ValidationError, match="expected_effect"):
        StrategyHypothesis.model_validate(
            {
                "summary": "B1을 T1으로 민다",
                "assignments": [
                    {
                        "box_id": "B1",
                        "target_id": "T1",
                        "reason": "직선으로 접근 가능",
                    }
                ],
                "protected_constraints": [],
                "subgoal": {
                    "kind": "push",
                    "box_id": "B1",
                    "target_id": "T1",
                    "direction": "UP",
                    "destination": {"row": 1, "col": 2},
                },
                "failure_conditions": [
                    {
                        "kind": "support_unreachable",
                        "description": "지지 칸에 갈 수 없음",
                    }
                ],
            }
        )


def test_board_analysis_rejects_two_box_ids_at_the_same_position() -> None:
    payload = _board_analysis().model_dump(mode="json")
    payload["boxes"].append(
        {"box_id": "B2", "position": {"row": 2, "col": 2}}
    )

    with pytest.raises(ValidationError, match="box positions must be unique"):
        BoardAnalysis.model_validate(payload)


def test_strategy_validation_reports_semantic_contradictions() -> None:
    payload = _strategy_hypothesis().model_dump(mode="json")
    payload["assignments"][0]["target_id"] = "T9"
    payload["protected_constraints"][0]["cells"] = [
        {"row": 1, "col": 2}
    ]
    payload["expected_effect"]["to_position"] = {"row": 1, "col": 1}
    strategy = StrategyHypothesis.model_validate(payload)

    violations = validate_strategy(_board_analysis(), strategy)

    assert {violation.code for violation in violations} == {
        "unknown_target",
        "assignment_conflict",
        "expected_effect_mismatch",
        "protected_cell_conflict",
    }


def test_plan_revision_requires_observable_evidence() -> None:
    revision = PlanRevision.model_validate(
        {
            "step": 1,
            "disposition": "modify",
            "changed_fields": ["subgoal"],
            "evidence": "B1 아래 지지 칸에 도달할 수 없음",
        }
    )

    assert revision.model_dump(mode="json") == {
        "step": 1,
        "disposition": "modify",
        "changed_fields": ["subgoal"],
        "evidence": "B1 아래 지지 칸에 도달할 수 없음",
    }
    with pytest.raises(ValidationError, match="evidence"):
        PlanRevision.model_validate(
            {
                "step": 1,
                "disposition": "modify",
                "changed_fields": ["subgoal"],
            }
        )

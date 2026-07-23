from typing import cast

from sokoban_agent.graph.agentic_execution_nodes import (
    detect_agentic_repetition,
    reflect_agentic_execution,
)
from sokoban_agent.graph.agentic_state import AgenticState


def _reflection_state() -> AgenticState:
    return cast(
        AgenticState,
        {
            "info": {
                "steps": 1,
                "success": False,
                "deadlock": False,
            },
            "expected_effect": {
                "box_id": "B1",
                "from_position": {"row": 2, "col": 2},
                "to_position": {"row": 1, "col": 2},
            },
            "active_subgoal": {
                "kind": "push",
                "box_id": "B1",
                "target_id": "T1",
                "direction": "UP",
                "destination": {"row": 1, "col": 2},
            },
            "board_analysis": {
                "boxes": [
                    {"box_id": "B1", "position": {"row": 2, "col": 2}}
                ],
                "targets": [
                    {"target_id": "T1", "position": {"row": 1, "col": 2}}
                ],
                "dead_squares": [],
                "reachable_cells": [{"row": 3, "col": 2}],
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
            },
            "execution_result": {
                "before_boxes": [[2, 2]],
                "after_boxes": [[2, 3]],
                "actions_requested": ["UP"],
                "actions_executed": ["UP"],
                "push_count": 1,
                "truncated": False,
            },
            "completed_subgoals": [],
            "plan_revisions": [],
            "attempt_keys": [],
            "strategy_attempts": 1,
            "effect_matches": 0,
            "effect_mismatches": 0,
        },
    )


def test_reflection_revises_only_falsified_subgoal() -> None:
    update = reflect_agentic_execution(_reflection_state())

    assert update["status"] == "reflection_failed"
    assert update["plan_revisions"] == [
        {
            "step": 1,
            "disposition": "modify",
            "changed_fields": ["subgoal", "expected_effect"],
            "evidence": "B1이 예상 위치 (1, 2)로 이동하지 않았습니다",
        }
    ]
    assert update["latest_strategy_feedback"] == [
        "unexpected_state: B1이 예상 위치 (1, 2)로 이동하지 않았습니다"
    ]
    assert "strategy_hypothesis" not in update


def test_repetition_ignores_player_position_for_same_planner_state() -> None:
    state = _reflection_state()
    state["observation"] = [[1, 1], [1, 1]]
    first = detect_agentic_repetition(state)
    state["attempt_keys"] = cast(list[str], first["attempt_keys"])
    state["observation"] = [[1, 2], [1, 1]]

    update = detect_agentic_repetition(state)

    assert update["cycle_detected"] is True
    assert update["status"] == "cycle_detected"
    assert update["attempt_keys"] == state["attempt_keys"]

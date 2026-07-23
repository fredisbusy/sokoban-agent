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
    assert "strategy_hypothesis" not in update


def test_repetition_detection_terminates_same_board_and_subgoal() -> None:
    state = _reflection_state()
    state["observation"] = [[1, 1], [1, 1]]
    state["attempt_keys"] = [
        '{"observation":[[1,1],[1,1]],"subgoal":{"box_id":"B1",'
        '"destination":{"col":2,"row":1},"direction":"UP","kind":"push",'
        '"target_id":"T1"}}'
    ]

    update = detect_agentic_repetition(state)

    assert update["cycle_detected"] is True
    assert update["status"] == "cycle_detected"
    assert update["attempt_keys"] == state["attempt_keys"]

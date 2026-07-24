from typing import cast

from sokoban_agent.graph.agentic.metrics import initial_agentic_metrics
from sokoban_agent.graph.agentic.nodes.execution import (
    detect_agentic_repetition,
    reflect_agentic_execution,
)
from sokoban_agent.graph.agentic.state import AgenticState


def _reflection_state() -> AgenticState:
    return cast(
        AgenticState,
        {
            "meta": {"max_steps": 10},
            "info": {
                "steps": 1,
                "success": False,
                "deadlock": False,
            },
            "planning": {
                "strategy_hypothesis": {
                    "subgoal": {
                        "kind": "push",
                        "box_id": "B1",
                        "target_id": "T1",
                        "direction": "UP",
                        "destination": {"row": 1, "col": 2},
                    },
                    "expected_effect": {
                        "box_id": "B1",
                        "from_position": {"row": 2, "col": 2},
                        "to_position": {"row": 1, "col": 2},
                    },
                },
                "completed_subgoals": [],
                "latest_strategy_feedback": [],
                "board_analysis": {
                    "boxes": [
                        {"box_id": "B1", "position": {"row": 2, "col": 2}}
                    ],
                    "targets": [
                        {
                            "target_id": "T1",
                            "position": {"row": 1, "col": 2},
                        }
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
                        {
                            "box_id": "B1",
                            "target_id": "T1",
                            "distance": 1,
                        }
                    ],
                },
            },
            "execution": {
                "result": {
                    "before_boxes": [[2, 2]],
                    "after_boxes": [[2, 3]],
                    "actions_requested": ["UP"],
                    "actions_executed": ["UP"],
                    "push_count": 1,
                    "invalid_move": False,
                    "truncated": False,
                },
                "reflection": None,
            },
            "plan_revisions": [],
            "memory": {"attempt_keys": [], "rejected_pushes": {}},
            "metrics": initial_agentic_metrics(),
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
    planning = cast(dict[str, object], update["planning"])
    assert planning["latest_strategy_feedback"] == [
        "unexpected_state: B1이 예상 위치 (1, 2)로 이동하지 않았습니다"
    ]
    assert "strategy_hypothesis" not in update


def test_repetition_ignores_player_position_for_same_planner_state() -> None:
    state = _reflection_state()
    state["observation"] = [[1, 1], [1, 1]]
    first = detect_agentic_repetition(state)
    memory = cast(dict[str, object], first["memory"])
    state["memory"]["attempt_keys"] = cast(list[str], memory["attempt_keys"])
    state["observation"] = [[1, 2], [1, 1]]

    update = detect_agentic_repetition(state)

    assert update["cycle_detected"] is True
    assert update["status"] == "cycle_detected"
    assert update["memory"] == state["memory"]

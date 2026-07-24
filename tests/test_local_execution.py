import numpy as np
import pytest

from sokoban_agent.env import (
    Action,
    FixedLevelProvider,
    SokobanEnv,
    parse_level,
)
from sokoban_agent.env.rules import (
    apply_action,
    decode_observation,
    is_success,
)
from sokoban_agent.planning import analyze_board
from sokoban_agent.planning.agentic.grounding import (
    SubgoalGroundingError,
    ground_push_subgoal,
    ground_push_subgoal_direct,
)
from sokoban_agent.planning.agentic.models import (
    BoardAnalysis,
    ProtectedConstraint,
    PushSubgoal,
)


def _observation(rows: list[str]) -> np.ndarray:
    level = parse_level("fixture", rows)
    env = SokobanEnv(level_provider=FixedLevelProvider([level]))
    observation, _ = env.reset()
    env.close()
    return observation


def _subgoal(
    *,
    direction: str,
    row: int,
    col: int,
) -> PushSubgoal:
    return PushSubgoal.model_validate(
        {
            "kind": "push",
            "box_id": "B1",
            "target_id": "T1",
            "direction": direction,
            "destination": {"row": row, "col": col},
        }
    )


def test_ground_subgoal_finds_player_path_then_one_push() -> None:
    observation = _observation(
        ["#######", "#  .  #", "#  $  #", "# @   #", "#######"]
    )
    analysis = analyze_board(observation)

    plan = ground_push_subgoal(
        observation,
        analysis,
        _subgoal(direction="UP", row=1, col=3),
        (),
    )

    assert plan.player_actions == ("RIGHT",)
    assert plan.push_action == "UP"
    assert plan.support.model_dump(mode="json") == {"row": 3, "col": 3}
    assert plan.push_count == 1


def test_ground_subgoal_rejects_static_deadlock_push() -> None:
    env = SokobanEnv()
    observation, _ = env.reset(options={"level_id": "tiny-push"})
    env.close()
    analysis = analyze_board(observation)

    with pytest.raises(SubgoalGroundingError) as caught:
        ground_push_subgoal(
            observation,
            analysis,
            _subgoal(direction="RIGHT", row=2, col=3),
            (),
        )

    assert caught.value.kind == "static_deadlock"


def test_ground_subgoal_rechecks_unreachable_support() -> None:
    env = SokobanEnv()
    observation, _ = env.reset(options={"level_id": "tiny-push"})
    env.close()
    payload = analyze_board(observation).model_dump(mode="json")
    payload["push_options"][0]["support"] = {"row": 0, "col": 2}
    stale_analysis = BoardAnalysis.model_validate(payload)

    with pytest.raises(SubgoalGroundingError) as caught:
        ground_push_subgoal(
            observation,
            stale_analysis,
            _subgoal(direction="UP", row=1, col=2),
            (),
        )

    assert caught.value.kind == "support_unreachable"


def test_ground_subgoal_rejects_protected_destination() -> None:
    env = SokobanEnv()
    observation, _ = env.reset(options={"level_id": "tiny-push"})
    env.close()
    analysis = analyze_board(observation)
    protected = ProtectedConstraint.model_validate(
        {
            "kind": "keep_clear",
            "cells": [{"row": 1, "col": 2}],
            "reason": "통로를 유지한다",
        }
    )

    with pytest.raises(SubgoalGroundingError) as caught:
        ground_push_subgoal(
            observation,
            analysis,
            _subgoal(direction="UP", row=1, col=2),
            (protected,),
        )

    assert caught.value.kind == "protected_constraint_violated"


def test_local_grounder_stops_after_one_push_before_puzzle_success() -> None:
    observation = _observation(["#######", "#. $@ #", "#######"])
    analysis = analyze_board(observation)
    plan = ground_push_subgoal(
        observation,
        analysis,
        _subgoal(direction="LEFT", row=1, col=2),
        (),
    )
    level, state = decode_observation(observation)

    pushes = 0
    for action_name in (*plan.player_actions, plan.push_action):
        move = apply_action(level, state, Action[action_name])
        pushes += int(move.pushed)
        state = move.state

    assert pushes == 1
    assert not is_success(level, state)
    assert state.boxes == {(1, 2)}


def test_direct_grounder_requires_player_on_push_support() -> None:
    observation = _observation(
        ["#######", "#  .  #", "#  $  #", "# @   #", "#######"]
    )
    analysis = analyze_board(observation)

    with pytest.raises(SubgoalGroundingError, match="지지 칸"):
        ground_push_subgoal_direct(
            observation,
            analysis,
            _subgoal(direction="UP", row=1, col=3),
            (),
        )


def test_direct_grounder_emits_only_the_available_push() -> None:
    observation = _observation(
        ["#######", "#  .  #", "#  $  #", "#  @  #", "#######"]
    )
    analysis = analyze_board(observation)

    plan = ground_push_subgoal_direct(
        observation,
        analysis,
        _subgoal(direction="UP", row=1, col=3),
        (),
    )

    assert plan.player_actions == ()
    assert plan.push_action == "UP"
    assert plan.push_count == 1

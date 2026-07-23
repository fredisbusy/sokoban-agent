import numpy as np

from sokoban_agent.env import Action, FixedLevelProvider, SokobanEnv, parse_level
from sokoban_agent.planning import analyze_board


def test_board_analysis_extracts_deterministic_push_facts() -> None:
    env = SokobanEnv()
    observation, _ = env.reset(options={"level_id": "tiny-push"})
    env.close()

    first = analyze_board(observation)
    second = analyze_board(observation)

    assert first == second
    assert [box.model_dump(mode="json") for box in first.boxes] == [
        {"box_id": "B1", "position": {"row": 2, "col": 2}}
    ]
    assert [target.model_dump(mode="json") for target in first.targets] == [
        {"target_id": "T1", "position": {"row": 1, "col": 2}}
    ]
    assert [
        distance.model_dump(mode="json")
        for distance in first.reverse_pull_distances
    ] == [{"box_id": "B1", "target_id": "T1", "distance": 1}]
    assert {
        option.direction: option.creates_static_deadlock
        for option in first.push_options
    } == {
        "UP": False,
        "RIGHT": True,
        "DOWN": True,
        "LEFT": True,
    }
    assert {"row": 3, "col": 2} in [
        cell.model_dump(mode="json") for cell in first.dead_squares
    ]


def test_board_analysis_preserves_box_identity_after_one_push() -> None:
    level = parse_level(
        "identity",
        ["#######", "# @   #", "# $ $ #", "# . . #", "#######"],
    )
    env = SokobanEnv(level_provider=FixedLevelProvider([level]))
    observation, _ = env.reset()
    before = analyze_board(observation)

    next_observation, _, _, _, _ = env.step(Action.DOWN)
    after = analyze_board(next_observation, previous=before)
    env.close()

    assert {
        box.box_id: box.position.model_dump(mode="json")
        for box in after.boxes
    } == {
        "B1": {"row": 3, "col": 2},
        "B2": {"row": 2, "col": 4},
    }


def test_board_analysis_rotates_push_direction_with_the_board() -> None:
    env = SokobanEnv()
    observation, _ = env.reset(options={"level_id": "tiny-push"})
    env.close()

    analysis = analyze_board(np.rot90(observation).copy())

    assert [
        option.direction
        for option in analysis.push_options
        if not option.creates_static_deadlock
    ] == ["LEFT"]
    assert analysis.reverse_pull_distances[0].distance == 1


def test_board_analysis_handles_a_narrow_push_corridor() -> None:
    level = parse_level(
        "corridor",
        ["#######", "#. $@ #", "#######"],
    )
    env = SokobanEnv(level_provider=FixedLevelProvider([level]))
    observation, _ = env.reset()
    env.close()

    analysis = analyze_board(observation)

    assert [option.direction for option in analysis.push_options] == ["LEFT"]
    assert not analysis.push_options[0].creates_static_deadlock
    assert analysis.reverse_pull_distances[0].distance == 2

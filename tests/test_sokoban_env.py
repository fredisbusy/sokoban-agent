import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from sokoban_agent.env import (
    ENV_ID,
    Action,
    FixedLevelProvider,
    SokobanEnv,
    Tile,
    parse_level,
)


def make_env(rows: list[str], *, max_steps: int = 120) -> SokobanEnv:
    level = parse_level("test", rows)
    return SokobanEnv(
        level_provider=FixedLevelProvider([level]),
        render_mode="ansi",
        max_steps=max_steps,
    )


def test_registered_environment_passes_gymnasium_checker() -> None:
    env = SokobanEnv()

    check_env(env, skip_render_check=True)


def test_registered_environment_can_be_made() -> None:
    env = gym.make(ENV_ID)

    observation, info = env.reset(seed=7, options={"level_id": "tiny-push"})

    assert observation.shape == (5, 5)
    assert info["level_id"] == "tiny-push"


def test_reset_is_reproducible_and_returns_valid_observation() -> None:
    env = SokobanEnv()

    first, first_info = env.reset(seed=11)
    second, second_info = env.reset(seed=11)

    assert np.array_equal(first, second)
    assert first_info["level_id"] == second_info["level_id"]
    assert env.observation_space.contains(first)


def test_player_walks_and_wall_move_is_invalid() -> None:
    env = make_env(["#####", "# . #", "# $ #", "#@  #", "#####"])
    observation, _ = env.reset()

    observation, reward, terminated, truncated, info = env.step(Action.RIGHT)

    assert observation[3, 2] == Tile.PLAYER
    assert reward == pytest.approx(-0.1)
    assert not terminated
    assert not truncated
    assert not info["invalid_move"]

    observation, _, _, _, info = env.step(Action.DOWN)

    assert observation[3, 2] == Tile.PLAYER
    assert info["invalid_move"]


def test_box_push_completes_level() -> None:
    env = make_env(["#####", "# . #", "# $ #", "# @ #", "#####"])
    env.reset()

    observation, reward, terminated, truncated, info = env.step(Action.UP)

    assert observation[1, 2] == Tile.BOX_ON_TARGET
    assert reward == pytest.approx(10.9)
    assert terminated
    assert not truncated
    assert info["success"]
    assert not info["deadlock"]


def test_box_cannot_be_pushed_through_wall() -> None:
    env = make_env(["######", "# .  #", "# #$@#", "#    #", "######"])
    observation, _ = env.reset()

    observation, _, terminated, _, info = env.step(Action.LEFT)

    assert observation[2, 3] == Tile.BOX
    assert observation[2, 4] == Tile.PLAYER
    assert not terminated
    assert info["invalid_move"]


def test_static_corner_deadlock_terminates_episode() -> None:
    env = make_env(["#####", "## .#", "# $@#", "#   #", "#####"])
    env.reset()

    _, reward, terminated, truncated, info = env.step(Action.LEFT)

    assert reward == pytest.approx(-10.1)
    assert terminated
    assert not truncated
    assert info["deadlock"]
    assert not info["success"]


def test_step_limit_truncates_episode() -> None:
    env = make_env(
        ["#####", "# . #", "# $ #", "#@  #", "#####"],
        max_steps=1,
    )
    env.reset()

    _, _, terminated, truncated, _ = env.step(Action.RIGHT)

    assert not terminated
    assert truncated


def test_rgb_render_has_expected_shape() -> None:
    level = parse_level("render", ["#####", "#@$.#", "#####"])
    env = SokobanEnv(
        level_provider=FixedLevelProvider([level]),
        render_mode="rgb_array",
        tile_size=4,
    )
    env.reset()

    frame = env.render()

    assert isinstance(frame, np.ndarray)
    assert frame.shape == (12, 20, 3)

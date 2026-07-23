"""Deterministic Gymnasium environment for Sokoban."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sokoban_agent.env.levels import (
    DEFAULT_LEVELS,
    LevelProvider,
    Position,
    SokobanLevel,
)


class Action(IntEnum):
    """Four primitive actions shared by every agent."""

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class Tile(IntEnum):
    """Integer encoding returned in board observations."""

    FLOOR = 0
    WALL = 1
    TARGET = 2
    BOX = 3
    PLAYER = 4
    BOX_ON_TARGET = 5
    PLAYER_ON_TARGET = 6


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Dense reward defaults compatible with common Boxoban experiments."""

    step: float = -0.1
    box_on_target: float = 1.0
    box_off_target: float = -1.0
    completion: float = 10.0
    deadlock: float = -10.0


_DEFAULT_REWARD_CONFIG = RewardConfig()

_DIRECTIONS: dict[Action, Position] = {
    Action.UP: (-1, 0),
    Action.RIGHT: (0, 1),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
}

_COLORS = np.asarray(
    [
        [238, 238, 238],
        [45, 45, 45],
        [255, 196, 64],
        [139, 90, 43],
        [55, 126, 184],
        [76, 175, 80],
        [126, 87, 194],
    ],
    dtype=np.uint8,
)
_ANSI_TILES: dict[Tile, str] = {
    Tile.FLOOR: " ",
    Tile.WALL: "#",
    Tile.TARGET: ".",
    Tile.BOX: "$",
    Tile.PLAYER: "@",
    Tile.BOX_ON_TARGET: "*",
    Tile.PLAYER_ON_TARGET: "+",
}


class SokobanEnv(gym.Env[np.ndarray, int]):
    """A small, testable Sokoban environment with Boxoban-compatible levels."""

    metadata = {
        "render_modes": ["ansi", "rgb_array", "human"],
        "render_fps": 4,
    }

    def __init__(
        self,
        *,
        level_provider: LevelProvider = DEFAULT_LEVELS,
        render_mode: str | None = None,
        max_steps: int = 120,
        reward_config: RewardConfig = _DEFAULT_REWARD_CONFIG,
        tile_size: int = 24,
    ) -> None:
        super().__init__()
        if render_mode not in {None, *self.metadata["render_modes"]}:
            raise ValueError(f"unsupported render mode: {render_mode}")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if tile_size <= 0:
            raise ValueError("tile_size must be positive")

        self.level_provider = level_provider
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.reward_config = reward_config
        self.tile_size = tile_size

        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Box(
            low=int(Tile.FLOOR),
            high=int(Tile.PLAYER_ON_TARGET),
            shape=self.level_provider.shape,
            dtype=np.uint8,
        )

        self._level: SokobanLevel | None = None
        self._boxes: set[Position] = set()
        self._player: Position | None = None
        self._steps = 0
        self._episode_done = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset to a requested level or sample one reproducibly."""

        super().reset(seed=seed)
        requested_level_id = (options or {}).get("level_id")
        if requested_level_id is not None and not isinstance(
            requested_level_id, str
        ):
            raise TypeError("options['level_id'] must be a string")

        if requested_level_id is None:
            level = self.level_provider.sample(self.np_random)
        else:
            level = self.level_provider.get(requested_level_id)
        if level.shape != self.level_provider.shape:
            raise ValueError("selected level does not match the observation shape")

        self._level = level
        self._boxes = set(level.boxes)
        self._player = level.player
        self._steps = 0
        self._episode_done = False

        observation = self._observation()
        info = self._info(invalid_move=False, pushed=False)
        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply one primitive move and report win, truncation, or deadlock."""

        level, player = self._require_state()
        if self._episode_done:
            raise RuntimeError("reset() must be called after an episode ends")
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action: {action}")

        selected_action = Action(int(action))
        row_delta, column_delta = _DIRECTIONS[selected_action]
        destination = (player[0] + row_delta, player[1] + column_delta)
        invalid_move = destination in level.walls
        pushed = False
        box_left_target = False
        box_entered_target = False

        if not invalid_move and destination in self._boxes:
            beyond = (
                destination[0] + row_delta,
                destination[1] + column_delta,
            )
            if beyond in level.walls or beyond in self._boxes:
                invalid_move = True
            else:
                pushed = True
                box_left_target = destination in level.targets
                box_entered_target = beyond in level.targets
                self._boxes.remove(destination)
                self._boxes.add(beyond)

        if not invalid_move:
            self._player = destination

        self._steps += 1
        success = self._boxes == set(level.targets)
        deadlock = not success and self._has_static_corner_deadlock()
        terminated = success or deadlock
        truncated = self._steps >= self.max_steps and not terminated
        self._episode_done = terminated or truncated

        reward = self.reward_config.step
        if pushed and box_left_target:
            reward += self.reward_config.box_off_target
        if pushed and box_entered_target:
            reward += self.reward_config.box_on_target
        if success:
            reward += self.reward_config.completion
        elif deadlock:
            reward += self.reward_config.deadlock

        observation = self._observation()
        info = self._info(invalid_move=invalid_move, pushed=pushed)
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    def render(self) -> str | np.ndarray | None:
        """Render the current board in the configured mode."""

        if self.render_mode is None:
            return None
        observation = self._observation()
        if self.render_mode == "rgb_array":
            image = _COLORS[observation]
            return np.repeat(
                np.repeat(image, self.tile_size, axis=0),
                self.tile_size,
                axis=1,
            )

        text = "\n".join(
            "".join(_ANSI_TILES[Tile(int(tile))] for tile in row)
            for row in observation
        )
        if self.render_mode == "human":
            print(text)
            return None
        return text

    def _observation(self) -> np.ndarray:
        level, player = self._require_state()
        board = np.full(level.shape, int(Tile.FLOOR), dtype=np.uint8)
        for row, column in level.walls:
            board[row, column] = int(Tile.WALL)
        for row, column in level.targets:
            board[row, column] = int(Tile.TARGET)
        for row, column in self._boxes:
            board[row, column] = int(
                Tile.BOX_ON_TARGET
                if (row, column) in level.targets
                else Tile.BOX
            )
        board[player] = int(
            Tile.PLAYER_ON_TARGET if player in level.targets else Tile.PLAYER
        )
        return board

    def _info(self, *, invalid_move: bool, pushed: bool) -> dict[str, Any]:
        level, _ = self._require_state()
        boxes_on_targets = len(self._boxes & set(level.targets))
        success = self._boxes == set(level.targets)
        return {
            "level_id": level.level_id,
            "steps": self._steps,
            "invalid_move": invalid_move,
            "pushed": pushed,
            "boxes_on_targets": boxes_on_targets,
            "success": success,
            "deadlock": not success and self._has_static_corner_deadlock(),
        }

    def _has_static_corner_deadlock(self) -> bool:
        level, _ = self._require_state()
        for row, column in self._boxes - set(level.targets):
            up = (row - 1, column) in level.walls
            right = (row, column + 1) in level.walls
            down = (row + 1, column) in level.walls
            left = (row, column - 1) in level.walls
            if (up and right) or (right and down) or (down and left) or (
                left and up
            ):
                return True
        return False

    def _require_state(self) -> tuple[SokobanLevel, Position]:
        if self._level is None or self._player is None:
            raise RuntimeError("reset() must be called before using the environment")
        return self._level, self._player

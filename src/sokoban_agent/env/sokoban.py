"""Deterministic Gymnasium environment for Sokoban."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sokoban_agent.env.catalog import DEFAULT_LEVELS
from sokoban_agent.env.levels import LevelProvider, SokobanLevel
from sokoban_agent.env.model import Action, SokobanState, Tile
from sokoban_agent.env.rules import (
    apply_action,
    has_static_corner_deadlock,
    initial_state,
    is_success,
    observation_for,
)


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Dense reward defaults compatible with common Boxoban experiments."""

    step: float = -0.1
    box_on_target: float = 1.0
    box_off_target: float = -1.0
    completion: float = 10.0
    deadlock: float = -10.0


_DEFAULT_REWARD_CONFIG = RewardConfig()

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
        self._state: SokobanState | None = None
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
        self._state = initial_state(level)
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

        level, state = self._require_state()
        if self._episode_done:
            raise RuntimeError("reset() must be called after an episode ends")
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action: {action}")

        selected_action = Action(int(action))
        move = apply_action(level, state, selected_action)
        self._state = move.state

        self._steps += 1
        success = is_success(level, move.state)
        deadlock = not success and has_static_corner_deadlock(level, move.state)
        terminated = success or deadlock
        truncated = self._steps >= self.max_steps and not terminated
        self._episode_done = terminated or truncated

        reward = self.reward_config.step
        if move.pushed and move.box_left_target:
            reward += self.reward_config.box_off_target
        if move.pushed and move.box_entered_target:
            reward += self.reward_config.box_on_target
        if success:
            reward += self.reward_config.completion
        elif deadlock:
            reward += self.reward_config.deadlock

        observation = self._observation()
        info = self._info(
            invalid_move=move.invalid_move,
            pushed=move.pushed,
        )
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
        level, state = self._require_state()
        return observation_for(level, state)

    def _info(self, *, invalid_move: bool, pushed: bool) -> dict[str, Any]:
        level, state = self._require_state()
        boxes_on_targets = len(state.boxes & level.targets)
        success = is_success(level, state)
        return {
            "level_id": level.level_id,
            "steps": self._steps,
            "invalid_move": invalid_move,
            "pushed": pushed,
            "boxes_on_targets": boxes_on_targets,
            "success": success,
            "deadlock": not success
            and has_static_corner_deadlock(level, state),
        }

    def _require_state(self) -> tuple[SokobanLevel, SokobanState]:
        if self._level is None or self._state is None:
            raise RuntimeError("reset() must be called before using the environment")
        return self._level, self._state

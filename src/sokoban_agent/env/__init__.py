"""Gymnasium-compatible Sokoban environments."""

from gymnasium.envs.registration import register, registry

from sokoban_agent.env.levels import (
    DEFAULT_LEVELS,
    BoxobanLevelProvider,
    FixedLevelProvider,
    LevelProvider,
    SokobanLevel,
    parse_boxoban_text,
    parse_level,
)
from sokoban_agent.env.sokoban import Action, RewardConfig, SokobanEnv, Tile

ENV_ID = "SokobanAgent-v0"


def register_envs() -> None:
    """Register the default environment once with Gymnasium."""

    if ENV_ID not in registry:
        register(
            id=ENV_ID,
            entry_point="sokoban_agent.env.sokoban:SokobanEnv",
        )


register_envs()

__all__ = [
    "DEFAULT_LEVELS",
    "ENV_ID",
    "Action",
    "BoxobanLevelProvider",
    "FixedLevelProvider",
    "LevelProvider",
    "RewardConfig",
    "SokobanEnv",
    "SokobanLevel",
    "Tile",
    "parse_boxoban_text",
    "parse_level",
    "register_envs",
]

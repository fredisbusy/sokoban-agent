"""Gymnasium-compatible Sokoban environments."""

from gymnasium.envs.registration import register, registry

from sokoban_agent.env.catalog import (
    DEFAULT_LEVEL_CATALOG,
    DEFAULT_LEVELS,
    LevelCatalog,
    LevelCatalogRecord,
    level_rows_sha256,
    load_level_catalog,
)
from sokoban_agent.env.levels import (
    BoxobanLevelProvider,
    FixedLevelProvider,
    LevelProvider,
    SokobanLevel,
    parse_boxoban_text,
    parse_level,
)
from sokoban_agent.env.model import Action, SokobanState, Tile
from sokoban_agent.env.sokoban import RewardConfig, SokobanEnv

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
    "DEFAULT_LEVEL_CATALOG",
    "ENV_ID",
    "Action",
    "BoxobanLevelProvider",
    "FixedLevelProvider",
    "LevelProvider",
    "LevelCatalog",
    "LevelCatalogRecord",
    "RewardConfig",
    "SokobanEnv",
    "SokobanLevel",
    "SokobanState",
    "Tile",
    "parse_boxoban_text",
    "parse_level",
    "level_rows_sha256",
    "load_level_catalog",
    "register_envs",
]

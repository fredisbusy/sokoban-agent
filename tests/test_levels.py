from pathlib import Path

import numpy as np
import pytest

from sokoban_agent.env import (
    BoxobanLevelProvider,
    FixedLevelProvider,
    parse_boxoban_text,
    parse_level,
)


def test_parse_boxoban_symbols_and_ids() -> None:
    levels = parse_boxoban_text(
        """; 7
#####
#+*$#
#####
""",
        source="000.txt",
    )

    assert len(levels) == 1
    level = levels[0]
    assert level.level_id == "000.txt:7"
    assert level.player == (1, 1)
    assert level.boxes == {(1, 2), (1, 3)}
    assert level.targets == {(1, 1), (1, 2)}


def test_level_requires_matching_boxes_and_targets() -> None:
    with pytest.raises(ValueError, match="same number"):
        parse_level("broken", ["#####", "#@$ #", "#####"])


def test_level_must_be_enclosed_by_walls() -> None:
    with pytest.raises(ValueError, match="enclosed"):
        parse_level("open", ["#####", "@ $.#", "#####"])


def test_fixed_provider_samples_reproducibly() -> None:
    levels = [
        parse_level("one", ["#####", "#@$.#", "#####"]),
        parse_level("two", ["#####", "#@$.#", "#####"]),
    ]
    provider = FixedLevelProvider(levels)

    first = provider.sample(np.random.default_rng(42))
    second = provider.sample(np.random.default_rng(42))

    assert first.level_id == second.level_id


def test_boxoban_provider_loads_file_lazily() -> None:
    sample = (
        Path(__file__).parents[1] / "assets" / "levels" / "boxoban_sample.txt"
    )
    provider = BoxobanLevelProvider(sample)

    assert provider.shape == (5, 5)
    assert provider.file_count == 1
    assert provider.level_count == 2
    assert provider.get("boxoban_sample.txt:sample-push").player == (3, 2)

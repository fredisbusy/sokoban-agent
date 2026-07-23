from pathlib import Path
from typing import cast

import numpy as np
import pytest

from sokoban_agent.env import (
    BoxobanLevelProvider,
    FixedLevelProvider,
    SokobanLevel,
    parse_boxoban_text,
    parse_level,
)


class _IndexRng:
    def __init__(self, index: int) -> None:
        self.index = index

    def integers(self, high: int) -> int:
        assert 0 <= self.index < high
        return self.index


def _rng_at(index: int) -> np.random.Generator:
    return cast(np.random.Generator, _IndexRng(index))


@pytest.fixture
def multi_file_boxoban(tmp_path: Path) -> Path:
    first = tmp_path / "a" / "000.txt"
    first.parent.mkdir()
    first.write_text(
        """; a0
#####
#@$.#
#   #
#   #
#####
; a1
#####
# . #
# $ #
#@  #
#####
""",
        encoding="utf-8",
    )
    second = tmp_path / "b" / "001.txt"
    second.parent.mkdir()
    second.write_text(
        """; b0
#####
#  .#
# $ #
# @ #
#####
; b1
#####
# . #
#  $#
# @ #
#####
""",
        encoding="utf-8",
    )
    return tmp_path


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


def test_boxoban_provider_maps_indices_across_file_boundaries(
    multi_file_boxoban: Path,
) -> None:
    provider = BoxobanLevelProvider(multi_file_boxoban)

    assert provider.file_count == 2
    assert provider.level_count == 4
    assert [
        provider.sample(_rng_at(index)).level_id for index in range(4)
    ] == [
        "a/000.txt:a0",
        "a/000.txt:a1",
        "b/001.txt:b0",
        "b/001.txt:b1",
    ]


def test_boxoban_provider_switches_single_file_cache(
    multi_file_boxoban: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = BoxobanLevelProvider(multi_file_boxoban)
    loaded_sources: list[str] = []
    original_load_file = provider._load_file

    def tracked_load_file(source: str) -> list[SokobanLevel]:
        loaded_sources.append(source)
        return original_load_file(source)

    monkeypatch.setattr(provider, "_load_file", tracked_load_file)

    assert provider.get("b/001.txt:b0").level_id == "b/001.txt:b0"
    assert provider.get("b/001.txt:b1").level_id == "b/001.txt:b1"
    assert loaded_sources == ["b/001.txt"]

    assert provider.get("a/000.txt:a0").level_id == "a/000.txt:a0"
    assert provider.get("a/000.txt:a1").level_id == "a/000.txt:a1"
    assert loaded_sources == ["b/001.txt", "a/000.txt"]


def test_boxoban_provider_rejects_later_shape_without_corrupting_cache(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.txt").write_text(
        """; valid
#####
#@$.#
#   #
#   #
#####
""",
        encoding="utf-8",
    )
    (tmp_path / "b.txt").write_text(
        """; wrong-shape
#####
#@$.#
#   #
#####
""",
        encoding="utf-8",
    )
    provider = BoxobanLevelProvider(tmp_path)

    with pytest.raises(ValueError, match="share a shape"):
        provider.get("b.txt:wrong-shape")

    assert provider.get("a.txt:valid").level_id == "a.txt:valid"

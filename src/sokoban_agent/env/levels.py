"""Level definitions and Boxoban-compatible level providers."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

Position = tuple[int, int]

_WALL = "#"
_FLOOR = " "
_TARGET = "."
_BOX = "$"
_PLAYER = "@"
_BOX_ON_TARGET = "*"
_PLAYER_ON_TARGET = "+"
_VALID_TILES = {
    _WALL,
    _FLOOR,
    _TARGET,
    _BOX,
    _PLAYER,
    _BOX_ON_TARGET,
    _PLAYER_ON_TARGET,
}


@dataclass(frozen=True, slots=True)
class SokobanLevel:
    """Immutable starting state for one Sokoban level."""

    level_id: str
    height: int
    width: int
    walls: frozenset[Position]
    targets: frozenset[Position]
    boxes: frozenset[Position]
    player: Position

    @property
    def shape(self) -> tuple[int, int]:
        """Return the board shape used by the observation space."""

        return self.height, self.width


class LevelProvider(Protocol):
    """Source of same-sized Sokoban levels."""

    @property
    def shape(self) -> tuple[int, int]:
        """Return the shape shared by every level."""
        ...

    def get(self, level_id: str) -> SokobanLevel:
        """Return a level by its stable identifier."""
        ...

    def sample(self, rng: np.random.Generator) -> SokobanLevel:
        """Sample a level using the environment RNG."""
        ...


def parse_level(level_id: str, rows: Sequence[str]) -> SokobanLevel:
    """Parse a rectangular Sokoban board using standard Boxoban symbols."""

    if not rows:
        raise ValueError(f"level {level_id!r} has no board rows")

    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError(f"level {level_id!r} must be a non-empty rectangle")

    walls: set[Position] = set()
    targets: set[Position] = set()
    boxes: set[Position] = set()
    players: list[Position] = []

    for row_index, row in enumerate(rows):
        for column_index, tile in enumerate(row):
            if tile not in _VALID_TILES:
                raise ValueError(
                    f"level {level_id!r} contains unsupported tile {tile!r}"
                )
            position = (row_index, column_index)
            if tile == _WALL:
                walls.add(position)
            if tile in {_TARGET, _BOX_ON_TARGET, _PLAYER_ON_TARGET}:
                targets.add(position)
            if tile in {_BOX, _BOX_ON_TARGET}:
                boxes.add(position)
            if tile in {_PLAYER, _PLAYER_ON_TARGET}:
                players.append(position)

    if len(players) != 1:
        raise ValueError(f"level {level_id!r} must contain exactly one player")
    if not boxes:
        raise ValueError(f"level {level_id!r} must contain at least one box")
    if len(boxes) != len(targets):
        raise ValueError(
            f"level {level_id!r} must contain the same number of boxes and targets"
        )

    boundary = {
        *((0, column) for column in range(width)),
        *((len(rows) - 1, column) for column in range(width)),
        *((row, 0) for row in range(len(rows))),
        *((row, width - 1) for row in range(len(rows))),
    }
    if not boundary <= walls:
        raise ValueError(f"level {level_id!r} must be enclosed by walls")

    return SokobanLevel(
        level_id=level_id,
        height=len(rows),
        width=width,
        walls=frozenset(walls),
        targets=frozenset(targets),
        boxes=frozenset(boxes),
        player=players[0],
    )


def parse_boxoban_text(text: str, *, source: str = "levels.txt") -> list[SokobanLevel]:
    """Parse all ``; level_id`` blocks from a Boxoban text file."""

    levels: list[SokobanLevel] = []
    current_id: str | None = None
    current_rows: list[str] = []

    def finish_level() -> None:
        nonlocal current_id, current_rows
        if current_id is None:
            return
        levels.append(parse_level(f"{source}:{current_id}", current_rows))
        current_id = None
        current_rows = []

    for raw_line in text.splitlines():
        if raw_line.startswith(";"):
            finish_level()
            current_id = raw_line.removeprefix(";").strip()
            if not current_id:
                raise ValueError(f"{source!r} contains an empty level id")
        elif current_id is not None and raw_line:
            current_rows.append(raw_line)
        elif raw_line.strip():
            raise ValueError(f"{source!r} has board content before a level id")

    finish_level()
    if not levels:
        raise ValueError(f"{source!r} contains no Boxoban levels")
    return levels


class FixedLevelProvider:
    """In-memory provider for tests and small reproducible experiments."""

    def __init__(self, levels: Sequence[SokobanLevel]) -> None:
        if not levels:
            raise ValueError("at least one level is required")
        shape = levels[0].shape
        if any(level.shape != shape for level in levels):
            raise ValueError("all levels in one provider must have the same shape")
        if len({level.level_id for level in levels}) != len(levels):
            raise ValueError("level ids must be unique")

        self._levels = {level.level_id: level for level in levels}
        self._level_ids = tuple(self._levels)
        self._shape = shape

    @property
    def shape(self) -> tuple[int, int]:
        """Return the common board shape."""

        return self._shape

    def get(self, level_id: str) -> SokobanLevel:
        """Return one fixed level."""

        try:
            return self._levels[level_id]
        except KeyError as error:
            raise KeyError(f"unknown level id: {level_id}") from error

    def sample(self, rng: np.random.Generator) -> SokobanLevel:
        """Sample one fixed level reproducibly."""

        index = int(rng.integers(len(self._level_ids)))
        return self._levels[self._level_ids[index]]


class BoxobanLevelProvider:
    """Lazy file-backed provider for an official Boxoban split or text file.

    Only the selected file is parsed and cached. This avoids materializing the
    full Boxoban training set in memory.
    """

    def __init__(self, path: str | Path) -> None:
        root = Path(path)
        if root.is_file():
            files = [root]
            source_names = [root.name]
        elif root.is_dir():
            files = sorted(root.rglob("*.txt"))
            source_names = [file.relative_to(root).as_posix() for file in files]
        else:
            raise FileNotFoundError(f"Boxoban path does not exist: {root}")
        if not files:
            raise ValueError(f"no Boxoban .txt files found in {root}")

        self._files = dict(zip(source_names, files, strict=True))
        self._counts: list[int] = []
        self._cumulative_counts: list[int] = []
        total = 0
        for file in files:
            count = sum(
                line.startswith(";")
                for line in file.read_text(encoding="utf-8").splitlines()
            )
            if count == 0:
                raise ValueError(f"{file} contains no Boxoban level headers")
            self._counts.append(count)
            total += count
            self._cumulative_counts.append(total)

        first_source = source_names[0]
        first_level = self._load_file(first_source)[0]
        self._shape = first_level.shape
        self._cached_source = first_source
        self._cached_levels = self._index_levels(
            self._load_file(first_source),
            expected_shape=self._shape,
        )

    @property
    def shape(self) -> tuple[int, int]:
        """Return the common Boxoban board shape."""

        return self._shape

    @property
    def level_count(self) -> int:
        """Return the number of indexed level headers."""

        return self._cumulative_counts[-1]

    @property
    def file_count(self) -> int:
        """Return the number of Boxoban text files."""

        return len(self._files)

    def get(self, level_id: str) -> SokobanLevel:
        """Load a level identified as ``relative/file.txt:level_header``."""

        try:
            source, _ = level_id.rsplit(":", maxsplit=1)
        except ValueError as error:
            raise KeyError(
                "Boxoban level id must look like '000.txt:0'"
            ) from error
        levels = self._levels_for_source(source)
        try:
            return levels[level_id]
        except KeyError as error:
            raise KeyError(f"unknown Boxoban level id: {level_id}") from error

    def sample(self, rng: np.random.Generator) -> SokobanLevel:
        """Sample uniformly over indexed Boxoban levels."""

        global_index = int(rng.integers(self.level_count))
        file_index = bisect_right(self._cumulative_counts, global_index)
        previous_total = (
            self._cumulative_counts[file_index - 1] if file_index > 0 else 0
        )
        source = tuple(self._files)[file_index]
        levels = tuple(self._levels_for_source(source).values())
        return levels[global_index - previous_total]

    def _levels_for_source(self, source: str) -> Mapping[str, SokobanLevel]:
        if source not in self._files:
            raise KeyError(f"unknown Boxoban source file: {source}")
        if source != self._cached_source:
            self._cached_levels = self._index_levels(
                self._load_file(source),
                expected_shape=self._shape,
            )
            self._cached_source = source
        return self._cached_levels

    def _load_file(self, source: str) -> list[SokobanLevel]:
        file = self._files[source]
        return parse_boxoban_text(file.read_text(encoding="utf-8"), source=source)

    @staticmethod
    def _index_levels(
        levels: Sequence[SokobanLevel],
        *,
        expected_shape: tuple[int, int],
    ) -> dict[str, SokobanLevel]:
        if any(level.shape != expected_shape for level in levels):
            raise ValueError("all levels in one Boxoban provider must share a shape")
        indexed = {level.level_id: level for level in levels}
        if len(indexed) != len(levels):
            raise ValueError("Boxoban level ids must be unique within a file")
        return indexed

"""Shared level catalog consumed by Python runtimes and the web viewer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from importlib.resources import files
from pathlib import Path
from typing import Any

from sokoban_agent.env.levels import (
    FixedLevelProvider,
    SokobanLevel,
    parse_level,
)


@dataclass(frozen=True, slots=True)
class LevelCatalogRecord:
    """One normalized Boxoban or project-authored level."""

    level_id: str
    title: str
    difficulty: str
    rows: tuple[str, ...]
    sha256: str
    recommended_max_steps: int
    source_type: str
    source: dict[str, object]
    tags: dict[str, object]
    reference: dict[str, object] | None

    def to_level(self) -> SokobanLevel:
        """Return the validated environment representation."""

        return parse_level(self.level_id, self.rows)


class LevelCatalog:
    """Validated lookup over the generated, version-controlled catalog."""

    def __init__(self, records: tuple[LevelCatalogRecord, ...]) -> None:
        if not records:
            raise ValueError("level catalog cannot be empty")
        self._records = records
        self._by_id = {record.level_id: record for record in records}
        if len(self._by_id) != len(records):
            raise ValueError("level catalog IDs must be unique")

    @property
    def records(self) -> tuple[LevelCatalogRecord, ...]:
        """Return catalog records in stable display order."""

        return self._records

    def get(self, level_id: str) -> LevelCatalogRecord:
        """Resolve a stable level ID."""

        try:
            return self._by_id[level_id]
        except KeyError as error:
            raise KeyError(f"unknown catalog level id: {level_id}") from error

    def provider(self, *level_ids: str) -> FixedLevelProvider:
        """Build a provider for a same-sized subset."""

        if not level_ids:
            raise ValueError("catalog provider requires at least one level ID")
        return FixedLevelProvider(
            [self.get(level_id).to_level() for level_id in level_ids]
        )


def level_rows_sha256(rows: list[str] | tuple[str, ...]) -> str:
    """Hash the exact JSON-safe board rows used by every runtime."""

    canonical = json.dumps(
        list(rows),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return sha256(canonical.encode()).hexdigest()


def load_level_catalog(path: str | Path | None = None) -> LevelCatalog:
    """Load and validate the generated catalog."""

    if path is None:
        resource = files("sokoban_agent").joinpath("data/level_catalog.json")
        raw = json.loads(resource.read_text(encoding="utf-8"))
    else:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    payload = _object(raw, "level catalog")
    if _integer(payload, "schema_version") != 1:
        raise ValueError("unsupported level catalog schema version")
    values = payload.get("levels")
    if not isinstance(values, list) or not values:
        raise ValueError("level catalog requires levels")
    records = tuple(_record(value) for value in values)
    digests = [record.sha256 for record in records]
    if len(set(digests)) != len(digests):
        raise ValueError("level catalog contains duplicate boards")
    return LevelCatalog(records)


def _record(value: object) -> LevelCatalogRecord:
    payload = _object(value, "level record")
    rows = _string_list(payload, "rows")
    level_id = _text(payload, "id")
    level = parse_level(level_id, rows)
    digest = _text(payload, "sha256")
    if digest != level_rows_sha256(rows):
        raise ValueError(f"{level_id} board checksum drifted")
    source = _object(payload.get("source"), f"{level_id} source")
    recommended_max_steps = _integer(payload, "recommended_max_steps")
    if recommended_max_steps < 1:
        raise ValueError(f"{level_id} max steps must be positive")
    return LevelCatalogRecord(
        level_id=level.level_id,
        title=_text(payload, "title"),
        difficulty=_text(payload, "difficulty"),
        rows=tuple(rows),
        sha256=digest,
        recommended_max_steps=recommended_max_steps,
        source_type=_text(source, "type"),
        source=dict(source),
        tags=dict(_object(payload.get("tags", {}), f"{level_id} tags")),
        reference=_optional_object(payload.get("reference"), level_id),
    )


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _optional_object(
    value: object,
    level_id: str,
) -> dict[str, object] | None:
    if value is None:
        return None
    return dict(_object(value, f"{level_id} reference"))


def _text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be non-empty text")
    return value


def _integer(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def _string_list(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        raise ValueError(f"{key} must contain strings")
    return value


DEFAULT_LEVEL_CATALOG = load_level_catalog()
DEFAULT_LEVELS = DEFAULT_LEVEL_CATALOG.provider("tiny-push", "tiny-walk")

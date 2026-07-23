"""Structurally tagged held-out manifests for agentic evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from sokoban_agent.env import SokobanLevel, parse_level
from sokoban_agent.graph.agentic_state import AgenticInput


@dataclass(frozen=True, slots=True)
class AgenticLevelCase:
    """One inline held-out level and its structural tags."""

    level_id: str
    rows: tuple[str, ...]
    height: int
    width: int
    box_count: int
    layout_family: str
    corridor_structure: str
    trap_types: tuple[str, ...]
    oracle_expectation: str

    def to_level(self) -> SokobanLevel:
        """Parse the exact level rows represented by this case."""

        return parse_level(self.level_id, self.rows)

    def graph_input(self, *, seed: int, max_steps: int) -> AgenticInput:
        """Return the JSON-safe input for the shared structured graph."""

        return {
            "level_id": self.level_id,
            "level_rows": list(self.rows),
            "seed": seed,
            "max_steps": max_steps,
        }


@dataclass(frozen=True, slots=True)
class AgenticCohortManifest:
    """Validated identity and cases for one held-out split."""

    schema_version: int
    version: str
    split: str
    description: str
    development_level_ids: tuple[str, ...]
    cohort_sha256: str
    levels: tuple[AgenticLevelCase, ...]


def load_agentic_cohort_manifest(
    manifest_path: str | Path,
) -> AgenticCohortManifest:
    """Load inline levels and reject metadata, split, or checksum drift."""

    raw = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("agentic cohort manifest must be an object")
    level_payloads = raw.get("levels")
    if not isinstance(level_payloads, list) or not level_payloads:
        raise ValueError("agentic cohort manifest requires levels")
    development_ids = _string_list(raw, "development_level_ids")
    levels = tuple(_level_case(payload) for payload in level_payloads)
    level_ids = [case.level_id for case in levels]
    if len(set(level_ids)) != len(level_ids):
        raise ValueError("agentic cohort contains duplicate level IDs")
    if set(level_ids) & set(development_ids):
        raise ValueError("test levels overlap development level IDs")

    expected_digest = _text(raw, "cohort_sha256")
    actual_digest = _cohort_digest(level_payloads)
    if expected_digest != actual_digest:
        raise ValueError(
            f"agentic cohort checksum mismatch: {actual_digest}"
        )
    return AgenticCohortManifest(
        schema_version=_integer(raw, "schema_version"),
        version=_text(raw, "version"),
        split=_text(raw, "split"),
        description=_text(raw, "description"),
        development_level_ids=tuple(development_ids),
        cohort_sha256=expected_digest,
        levels=levels,
    )


def _level_case(payload: object) -> AgenticLevelCase:
    if not isinstance(payload, dict):
        raise ValueError("agentic cohort levels must be objects")
    rows = _string_list(payload, "rows")
    trap_types = _string_list(payload, "trap_types")
    case = AgenticLevelCase(
        level_id=_text(payload, "level_id"),
        rows=tuple(rows),
        height=_integer(payload, "height"),
        width=_integer(payload, "width"),
        box_count=_integer(payload, "box_count"),
        layout_family=_text(payload, "layout_family"),
        corridor_structure=_text(payload, "corridor_structure"),
        trap_types=tuple(trap_types),
        oracle_expectation=_text(payload, "oracle_expectation"),
    )
    level = case.to_level()
    if level.shape != (case.height, case.width):
        raise ValueError(f"{case.level_id} board dimensions drifted")
    if len(level.boxes) != case.box_count:
        raise ValueError(f"{case.level_id} box count drifted")
    return case


def _cohort_digest(level_payloads: list[object]) -> str:
    canonical = json.dumps(
        level_payloads,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return sha256(canonical).hexdigest()


def _text(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"agentic cohort {key} must be text")
    return value


def _integer(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"agentic cohort {key} must be positive")
    return value


def _string_list(payload: dict[str, object], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        raise ValueError(f"agentic cohort {key} must contain strings")
    return value

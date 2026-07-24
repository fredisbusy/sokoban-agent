"""Build the shared runtime catalog from Boxoban and custom level sources."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
from typing import Any

BOXOBAN_MANIFEST = Path("benchmarks/boxoban_research_v1.json")
CUSTOM_ROOT = Path("levels/custom")
OUTPUT = Path("src/sokoban_agent/data/level_catalog.json")


def main() -> None:
    """Validate source records and write one catalog for Python and Viewer."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when the generated catalog is stale",
    )
    args = parser.parse_args()
    boxoban = _object(
        json.loads(BOXOBAN_MANIFEST.read_text(encoding="utf-8")),
        "Boxoban manifest",
    )
    levels = [_custom_record(path) for path in sorted(CUSTOM_ROOT.glob("*.json"))]
    levels.extend(_boxoban_records(boxoban))
    _validate_catalog(levels)
    payload = {
        "schema_version": 1,
        "sources": {
            "boxoban_manifest": BOXOBAN_MANIFEST.as_posix(),
            "custom_root": CUSTOM_ROOT.as_posix(),
        },
        "levels": levels,
    }
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != content:
            raise SystemExit(
                f"{OUTPUT} is stale; run scripts/build_level_catalog.py"
            )
        print(f"{OUTPUT}: verified {len(levels)} levels")
    else:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(content, encoding="utf-8")
        print(f"{OUTPUT}: wrote {len(levels)} validated levels")


def _custom_record(path: Path) -> dict[str, object]:
    payload = _object(
        json.loads(path.read_text(encoding="utf-8")),
        path.as_posix(),
    )
    if _positive_int(payload, "schema_version") != 1:
        raise ValueError(f"{path} has an unsupported schema version")
    level_id = _text(payload, "id")
    rows = _rows(payload, "rows")
    return {
        "id": level_id,
        "title": _text(payload, "title", default=level_id),
        "difficulty": _text(payload, "difficulty", default="custom"),
        "rows": rows,
        "sha256": _board_digest(rows),
        "recommended_max_steps": _positive_int(
            payload,
            "recommended_max_steps",
            default=120,
        ),
        "reference": payload.get("reference"),
        "tags": _object(payload.get("tags", {}), f"{path} tags"),
        "source": {
            "type": "custom",
            "path": path.as_posix(),
            "revision": _positive_int(payload, "revision", default=1),
            "status": _text(payload, "status", default="draft"),
        },
    }


def _boxoban_records(manifest: dict[str, Any]) -> list[dict[str, object]]:
    source = _object(manifest.get("source"), "Boxoban source")
    commit = _text(source, "commit")
    license_name = _text(source, "license")
    records = []
    for value in _list(manifest, "levels"):
        payload = _object(value, "Boxoban level")
        level_id = _text(payload, "level_id")
        rows = _rows(payload, "rows")
        difficulty = _text(payload, "difficulty")
        reference = _object(payload.get("reference"), f"{level_id} reference")
        action_count = _positive_int(reference, "action_count")
        records.append(
            {
                "id": level_id,
                "title": level_id,
                "difficulty": difficulty,
                "rows": rows,
                "sha256": _board_digest(rows),
                "recommended_max_steps": max(120, action_count * 3),
                "reference": reference,
                "tags": {
                    "layout_family": _text(payload, "layout_family"),
                    "corridor_structure": _text(
                        payload,
                        "corridor_structure",
                    ),
                    "trap_types": _string_list(payload, "trap_types"),
                },
                "source": {
                    "type": "boxoban",
                    "source_level_id": _text(payload, "source_level_id"),
                    "commit": commit,
                    "license": license_name,
                },
            }
        )
    return records


def _validate_catalog(levels: list[dict[str, object]]) -> None:
    if not levels:
        raise ValueError("level catalog cannot be empty")
    ids: set[str] = set()
    digests: set[str] = set()
    for record in levels:
        level_id = _text(record, "id")
        rows = _rows(record, "rows")
        _validate_rows(level_id, rows)
        digest = _text(record, "sha256")
        if digest != _board_digest(rows):
            raise ValueError(f"{level_id} board checksum drifted")
        if level_id in ids:
            raise ValueError(f"duplicate level id: {level_id}")
        if digest in digests:
            raise ValueError(f"duplicate board: {level_id}")
        ids.add(level_id)
        digests.add(digest)


def _validate_rows(level_id: str, rows: list[str]) -> None:
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError(f"{level_id} must be a non-empty rectangle")
    valid_tiles = frozenset("# .$@*+")
    if any(tile not in valid_tiles for row in rows for tile in row):
        raise ValueError(f"{level_id} contains an unsupported tile")
    players = sum(row.count("@") + row.count("+") for row in rows)
    boxes = sum(row.count("$") + row.count("*") for row in rows)
    targets = sum(row.count(".") + row.count("*") + row.count("+") for row in rows)
    if players != 1 or boxes < 1 or boxes != targets:
        raise ValueError(f"{level_id} has invalid player, box, or target counts")
    if (
        set(rows[0]) != {"#"}
        or set(rows[-1]) != {"#"}
        or any(row[0] != "#" or row[-1] != "#" for row in rows)
    ):
        raise ValueError(f"{level_id} must be enclosed by walls")


def _board_digest(rows: list[str]) -> str:
    canonical = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return sha256(canonical.encode()).hexdigest()


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _list(payload: dict[str, Any], key: str) -> list[object]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return value


def _rows(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(
        isinstance(row, str) for row in value
    ):
        raise ValueError(f"{key} must contain strings")
    return value


def _string_list(payload: dict[str, Any], key: str) -> list[str]:
    return _rows(payload, key)


def _text(
    payload: dict[str, Any],
    key: str,
    *,
    default: str | None = None,
) -> str:
    value = payload.get(key, default)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be non-empty text")
    return value


def _positive_int(
    payload: dict[str, Any],
    key: str,
    *,
    default: int | None = None,
) -> int:
    value = payload.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{key} must be a positive integer")
    return value


if __name__ == "__main__":
    main()

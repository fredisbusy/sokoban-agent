"""Download and verify the curated official-difficulty Boxoban cohort."""

from __future__ import annotations

import argparse
import json
import ssl
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import certifi

from sokoban_agent.env import SokobanLevel, parse_boxoban_text

DEFAULT_MANIFEST = Path("benchmarks/boxoban_research_v1.json")
DEFAULT_DATA_ROOT = Path("data/boxoban")


def main() -> None:
    """Prepare only the three pinned source files used by the research set."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--download",
        action="store_true",
        help="download missing pinned source files",
    )
    args = parser.parse_args()
    manifest = _object(
        json.loads(args.manifest.read_text(encoding="utf-8")),
        "manifest",
    )
    source = _object(manifest.get("source"), "source")
    repository = _text(source, "repository")
    commit = _text(source, "commit")
    files = _objects(source.get("files"), "source files")

    parsed_by_difficulty: dict[str, list[SokobanLevel]] = {}
    for file_payload in files:
        difficulty = _text(file_payload, "difficulty")
        source_path = _text(file_payload, "path")
        expected_digest = _text(file_payload, "sha256")
        destination = args.data_root / source_path
        if not destination.is_file():
            if not args.download:
                raise SystemExit(
                    f"{destination} is missing. Re-run with --download."
                )
            _download(
                _raw_url(repository, commit, source_path),
                destination,
            )
        actual_digest = sha256(destination.read_bytes()).hexdigest()
        if actual_digest != expected_digest:
            raise SystemExit(
                f"{source_path} checksum mismatch: "
                f"expected {expected_digest}, got {actual_digest}"
            )
        parsed_by_difficulty[difficulty] = parse_boxoban_text(
            destination.read_text(encoding="utf-8"),
            source=source_path,
        )

    levels = _objects(manifest.get("levels"), "levels")
    _verify_selection(_text(manifest, "version"), levels, parsed_by_difficulty)
    print(
        f"{_text(manifest, 'version')}: verified {len(levels)} levels "
        f"from {len(files)} files at {commit[:8]}"
    )


def _verify_selection(
    version: str,
    manifest_levels: list[dict[str, Any]],
    parsed_by_difficulty: dict[str, list[SokobanLevel]],
) -> None:
    for difficulty, parsed in parsed_by_difficulty.items():
        expected = sorted(
            parsed,
            key=lambda level: sha256(
                f"{version}:{difficulty}:{level.level_id}".encode()
            ).hexdigest(),
        )[:5]
        selected = [
            payload
            for payload in manifest_levels
            if _text(payload, "difficulty") == difficulty
        ]
        selected_ids = [_text(payload, "source_level_id") for payload in selected]
        expected_ids = [level.level_id for level in expected]
        if selected_ids != expected_ids:
            raise ValueError(
                f"{difficulty} deterministic selection drifted: {expected_ids}"
            )
        by_id = {level.level_id: level for level in expected}
        for payload in selected:
            source_level_id = _text(payload, "source_level_id")
            rows = payload.get("rows")
            if rows != _level_rows(by_id[source_level_id]):
                raise ValueError(f"{source_level_id} board rows drifted")


def _level_rows(level: SokobanLevel) -> list[str]:
    rows = []
    for row in range(level.height):
        tiles = []
        for column in range(level.width):
            position = (row, column)
            tile = "#" if position in level.walls else " "
            if position in level.targets:
                tile = "."
            if position in level.boxes:
                tile = "*" if position in level.targets else "$"
            if position == level.player:
                tile = "+" if position in level.targets else "@"
            tiles.append(tile)
        rows.append("".join(tiles))
    return rows


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "sokoban-agent"})
    context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(request, timeout=60, context=context) as response:
        content = response.read()
    temporary = destination.with_suffix(f"{destination.suffix}.download")
    temporary.write_bytes(content)
    temporary.replace(destination)


def _raw_url(repository: str, commit: str, source_path: str) -> str:
    prefix = "https://github.com/"
    if not repository.startswith(prefix):
        raise ValueError("Boxoban repository must be hosted on GitHub")
    slug = repository.removeprefix(prefix).removesuffix(".git")
    return (
        f"https://raw.githubusercontent.com/{slug}/{commit}/{source_path}"
    )


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _objects(value: object, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return [_object(item, label) for item in value]


def _text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be non-empty text")
    return value


if __name__ == "__main__":
    main()

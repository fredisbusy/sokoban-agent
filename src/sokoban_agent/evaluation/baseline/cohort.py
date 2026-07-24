"""Reproducible Boxoban cohort manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from sokoban_agent.env import BoxobanLevelProvider, SokobanLevel


@dataclass(frozen=True, slots=True)
class CohortManifest:
    """Validated external dataset identity and selected level IDs."""

    version: str
    repository: str
    commit: str
    license: str
    split_path: str
    source_file: str
    source_sha256: str
    level_ids: tuple[str, ...]


def load_cohort_manifest(
    manifest_path: str | Path,
    data_root: str | Path,
) -> tuple[CohortManifest, BoxobanLevelProvider]:
    """Load a manifest and reject drift in its external source file."""

    raw = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("cohort manifest must be a JSON object")
    level_ids = raw.get("level_ids")
    if not isinstance(level_ids, list) or not all(
        isinstance(level_id, str) for level_id in level_ids
    ):
        raise ValueError("cohort manifest level_ids must be strings")
    if len(level_ids) < 30:
        raise ValueError("cohort manifest must contain at least 30 levels")
    if len(set(level_ids)) != len(level_ids):
        raise ValueError("cohort manifest contains duplicate level IDs")

    manifest = CohortManifest(
        version=_text(raw, "version"),
        repository=_text(raw, "repository"),
        commit=_text(raw, "commit"),
        license=_text(raw, "license"),
        split_path=_text(raw, "split_path"),
        source_file=_text(raw, "source_file"),
        source_sha256=_text(raw, "source_sha256"),
        level_ids=tuple(level_ids),
    )
    split = Path(data_root) / manifest.split_path
    source = split / manifest.source_file
    if not source.is_file():
        raise FileNotFoundError(f"Boxoban source file is missing: {source}")
    digest = sha256(source.read_bytes()).hexdigest()
    if digest != manifest.source_sha256:
        raise ValueError(
            f"Boxoban checksum mismatch: expected {manifest.source_sha256}, "
            f"got {digest}"
        )

    provider = BoxobanLevelProvider(split)
    board_hashes: set[str] = set()
    for level_id in manifest.level_ids:
        level = provider.get(level_id)
        board_key = _level_hash(level)
        if board_key in board_hashes:
            raise ValueError("cohort manifest contains duplicate level boards")
        board_hashes.add(board_key)
    return manifest, provider


def _text(raw: dict[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"cohort manifest {key} must be a non-empty string")
    return value


def _level_hash(level: SokobanLevel) -> str:
    content = (
        level.shape,
        tuple(sorted(level.walls)),
        tuple(sorted(level.targets)),
        tuple(sorted(level.boxes)),
        level.player,
    )
    return sha256(repr(content).encode()).hexdigest()

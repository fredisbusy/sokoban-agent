"""Prepare and verify the pinned external Boxoban pilot dataset."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from sokoban_agent.evaluation import load_cohort_manifest

DEFAULT_MANIFEST = Path("benchmarks/boxoban_pilot_v1.json")
DEFAULT_DATA_ROOT = Path("data/boxoban")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--download",
        action="store_true",
        help="clone the pinned external dataset when it is missing",
    )
    args = parser.parse_args()
    raw = json.loads(args.manifest.read_text(encoding="utf-8"))
    repository = _required_text(raw, "repository")
    commit = _required_text(raw, "commit")

    if not (args.data_root / ".git").is_dir():
        if not args.download:
            raise SystemExit(
                "Boxoban data is missing. Re-run with --download."
            )
        args.data_root.parent.mkdir(parents=True, exist_ok=True)
        _run(
            "git",
            "clone",
            "--filter=blob:none",
            repository,
            str(args.data_root),
        )
        _run("git", "-C", str(args.data_root), "checkout", commit)

    actual_commit = subprocess.check_output(
        ["git", "-C", str(args.data_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if actual_commit != commit:
        raise SystemExit(
            f"Boxoban commit mismatch: expected {commit}, got {actual_commit}"
        )

    manifest, provider = load_cohort_manifest(
        args.manifest,
        args.data_root,
    )
    print(
        f"{manifest.version}: verified {len(manifest.level_ids)} levels "
        f"from {provider.level_count} source levels at {actual_commit[:8]}"
    )


def _required_text(raw: object, key: str) -> str:
    if not isinstance(raw, dict):
        raise ValueError("manifest must be a JSON object")
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"manifest {key} must be a non-empty string")
    return value


def _run(*command: str) -> None:
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

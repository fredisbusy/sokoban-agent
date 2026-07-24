import json
from pathlib import Path

import pytest
from langgraph.runtime import Runtime

from sokoban_agent.env import (
    DEFAULT_LEVEL_CATALOG,
    DEFAULT_LEVELS,
    level_rows_sha256,
    load_level_catalog,
)
from sokoban_agent.graph.agentic.builder import initialize_agentic_state
from sokoban_agent.graph.agentic.state import AgenticRuntimeContext


def test_shared_catalog_contains_custom_and_boxoban_levels() -> None:
    records = DEFAULT_LEVEL_CATALOG.records

    assert len(records) == 17
    assert {
        record.difficulty: sum(
            item.difficulty == record.difficulty for item in records
        )
        for record in records
    } == {
        "builtin": 2,
        "unfiltered": 5,
        "medium": 5,
        "hard": 5,
    }
    assert DEFAULT_LEVEL_CATALOG.get("tiny-walk").source_type == "custom"
    assert (
        DEFAULT_LEVEL_CATALOG.get("boxoban-medium-564").source_type
        == "boxoban"
    )
    assert DEFAULT_LEVELS.get("tiny-push").level_id == "tiny-push"


def test_catalog_board_hash_covers_exact_rows() -> None:
    record = DEFAULT_LEVEL_CATALOG.get("tiny-walk")

    assert record.sha256 == level_rows_sha256(record.rows)


def test_catalog_rejects_board_drift(tmp_path: Path) -> None:
    source = (
        Path(__file__).parents[1]
        / "src"
        / "sokoban_agent"
        / "data"
        / "level_catalog.json"
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["levels"][0]["rows"][1] = "#  .#"
    changed = tmp_path / "catalog.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="checksum"):
        load_level_catalog(changed)


def test_agentic_graph_resolves_catalog_id_and_checksum() -> None:
    record = DEFAULT_LEVEL_CATALOG.get("boxoban-medium-564")

    state = initialize_agentic_state(
        {
            "level_id": record.level_id,
            "level_sha256": record.sha256,
            "max_steps": record.recommended_max_steps,
        },
        Runtime[AgenticRuntimeContext](context={}),
    )

    assert state["level_id"] == record.level_id
    assert state["level_sha256"] == record.sha256
    assert state["level_rows"] == list(record.rows)


def test_agentic_graph_rejects_stale_catalog_reference() -> None:
    with pytest.raises(ValueError, match="checksum mismatch"):
        initialize_agentic_state(
            {
                "level_id": "tiny-push",
                "level_sha256": "stale",
            },
            Runtime[AgenticRuntimeContext](context={}),
        )

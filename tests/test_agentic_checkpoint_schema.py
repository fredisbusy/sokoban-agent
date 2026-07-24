from __future__ import annotations

import pytest

from sokoban_agent.graph.agentic.runtime import (
    AgenticGraphRunner,
    CheckpointSchemaMismatch,
)


def test_runner_rejects_checkpoint_without_schema_version() -> None:
    runner = AgenticGraphRunner()
    config = {"configurable": {"thread_id": "legacy-schema"}}
    runner.graph.update_state(config, {"status": "initialized"})

    with pytest.raises(CheckpointSchemaMismatch, match="start a new thread"):
        runner.run(
            {"level_id": "tiny-push", "max_steps": 2},
            thread_id="legacy-schema",
        )

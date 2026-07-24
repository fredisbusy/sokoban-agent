"""Agent Server persistence ownership contract."""

import importlib
import json
from pathlib import Path
from typing import Any


def _configured_graph() -> Any:
    config = json.loads(Path("langgraph.json").read_text(encoding="utf-8"))
    target = config["graphs"]["sokoban_agent"]
    path, variable = target.split(":", maxsplit=1)
    module_path = Path(path).with_suffix("").relative_to("src")
    module = importlib.import_module(".".join(module_path.parts))
    return getattr(module, variable)


def test_agent_server_graph_delegates_store_to_platform() -> None:
    assert _configured_graph().store is None

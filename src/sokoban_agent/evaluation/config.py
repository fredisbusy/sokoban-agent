"""Configuration contracts for reproducible evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ResearchRunConfig:
    """Reproducibility identity shared by every comparison record."""

    prompt_name: str
    prompt_commit: str
    graph_id: str
    graph_revision: str
    model_name: str
    seeds: tuple[int, ...]
    max_steps: int = 100
    max_planning_attempts: int = 3
    oracle_max_expanded_states: int = 100_000
    model_config: dict[str, object] = field(default_factory=dict)
    dirty_worktree: bool = False

    def __post_init__(self) -> None:
        if self.prompt_commit in {"", "unresolved", "latest"}:
            raise ValueError("research runs require an immutable prompt commit")
        if not self.seeds:
            raise ValueError("research runs require at least one seed")
        if self.max_steps < 1 or self.oracle_max_expanded_states < 1:
            raise ValueError("research limits must be positive")

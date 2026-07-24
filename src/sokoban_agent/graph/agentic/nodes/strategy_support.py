"""Validation feedback and metric helpers for strategy nodes."""

from __future__ import annotations

from pydantic import ValidationError

from sokoban_agent.graph.agentic.metrics import update_agentic_metrics
from sokoban_agent.graph.agentic.state import AgenticState, StrategySchemaIssue
from sokoban_agent.planning.llm.client import TextCompletion


def schema_issues(error: ValidationError) -> list[StrategySchemaIssue]:
    """Return bounded, JSON-safe issues for an LLM correction turn."""

    issues: list[StrategySchemaIssue] = []
    for detail in error.errors(include_url=False)[:8]:
        location = detail.get("loc", ())
        path = ".".join(str(part) for part in location) or "$"
        issue_type = detail.get("type", "validation_error")
        message = detail.get("msg", "유효하지 않은 값입니다")
        issues.append(
            {
                "path": path,
                "code": str(issue_type),
                "message": str(message),
            }
        )
    return issues or [
        {
            "path": "$",
            "code": "validation_error",
            "message": "전략 JSON이 스키마와 일치하지 않습니다",
        }
    ]


def schema_feedback(issue: StrategySchemaIssue) -> str:
    return (
        "strategy_schema_error: 스키마 오류; "
        f"path={issue['path']}; code={issue['code']}; "
        f"message={issue['message']}. 이 항목을 수정해 전체 JSON을 다시 제출하세요"
    )


def completion_metrics(
    state: AgenticState,
    completion: TextCompletion,
    *,
    strategy: dict[str, int] | None = None,
) -> dict[str, object]:
    """Return one nested metrics update for a completed model call."""

    metrics = state["metrics"]
    return {
        "metrics": update_agentic_metrics(
            metrics,
            strategy=strategy,
            llm={
                "calls": metrics["llm"]["calls"] + 1,
                "elapsed_seconds": (
                    metrics["llm"]["elapsed_seconds"]
                    + completion.metrics.total_seconds
                ),
                "prompt_tokens": (
                    metrics["llm"]["prompt_tokens"]
                    + completion.metrics.prompt_tokens
                ),
                "output_tokens": (
                    metrics["llm"]["output_tokens"]
                    + completion.metrics.output_tokens
                ),
            },
        ),
    }

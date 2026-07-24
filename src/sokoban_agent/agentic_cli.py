"""Command-line invocation of the shared structured LangGraph."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from dotenv import load_dotenv

from sokoban_agent.graph.agentic.runtime import AgenticGraphRunner
from sokoban_agent.planning.llm.client import OllamaSettings


def build_parser() -> argparse.ArgumentParser:
    """Build the structured agent CLI parser."""

    parser = argparse.ArgumentParser(
        description="LangGraph 구조화 전략으로 Sokoban을 실행합니다.",
    )
    parser.add_argument("--level-id", default="tiny-push")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--prompt-name", default="sokoban-strategy")
    parser.add_argument(
        "--prompt-commit",
        required=True,
        help="LangSmith의 immutable prompt commit hash",
    )
    parser.add_argument("--model")
    parser.add_argument(
        "--rationale-mode",
        choices=("on", "off"),
        default="on",
    )
    parser.add_argument(
        "--grounding-mode",
        choices=("direct", "local-search"),
        default="local-search",
    )
    parser.add_argument(
        "--memory-mode",
        choices=("off", "episode", "shared"),
        default="episode",
        help="episode은 실패 후보만, shared는 검증된 전략·접지 캐시도 재사용",
    )
    parser.add_argument(
        "--memory-namespace",
        default="default",
        help="shared 메모리를 분리할 실험 또는 사용자 namespace",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one episode and print a safe JSON result summary."""

    args = build_parser().parse_args(argv)
    load_dotenv(".env", override=True)
    model_name = args.model or OllamaSettings.from_env().model
    state = AgenticGraphRunner().run(
        {
            "level_id": args.level_id,
            "seed": args.seed,
            "max_steps": args.max_steps,
        },
        context={
            "prompt_name": args.prompt_name,
            "prompt_commit": args.prompt_commit,
            "model_name": model_name,
            "rationale_mode": args.rationale_mode,
            "grounding_mode": args.grounding_mode,
            "memory_mode": args.memory_mode,
            "memory_namespace": args.memory_namespace,
        },
    )
    print(
        json.dumps(
            {
                "level_id": state["meta"]["level_id"],
                "status": state["status"],
                "success": state["info"].get("success") is True,
                "actions": state["action_history"],
                "push_count": state["push_count"],
                "prompt": state["meta"]["prompt"],
                "model_name": state["meta"]["model_name"],
                "memory": {
                    "mode": state["meta"]["memory_mode"],
                    "requests": state["metrics"]["memory"]["requests"],
                    "hits": state["metrics"]["memory"]["hits"],
                    "writes": state["metrics"]["memory"]["writes"],
                    "llm_calls_saved": state["metrics"]["memory"][
                        "llm_calls_saved"
                    ],
                },
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

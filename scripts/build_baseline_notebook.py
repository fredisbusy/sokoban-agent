"""Build the reproducible Random-vs-BFS baseline notebook."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

LEVEL_IDS = ["tiny-push", "tiny-walk"]
SEEDS = list(range(10))
MAX_STEPS = 40

_new_code_cell = cast(Callable[[str], Any], new_code_cell)
_new_markdown_cell = cast(Callable[[str], Any], new_markdown_cell)
_new_notebook = cast(Callable[..., Any], new_notebook)
_write_notebook = cast(Callable[[Any, Path], None], nbformat.write)


def build_notebook(output_path: Path) -> None:
    """Create a notebook with visible inputs and bounded outputs."""

    notebook = _new_notebook(
        cells=[
            _new_markdown_cell(
                "# Random vs BFS 기준선\n\n"
                "## Experiment Goal\n\n"
                "- LangGraph 실행기에서 Random과 BFS 기준선이 같은 cohort를 "
                "사용하는지 확인한다.\n"
                "- 저장 출력은 없다. 위에서 아래로 실행해 현재 결과를 만든다."
            ),
            _new_markdown_cell(
                "## Context & Methods\n\n"
                "두 기준선을 같은 내장 레벨과 seed 조합에서 실행한다. "
                "환경의 `max_steps`는 모든 실행에 동일하게 적용한다.\n\n"
                "### Key Assumptions\n\n"
                "- 현재 비교 범위는 `tiny-push`, `tiny-walk`다.\n"
                "- BFS는 primitive action 최단 경로를 완전 탐색한다.\n"
                "- 실행 시간에는 LangGraph 초기화·계획·검증·실행이 포함된다."
            ),
            _new_markdown_cell("### 1. Setup"),
            _new_code_cell(
                "from dataclasses import asdict\n\n"
                "import matplotlib.pyplot as plt\n"
                "import pandas as pd\n\n"
                "from sokoban_agent.env import SokobanEnv\n"
                "from sokoban_agent.evaluation import (\n"
                "    run_benchmark,\n"
                "    summarize_by_planner,\n"
                ")\n"
                "from sokoban_agent.planning import BFSPlanner, RandomPlanner\n\n"
                f"LEVEL_IDS = {LEVEL_IDS!r}\n"
                f"SEEDS = {SEEDS!r}\n"
                f"MAX_STEPS = {MAX_STEPS}"
            ),
            _new_markdown_cell("## Data\n\n### 2. Run identical cases"),
            _new_code_cell(
                "env = SokobanEnv(max_steps=MAX_STEPS)\n"
                "try:\n"
                "    results = run_benchmark(\n"
                "        env,\n"
                "        [RandomPlanner(), BFSPlanner()],\n"
                "        level_ids=LEVEL_IDS,\n"
                "        seeds=SEEDS,\n"
                "    )\n"
                "finally:\n"
                "    env.close()\n\n"
                "results_df = pd.DataFrame.from_records(\n"
                "    asdict(result) for result in results\n"
                ")\n"
                "results_df.head(8)"
            ),
            _new_markdown_cell("## Results\n\n### 3. Compare metrics"),
            _new_code_cell(
                "summary_df = pd.DataFrame.from_records(\n"
                "    asdict(summary) for summary in summarize_by_planner(results)\n"
                ").set_index('planner_name')\n"
                "summary_df"
            ),
            _new_code_cell(
                "fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))\n"
                "summary_df['success_rate'].plot.bar(\n"
                "    ax=axes[0], ylim=(0, 1), title='Success rate'\n"
                ")\n"
                "summary_df['mean_actions'].plot.bar(\n"
                "    ax=axes[1], title='Mean actions per episode'\n"
                ")\n"
                "for axis in axes:\n"
                "    axis.set_xlabel('Graph planner')\n"
                "    axis.tick_params(axis='x', rotation=0)\n"
                "fig.tight_layout()\n"
                "plt.show()"
            ),
            _new_markdown_cell("### 4. Validate the comparison cohort"),
            _new_code_cell(
                "case_sets = {\n"
                "    planner_name: set(\n"
                "        zip(group['level_id'], group['seed'], strict=True)\n"
                "    )\n"
                "    for planner_name, group in results_df.groupby('planner_name')\n"
                "}\n"
                "assert case_sets['graph:random'] == case_sets['graph:bfs']\n"
                "assert len(case_sets['graph:bfs']) == len(LEVEL_IDS) * len(SEEDS)\n"
                "assert summary_df.loc['graph:bfs', 'success_rate'] == 1.0\n"
                "case_sets"
            ),
            _new_markdown_cell(
                "## Takeaways\n\n"
                "실행 후 위 집계표와 검증 셀을 기준선 결과로 사용한다. "
                "이 실험은 작은 내장 레벨만 다룬다."
            ),
        ],
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_notebook(notebook, output_path)


def main() -> None:
    """Write the notebook at the repository's standard path."""

    repository_root = Path(__file__).resolve().parents[1]
    build_notebook(repository_root / "notebooks" / "baseline_comparison.ipynb")


if __name__ == "__main__":
    main()

"""Build the reproducible LLM-vs-baselines experiment notebook."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

LEVEL_IDS = ["tiny-push", "tiny-walk", "heldout-turn"]
SEEDS = [0, 1]
MAX_STEPS = 15
MAX_ATTEMPTS = 3

_new_code_cell = cast(Callable[[str], Any], new_code_cell)
_new_markdown_cell = cast(Callable[[str], Any], new_markdown_cell)
_new_notebook = cast(Callable[..., Any], new_notebook)
_write_notebook = cast(Callable[[Any, Path], None], nbformat.write)


def build_notebook(output_path: Path) -> None:
    """Create the reader-facing LLM comparison notebook."""

    notebook = _new_notebook(
        cells=[
            _new_markdown_cell(
                "# LLM Sokoban Agent 비교 실험\n\n"
                "## tl;dr\n\n"
                "- 저장된 `gemma4:26b` 실행은 동일한 3개 레벨 × 2개 seed 중 "
                "2/6 에피소드(33%)를 해결했다. 성공은 `tiny-push` 2회였다.\n"
                "- BFS는 6/6, Random은 0/6을 해결했다.\n"
                "- LLM은 총 88회 호출했고 30회 재시도했으며, Python 검증기가 "
                "막힌 행동 32회를 환경 실행 전에 거절했다.\n"
                "- 이 수치는 아래 고정 설정으로 저장된 실행 결과이며, 모델이나 "
                "설정을 바꾸면 다시 실행해 갱신해야 한다."
            ),
            _new_markdown_cell(
                "## Context & Methods\n\n"
                "상태 배열을 표준 Sokoban 문자 보드로 직렬화하고, 모델에는 "
                '`{"action":"UP"}` 형태의 JSON Schema 출력을 요구한다. '
                "Python Agent가 응답 형식과 현재 보드에서의 이동 가능 여부를 "
                "검증한 뒤 환경에 전달한다.\n\n"
                "### Key Assumptions\n\n"
                "- `.env`의 Ollama 서버와 모델이 실행 시점에 사용 가능하다.\n"
                "- `temperature=0`과 episode seed를 모델 호출에 전달하지만, "
                "런타임·모델 버전에 따라 완전한 결정성은 보장되지 않는다.\n"
                "- `heldout-turn`은 제품 기본 레벨에 포함하지 않은 실험용 "
                "소형 레벨이다.\n"
                "- 실행 시간은 로컬/원격 Ollama 상태의 영향을 받는다."
            ),
            _new_markdown_cell("### 1. Load dependencies and fixed parameters"),
            _new_code_cell(
                "from dataclasses import asdict\n\n"
                "import matplotlib.pyplot as plt\n"
                "import pandas as pd\n"
                "from IPython.display import Markdown, display\n\n"
                "from sokoban_agent.agents import BFSAgent, LLMAgent, RandomAgent\n"
                "from sokoban_agent.agents.llm import OllamaClient, OllamaSettings\n"
                "from sokoban_agent.env import (\n"
                "    DEFAULT_LEVELS,\n"
                "    FixedLevelProvider,\n"
                "    SokobanEnv,\n"
                "    parse_level,\n"
                ")\n"
                "from sokoban_agent.evaluation import (\n"
                "    run_benchmark,\n"
                "    summarize_by_agent,\n"
                ")\n\n"
                f"LEVEL_IDS = {LEVEL_IDS!r}\n"
                f"SEEDS = {SEEDS!r}\n"
                f"MAX_STEPS = {MAX_STEPS}\n"
                f"MAX_ATTEMPTS = {MAX_ATTEMPTS}\n\n"
                "settings = OllamaSettings.from_env()\n"
                "experiment_config = {\n"
                "    'model': settings.model,\n"
                "    'temperature': settings.temperature,\n"
                "    'level_ids': LEVEL_IDS,\n"
                "    'seeds': SEEDS,\n"
                "    'max_steps': MAX_STEPS,\n"
                "    'max_attempts': MAX_ATTEMPTS,\n"
                "}\n"
                "experiment_config"
            ),
            _new_markdown_cell(
                "## Data\n\n"
                "### 2. Build the fixed and held-out level cohort"
            ),
            _new_code_cell(
                "heldout_level = parse_level(\n"
                "    'heldout-turn',\n"
                "    [\n"
                "        '#####',\n"
                "        '#.  #',\n"
                "        '# $ #',\n"
                "        '#  @#',\n"
                "        '#####',\n"
                "    ],\n"
                ")\n"
                "level_provider = FixedLevelProvider(\n"
                "    [\n"
                "        DEFAULT_LEVELS.get('tiny-push'),\n"
                "        DEFAULT_LEVELS.get('tiny-walk'),\n"
                "        heldout_level,\n"
                "    ]\n"
                ")\n"
                "LEVEL_IDS"
            ),
            _new_markdown_cell("### 3. Run identical cases"),
            _new_code_cell(
                "client = OllamaClient(settings)\n"
                "agents = [\n"
                "    RandomAgent(),\n"
                "    BFSAgent(),\n"
                "    LLMAgent(\n"
                "        client,\n"
                "        model_name=settings.model,\n"
                "        max_attempts=MAX_ATTEMPTS,\n"
                "    ),\n"
                "]\n"
                "env = SokobanEnv(\n"
                "    level_provider=level_provider,\n"
                "    max_steps=MAX_STEPS,\n"
                ")\n"
                "try:\n"
                "    results = run_benchmark(\n"
                "        env,\n"
                "        agents,\n"
                "        level_ids=LEVEL_IDS,\n"
                "        seeds=SEEDS,\n"
                "    )\n"
                "finally:\n"
                "    env.close()\n\n"
                "results_df = pd.DataFrame.from_records(\n"
                "    asdict(result) for result in results\n"
                ")\n"
                "results_df"
            ),
            _new_markdown_cell(
                "## Results\n\n"
                "### 4. Compare success, efficiency, recovery, and latency"
            ),
            _new_code_cell(
                "summary_df = pd.DataFrame.from_records(\n"
                "    asdict(summary) for summary in summarize_by_agent(results)\n"
                ").set_index('agent_name')\n"
                "summary_columns = [\n"
                "    'episode_count',\n"
                "    'success_count',\n"
                "    'success_rate',\n"
                "    'mean_actions',\n"
                "    'mean_actions_on_success',\n"
                "    'mean_elapsed_seconds',\n"
                "    'total_llm_calls',\n"
                "    'total_llm_retries',\n"
                "    'total_llm_format_errors',\n"
                "    'total_llm_invalid_actions',\n"
                "    'mean_llm_elapsed_seconds',\n"
                "]\n"
                "summary_df[summary_columns]"
            ),
            _new_code_cell(
                "fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))\n"
                "summary_df['success_rate'].plot.bar(\n"
                "    ax=axes[0], ylim=(0, 1), title='Success rate'\n"
                ")\n"
                "summary_df['mean_actions'].plot.bar(\n"
                "    ax=axes[1], title='Mean actions per episode'\n"
                ")\n"
                "summary_df['mean_elapsed_seconds'].plot.bar(\n"
                "    ax=axes[2], title='Mean elapsed seconds'\n"
                ")\n"
                "for axis in axes:\n"
                "    axis.set_xlabel('Agent')\n"
                "    axis.tick_params(axis='x', rotation=20)\n"
                "fig.tight_layout()\n"
                "plt.show()"
            ),
            _new_markdown_cell("### 5. Inspect LLM failures by level and seed"),
            _new_code_cell(
                "llm_name = f'llm:{settings.model}'\n"
                "llm_columns = [\n"
                "    'level_id',\n"
                "    'seed',\n"
                "    'success',\n"
                "    'deadlock',\n"
                "    'truncated',\n"
                "    'action_count',\n"
                "    'llm_calls',\n"
                "    'llm_retries',\n"
                "    'llm_client_errors',\n"
                "    'llm_format_errors',\n"
                "    'llm_invalid_actions',\n"
                "    'llm_elapsed_seconds',\n"
                "    'failure_reason',\n"
                "]\n"
                "llm_results_df = results_df[\n"
                "    results_df['agent_name'] == llm_name\n"
                "][llm_columns]\n"
                "llm_results_df"
            ),
            _new_markdown_cell("### 6. Validate the comparison cohort"),
            _new_code_cell(
                "case_sets = {\n"
                "    agent_name: set(\n"
                "        zip(group['level_id'], group['seed'], strict=True)\n"
                "    )\n"
                "    for agent_name, group in results_df.groupby('agent_name')\n"
                "}\n"
                "expected_cases = set(\n"
                "    (level_id, seed)\n"
                "    for level_id in LEVEL_IDS\n"
                "    for seed in SEEDS\n"
                ")\n"
                "assert all(cases == expected_cases for cases in case_sets.values())\n"
                "assert summary_df.loc['bfs', 'success_rate'] == 1.0\n"
                "assert summary_df.loc[llm_name, 'total_llm_client_errors'] == 0\n"
                "{name: len(cases) for name, cases in case_sets.items()}"
            ),
            _new_markdown_cell("## Takeaways"),
            _new_code_cell(
                "llm_summary = summary_df.loc[llm_name]\n"
                "bfs_summary = summary_df.loc['bfs']\n"
                "solved_levels = llm_results_df.loc[\n"
                "    llm_results_df['success'], 'level_id'\n"
                "].value_counts().to_dict()\n"
                "display(Markdown(\n"
                "    f'- LLM은 {int(llm_summary.success_count)}/'\n"
                "    f'{int(llm_summary.episode_count)} 에피소드를 해결했다 '\n"
                "    f'({llm_summary.success_rate:.0%}).\\n'\n"
                "    f'- BFS는 같은 cohort에서 '\n"
                "    f'{bfs_summary.success_rate:.0%} 성공했다.\\n'\n"
                "    f'- LLM의 총 호출은 {int(llm_summary.total_llm_calls)}회, '\n"
                "    f'재시도는 {int(llm_summary.total_llm_retries)}회, '\n"
                "    f'막힌 행동 거절은 '\n"
                "    f'{int(llm_summary.total_llm_invalid_actions)}회였다.\\n'\n"
                "    f'- 성공 레벨별 횟수: `{solved_levels}`. 이 결과는 소형 '\n"
                "    '3개 레벨의 탐색 실험이며 전체 Boxoban 성능을 대표하지 않는다.'\n"
                "))"
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
    build_notebook(
        repository_root / "notebooks" / "llm_agent_comparison.ipynb"
    )


if __name__ == "__main__":
    main()

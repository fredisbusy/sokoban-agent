# Sokoban Agent

정답 경로 없이 Sokoban을 관찰하고 계획하는 AI 에이전트를 단계적으로
연구하는 Python 프로젝트입니다.

LangGraph를 실행 코어로 사용합니다. `plan → validate → execute` 그래프가
계획, 규칙 검증, 실행, 재시도와 체크포인트를 관리하고
Random/BFS/A*/LLM Planner를 같은 흐름에 연결합니다. 탐색 알고리즘은 전체
계획을, LLM은 JSON Schema로 최대 8개 행동을 제안하며 환경 규칙 노드가 전체
계획을 실행 전에 검증합니다.
목표와 범위는 [PROJECT](docs/PROJECT.md), 작업 순서는 [TODO](TODO.md)에서
관리합니다.

## 빠른 시작

Python 3.11~3.13과 [uv](https://docs.astral.sh/uv/)가 필요합니다.

```bash
git clone https://github.com/fredisbusy/sokoban-agent.git
cd sokoban-agent
uv sync
uv run sokoban-play --level tiny-push
```

Ollama 없이도 LangGraph와 Random/BFS Planner를 사용할 수 있습니다.

### 필요한 기능만 설치

기본 환경, LangGraph와 알고리즘 Planner만 설치하려면 개발 의존성을
제외합니다.

```bash
uv sync --no-dev
```

LLM, vision, notebook 기능은 각각 명시적으로 설치합니다.

```bash
uv sync --no-dev --extra llm
uv sync --no-dev --extra vision
uv sync --group notebook
```

기능별 선택 기준은 [의존성과 기술 선택](docs/DEPENDENCIES.md)에 정리했습니다.

## Sokoban 환경 사용

기본 환경은 작은 고정 레벨을 포함하며 Gymnasium에
`SokobanAgent-v0`로 등록됩니다.

```python
import gymnasium as gym
import sokoban_agent.env

env = gym.make("SokobanAgent-v0", render_mode="ansi")
observation, info = env.reset(seed=42, options={"level_id": "tiny-push"})
observation, reward, terminated, truncated, info = env.step(0)

print(env.render())
print(info)
env.close()
```

행동은 `0=UP`, `1=RIGHT`, `2=DOWN`, `3=LEFT`입니다. 관찰은 각 칸을
`FLOOR`, `WALL`, `TARGET`, `BOX`, `PLAYER`, `BOX_ON_TARGET`,
`PLAYER_ON_TARGET` 중 하나로 표현한 `(높이, 너비)` 형태의 `uint8`
배열입니다.

### 터미널에서 직접 플레이

내장 레벨은 터미널에서 방향키 또는 `WASD`로 플레이할 수 있습니다.

```bash
uv run sokoban-play
```

`R`은 현재 레벨을 다시 시작하고 `Q`는 게임을 종료합니다. 한 번 밀면
완료되는 입문 레벨은 다음처럼 선택합니다.

```bash
uv run sokoban-play --level tiny-push
```

터미널 플레이는 현재 macOS와 Linux의 대화형 터미널을 지원합니다.

### Boxoban 레벨 사용

[Google DeepMind Boxoban 레벨](https://github.com/google-deepmind/boxoban-levels)은
외부 데이터셋으로 관리합니다. 필요한 경우 다음 위치에 내려받습니다.

```bash
git clone --depth 1 \
  https://github.com/google-deepmind/boxoban-levels.git \
  data/boxoban
```

한 개의 텍스트 파일이나 `train`, `valid`, `test` 디렉터리를 레벨 공급자로
연결할 수 있습니다. 선택된 파일만 파싱하여 전체 학습 세트를 메모리에
올리지 않습니다. 한 공급자에 포함된 레벨은 보드 크기가 같아야 하며,
레벨 ID는 `상대/파일.txt:헤더` 형식입니다.

```python
from sokoban_agent.env import BoxobanLevelProvider, SokobanEnv

levels = BoxobanLevelProvider("data/boxoban/unfiltered/train")
env = SokobanEnv(level_provider=levels, render_mode="ansi")

observation, info = env.reset(seed=42)
print(info["level_id"], env.render())
env.close()
```

저장소에는 전체 데이터셋 없이도 파서와 환경을 확인할 수 있는
`assets/levels/boxoban_sample.txt`가 포함됩니다.

## Ollama 연결 (선택)

Ollama 서버가 실행 중이고 사용할 모델이 서버에 설치되어 있어야 합니다.
예제 설정을 복사한 뒤 서버 주소와 모델명을 환경에 맞게 수정합니다.

```bash
cp .env.example .env
uv run python scripts/check_ollama.py
```

`.env`에서 필수 값은 `OLLAMA_API_BASE`입니다. 모델과 제한 시간에는
기본값이 있습니다.

```dotenv
OLLAMA_API_BASE=https://model.byfred.io
OLLAMA_MODEL=qwen3.6:27b-mlx
OLLAMA_TIMEOUT_SECONDS=300
OLLAMA_TEMPERATURE=0
OLLAMA_NUM_CTX=4096
OLLAMA_MAX_OUTPUT_TOKENS=256
OLLAMA_KEEP_ALIVE=30m
OLLAMA_THINK=false
LANGSMITH_TRACING=false
```

Python에서는 다음처럼 텍스트 응답을 요청합니다. 이 클라이언트는 아직
독립적으로 사용할 수도 있습니다.

```python
from sokoban_agent.planning.llm import OllamaClient

client = OllamaClient.from_env()
answer = client.complete("다음 행동을 UP, DOWN, LEFT, RIGHT 중 하나로 답해줘.")
print(answer.content)
print(answer.metrics)
```

실제 보드 플레이에는 같은 클라이언트를 `LLMPlanner`에 연결하고,
`SokobanGraph`가 실행과 복구를 담당합니다.

```python
from sokoban_agent.planning import LLMPlanner
from sokoban_agent.planning.llm import OllamaClient
from sokoban_agent.env import SokobanEnv
from sokoban_agent.graph import SokobanGraph

client = OllamaClient.from_env()
planner = LLMPlanner(client, model_name=client.settings.model)
graph = SokobanGraph(SokobanEnv(), planner, max_planning_attempts=3)
state = graph.run(level_id="tiny-push", thread_id="example-episode")
print(state["info"]["success"], state["action_history"])
```

LLM 제안을 push 기반 A*로 검사하고, 안전하면 탐색한 후속 경로를 재사용하며,
위험하면 현재 상태의 안전한 전체 계획으로 대체할 수도 있습니다.

```python
from sokoban_agent.planning import SearchGuardPlanner

hybrid = SearchGuardPlanner(planner)
graph = SokobanGraph(SokobanEnv(), hybrid)
state = graph.run(level_id="tiny-push")
print(state["algorithm_calls"], state["algorithm_fallbacks"])
```

## LangGraph Studio

로컬 Studio에서는 `initialize`, `llm_plan`, `astar_guard`,
`validate_plan`, `execute_action` 노드를 단계별로 살펴볼 수 있습니다.
Studio용 의존성을 설치하고 개발 서버를 실행합니다.

```bash
uv sync --group studio
uv run --group studio langgraph dev
```

Studio 입력에는 다음처럼 내장 레벨과 실행 한도를 지정합니다.

```json
{"level_id": "tiny-walk", "seed": 0, "max_steps": 15}
```

각 단계에서 `board`, `planner_goal`, `decision_summary`, `risk`,
`proposed_plan`, `guard_summary`, `decision_log`를 확인할 수 있습니다.
노드와 필드 이름은 영어로 유지합니다. 각 필드의 목표, 판단 근거, 위험,
A* 검사 결과 내용은 한국어로 기록되며 숨은 추론 원문은 포함하지 않습니다.

Studio 진입점은 `langgraph.json`과
`src/sokoban_agent/graph/studio.py`에 정의되어 있습니다.
`LANGSMITH_TRACING=false`가 기본값이므로 실행 기록은 외부 LangSmith로
전송하지 않습니다.

## 연구와 검증

```bash
uv run pytest
uv run ruff check .
uv run mypy
```

`make test`, `make lint`, `make typecheck`는 같은 검사의 단축 명령입니다.
기준선 비교 노트북은 다음 명령으로 다시 생성하고 실행할 수 있습니다.

```bash
uv run --group notebook python scripts/build_baseline_notebook.py
uv run --group notebook python -m jupyter nbconvert \
  --execute --to notebook --inplace notebooks/baseline_comparison.ipynb
```

주 실험은 `.env`의 모델을 사용해 Random, BFS, push 기반 A*, LLM,
LLM+BFS Guard, LLM+A* Guard를 비교합니다.

```bash
uv run --group notebook python scripts/build_langgraph_comparison_notebook.py
uv run --group notebook python -m jupyter nbconvert \
  --execute --to notebook --inplace notebooks/langgraph_planner_comparison.ipynb
```

`baseline_comparison.ipynb`는 LangGraph 실행기와 BFS 기준선을 빠르게
확인하는 사전 점검용이다. 실제 모델과 하이브리드 성능을 비교할 때는
`langgraph_planner_comparison.ipynb`를 사용한다. 주 실험 노트북에는 마지막
실행의 설정, 표, 차트와 이동 재생 출력을 저장한다.

노트북은 실험과 시각화에만 사용하고, 재사용 코드는 `src/`에 둡니다.

## 프로젝트 구조

현재 기능이 들어 있는 경로는 다음과 같습니다.

```text
src/sokoban_agent/
├── env/               # 게임 규칙, 렌더링, 레벨 공급자
├── planning/          # Planner 계약과 Random/BFS/A*/LLM 계획 노드
├── graph/             # LangGraph 상태, 검증, 실행, 복구, 체크포인트
├── evaluation/        # 그래프 벤치마크, 결과, 집계, trajectory
└── play.py            # 터미널 플레이
assets/levels/         # 저장소에 포함된 예제 레벨
notebooks/             # 실행 결과를 포함한 기준선 비교
scripts/               # Ollama 확인과 노트북 생성
tests/                 # 환경, 레벨, 클라이언트, 플레이 검증
docs/                  # 목표와 아키텍처
```

## 문서

- [프로젝트 목표와 연구 범위](docs/PROJECT.md)
- [핵심 아키텍처와 실행 흐름](docs/ARCHITECTURE.md)
- [의존성과 기술 선택](docs/DEPENDENCIES.md)
- [현재 우선순위와 완료 조건](TODO.md)
- [에이전트 작업 규칙](AGENTS.md)

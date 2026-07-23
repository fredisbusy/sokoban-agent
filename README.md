# Sokoban Agent

정답 경로 없이 처음 보는 Sokoban을 관찰하고, 구조화된 가설과 하위 목표를
세워 실행 결과로 수정하는 AI 에이전트를 단계적으로 연구하는 Python
프로젝트입니다.

LangGraph를 실행 코어로 사용합니다. `plan → validate → execute` 그래프가
계획, 규칙 검증, 실행, 재시도와 체크포인트를 관리하고
Random/BFS/A*/LLM Planner를 같은 흐름에 연결합니다. 탐색 알고리즘은 전체
계획을, LLM은 JSON Schema로 최대 8개 행동을 제안하며 환경 규칙 노드가 전체
계획을 실행 전에 검증합니다.
목표와 범위는 [PROJECT](docs/PROJECT.md), 작업 순서는 [TODO](TODO.md)에서
관리합니다.

현재 원시 행동 LLM과 LLM+A* Search Guard는 비교 baseline입니다. 다음
단계에서는 전역 A*를 실행 중 정답 제공자가 아닌 평가 oracle로 분리하고,
에이전트가 보드 분석, 상자-목표 배정, 보호 제약과 실행 가능한 하위 목표를
직접 유지하게 합니다. 상태 전이·재시도·체크포인트는 LangGraph가, prompt
본문과 버전은 LangSmith Prompt Management가 맡으며 자체 workflow나 prompt
registry를 만들지 않습니다. 구체적인 전환 계획은
[구조화된 문제 해결 에이전트 연구 계획](docs/AGENTIC_PLANNING.md)에
정리했습니다.

## 빠른 시작

Python 3.11~3.13과 [uv](https://docs.astral.sh/uv/)가 필요합니다.

```bash
git clone https://github.com/fredisbusy/sokoban-agent.git
cd sokoban-agent
make sync
make play LEVEL=tiny-push
```

Ollama 없이도 LangGraph와 Random/BFS Planner를 사용할 수 있습니다.

### 의존성 설치

개발, LLM, vision, notebook, Studio 의존성을 한 번에 설치합니다.

```bash
make sync
```

기능별 선택 기준은 [의존성과 기술 선택](docs/DEPENDENCIES.md)에 정리했습니다.
사용 가능한 명령 전체는 `make help`로 확인할 수 있습니다.

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
make play
```

`R`은 현재 레벨을 다시 시작하고 `Q`는 게임을 종료합니다. 한 번 밀면
완료되는 입문 레벨은 다음처럼 선택합니다.

```bash
make play LEVEL=tiny-push
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
make ollama-check
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

로컬 Studio에서는 `initialize`, `analyze`, `resolve_prompt`,
`compose_strategy_input`, `propose_strategy`, `verify_strategy` 노드를
단계별로 살펴볼 수 있습니다. 전역 A* 정답 경로나 전체 계획 대체는 이
주 그래프에 연결하지 않습니다.
Studio용 의존성 설치와 개발 서버 실행은 다음 명령으로 한 번에 처리합니다.

```bash
make studio
```

Agent Server와 Studio 연결 주소는 브라우저 호환성을 위해
`http://localhost:2024`로 고정됩니다.

Studio 입력에는 다음처럼 내장 레벨과 실행 한도를 지정합니다.

```json
{"level_id": "tiny-walk", "seed": 0, "max_steps": 15}
```

Studio의 **Manage Assistants**에서 다음 context를 설정합니다. `prompt_commit`
에는 `latest` 같은 mutable selector보다 LangSmith에서 확인한 commit hash를
사용해야 연구 실행을 재현할 수 있습니다.

```json
{
  "prompt_name": "sokoban-strategy",
  "prompt_commit": "<LANGSMITH_PROMPT_COMMIT>",
  "model_name": "qwen3.6:27b-mlx",
  "rationale_mode": "on"
}
```

각 단계에서 `board_analysis`, `prompt`, `strategy_input`,
`strategy_hypothesis`, `strategy_violations`, `active_subgoal`,
`protected_constraints`, `expected_effect`, `feedback`,
`decision_events`를 확인할 수 있습니다. prompt 본문과 숨은 추론 원문은
checkpoint에 저장하지 않습니다.

Studio 진입점은 `langgraph.json`과
`src/sokoban_agent/graph/agentic.py`에 정의되어 있습니다.
`LANGSMITH_TRACING=false`가 기본값이므로 실행 기록은 외부 LangSmith로
전송하지 않습니다.

## 연구와 검증

```bash
make check
```

각 검사는 `make test`, `make lint`, `make typecheck`로 따로 실행할 수 있습니다.
기준선 비교 노트북은 다음 명령으로 다시 생성하고 실행할 수 있습니다.

```bash
make baseline-notebook
```

주 실험은 `.env`의 모델을 사용해 Random, BFS, push 기반 A*, LLM,
LLM+BFS Guard, LLM+A* Guard를 비교합니다.

```bash
make experiment-notebook
```

`baseline_comparison.ipynb`는 LangGraph 실행기와 BFS 기준선을 빠르게
확인하는 사전 점검용이다. 실제 모델과 하이브리드 성능을 비교할 때는
`langgraph_planner_comparison.ipynb`를 사용한다. 주 실험 노트북에는 마지막
실행의 설정, 표, 차트와 이동 재생 출력을 저장한다.

노트북은 실험과 시각화에만 사용하고, 재사용 코드는 `src/`에 둡니다.

### LLM 기여도 파일럿

LLM이 실제로 탐색을 돕는지 확인하는 주 실험은 고정된 Boxoban 코호트에서
다음 다섯 정책을 비교합니다.

- `astar-only`: 모델 없이 bounded A*만 사용
- `llm-common-validation`: LLM 제안을 공통 규칙 노드로만 검사
- `llm-suffix-only`: 유효한 LLM prefix 뒤에서만 A* 실행, 전체 대체 금지
- `llm-full-guard`: prefix 보강이 실패하면 현재 상태를 A*로 전체 대체
- `llm-always-replace`: LLM을 호출하되 제안을 버리는 음성 대조군

외부 데이터는 저장소에 커밋하지 않습니다. Apache-2.0 Boxoban 데이터를
고정 커밋으로 내려받고 체크섬과 50개 레벨 manifest를 검증합니다. 앞의
30개가 파일럿, 전체 50개가 확인 코호트입니다.

```bash
uv run python scripts/prepare_boxoban_pilot.py --download
uv run python scripts/run_boxoban_pilot.py
```

빠른 연결 점검은 `--cohort-size 1`, 확인 코호트는 `--cohort-size 50`으로
실행합니다. 결과는 기본적으로
`_workspace/benchmarks/boxoban_pilot_v1.jsonl`에 에피소드마다 저장되며,
같은 설정으로 다시 실행하면 완료된 항목을 건너뜁니다.

성공률과 행동 수 외에도 push 수, 상태 재방문, 반복 계획, LLM 제안·합법
prefix·실제 채택 행동 수, guard 판정, suffix 탐색량, 처음 상태에서 다시 푼
bounded A* reference와의 행동·push·확장 상태 차이를 기록합니다. 이
reference는 탐색 한도가 있는 비교 기준이며 수학적 최적해라고 부르지 않습니다.

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
- [구조화된 문제 해결 에이전트 연구 계획](docs/AGENTIC_PLANNING.md)
- [LangGraph 실시간 Sokoban 관찰 화면 계획](docs/LIVE_VIEWER.md)
- [핵심 아키텍처와 실행 흐름](docs/ARCHITECTURE.md)
- [의존성과 기술 선택](docs/DEPENDENCIES.md)
- [현재 우선순위와 완료 조건](TODO.md)
- [에이전트 작업 규칙](AGENTS.md)

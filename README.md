# Sokoban Agent

정답 경로 없이 처음 보는 Sokoban을 관찰하고, 구조화된 가설과 하위 목표를
세워 실행 결과로 수정하는 AI 에이전트를 단계적으로 연구하는 Python
프로젝트입니다.

LangGraph를 실행 코어로 사용합니다. 비교 baseline의
`plan → validate → execute` 그래프와, 보드 분석부터 전략 제안·검증,
단일 push 실행, 관찰·성찰을 반복하는 구조화 `StateGraph`를 함께
유지합니다. Random/BFS/A*/원시 행동 LLM은 같은 평가 흐름에 연결하고,
주 구조화 정책에서 전역 A*는 실행 중 정답 제공자가 아니라 평가 oracle로만
사용합니다.
목표와 범위는 [PROJECT](docs/PROJECT.md), 작업 순서는 [TODO](TODO.md)에서
관리합니다.

현재 원시 행동 LLM과 LLM+A* Search Guard는 비교 baseline입니다. 주
에이전트는 보드 분석, 상자-목표 배정, 보호 제약과 실행 가능한 하위 목표를
직접 유지합니다. 상태 전이·재시도·체크포인트는 LangGraph가, prompt
본문과 버전은 LangSmith Prompt Management가 맡으며 자체 workflow나 prompt
registry를 만들지 않습니다. 구체적인 연구 설계는
[구조화된 문제 해결 에이전트 연구 계획](docs/AGENTIC_PLANNING.md)에
정리했습니다.

## 빠른 시작

Python 3.13과 [uv](https://docs.astral.sh/uv/)가 필요합니다.

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

연구와 웹 선택기에 쓰는 공식 난이도별 코호트는 전체 데이터셋 대신 고정
원본 파일 3개만 내려받습니다.

```bash
make boxoban-research-data
```

`benchmarks/boxoban_research_v1.json`에는 Boxoban이 정의한
`unfiltered`, `medium`, `hard`에서 SHA-256 규칙으로 선정한 5개씩, 총 15개
보드가 원본 commit·파일 checksum·Apache-2.0 출처와 함께 고정되어 있습니다.
모두 bounded A*로 손상 여부와 해 경로를 확인했지만, 그 경로는 주 에이전트
실행에 제공하지 않습니다.

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
OLLAMA_MAX_OUTPUT_TOKENS=512
OLLAMA_KEEP_ALIVE=30m
OLLAMA_THINK=false
LANGSMITH_TRACING=true
```

모델 호출은 provider HTTP를 직접 구현하지 않고 공식 `ChatLiteLLM`
LangChain 통합을 사용합니다. 따라서 LangGraph 실행 안에서 LangSmith
callback과 trace context가 모델 호출까지 이어집니다. Python에서는 다음처럼
동일한 Adapter를 독립적으로 확인할 수도 있습니다.

```python
from sokoban_agent.planning.llm import LiteLLMClient

client = LiteLLMClient.from_env()
answer = client.complete("다음 행동을 UP, DOWN, LEFT, RIGHT 중 하나로 답해줘.")
print(answer.content)
print(answer.metrics)
```

실제 보드 플레이에는 같은 클라이언트를 `LLMPlanner`에 연결하고,
`SokobanGraph`가 실행과 복구를 담당합니다.

```python
from sokoban_agent.planning import LLMPlanner
from sokoban_agent.planning.llm import LiteLLMClient
from sokoban_agent.env import SokobanEnv
from sokoban_agent.graph import SokobanGraph

client = LiteLLMClient.from_env()
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
`recall_failures`, `compose_strategy_input`, `recall_strategy`,
`propose_strategy`, `verify_strategy`, `remember_failure`,
`detect_repetition`, `recall_grounding`, `ground_subgoal`,
`execute_until_push`, `reflect`, `remember_outcome`, `observe` 노드를 단계별로
살펴볼 수 있습니다. 각 실행 node는 첫 push에서 멈춘 뒤 실행 효과를
성찰하며, 미해결이면 새 관찰로 돌아갑니다. 전역 A* 정답 경로나 전체 계획
대체는 이 주 그래프에 연결하지 않습니다.
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

Assistant context를 생략하면 prompt는 `sokoban-strategy:latest`를 기본으로
해석하고, 실제 선택된 immutable commit을 실행 state에 기록합니다.
Studio의 **Manage Assistants**에서 다음 context를 설정합니다. `prompt_commit`
에는 `latest` 같은 mutable selector보다 LangSmith에서 확인한 commit hash를
사용해야 연구 실행을 재현할 수 있습니다.

```json
{
  "prompt_name": "sokoban-strategy",
  "prompt_commit": "731d3f516b225cc0e1d11b87cfc5abe45c1a92ed63b41ef3e23796e805006b77",
  "model_name": "qwen3.6:27b-mlx",
  "rationale_mode": "on",
  "grounding_mode": "local-search",
  "memory_mode": "episode",
  "memory_namespace": "default"
}
```

`memory_mode=episode`은 현재 run의 실패 push만 checkpoint state에
기억합니다. `shared`는 LangGraph Store에서 동일한 보드 구조 분석, 환경으로
효과가 확인된 전략과 선형 재검증을 통과한 접지 경로를 thread 사이에
재사용합니다. 연구 비교는 정보 누설을 막기 위해 `off`로 실행하며, 공유
메모리는 prompt commit·model·실행 모드와 `memory_namespace`가 같은 경우에만
적중합니다.

위 commit은 private `sokoban-strategy` prompt의 실제 모델 종단간 검증에
사용한 고정 버전입니다. 새 prompt 버전을 연구에 사용할 때는 먼저 같은
검증을 통과한 뒤 hash를 명시적으로 교체합니다.

실시간 관찰 화면은 prompt 이름, 모델과 실행 모드를 받아 Agent Server의
공식 per-run `context`로 전달합니다. prompt commit은 사용자가 입력하지
않아도 됩니다. 화면은 `latest` selector를 전달하고 `resolve_prompt` node가
실행 시점의 실제 immutable commit으로 해석해 state에 기록합니다. 엄격한
재현성이 필요한 연구 배치와 CLI에서는 기존처럼 고정 commit을 직접
지정합니다.

각 단계에서 `board_analysis`, `prompt`, `strategy_input`,
`strategy_hypothesis`, `strategy_violations`, `active_subgoal`,
`protected_constraints`, `expected_effect`, `feedback`,
`grounded_actions`, `execution_result`, `reflection_result`,
`plan_revisions`, `action_history`, `decision_events`, `memory_requests`,
`memory_hits`, `memory_writes`, `llm_calls_saved`를 확인할 수 있습니다. prompt
본문과 숨은 추론 원문은 checkpoint에 저장하지 않습니다.

Studio 진입점은 `langgraph.json`과
`src/sokoban_agent/graph/agentic.py`에 정의되어 있습니다.
개발 기본 예시는 `LANGSMITH_TRACING=true`이며 LangGraph node와
`ChatLiteLLM` 호출, 구조화 응답 parser를 같은 LangSmith project에서
추적합니다. parser span에는 스키마 검증 오류가 필드 단위로 기록됩니다.
모델 입력과 출력을 외부에 남기면 안 되는 환경에서는 tracing을 끄거나
LangSmith input/output masking을 적용합니다.

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

### 구조화 문제 해결 일반화 실험

새 주 실험은 개발 fixture와 ID가 겹치지 않는
`benchmarks/boxoban_research_v1.json`을 기본으로 사용합니다. 공식 Boxoban의
`unfiltered`, `medium`, `hard`에서 5개씩 고정하고 원본 commit, 파일과
전체 level payload checksum을 검증합니다. 빠른 연결 점검은
`--cases-per-difficulty 1 --seeds 0`으로 난이도별 한 사례만 실행합니다.

다음 여섯 정책을 같은 레벨·seed grid에서 비교합니다.

- `primitive-llm`: 원시 방향 행동 LLM baseline
- `structured-llm`: 구조화된 단일 push를 직접 실행하며 국소 탐색 없음
- `structured-local-search`: 구조화된 단일 push와 지지 칸까지의 국소 탐색
- `structured-no-rationale`: 실행 필드는 유지하고 자연어 근거만 제거
- `current-full-guard`: 기존 LLM+A* 전체 대체 시스템 상한
- `astar-oracle`: 에피소드 외부의 bounded A* 사후 비교 기준

연구 실행은 검증된 immutable LangSmith prompt commit을 고정합니다.

```bash
uv run python scripts/run_agentic_research.py \
  --prompt-commit \
  731d3f516b225cc0e1d11b87cfc5abe45c1a92ed63b41ef3e23796e805006b77

uv run python scripts/run_agentic_research.py \
  --prompt-commit \
  731d3f516b225cc0e1d11b87cfc5abe45c1a92ed63b41ef3e23796e805006b77 \
  --cases-per-difficulty 1 --seeds 0
make agentic-notebook
```

실행기는 prompt commit, graph revision, 모델 설정, seed와 한도를 결과
manifest에 고정합니다. 결과에는 action trajectory, 하위 목표 성공·실패,
배정·가설 수정, 보호 제약 위반, 예상 효과 일치, 규칙·도달성·국소 탐색과
LLM·전역 알고리즘 비용이 분리되어 들어갑니다. rationale 제거의 기여는
문자열 유사도가 아니라 같은 사례의 정확한 action sequence와 성공 변화로
측정합니다.

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

## 실시간 Sokoban 관찰 화면

에이전트 실행과 동시에 플레이어·상자 이동 및 구조화된 전략 상태를 보려면
두 터미널에서 Agent Server와 Next.js·TypeScript 관찰 화면을 각각 실행한다.

```bash
# 터미널 1
make studio

# 터미널 2
make viewer
```

브라우저에서 <http://127.0.0.1:4173>을 열고 개발용 맵이나 공식 Boxoban
난이도별 연구 맵을 선택해 `실행`을 누른다. 기본값은 `sokoban_agent`,
`tiny-walk`, seed `0`, 최대 15행동이다. 맵을 바꾸면 검증된 bounded A*
행동 수를 기준으로 권장 행동 제한이 자동 설정되며 사용자가 수정할 수 있다.
화면의 일시정지는 표시 queue만 멈추며 실제 LangGraph 실행은 계속된다.
`한 단계`와 `최신으로` 버튼으로 대기 중인 state update를 탐색할 수 있다.

관찰 화면 자체 테스트는 다음처럼 실행한다.

```bash
cd viewer
npm test
```

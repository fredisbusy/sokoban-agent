# Sokoban Agent

정답 경로 없이 Sokoban을 관찰하고 계획하는 AI 에이전트를 단계적으로
연구하는 Python 프로젝트입니다.

현재 구현된 기능은 Gymnasium 환경, 고정·Boxoban 레벨 로더, 터미널 플레이,
Random/BFS 기준선, 공통 실행기와 측정 구조, 비교 노트북, LiteLLM 기반
Ollama 클라이언트와 구조화 출력 LLM Agent입니다. LLM Agent는 매 상태에서
JSON 행동 하나를 생성하고, 형식 오류와 막힌 행동을 제한 횟수 안에서
재시도합니다.
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

Ollama 없이도 환경과 터미널 플레이를 사용할 수 있습니다.

### 필요한 기능만 설치

기본 환경과 기준선만 설치하려면 개발 의존성을 제외합니다.

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
OLLAMA_API_BASE=http://localhost:11434
OLLAMA_MODEL=llama3.2
OLLAMA_TIMEOUT_SECONDS=120
```

Python에서는 다음처럼 텍스트 응답을 요청합니다. 이 클라이언트는 아직
독립적으로 사용할 수도 있습니다.

```python
from sokoban_agent.agents.llm import OllamaClient

client = OllamaClient.from_env()
answer = client.complete("다음 행동을 UP, DOWN, LEFT, RIGHT 중 하나로 답해줘.")
print(answer)
```

실제 보드 플레이에는 같은 클라이언트를 `LLMAgent`에 연결합니다.

```python
from sokoban_agent.agents import LLMAgent
from sokoban_agent.agents.llm import OllamaClient

client = OllamaClient.from_env()
agent = LLMAgent(client, model_name=client.settings.model, max_attempts=3)
```

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

LLM 비교 실험은 `.env`의 모델을 사용하며 다음 명령으로 재생성하고
실행합니다.

```bash
uv run --group notebook python scripts/build_llm_comparison_notebook.py
uv run --group notebook python -m jupyter nbconvert \
  --execute --to notebook --inplace notebooks/llm_agent_comparison.ipynb
```

실행된 비교 노트북은 모든 에피소드의 실제 상태·행동 trajectory를 함께
보관한다. 기본 애니메이션은 LLM의 `tiny-walk`, seed 0 경로를 재생하며,
노트북의 `ANIMATION_CASE`를 바꾸면 같은 실험의 다른 Agent·레벨·seed도
확인할 수 있다.

노트북은 실험과 시각화에만 사용하고, 재사용 코드는 `src/`에 둡니다.

## 프로젝트 구조

현재 기능이 들어 있는 경로는 다음과 같습니다.

```text
src/sokoban_agent/
├── env/               # 게임 규칙, 렌더링, 레벨 공급자
├── agents/            # Agent 계약과 Random/BFS/LLM 클라이언트
├── evaluation/        # 에피소드 실행, 결과와 집계
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

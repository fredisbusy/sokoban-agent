# Sokoban Agent

정답 경로 없이 Sokoban을 관찰하고 계획하는 AI 에이전트를 단계적으로
연구하는 Python 프로젝트입니다.

현재 구현된 기능은 Gymnasium 환경, 고정·Boxoban 레벨 로더, 터미널 플레이,
LiteLLM 기반 Ollama 텍스트 클라이언트입니다. Random/BFS 기준선, 실제로
보드를 푸는 LLM 에이전트, 공통 실행기와 측정 구조는 아직 구현되지 않았습니다.
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
Sokoban 보드나 행동 실행기와 연결되어 있지 않습니다.

```python
from sokoban_agent.agents.llm import OllamaClient

client = OllamaClient.from_env()
answer = client.complete("다음 행동을 UP, DOWN, LEFT, RIGHT 중 하나로 답해줘.")
print(answer)
```

## 연구와 검증

```bash
uv run pytest
uv run ruff check .
uv run mypy
```

`make test`, `make lint`, `make typecheck`는 같은 검사의 단축 명령입니다.
향후 노트북은 실험과 시각화에만 사용하고, 재사용 코드는 `src/`에 둡니다.

## 프로젝트 구조

현재 기능이 들어 있는 경로는 다음과 같습니다.

```text
src/sokoban_agent/
├── env/               # 게임 규칙, 렌더링, 레벨 공급자
├── agents/llm.py      # Ollama 설정과 텍스트 클라이언트
└── play.py            # 터미널 플레이
assets/levels/         # 저장소에 포함된 예제 레벨
scripts/               # Ollama 연결 확인
tests/                 # 환경, 레벨, 클라이언트, 플레이 검증
docs/                  # 목표와 아키텍처
```

## 문서

- [프로젝트 목표와 연구 범위](docs/PROJECT.md)
- [핵심 아키텍처와 실행 흐름](docs/ARCHITECTURE.md)
- [현재 우선순위와 완료 조건](TODO.md)
- [에이전트 작업 규칙](AGENTS.md)

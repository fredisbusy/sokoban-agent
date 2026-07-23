# Sokoban Agent

Sokoban을 관찰하고 계획하며 직접 행동하는 AI 에이전트를 연구하는 Python
프로젝트입니다. 정답 경로를 미리 주지 않고 다음 능력을 단계적으로 비교합니다.

- 보드 상태를 이해하고 상자를 목표까지 옮기는 계획
- 무효 이동과 데드락을 감지하고 다른 계획으로 재시도하는 능력
- Random/BFS 기준선과 LLM 에이전트의 성공률·행동 수·응답 시간
- 구조화된 보드에서 화면 기반 관찰로 확장할 수 있는지

목표 구조는 `관찰 → 계획 → 행동 검증 → 환경 실행 → 결과 기록`입니다.
게임 규칙은 Python 환경이 판정하고, LLM은 LiteLLM을 통해 Ollama 모델에
계획과 행동을 요청합니다. 현재 바로 사용할 수 있는 인터페이스는
`.env` 기반 Ollama 연결과 텍스트 생성이며, 개발 순서는 [TODO](TODO.md)에서
관리합니다.

## 빠른 시작

Python 3.11~3.13과 [uv](https://docs.astral.sh/uv/)가 필요합니다.

```bash
git clone git@github.com:fredisbusy/sokoban-agent.git
cd sokoban-agent
uv sync
cp .env.example .env
```

`.env`에 실행 중인 Ollama 서버와 모델을 지정합니다.

```dotenv
OLLAMA_API_BASE=http://<ollama-host>:11434
OLLAMA_MODEL=gemma4:26b
OLLAMA_TIMEOUT_SECONDS=120
```

연결을 확인합니다.

```bash
uv run python scripts/check_ollama.py
```

정상이라면 모델이 `연결 확인`처럼 짧은 응답을 반환합니다.

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
```

행동은 `0=UP`, `1=RIGHT`, `2=DOWN`, `3=LEFT`입니다. 관찰은 각 칸을
`FLOOR`, `WALL`, `TARGET`, `BOX`, `PLAYER`, `BOX_ON_TARGET`,
`PLAYER_ON_TARGET` 중 하나로 표현한 `uint8` 배열입니다.

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
올리지 않습니다.

```python
from sokoban_agent.env import BoxobanLevelProvider, SokobanEnv

levels = BoxobanLevelProvider("data/boxoban/unfiltered/train")
env = SokobanEnv(level_provider=levels, render_mode="ansi")

observation, info = env.reset(seed=42)
print(info["level_id"], env.render())
```

저장소에는 전체 데이터셋 없이도 파서와 환경을 확인할 수 있는
`assets/levels/boxoban_sample.txt`가 포함됩니다.

## Ollama를 Python에서 사용

```python
from sokoban_agent.agents.llm import OllamaClient

client = OllamaClient.from_env()
answer = client.complete("다음 행동을 UP, DOWN, LEFT, RIGHT 중 하나로 답해줘.")
print(answer)
```

## 연구와 검증

```bash
uv run jupyter lab  # 탐색과 시각화
make test           # pytest
make lint           # Ruff
make typecheck      # mypy
```

노트북은 실험과 시각화에 사용하고, 재사용 가능한 환경·에이전트 로직은
`src/`에 둡니다.

## 프로젝트 구조

디렉터리별 책임은 다음과 같습니다.

```text
src/sokoban_agent/
├── env/          # Gymnasium 환경, 게임 규칙, Boxoban 레벨 로더
├── agents/       # 기준선과 LLM 에이전트
├── perception/   # 보드·화면 인식
├── memory/       # 상태·계획·실패 기록
└── utils/        # 공통 도구
notebooks/        # 탐색과 결과 시각화
configs/          # 재현 가능한 실험 설정
scripts/          # 실행 진입점
tests/            # 동작 검증
assets/           # 레벨·이미지 등 정적 자료
```

## 문서

- [프로젝트 목표와 연구 범위](docs/PROJECT.md)
- [현재 우선순위와 완료 조건](TODO.md)
- [에이전트 작업 규칙](AGENTS.md)

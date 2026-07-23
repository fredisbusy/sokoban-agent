# Sokoban Agent

Sokoban을 플레이하는 AI 에이전트를 처음부터 구현하며 연구하기 위한 Python
프로젝트 뼈대입니다.

## 시작하기

```bash
uv sync
uv run jupyter lab
```

재사용 가능한 환경·에이전트 로직은 `src/`에, 탐색과 시각화는
`notebooks/`에 둡니다. 반복 가능한 실험 설정은 `configs/`, 실행 진입점은
`scripts/`, 검증 코드는 `tests/`에 둡니다.

현재는 라이브러리와 폴더 구조만 준비되어 있으며 Sokoban 환경이나 에이전트
구현은 포함하지 않습니다.

## 선택 가능한 기존 환경

- `gym-sokoban`: 빠른 참고용이지만 오래된 Gym API 기반이라 직접 의존하지 않습니다.
- MiniHack Boxoban: 대규모 벤치마크에 적합하지만 첫 구현에는 무겁습니다.
- Griddly Sokoban: 규칙 실험에는 유연하지만 별도 프레임워크 학습이 필요합니다.


# Boxoban LLM contribution baseline

`boxoban_pilot_v1.json`은 Google DeepMind의
[boxoban-levels](https://github.com/google-deepmind/boxoban-levels)에서
선택한 재현 가능한 LLM prefix 기여도 baseline 코호트다. 원본 라이선스는
Apache-2.0이다.

이 v1 코호트와 runner는 원시 행동 LLM과 A* Search Guard의 기존 결과를
보존한다. 구조화된 가설·하위 목표 에이전트의 일반화 실험은 v1 결과 형식을
변경하지 않고 별도 manifest와 runner로 추가한다. 새 주 정책의 실행 중에는
전역 A* 정답 suffix나 전체 대체를 사용하지 않으며, bounded A*는 에피소드
후 평가 oracle로만 호출한다.

- 원본 커밋과 파일 SHA-256을 고정한다.
- `boxoban-pilot-v1:<level_id>`의 SHA-256 오름차순으로 50개를 선택한다.
- 앞의 30개는 파일럿, 전체 50개는 확인 코호트로 사용한다.
- 서로 같은 보드가 선택되지 않았는지 로더가 검사한다.

데이터 준비와 실행:

```bash
uv run python scripts/prepare_boxoban_pilot.py --download
uv run python scripts/run_boxoban_pilot.py
```

실행 결과는 `_workspace/benchmarks/` 아래에 저장되며 Git에서 제외된다.
JSONL은 중간 결과와 재개 상태를, 같은 이름의 `.summary.json`은 정책별 집계를
담는다. 모델 비교에는 최소 30개 독립 레벨을 사용하고 한 레벨 스모크 결과를
성능 결론으로 사용하지 않는다.

## 구조화 에이전트 공식 난이도 코호트

`boxoban_research_v1.json`은 같은 원본 commit에서 Boxoban 공식 난이도
`unfiltered`, `medium`, `hard`를 각각 5개씩 고정한다. 각 원본 파일의
SHA-256과 인라인 보드 payload checksum을 모두 검증하며, 선정 규칙은
`sha256('boxoban-research-v1:' + difficulty + ':' + source_level_id)`
오름차순이다. 따라서 보기 좋은 맵만 사후 선별하지 않는다.

```bash
make boxoban-research-data
uv run python scripts/run_agentic_research.py \
  --prompt-commit <IMMUTABLE_LANGSMITH_COMMIT>
```

원본 3개 파일은 `data/boxoban/`에 내려받고 Git에서 제외한다. 연구 manifest의
15개 보드만 웹 선택과 정확한 실행 입력을 위해 출처·commit·Apache-2.0
라이선스와 함께 저장한다. bounded A* reference는 보드 무결성 및 사후 비교
기준이며 구조화 정책에 정답 경로로 전달하지 않는다.

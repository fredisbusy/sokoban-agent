# Boxoban LLM contribution pilot

`boxoban_pilot_v1.json`은 Google DeepMind의
[boxoban-levels](https://github.com/google-deepmind/boxoban-levels)에서
선택한 재현 가능한 평가 코호트다. 원본 라이선스는 Apache-2.0이다.

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

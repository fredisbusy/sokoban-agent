# 회귀 검증 결과

## 추가한 계약

- baseline episode v1은 정확히 57개 ordered column을 유지한다.
- 필수 10개 필드만 있는 additive v1 row를 읽고 canonical default를 채운다.
- 잘못된 boolean, unknown field, 불일치하는 파생값을 명시적으로 거절한다.
- `policy_elapsed_seconds`는 기존 `max(0, elapsed-reference)` 의미를 유지한다.
- agentic checkpoint는 schema v2를 기록하며 version이 없는 thread를 거절한다.
- `AgenticState` 최상위 필드 수는 15개 이하로 고정한다.
- viewer는 agentic/baseline decoder를 명시적으로 분리한다.

## 실행 결과

- `uv run pytest`: 147 passed
- `uv run ruff check .`: passed
- `uv run mypy`: 104 source files passed
- `viewer npm test`: 17 passed
- `viewer npm run typecheck`: passed
- `viewer npm run build`: passed
- `git diff --check`: passed

노트북 세 개는 generator-only 방식으로 재생성했으며 실제 모델 실행은 자동
회귀 게이트에 포함하지 않았다.

# 회귀 검증 계획

## 현재 호환성 위험

- pilot은 `asdict(EpisodeResult)`와 `EpisodeResult(**payload)`에 직접 의존한다.
- notebook은 `asdict(summary)` 컬럼을 직접 선택한다.
- research artifact는 record schema v2의 평면 컬럼과 N/A=`null` 의미를 갖는다.
- agentic history 세 채널은 `operator.add` reducer delta 의미를 갖는다.
- viewer는 `updates` stream을 얕게 누적하므로 nested group은 항상 완전한
  snapshot update여야 한다.

## 추가할 계약 테스트

1. episode v1 canonical writer key와 optional key가 빠진 additive v1 JSONL
   reader round trip
2. baseline summary exact ordered column set
3. research record/summary 전체 column set과 정책 순서
4. `null`과 측정값 `0` 구분
5. strict `JsonPlusSerializer` agentic checkpoint round trip
6. `feedback`, `plan_revisions`, `decision_events`가 resume 후 한 번씩만 누적
7. fresh thread와 same-thread resume 분리
8. agentic update golden fixture의 ViewEvent 전체 필드
9. notebook generator가 요청 컬럼을 실제 schema에서 찾는지 검증
10. v2 research artifact reader validation
11. old state schema checkpoint resume 명시적 거절
12. memory recall `Command` hit/miss update stream

## 단계별 게이트

| 단계 | 필수 검증 |
|---|---|
| P0/P1 | schema golden, additive v1 JSONL round trip, notebook column contract |
| P2 | agentic/baseline viewer decoder, `langgraph.json` 단일 graph 검사 |
| P3/P5 | strict checkpoint, reducer, same-thread resume, action stream/final state |
| P4 | baseline 집계 수치, 6정책 순서, artifact key diff |
| P6 | notebook regeneration diff, docs link 검사 |

모든 단계에서 다음 전체 게이트를 통과한다.

- `uv run pytest`
- `uv run ruff check .`
- `uv run mypy`
- `cd viewer && npm test`
- `cd viewer && npm run typecheck`
- `cd viewer && npm run build`
- `git diff --check`

`policy_elapsed_seconds` adapter 검증은 기존 `max(0, elapsed -
guard_reference_elapsed)` clamp 의미까지 고정한다.

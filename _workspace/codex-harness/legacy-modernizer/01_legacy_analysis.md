# 대형 클래스 필드 분석

## 결론

다섯 클래스 중 선언만 되고 생성되지 않는 필드는 없다. 평가 클래스의 많은
필드는 Python 코드에서 직접 읽히지 않더라도 JSONL, summary JSON, pandas
컬럼으로 소비된다. 따라서 단순 참조 횟수만으로 삭제하면 안 된다.

가장 명확한 제거 대상은 다음과 같다.

1. `StudioState` 전체
   - `langgraph.json`은 agentic graph만 등록한다.
   - Studio 전용 graph의 import 소비자는 `tests/test_studio.py`뿐이다.
   - 별도 LLM/A* 정책을 중복 구현하므로 상태를 재분류하기보다 graph와 함께
     제거하는 편이 맞다.
2. `AgenticState.failure_conditions`
   - `strategy_hypothesis`에서 최상위 state로 복사되지만 이후 어느 node도
     읽지 않는다.
   - 원본 `StrategyHypothesis.failure_conditions`에는 계속 남으므로 연구
     계약도 보존된다.

## `EpisodeResult`

57개 필드는 다음 책임을 혼합한다.

- identity: `planner_name`, `level_id`, `seed`
- outcome: terminal flags, action/push/invalid/reward, failure, 반복 상태
- planning usage
- algorithm usage
- LLM usage와 provider timing
- guard 기여도와 diagnostic reference
- 별도 bounded reference
- 전체 시간과 정책 시간

모든 필드는 baseline runner에서 생성되고 전체 `asdict()` 결과가 pilot JSONL과
노트북으로 나간다. 특히 `scripts/run_boxoban_pilot.py`는 저장한 dict를 다시
`EpisodeResult(**payload)`로 읽으므로 현재 shape는 resume 계약이다.

### 내부 저장에서 파생으로 전환할 후보

- `action_overhead_vs_reference`
- `push_overhead_vs_reference`
- `policy_elapsed_seconds`

이 값들은 outcome, reference, guard diagnostic time에서 결정론적으로 계산된다.
외부 v1 row에는 adapter property로 같은 키를 계속 제공할 수 있다.

### v2 artifact에서 삭제를 검토할 후보

- `total_reward`
  - 현재 프로젝트 범위와 성공 기준에서 사용하지 않고 집계에도 포함되지 않는다.
- `guard_reference_action_count`
  - planning outcome에서 graph metrics와 episode raw row까지 생산되지만
    summary나 연구 record로는 투영되지 않는다. 제거 여부는 P0에서 이
    raw diagnostic의 의미와 후속 분석 수요를 확인한 뒤 결정한다.

다음은 직접 소비가 적어도 재현성과 비용 분석 원시값이므로 유지한다.

- `llm_load_seconds`, `llm_prompt_eval_seconds`, `llm_eval_seconds`
- `guard_reference_elapsed_seconds`
- `reference_expanded_states`, `reference_elapsed_seconds`

## `PlannerSummary`

54개 필드는 outcome, action, latency distribution, planning, algorithm, LLM,
guard, reference 집계를 혼합한다. truly unused 필드는 없다. 전체 shape가
summary JSON과 DataFrame 컬럼이다.

다음은 저장할 원천값이 아니라 계산 property로 바꿀 수 있다.

- `success_rate`
- `deadlock_rate`
- `guard_adoption_rate`

`p50`/`p95`, 성공 episode 평균, provider timing 평균은 원시 episode 없이
복원할 수 없으므로 summary 값으로 유지한다.

## `ResearchPolicySummary`

20개 필드는 outcome, strategy quality, LLM/memory, rule/search cost, timing을
혼합한다. 모든 값이 artifact와 notebook에 노출되므로 즉시 삭제할 필드는
없다.

다음은 materialized 계산값으로 분류한다.

- `success_rate`
- `subgoal_success_rate`
- `effect_match_rate`
- `total_llm_tokens`

내부에서는 책임별 aggregate를 조합하고 flat summary adapter에서 기존 컬럼을
만드는 것이 기존 `ResearchEpisodeRecord.to_flat_dict()`와 일관된다.

## `AgenticState`

45개 필드는 case/runtime, environment, strategy, grounding, execution,
reflection, memory, metrics, history channel을 혼합한다.

### 중복 저장

`strategy_hypothesis`가 이미 아래 값을 포함하지만 verify node가 최상위에 다시
복사한다.

- `active_subgoal`
- `protected_constraints`
- `expected_effect`
- `failure_conditions`

앞의 세 값은 grounding/execution/viewer에서 실제 사용하고 마지막 값만
write-only다. 장기적으로는 검증된 hypothesis를 canonical source로 삼아 네
복제 필드를 모두 없애고 typed accessor/adapter가 하위 값을 읽게 한다.

추가 파생 후보:

- `grounded_actions`: `grounded_plan.player_actions + push_action`
- `strategy_memory_hit`, `grounding_memory_hit`: recall 직후 routing에만 쓰는
  transient boolean이므로 recall node의 `Command(goto=...)`로 대체 가능
- `info.success`, `info.deadlock`, execution `truncated`, `status`의 terminal
  의미: typed status를 canonical source로 정리

### 유지할 필드

- `level_rows`, `level_sha256`: graph 실행 중에는 write-only지만 catalog
  provenance와 checksum 검증 계약이다.
- `strategy_schema_issues`: routing에는 error code만 필요하지만 구조화된 모델
  교정 및 trace 계약이다.
- `decision_events`, `feedback`, `plan_revisions`: `operator.add` reducer가
  붙은 checkpoint history다.
- `board_analysis`: observation에서 재계산 가능해도 logical ID 안정성과
  checkpoint 관찰을 위해 필요하다.
- `execution_result`, `reflection_result`: memory와 viewer가 소비한다.

## `StudioState`

현재 활성 graph가 아니다. 전체 제거가 승인되기 전 개별 필드만 본다면 다음이
write-only 또는 파생 가능하다.

- write-only: `proposed_plan`, `validation_summary`, `execution_summary`,
  `llm_elapsed_seconds`, `algorithm_fallbacks`,
  `algorithm_expanded_states`, `algorithm_elapsed_seconds`
- 파생 가능: `board`, `action_count`, terminal booleans
- test/표시 전용: `decision_summary`, `decision_log`, `planner_goal`, `risk`,
  `guard_summary`

이 목록을 미세 조정하는 것보다 `graph/studio/`와 `tests/test_studio.py`를
제거하고 실제 agentic/baseline state adapter를 사용하는 것이 우선이다.

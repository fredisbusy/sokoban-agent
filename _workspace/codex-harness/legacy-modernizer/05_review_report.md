# 최종 교차 검토

## 반영한 필수 수정

- `guard_reference_action_count`는 생산 경로가 있는 raw diagnostic임을
  명시하고, 단순 미사용 삭제가 아니라 P0 의미 검토 대상으로 낮췄다.
- viewer의 baseline decoder를 선택 사항이 아닌 기존 회귀 계약으로 확정했다.
- artifact writer의 canonical key와 additive v1 reader의 누락 optional key
  허용을 분리했다.
- old checkpoint를 실제로 차단할 state version guard 또는 versioned
  graph/assistant identity를 완료 조건에 추가했다.
- 실행 순서를 adapter 선행, Studio 제거, state 타입 강화, 평가 합성,
  checkpoint-breaking state 중첩 순으로 통일했다.

## 타당성 확인

- `StudioState` 전체 제거 방향은 현재 `langgraph.json` 진입점과 import graph에
  부합한다.
- `AgenticState.failure_conditions` 최상위 복제 제거 후에도 canonical
  hypothesis에 실패 조건이 남는다.
- memory recall의 hit flag를 `Command(update=..., goto=...)`로 대체하는 것은
  update와 routing을 함께 수행하는 경우에만 Command를 쓴다는 프로젝트 원칙과
  부합한다.
- `policy_elapsed_seconds`는 adapter property로 파생 가능하며 기존 `max(0)`
  의미를 golden test로 보존해야 한다.

## 남은 의사결정

구현 P0에서 다음 두 raw metric을 v2 artifact에서 계속 제공할지 확정한다.

- `total_reward`
- `guard_reference_action_count`

둘 다 v1 reader/writer 호환과 별개인 명시적 v2 schema 결정으로 다룬다.

# 책임 분리 전략

## 원칙

1. 평가 내부 객체는 책임별 값 객체로 합성한다.
2. JSONL, summary JSON, pandas의 평면 shape는 versioned adapter가 소유한다.
3. LangGraph state는 모든 필드를 기계적으로 한 dict에 넣지 않는다.
   reducer 이력과 routing hot channel은 최상위에 유지한다.
4. checkpoint schema가 바뀌면 새 graph revision과 새 thread를 사용한다.
5. 현재 활성 graph가 아닌 `StudioState`는 재설계하지 않고 제거한다.

## 목표 구조

### Baseline episode

```text
BaselineEpisodeResult
├─ identity: PlannerEpisodeIdentity
├─ outcome: BaselineEpisodeOutcome
├─ planning: PlanningUsage
├─ algorithm: AlgorithmUsage
├─ llm: BaselineLLMUsage
├─ guard: GuardUsage
├─ reference: ReferenceComparison
└─ timing: EpisodeTiming
```

flat `BaselineEpisodeRowV1`이 기존 57개 JSONL 키를 읽고 쓴다. 내부 결과
객체에는 flat alias를 장기간 남기지 않는다.

### Baseline planner summary

```text
PlannerSummary
├─ planner_name
├─ sample: EpisodeSampleSummary
├─ outcome: OutcomeSummary
├─ actions: ActionSummary
├─ timing: LatencySummary
├─ planning: PlanningSummary
├─ algorithm: AlgorithmSummary
├─ llm: LLMSummary
├─ guard: GuardSummary
└─ reference: ReferenceSummary
```

`PlannerSummaryRowV1.to_flat_dict()`가 기존 notebook/summary 컬럼을 보존한다.

### Research policy summary

```text
ResearchPolicySummary
├─ policy_name
├─ sample
├─ outcome
├─ strategy
├─ llm
├─ memory
├─ rules
├─ local_search
├─ algorithm
└─ timing
```

`ResearchPolicySummary.to_flat_dict()`와 `summary_schema_version`을 추가한다.
현재 `ResearchEpisodeRecord`의 nested model + flat adapter 패턴을 그대로
따른다.

### Agentic graph state

```text
AgenticState
├─ state_schema_version
├─ case: AgenticCaseState
├─ runtime: AgenticRuntimeState
├─ observation
├─ info
├─ status: AgenticStatus
├─ action_history
├─ push_count
├─ cycle_detected
├─ analysis: AnalysisState
├─ strategy: StrategyState
├─ grounding: GroundingState
├─ execution_result: ExecutionResultState | None
├─ reflection_result: ReflectionResultState | None
├─ memory: AgenticMemoryState
├─ metrics: AgenticMetrics
├─ plan_revisions: Annotated[..., add]
├─ feedback: Annotated[..., add]
└─ decision_events: Annotated[..., add]
```

- reducer 세 채널은 최상위에 유지한다.
- `observation`, `info`, `status`, action/push는 실행 hot channel로 유지한다.
- 중첩 group update는 shallow merge하지 않고 항상 완전한 새 snapshot을
  반환한다.
- 먼저 `status`, execution/reflection/info payload를 TypedDict와 Literal로
  강화한 뒤 shape를 바꾼다.
- viewer, CLI, evaluation은 raw state를 직접 해석하지 않고 각 adapter만
  소비한다.

### Viewer

raw `GraphState = Record<string, unknown>` fallback chain을 명시적 decoder로
바꾼다.

- `decodeAgenticUpdateV1 -> ViewEventV1`
- `decodeBaselineUpdateV1 -> ViewEventV1`
- legacy Studio decoder는 제거한다.

`ViewEventV1`은 UI 관심사에 맞춰 strategy/effect/metrics 중첩 구조를 유지한다.

## 하지 않을 것

- 필드 수만 줄이기 위한 임의의 10개 단위 분할
- nested dict의 generic merge reducer
- 새 내부 API와 57개 property alias를 동시에 장기 유지
- 구 checkpoint를 묵시적으로 새 schema로 resume
- pandas 편의를 graph state나 domain model의 평면 구조로 전파

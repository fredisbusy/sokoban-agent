# LangGraph 중심 아키텍처

이 프로젝트의 실행 코어는 LangGraph다. Gymnasium 환경은 게임 규칙과 상태
전이의 단일 기준이고, Planner는 상태를 직접 변경하지 않은 채 행동 계획만
제안한다.

## 실행 그래프

```mermaid
flowchart TD
    S["START · 환경 reset"] --> P["plan · Planner 호출"]
    P -->|"계획 있음"| V["validate · 순수 규칙 검사"]
    P -->|"복구 가능 오류"| P
    P -->|"재시도 소진"| E["END"]
    V -->|"유효"| X["execute · env.step"]
    V -->|"막힌 행동"| P
    V -->|"재시도 소진"| E
    X -->|"남은 계획"| V
    X -->|"새 계획 필요"| P
    X -->|"성공 · 데드락 · 제한"| E
```

`SokobanGraphState`에는 관찰, 환경 정보, 남은 계획, 행동 이력, 거절 피드백,
계획 시도 횟수와 평가 지표가 들어간다. `InMemorySaver`는 에피소드
`thread_id`별로 각 노드 이후 상태를 체크포인트한다.

## Planner 경계

모든 계획 방식은 같은 `Planner` Protocol을 구현한다.

- `RandomPlanner`: 한 행동을 표본 추출한다.
- `BFSPlanner`: 현재 상태에서 완전한 최단 행동열을 계산한다.
- `LLMPlanner`: JSON Schema로 한 행동을 제안한다.
- `SearchGuardPlanner`: 주 Planner의 제안 이후 상태를 BFS로 검사하고,
  막혔거나 풀 수 없으면 현재 상태의 BFS 계획으로 대체한다.

그래프는 Planner 종류를 알 필요가 없다. 알고리즘 Planner가 여러 행동을
반환하면 그래프가 각 행동을 다시 검증하며 순서대로 실행한다. LLM의 형식
오류나 막힌 행동은 상태의 feedback에 기록되고 `plan` 노드로 되돌아간다.

## 책임 경계

- `env/`: 레벨, 규칙, 상태 전이, 성공과 데드락 판정
- `planning/`: BFS·Random·LLM 계획 생성과 Ollama 연결
- `graph/`: 계획, 검증, 실행, 재시도, 체크포인트
- `evaluation/`: 동일한 그래프를 사용한 벤치마크, 집계, trajectory

평가 실행기는 별도 행동 루프를 구현하지 않는다. 반드시 `SokobanGraph`를
호출하므로 기준선과 LLM이 같은 검증·복구 정책을 통과한다.

## 다음 확장

장기 기억은 그래프 상태와 별도로 JSONL 또는 SQLite에 실패 상태를 저장한 뒤
`recall` 노드를 `plan` 앞에 추가한다. 다음 알고리즘 확장은 A* 휴리스틱과
상자 단위 부분 계획을 별도 Planner 또는 subgraph로 추가한다.

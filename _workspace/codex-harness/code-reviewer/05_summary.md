# 코드 구조 리뷰 요약

## 결론

도메인 분리(`env`, `planning`, `graph`, `evaluation`)와 주 agentic graph의
방향은 적절하다. 파일 크기도 모두 기준 이하다. 전면 재설계보다 다음 세
경계의 정리가 효과가 크다.

1. **Composition root와 수명주기**
   - 실제 model/prompt adapter와 provenance를 한 객체에서 소유한다.
   - compiled graph/store/runner를 experiment 또는 server scope로 올린다.
2. **Graph state 계약**
   - terminal routing, episode/thread 경계, status/payload 타입을 명시한다.
   - viewer와 evaluation은 같은 versioned state adapter를 사용한다.
3. **중복·검증 경계**
   - 사용되지 않는 Studio graph를 제거한다.
   - viewer tests/typecheck/build와 기본 설치 import smoke를 품질 게이트에 넣는다.

## 권장 순서

### 1단계: 즉시 수정

- effective model config를 generator에 결합하고 기록값을 실제 설정에서 파생
- initialize terminal route 추가
- viewer nested metrics adapter와 실제 state contract test 추가
- viewer test command 및 root check 통합
- manifest download path containment와 크기 제한

### 2단계: 구조 정리

- experiment-scoped runner/store 도입
- legacy Studio graph 제거
- thread와 episode 수명 계약 명시
- core/LLM dependency 경계 정리

### 3단계: 성능 후속

- A* reachability tree와 path reconstruction 분리
- Boxoban index/header scan 최적화
- graph recursion budget을 위상 기반 정책으로 교체

## 잔여 위험

- 실제 Ollama/LangSmith 네트워크 호출은 이번 리뷰에서 실행하지 않았다.
- viewer dependency가 설치되지 않아 typecheck/build는 실행하지 못했다.
- Agent Server가 localhost 밖에 노출될 경우 request limit과 tenant memory
  namespace는 별도 보안 완료 조건이 필요하다.

## 2026-07-24 LangGraph 발견성 결론

실행 코어는 충분히 LangGraph-first다. StateGraph가 전이를 소유하고, state
reducer, runtime context, retry policy, checkpointer/store를 프레임워크 방식으로
사용하며 CLI와 평가가 주 structured graph factory를 공유한다.

현재 불편의 원인은 LangGraph 오용이 아니라 topology 발견성이다. 세 graph
family가 공존하고, legacy graph가 현재 Studio entrypoint처럼 이름 붙어 있으며,
production provider 조립이 여러 파일에 흩어져 있다.

우선순위는 다음과 같다.

1. 실제 모델 선택과 provenance 불일치 수정
2. legacy Studio graph 제거 또는 baseline history로 명확히 이동
3. graph inventory와 단일 composition root 추가
4. structured planning port와 외부 adapter 분리
5. current/historical architecture 문서 분리와 import boundary 검사

추가 검증으로 graph 관련 표적 테스트 17개가 통과했다.

## 2026-07-24 구현 후 정정

위의 “세 graph family”는 구현 전 스냅샷의 표현이다. legacy Studio graph는
`47f2c8b`에서 제거되었고, 현재 실행 graph family는 구조화 agentic과 원시
행동 baseline 두 개다. 이번 변경에서 composition root, graph inventory,
effective model provenance, 초기 terminal routing과 experiment-scoped runner를
반영했다. manifest 경로·크기 방어와 planning port/adapter 파일 분리는 별도
후속 범위로 남긴다.

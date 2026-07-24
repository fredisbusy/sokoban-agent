# 보안 검토

## P1. 외부 manifest의 path가 data root 밖 쓰기를 허용한다

- 근거:
  - `scripts/prepare_boxoban_research.py:43-57, 125-143`은 manifest의
    `source.path`를 containment 검사 없이 destination에 결합한다.
  - 절대 경로와 `..` 경로를 거부하지 않고 다운로드 응답 전체를 메모리에 읽는다.
- 영향:
  - 신뢰하지 않는 manifest와 `--download`를 함께 쓰면 프로세스 권한 범위의
    파일 덮어쓰기와 메모리·디스크 고갈이 가능하다.
- 방향:
  - repository/commit/path schema를 검증하고 resolved destination이 resolved
    data root 아래인지 강제한다.
  - 크기 제한 스트리밍 다운로드, checksum 검증, 임시 파일의 원자적 이동으로
    다운로드 adapter를 분리한다.

## P2. Agent Server 입력과 shared memory가 서버 정책 경계가 아니다

- 근거:
  - `AgenticInput`은 `max_steps`, `level_rows`의 런타임 상한을 갖지 않는다.
  - `memory_mode`와 `memory_namespace`도 호출자가 직접 지정한다.
- 영향:
  - 서버를 localhost 밖에 노출하면 큰 보드·step으로 CPU, checkpoint, LLM 비용을
    키우거나 다른 호출자의 shared namespace를 오염시킬 수 있다.
- 방향:
  - Pydantic request schema에서 보드 면적, 상자 수, step, payload 크기를 제한한다.
  - tenant/deployment/assistant namespace는 인증 계층에서 주입하고 quota를 둔다.

## 확인 결과

- subprocess는 인자 배열을 사용해 shell injection 징후가 없다.
- pickle, eval, 동적 코드 실행과 추적된 실제 secret은 발견하지 못했다.
- viewer는 React escaping을 사용하며 확인 범위에서 직접 XSS 징후가 없다.

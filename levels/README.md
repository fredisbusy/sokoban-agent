# Level catalog

맵 원본은 두 종류만 관리한다.

- Boxoban 연구 맵: `benchmarks/boxoban_research_v1.json`
- 직접 만든 맵: `levels/custom/*.json`

두 원본은 `scripts/build_level_catalog.py`가
`src/sokoban_agent/data/level_catalog.json`으로 합친다. 생성된 catalog가
Python 환경, LangGraph Agent Server와 Viewer가 함께 읽는 실행 기준이다.

## 사용자 맵 추가

`levels/custom/`에 JSON 파일 하나를 추가한다.

```json
{
  "schema_version": 1,
  "id": "custom/my-level",
  "revision": 1,
  "title": "My Level",
  "status": "draft",
  "difficulty": "custom",
  "rows": [
    "#####",
    "#@$.#",
    "#####"
  ],
  "recommended_max_steps": 30
}
```

`id`는 catalog 전체에서 고유해야 한다. 연구에 사용한 보드를 수정할 때는
기존 파일을 덮어써서 과거 실험을 바꾸지 말고 새 ID 또는 revision으로
추가한다.

추가 후 catalog를 생성하고 검증한다.

```bash
make levels
make levels-check
```

생성기는 타일, 직사각형, 닫힌 경계, 플레이어 수, 상자와 목표 수, ID와 보드
중복을 검사하고 보드 `sha256`을 계산한다. Viewer는 `level_id`와 `sha256`만
Agent Server에 보내며, 서버가 같은 catalog에서 실제 `rows`를 해석한다.

## 연구 코호트

맵 원본과 연구 대상 선택을 섞지 않는다. 코호트 manifest는 실행할 맵의
ID와 보드 checksum을 고정하는 실험 설정이며, 보드 원본은 catalog에 둔다.
현재 공식 난이도 코호트의 선정 근거와 A* reference는
`benchmarks/boxoban_research_v1.json`에 유지한다.

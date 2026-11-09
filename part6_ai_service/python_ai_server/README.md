# python_ai_server (P6-01~P6-04)

> 상태: **자리표시자** — 실습 내용은 해당 주차에 작성합니다.

| 항목 | 값 |
|---|---|
| 실습 ID | P6-01~P6-04 |
| 구분 | 메인 |
| 주차 | W16–19 |
| 소속 | [Part 6 — AI 서비스](../README.md) |

## 학습 목표

- `/api/v1/documents`, `embeddings`, `search`, `rag/query`, `graphrag/query`, `health`를 구현한다
- SSE 스트리밍, 외부 API 타임아웃·재시도를 구현한다
- 요청별 토큰 사용량·비용·지연을 로그로 남긴다

## 서비스 관점

외부 API 장애·지연이 곧 서비스 장애다. 타임아웃·재시도·비용 로그가 운영의 기본이다.

## 작성 예정

- `README.md` 12항목 + 서비스 관점: 목표 · 선수지식 · 핵심개념 · 내부원리 · 서비스 관점 · 수식/그림 · 실행법 · TODO 목록 · 예상 결과 · 자주 나는 오류 · 과제 · 체크리스트 · 다음 단계 연결
- `starter.py` (TODO 3~6개 + 개념/HINT/WHY/서비스 관점/REF 주석), `solution.py`, `tests/`
- 규약: [ARCHITECTURE §3–4](../../docs/ARCHITECTURE.md#3-실습-파일-규약-startersolutiontest)

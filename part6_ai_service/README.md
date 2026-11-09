# Part 6 — AI 서비스

> 상태: **골격만 있음** — 실습 코드는 아직 작성하지 않았습니다. 진행 상태는 [PROGRESS](../docs/PROGRESS.md).

> 방향: **개념은 깊게, 구현은 얕게 — 서비스 설계 중심**([PROJECT_PLAN](../docs/PROJECT_PLAN.md)).

Python AI 서버(FastAPI)와 이를 호출하는 Spring Boot 서버를 만들고, 스트리밍·타임아웃·재시도·로깅·비용 추적·Docker까지 운영 요소를 갖춘 RAG vs GraphRAG 비교 웹 서비스로 마무리한다.

- 기간: W16–21
- 완료 조건: 사용자 → Spring Boot → Python AI Server → RAG/GraphRAG → VectorDB/Neo4j → Claude API 흐름의 비교 웹 서비스(Docker Compose로 실행).

## 실습 목록

| 디렉터리 | 실습 ID | 구분 | 주차 |
|---|---|---|---|
| [python_ai_server](./python_ai_server/README.md) | P6-01~P6-04 | 메인 | W16–19 |
| [spring_boot_server](./spring_boot_server/README.md) | P6-02~P6-04 | 메인 | W17–19 |

주차별 세부 계획: [LEARNING_ROADMAP](../docs/LEARNING_ROADMAP.md)

# Part 4 — GraphRAG

> 상태: **골격만 있음** — 실습 코드는 아직 작성하지 않았습니다. 진행 상태는 [PROGRESS](../docs/PROGRESS.md).

> 방향: **개념은 깊게, 구현은 얕게 — 서비스 설계 중심**([PROJECT_PLAN](../docs/PROJECT_PLAN.md)).

지식 그래프를 만들고 Cypher·Graph+Vector 검색으로 질의한 뒤, 멀티홉 질문에서 Vector RAG와 비교해 "언제 GraphRAG가 필요한가"를 판단한다.

- 기간: W11–13
- 완료 조건: Vector RAG vs GraphRAG를 멀티홉 질의 세트로 비교하고, 구축 비용·질의 지연·정답률을 근거로 선택 기준을 정리한다.

## 실습 목록

| 디렉터리 | 실습 ID | 구분 | 주차 |
|---|---|---|---|
| [knowledge_graph](./knowledge_graph/README.md) | P4-01 | 메인 | W11 |
| [entity_extraction](./entity_extraction/README.md) | P4-02 | 메인 | W11 |
| [relationship_extraction](./relationship_extraction/README.md) | P4-03 | 메인 | W11 |
| [graph_retrieval](./graph_retrieval/README.md) | P4-04 | 메인 | W12 |
| [evaluation](./evaluation/README.md) | P4-05 | 메인 | W13 |
| [graph_traversal](./graph_traversal/README.md) | P4-S1 | 선택 | — |

주차별 세부 계획: [LEARNING_ROADMAP](../docs/LEARNING_ROADMAP.md)

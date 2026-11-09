# Part 3 — RAG

> 상태: **골격만 있음** — 실습 코드는 아직 작성하지 않았습니다. 진행 상태는 [PROGRESS](../docs/PROGRESS.md).

> 방향: **개념은 깊게, 구현은 얕게 — 서비스 설계 중심**([PROJECT_PLAN](../docs/PROJECT_PLAN.md)).

문서를 청크로 나누고 Dense(OpenAI 임베딩)·BM25·Hybrid·Reranker로 검색한 뒤, 직접 설계한 평가 세트로 품질·지연·비용을 비교한다.

- 기간: W7–10
- 완료 조건: 동일 데이터셋에서 BM25 vs Dense vs Hybrid vs Hybrid+Rerank를 IR 지표·지연·비용으로 비교하고, 직접 구현한 Hybrid를 프레임워크 구현과 비교한다.

## 실습 목록

| 디렉터리 | 실습 ID | 구분 | 주차 |
|---|---|---|---|
| [document_loader](./document_loader/README.md) | P3-01 | 메인 | W7 |
| [chunking](./chunking/README.md) | P3-02 | 메인 | W7 |
| [dense_retrieval](./dense_retrieval/README.md) | P3-03 | 메인 | W8 |
| [sparse_retrieval](./sparse_retrieval/README.md) | P3-04 | 메인 | W8 |
| [hybrid_search](./hybrid_search/README.md) | P3-05 | 메인 | W9 |
| [reranking](./reranking/README.md) | P3-06 | 메인 | W9 |
| [evaluation](./evaluation/README.md) | P3-07 | 메인 | W10 |

주차별 세부 계획: [LEARNING_ROADMAP](../docs/LEARNING_ROADMAP.md)

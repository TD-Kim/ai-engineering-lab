# Part 3 — RAG

> 상태: **골격만 있음** — 실습 코드는 아직 작성하지 않았습니다. 진행 상태는 [PROGRESS](../docs/PROGRESS.md).

검색 증강 생성의 검색 단계를 직접 구현하고, 검색 방식별 성능을 같은 데이터셋에서 수치로 비교한다.

- 기간: W11–15
- 완료 조건: 동일 데이터셋에서 BM25 vs Dense vs Hybrid vs Hybrid+Rerank를 IR 지표·Latency로 비교한다.

## 실습 목록

| 디렉터리 | 실습 ID | 난이도 | 주차 |
|---|---|---|---|
| [document_loader](./document_loader/README.md) | P3-01 | L1 | W11 |
| [chunking](./chunking/README.md) | P3-02 | L1 | W11 |
| [dense_retrieval](./dense_retrieval/README.md) | P3-03 | L2 | W12 |
| [sparse_retrieval](./sparse_retrieval/README.md) | P3-04 | L2 | W13 |
| [hybrid_search](./hybrid_search/README.md) | P3-05 | L2–L3 | W14–15 |
| [reranking](./reranking/README.md) | P3-06 | L2 | W14 |
| [evaluation](./evaluation/README.md) | P3-07 | L3 | W15 |

주차별 세부 계획: [LEARNING_ROADMAP](../docs/LEARNING_ROADMAP.md)

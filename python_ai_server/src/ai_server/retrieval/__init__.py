"""검색.

역할: BM25(키워드)·Dense(벡터)·Hybrid(결합)·Rerank(재정렬).
쓰이는 곳: /api/v1/search, RAG 파이프라인.
설계 포인트: 같은 평가 세트로 방식별 수치(Recall@K, MRR, NDCG, 지연)를 비교한다.
"""

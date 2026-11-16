"""GraphRAG.

역할: 지식 그래프 구축(엔티티·관계 추출 → Neo4j), 그래프 탐색, Graph+Vector 검색.
쓰이는 곳: /api/v1/graphrag/query.
설계 포인트: 멀티홉 질문에서 Vector RAG 보다 나은지 수치로 판단한다.
"""

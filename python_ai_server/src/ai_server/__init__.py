"""Python AI 서버.

최종 서비스 흐름: 사용자 → Spring Boot → **Python AI 서버** → RAG/GraphRAG → 벡터DB/Neo4j → LLM
하위 모듈은 기능별로 나눈다. 각 모듈의 역할은 그 모듈 __init__.py 에 적혀 있다.
"""

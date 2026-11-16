"""FastAPI 라우터.

역할: /api/v1/* 엔드포인트(documents, embeddings, search, rag/query, graphrag/query, health).
쓰이는 곳: Spring Boot 서버가 이 API 를 호출한다. 스트리밍은 SSE.
설계 포인트: 타임아웃·재시도, 오류 응답 형식, 요청별 로그.
"""

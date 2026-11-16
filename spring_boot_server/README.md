# Spring Boot 서버

최종 서비스 흐름: 사용자 → **Spring Boot** → Python AI 서버 → RAG/GraphRAG → 벡터DB/Neo4j → LLM

Python AI 서버의 `/api/v1/*`를 호출하고, 답변 스트리밍은 SSE로 사용자에게 중계합니다.
WebClient, DTO, 예외 처리, 타임아웃, 로깅을 다룹니다(기획서 §4 Part 6).

아직 비어 있습니다. Python AI 서버에 첫 API가 생기면 여기서 프로젝트를 만듭니다.

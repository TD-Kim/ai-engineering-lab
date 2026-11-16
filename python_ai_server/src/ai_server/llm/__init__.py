"""LLM 호출부 (공급자 어댑터).

역할: Claude API·로컬 모델(Ollama) 등 LLM 을 같은 모양의 함수로 부른다.
쓰이는 곳: RAG 답변 생성, GraphRAG 엔티티·관계 추출, /api/v1/rag/query 의 스트리밍 응답.
설계 포인트: 공급자는 환경 변수(LLM_PROVIDER, LLM_MODEL, OLLAMA_MODEL)로 바꾼다. 토큰 사용량을 기록한다.
"""

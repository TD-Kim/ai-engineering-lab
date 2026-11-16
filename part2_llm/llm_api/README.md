# llm_api (P2-02, P2-03)

Claude API를 서비스 코드로 다루는 개념 묶음입니다. 소속: [Part 2 — LLM 활용 원리](../README.md)

| 개념 README | 내용 | 최종 서비스에서 쓰이는 곳 |
|---|---|---|
| [basics](./basics/README.md) | 요청 본문(system·멀티턴), 응답 검사(stop_reason·블록·서버 측 폴백), 구조화 출력, 스트리밍, 토큰 세기, 오류 분류 | Python AI 서버의 Claude 호출부, 엔티티·관계 추출(Part 4), SSE(Part 6) |
| [cost_latency](./cost_latency/README.md) | 프롬프트 캐싱(접두어·breakpoint·usage 세 칸·깨짐 찾기), 배치(custom_id·50%·최대 24시간), 사용량 로그 집계(적중률·p95·총비용) | RAG 프롬프트 캐싱(Part 3), 색인·평가 배치, 비용 모니터링(Part 6) |

## 서비스 관점

요청당 토큰 수가 곧 비용과 지연이다. 캐싱·배치·스트리밍은 같은 품질에서 비용과 체감 지연을 줄이는 수단이다. 공급자 교체·폴백은 [P2-01B](../local_llm_serving/README.md)의 공통 인터페이스 설계를 쓴다.

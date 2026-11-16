# llm_api (P2-02, P2-03)

Claude API를 서비스 코드로 다루는 실습 묶음입니다. 소속: [Part 2 — LLM 활용 원리](../README.md)

| 디렉터리 | 실습 ID | 내용 | 상태 |
|---|---|---|---|
| [basics](./basics/README.md) | P2-02 | 요청 본문(system·멀티턴), 응답 검사(stop_reason·블록·서버 측 폴백), 구조화 출력, 스트리밍, 토큰 세기, 오류 분류 | 작성됨 |
| [cost_latency](./cost_latency/README.md) | P2-03 | 프롬프트 캐싱(접두어·breakpoint·usage 세 칸·깨짐 찾기), 배치(custom_id·50%·최대 24시간), 사용량 로그 집계(적중률·p95·총비용) | 작성됨 |

## 서비스 관점

요청당 토큰 수가 곧 비용과 지연이다. 캐싱·배치·스트리밍은 같은 품질에서 비용과 체감 지연을 줄이는 수단이다. 공급자 교체·폴백은 [P2-01B](../local_llm_serving/README.md)의 공통 인터페이스를 쓴다.

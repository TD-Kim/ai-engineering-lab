# Part 2 — LLM 활용 원리

> 방향: **개념은 깊게, 구현은 얕게 — 서비스 설계 중심**([PROJECT_PLAN](../docs/PROJECT_PLAN.md)). 학습 방식(만들고 싶은 것을 골라 직접 구현)은 [루트 README](../README.md#지금-어디를-보면-되나-학습-방식)를 보세요.

로컬 소형 모델로 다음 토큰 예측·샘플링을 눈으로 확인하고, Claude API로 토큰·프롬프트 구조·구조화 출력·스트리밍·캐싱과 비용을 다룬다. 마지막으로 프롬프트 vs RAG vs Fine-tuning 선택 기준을 정리한다.
이 폴더에는 **개념 README만** 남겼다(이전 실습 코드는 커밋 `aa46d2f`).

- 권장 기간: W5–6(강제 아님)
- 이 Part를 마쳤다고 볼 수 있는 기준: 같은 질문에 대해 "프롬프트만 / RAG / Fine-tuning" 중 무엇을 고를지 비용·지연·품질 근거로 설명할 수 있다.

## 개념 README

| 개념 | 최종 서비스에서 쓰이는 곳 |
|---|---|
| [next_token_prediction](./next_token_prediction/README.md) — 다음 토큰 예측·샘플링 | 답변 생성 설정, 로컬 모델 비교 |
| [local_llm_serving](./local_llm_serving/README.md) — 로컬 LLM 서빙, 공급자 교체(Claude·로컬·가짜), TTFT·VRAM | Python AI 서버의 LLM 어댑터(Part 6), 엔티티 추출 비교(Part 4) |
| [llm_api](./llm_api/README.md) — [basics](./llm_api/basics/README.md): Claude 요청·응답·구조화 출력·스트리밍 / [cost_latency](./llm_api/cost_latency/README.md): 캐싱·배치·사용량 집계 | LLM 호출부, 엔티티·관계 추출, SSE, 비용 추적 |
| [finetuning](./finetuning/README.md) | 프롬프트 vs RAG vs Fine-tuning 선택 기준 (작성 예정) |
| [loss](./loss/README.md) · [backpropagation](./backpropagation/README.md) · [training](./training/README.md) | 선택 주제(작성 예정) |

> P2-01B의 공급자 인터페이스(`LLMClient`)는 Part 4 엔티티·관계 추출의 로컬 LLM 비교와 Part 6 AI 서버에서 재사용하는 설계입니다.

> **순서 메모**: P2-01B·P2-02·P2-03은 임베딩([P1-02](../part1_ai_fundamentals/embedding/README.md)) 없이 이해할 수 있습니다. P2-01B README §2의 개념 선수지식(P2-01·P1-05의 prefill·decode·TTFT)을 먼저 읽고 오면 됩니다. **Part 3 전에는 임베딩 개념**을 알고 있어야 합니다. 전체 의존 표는 [루트 README](../README.md#지금-어디를-보면-되나-학습-방식).

주제별 확인 질문: [LEARNING_ROADMAP](../docs/LEARNING_ROADMAP.md)

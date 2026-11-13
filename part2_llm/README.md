# Part 2 — LLM 활용 원리

> 상태: **P2-01 작성됨**, 나머지는 순서대로 작성 예정. 진행 상태는 [PROGRESS](../docs/PROGRESS.md).
> 테스트 설정: [`conftest.py`](./conftest.py) (Part 1 과 같은 규약: `IMPL=starter`, TODO 미구현 → skip).

> 방향: **개념은 깊게, 구현은 얕게 — 서비스 설계 중심**([PROJECT_PLAN](../docs/PROJECT_PLAN.md)).

로컬 소형 모델로 다음 토큰 예측·샘플링을 눈으로 확인하고, Claude API로 토큰·프롬프트 구조·구조화 출력·스트리밍·캐싱과 비용을 다룬다. 마지막으로 프롬프트 vs RAG vs Fine-tuning 선택 기준을 정리한다.

- 기간: W5–6
- 완료 조건: 같은 질문에 대해 "프롬프트만 / RAG / Fine-tuning" 중 무엇을 고를지 비용·지연·품질 근거로 설명할 수 있다.

## 실습 목록

| 디렉터리 | 실습 ID | 구분 | 주차 |
|---|---|---|---|
| [next_token_prediction](./next_token_prediction/README.md) | P2-01 | 메인 | W5 |
| [local_llm_serving](./local_llm_serving/README.md) | P2-01B | 메인 | W5(–6) |
| [llm_api](./llm_api/README.md) | P2-02, P2-03 | 메인 | W5–6 |
| [finetuning](./finetuning/README.md) | P2-04 | 메인 | W6 |
| [loss](./loss/README.md) | P2-S1 | 선택 | — |
| [backpropagation](./backpropagation/README.md) | P2-S1 | 선택 | — |
| [training](./training/README.md) | P2-S1 | 선택 | — |

> P2-01B의 공급자 인터페이스(`LLMClient`)는 Part 4 엔티티·관계 추출의 로컬 LLM 비교와 Part 6 AI 서버에서 재사용한다.

주차별 세부 계획: [LEARNING_ROADMAP](../docs/LEARNING_ROADMAP.md)

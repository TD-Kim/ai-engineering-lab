# Part 5 — Fine-tuning

> 상태: **골격만 있음** — 실습 코드는 아직 작성하지 않았습니다. 진행 상태는 [PROGRESS](../docs/PROGRESS.md).

> 방향: **개념은 깊게, 구현은 얕게 — 서비스 설계 중심**([PROJECT_PLAN](../docs/PROJECT_PLAN.md)).

OpenAI 임베딩을 기준선으로 두고 로컬 임베딩·Reranker를 fine-tuning한 전후를 같은 평가 세트로 비교해, fine-tuning이 언제 값어치가 있는지 판단한다.

- 기간: W14–15
- 완료 조건: 동일 평가 세트에서 품질·비용·지연·운영 부담을 비교한 판단 정리(P5-03).

## 실습 목록

| 디렉터리 | 실습 ID | 구분 | 주차 |
|---|---|---|---|
| [embedding](./embedding/README.md) | P5-01, P5-03 | 메인 | W14–15 |
| [reranker](./reranker/README.md) | P5-02 | 메인 | W15 |
| [llm_lora](./llm_lora/README.md) | P5-S1 | 선택 | — |

주차별 세부 계획: [LEARNING_ROADMAP](../docs/LEARNING_ROADMAP.md)

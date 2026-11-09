# Part 1 — AI 원리

> 상태: **골격만 있음** — 실습 코드는 아직 작성하지 않았습니다. 진행 상태는 [PROGRESS](../docs/PROGRESS.md).

Tokenizer부터 Transformer까지 핵심 부품을 직접 구현하고, 작은 Transformer(미니 GPT)를 RTX 3070에서 학습한다.

- 기간: W2–6
- 완료 조건: 작은 Transformer(미니 GPT)를 직접 구현해 학습하고 손실 곡선을 해석한다.

## 실습 목록

진행 순서: **P1-04 PyTorch 기초를 선수 실습으로 맨 앞에 둔다**(이후 모든 실습이 Tensor·autograd·학습 루프를 쓰기 때문). 그 뒤는 기획서 순서 1-1 Tokenizer(P1-01) → 1-2 Embedding(P1-02) → 1-3 Word2Vec(P1-03) → 1-4 Transformer(P1-05).

| 디렉터리 | 실습 ID | 난이도 | 주차 |
|---|---|---|---|
| [pytorch](./pytorch/README.md) (선수) | P1-04 | L1–L3 | W2 |
| [tokenizer](./tokenizer/README.md) | P1-01 | L1–L3 | W3 |
| [embedding](./embedding/README.md) | P1-02 | L1 | W4 |
| [word2vec](./word2vec/README.md) | P1-03 | L2 | W4 |
| [transformer](./transformer/README.md) | P1-05a, P1-05b | L2–L3 | W5–6 |

주차별 세부 계획: [LEARNING_ROADMAP](../docs/LEARNING_ROADMAP.md)

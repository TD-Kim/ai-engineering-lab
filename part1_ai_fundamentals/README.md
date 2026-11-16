# Part 1 — AI 원리

> 방향: **개념은 깊게, 구현은 얕게 — 서비스 설계 중심**([PROJECT_PLAN](../docs/PROJECT_PLAN.md)). 학습 방식(만들고 싶은 것을 골라 직접 구현)은 [루트 README](../README.md#지금-어디를-보면-되나-학습-방식)를 보세요.

서비스 개발자가 RAG·GraphRAG를 설계하고 판단하는 데 필요한 만큼 원리를 익힌다.
토큰·임베딩·어텐션을 **비용·지연·품질** 관점에서 설명할 수 있게 하는 것이 목표다.
이 폴더에는 **개념 README만** 남겼다(정해진 실습·starter 방식은 폐지, 이전 실습 코드는 커밋 `aa46d2f`).

- 권장 기간: W2–4(강제 아님)
- 이 Part를 마쳤다고 볼 수 있는 기준: 토큰 수로 비용을 추정하고, 임베딩·코사인 유사도로 문장 검색을 설명하며, 컨텍스트 길이가 지연·비용에 주는 영향을 설명할 수 있다.

## 개념 README

| 개념 | 최종 서비스에서 쓰이는 곳 |
|---|---|
| [pytorch](./pytorch/README.md) (선수) — Tensor, autograd, `nn.Module`, DataLoader | 로컬 모델 구동, Part 5 fine-tuning |
| [tokenizer](./tokenizer/README.md) — 토큰 수, 한국어/영어 차이, 비용 추정, 토큰 기준 청킹, 컨텍스트 창 | 청크 크기(Part 3), 비용 계산·추적(Part 6) |
| [embedding](./embedding/README.md) — 정규화·코사인, top-k, OpenAI·로컬 임베딩 | Dense 검색(Part 3), GraphRAG 벡터 검색(Part 4) |
| [word2vec](./word2vec/README.md) — 분포 가설, Skip-gram·CBOW, 한계 | 임베딩 품질 판단(Part 3·5) |
| [transformer](./transformer/README.md) — 어텐션 계산, 구조, 컨텍스트 길이와 지연·비용, KV 캐시 | 프롬프트에 넣을 청크 수·캐싱 설계(Part 2·3), Reranker 이해 |

각 폴더의 `advanced/README.md`는 원리를 직접 구현해 볼 때 참고하는 심화 개념 노트입니다(BPE, 임베딩 층, Skip-gram, Self-Attention~미니 GPT).

공통:
- 말뭉치: [data/corpus_ko_en.txt](./data/corpus_ko_en.txt). 직접 작성한 한국어·영어 문장이며 다운로드가 필요 없습니다.
- 사용자가 직접 만든 코드: [pytorch/starter.py](./pytorch/starter.py), [tokenizer/starter.py](./tokenizer/starter.py)는 그대로 보존합니다.

> **순서 메모**: P1-02(임베딩)는 다른 개념 문서나 코드에 의존하지 않으므로 먼저 해도 되고 나중에 해도 됩니다. 단 **Part 3(RAG)를 만들기 전에는** 임베딩 개념을 알고 있어야 합니다(Dense 검색). 전체 의존 표는 [루트 README](../README.md#지금-어디를-보면-되나-학습-방식).

주제별 확인 질문: [LEARNING_ROADMAP](../docs/LEARNING_ROADMAP.md)

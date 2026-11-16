# Part 1 — AI 원리

> 진행 상태는 [PROGRESS](../docs/PROGRESS.md). 방향: **개념은 깊게, 구현은 얕게 — 서비스 설계 중심**([PROJECT_PLAN](../docs/PROJECT_PLAN.md)).

서비스 개발자가 RAG·GraphRAG를 설계하고 판단하는 데 필요한 만큼 원리를 익힌다.
토큰·임베딩·어텐션을 **비용·지연·품질** 관점에서 설명할 수 있게 하는 것이 목표다.
원리를 직접 구현하는 심화 과제는 각 실습의 `advanced/`에 **선택 과제**로 둔다.

- 기간: W2–4
- 완료 조건: 토큰 수로 비용을 추정하고, 임베딩·코사인 유사도로 문장 검색을 설명하며, 컨텍스트 길이가 지연·비용에 주는 영향을 설명할 수 있다.

## 실습 목록

진행 순서: **P1-04 PyTorch 기초는 선수 실습으로 맨 앞**(학습자가 이미 학습함). 그 뒤 Tokenizer(P1-01) → Embedding(P1-02) → Word2Vec(P1-03) → Transformer(P1-05).

| 디렉터리 | 실습 ID | 메인 실습 | 주차 | 선택 과제 |
|---|---|---|---|---|
| [pytorch](./pytorch/README.md) (선수) | P1-04 | Tensor, autograd, `nn.Module`, DataLoader | W2 | — |
| [tokenizer](./tokenizer/README.md) | P1-01 | 토큰 수, 한국어/영어 차이, 비용 추정, 토큰 기준 청킹, 컨텍스트 창 맞추기 | W2 | `advanced/`: BPE 직접 구현 |
| [embedding](./embedding/README.md) | P1-02 | 정규화·코사인, OpenAI 임베딩으로 문장 유사도·최근접 | W3 | `advanced/`: 임베딩 층·위치 임베딩 직접 구현 |
| [word2vec](./word2vec/README.md) | P1-03 | 분포 가설, gensim 짧은 실행 | W3 | `advanced/`: Skip-gram/CBOW·Negative Sampling 직접 구현 |
| [transformer](./transformer/README.md) | P1-05 | 어텐션 1회 계산, 구조, 컨텍스트 길이와 지연·비용, KV 캐시 | W4 | `advanced/`: Self-Attention~미니 GPT 직접 구현·학습 |

공통:
- 말뭉치: [data/corpus_ko_en.txt](./data/corpus_ko_en.txt). 직접 작성한 한국어·영어 문장이며 다운로드가 필요 없습니다.
- 테스트: 저장소 루트에서 `pytest`. `IMPL=starter`로 직접 채운 코드를 검사합니다([conftest.py](./conftest.py)).

> **순서를 바꿔도 되는 이유·조건 (P1-02)**: P1-02는 다른 실습의 코드를 불러오지 않고 P2-01B·P2-02·P2-03도 P1-02를 쓰지 않으므로, 로컬 LLM(P2-01B)을 먼저 하고 P1-02(로컬 임베딩 옵션)를 뒤에 해도 됩니다. 단 P1-03·P1-05는 P1-02의 코사인·내적을 개념 선수지식으로 쓰니 그 앞에 두는 것이 좋고, **Part 3 전에는 반드시 끝내야** 합니다. 전체 의존 표: [루트 README](../README.md#지금-어디를-보면-되나-학습-순서진행-안내).

주차별 세부 계획: [LEARNING_ROADMAP](../docs/LEARNING_ROADMAP.md)

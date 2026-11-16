# AI Engineering Learning Lab

AI 서비스 개발자로서 **RAG·GraphRAG 서비스를 설계하고 판단하는 능력**을 실습으로 기르는 저장소입니다.
원리는 자세한 설명으로 깊게 이해하고, 구현은 라이브러리·API로 얕게 합니다(원리 직접 구현은 `advanced/` 선택 과제).
최종적으로 Java/Spring Boot 경력과 결합한 **AI 서비스 포트폴리오**로 마무리합니다.

> 원칙: 개념은 깊게, 구현은 얕게(서비스 설계 중심) · LLM은 Claude API, 임베딩은 OpenAI API 기본 ·
> 로컬 모델은 개념 실습과 fine-tuning 비교에 · 모델 라이선스 확인 · "개선되었다"는 수치로만 말한다.

## 지금 어디를 보면 되나 (학습 순서·진행 안내)

### 1. 학습 순서

- **권장 순서(기획서)**: P1-04 PyTorch(선수) → P1-01 Tokenizer → P1-02 Embedding → P1-03 Word2Vec → P1-05 Transformer → Part 2(P2-01 → P2-01B → P2-02 → P2-03 → P2-04) → Part 3 …. 주차 일정은 [LEARNING_ROADMAP](docs/LEARNING_ROADMAP.md).
- **지금 택한 순서(2026-11-16)**: P1-04(이미 학습함) → P1-01(진행 중) → **로컬 LLM [P2-01B](part2_llm/local_llm_serving/README.md)** → **[P1-02 로컬 임베딩](part1_ai_fundamentals/embedding/README.md)**(OpenAI 키 없이 `EMBEDDING_PROVIDER`로 로컬 모델 사용) → 그다음은 정해지는 대로 [PROGRESS의 학습 순서 메모](docs/PROGRESS.md#학습-순서-메모)에 적습니다.
- 순서를 바꿔도 되지만 **의존 관계(아래 3)를 지켜야** 합니다. 특히 **Part 3(RAG)를 시작하기 전에는 P1-02를 끝내야** 합니다.

### 2. 실습으로 가는 링크 (상태는 [PROGRESS](docs/PROGRESS.md)에서 확인)

| 실습 | 폴더 README | 한 줄 설명 |
|---|---|---|
| P1-04 | [pytorch](part1_ai_fundamentals/pytorch/README.md) | Tensor·autograd·`nn.Module`·DataLoader (선수) |
| P1-01 | [tokenizer](part1_ai_fundamentals/tokenizer/README.md) | 토큰 수·비용·토큰 청킹 |
| P1-02 | [embedding](part1_ai_fundamentals/embedding/README.md) | 정규화·코사인·top-k, OpenAI 또는 로컬 임베딩 |
| P1-03 | [word2vec](part1_ai_fundamentals/word2vec/README.md) | 분포 가설, gensim |
| P1-05 | [transformer](part1_ai_fundamentals/transformer/README.md) | 어텐션·컨텍스트 길이·KV 캐시 |
| P2-01 | [next_token_prediction](part2_llm/next_token_prediction/README.md) | 다음 토큰 예측·샘플링(로컬 소형 모델) |
| P2-01B | [local_llm_serving](part2_llm/local_llm_serving/README.md) | Ollama 로컬 서빙, 공급자 교체 |
| P2-02 | [llm_api/basics](part2_llm/llm_api/basics/README.md) | Claude API 요청·응답·구조화 출력·스트리밍 |
| P2-03 | [llm_api/cost_latency](part2_llm/llm_api/cost_latency/README.md) | 프롬프트 캐싱·배치·사용량 집계 |
| P2-04 이후 | [part2_llm](part2_llm/README.md) · [part3_rag](part3_rag/README.md) · [part4_graphrag](part4_graphrag/README.md) · [part5_finetuning](part5_finetuning/README.md) · [part6_ai_service](part6_ai_service/README.md) | 작성 예정(골격) |

### 3. 의존 관계 — 무엇을 먼저 해야 하나

"코드 의존"은 실제로 다른 실습의 파일을 불러오는 경우(저장소 코드로 확인), "개념 선수지식"은 각 실습 README §2에 적힌 내용입니다. 코드 의존은 **그 실습의 `solution.py`를 불러오므로, 학습자가 그 실습을 풀지 않아도 동작**합니다.

| 실습 | 코드 의존 | 개념 선수지식 (각 README §2) | 순서를 바꿀 때 |
|---|---|---|---|
| P1-04 | 없음 | — | 맨 앞 |
| P1-01 | 없음 | 파이썬·UTF-8 | 어디서든 가능 |
| P1-02 | 없음 | P1-04(shape·브로드캐스팅), P1-01(토큰·비용 개념) | P2-01B 뒤에 해도 됨 |
| P1-03 | 데모가 P1-02의 `fake_embed`를 불러옴 | P1-02(코사인·top-k), P1-01 | P1-02 뒤 권장 |
| P1-05 | 없음 | P1-04, P1-02(내적·코사인), P1-01 | P1-02 뒤 권장 |
| P2-01 | Part 1 말뭉치 파일 | P1-05, P1-01, P1-04 | |
| P2-01B | 없음 | P2-01, P1-05(prefill·decode·TTFT), P1-01 | **P1-02 없이 가능.** 개념 선수지식(P2-01·P1-05)은 해당 README의 §3·§4를 먼저 읽고 오면 됩니다 |
| P2-02 | P2-01B의 `Usage`·`ProviderError`·`log_usage` | P1-01, P1-05(TTFT·캐싱 접두어 원리), P2-01B | P1-02 없이 가능 |
| P2-03 | P2-02(와 그 안의 P2-01B) | P1-01, P1-05(KV 캐시·같은 앞부분은 다시 계산하지 않는 원리), P2-01B, P2-02 | P1-02 없이 가능 |
| Part 3 (P3-01~07) | **(계획 — Part 3는 아직 골격)** P3-03/04에서 P1-02의 `top_k`와 `make_embed_fn`(공급자 교체)이 Dense 검색이 된다고 [P1-02 §12](part1_ai_fundamentals/embedding/README.md#12-다음-단계-연결)에 적혀 있음 | **P1-02 필수**, P1-01(토큰 청킹) 권장 | P1-02를 끝내지 않고 Part 3 시작 금지 |

> "개념 선수지식" 칸은 각 README §2 중 다른 실습에 대한 항목만 요약한 것입니다(파이썬 문법 등은 생략).
>
> 위 표의 "순서를 바꿀 때" 칸은 사용자가 택한 순서(P2-01B → P1-02)와 코드·README 확인에 근거한 것입니다. P2-01B가 개념으로 기대하는 P2-01·P1-05를 건너뛰는 선택은 학습자 결정이며, 막히면 그 README의 해당 개념 절을 먼저 읽으세요.

### 4. 어디에 무엇이 있나

| 알고 싶은 것 | 문서 |
|---|---|
| 실습별 진행 상태(`NOT_STARTED`/`IN_PROGRESS`/`COMPLETED`) | [docs/PROGRESS.md](docs/PROGRESS.md) |
| 주차별 일정·RTX 3070 가능 범위 | [docs/LEARNING_ROADMAP.md](docs/LEARNING_ROADMAP.md) |
| 작업 세션(총괄·작업·검증) 규칙·사용자 결정 이력 | [docs/ORCHESTRATION.md](docs/ORCHESTRATION.md) |
| VS Code + Claude Code로 학습하는 법 | [docs/ENVIRONMENT.md §7](docs/ENVIRONMENT.md#7-vs-code--claude-code로-학습하기) |

## 학습 환경

| 항목 | 값 |
|---|---|
| OS | Windows |
| GPU | NVIDIA RTX 3070 (VRAM 8GB) |
| RAM | 32GB |
| Python | 3.12 (저장소 자체 `.venv`, uv 관리) |
| 패키지 관리 | [uv](https://docs.astral.sh/uv/) |

설치 절차는 [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)를 보세요.

API 키는 저장소 루트 `.env`에만 둡니다(커밋 금지). 템플릿을 복사해 값을 채우세요.

```powershell
Copy-Item .env.example .env   # bash: cp .env.example .env
```

## 로드맵 한눈에 보기

방향: **개념은 깊게, 구현은 얕게 — 서비스 설계 중심.** 원리 직접 구현은 각 실습의 `advanced/` 선택 과제로 둔다.

| Part | 주제 | 주차 | 핵심 산출물 |
|---|---|---|---|
| 0 | 환경·기획 | W1 | 이 문서들, uv 환경 |
| 1 | [AI 원리](part1_ai_fundamentals/README.md) | W2–4 | 토큰·임베딩·어텐션을 비용·지연·품질로 설명 |
| 2 | [LLM 활용 원리](part2_llm/README.md) | W5–6 | 로컬 소형 모델 샘플링, Claude API(토큰·구조화 출력·스트리밍·캐싱) |
| 3 | [RAG](part3_rag/README.md) | W7–10 | BM25 vs Dense vs Hybrid vs Hybrid+Rerank 비교 + 지연·비용 |
| 4 | [GraphRAG](part4_graphrag/README.md) | W11–13 | "언제 GraphRAG가 필요한가" 멀티홉 비교 |
| 5 | [Fine-tuning](part5_finetuning/README.md) | W14–15 | 임베딩·Reranker fine-tuning이 언제 값어치가 있는가 |
| 6 | [AI 서비스](part6_ai_service/README.md) | W16–21 | FastAPI + Spring Boot, SSE·타임아웃·재시도·비용 추적·Docker, 비교 웹 서비스 |
| — | 포트폴리오 정리 | W22 | 실험 기록·회고·데모 |

자세한 주차별 계획은 [docs/LEARNING_ROADMAP.md](docs/LEARNING_ROADMAP.md)에 있습니다.

## 문서

| 문서 | 내용 |
|---|---|
| [PROJECT_PLAN](docs/PROJECT_PLAN.md) | 목적, 원칙, Part별 범위, 코드·문서 규칙 |
| [REPOSITORY_ANALYSIS](docs/REPOSITORY_ANALYSIS.md) | 참고 저장소(graphRAG_learning) 분석: 배울 점, Part별 참고, 다르게 할 점 |
| [LEARNING_ROADMAP](docs/LEARNING_ROADMAP.md) | 22주 일정(서비스 설계 중심), RTX 3070 가능/불가능 표, Hybrid 비교 계획 |
| [ARCHITECTURE](docs/ARCHITECTURE.md) | 디렉터리 구조, 서비스 아키텍처, starter/solution/test·힌트 주석 규약 |
| [ENVIRONMENT](docs/ENVIRONMENT.md) | uv 환경 구성, 라이브러리, 모델·VRAM·라이선스, GPU 검증, 트러블슈팅 |
| [PROGRESS](docs/PROGRESS.md) | 실습별 진행 상태, 주간 점검 템플릿 |

## 디렉터리

```text
ai-engineering-lab/
├── docs/                    # 기획·로드맵·아키텍처·환경·진행 문서
├── requirements/            # Part별 의존성 (base, dev, part1~6)
├── part1_ai_fundamentals/   # Tokenizer, Embedding, Word2Vec, PyTorch, Transformer
├── part2_llm/               # Next token, Loss, Backprop, Training, Fine-tuning 원리
├── part3_rag/               # 로딩, 청킹, Dense/Sparse/Hybrid, Rerank, 평가
├── part4_graphrag/          # KG, 엔티티·관계 추출, 그래프 탐색·검색, 평가
├── part5_finetuning/        # Embedding, Reranker, LLM LoRA/QLoRA
├── part6_ai_service/        # Python AI Server(FastAPI) + Spring Boot Server
├── datasets/                # 실습 데이터셋 설명 (대용량 원본은 커밋하지 않음)
├── experiments/             # 실험 기록
└── tests/                   # 공통 테스트
```

## 실습 진행 방식

각 실습은 `starter.py`(TODO + 힌트)를 직접 채우고, 막히면 `solution.py`와 비교합니다.
진행 상태는 [docs/PROGRESS.md](docs/PROGRESS.md)에서 관리하며,
**학습자가 직접 확인한 실습만 `COMPLETED`** 로 바꿉니다.

VS Code에서 Claude Code와 함께 학습할 때의 규칙은 [CLAUDE.md](CLAUDE.md), 준비 방법은 [ENVIRONMENT §7](docs/ENVIRONMENT.md#7-vs-code--claude-code로-학습하기)에 있습니다(정답 대신 단계별 힌트, 내 코드는 설명만).

## 참고

- 참고 저장소: [TD-Kim/graphRAG_learning](https://github.com/TD-Kim/graphRAG_learning) —
  강의 실습 코드이므로 **코드·데이터를 복사하지 않고 링크로만 참고**합니다.

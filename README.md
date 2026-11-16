# AI Engineering Learning Lab

AI 서비스 개발자로서 **RAG·GraphRAG 서비스를 설계하고 판단하는 능력**을 실습으로 기르는 저장소입니다.
원리는 자세한 설명으로 깊게 이해하고, 구현은 라이브러리·API로 얕게 합니다(원리 직접 구현은 선택, 심화 개념 노트는 각 폴더의 `advanced/README.md`).
최종적으로 Java/Spring Boot 경력과 결합한 **AI 서비스 포트폴리오**로 마무리합니다.

> 원칙: 개념은 깊게, 구현은 얕게(서비스 설계 중심) · LLM은 Claude API, 임베딩은 OpenAI API 기본 ·
> 로컬 모델은 개념 실습과 fine-tuning 비교에 · 모델 라이선스 확인 · "개선되었다"는 수치로만 말한다.

## 지금 어디를 보면 되나 (학습 방식)

> **2026-11-16 학습 방식 전환:** 정해진 실습 순서와 `starter.py` 채우기 방식은 폐지했습니다. **기획서(방향)만 두고, 만들고 싶은 것을 그때그때 골라 직접 구현**합니다. 막히면 VS Code(Claude Code)나 총괄 세션에 질문하세요.

### 1. 이렇게 진행합니다

1. **만들고 싶은 기능을 고른다.** (예: 문서 청킹, Dense 검색, LLM 어댑터, 엔티티 추출, FastAPI 엔드포인트)
2. **기획서의 어느 Part인지, 최종 서비스의 어디에 들어가는지 확인한다.** → [로드맵의 주제 지도](docs/LEARNING_ROADMAP.md), [최종 서비스 구조](docs/ARCHITECTURE.md#1-최종-서비스-구조)
3. **개념 README를 읽는다.** (아래 2)
4. **직접 구현한다.** 코드는 [권장 위치](docs/ARCHITECTURE.md#4-코드-권장-위치-제안)에 둡니다(제안, 강제 아님).
5. **막히면 질문한다.** VS Code의 Claude Code는 [CLAUDE.md](CLAUDE.md) 규칙대로 "이것이 최종 서비스의 어디에 쓰이는지"부터 짚고, 정답 대신 방향 → API 이름 → 예시 순 힌트를 줍니다.
6. **만든 것을 기록한다.** → [docs/PROGRESS.md](docs/PROGRESS.md) "내가 만든 것 기록"

순서는 권장일 뿐입니다. 다만 **Part 3(RAG)를 만들기 전에는 임베딩 개념을 알고 있어야** 합니다(아래 3).

### 2. 개념 README

| 개념 | 폴더 README | 최종 서비스에서 쓰이는 곳 |
|---|---|---|
| PyTorch 기초 (선수) | [pytorch](part1_ai_fundamentals/pytorch/README.md) | 로컬 모델 구동, Part 5 fine-tuning |
| Tokenizer | [tokenizer](part1_ai_fundamentals/tokenizer/README.md) | 청크 크기, 비용 계산·추적 |
| Embedding (OpenAI·로컬) | [embedding](part1_ai_fundamentals/embedding/README.md) | Dense 검색, GraphRAG 벡터 검색 |
| Word2Vec | [word2vec](part1_ai_fundamentals/word2vec/README.md) | 임베딩 품질 판단 |
| Transformer | [transformer](part1_ai_fundamentals/transformer/README.md) | 컨텍스트 길이 비용·캐싱 설계, Reranker 이해 |
| 다음 토큰 예측·샘플링 | [next_token_prediction](part2_llm/next_token_prediction/README.md) | 생성 설정 |
| 로컬 LLM 서빙·공급자 교체 | [local_llm_serving](part2_llm/local_llm_serving/README.md) | LLM 어댑터(Part 6), 엔티티 추출 비교(Part 4) |
| Claude API 기초 | [llm_api/basics](part2_llm/llm_api/basics/README.md) | Claude 호출부, 구조화 출력, SSE |
| 비용·지연(캐싱·배치) | [llm_api/cost_latency](part2_llm/llm_api/cost_latency/README.md) | 프롬프트 캐싱, 배치, 비용 모니터링 |
| 심화 노트(선택) | [tokenizer](part1_ai_fundamentals/tokenizer/advanced/README.md) · [embedding](part1_ai_fundamentals/embedding/advanced/README.md) · [word2vec](part1_ai_fundamentals/word2vec/advanced/README.md) · [transformer](part1_ai_fundamentals/transformer/advanced/README.md) `advanced/` | 원리를 직접 구현해 볼 때 참고 |
| Part 3~6 주제(자리표시자) | [part3_rag](part3_rag/README.md) · [part4_graphrag](part4_graphrag/README.md) · [part5_finetuning](part5_finetuning/README.md) · [part6_ai_service](part6_ai_service/README.md) | 만들 때 개념 README를 채움 |

### 3. 의존 관계 — 무엇을 알고 시작하면 좋은가

"개념 선수지식"은 각 README §2에서 다른 문서에 대한 항목만 요약한 것입니다(파이썬 문법 등은 생략).

| 개념 | 개념 선수지식 | 순서를 바꿀 때 |
|---|---|---|
| pytorch | — | 맨 앞 |
| tokenizer | 파이썬·UTF-8 | 어디서든 가능 |
| embedding | pytorch(shape·브로드캐스팅), tokenizer(토큰·비용) | 다른 개념 없이도 가능 |
| word2vec | embedding(코사인·top-k), tokenizer | embedding 뒤 권장 |
| transformer | pytorch, embedding(내적·코사인), tokenizer | embedding 뒤 권장 |
| next_token_prediction | transformer, tokenizer, pytorch | |
| local_llm_serving | next_token_prediction, transformer(prefill·decode·TTFT), tokenizer | embedding 없이 가능 |
| llm_api/basics | tokenizer, transformer(TTFT·캐싱 접두어 원리), local_llm_serving | embedding 없이 가능 |
| llm_api/cost_latency | tokenizer, transformer(KV 캐시), local_llm_serving, llm_api/basics | embedding 없이 가능 |
| **Part 3 (RAG)** | **embedding 필수**, tokenizer(토큰 청킹) 권장. [embedding §9](part1_ai_fundamentals/embedding/README.md#9-다음-단계-연결)는 Part 3 Dense 검색이 `top_k`와 공급자 교체 구조를 이어 쓴다고 설명 | embedding 개념 없이 Part 3 시작 금지 |

### 4. 어디에 무엇이 있나

| 알고 싶은 것 | 문서 |
|---|---|
| 주제 지도·각 주제에서 확인할 질문·RTX 3070 가능 범위·Hybrid 비교 계획 | [docs/LEARNING_ROADMAP.md](docs/LEARNING_ROADMAP.md) |
| 최종 서비스 구조·코드 권장 위치 | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| 내가 만든 것 기록·주간 점검 | [docs/PROGRESS.md](docs/PROGRESS.md) |
| VS Code + Claude Code로 학습하는 법 | [docs/ENVIRONMENT.md §7](docs/ENVIRONMENT.md#7-vs-code--claude-code로-학습하기) |
| 학습 도우미 규칙 | [CLAUDE.md](CLAUDE.md) |
| 세션 운영(총괄 ①·질문·기획 점검)과 이전 방식 기록 | [docs/ORCHESTRATION.md](docs/ORCHESTRATION.md) |

### 5. 이전 실습 코드 복구 (`labs-v1`)

이전 방식의 실습 코드·테스트·실험 기록은 삭제했지만 **git 기록에 남아 있습니다.** 삭제 직전 커밋은 **`aa46d2f`**입니다.

```powershell
git checkout aa46d2f      # 옛 실습 전체를 본다(읽기용). 돌아오려면: git switch main
git tag labs-v1 aa46d2f   # 이름 붙여 두면 편하다. 원격에도 올리려면: git push origin labs-v1
```

> `labs-v1` 태그는 작업 환경에서 원격 push가 거부되어(HTTP 403) **아직 원격에 없습니다.** 위 두 명령을 PC에서 한 번 실행하면 됩니다. 사용자가 직접 만든 `pytorch/starter.py`·`tokenizer/starter.py`는 삭제하지 않고 그대로 남아 있습니다.

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

방향: **개념은 깊게, 구현은 얕게 — 서비스 설계 중심.** 아래 주차는 **권장 일정일 뿐 강제가 아닙니다.**

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

각 주제에서 확인할 질문은 [docs/LEARNING_ROADMAP.md](docs/LEARNING_ROADMAP.md)에 있습니다.

## 문서

| 문서 | 내용 |
|---|---|
| [PROJECT_PLAN](docs/PROJECT_PLAN.md) | 목적, 원칙, Part별 범위, 코드·문서 규칙 |
| [REPOSITORY_ANALYSIS](docs/REPOSITORY_ANALYSIS.md) | 참고 저장소(graphRAG_learning) 분석: 배울 점, Part별 참고, 다르게 할 점 |
| [LEARNING_ROADMAP](docs/LEARNING_ROADMAP.md) | 주제 지도(Part 1~6)와 확인할 질문, RTX 3070 가능/불가능 표, Hybrid 비교 계획 |
| [ARCHITECTURE](docs/ARCHITECTURE.md) | 최종 서비스 구조, 개념→서비스 위치, 코드 권장 위치(제안) |
| [ENVIRONMENT](docs/ENVIRONMENT.md) | uv 환경 구성, 라이브러리, 모델·VRAM·라이선스, GPU 검증, 트러블슈팅 |
| [PROGRESS](docs/PROGRESS.md) | 내가 만든 것 기록, 주간 점검 템플릿 |
| [ORCHESTRATION](docs/ORCHESTRATION.md) · [HANDOFF](docs/HANDOFF.md) | 세션 운영과 이전 방식(labs-v1) 기록 |

## 디렉터리

```text
ai-engineering-lab/
├── docs/                    # 기획·로드맵·아키텍처·환경·진행 문서
├── requirements/            # Part별 의존성 (base, dev, part1~6)
├── part1_ai_fundamentals/   # 개념 README: PyTorch, Tokenizer, Embedding, Word2Vec, Transformer
├── part2_llm/               # 개념 README: 샘플링, 로컬 LLM 서빙, Claude API, 비용·지연
├── part3_rag/               # (자리표시자) 로딩, 청킹, Dense/Sparse/Hybrid, Rerank, 평가
├── part4_graphrag/          # KG, 엔티티·관계 추출, 그래프 탐색·검색, 평가
├── part5_finetuning/        # Embedding, Reranker, LLM LoRA/QLoRA
├── part6_ai_service/        # Python AI Server(FastAPI) + Spring Boot Server — 최종 서비스 코드의 자리
├── datasets/                # 실습 데이터셋 설명 (대용량 원본은 커밋하지 않음)
└── experiments/             # 실험 기록(템플릿)
```

## 진행 방식

위 [지금 어디를 보면 되나](#지금-어디를-보면-되나-학습-방식)를 따릅니다. 진행 기록은 [docs/PROGRESS.md](docs/PROGRESS.md)에 쓰고, **학습자가 직접 확인한 것만 `COMPLETED`** 로 바꿉니다.

VS Code에서 Claude Code와 함께 학습할 때의 규칙은 [CLAUDE.md](CLAUDE.md), 준비 방법은 [ENVIRONMENT §7](docs/ENVIRONMENT.md#7-vs-code--claude-code로-학습하기)에 있습니다(정답 대신 단계별 힌트, 내 코드는 이유만 설명).

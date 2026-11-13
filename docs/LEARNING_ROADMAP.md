# 학습 로드맵 (LEARNING ROADMAP)

전체 약 **22주**, 주 **8~10시간** 기준. 각 주의 마지막에 [주간 점검](PROGRESS.md#주간-점검-템플릿)을 한다.
실습 ID는 [PROGRESS](PROGRESS.md)와 각 디렉터리 README에서 같은 값을 쓴다. `P?-S?`는 **선택 과제**다.

> 방향(2026-11-09): **개념은 깊게, 구현은 얕게 — 서비스 설계 중심.** 원리 직접 구현은 각 실습의 `advanced/` 또는 선택 과제로 둔다. 자세한 배경은 [PROJECT_PLAN §1](PROJECT_PLAN.md#1-목적).

## 1. 전체 일정

| 주차 | Part | 주제 | 마일스톤 |
|---|---|---|---|
| W1 | 0 | 환경·기획 | uv 3.12 환경, 문서 확정 |
| W2–4 | 1 | AI 원리 | 토큰·임베딩·어텐션을 **비용·지연·품질** 관점으로 설명할 수 있음 |
| W5–6 | 2 | LLM 활용 원리 | 로컬 소형 모델로 샘플링 이해, Claude API로 토큰·구조화 출력·스트리밍·캐싱 |
| W7–10 | 3 | RAG | **BM25 vs Dense vs Hybrid vs Hybrid+Rerank** 비교표 + 지연·비용 |
| W11–13 | 4 | GraphRAG | **"언제 GraphRAG가 필요한가"** 멀티홉 비교와 판단 기준 |
| W14–15 | 5 | Fine-tuning | **임베딩·Reranker fine-tuning이 언제 값어치가 있는가** 판단 |
| W16–21 | 6 | AI 서비스 | **RAG vs GraphRAG 비교 웹 서비스** (FastAPI + Spring Boot, 운영 요소 포함) |
| W22 | — | 포트폴리오 | 실험 기록·비교표 정리, 회고, 데모 |

## 2. 주차별 계획

### Part 0 — 환경·기획
| 주차 | 실습 ID | 내용 |
|---|---|---|
| W1 | P0-01 | 기획·분석·로드맵·아키텍처·환경 문서 |
| W1 | P0-02 | uv + Python 3.12 `.venv`, PyTorch 설치, [GPU 검증](ENVIRONMENT.md#5-gpu-검증-스크립트) |

### Part 1 — AI 원리 ([part1_ai_fundamentals](../part1_ai_fundamentals/README.md))

> **P1-04 PyTorch 기초는 선수 실습으로 맨 앞에 둔다**(학습자가 이미 학습함). 그 뒤는 기획서 순서 Tokenizer(P1-01) → Embedding(P1-02) → Word2Vec(P1-03) → Transformer(P1-05). 각 실습의 기존 심화 구현은 `advanced/`(선택)에 있다.

| 주차 | 실습 ID | 내용 | 선택 과제(`advanced/`) |
|---|---|---|---|
| W2 | P1-04 | **(선수)** PyTorch 기초: Tensor, autograd, `nn.Module`, DataLoader | — |
| W2 | P1-01 | Tokenizer: 토큰 수 세기, 한국어/영어 차이, **토큰 기준 비용 추정**, 토큰 예산 청킹(오버랩), 컨텍스트 창 맞추기. BPE 원리는 손 계산 예시로 | BPE 직접 구현, HF와 비교 |
| W3 | P1-02 | Embedding: L2 정규화·코사인 유사도, **OpenAI 임베딩**으로 문장 유사도·최근접 검색 | 임베딩 층·위치 임베딩 직접 구현 |
| W3 | P1-03 | Word2Vec(짧게): 분포 가설, gensim으로 짧게 학습·최근접 | Skip-gram/CBOW·Negative Sampling 직접 구현 |
| W4 | P1-05 | Transformer: 어텐션 1회 계산, 구조 그림, **컨텍스트 길이와 지연·비용**, KV 캐시 개념 | Self-Attention~미니 GPT 직접 구현·학습 |

### Part 2 — LLM 활용 원리 ([part2_llm](../part2_llm/README.md))
| 주차 | 실습 ID | 내용 |
|---|---|---|
| W5 | P2-01 | 다음 토큰 예측과 샘플링: **로컬 소형 모델(Qwen2.5-0.5B, Apache-2.0)** 로 logits·temperature·top-k·top-p, 컨텍스트 길이 |
| W5(–6) | P2-01B | **로컬 LLM 서빙과 공급자 교체:** Ollama 로컬 서버를 API처럼 호출, 공통 `LLMClient`(Claude·로컬·가짜)로 설정 교체, TTFT·토큰/초·VRAM 측정(RTX 3070) |
| W5 | P2-02 | **Claude API 기초:** 토큰 세기, 메시지·시스템 프롬프트 구조, 구조화 출력, 스트리밍 |
| W6 | P2-03 | **비용·지연:** 프롬프트 캐싱, 배치, 토큰 사용량 기록 |
| W6 | P2-04 | **프롬프트 vs RAG vs Fine-tuning 선택 기준** (+ LoRA 개념) |
| 선택 | P2-S1 | loss·역전파·학습 루프·LoRA 직접 구현 (`loss/`, `backpropagation/`, `training/`) |

> P2-01B의 공급자 인터페이스(`LLMClient`)는 Part 4 엔티티·관계 추출(P4-02/03)의 Claude vs 로컬 LLM 비교와 Part 6 AI 서버(P6-01~03의 공급자 교체·타임아웃·폴백)에서 재사용한다.

### Part 3 — RAG ([part3_rag](../part3_rag/README.md))
| 주차 | 실습 ID | 내용 |
|---|---|---|
| W7 | P3-01, P3-02 | 문서 로딩, **청킹 전략 비교**(고정 길이·문장·토큰 기준·오버랩) |
| W8 | P3-03, P3-04 | Dense 검색(**OpenAI 임베딩** + FAISS), Sparse 검색(BM25, Inverted Index 개념) |
| W9 | P3-05, P3-06 | **Hybrid(RRF, weighted) 직접 구현**, Reranker 적용 |
| W10 | P3-07 | **평가 세트 설계**, IR 지표, **지연·비용 측정**, 4방식 비교 + 프레임워크 Hybrid 비교 |

### Part 4 — GraphRAG ([part4_graphrag](../part4_graphrag/README.md))
| 주차 | 실습 ID | 내용 |
|---|---|---|
| W11 | P4-01 | Knowledge Graph 모델링, Neo4j(별도 DB) 적재, **Cypher 활용** |
| W11 | P4-02, P4-03 | 엔티티·관계 추출: **Claude 기반** 추출(규칙 기반은 짧은 비교) |
| W12 | P4-04 | Graph + Vector 검색(neo4j-graphrag), Cypher 기반 멀티홉 질의 |
| W13 | P4-05 | **Vector RAG vs GraphRAG 멀티홉 비교**, "언제 GraphRAG가 필요한가" 판단 기준 정리 |
| 선택 | P4-S1 | BFS/DFS/최단경로 직접 구현 (`graph_traversal/`) |

### Part 5 — Fine-tuning ([part5_finetuning](../part5_finetuning/README.md))
| 주차 | 실습 ID | 내용 |
|---|---|---|
| W14 | P5-01 | 임베딩: **OpenAI 임베딩 기준선 vs 로컬 임베딩(e5, bge-m3) fine-tuning 전후** |
| W15 | P5-02 | Reranker: 적용 효과 측정, 로컬 Cross-Encoder(bge-reranker-v2-m3) fine-tuning |
| W15 | P5-03 | **"fine-tuning이 언제 값어치가 있는가"** 판단 정리(품질·비용·지연·운영 부담) |
| 선택 | P5-S1 | LLM LoRA/QLoRA (개념은 메인, 실행은 선택, `llm_lora/`) |

### Part 6 — AI 서비스 ([part6_ai_service](../part6_ai_service/README.md))
| 주차 | 실습 ID | 내용 |
|---|---|---|
| W16 | P6-01 | FastAPI AI 서버: documents, embeddings, search, rag/query, graphrag/query, health |
| W17 | P6-02 | **SSE 스트리밍**(FastAPI) + Spring Boot WebClient·DTO·SSE 중계 |
| W18 | P6-03 | **타임아웃·재시도**·예외 처리(양쪽 서버), 실패 시 사용자 응답 설계 |
| W19 | P6-04 | **로깅·모니터링**, 요청별 **비용(토큰) 추적** |
| W20 | P6-05 | **Docker(Compose)**: AI 서버 + Spring Boot + Neo4j + 벡터 저장소 |
| W21 | P6-06 | **RAG vs GraphRAG 비교 웹 서비스** 통합 |

### 포트폴리오
| 주차 | 내용 |
|---|---|
| W22 | 실험 기록·비교표 정리, 회고, 데모, 확장 과제(gRPC 등) 선정 |

## 3. RTX 3070 (8GB) 학습 가능 범위

> 수치는 일반적인 경험치이며 시퀀스 길이·배치 크기에 따라 달라진다. 실제 사용량은 실험 기록에 측정해 남긴다.
> 이 저장소의 메인 실습에서 로컬 **학습**은 Part 5의 임베딩·Reranker fine-tuning뿐이다. 나머지 "직접 학습"은 **선택 과제**다.

| 작업 | 판정 | 조건·비고 |
|---|---|---|
| 미니 GPT(수백만~수천만 파라미터) 처음부터 학습 | **가능** | **선택 과제** (P1-05 `advanced/`) |
| Word2Vec, 작은 분류기 학습 | **가능** | 메인은 gensim 짧은 실행, 직접 구현은 **선택 과제** |
| 0.5~1.5B LLM 추론 | **가능** | fp16/bf16 (P2-01 개념 실습) |
| 0.5~1.5B LLM LoRA | **가능** | **선택 과제**. 1.5B는 gradient checkpointing 필수, 작은 배치 |
| 3B LLM 4bit 추론 / QLoRA | **가능** | **선택 과제**. bitsandbytes 4bit |
| 7~8B LLM QLoRA | **조건부** | **선택 과제**. 짧은 시퀀스·배치 1·checkpointing. OOM 위험 |
| 13B 이상 학습 | **사실상 불가** | |
| 임베딩 모델(~0.6B) fine-tuning | **가능** | **메인**(P5-01). 배치 크기가 MNRL 성능에 영향 → 메모리와 절충 |
| Cross-Encoder reranker fine-tuning | **가능** | **메인**(P5-02) |
| **LLM 사전학습(pre-training)** | **목표에서 제외** | 연산량·데이터 규모상 불가. 원리는 미니 GPT(선택)로 |

## 4. Hybrid Search 비교 계획

### 4-1. 순서
1. **직접 구현(Part 3, W8–9):** BM25(라이브러리 또는 간단 구현), Dense(FAISS), 결합 방식 두 가지
   - RRF: `score(d) = Σ 1 / (k + rank_i(d))` (Cormack et al., 2009)
   - Weighted: 점수 정규화(min-max 등) 후 `α·dense + (1-α)·sparse`
2. **프레임워크 구현(W10):** 같은 코퍼스·같은 질의·같은 K로 실행

| 구현 | 결합 방식(문서 기준, 버전별 확인 필요) |
|---|---|
| 직접 구현 | RRF, weighted(정규화 방식 선택 가능) |
| LangChain `EnsembleRetriever` | 가중 RRF |
| LlamaIndex `QueryFusionRetriever` | `mode`로 선택(reciprocal rerank, relative score, distance-based score 등) |
| Haystack `DocumentJoiner` | `join_mode`로 선택(RRF, merge, distribution-based rank fusion 등) |
| neo4j-graphrag `HybridRetriever` | Neo4j 벡터 인덱스 + 풀텍스트 인덱스 점수 결합 |

### 4-2. 비교 항목
- 정확도: Precision@K, Recall@K, HitRate@K, MRR, NDCG@K
- 속도: 질의당 Latency(p50/p95), 인덱싱 시간
- **비용:** 임베딩·LLM 토큰 사용량과 추정 비용
- 일관성: 같은 설정에서 직접 구현과 프레임워크의 **상위 K 결과 겹침 비율**
- 구현 비용: 코드 줄 수, 설정 가능성(가중치·k·정규화), 의존성 크기

### 4-3. 공정성 규칙
- 같은 청크, 같은 임베딩 모델, 같은 BM25 토크나이저(가능한 범위에서), 같은 K
- 프레임워크 기본값이 다르면 **기본값 실행 + 맞춘 설정 실행** 두 번 기록
- 결과는 `experiments/`에 CSV로 저장하고 차이의 원인을 해석한다

## 5. 주석 규약 요약

| 태그 | 내용 |
|---|---|
| `# 개념:` | 무엇인지 쉬운 말로(처음 보는 용어 전제, 작은 숫자 예시) |
| `# HINT 1:` | 방향 (starter만) |
| `# HINT 2:` | 함수·API·자료 구조 (starter만) |
| `# WHY:` | 왜 필요한가 / 빠지면 생기는 문제 |
| `# 서비스 관점:` | 비용·지연·품질·설계 판단에서의 의미 |
| `# REF:` | 실존 논문(저자·연도·절) 또는 공식 문서 URL. 불확실하면 생략 |

정답 코드는 힌트에 넣지 않는다. 자세한 예시는 [ARCHITECTURE §4](ARCHITECTURE.md#4-힌트-주석-규약과-예시).

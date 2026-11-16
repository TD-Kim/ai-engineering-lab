# 학습 로드맵 (LEARNING ROADMAP)

> **2026-11-16 학습 방식 전환:** 정해진 22주 실습표와 starter 방식은 폐지했습니다. 방향(기획서)만 두고, **만들고 싶은 것을 그때그때 골라 직접 구현**합니다. 이 문서는 "무슨 주제들이 있고, 각 주제를 만들 때 무엇을 확인해야 하나"를 정리한 지도입니다. 순서는 **권장일 뿐 강제가 아닙니다.** 이전 주차표(W1–W22)와 실습 ID(P1-01 등)는 커밋 `aa46d2f`의 이 문서에 있습니다.
> 방향: **개념은 깊게, 구현은 얕게 — 서비스 설계 중심.** 자세한 배경은 [PROJECT_PLAN §1](PROJECT_PLAN.md#1-목적).

## 1. 기획서 방향의 주제 (Part 1~6)

최종 목표는 **RAG·GraphRAG 서비스를 설계·판단하는 능력**과 [포트폴리오 서비스](ARCHITECTURE.md#1-최종-서비스-구조)입니다. 아래 주제 중 만들고 싶은 것을 고르고, 먼저 "[최종 서비스의 어디에 들어가는가](ARCHITECTURE.md#1-2-개념이-서비스의-어디에-쓰이나)"를 확인하세요.

**의존 관계(권장 순서의 이유):** 임베딩(Part 1) 개념은 Part 3(RAG) 전에 알고 있어야 하고, LLM 호출부(Part 2)는 Part 3~4의 생성·추출에 쓰입니다. 그 밖에는 서로 독립적이라 순서를 바꿔도 됩니다. 표는 [루트 README](../README.md#지금-어디를-보면-되나-학습-방식)에 있습니다.

### Part 1 — AI 원리 ([개념 README](../part1_ai_fundamentals/README.md))
| 주제 | 확인할 질문 |
|---|---|
| PyTorch 기초 (선수) | shape·device·`.grad` 누적 오류를 읽을 수 있나? 학습 루프 5단계를 쓸 수 있나? |
| Tokenizer | 한국어는 왜 같은 글자 수에 토큰이 많나? 토큰 수로 비용·청크 크기를 어떻게 계산하나? 로컬 토큰 수와 청구 토큰 수는 왜 다른가? |
| Embedding | 정규화 후 내적이 왜 코사인인가? 모델을 바꾸면 왜 전부 재임베딩인가? API와 로컬 임베딩은 비용·지연·프라이버시에서 어떻게 다른가? 한국어 품질은 내 데이터로 어떻게 재나? |
| Word2Vec | 분포 가설이 무엇이고 "단어 하나에 벡터 하나"의 한계는? |
| Transformer | 컨텍스트 길이가 지연·비용에 주는 영향은? KV 캐시가 무엇을 아끼나? 어텐션 점수는 어떻게 계산되나? |

### Part 2 — LLM 활용 원리 ([개념 README](../part2_llm/README.md))
| 주제 | 확인할 질문 |
|---|---|
| 다음 토큰 예측·샘플링 | temperature·top-k·top-p가 확률 분포를 어떻게 바꾸나? |
| 로컬 LLM 서빙·공급자 교체 | 공통 인터페이스로 Claude·로컬·가짜를 어떻게 바꿔 끼우나? RTX 3070(8GB)에서 어떤 크기까지 가능한가? 일시 장애에만 폴백하는 이유는? |
| Claude API 기초 | 응답은 stop_reason부터 왜 확인하나? 구조화 출력·스트리밍·거절·폴백은 어떻게 다루나? 호출 전 비용은 어떻게 추정하나? |
| 비용·지연 | 프롬프트 캐싱은 무엇을 앞에 두어야 읽히나? 배치는 언제 쓰나(50% 할인과 최대 24시간)? 캐시 적중률·p95를 어떻게 모니터링하나? |
| 프롬프트 vs RAG vs Fine-tuning | 같은 질문에 무엇을 고를지 비용·지연·품질 근거는? LoRA 개념은? (작성 예정) |

### Part 3 — RAG ([개념 README](../part3_rag/README.md))
| 주제 | 확인할 질문 |
|---|---|
| 문서 로딩·청킹 | 청크 크기·오버랩이 검색 품질과 임베딩 비용에 주는 영향은? 고정 길이·문장·토큰 기준 중 무엇을 고르나? |
| Dense / Sparse(BM25) 검색 | 각각 어떤 질의에 강하고 약한가? 벡터DB(FAISS/ChromaDB)는 어떻게 쓰나? |
| Hybrid(RRF·weighted)·Reranker | 결합 방식에 따라 결과가 어떻게 바뀌나? Reranker는 얼마의 지연·비용으로 얼마를 개선하나? |
| 평가 | 평가 세트를 어떻게 설계하나? IR 지표(Precision/Recall/HitRate/MRR/NDCG)·지연·비용을 어떻게 같이 보나? ([Hybrid 비교 계획](#3-hybrid-search-비교-계획)) |

### Part 4 — GraphRAG ([개념 README](../part4_graphrag/README.md))
| 주제 | 확인할 질문 |
|---|---|
| Knowledge Graph·Neo4j·Cypher | 무엇을 노드·관계로 모델링하나? Cypher로 멀티홉을 어떻게 묻나? |
| 엔티티·관계 추출 | Claude 구조화 출력으로 어떻게 추출하나? 규칙 기반·로컬 LLM과 비교하면? |
| Graph + Vector 검색, 평가 | 멀티홉 질문에서 Vector RAG와 GraphRAG의 차이는? **언제 GraphRAG가 필요한가**의 판단 기준은? |

### Part 5 — Fine-tuning ([개념 README](../part5_finetuning/README.md))
| 주제 | 확인할 질문 |
|---|---|
| 임베딩·Reranker fine-tuning | OpenAI 기준선 대비 로컬 fine-tuning 전후 수치는? 학습 시간·VRAM은? |
| 판단 정리 | **fine-tuning이 언제 값어치가 있는가**(품질·비용·지연·운영 부담)? LLM LoRA는 개념 + 선택 |

### Part 6 — AI 서비스 ([개념 README](../part6_ai_service/README.md))
| 주제 | 확인할 질문 |
|---|---|
| FastAPI AI 서버 | 엔드포인트(documents, embeddings, search, rag/query, graphrag/query, health)는 어떻게 나누나? |
| SSE 스트리밍·Spring Boot 연동 | 스트림을 WebClient로 어떻게 중계하나? DTO는? |
| 타임아웃·재시도·예외 | 외부 API 실패 시 사용자에게 무엇을 돌려주나? 재시도는 어디까지? |
| 로깅·모니터링·비용 추적 | 요청별 토큰·비용·캐시 적중률을 어떻게 남기나? 원문은 왜 로그에 안 남기나? |
| Docker Compose·통합 | AI 서버 + Spring Boot + Neo4j + 벡터 저장소를 한 번에 어떻게 띄우나? **RAG vs GraphRAG 비교 웹 서비스**로 어떻게 묶나? |

### 포트폴리오
실험 기록·비교표 정리, 회고, 데모, 확장 과제(gRPC 등) 선정. 기록은 [PROGRESS](PROGRESS.md)와 [experiments](../experiments/README.md)에 쌓습니다.

## 2. RTX 3070 (8GB) 학습 가능 범위

> 수치는 일반적인 경험치이며 시퀀스 길이·배치 크기에 따라 달라진다. 실제 사용량은 실험 기록에 측정해 남긴다.
> 로컬 **학습**의 기본 범위는 Part 5의 임베딩·Reranker fine-tuning이다. 나머지 "직접 학습"은 **선택**이다.

| 작업 | 판정 | 조건·비고 |
|---|---|---|
| 미니 GPT(수백만~수천만 파라미터) 처음부터 학습 | **가능** | **선택** (P1-05 심화 노트) |
| Word2Vec, 작은 분류기 학습 | **가능** | gensim 짧은 실행이 기본, 직접 구현은 **선택** |
| 0.5~1.5B LLM 추론 | **가능** | fp16/bf16 (P2-01 개념) |
| 0.5~1.5B LLM LoRA | **가능** | **선택**. 1.5B는 gradient checkpointing 필수, 작은 배치 |
| 3B LLM 4bit 추론 / QLoRA | **가능** | **선택**. bitsandbytes 4bit |
| 7~8B LLM QLoRA | **조건부** | **선택**. 짧은 시퀀스·배치 1·checkpointing. OOM 위험 |
| 13B 이상 학습 | **사실상 불가** | |
| 임베딩 모델(~0.6B) fine-tuning | **가능** | **기본 범위**(Part 5 임베딩). 배치 크기가 MNRL 성능에 영향 → 메모리와 절충 |
| Cross-Encoder reranker fine-tuning | **가능** | **기본 범위**(Part 5 Reranker) |
| **LLM 사전학습(pre-training)** | **목표에서 제외** | 연산량·데이터 규모상 불가. 원리는 미니 GPT(선택)로 |

## 3. Hybrid Search 비교 계획

### 3-1. 순서
1. **직접 구현(Part 3):** BM25(라이브러리 또는 간단 구현), Dense(FAISS), 결합 방식 두 가지
   - RRF: `score(d) = Σ 1 / (k + rank_i(d))` (Cormack et al., 2009)
   - Weighted: 점수 정규화(min-max 등) 후 `α·dense + (1-α)·sparse`
2. **프레임워크 구현:** 같은 코퍼스·같은 질의·같은 K로 실행

| 구현 | 결합 방식(문서 기준, 버전별 확인 필요) |
|---|---|
| 직접 구현 | RRF, weighted(정규화 방식 선택 가능) |
| LangChain `EnsembleRetriever` | 가중 RRF |
| LlamaIndex `QueryFusionRetriever` | `mode`로 선택(reciprocal rerank, relative score, distance-based score 등) |
| Haystack `DocumentJoiner` | `join_mode`로 선택(RRF, merge, distribution-based rank fusion 등) |
| neo4j-graphrag `HybridRetriever` | Neo4j 벡터 인덱스 + 풀텍스트 인덱스 점수 결합 |

### 3-2. 비교 항목
- 정확도: Precision@K, Recall@K, HitRate@K, MRR, NDCG@K
- 속도: 질의당 Latency(p50/p95), 인덱싱 시간
- **비용:** 임베딩·LLM 토큰 사용량과 추정 비용
- 일관성: 같은 설정에서 직접 구현과 프레임워크의 **상위 K 결과 겹침 비율**
- 구현 비용: 코드 줄 수, 설정 가능성(가중치·k·정규화), 의존성 크기

### 3-3. 공정성 규칙
- 같은 청크, 같은 임베딩 모델, 같은 BM25 토크나이저(가능한 범위에서), 같은 K
- 프레임워크 기본값이 다르면 **기본값 실행 + 맞춘 설정 실행** 두 번 기록
- 결과는 `experiments/`에 CSV로 저장하고 차이의 원인을 해석한다

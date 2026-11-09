# 학습 로드맵 (LEARNING ROADMAP)

전체 약 **30주**, 주 **8~10시간** 기준. 각 주의 마지막에 [주간 점검](PROGRESS.md#주간-점검-템플릿)을 한다.
실습 ID는 [PROGRESS](PROGRESS.md)와 각 디렉터리 README에서 같은 값을 쓴다.

## 1. 전체 일정

| 주차 | Part | 주제 | 마일스톤 |
|---|---|---|---|
| W1 | 0 | 환경·기획 | uv 3.12 환경, GPU 검증 통과, 문서 확정 |
| W2–6 | 1 | AI 원리 | **미니 GPT** 직접 구현·학습 |
| W7–10 | 2 | LLM 학습 원리 | **소형 LM** 학습, LoRA 원리 이해 |
| W11–15 | 3 | RAG | **BM25 vs Dense vs Hybrid vs Hybrid+Rerank** 비교표 |
| W16–19 | 4 | GraphRAG | **Vector RAG vs GraphRAG** 멀티홉 평가 |
| W20–23 | 5 | Fine-tuning | 임베딩·Reranker·LLM LoRA **전후 비교** |
| W24–28 | 6 | AI 서비스 | **RAG vs GraphRAG 비교 웹 서비스** |
| W29–30 | — | 포트폴리오 | 실험 기록 정리, 회고, 데모 |

## 2. 주차별 계획

### Part 0 — 환경·기획
| 주차 | 실습 ID | 내용 |
|---|---|---|
| W1 | P0-01 | uv + Python 3.12 `.venv`, PyTorch cu128 설치, [GPU 검증](ENVIRONMENT.md#5-gpu-검증-스크립트), 문서 검토 |

### Part 1 — AI 원리 ([part1_ai_fundamentals](../part1_ai_fundamentals/README.md))
| 주차 | 실습 ID | 내용 | 난이도 |
|---|---|---|---|
| W2 | P1-01 | Tokenizer: 문자/단어/서브워드, **BPE 직접 구현**, HF `tokenizers`와 비교 | L1–L2 |
| W3 | P1-02 | Embedding: one-hot vs dense, 코사인 유사도, `nn.Embedding` | L1 |
| W3 | P1-03 | Word2Vec: CBOW, Skip-gram, **Negative Sampling** 직접 구현, gensim과 비교 | L2 |
| W4 | P1-04 | PyTorch 기초: Tensor, autograd, `nn.Module`, DataLoader, GPU 이동 | L1 |
| W5 | P1-05a | Transformer 부품: Self-Attention, MHA, FFN, Residual, LayerNorm, Positional Encoding | L2 |
| W6 | P1-05b | **미니 GPT** 조립·학습(문자 단위 또는 작은 BPE 어휘) | L3 |

### Part 2 — LLM 학습 원리 ([part2_llm](../part2_llm/README.md))
| 주차 | 실습 ID | 내용 | 난이도 |
|---|---|---|---|
| W7 | P2-01, P2-02 | Next token prediction, Cross-Entropy loss와 perplexity | L1–L2 |
| W8 | P2-03 | Backpropagation(작은 계산 그래프 수동 미분 → autograd 검증), GD·SGD·Adam/AdamW | L2 |
| W9 | P2-04 | 소형 LM 학습: 학습률 스케줄, overfitting 관찰, 검증 손실 | L2–L3 |
| W10 | P2-05 | Pre-training vs Fine-tuning, Instruction tuning 데이터 형식, **LoRA 직접 구현**(작은 Linear) | L2 |

### Part 3 — RAG ([part3_rag](../part3_rag/README.md))
| 주차 | 실습 ID | 내용 | 난이도 |
|---|---|---|---|
| W11 | P3-01, P3-02 | 문서 로딩(PDF·텍스트), 청킹(고정 길이·문장·오버랩) | L1 |
| W12 | P3-03 | Dense 검색: Sentence Transformers 임베딩, FAISS(Flat/IVF), brute-force와 비교 | L2 |
| W13 | P3-04 | Sparse 검색: **Inverted Index + BM25 직접 구현**, `rank_bm25`와 비교 | L2 |
| W14 | P3-05, P3-06 | **Hybrid(RRF, weighted) 직접 구현**, Cross-Encoder Rerank | L2–L3 |
| W15 | P3-07 | 평가: Precision/Recall/HitRate@K, MRR, NDCG, Latency → CSV·시각화, **4방식 비교 + 프레임워크 비교** | L3 |

### Part 4 — GraphRAG ([part4_graphrag](../part4_graphrag/README.md))
| 주차 | 실습 ID | 내용 | 난이도 |
|---|---|---|---|
| W16 | P4-01 | Knowledge Graph 모델링, Neo4j(별도 DB) 적재, Cypher 기초 | L1 |
| W17 | P4-02, P4-03 | 엔티티·관계 추출: **규칙 기반 vs 로컬 LLM 기반** 정확도 비교 | L2 |
| W18 | P4-04, P4-05 | BFS/DFS/최단경로/멀티홉 Python 구현 vs Cypher, **Graph+Vector 검색** | L2–L3 |
| W19 | P4-06 | **Vector RAG vs GraphRAG** 멀티홉 질의 정량 평가 | L3 |

### Part 5 — Fine-tuning ([part5_finetuning](../part5_finetuning/README.md))
| 주차 | 실습 ID | 내용 | 난이도 |
|---|---|---|---|
| W20 | P5-00 | 베이스라인 측정, 학습/검증/테스트 분리, hard negative 마이닝 | L2 |
| W21 | P5-01 | 임베딩 fine-tuning: MNRL, Triplet, hard negative | L2 |
| W22 | P5-02 | Reranker(Cross-Encoder) fine-tuning | L2 |
| W23 | P5-03 | LLM LoRA/QLoRA(0.5~1.5B LoRA, 3B QLoRA), GPU 메모리·시간 측정 | L3 |

### Part 6 — AI 서비스 ([part6_ai_service](../part6_ai_service/README.md))
| 주차 | 실습 ID | 내용 | 난이도 |
|---|---|---|---|
| W24 | P6-01a | FastAPI: documents, embeddings, search, health | L2 |
| W25 | P6-01b | rag/query, graphrag/query, 스트리밍 응답, Docker | L2–L3 |
| W26 | P6-02a | Spring Boot: WebClient, DTO, 예외 처리 | L2 |
| W27 | P6-02b | SSE 중계, 타임아웃, 로깅, 장애 처리(재시도·폴백) | L3 |
| W28 | P6-03 | **RAG vs GraphRAG 비교 웹 서비스** 통합 | L3 |

### 포트폴리오
| 주차 | 내용 |
|---|---|
| W29 | 실험 기록 정리, 비교표·그래프 정리, README 데모 섹션 |
| W30 | 회고, 확장 과제(gRPC 등) 선정, 발표 자료 |

## 3. RTX 3070 (8GB) 학습 가능 범위

> 수치는 일반적인 경험치이며 시퀀스 길이·배치 크기에 따라 달라진다. 실제 사용량은 실험 기록에 측정해 남긴다.

| 작업 | 판정 | 조건·비고 |
|---|---|---|
| 미니 GPT(수백만~수천만 파라미터) 처음부터 학습 | **가능** | Part 1·2의 핵심. 직접 학습 |
| Word2Vec, 작은 분류기 학습 | **가능** | |
| 0.5~1.5B LLM 추론 | **가능** | fp16/bf16 |
| 0.5~1.5B LLM LoRA | **가능** | 1.5B는 **gradient checkpointing 필수**, 작은 배치 |
| 3B LLM 4bit 추론 / QLoRA | **가능** | bitsandbytes 4bit |
| 7~8B LLM QLoRA | **조건부** | 짧은 시퀀스·배치 1·checkpointing. **OOM 위험** |
| 13B 이상 학습 | **사실상 불가** | |
| 임베딩 모델(~0.6B) fine-tuning | **가능** | 배치 크기가 MNRL 성능에 영향 → 메모리와 절충 |
| Cross-Encoder reranker fine-tuning | **가능** | |
| **LLM 사전학습(pre-training)** | **목표에서 제외** | 연산량·데이터 규모상 불가. 원리는 미니 GPT로 학습 |

## 4. Hybrid Search 비교 계획

### 4-1. 순서
1. **직접 구현(Part 3, W13–14):** Inverted Index + BM25, Dense(FAISS), 결합 방식 두 가지
   - RRF: `score(d) = Σ 1 / (k + rank_i(d))` (Cormack et al., 2009)
   - Weighted: 점수 정규화(min-max 등) 후 `α·dense + (1-α)·sparse`
2. **프레임워크 구현(W15):** 같은 코퍼스·같은 질의·같은 K로 실행

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
- 일관성: 같은 설정에서 직접 구현과 프레임워크의 **상위 K 결과 겹침 비율**
- 구현 비용: 코드 줄 수, 설정 가능성(가중치·k·정규화), 의존성 크기

### 4-3. 공정성 규칙
- 같은 청크, 같은 임베딩 모델, 같은 BM25 토크나이저(가능한 범위에서), 같은 K
- 프레임워크 기본값이 다르면 **기본값 실행 + 맞춘 설정 실행** 두 번 기록
- 결과는 `experiments/`에 CSV로 저장하고 차이의 원인을 해석한다

## 5. 주석 규약 요약

| 태그 | 내용 |
|---|---|
| `# HINT 1:` | 방향 |
| `# HINT 2:` | shape·API |
| `# WHY:` | 왜 필요한가 / 빠지면 생기는 문제 |
| `# REF:` | 실존 논문(저자·연도·절) 또는 공식 문서 URL. 불확실하면 생략 |

정답 코드는 힌트에 넣지 않는다. 자세한 예시는 [ARCHITECTURE §4](ARCHITECTURE.md#4-힌트-주석-규약과-예시).

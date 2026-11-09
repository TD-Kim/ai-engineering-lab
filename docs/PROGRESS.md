# 진행 상황 (PROGRESS)

## 상태 정의

| 상태 | 의미 |
|---|---|
| `NOT_STARTED` | 시작 전 |
| `IN_PROGRESS` | 진행 중 |
| `COMPLETED` | **학습자가 직접 확인한 것만** |

> `COMPLETED`는 **학습자가 직접 확인한 뒤에만** 표시한다. 산출물 검증은 별도 검증 단계가 맡고, 학습 중 빠진 부분은 학습자가 그때그때 알려 준다.

## Part 0 — 환경·기획

| ID | 실습 | 주차 | 상태 | 확인일 | 메모 |
|---|---|---|---|---|---|
| P0-01 | 기획·분석·로드맵·아키텍처·환경 문서, 디렉터리 골격 | W1 | `NOT_STARTED` | | 문서 초안 커밋됨, 학습자 확인 전 |
| P0-02 | uv 3.12 `.venv` + PyTorch cu128 설치, GPU 검증 통과 | W1 | `NOT_STARTED` | | 로컬 PC에서 실행 필요 |

## Part 1 — AI 원리

> 진행 순서: **P1-04 PyTorch 기초를 선수 실습으로 맨 앞에 둔다**(이후 모든 실습이 Tensor·autograd·학습 루프를 쓰기 때문). 그 뒤는 기획서 순서 1-1 Tokenizer(P1-01) → 1-2 Embedding(P1-02) → 1-3 Word2Vec(P1-03) → 1-4 Transformer(P1-05).

| ID | 실습 | 주차 | 난이도 | 상태 | 확인일 | 메모 |
|---|---|---|---|---|---|---|
| P1-04 | (선수) PyTorch 기초 | W2 | L1–L3 | `NOT_STARTED` | | 실습 자료 작성 완료, 학습자 확인 전 |
| P1-01 | Tokenizer · BPE 직접 구현 | W3 | L1–L3 | `NOT_STARTED` | | 실습 자료 작성 완료, 학습자 확인 전 |
| P1-02 | Embedding 기초 | W4 | L1–L3 | `NOT_STARTED` | | |
| P1-03 | Word2Vec (CBOW, Skip-gram, Negative Sampling) | W4 | L1–L3 | `NOT_STARTED` | | |
| P1-05a | Transformer 부품 (Attention, MHA, FFN, Residual, LayerNorm, PE) | W5 | L2 | `NOT_STARTED` | | |
| P1-05b | 미니 GPT 조립·학습 | W6 | L3 | `NOT_STARTED` | | |

## Part 2 — LLM 학습 원리

| ID | 실습 | 주차 | 난이도 | 상태 | 확인일 | 메모 |
|---|---|---|---|---|---|---|
| P2-01 | Next token prediction | W7 | L1 | `NOT_STARTED` | | |
| P2-02 | Cross-Entropy loss · perplexity | W7 | L2 | `NOT_STARTED` | | |
| P2-03 | Backpropagation · GD · Optimizer | W8 | L2 | `NOT_STARTED` | | |
| P2-04 | 소형 LM 학습 · overfitting | W9 | L2–L3 | `NOT_STARTED` | | |
| P2-05 | Pre-training vs Fine-tuning · Instruction tuning · LoRA 원리 | W10 | L2 | `NOT_STARTED` | | |

## Part 3 — RAG

| ID | 실습 | 주차 | 난이도 | 상태 | 확인일 | 메모 |
|---|---|---|---|---|---|---|
| P3-01 | 문서 로딩 | W11 | L1 | `NOT_STARTED` | | |
| P3-02 | 청킹 | W11 | L1 | `NOT_STARTED` | | |
| P3-03 | Dense 검색 · FAISS | W12 | L2 | `NOT_STARTED` | | |
| P3-04 | Inverted Index · BM25 직접 구현 | W13 | L2 | `NOT_STARTED` | | |
| P3-05 | Hybrid (RRF, weighted) 직접 구현 | W14 | L2–L3 | `NOT_STARTED` | | |
| P3-06 | Cross-Encoder Rerank | W14 | L2 | `NOT_STARTED` | | |
| P3-07 | 평가 · 4방식 비교 · 프레임워크 Hybrid 비교 | W15 | L3 | `NOT_STARTED` | | |

## Part 4 — GraphRAG

| ID | 실습 | 주차 | 난이도 | 상태 | 확인일 | 메모 |
|---|---|---|---|---|---|---|
| P4-01 | Knowledge Graph · Neo4j 기초 | W16 | L1 | `NOT_STARTED` | | |
| P4-02 | 엔티티 추출 (규칙 vs LLM) | W17 | L2 | `NOT_STARTED` | | |
| P4-03 | 관계 추출 (규칙 vs LLM) | W17 | L2 | `NOT_STARTED` | | |
| P4-04 | 그래프 탐색 (BFS/DFS/최단경로/멀티홉) vs Cypher | W18 | L2 | `NOT_STARTED` | | |
| P4-05 | Graph + Vector 검색 | W18 | L3 | `NOT_STARTED` | | |
| P4-06 | Vector RAG vs GraphRAG 멀티홉 평가 | W19 | L3 | `NOT_STARTED` | | |

## Part 5 — Fine-tuning

| ID | 실습 | 주차 | 난이도 | 상태 | 확인일 | 메모 |
|---|---|---|---|---|---|---|
| P5-00 | 베이스라인 · 데이터 분리 · hard negative 마이닝 | W20 | L2 | `NOT_STARTED` | | |
| P5-01 | 임베딩 fine-tuning (MNRL, Triplet) | W21 | L2 | `NOT_STARTED` | | |
| P5-02 | Reranker fine-tuning | W22 | L2 | `NOT_STARTED` | | |
| P5-03 | LLM LoRA / QLoRA | W23 | L3 | `NOT_STARTED` | | |

## Part 6 — AI 서비스

| ID | 실습 | 주차 | 난이도 | 상태 | 확인일 | 메모 |
|---|---|---|---|---|---|---|
| P6-01a | FastAPI: documents · embeddings · search · health | W24 | L2 | `NOT_STARTED` | | |
| P6-01b | FastAPI: rag/query · graphrag/query · 스트리밍 · Docker | W25 | L2–L3 | `NOT_STARTED` | | |
| P6-02a | Spring Boot: WebClient · DTO · 예외 처리 | W26 | L2 | `NOT_STARTED` | | |
| P6-02b | Spring Boot: SSE · 타임아웃 · 로깅 · 장애 처리 | W27 | L3 | `NOT_STARTED` | | |
| P6-03 | RAG vs GraphRAG 비교 웹 서비스 | W28 | L3 | `NOT_STARTED` | | |

## 포트폴리오

| ID | 항목 | 주차 | 상태 | 확인일 | 메모 |
|---|---|---|---|---|---|
| PF-01 | 실험 기록·비교표 정리 | W29 | `NOT_STARTED` | | |
| PF-02 | 회고 · 데모 · 확장 과제 선정 | W30 | `NOT_STARTED` | | |

---

## 주간 점검 템플릿

아래 블록을 복사해 이 파일 맨 아래 "주간 점검 기록"에 추가한다.

```markdown
### W__ (YYYY-MM-DD ~ YYYY-MM-DD)

- 계획한 실습: P_-__, P_-__
- 실제 학습 시간: __ 시간 (목표 8~10시간)
- 상태 변경: P_-__ NOT_STARTED → IN_PROGRESS → COMPLETED
- 직접 설명할 수 있게 된 개념 (3줄):
  1.
  2.
  3.
- 아직 설명하지 못하는 것 / 막힌 곳:
- 실험 기록 링크: experiments/...
- 측정 수치 (있다면): 지표, GPU 메모리, 학습 시간
- 다음 주 조정 사항:
- 포트폴리오에 남길 한 줄:
```

## 주간 점검 기록

(아직 없음)

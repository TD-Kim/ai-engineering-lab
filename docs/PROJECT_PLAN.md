# 프로젝트 기획 (PROJECT PLAN)

## 1. 목적

- AI 엔지니어링을 **실습형**으로 학습한다.
- 라이브러리를 사용하는 능력과 **내부 원리를 직접 구현**하는 능력을 동시에 기른다.
- 최종적으로 Java/Spring Boot 경력과 결합한 **AI 서비스 포트폴리오**를 만든다.
  진행 과정 자체(실험 기록, 비교 수치, 회고)를 포트폴리오 자산으로 관리한다.

## 2. 학습 환경과 제약

| 항목 | 값 | 비고 |
|---|---|---|
| OS | Windows | DataLoader 사용 스크립트는 `if __name__ == "__main__":` 필수 |
| GPU | RTX 3070, VRAM 8GB | 가능/불가능 구분은 [LEARNING_ROADMAP](LEARNING_ROADMAP.md#3-rtx-3070-8gb-학습-가능-범위) |
| RAM | 32GB | |
| Python | 3.12 권장 | gensim·faiss 계열의 3.14 휠 부재 때문 |
| 패키지 관리 | uv | 이 저장소 자체 `.venv` 사용 |

- 참고 저장소의 `part1\.venv`(Python 3.14.7)는 **건드리지 않는다.** 새 저장소는 독립 환경을 쓴다.
- 자세한 설치는 [ENVIRONMENT](ENVIRONMENT.md).

## 3. 원칙

1. **무료 오픈소스·로컬 모델 우선.** 유료 API(OpenAI, Claude 등)를 전제로 한 실습은 만들지 않는다.
   유료 API는 선택적 비교군으로만 언급할 수 있다.
2. **모델 라이선스 확인.** 모델 카드의 라이선스를 기록하고, 비상업 라이선스 모델은 포트폴리오 서비스 기본값으로 쓰지 않는다.
3. **기술은 필요한 단계에서만 도입한다.**
   PyTorch(Part 1) → HF Transformers(Part 2) → Sentence Transformers·FAISS·ChromaDB(Part 3) →
   Neo4j(Part 4) → PEFT(Part 2 원리, Part 5 실전) → FastAPI·Spring Boot·Docker(Part 6).
4. **개선을 가정하지 않는다.** 모든 "더 좋다"는 동일 데이터셋·동일 테스트셋의 수치로만 말한다.
5. **저작권 존중.** 참고 저장소(강의 실습 코드)의 코드·데이터는 복사하지 않고 링크로만 참고한다.

## 4. Part별 범위

### Part 0 — 환경·기획 (W1)
기획·분석·로드맵·아키텍처·환경 문서, uv 환경, GPU 검증, 디렉터리 골격.

### Part 1 — AI 원리 (W2–6)
- Tokenizer, BPE 직접 구현
- Embedding, Word2Vec(CBOW, Skip-gram, Negative Sampling)
- PyTorch 기초(Tensor, autograd, nn.Module, DataLoader)
- Transformer: Self-Attention, Multi-Head Attention, FFN, Residual, LayerNorm, Positional Encoding
- **완료 조건:** 작은 Transformer(미니 GPT)를 직접 구현해 RTX 3070에서 학습

### Part 2 — LLM 학습 원리 (W7–10)
- Next token prediction, Cross-Entropy loss, Backpropagation, Gradient Descent, Optimizer
- Overfitting, Pre-training vs Fine-tuning, Instruction tuning, LoRA/QLoRA 원리
- **완료 조건:** 소형 LM 학습 및 학습 곡선 해석

### Part 3 — RAG (W11–15)
- 문서 로딩, 청킹, Dense 검색, FAISS, BM25, Inverted Index
- Hybrid(RRF, weighted), Cross-Encoder Rerank
- 평가: Precision@K, Recall@K, HitRate@K, MRR, NDCG, Latency → CSV·시각화
- **완료 조건:** 동일 데이터셋에서 **BM25 vs Dense vs Hybrid vs Hybrid+Rerank** 수치 비교
- 직접 구현한 Hybrid를 프레임워크 구현과 비교(§5-1)

### Part 4 — GraphRAG (W16–19)
- Knowledge Graph, 규칙 기반 vs LLM 기반 엔티티·관계 추출
- BFS/DFS/최단경로/멀티홉 탐색(Python 직접 구현) vs Neo4j Cypher 비교
- Graph + Vector 검색
- **완료 조건:** Vector RAG vs GraphRAG **멀티홉 질의 정량 평가**

### Part 5 — Fine-tuning (W20–23)
- Embedding: MNRL(MultipleNegativesRankingLoss), Triplet, hard negative
- Reranker fine-tuning
- LLM LoRA/QLoRA
- **완료 조건:** 동일 테스트셋 **전후 비교**, GPU 메모리·학습 시간 측정. 개선을 가정하지 않음

### Part 6 — AI 서비스 (W24–28)
- FastAPI AI 서버: `/api/v1/documents`, `/api/v1/embeddings`, `/api/v1/search`,
  `/api/v1/rag/query`, `/api/v1/graphrag/query`, `/api/v1/health`, 스트리밍 응답
- Spring Boot: WebClient, DTO, SSE, 예외 처리, 타임아웃, 로깅, 장애 처리
- 흐름: 사용자 → Spring Boot → Python AI Server → RAG/GraphRAG → VectorDB/Neo4j → LLM
- REST 먼저, gRPC는 확장 과제
- **완료 조건:** RAG vs GraphRAG 비교 웹 서비스

## 5. 사용자 강조 사항

### 5-1. Hybrid Search 직접 구현 후 프레임워크와 비교
직접 구현(BM25 + Dense + RRF/weighted)을 먼저 완성한 뒤 다음과 비교한다.
- LangChain `EnsembleRetriever`
- LlamaIndex `QueryFusionRetriever`
- Haystack 하이브리드 파이프라인(`DocumentJoiner`)
- neo4j-graphrag `HybridRetriever`

비교 항목과 절차는 [LEARNING_ROADMAP §4](LEARNING_ROADMAP.md#4-hybrid-search-비교-계획).

### 5-2. RTX 3070에서 가능한 학습과 불가능한 학습 구분
- LLM **사전학습은 목표에서 제외**한다.
- 작은 Transformer는 **직접 학습**한다.
- 실제 LLM은 **사전학습 모델 + LoRA/QLoRA**로 다룬다.

### 5-3. Spring Boot가 Python AI 서버를 호출하는 구조로 마무리
- 진행 과정 자체를 포트폴리오로 관리하고, **주차별 점검**([PROGRESS](PROGRESS.md#주간-점검-템플릿))을 한다.

## 6. 코드·문서 규칙

### 6-1. 실습 구성 순서
각 실습은 다음 순서로 설명한다.

> 개념 → 수학 → 데이터 흐름 → 코드 구조 → 구현할 부분 → 실행법 → 예상 출력 → 해석 → 과제 → 정답

- `starter.py`(TODO + 힌트)와 `solution.py`를 분리한다. 가능하면 `pytest` 테스트를 둔다.
- 난이도: **Level 1**(따라 하기) / **Level 2**(빈칸 구현) / **Level 3**(설계부터 직접).
- 세부 규약은 [ARCHITECTURE §3](ARCHITECTURE.md#3-실습-파일-규약-startersolutiontest).

### 6-2. 디렉터리 README 12항목
1. 목표 2. 선수지식 3. 핵심개념 4. 내부원리 5. 수식/그림 6. 실행법
7. TODO 목록 8. 예상 결과 9. 자주 나는 오류 10. 과제 11. 체크리스트 12. 다음 단계 연결

### 6-3. 힌트 주석 규약
`starter.py`의 각 TODO 아래에 다음 주석을 단다.

| 태그 | 내용 |
|---|---|
| `# HINT 1:` | 방향(무엇을 해야 하는가) |
| `# HINT 2:` | shape·API(어떤 함수, 어떤 차원) |
| `# WHY:` | 왜 필요한가 / 빠지면 무슨 문제가 생기는가 |
| `# REF:` | 실존하는 논문(저자·연도·절) 또는 공식 문서 URL만. **불확실하면 생략, 지어내기 금지** |

- 힌트에 정답 코드를 넣지 않는다.
- `solution.py`에는 "왜" 주석과 shape 주석을 달고, **같은 TODO 번호와 REF를 유지**한다.
- 예시: [ARCHITECTURE §4](ARCHITECTURE.md#4-힌트-주석-규약과-예시).

### 6-4. 실험 기록
`experiments/`에 실험마다 다음을 기록한다: 목적, 데이터셋, 모델, 하이퍼파라미터, GPU 사용량, 학습 시간, 결과, 비교, 개선점.
템플릿은 [experiments/README.md](../experiments/README.md).

### 6-5. 진행 상태
[PROGRESS](PROGRESS.md)에서 `NOT_STARTED` / `IN_PROGRESS` / `COMPLETED` 세 가지로 관리한다.
**학습자가 직접 확인한 것만 `COMPLETED`.**

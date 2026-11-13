# 진행 상황 (PROGRESS)

## 상태 정의

| 상태 | 의미 |
|---|---|
| `NOT_STARTED` | 시작 전 |
| `IN_PROGRESS` | 진행 중 |
| `COMPLETED` | **학습자가 직접 확인한 것만** |

> `COMPLETED`는 **학습자가 직접 확인한 뒤에만** 표시한다. 산출물 검증은 별도 검증 단계가 맡고, 학습 중 빠진 부분은 학습자가 그때그때 알려 준다.

> 계획 변경(2026-11-09): **개념은 깊게, 구현은 얕게 — 서비스 설계 중심.** 22주 계획. 구분: **메인** / **선택**(advanced·선택 과제). 자세한 내용은 [LEARNING_ROADMAP](LEARNING_ROADMAP.md).

## Part 0 — 환경·기획

| ID | 실습 | 주차 | 구분 | 상태 | 확인일 | 메모 |
|---|---|---|---|---|---|---|
| P0-01 | 기획·분석·로드맵·아키텍처·환경 문서, 디렉터리 골격 | W1 | 메인 | `NOT_STARTED` | | 문서 초안 커밋됨, 학습자 확인 전 |
| P0-02 | uv 3.12 `.venv` + PyTorch 설치, GPU 검증 | W1 | 메인 | `NOT_STARTED` | | 로컬 PC에서 실행 필요 |

## Part 1 — AI 원리

> **P1-04 PyTorch 기초는 선수 실습으로 맨 앞**(학습자가 이미 학습함). 그 뒤는 Tokenizer(P1-01) → Embedding(P1-02) → Word2Vec(P1-03) → Transformer(P1-05).

| ID | 실습 | 주차 | 구분 | 상태 | 확인일 | 메모 |
|---|---|---|---|---|---|---|
| P1-04 | (선수) PyTorch 기초 | W2 | 메인 | `NOT_STARTED` | | 실습 자료 작성 완료, 학습자 확인 전 |
| P1-01 | Tokenizer: 토큰 수·한국어/영어·비용·토큰 청킹·컨텍스트 | W2 | 메인 | `NOT_STARTED` | | 실습 자료 작성 완료, 학습자 확인 전 |
| P1-01A | Tokenizer 심화: BPE 직접 구현 (`tokenizer/advanced/`) | — | 선택 | `NOT_STARTED` | | 기존 심화 실습 |
| P1-02 | Embedding: 정규화·코사인, OpenAI 임베딩 유사도·최근접 | W3 | 메인 | `NOT_STARTED` | | 실습 자료 작성 완료, 학습자 확인 전 |
| P1-02A | Embedding 심화: 임베딩 층·위치 임베딩 직접 구현 (`embedding/advanced/`) | — | 선택 | `NOT_STARTED` | | 기존 심화 실습 |
| P1-03 | Word2Vec(짧게): 분포 가설, gensim | W3 | 메인 | `NOT_STARTED` | | 실습 자료 작성 완료, 학습자 확인 전 |
| P1-03A | Word2Vec 심화: Skip-gram/CBOW·Negative Sampling 직접 구현 (`word2vec/advanced/`) | — | 선택 | `NOT_STARTED` | | 기존 심화 실습 |
| P1-05 | Transformer: 어텐션 1회 계산, 구조, 컨텍스트 길이와 지연·비용, KV 캐시 | W4 | 메인 | `NOT_STARTED` | | 실습 자료 작성 완료, 학습자 확인 전 |
| P1-05A | Transformer 심화: Self-Attention~미니 GPT 직접 구현·학습 (`transformer/advanced/`) | — | 선택 | `NOT_STARTED` | | 기존 심화 실습 |

## Part 2 — LLM 활용 원리

| ID | 실습 | 주차 | 구분 | 상태 | 확인일 | 메모 |
|---|---|---|---|---|---|---|
| P2-01 | 다음 토큰 예측·샘플링 (로컬 소형 모델) | W5 | 메인 | `NOT_STARTED` | | 실습 자료 작성 완료, 학습자 확인 전 |
| P2-01B | 로컬 LLM 서빙과 공급자 교체 (Ollama, 공통 LLMClient, TTFT·토큰/초·VRAM) | W5(–6) | 메인 | `NOT_STARTED` | |  |
| P2-02 | Claude API 기초: 토큰 세기, 프롬프트 구조, 구조화 출력, 스트리밍 | W5 | 메인 | `NOT_STARTED` | |  |
| P2-03 | 비용·지연: 프롬프트 캐싱, 배치, 토큰 사용량 기록 | W6 | 메인 | `NOT_STARTED` | |  |
| P2-04 | 프롬프트 vs RAG vs Fine-tuning 선택 기준 (+ LoRA 개념) | W6 | 메인 | `NOT_STARTED` | |  |
| P2-S1 | loss·역전파·학습 루프·LoRA 직접 구현 | — | 선택 | `NOT_STARTED` | |  |

## Part 3 — RAG

| ID | 실습 | 주차 | 구분 | 상태 | 확인일 | 메모 |
|---|---|---|---|---|---|---|
| P3-01 | 문서 로딩 | W7 | 메인 | `NOT_STARTED` | |  |
| P3-02 | 청킹 전략 비교 | W7 | 메인 | `NOT_STARTED` | |  |
| P3-03 | Dense 검색 (OpenAI 임베딩 + FAISS) | W8 | 메인 | `NOT_STARTED` | |  |
| P3-04 | Sparse 검색 (BM25) | W8 | 메인 | `NOT_STARTED` | |  |
| P3-05 | Hybrid (RRF, weighted) 직접 구현 | W9 | 메인 | `NOT_STARTED` | |  |
| P3-06 | Reranker 적용 | W9 | 메인 | `NOT_STARTED` | |  |
| P3-07 | 평가 세트 설계 · IR 지표 · 지연·비용 · 4방식/프레임워크 비교 | W10 | 메인 | `NOT_STARTED` | |  |

## Part 4 — GraphRAG

| ID | 실습 | 주차 | 구분 | 상태 | 확인일 | 메모 |
|---|---|---|---|---|---|---|
| P4-01 | Knowledge Graph · Neo4j · Cypher | W11 | 메인 | `NOT_STARTED` | |  |
| P4-02 | 엔티티 추출 (Claude 기반, 규칙 기반 짧은 비교) | W11 | 메인 | `NOT_STARTED` | |  |
| P4-03 | 관계 추출 | W11 | 메인 | `NOT_STARTED` | |  |
| P4-04 | Graph + Vector 검색, Cypher 멀티홉 질의 | W12 | 메인 | `NOT_STARTED` | |  |
| P4-05 | Vector RAG vs GraphRAG 멀티홉 비교 · 선택 기준 | W13 | 메인 | `NOT_STARTED` | |  |
| P4-S1 | 그래프 탐색 알고리즘 직접 구현 | — | 선택 | `NOT_STARTED` | |  |

## Part 5 — Fine-tuning

| ID | 실습 | 주차 | 구분 | 상태 | 확인일 | 메모 |
|---|---|---|---|---|---|---|
| P5-01 | 임베딩: OpenAI 기준선 vs 로컬 fine-tuning 전후 | W14 | 메인 | `NOT_STARTED` | |  |
| P5-02 | Reranker 적용 효과 · Cross-Encoder fine-tuning | W15 | 메인 | `NOT_STARTED` | |  |
| P5-03 | fine-tuning이 언제 값어치가 있는가 판단 정리 | W15 | 메인 | `NOT_STARTED` | |  |
| P5-S1 | LLM LoRA / QLoRA 실행 | — | 선택 | `NOT_STARTED` | |  |

## Part 6 — AI 서비스

| ID | 실습 | 주차 | 구분 | 상태 | 확인일 | 메모 |
|---|---|---|---|---|---|---|
| P6-01 | FastAPI AI 서버 엔드포인트 | W16 | 메인 | `NOT_STARTED` | |  |
| P6-02 | SSE 스트리밍 + Spring Boot WebClient·DTO·SSE 중계 | W17 | 메인 | `NOT_STARTED` | |  |
| P6-03 | 타임아웃·재시도·예외 처리 | W18 | 메인 | `NOT_STARTED` | |  |
| P6-04 | 로깅·모니터링·비용(토큰) 추적 | W19 | 메인 | `NOT_STARTED` | |  |
| P6-05 | Docker Compose | W20 | 메인 | `NOT_STARTED` | |  |
| P6-06 | RAG vs GraphRAG 비교 웹 서비스 통합 | W21 | 메인 | `NOT_STARTED` | |  |

## 포트폴리오

| ID | 실습 | 주차 | 구분 | 상태 | 확인일 | 메모 |
|---|---|---|---|---|---|---|
| PF-01 | 실험 기록·비교표 정리, 회고, 데모, 확장 과제 선정 | W22 | 메인 | `NOT_STARTED` | |  |

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

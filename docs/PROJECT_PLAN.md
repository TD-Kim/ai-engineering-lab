# 프로젝트 기획 (PROJECT PLAN)

## 1. 목적

- **AI 서비스 개발자로서 RAG·GraphRAG 서비스를 설계하고 판단하는 능력**을 실습으로 기른다.
  - 예: 어떤 검색 방식을 쓸지, 언제 GraphRAG가 필요한지, 언제 fine-tuning이 값어치가 있는지, 비용·지연·품질을 어떻게 맞출지
- 원리는 **깊게 이해**하되 구현은 **라이브러리·API를 사용해 얕게** 한다. 원리 직접 구현은 선택이며, 심화 개념 노트는 각 폴더 `advanced/README.md`에 남겨 두었다.
- 최종적으로 Java/Spring Boot 경력과 결합한 **AI 서비스 포트폴리오**를 만든다.
  진행 과정 자체(실험 기록, 비교 수치, 회고)를 포트폴리오 자산으로 관리한다.

### 원칙 변경 이력

| 날짜 | 변경 |
|---|---|
| 2026-11-09 | **API 기반 전환:** LLM은 Claude API, 임베딩은 OpenAI API를 기본으로 쓴다. 로컬 모델은 Part 1·2 개념 실습과 Part 5 fine-tuning에 쓴다 |
| 2026-11-09 | **개념은 깊게, 구현은 얕게 — 서비스 설계 중심:** 기존 Part 1 심화 구현은 각 실습의 `advanced/`(선택 과제)로 옮긴다. 모든 실습에 "서비스 관점"을 넣는다 |
| 2026-11-16 | **학습 방식 전환:** 정해진 실습 순서·starter/solution/tests 방식을 폐지한다. 기획서(방향)만 두고 사용자가 만들고 싶은 것을 골라 **직접 구현**하며, VS Code(Claude Code)와 총괄 세션에 질문한다. 기존 실습 자료는 개념 README만 남기고 코드는 지웠다(복구: 커밋 `aa46d2f`) |

## 2. 학습 환경과 제약

| 항목 | 값 | 비고 |
|---|---|---|
| OS | Windows | DataLoader 사용 스크립트는 `if __name__ == "__main__":` 필수 |
| GPU | RTX 3070, VRAM 8GB | 가능/불가능 구분은 [LEARNING_ROADMAP](LEARNING_ROADMAP.md#2-rtx-3070-8gb-학습-가능-범위) |
| RAM | 32GB | |
| Python | 3.12 권장 | gensim 등의 3.14 휠 부재 때문 |
| 패키지 관리 | uv | 이 저장소 자체 `.venv` 사용 |
| API 키 | `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` | 사용자 PC의 `.env`에만 둔다(커밋 금지). 테스트는 API 호출을 모킹한다 |

- 참고 저장소의 `part1\.venv`(Python 3.14.7)는 **건드리지 않는다.** 새 저장소는 독립 환경을 쓴다.
- 자세한 설치는 [ENVIRONMENT](ENVIRONMENT.md).

## 3. 원칙

1. **개념은 깊게, 구현은 얕게.** 처음 보는 용어도 이해할 수 있도록 코드 주석과 README에서 자세히 설명한다(`# 개념:`). 구현은 라이브러리·API를 쓰고, 실습당 TODO는 3~6개로 둔다. 손 미분·학습 루프 직접 작성 같은 원리 구현은 `advanced/` 선택 과제로 돌린다.
2. **서비스 관점.** 모든 실습에서 "이것이 비용·지연·품질·설계 판단에 어떤 의미인가"를 설명한다(`# 서비스 관점:`).
3. **API 기본, 로컬은 목적이 있을 때만.**
   - LLM(답변 생성, 추출, LLM-as-judge)은 Anthropic Claude API를 쓴다. 모델 ID는 환경 변수로 바꿀 수 있게 한다.
   - 임베딩은 OpenAI API를 쓴다.
   - 로컬 모델은 개념 실습(Part 1·2)과 fine-tuning 비교(Part 5)에 쓴다.
   - 이 클라우드 작업 환경에는 키가 없으므로 실제 호출 결과를 지어내지 않는다. 테스트는 모킹하고, 실제 호출 스크립트는 사용자 PC에서 실행한다.
4. **모델 라이선스 확인.** 로컬 모델은 모델 카드의 라이선스를 기록하고, 비상업 라이선스 모델은 서비스 기본값으로 쓰지 않는다.
5. **개선을 가정하지 않는다.** 모든 "더 좋다"는 동일 데이터셋·동일 평가 세트의 수치로만 말한다.
6. **저작권 존중.** 참고 저장소(강의 실습 코드)의 코드·데이터는 복사하지 않고 링크로만 참고한다.

## 4. Part별 범위

주제 목록과 확인할 질문은 [LEARNING_ROADMAP](LEARNING_ROADMAP.md)에 있다.

### Part 0 — 환경·기획
기획·분석·로드맵·아키텍처·환경 문서, uv 환경, 디렉터리 골격.

### Part 1 — AI 원리 (서비스 개발자에게 필요한 만큼)
- **PyTorch 기초(P1-04):** 학습자가 이미 학습한 선수 실습. 그대로 둔다.
- **Tokenizer(P1-01):** 토큰 수, 한국어/영어 차이, 비용 추정, 토큰 기준 청킹, 컨텍스트 창 맞추기. BPE 원리는 손으로 따라가는 예시로 설명한다.
- **Embedding(P1-02):** 정규화·코사인 유사도, OpenAI 임베딩으로 문장 유사도·최근접 검색.
- **Word2Vec(P1-03):** 분포 가설 개념과 gensim 짧은 실행.
- **Transformer(P1-05):** 어텐션 1회 계산, 구조 그림, 컨텍스트 길이와 지연·비용의 관계, KV 캐시 개념.
- **선택 과제(`advanced/`):** BPE·임베딩 층·Word2Vec·Transformer 직접 구현과 미니 GPT 학습(심화 개념 노트, 코드는 `aa46d2f`).

### Part 2 — LLM 활용 원리 (로컬 소형 모델 + Claude API)
- **개념(로컬 소형 모델):** 다음 토큰 예측, 샘플링(temperature, top-k, top-p), 컨텍스트. Qwen2.5-0.5B 등 Apache-2.0 모델을 쓴다.
- **서비스 활용(Claude API):** 토큰 세기, 프롬프트 구조(시스템 프롬프트, 메시지), 구조화 출력, 스트리밍, 캐싱·배치와 비용.
- **판단:** 프롬프트 vs RAG vs Fine-tuning 선택 기준.
- **선택 과제:** loss·역전파·학습 루프·LoRA 직접 구현.

### Part 3 — RAG
- 문서 로딩, **청킹 전략 비교**, Dense(OpenAI 임베딩)·BM25·Hybrid(RRF, weighted) 검색, Reranker
- **평가 세트 설계**(질문·정답 문서 만들기), IR 지표(Precision/Recall/HitRate@K, MRR, NDCG), **지연·비용 측정**
- **완료 조건:** 동일 데이터셋에서 **BM25 vs Dense vs Hybrid vs Hybrid+Rerank** 수치 비교. 직접 구현한 Hybrid를 프레임워크 구현과 비교(§5-1)

### Part 4 — GraphRAG
- **"언제 GraphRAG가 필요한가" 판단 실험 중심.**
- Knowledge Graph 모델링, Neo4j·Cypher 활용, LLM(Claude) 기반 엔티티·관계 추출
- Graph + Vector 검색(neo4j-graphrag 활용)
- **완료 조건:** Vector RAG vs GraphRAG **멀티홉 질의 정량 비교**와 선택 기준 정리
- **선택 과제:** 그래프 탐색 알고리즘(BFS/DFS/최단경로) 직접 구현

### Part 5 — Fine-tuning (임베딩·Reranker 비교)
- **임베딩:** OpenAI 임베딩 기준선 vs 로컬 임베딩(multilingual-e5, bge-m3) fine-tuning 전후
- **Reranker:** 적용 효과 측정, 로컬 Cross-Encoder(bge-reranker-v2-m3) fine-tuning
- **완료 조건:** 동일 평가 세트로 "fine-tuning이 언제 값어치가 있는가"를 수치(품질·비용·지연·운영 부담)로 판단
- **LLM LoRA/QLoRA:** 개념 설명 + 선택 과제(RTX 3070 범위)

### Part 6 — AI 서비스 (가장 큰 Part)
- FastAPI AI 서버: `/api/v1/documents`, `/api/v1/embeddings`, `/api/v1/search`,
  `/api/v1/rag/query`, `/api/v1/graphrag/query`, `/api/v1/health`, SSE 스트리밍
- Spring Boot: WebClient, DTO, SSE 중계, 예외 처리
- 운영: **타임아웃·재시도**, 로깅·모니터링, **비용(토큰) 추적**, Docker(Compose)
- 흐름: 사용자 → Spring Boot → Python AI Server → RAG/GraphRAG → VectorDB/Neo4j → Claude API
- REST 먼저, gRPC는 확장 과제
- **완료 조건:** RAG vs GraphRAG 비교 웹 서비스

## 5. 사용자 강조 사항

### 5-1. Hybrid Search 직접 구현 후 프레임워크와 비교
직접 구현(BM25 + Dense + RRF/weighted)을 먼저 완성한 뒤 다음과 비교한다.
- LangChain `EnsembleRetriever`
- LlamaIndex `QueryFusionRetriever`
- Haystack 하이브리드 파이프라인(`DocumentJoiner`)
- neo4j-graphrag `HybridRetriever`

비교 항목과 절차는 [LEARNING_ROADMAP §3](LEARNING_ROADMAP.md#3-hybrid-search-비교-계획).

### 5-2. RTX 3070에서 가능한 학습과 불가능한 학습 구분
- LLM **사전학습은 목표에서 제외**한다.
- 작은 Transformer 직접 학습은 **선택 과제**(`part1_ai_fundamentals/transformer/advanced/`)다.
- 로컬 학습은 Part 5의 임베딩·Reranker fine-tuning이 중심이다. LLM LoRA/QLoRA는 선택 과제다.

### 5-3. Spring Boot가 Python AI 서버를 호출하는 구조로 마무리
- 진행 과정 자체를 포트폴리오로 관리하고, **주차별 점검**([PROGRESS](PROGRESS.md#주간-점검-템플릿))을 한다.

## 6. 코드·문서 규칙

### 6-1. 만드는 방식 (2026-11-16 이후)
1. 만들고 싶은 기능을 고른다 → 2. 기획서의 어느 Part인지, [최종 서비스의 어디에 들어가는지](ARCHITECTURE.md#1-최종-서비스-구조) 확인한다 → 3. 개념 README를 읽는다 → 4. **직접 구현**한다 → 5. 막히면 VS Code(Claude Code)나 총괄 세션에 질문한다.
- 코드는 [권장 위치](ARCHITECTURE.md#4-코드-권장-위치-제안)에 둔다(제안, 강제 아님).
- 이전 방식(실습 순서, `starter.py`/`solution.py`/`tests`, 실습당 TODO 3~6개)은 폐지했고 규약은 커밋 `aa46d2f`의 이 문서에 남아 있다.

### 6-2. 개념 README 구성
목표 · 선수지식 · 핵심 개념 · 내부 원리 · 서비스 관점 · 수식 · 자주 나는 오류 · 과제(직접 만들어 볼 것) · 체크리스트 · 다음 단계 · REF. 맨 위에 **"최종 서비스에서의 위치"**를 적는다.

### 6-3. 주석 규약 (권장)
직접 구현한 코드에도 다음 주석 스타일을 권장한다.

| 태그 | 내용 |
|---|---|
| `# 개념:` | 무엇인지 쉬운 말로. 처음 보는 용어를 전제로, 필요하면 작은 숫자 예시 |
| `# WHY:` | 왜 필요한가 / 빠지면 무슨 문제가 생기는가 |
| `# 서비스 관점:` | 비용·지연·품질·설계 판단에서 어떤 의미인가 |
| `# REF:` | 실존하는 논문(저자·연도·절) 또는 공식 문서 URL만. **불확실하면 생략, 지어내기 금지** |

### 6-4. 실험 기록
`experiments/`에 실험마다 다음을 기록한다: 목적, 데이터셋, 모델, 하이퍼파라미터, (해당 시) GPU 사용량·학습 시간, **토큰 사용량·비용**, 결과, 비교, 개선점.
템플릿은 [experiments/README.md](../experiments/README.md).

### 6-5. 진행 기록
[PROGRESS](PROGRESS.md)에 "내가 만든 것"을 기록한다(날짜·만든 기능·Part·배운 점·남은 질문). 상태값을 쓸 때는 `NOT_STARTED` / `IN_PROGRESS` / `COMPLETED` 세 가지만 쓰고, **학습자가 직접 확인한 것만 `COMPLETED`.**

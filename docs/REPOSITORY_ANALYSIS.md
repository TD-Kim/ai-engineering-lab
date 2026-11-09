# 참고 저장소 분석 (REPOSITORY ANALYSIS)

대상: [TD-Kim/graphRAG_learning](https://github.com/TD-Kim/graphRAG_learning)

> **이관 방침(B안): 이관하지 않는다.**
> 참고 저장소는 패스트캠퍼스 GraphRAG 강의 실습 코드이므로 **코드·데이터를 이 저장소로 복사하지 않고 링크로만 참고**한다.
>
> **경로 표기:** 이 문서를 작성할 때 참고 저장소의 파일 경로를 직접 확인하지 못했다.
> 따라서 아래에는 저장소 루트 링크와 **디렉터리 이름 수준**만 적고, 세부 파일 경로는 지어내지 않았다.
> 학습 중 실제 경로를 확인하면 이 문서에 링크를 보강한다.

## 1. 개요

| 항목 | 내용 |
|---|---|
| 성격 | 강의 실습 코드 + 학습자 복습 파일(GUIDE, step1~3) |
| 규모 | Python 약 12,700줄, uv 프로젝트 20개 |
| Python | 20개 중 13개가 Python >= 3.14 요구 |
| 임베딩·LLM | 모두 유료 API(OpenAI, Claude) |

## 2. 있는 것 / 없는 것

| 있는 것 | 없는 것 |
|---|---|
| Neo4j 기초 | 원리 직접 구현(토크나이저, 어텐션 등) |
| KG 구축 6종 | 로컬 모델 사용 |
| neo4j-graphrag 기반 검색 패턴 7종 | BM25 / Sparse 검색 |
| RAGAS 평가 | Reranker |
| LangGraph 에이전트 | IR 지표(Precision/Recall@K, MRR, NDCG) |
| Streamlit 챗봇 | FastAPI 서버, Spring Boot 연동 |
| | 테스트 코드 |

## 3. 배울 점

- **KG 구축 흐름:** 문서 → 엔티티·관계 추출 → Neo4j 적재까지의 전체 파이프라인을 여러 방식으로 보여 준다.
- **검색 패턴 다양성:** neo4j-graphrag의 여러 Retriever(벡터, Cypher, 하이브리드 등)를 실제로 연결해 본 예제가 있다.
- **평가 경험:** RAGAS로 생성 품질을 평가하는 흐름이 있어 Part 3·4 평가 설계의 출발점이 된다.
- **서비스 흐름:** LangGraph 에이전트와 Streamlit 챗봇은 Part 6 UI·대화 흐름 설계에 참고가 된다.

## 4. Part별 참고 매핑

| Part | 참고 여부 | 참고 대상(디렉터리 이름 수준) | 이 저장소에서의 사용 방식 |
|---|---|---|---|
| 1 AI 원리 | 없음 | — | 전부 신규 작성 |
| 2 LLM 학습 원리 | 없음 | — | 전부 신규 작성 |
| 3 RAG | 일부 | `pdf2kg`(문서 로딩), `vectorsearch`(벡터 검색), `ragas`(평가) | 흐름만 참고. **BM25·Hybrid·Rerank·IR 지표는 직접 구현** |
| 4 GraphRAG | 대부분 있음 | KG 구축 6종, neo4j-graphrag 검색 패턴 | **비교군으로만 사용.** 규칙 기반 추출, Python 그래프 탐색, 직접 만든 Graph+Vector 검색, 멀티홉 정량 평가로 비교 |
| 5 Fine-tuning | 없음 | — | 전부 신규 작성 |
| 6 AI 서비스 | 일부 | Streamlit 챗봇, LangGraph 에이전트 | UI·대화 흐름만 참고. FastAPI·Spring Boot는 신규 |

## 5. 다르게 할 점

| 참고 저장소 | 이 저장소 |
|---|---|
| 유료 API(OpenAI, Claude) 전제 | **로컬·오픈소스 모델**(Qwen 계열, e5/bge 임베딩, bge reranker, Ollama) |
| 라이브러리 호출 위주 | **원리 직접 구현 → 라이브러리와 비교** |
| Python >= 3.14 다수 | **Python 3.12** 단일 `.venv` (gensim·faiss 휠 호환) |
| Dense/그래프 검색 중심 | BM25·Hybrid·Rerank까지 포함한 **검색 방식별 수치 비교** |
| RAGAS(생성 품질) 중심 평가 | **IR 지표 + Latency** 를 CSV·그래프로 기록, 멀티홉 정량 평가 |
| 테스트 없음 | `pytest` 기반 starter/solution 검증 |
| Streamlit 단일 앱 | **Spring Boot → FastAPI AI 서버** 구조의 서비스 |

## 6. 주의 사항

- **Neo4j DB 분리:** 참고 저장소의 일부 스크립트는 실행 시 DB 내용을 지운다.
  Part 4 비교 실험은 **별도 Neo4j 데이터베이스(또는 별도 컨테이너)** 에서 수행한다.
- **가상환경 분리:** 참고 저장소의 `part1\.venv`(Python 3.14.7)는 수정하지 않는다.
- **저작권:** 강의 코드·데이터를 복사·재배포하지 않는다. 필요하면 링크와 "무엇을 참고했는지"만 기록한다.

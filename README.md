# AI Engineering Learning Lab

AI 엔지니어링을 **실습형으로** 학습하는 저장소입니다.
라이브러리를 "쓰는 법"과 그 내부 원리를 "직접 구현하는 법"을 함께 익히고,
최종적으로 Java/Spring Boot 경력과 결합한 **AI 서비스 포트폴리오**로 마무리합니다.

> 원칙: 무료 오픈소스·로컬 모델 우선 · 유료 API 전제 금지 · 모델 라이선스 확인 ·
> 기술은 필요한 단계에서만 도입 · "개선되었다"는 수치로만 말한다.

## 학습 환경

| 항목 | 값 |
|---|---|
| OS | Windows |
| GPU | NVIDIA RTX 3070 (VRAM 8GB) |
| RAM | 32GB |
| Python | 3.12 (저장소 자체 `.venv`, uv 관리) |
| 패키지 관리 | [uv](https://docs.astral.sh/uv/) |

설치 절차는 [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)를 보세요.

## 로드맵 한눈에 보기

| Part | 주제 | 주차 | 핵심 산출물 |
|---|---|---|---|
| 0 | 환경·기획 | W1 | 이 문서들, uv 환경, GPU 검증 |
| 1 | [AI 원리](part1_ai_fundamentals/README.md) | W2–6 | BPE·Word2Vec·Transformer 직접 구현, 미니 GPT |
| 2 | [LLM 학습 원리](part2_llm/README.md) | W7–10 | 소형 LM 학습, LoRA 원리 |
| 3 | [RAG](part3_rag/README.md) | W11–15 | BM25 vs Dense vs Hybrid vs Hybrid+Rerank 수치 비교 |
| 4 | [GraphRAG](part4_graphrag/README.md) | W16–19 | Vector RAG vs GraphRAG 멀티홉 평가 |
| 5 | [Fine-tuning](part5_finetuning/README.md) | W20–23 | 임베딩·Reranker·LLM LoRA 전후 비교 |
| 6 | [AI 서비스](part6_ai_service/README.md) | W24–28 | FastAPI AI 서버 + Spring Boot, 비교 웹 서비스 |
| — | 포트폴리오 정리 | W29–30 | 실험 기록·회고·데모 |

자세한 주차별 계획은 [docs/LEARNING_ROADMAP.md](docs/LEARNING_ROADMAP.md)에 있습니다.

## 문서

| 문서 | 내용 |
|---|---|
| [PROJECT_PLAN](docs/PROJECT_PLAN.md) | 목적, 원칙, Part별 범위, 코드·문서 규칙 |
| [REPOSITORY_ANALYSIS](docs/REPOSITORY_ANALYSIS.md) | 참고 저장소(graphRAG_learning) 분석: 배울 점, Part별 참고, 다르게 할 점 |
| [LEARNING_ROADMAP](docs/LEARNING_ROADMAP.md) | 30주 일정, RTX 3070 가능/불가능 표, Hybrid 비교 계획 |
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

## 참고

- 참고 저장소: [TD-Kim/graphRAG_learning](https://github.com/TD-Kim/graphRAG_learning) —
  강의 실습 코드이므로 **코드·데이터를 복사하지 않고 링크로만 참고**합니다.

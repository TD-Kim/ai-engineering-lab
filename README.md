# AI Engineering Learning Lab

AI 서비스 개발자로서 **RAG·GraphRAG 서비스를 설계하고 판단하는 능력**을 실습으로 기르는 저장소입니다.
원리는 자세한 설명으로 깊게 이해하고, 구현은 라이브러리·API로 얕게 합니다(원리 직접 구현은 `advanced/` 선택 과제).
최종적으로 Java/Spring Boot 경력과 결합한 **AI 서비스 포트폴리오**로 마무리합니다.

> 원칙: 개념은 깊게, 구현은 얕게(서비스 설계 중심) · LLM은 Claude API, 임베딩은 OpenAI API 기본 ·
> 로컬 모델은 개념 실습과 fine-tuning 비교에 · 모델 라이선스 확인 · "개선되었다"는 수치로만 말한다.

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

방향: **개념은 깊게, 구현은 얕게 — 서비스 설계 중심.** 원리 직접 구현은 각 실습의 `advanced/` 선택 과제로 둔다.

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

자세한 주차별 계획은 [docs/LEARNING_ROADMAP.md](docs/LEARNING_ROADMAP.md)에 있습니다.

## 문서

| 문서 | 내용 |
|---|---|
| [PROJECT_PLAN](docs/PROJECT_PLAN.md) | 목적, 원칙, Part별 범위, 코드·문서 규칙 |
| [REPOSITORY_ANALYSIS](docs/REPOSITORY_ANALYSIS.md) | 참고 저장소(graphRAG_learning) 분석: 배울 점, Part별 참고, 다르게 할 점 |
| [LEARNING_ROADMAP](docs/LEARNING_ROADMAP.md) | 22주 일정(서비스 설계 중심), RTX 3070 가능/불가능 표, Hybrid 비교 계획 |
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

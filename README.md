# AI Engineering Learning Lab

AI 서비스 개발자로서 **RAG·GraphRAG 서비스를 설계하고 판단하는 능력**을 기르는 저장소입니다.
만들고 싶은 기능을 골라 직접 구현하며 배우고, 최종적으로 **Spring Boot가 Python AI 서버를 호출하는 서비스**로 마무리합니다.

## 시작하기

1. **기획서를 읽습니다:** [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md)
   - 목적과 학습 방식(§1~2), 최종 서비스 구조(§3), 주제 지도(§4), 환경 설치(§7)
2. **만들고 싶은 기능을 고르고**, 최종 서비스의 어디에 들어가는지 확인합니다.
3. **직접 구현하고**, 막히면 VS Code의 Claude Code에 질문합니다. 질문 규칙은 [CLAUDE.md](CLAUDE.md)에 있습니다(정답 대신 힌트부터).

## 구조

| 폴더·파일 | 내용 |
|---|---|
| [python_ai_server/](python_ai_server/README.md) | Python AI 서버 (`ai_server` 패키지: llm · embeddings · chunking · retrieval · graph · api · observability) |
| [spring_boot_server/](spring_boot_server/README.md) | Spring Boot 서버 (나중에 시작) |
| [experiments/](experiments/README.md) | 한 번 해 보는 연습·측정 |
| [data/](data/README.md) | 샘플 문서·평가 세트 |
| [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) | 기획서 (방향·구조·주제·환경·결정 이력) |
| [docs/TIMELINE.md](docs/TIMELINE.md) | 타임라인: 언제 무엇을 만들고 어떤 함수를 썼나 (커밋할 때 자동 기록) |
| [CLAUDE.md](CLAUDE.md) | Claude Code 학습 도우미 규칙 |
| [.env.example](.env.example) | API 키·모델 설정 견본 (`Copy-Item .env.example .env` 후 값 채우기, `.env`는 커밋 금지) |

## 처음 한 번 설정 (PowerShell, 저장소 루트)

```powershell
.\.venv\Scripts\Activate.ps1
uv pip install -e python_ai_server        # ai_server 패키지 설치 (코드 수정이 바로 반영됨)
git config core.hooksPath .githooks       # 커밋할 때 타임라인 자동 기록 켜기
```

이후에는 평소처럼 커밋하면 `docs/TIMELINE.md`에 기록이 붙습니다. 커밋 메시지는 "무엇을 만들었는지" 알 수 있게 쓰세요(타임라인 제목이 됩니다).

## 이전 자료

이 저장소는 2026-11-16에 기획서만 남기고 정리했습니다. 이전 자료는 git 기록에 남아 있습니다.

- 실습 코드(starter·solution·tests)가 있던 상태: `git checkout aa46d2f`
- 개념 README가 있던 상태: `git checkout 26f20bd`

# AI Engineering Learning Lab

AI 서비스 개발자로서 **RAG·GraphRAG 서비스를 설계하고 판단하는 능력**을 기르는 저장소입니다.
만들고 싶은 기능을 골라 직접 구현하며 배우고, 최종적으로 **Spring Boot가 Python AI 서버를 호출하는 서비스**로 마무리합니다.

## 시작하기

1. **기획서를 읽습니다:** [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md)
   - 목적과 학습 방식(§1~2), 최종 서비스 구조(§3), 주제 지도(§4), 환경 설치(§7)
2. **만들고 싶은 기능을 고르고**, 최종 서비스의 어디에 들어가는지 확인합니다.
3. **직접 구현하고**, 막히면 VS Code의 Claude Code에 질문합니다. 질문 규칙은 [CLAUDE.md](CLAUDE.md)에 있습니다(정답 대신 힌트부터).

## 파일

| 파일 | 내용 |
|---|---|
| [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) | 기획서 (방향·구조·주제·환경·결정 이력) |
| [CLAUDE.md](CLAUDE.md) | Claude Code 학습 도우미 규칙 |
| [.env.example](.env.example) | API 키·모델 설정 견본 (`Copy-Item .env.example .env` 후 값 채우기, `.env`는 커밋 금지) |

## 이전 자료

이 저장소는 2026-11-16에 기획서만 남기고 정리했습니다. 이전 자료는 git 기록에 남아 있습니다.

- 실습 코드(starter·solution·tests)가 있던 상태: `git checkout aa46d2f`
- 개념 README가 있던 상태: `git checkout 26f20bd`

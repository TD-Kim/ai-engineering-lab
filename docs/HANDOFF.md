# 인계 메모 (총괄 세션용)

> 총괄 세션(claude.ai 클라우드 세션)이 `/clear` 뒤에 맥락을 되찾기 위한 짧은 메모입니다.
> 학습 내용 자체는 [PROJECT_PLAN.md](PROJECT_PLAN.md), 만든 것의 기록은 [TIMELINE.md](TIMELINE.md)에 있습니다.
> 마지막 갱신: 2026-11-17

## 1. 지금 상태

- **학습 방식:** 정해진 실습 없이, 사용자가 기획서 방향 안에서 만들고 싶은 것을 골라 **직접 구현**한다.
  - 코드 질문은 사용자 PC의 VS Code(Claude Code)에서, 기획·방향 질문은 총괄 세션에서 한다.
  - 힌트는 [CLAUDE.md](../CLAUDE.md) §3의 5단계(질문 → 개념 → 의사코드 → 작은 예제 → 정답은 요청 시만).
- **지금 만드는 것:** 로컬 LLM 대화 — `python_ai_server/src/ai_server/llm/ollama_chat.py` (사용자가 작성 시작, Ollama `/api/chat`)
  - 안내했던 단계: 0) `ollama run qwen2.5:1.5b`로 확인 → 1) 질문 하나 보내고 답 받기 → 2) 대화 기록을 쌓아 이어 가기 → 3) 스트리밍과 TTFT → 4) 시스템 프롬프트(선택) → 다음: Claude와 같은 모양의 LLM 어댑터로 감싸기
  - 11-17 진행: `chat()` 작성, 문서 3개를 시스템 프롬프트에 넣은 간단한 RAG 테스트(`c405f56`). 문서가 1개면 규칙을 지키고, 3개면 규칙을 벗어난 답이 나옴
  - **다음(사용자가 정함): 규칙 강화부터.** 그다음 근거(문서 번호) 인용 요구 → 더 큰 모델과 비교. 1단계 힌트(방향 질문)까지 줌: 어긋난 답의 모양, 규칙의 위치, 없는 내용 질문에 대한 지시
- **세션:** 총괄 세션 하나만 쓴다. 이전의 작업 세션 ②·검증 세션 ③은 보관됨(더 만들지 않음).
- **저장소:** `TD-Kim/ai-engineering-lab`, `main`에 직접 커밋. 구조는 기획서 §3-2.

## 2. 사용자 선호

- 한국어, 처음 배우는 사람 기준의 쉬운 설명과 작은 숫자 예시.
- 순서는 사용자가 정한다. 의존 관계(RAG 전 임베딩 개념)만 알려 준다.
- 정답 코드는 요청할 때만. 파일은 요청 없이 고치지 않는다.
- 로컬 LLM·로컬 임베딩도 쓰고 싶어 한다(공급자 분리 설계). Claude는 LLM, OpenAI는 임베딩 기본.
- 실행하지 않은 결과·숫자는 지어내지 않는다. 확인 못 한 것은 "미확인".

## 3. 사용자 PC에서 아직 할 일

- `git pull`, `uv pip install -e python_ai_server`, `git config core.hooksPath .githooks`(타임라인 자동 기록)
- 남은 `part1_ai_fundamentals`, `part2_llm` 폴더 확인 후 정리(직접 쓴 파일이 있으면 옮기기)
- 복구용 태그(이 클라우드 환경에서는 태그 push가 403): `git tag labs-v1 aa46d2f; git push origin labs-v1`
- Ollama 설치·모델 측정(TTFT·토큰/초·VRAM)
- 이전 개념 메모: e5 임베딩의 `query:`/`passage:` 접두어는 미확인(모델 카드 확인 필요)

## 4. 이 클라우드 환경에서 알아 둘 것

- 클론 위치: `/home/user/ai-engineering-lab`. 커밋하면 훅이 `docs/TIMELINE.md`에 KST 기준으로 기록한다(`core.hooksPath=.githooks` 설정됨).
- 원격 브랜치 삭제와 태그 push는 이 환경에서 막혀 있다(HTTP 403). 사용자 PC에서 해야 한다.
- 하위 세션의 저장소 연결(add_repo)·세션 간 파일 전송은 권한 검사에서 거부된 적이 있다.
- 참고 저장소 `TD-Kim/graphRAG_learning`은 읽기 전용 참고용. 그 저장소의 `claude/laughing-knuth-ey4pwi` 브랜치는 사용자가 남겨 두기로 함.
- 이전 자료: 실습 코드 `aa46d2f`, 개념 README `26f20bd`.

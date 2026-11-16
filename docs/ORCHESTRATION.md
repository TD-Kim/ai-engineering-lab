# ORCHESTRATION — 세션 운영 규약

> **2026-11-16 학습 방식 전환 이후의 운영**을 먼저 적고, 이전 방식(실습 자료를 ②·③이 만들고 검증하던 방식)은 §4 "이전 방식 (labs-v1)"에 짧게 남깁니다. 현재 상태·인계는 [HANDOFF.md](HANDOFF.md), 학습자용 규칙은 [CLAUDE.md](../CLAUDE.md).

## 1. 지금의 운영

| 누가 | 하는 일 | 하지 않는 일 |
|---|---|---|
| **사용자** | 만들고 싶은 것을 골라 **VS Code(로컬 Claude Code)에서 직접 구현**, [PROGRESS](PROGRESS.md)에 기록 | |
| **VS Code의 Claude Code** | [CLAUDE.md](../CLAUDE.md) 규칙대로 질문에 답하기: 최종 서비스의 어디에 쓰이는지 먼저 짚고, 힌트는 방향 → API 이름 → 예시 순, 정답 코드는 요청할 때만, 코드 리뷰는 요청 시 이유 설명 | 요청 없이 사용자 코드 수정 |
| **① 총괄 세션** | **기획 점검·질문 응답**: 방향(서비스 설계 중심, 개념은 깊게·구현은 얕게) 점검, 다음에 만들 것 추천 | 코드를 대신 짜 주기 |
| ② 작업 · ③ 검증 | **보관됨**(2026-11-16). 실습 자료(starter·solution·tests)를 만들고 검증하는 일은 더 하지 않음 | |

- 한 번에 정해진 실습 하나씩 진행하는 흐름은 폐지했습니다.
- 파일은 git(`main`)으로만 주고받습니다.

## 2. 사용자 결정 이력 (바꾸지 말 것)

| 결정 | 내용 |
|---|---|
| 방향 전환(2026-11-09) | **개념은 깊게, 구현은 얕게 — 서비스 설계 중심.** 목표: RAG·GraphRAG 서비스를 설계·판단하는 능력 |
| Part 2 | 로컬 소형 모델(개념) + Claude API(서비스 활용) 둘 다. 프롬프트 vs RAG vs Fine-tuning 선택 기준 포함 |
| Part 5 | 임베딩·Reranker fine-tuning 비교만(OpenAI 기준선 vs 로컬 fine-tuning 전후, "언제 값어치가 있나"). LLM LoRA 는 개념 + 선택 |
| 공급자 | LLM = Anthropic(`claude-opus-5-5`, `LLM_MODEL` 로 변경), 임베딩 = OpenAI(`text-embedding-3-small`, `EMBEDDING_MODEL`) 기본 + **로컬 임베딩 옵션**(`EMBEDDING_PROVIDER` = fake·openai·sentence_transformers·ollama, 2026-11-16: 키 없이도 가능하게) |
| 키 | 저장소 루트 `.env`에만(커밋 금지), `.env.example` 참고 |
| 진행 상태 | `NOT_STARTED`/`IN_PROGRESS`/`COMPLETED`만. `COMPLETED`는 사용자가 확인한 뒤에만 |
| 참고 저장소 | `TD-Kim/graphRAG_learning` 은 참고용 **읽기 전용**(푸시 금지). 브랜치 `claude/laughing-knuth-ey4pwi` 는 사용자가 직접 삭제 예정(보류) |
| **학습 방식(2026-11-16)** | **기획서(방향)만 두고 사용자가 만들고 싶은 것을 그때그때 골라 직접 구현.** 정해진 실습 순서·starter 방식 폐지. 기존 실습 자료는 개념 README만 남기고 코드는 삭제(커밋 `aa46d2f`로 복구). ②·③ 세션은 이 작업 후 보관 |

## 3. 환경 제약 · 알려진 함정 (계속 유효)

- **네트워크(클라우드 세션)**
  - 접속 불가: pytorch.org·download.pytorch.org, huggingface.co, platform.openai.com, numpy.org, radimrehurek.com(gensim), arxiv.org, ollama.com·docs.ollama.com, docs.python.org, learn.microsoft.com.
  - 접속 가능: `raw.githubusercontent.com`(Ollama·FlagEmbedding 문서), `platform.claude.com/docs/...md`(가격·캐싱·배치).
  - 확인하지 못한 출처는 "미확인"으로 표기한다.
  - 작업 환경에서 **태그 push가 HTTP 403**으로 거부된다(브랜치 push는 됨). 태그가 필요하면 사용자 PC에서 만든다.
- **API 결과 지어내지 않기**: 키·GPU·모델이 없는 환경에서는 실제 호출·측정 값을 쓰지 않는다. 측정 칸은 "측정 필요"로 비운다.
- **Anthropic SDK 코드**: 반드시 claude-api 스킬과 설치된 패키지(anthropic 2.6.0)로 확인한다. 쓰지 않는 것:
  - `temperature`·`top_p`·`top_k`(API 버전 2026-10-21 에서 400)
  - `budget_tokens`, `thinking: {"type": "disabled"}`(Opus 5.5 400)
  - assistant prefill, 강제 `tool_choice`
  - 스킬 예시와 SDK 가 다를 때는 설치된 SDK 타입·공식 문서를 따른다. 예: 배치 errored 원인은 `result.result.error.error.type == "invalid_request_error"`.
- **사용자가 직접 만든 파일은 수정하지 않는다**: `part1_ai_fundamentals/pytorch/starter.py`, `part1_ai_fundamentals/tokenizer/starter.py`(그리고 같은 사용자 커밋의 `pytorch/solution.py`, `check_gpu.py`).
- **테스트를 직접 만들 때**: 스레드·서버·무한 대기를 쓰지 않는다(이전에 스레드 시간 제한 가드가 버려진 스레드로 8.6GB 메모리를 먹은 사례). 호출 횟수 가드나 길이가 정해진 `for`를 쓴다. 외부 API는 가짜 객체로 대체한다.

## 4. 이전 방식 (labs-v1) — 짧은 기록

> 2026-11-16까지의 운영입니다. 코드·테스트·상세 규약은 커밋 `aa46d2f`(와 그 이전)에 남아 있습니다.

- **역할**: ① 총괄(사용자와 소통, 지시, ③ 판정 전달) · ② 작업(실습 코드·문서 작성, 테스트·ruff, 커밋·푸시, HANDOFF 갱신) · ③ 검증(푸시된 커밋을 읽기 전용으로 검수, 변이 검사 포함, `PASS`/`PASS_WITH_NOTES`/`FAIL` 판정).
- **세션 ID(2026-11-16)**: ① `session_01X4hFs1UCHqKvdWQBdNx9nq`, ② `session_01X4hFrztTmSsMhYoLT1ToF1`, ③ `session_01X4hFrzB2tvoitkbEecYem6` (②·③ 보관됨).
- **흐름**: 사용자 "다음" → ① 지시 → ② 구현·푸시 → ③ 검증 → ① 판정 전달 → 수정·재검증. 한 번에 한 실습.
- **만들어진 실습**: P1-04 PyTorch, P1-01 Tokenizer, P1-02 Embedding(로컬 임베딩 옵션 포함), P1-03 Word2Vec, P1-05 Transformer, P2-01 샘플링, P2-01B 로컬 LLM 서빙, P2-02 Claude API 기초, P2-03 비용·지연. 모두 ③ 검증 `PASS` 또는 `PASS_WITH_NOTES`(권장 사항 반영 완료)였고 상태는 `NOT_STARTED`("학습자 확인 전")였다.
- **이 방식에서 얻은 교훈(개념 README에 반영)**: 스킬 예시와 SDK 타입이 다를 수 있다(배치 errored), 캐시 단가는 공식 가격표로 확인, e5 접두어는 모델 카드 확인(미확인 표기), 정규식 `$`는 끝 줄바꿈을 통과시킨다(`fullmatch`).

# HANDOFF — 인계 문서

> 새 세션은 이 문서와 [ORCHESTRATION.md](ORCHESTRATION.md), [CLAUDE.md](../CLAUDE.md)를 먼저 읽는다. **2026-11-16 학습 방식 전환**으로 작업 세션 ②·검증 세션 ③은 보관되었고, 운영은 **① 총괄(질문·기획 점검) + 사용자 VS Code(Claude Code)** 입니다.

## 1. 현재 상태 (2026-11-16)

- **방식**: 기획서(방향)만 두고, 사용자가 만들고 싶은 것을 골라 직접 구현. 정해진 실습 순서·starter 방식은 폐지. 안내는 [루트 README](../README.md#지금-어디를-보면-되나-학습-방식).
- **남은 것**: 개념 README(part1·part2 + 각 `advanced/` 심화 노트), 자리표시자 README(part3~6), 문서(docs/), `requirements/`, `.env.example`, `data/corpus_ko_en.txt`.
- **남은 .py 파일(4개, 모두 사용자가 커밋한 파일, 변경 없음)**: `part1_ai_fundamentals/pytorch/starter.py`(a4fdc61), `part1_ai_fundamentals/tokenizer/starter.py`(60349ae), `part1_ai_fundamentals/pytorch/solution.py`(a4fdc61, 사용자가 공백 1줄 수정), `check_gpu.py`(a4fdc61).
- **삭제한 것**: part1·part2 의 starter/solution/tests/advanced 코드, 선택 스크립트(run_demo·embed_with_*·call_claude·measure_* 등), conftest.py, 루트 `tests/`·`pytest.ini`, 코드에 묶인 `experiments/` 실습 기록 → 총 80개 파일(커밋 `2d20447`). 이어서 개념 README 정리, 문서 갱신.

## 2. 이전 실습 복구

이전 방식의 마지막 상태는 커밋 **`aa46d2f`**입니다(`git checkout aa46d2f`, 돌아오려면 `git switch main`).

- 작업 환경에서 `labs-v1` 태그 push 가 HTTP 403 으로 거부되어 **태그는 원격에 없습니다.** PC에서 `git tag labs-v1 aa46d2f && git push origin labs-v1`.
- 복구되는 것: 9개 실습(P1-04·P1-01·P1-02(로컬 임베딩 포함)·P1-03·P1-05·P2-01·P2-01B·P2-02·P2-03)의 starter·solution·tests·선택 스크립트, experiments 기록, 옛 문서(주차표 포함).

## 3. 이전 방식에서 완료된 일 (요약)

| 작업 | 커밋 | ③ 판정 |
|---|---|---|
| 방향 전환 1~3단계(기획 문서, advanced/ 이동, Part 1 메인 5개) | 99a40ff · f655119 · 16ecd91 … 093a0fa | PASS 또는 PASS_WITH_NOTES(반영 완료) |
| P2-01 샘플링, P2-01B 로컬 LLM 서빙 | e401983/f87c91a, 6ac51fd | 반영 완료 |
| P2-02 Claude API 기초 | 1e86636 → d01b39d → c2aded7 | PASS |
| P2-03 비용·지연 | d1d6fbd → bf7855b | PASS |
| P1-02 로컬 임베딩 옵션 + 학습 순서 문서 | dfbc0cb · 6f9dc58 · aa46d2f | 권장 5건 반영(재검증 요청 중 방식 전환) |
| CLAUDE.md(학습 도우미 규칙) | c72bc95 | PASS |

모든 이전 실습의 상태는 `NOT_STARTED`("학습자 확인 전")였고, 상태 갱신은 학습자 확인 후입니다([PROGRESS](PROGRESS.md)).

## 4. 열려 있는 것

- `labs-v1` 태그를 PC에서 만들어 push(위 §2).
- 브랜치 `claude/laughing-knuth-ey4pwi`(참고 저장소 `TD-Kim/graphRAG_learning`)는 사용자가 직접 삭제 예정.
- P1-04·P1-01 상태값 확인은 사용자가 한다(현재 표기: [PROGRESS](PROGRESS.md) "내가 만든 것 기록").
- 이전 실습의 "사용자 PC 측정 필요" 칸(Ollama·VRAM, Qwen, OpenAI/로컬 임베딩, Claude 실제 호출 등)은 옛 기록(`aa46d2f`)에만 있고 비어 있었다. 직접 측정하면 [experiments 템플릿](../experiments/README.md)에 새로 기록한다.
- 보류 메모(`aa46d2f` 이전 HANDOFF §3): 격리 테스트 세분화, advanced 스크립트 경로 로딩 — 코드가 삭제되어 **무효**.
- `docs/REPOSITORY_ANALYSIS.md`의 일부 표현("`pytest` 기반 starter/solution 검증" 등)은 이전 방식 기준이다(참고 저장소 분석 문서라 그대로 둠).

## 5. 다음에 할 일

사용자가 만들고 싶은 것을 고릅니다. 추천은 ①(또는 VS Code의 Claude Code)이 [LEARNING_ROADMAP](LEARNING_ROADMAP.md)의 주제 지도와 [최종 서비스 구조](ARCHITECTURE.md#1-최종-서비스-구조)를 기준으로 합니다. 코드는 [권장 위치](ARCHITECTURE.md#4-코드-권장-위치-제안)에 둡니다.

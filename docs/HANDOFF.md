# HANDOFF — 방향 전환(2026-11-09) 이후

방향: **개념은 깊게, 구현은 얕게 — 서비스 설계 중심.** 목표는 RAG·GraphRAG 서비스를 설계·판단하는 능력이다.
진행 방식: 한 단계(실습)씩 완성한 뒤 멈춘다. 다음 실습은 사용자가 "다음"이라고 할 때 ①(총괄 세션)의 지시를 받고 시작한다.

## 방향 전환 단계

| 단계 | 내용 | 커밋 | 상태 |
|---|---|---|---|
| 1 | 기획 문서 수정 (PROJECT_PLAN, LEARNING_ROADMAP 22주, PROGRESS, ARCHITECTURE, README들) | 99a40ff | 완료(검증 대기) |
| 2 | 기존 Part 1 심화 실습(tokenizer·embedding·word2vec·transformer)을 각 `<실습>/advanced/`로 이동(24개 파일) | f655119 | 완료(검증 대기) |
| 3 | 새 Tokenizer 메인 실습 P1-01 | 이 HANDOFF 를 갱신한 커밋 | 완료(검증 대기) |

## Part 1 실습 상태

| 실습 | 메인 | 선택 과제(advanced/) |
|---|---|---|
| P1-04 pytorch | 그대로 유지(③ PASS_WITH_NOTES). 사용자가 starter.py 를 직접 채움(a4fdc61) | — |
| P1-01 tokenizer | **새로 작성(3단계)**. TODO 5개: 토큰 수, 한/영 비교, 비용 추정, 토큰 예산 청크(오버랩), 컨텍스트 맞춤. solution 12 passed / starter 11 skipped + 1 passed(가짜 Claude 클라이언트 테스트, impl 무관). README 12항목 + 서비스 관점, experiments/P1-01-20261109-tokenizer | P1-01A: BPE 직접 구현(기존) |
| P1-02 embedding | 자리표시 README(메인 작성 예정) | P1-02A(기존) |
| P1-03 word2vec | 자리표시 README(메인: 개념 + gensim 짧은 실행 예정) | P1-03A(기존) |
| P1-05 transformer | 자리표시 README(메인 작성 예정) | P1-05A(기존) |

`docs/PROGRESS.md`는 모두 `NOT_STARTED`로 유지한다. 학습자가 직접 확인해야 `COMPLETED`가 된다.

## P1-01 메인 실습 메모
- 토크나이저: HF `tokenizers` 바이트 단위 BPE 를 저장소 말뭉치로 학습(다운로드 없음, vocab 500).
- 한글 한 글자가 토큰 여러 개가 되고, 토큰 하나가 두 글자에 걸칠 수 있다(offsets 겹침). 청크·자르기는 offsets 로 원문을 자르고, 제공 함수 `is_char_boundary`로 글자 경계에서만 자른다. 겹침 0 청크를 이어 붙이면 원문과 같다.
- 토큰 수 기준은 "원문 전체 인코딩"이다. 조각만 다시 인코딩하면 경계에서 몇 토큰 달라질 수 있음을 README 에 명시했다.
- 선택 스크립트: `count_with_claude.py`(`messages.count_tokens`, `.env`의 `ANTHROPIC_API_KEY`, 모델은 `LLM_MODEL` 기본 `claude-opus-5-5`), `pretrained_tokenizer.py`(Qwen/Qwen2.5-0.5B tokenizer.json). 클라우드 환경에서는 키·인터넷이 없어 둘 다 "건너뜀" → **사용자 PC에서 실행 필요**.
- 단가는 공식 가격 페이지 기준 "작성 시점 2026-10"으로 표기했다.
- requirements/part1.txt 에 선택 의존성 anthropic, python-dotenv 추가.

## 공통 규약 (Part 1)
- 테스트는 `part1_ai_fundamentals/conftest.py`의 `impl` 픽스처를 쓴다.
  - 실습 디렉터리(또는 `advanced/`)의 `{IMPL}.py`를 파일 경로로, `part1_` + 상대 경로 기반 고유 모듈명으로 불러온다(예: `part1_tokenizer_advanced_solution`).
  - starter에서 `NotImplementedError`가 나면 skip("TODO 미구현: TODO n")으로 보고한다.
- 테스트 파일명: 메인 `test_p1_0x_<이름>.py`, 선택 과제 `test_p1_0x_<이름>_advanced.py`.
- API 호출은 테스트에서 가짜 클라이언트로 대체한다. 키는 `.env`에만, 커밋 금지.
- 루트 `pytest`: 88 passed, 1 skipped(P1-02A 사전학습 비교, sentence-transformers 미설치).
- ruff: 사용자가 채운 `part1_ai_fundamentals/pytorch/starter.py`에 RUF100(쓰지 않는 noqa) 1건이 있어 `--exclude`로 검사했다. 사용자 학습 파일이라 수정하지 않았다.

## 환경 메모
- 클라우드: download.pytorch.org·huggingface.co 차단. PyPI torch 2.15.0+cu132(CPU).
- `.venv`: Python 3.12.13, numpy 2.5.4, gensim 4.4.0, tokenizers 0.23.3, anthropic 2.6.0, python-dotenv.

## 보류
- Part 0 문서: REPOSITORY_ANALYSIS 실제 경로, 실험 디렉터리 이름 통일.
- 정리됨: `.env.example`(ANTHROPIC_API_KEY·OPENAI_API_KEY 빈 값, LLM_MODEL, EMBEDDING_MODEL) 추가, 루트 README·ENVIRONMENT §2·P1-01 README §6 에 복사 안내.

## ③ 검증 반영 (16ecd91 PASS_WITH_NOTES)
- `.env.example` 추가와 안내.
- 출력 단가가 비싼 이유 정정: 토큰당 연산량은 비슷, 입력은 병렬 prefill·출력은 순차 decode → 처리량 차이.
- TODO 4 가볍게: 글자 경계·과대 글자·무한 반복 방지를 제공 함수 `snap_chunk_end`·`snap_next_start`로 분리. 학습자는 start/end/overlap 이동만. TODO 총 5개 유지.
- 청크 테스트에 시간 제한(스레드 10초)과 조각 수 상한 → 무한 반복 구현은 멈추지 않고 실패 메시지.
- HINT 1 문구를 "글자 경계를 확인하지 않고 decode 하면 한글이 깨질 수 있음"으로 조건 명시.

## 다음에 재개할 첫 단계
사용자가 "다음"이라고 하면 P1-02 Embedding 메인 실습을 시작한다(①의 지시 확인 후).

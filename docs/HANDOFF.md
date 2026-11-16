# HANDOFF — 작업 세션(②) 인계 문서

> 새 세션은 이 문서와 [ORCHESTRATION.md](ORCHESTRATION.md)를 먼저 읽는다. 세션 역할·메시지 규약·사용자 결정 이력·알려진 함정은 ORCHESTRATION에 있다.
> **인계 시점 main = `c72bc95`** (이 문서를 갱신한 문서 전용 커밋 직전, 2026-11-16).
> 방향: **개념은 깊게, 구현은 얕게 — 서비스 설계 중심.** 한 번에 한 실습만 만들고 멈춘다. 다음 실습은 사용자가 "다음"이라고 할 때 ①의 지시로 시작한다.

## 1. 완료된 작업 (최종 커밋 · ③ 판정)

| 작업 | 커밋(최초 → 검증 반영) | ③ 판정 | 비고 |
|---|---|---|---|
| P1-04 PyTorch 기초 | 791e412 등(방향 전환 전) | PASS_WITH_NOTES → 반영 | 사용자가 starter 를 직접 채움(a4fdc61). 손대지 않음 |
| 방향 전환 1단계: 기획 문서 | 99a40ff | (3단계와 함께 검증) | PROJECT_PLAN·LEARNING_ROADMAP(22주)·PROGRESS·ARCHITECTURE·README |
| 방향 전환 2단계: Part 1 심화 → `advanced/` | f655119 | (3단계와 함께 검증) | 24개 파일 git mv, 테스트명 `*_advanced.py` |
| P1-01 Tokenizer (메인) | 16ecd91 → a8d675d → c9f495a | PASS_WITH_NOTES ×2 → 반영 후 재검증 요청, 이후 다음 실습 진행 | 청크 무한 반복 가드는 호출 횟수 방식(스레드 금지) |
| P1-02 Embedding (메인) | e4b03e6 | PASS (사소 2건은 a8ef12d 에 반영) | |
| P1-03 Word2Vec (메인, 짧게) | a8ef12d | PASS (사소 1건은 7204c31 에 반영) | |
| P1-05 Transformer (메인) | 7204c31 → 093a0fa | PASS_WITH_NOTES → 반영 후 재검증 요청, 이후 다음 실습 진행 | Part 1 메인 5개 완료 |
| P2-01 다음 토큰 예측·샘플링 | e401983 → f87c91a | PASS_WITH_NOTES → 반영 | `part2_llm/conftest.py`, pytest.ini 에 part2_llm 추가 |
| P2-01B 로컬 LLM 서빙·공급자 교체 | 93dce4d(문서) + 6ac51fd | 검증 완료, 권장(거절 폴백·TTFT 의 thinking)은 P2-02 에서 반영 | 사용자 결정으로 추가된 실습 |
| P2-02 Claude API 기초 | 1e86636 → d01b39d → c2aded7 | 1e86636 PASS_WITH_NOTES → d01b39d 재검증 **PASS**(권장 4건 반영, 변형 6종 포착, 폴백 설명 스킬 대조 일치) · 후속 c2aded7 **PASS** | c2aded7 = 선택 스크립트 6개 경로 로딩 통일 + 격리 테스트 8개(`tests/test_script_isolation.py`) |
| CLAUDE.md(학습 도우미 규칙) + ENVIRONMENT §7 | c72bc95 | **PASS** (권장 3건은 P2-03 커밋에 포함 예정 → §3) | |

모든 PROGRESS 상태는 `NOT_STARTED`(작성된 실습은 메모 "실습 자료 작성 완료, 학습자 확인 전"). `COMPLETED`는 사용자만.

마지막 확인 수치(c72bc95): 루트 `pytest` 184 passed, 1 skipped(P1-02A sentence-transformers 미설치) · `ruff check . --exclude part1_ai_fundamentals/pytorch/starter.py` 통과 · 상대 링크 0 깨짐.

## 2. 실습별 핵심 메모 (다시 손댈 때 필요한 것)

- **P1-01** `part1_ai_fundamentals/tokenizer/`: 로컬 바이트 BPE(vocab 500). 청크는 offsets 로 원문 자르기, 제공 함수 `is_char_boundary`·`snap_chunk_end`·`snap_next_start`. 사용자가 starter 진행 중(60349ae) — **사용자 파일 수정 금지**.
- **P1-02** `embedding/`: numpy, `embed_fn` 주입, `fake_embed`(글자 2-gram md5 256차원), OpenAI 실제 호출은 `embed_with_openai.py`(사용자 PC).
- **P1-03** `word2vec/`: gensim 4.4.0, 상대 비교 테스트(cat–dog > cat–coffee), 데모 [5]는 P1-02 `fake_embed` 를 경로로 로드.
- **P1-05** `transformer/`: 마스크 규약 True=볼 수 있음(SDPA bool 과 같음, `key_padding_mask` 는 반대). 한 행 전부 가리면 직접 계산 NaN / torch 2.15 SDPA 는 0. KV 캐시는 미리 할당한 버퍼.
- **P2-01** `part2_llm/next_token_prediction/`: 단어 `BigramLM`(alpha 0.001), 필터 순서 T → top-k → top-p(테스트로 고정), Qwen 실제 실행은 `run_qwen.py`.
- **P2-01B** `part2_llm/local_llm_serving/`: 공통 인터페이스 `generate(prompt) -> (text, Usage)`, Fake/Ollama/Claude, 폴백은 `ProviderError`만, 클라이언트 주소 변수 `OLLAMA_BASE_URL`(서버용 `OLLAMA_HOST` 와 구분). GPU 측정은 `measure_local_models.py`(사용자 PC).
- **P2-02** `part2_llm/llm_api/basics/`: `client.beta.messages.create/stream(**req)`, req = model·max_tokens 16000·messages·output_config{effort medium}·system·betas[`server-side-fallback-2026-07-01`]·fallbacks `"default"`. 응답은 stop_reason 먼저 → text 블록만 → 검증 → 폴백 여부. `messages.parse` 대신 `output_config.format` + `anthropic.transform_schema`. 실제 호출 `call_claude.py`(유료). P2-01B 의 `Usage`·`ProviderError`·`log_usage` 를 경로로 로드해 재사용.

## 3. 보류·남은 작업

**다음 실습 커밋에 함께 넣기로 한 것**(① 지시):
1. `CLAUDE.md` §6 Claude API 금지 목록에 `thinking: {"type": "disabled"}` 추가(Opus 5.5 에서 400).
2. `docs/ENVIRONMENT.md` §7 1단계 문장 순서를 실제 순서(clone → cd → §2 venv·설치)로.
3. `docs/ENVIRONMENT.md` §2 에 PowerShell 실행 정책 안내: `Activate.ps1` 이 "스크립트를 실행할 수 없습니다"로 막히면 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` 한 번(현재 사용자에게만 적용).
4. `docs/ARCHITECTURE.md` §3: 선택 스크립트 로딩 규칙 한 줄(`_load_sibling`, 이름 `lab_<상위>_<실습>_<파일>`) 추가, pytest.ini 예시 `testpaths` 를 실제 값 `part1_ai_fundamentals part2_llm tests` 로 정정(현재 예시는 part2_llm 누락).
5. (낮은 우선순위, ③ 권장) `tests/test_script_isolation.py` 의 skip 세분화: 지금은 tokenizers·torch 중 하나라도 없으면 fixture 전체가 skip — 라이브러리별 importorskip 으로 스크립트별로만 skip.
6. (낮은 우선순위, ③ 권장) `advanced/` 의 스크립트 9개(`run_demo.py`·`compare_*.py` 등)가 `main()` 을 테스트가 부르게 되는 시점에 `_load_sibling` 경로 로딩으로 변경.

**기타 보류**
- Part 0 문서: REPOSITORY_ANALYSIS 의 실제 경로, 실험 디렉터리 이름 통일.
- P2-01B `ClaudeClient`(최소 어댑터)는 408·409 를 `ProviderError` 로 매핑하지 않음 — Part 6 에서 P2-02 요청 빌더·`classify_error` 와 통일.
- 사용자 PC 측정이 비어 있는 실행 기록: P1-01 선택 스크립트, P1-02 OpenAI, P2-01 Qwen, P2-01B Ollama·VRAM, P2-02 실제 호출(모두 "측정 필요" 칸).

## 4. 다음에 재개할 첫 단계 — P2-03 (사용자 "다음" + ① 지시 후)

- 실습: **P2-03 비용·지연 — 프롬프트 캐싱, 배치, 토큰 사용량 기록**(LEARNING_ROADMAP W6). 위치 `part2_llm/llm_api/cost_latency/`(llm_api README 색인에 "작성 예정"으로 있음).
- 재사용: P2-02 `build_request`·`estimate_cost`·`to_usage`·`read_response`, P2-01B `log_usage`·`Usage`. 복사하지 말고 경로 로드(`_load_p2_01b` 방식).
- SDK 확인 출처(claude-api 스킬): `shared/prompt-caching.md`(접두어 일치, 바뀐 지점 이후만 무효, 모델별 최소 길이, `cache_control`, `usage.cache_read_input_tokens`·`cache_creation_input_tokens`, `input_tokens` 는 캐시 제외 나머지), `python/claude-api/batches.md`(배치 — 서버 측 `fallbacks` 는 Batches 에서 거부되므로 배치 요청은 `fallbacks=False`), `shared/cost-optimization.md`.
- 규약: TODO 4~6, 테스트는 가짜 SDK·가짜 시계(키·네트워크 불필요), 실제 호출은 선택 스크립트(유료, 키 없으면 "건너뜀"), 단가는 공식 페이지 링크 + 기준일, README 12항목 + 서비스 관점, experiments 는 사용자 PC 칸 비우기, ruff. §3 의 1~4번을 같은 커밋에 반영.
- 완료 후: 커밋·푸시 → HANDOFF 갱신 → ③ 검증 요청 → ①에 3~5줄 보고 → 대기.

# HANDOFF — 방향 전환(2026-11-09) 이후

방향: **개념은 깊게, 구현은 얕게 — 서비스 설계 중심.** 목표는 RAG·GraphRAG 서비스를 설계·판단하는 능력이다.
진행 방식: 한 단계(실습)씩 완성한 뒤 멈춘다. 다음 실습은 사용자가 "다음"이라고 할 때 ①(총괄 세션)의 지시를 받고 시작한다.

## 방향 전환 단계

| 단계 | 내용 | 커밋 | 상태 |
|---|---|---|---|
| 1 | 기획 문서 수정 (PROJECT_PLAN, LEARNING_ROADMAP 22주, PROGRESS, ARCHITECTURE, README들) | 99a40ff | 완료(검증 대기) |
| 2 | 기존 Part 1 심화 실습(tokenizer·embedding·word2vec·transformer)을 각 `<실습>/advanced/`로 이동(24개 파일) | f655119 | 완료(검증 대기) |
| 3 | 새 Tokenizer 메인 실습 P1-01 | 16ecd91 → 검증 반영 a8d675d, c9f495a | 완료(③ PASS_WITH_NOTES 반영) |
| 4 | 새 Embedding 메인 실습 P1-02 | e4b03e6 | 완료(③ PASS, 사소 권장 2건은 5단계 커밋에 반영) |
| 5 | 새 Word2Vec 메인 실습 P1-03 | a8ef12d | 완료(③ PASS, 사소 권장 1건은 6단계 커밋에 반영) |
| 6 | 새 Transformer 메인 실습 P1-05 | 7204c31 → 검증 반영 093a0fa | 완료(③ PASS_WITH_NOTES 반영) — Part 1 메인 5개 모두 작성 |
| 7 | Part 2 첫 실습 P2-01 다음 토큰 예측·샘플링 | 이 HANDOFF 를 갱신한 커밋 | 완료(검증 대기) |

## Part 1 실습 상태

| 실습 | 메인 | 선택 과제(advanced/) |
|---|---|---|
| P1-04 pytorch | 그대로 유지(③ PASS_WITH_NOTES). 사용자가 starter.py 를 직접 채움(a4fdc61) | — |
| P1-01 tokenizer | **새로 작성(3단계)**. TODO 5개: 토큰 수, 한/영 비교, 비용 추정, 토큰 예산 청크(오버랩), 컨텍스트 맞춤. solution 13 passed / starter 12 skipped + 1 passed(가짜 Claude 클라이언트 테스트, impl 무관). README 12항목 + 서비스 관점, experiments/P1-01-20261109-tokenizer | P1-01A: BPE 직접 구현(기존) |
| P1-02 embedding | **새로 작성(2026-11-12)**. TODO 5개: L2 정규화, 코사인 유사도, embed_fn 주입 배치 임베딩, top-k 검색, (선택) 임베딩 캐시. solution 15 passed / starter 15 skipped. README 12항목 + 서비스 관점, experiments/P1-02-20261112-embedding-main | P1-02A: 임베딩 층·위치 임베딩 직접 구현(기존) |
| P1-03 word2vec | **새로 작성(2026-11-12, 짧게)**. TODO 4개: 전처리, gensim 학습(build_vocab→train, workers=1), 학습 전후 쌍 유사도·최근접(OOV 처리), (선택) 평균 문장 벡터. solution 11 passed / starter 11 skipped. README 12항목 + 서비스 관점, experiments/P1-03-20261112-word2vec-main | P1-03A: Skip-gram·CBOW·NS 직접 구현(기존) |
| P1-05 transformer | **새로 작성(2026-11-12)**. TODO 5개: Q·K·V, 스케일드 닷-프로덕트 어텐션, causal 마스크, 길이 T 별 비용 측정, (선택) KV 캐시. solution 11 passed / starter 11 skipped. 나머지 부품은 README mermaid 그림·설명. README 12항목 + 서비스 관점, experiments/P1-05-20261112-transformer-main | P1-05A: MHA~미니 GPT 직접 구현(기존) |

`docs/PROGRESS.md`는 모두 `NOT_STARTED`로 유지한다. 학습자가 직접 확인해야 `COMPLETED`가 된다.

## P1-01 메인 실습 메모
- 토크나이저: HF `tokenizers` 바이트 단위 BPE 를 저장소 말뭉치로 학습(다운로드 없음, vocab 500).
- 한글 한 글자가 토큰 여러 개가 되고, 토큰 하나가 두 글자에 걸칠 수 있다(offsets 겹침). 청크·자르기는 offsets 로 원문을 자르고, 제공 함수 `is_char_boundary`로 글자 경계에서만 자른다. 겹침 0 청크를 이어 붙이면 원문과 같다.
- 토큰 수 기준은 "원문 전체 인코딩"이다. 조각만 다시 인코딩하면 경계에서 몇 토큰 달라질 수 있음을 README 에 명시했다.
- 선택 스크립트: `count_with_claude.py`(`messages.count_tokens`, `.env`의 `ANTHROPIC_API_KEY`, 모델은 `LLM_MODEL` 기본 `claude-opus-5-5`), `pretrained_tokenizer.py`(Qwen/Qwen2.5-0.5B tokenizer.json). 클라우드 환경에서는 키·인터넷이 없어 둘 다 "건너뜀" → **사용자 PC에서 실행 필요**.
- 단가는 공식 가격 페이지 기준 "작성 시점 2026-10"으로 표기했다.
- requirements/part1.txt 에 선택 의존성 anthropic, python-dotenv 추가.

## P1-02 메인 실습 메모
- numpy 사용(구현은 얕게). 임베딩 함수(`embed_fn`: 문자열 목록 → 벡터 목록)를 주입받는 구조.
- 제공: `fake_embed`(글자 2-gram md5 feature hashing, 256차원, 결정적), `make_openai_embed_fn`(OpenAI `client.embeddings.create(model, input)`, 응답을 `index`로 정렬, `usage` 누적). SDK 시그니처는 설치된 openai 3.49.0 에서 확인.
- 데모는 가짜 임베딩의 한계를 실제 출력으로 보여 준다: 글자가 겹치지 않는 `강아지와 고양이` 질의는 정답(cat·dog 문장)을 못 찾음 → Part 3 Hybrid 비교의 동기.
- 선택 스크립트 `embed_with_openai.py`(.env 의 OPENAI_API_KEY, 모델 EMBEDDING_MODEL 기본 text-embedding-3-small): 키가 없어 "건너뜀"만 확인 → **사용자 PC에서 실행 필요**.
- OpenAI 문서·가격 페이지와 numpy.org 는 작성 환경에서 접속 불가 → README REF 에 "미확인" 표기, 가격 수치는 쓰지 않고 링크만.
- 테스트의 반복은 모두 길이가 정해진 for 문(무한 반복·메모리 위험 없음). 가짜 OpenAI 클라이언트는 역순 응답을 돌려 index 정렬을 검사.
- requirements/part1.txt 에 선택 의존성 openai 추가.

## P1-03 메인 실습 메모
- gensim 4.4.0 `Word2Vec(vector_size=32, window=2, min_count=1, negative=5, hs=0, sample=0, epochs=200, seed=0, workers=1)`. 같은 시드면 다른 프로세스에서도 같은 값(cat–dog 0.967357)을 확인.
- 테스트는 상대 비교: cat–dog > cat–coffee, mat–log > mat–coffee(Skip-gram·CBOW 모두), 학습 후 > 학습 전. 탐색에서 시드 0~9 모두 성립(CBOW epochs 50 은 8/10 이라 epochs 200 사용).
- 데모 [5]는 P1-02 `embedding/solution.py`의 `fake_embed`를 파일 경로로 불러와 비교한다(P1-02 파일 이름·함수 이름을 바꾸면 이 데모도 고쳐야 함).
- 실제 결과로 드러난 교육 포인트: 대조군도 0.5~0.65(작은 말뭉치), `a dog and a cat` 평균 벡터는 흔한 단어에 끌려 정답 실패·`dog cat` 은 성공, 한국어 질의는 전부 OOV.
- gensim 문서·arXiv 는 작성 환경에서 접속 불가 → REF "미확인" 표기(논문 저자·연도·절은 기재).

## P1-05 메인 실습 메모
- torch. 마스크 규약 True = 볼 수 있음(F.scaled_dot_product_attention bool attn_mask 와 같음, nn.MultiheadAttention key_padding_mask 는 반대 — README 명시).
- 가린 칸은 -inf. 한 행 전체를 가리면 직접 계산은 NaN, 이 torch 2.15 SDPA 는 0 을 돌려줌(관찰) → SDPA 비교 테스트는 모든 행에 True 가 있는 마스크로만, NaN 은 별도 테스트.
- 시간 측정 테스트는 모양·원소 수·MB 만 단언. 데모 시간 비율은 4회 실행에서 2.2~5.7배로 흔들림(README·experiments 에 그대로 기록).
- KV 캐시: 처음 `torch.cat` 누적 구현은 캐시 쪽이 더 느렸음(19.7 vs 28.1ms) → 미리 할당한 버퍼로 수정, 약 1/5. 결과는 한 번에 계산한 causal 과 같음(2e-06).
- "lost in the middle" 등 출처를 확인할 수 없는 주장은 넣지 않음. Vaswani et al. 2017 은 절 번호로 인용, arXiv·pytorch.org 링크는 미확인 표기.
- 데모는 `torch.set_num_threads(1)`; 데모 실행 테스트는 끝나면 스레드 수를 되돌린다.

## ③ P1-05 검증 반영 (7204c31 PASS_WITH_NOTES)
- `test_make_qkv_shapes_and_values` 에 `k == x @ wk` 단언 추가. 스크래치 변이(K = Q)로 이 테스트만 실패 확인.
- README 서비스 관점 정정: 어텐션 계산은 T², 점수 행렬을 그대로 만들면 메모리도 T²(FlashAttention 류는 메모리 T 수준·계산 T²), 모델 전체로는 T 비례 항(투영·FFN)도 있어 길수록 T² 비중이 커짐.
- 프롬프트 캐싱: 접두어가 바뀌면 그 지점 이후 캐시 무효(앞 구간은 재사용 가능), 최소 길이는 모델마다 다름(수치는 Part 2), `cache_control` 표시·`usage.cache_read_input_tokens` 확인 — claude-api 스킬 문서(prompt-caching)로 확인.
- KV 캐시 메모리 식 = 토큰 수 × 층 수 × 2 × (KV 헤드 수 × 헤드 차원) × 자료형 바이트, GQA/MQA 는 KV 헤드 수가 줄어 메모리 감소.

## P1-03 사소 권장 반영 (③ PASS 메모)
- `test_train_word2vec_settings` 에 `m.sample == 0 and m.hs == 0` 단언 추가.

## P1-02 사소 권장 반영 (③ PASS 메모)
- `test_top_k_many_ties_ascending_index`: 같은 벡터 25개가 섞인 문서에서 동점 번호 오름차순. `kind="stable"` 제거 변이로 이 테스트가 실패하는 것을 확인.
- `fake_embed` docstring 2-gram 예시를 실제 목록 `[' 손', '손실', '실 ', ' 값', '값 ']`으로 수정(solution·starter).

## Part 2 실습 상태

| 실습 | 상태 |
|---|---|
| P2-01 next_token_prediction | **작성(2026-11-12)**. TODO 5개: temperature softmax, top-k, top-p, sample_next(greedy/sampling), generate(컨텍스트 창·max_new_tokens·eos). solution 11 passed / starter 11 skipped. README 12항목 + 서비스 관점, experiments/P2-01-20261112-next-token |
| P2-02 ~ P2-04, P2-S1 | 자리표시 README (순서대로 작성 예정) |

## P2-01 메모
- 모델: 다운로드 없는 단어 `BigramLM`(제공, Part 1 말뭉치, alpha=0.001). "모델 = 토큰 목록 → 다음 토큰 logits 함수" 인터페이스라 같은 `generate`가 Qwen 에서도 동작.
- 선택 `run_qwen.py`: transformers `AutoTokenizer`/`AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")`, CUDA 면 `.half()`, 상위 5 토큰·T별 엔트로피·생성·프롬프트 길이별 시간(cuda.synchronize). 이 환경은 transformers 미설치로 "건너뜀"만 확인 → **사용자 PC 실행 필요**. 테스트하지 않음(실행하면 다운로드가 일어나므로).
- 실제 결과로 드러난 교육 포인트: greedy 되풀이("the same data and"), 시드 재현, alpha 0.1 이면 엔트로피 2.69→8.31비트로 무의미한 문장, 컨텍스트 창 4 → 입력 [4, 4, 4].
- 서비스 관점: Claude API(API 버전 2026-10-21)에서 temperature·top_p·top_k 를 보내면 400 — claude-api 스킬 shared/api-version-migration.md 로 확인. 일관성은 프롬프트·구조화 출력(P2-02)으로 조절.
- `part2_llm/conftest.py` 신설(Part 1 conftest 와 같은 규약, 모듈명 접두어 `part2_`), `pytest.ini` testpaths 에 part2_llm 추가.
- Anthropic SDK 코드는 이번 실습에 없음(P2-02 부터).

## 공통 규약 (Part 1)
- 테스트는 `part1_ai_fundamentals/conftest.py`의 `impl` 픽스처를 쓴다.
  - 실습 디렉터리(또는 `advanced/`)의 `{IMPL}.py`를 파일 경로로, `part1_` + 상대 경로 기반 고유 모듈명으로 불러온다(예: `part1_tokenizer_advanced_solution`).
  - starter에서 `NotImplementedError`가 나면 skip("TODO 미구현: TODO n")으로 보고한다.
- 테스트 파일명: 메인 `test_p1_0x_<이름>.py`, 선택 과제 `test_p1_0x_<이름>_advanced.py`.
- API 호출은 테스트에서 가짜 클라이언트로 대체한다. 키는 `.env`에만, 커밋 금지.
- 루트 `pytest`: 137 passed, 1 skipped(P1-02A 사전학습 비교, sentence-transformers 미설치).
- ruff: 사용자가 채운 `part1_ai_fundamentals/pytorch/starter.py`에 RUF100(쓰지 않는 noqa) 3건(2026-11-12 기준)이 있어 `--exclude`로 검사했다. 사용자 학습 파일이라 수정하지 않았다. 사용자가 진행 중인 `tokenizer/starter.py`(60349ae)도 손대지 않았다.

## 환경 메모
- 클라우드: download.pytorch.org·pytorch.org·huggingface.co·platform.openai.com·numpy.org·radimrehurek.com·arxiv.org 접속 불가. PyPI torch 2.15.0+cu132(CPU).
- `.venv`: Python 3.12.13, numpy 2.5.4, gensim 4.4.0, tokenizers 0.23.3, anthropic 2.6.0, python-dotenv, openai 3.49.0.

## 보류
- Part 0 문서: REPOSITORY_ANALYSIS 실제 경로, 실험 디렉터리 이름 통일.
- 정리됨: `.env.example`(ANTHROPIC_API_KEY·OPENAI_API_KEY 빈 값, LLM_MODEL, EMBEDDING_MODEL) 추가, 루트 README·ENVIRONMENT §2·P1-01 README §6 에 복사 안내.

## ③ 검증 반영 (16ecd91 PASS_WITH_NOTES)
- `.env.example` 추가와 안내.
- 출력 단가가 비싼 이유 정정: 토큰당 연산량은 비슷, 입력은 병렬 prefill·출력은 순차 decode → 처리량 차이.
- TODO 4 가볍게: 글자 경계·과대 글자·무한 반복 방지를 제공 함수 `snap_chunk_end`·`snap_next_start`로 분리. 학습자는 start/end/overlap 이동만. TODO 총 5개 유지.
- 청크 테스트 무한 반복 가드: (a8d675d 의 스레드 10초 가드는 버려진 스레드가 메모리를 계속 써서 폐기) → 테스트가 impl 의 제공 함수 `snap_chunk_end`를 감싸 호출이 "토큰 수 + 1"번을 넘으면 즉시 실패. 변이 `start = start`로 0.3초·최대 RSS 47MB 에 실패 메시지 확인. 플랫폼 무관.
- 종료 조건 테스트 추가: overlap>0 에서 마지막 조각이 직전 조각 꼬리에 완전히 포함되지 않음(break 제거 변이로 실패 확인).
- `.env.example` OPENAI_API_KEY 주석: Part 1 P1-02(Embedding) 이후 사용.
- HINT 1 문구를 "글자 경계를 확인하지 않고 decode 하면 한글이 깨질 수 있음"으로 조건 명시.

## 다음에 재개할 첫 단계
Part 1 메인 5개와 P2-01 이 작성되었다. 사용자가 "다음"이라고 하면 ①의 지시에 따라 P2-02(Claude API 기초)를 시작한다. Anthropic SDK 코드는 claude-api 스킬로 현재 형태를 확인해 작성한다.

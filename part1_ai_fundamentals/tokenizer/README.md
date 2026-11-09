# P1-01 Tokenizer — 토큰은 비용·지연·컨텍스트의 단위

| 항목 | 값 |
|---|---|
| 실습 ID | P1-01 (메인) |
| 주차 · 진행 순서 | W2 · P1-04(PyTorch) 다음 |
| 예상 시간 | 2~3시간 |
| 파일 | `starter.py`(직접 채우기), `solution.py`(정답), `tests/test_p1_01_tokenizer.py`, `run_demo.py`, 선택: `count_with_claude.py`(API 키 필요), `pretrained_tokenizer.py`(인터넷 필요) |
| 소속 | [Part 1 — AI 원리](../README.md) |
| 선택 과제 | [advanced/](./advanced/README.md) — BPE 직접 구현 (P1-01A) |

> 방향: **개념은 깊게, 구현은 얕게.** BPE 알고리즘은 라이브러리(Hugging Face `tokenizers`)에 맡기고, 토큰 수를 "서비스 설계 숫자"로 쓰는 법을 익힙니다.

## 1. 목표

- 텍스트의 토큰 수를 세고, 한국어와 영어의 **문자 수 대비 토큰 수**를 비교한다.
- 토큰 단가표로 **API 비용**을 추정한다.
- 토큰 예산으로 문서를 **청크로 나누고(오버랩 포함)**, **컨텍스트 창**에 맞게 자른다.
- "토큰 수는 모델(토크나이저)마다 다르다 → 실제 요금은 그 모델의 토큰 세기로 확인한다"를 설명할 수 있다.

## 2. 선수지식

- Python 문자열·리스트·딕셔너리, 슬라이싱 `text[a:b]`
- UTF-8: 영어 글자는 1바이트, 한글 한 글자는 3바이트, 이모지는 보통 4바이트
- 환경 구성: [docs/ENVIRONMENT.md](../../docs/ENVIRONMENT.md), `pip install -r requirements/part1.txt`

## 3. 핵심 개념

| 개념 | 한 줄 설명 |
|---|---|
| 토큰(token) | 모델이 읽고 쓰는 최소 단위. 단어일 수도, 단어 조각·바이트 조각일 수도 있다 |
| 토크나이저 | 텍스트 ↔ 토큰 번호 목록을 바꾸는 규칙. **모델마다 다르다** |
| BPE | 자주 붙어 나오는 조각을 하나로 합치는 것을 반복해 어휘를 만드는 방법 |
| 바이트 단위(byte-level) BPE | 256개 바이트에서 시작 → 어떤 글자도 `<unk>`(모르는 토큰) 없이 표현. 대신 드문 글자는 여러 토큰 |
| offsets | 각 토큰이 원문의 몇 번째 글자부터 몇 번째 글자 앞까지인지 `(시작, 끝)` |
| 컨텍스트 창 | 모델이 한 번에 볼 수 있는 토큰 수 상한. **입력 + 출력** 합계 |
| 청크(chunk) | RAG에서 검색·임베딩 단위로 쓰려고 문서를 나눈 조각 |

## 4. 내부 원리

### 4-1. BPE는 "합치기 규칙 목록"이다

말뭉치에서 가장 자주 붙어 나오는 쌍을 찾아 하나로 합치고, 이를 정해진 어휘 크기까지 반복합니다. 학습 결과는 **합치기 규칙의 순서 목록**입니다.

**예시 말뭉치**: Sennrich et al., 2016, §3.2 Algorithm 1 의 예시 어휘(`low` ×5, `lower` ×2, `newest` ×6, `widest` ×3)를 [advanced/](./advanced/README.md)의 직접 구현으로 실행한 첫 병합들입니다(`</w>`는 단어 끝 표시).

| 단계 | 선택된 쌍 (빈도) | 병합 후 |
|---|---|---|
| 0 | — | `l o w </w>` · `l o w e r </w>` · `n e w e s t </w>` · `w i d e s t </w>` |
| 1 | e + s (9) | `n e w es t </w>` · `w i d es t </w>` |
| 2 | es + t (9) | `… est </w>` |
| 3 | est + `</w>` (9) | `… est</w>` |
| 4 | l + o (7) | `lo w </w>` · `lo w e r </w>` |
| 5 | lo + w (7) | `low </w>` · `low e r </w>` |

처음 보는 단어 `lowest`는 규칙을 순서대로 적용해 `low` + `est</w>`가 됩니다. **모르는 단어도 조각으로 표현**하는 것이 BPE의 핵심입니다. 직접 구현해 보고 싶다면 선택 과제 P1-01A.

### 4-2. 바이트 단위라서 한글이 여러 토큰이 된다

이 실습의 로컬 토크나이저(말뭉치 2,492자로 학습, 어휘 500개)로 실제로 인코딩한 결과입니다.

```
'the newest model' → 5 토큰
  tokens  ['the', 'Ġnew', 'es', 't', 'Ġmodel']          # Ġ = 앞의 공백
  offsets [(0, 3), (3, 7), (7, 9), (9, 10), (10, 16)]

'토크나이저' → 10 토큰
  tokens  ['í', 'Ĩ', 'ł', 'í', 'ģ', '¬ë', 'Ĥĺ', 'ìĿ´', 'ìł', 'Ģ']   # 바이트를 글자처럼 보이게 바꾼 표시
  offsets [(0, 1), (0, 1), (0, 1), (1, 2), (1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (4, 5)]
```

- `토`(3바이트)는 합쳐진 적이 없어 **토큰 3개**이고, 셋 다 offsets `(0, 1)`로 같은 글자를 가리킵니다.
- `¬ë`는 `크`의 마지막 바이트와 `나`의 첫 바이트를 **함께** 담아 offsets가 `(1, 3)`입니다.
- 그래서 토큰 번호 일부만 `decode`하면 바이트가 잘려 `�`(깨진 글자)가 나올 수 있습니다. 이 실습은 토큰 대신 **offsets로 원문을 잘라** 글자를 지킵니다(제공 함수 `is_char_boundary`).

## 서비스 관점

| 질문 | 토큰으로 본 답 |
|---|---|
| 요청 1건 비용은? | (입력 토큰 × 입력 단가 + 출력 토큰 × 출력 단가) / 1,000,000. 출력 단가가 입력의 5배 |
| 한국어 서비스는 비용이 다른가? | 같은 뜻이라도 토크나이저에 따라 토큰 수가 다르다. **가정하지 말고 그 모델로 잰다** |
| RAG 비용은 어디서 커지나? | 검색 문서를 입력에 넣으므로 **입력 토큰**이 커진다 → 청크 크기 × 상위 k 개가 비용을 정한다 |
| 청크는 몇 토큰으로? | 작으면 검색은 정밀하지만 문맥 부족, 크면 비용·지연 증가. 오버랩은 저장·임베딩 비용을 `max/(max−overlap)`배 늘린다 → Part 3 P3-02에서 측정 |
| 문서가 컨텍스트 창보다 길면? | 출력 몫을 먼저 예약하고, 남은 예산 안에서 **무엇을 넣을지 고른다**(RAG·요약). 자르면 로그를 남긴다 |
| 로컬에서 센 토큰 수로 청구액을 맞출 수 있나? | 없다. 이 실습의 로컬 BPE는 원리용이다. Claude 비용은 `messages.count_tokens`(→ `count_with_claude.py`), 실제 사용량은 응답의 `usage`로 확인 |

## 5. 수식

- 비용(USD) = 요청 수 × (입력 토큰 / 10⁶ × 입력 단가 + 출력 토큰 / 10⁶ × 출력 단가)
- 글자/토큰 = 문자 수 / 토큰 수 (클수록 압축이 잘 됨 = 같은 글이 싸다)
- 청크 시작 간격 step = max_tokens − overlap, 조각 수 ≈ ⌈(N − overlap) / step⌉, 저장 토큰 합 ≈ N × max_tokens / step
- 문서에 쓸 예산 = context_window − reserved_tokens (시스템 프롬프트 + 질문 + **출력 최대 길이**)

단가(USD / 100만 토큰, 입력/출력, **작성 시점 2026-10 기준** — 사용 전 [공식 가격 페이지](https://platform.claude.com/docs/en/about-claude/pricing)에서 재확인):

| 모델 ID | 입력 | 출력 |
|---|---|---|
| `claude-opus-5-5` | 4.00 | 20.00 |
| `claude-sonnet-5-5` | 2.00 | 10.00 |
| `claude-haiku-5-5` | 0.10 | 0.50 (프롬프트 10만 토큰 이하 기준) |

## 6. 실행법

저장소 루트에서 실행합니다.

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"
pip install -r requirements/part1.txt

# 1) 남은 TODO 확인
python part1_ai_fundamentals/tokenizer/starter.py

# 2) 직접 구현 검사 (TODO 가 남은 테스트는 skip)
$env:IMPL = "starter"; pytest part1_ai_fundamentals/tokenizer/tests; Remove-Item Env:IMPL

# 3) 정답 코드 검사
pytest part1_ai_fundamentals/tokenizer/tests

# 4) 데모 (다운로드·API 키 불필요)
python part1_ai_fundamentals/tokenizer/run_demo.py --impl starter

# 5) (선택) Claude 기준 토큰 수 — 저장소 루트 .env 에 ANTHROPIC_API_KEY=... (커밋 금지)
python part1_ai_fundamentals/tokenizer/count_with_claude.py

# 6) (선택) 공개 모델 토크나이저와 비교 — 인터넷 필요(tokenizer.json 만 받고 모델 가중치는 받지 않음)
python part1_ai_fundamentals/tokenizer/pretrained_tokenizer.py
```

**bash (Linux/macOS)**

```bash
source .venv/bin/activate
python part1_ai_fundamentals/tokenizer/starter.py
IMPL=starter pytest part1_ai_fundamentals/tokenizer/tests
pytest part1_ai_fundamentals/tokenizer/tests
python part1_ai_fundamentals/tokenizer/run_demo.py --impl solution
```

- 테스트는 다운로드도 API 호출도 하지 않습니다. 토크나이저는 `part1_ai_fundamentals/data/corpus_ko_en.txt`로 몇 초 안에 학습하고, Claude 토큰 세기는 **가짜 클라이언트**로 검사합니다(이 테스트는 starter와 무관해 starter 모드에서도 통과).
- `count_with_claude.py`의 모델은 환경 변수 `LLM_MODEL`로 바꿉니다(기본 `claude-opus-5-5`). 토큰 세기 API는 모델을 실행하지 않고 개수만 돌려줍니다. 요금·호출 한도는 공식 문서에서 확인하세요(미확인).

## 7. TODO 목록

| TODO | 함수 | 내용 | 서비스에서 쓰는 곳 |
|---|---|---|---|
| 1 | `count_tokens` | 토큰 수 세기 | 모든 비용·예산 계산의 출발점 |
| 2 | `compare_languages` | 언어별 문자 수 / 토큰 수 / 글자당 토큰 | 한국어 서비스 비용 보정 |
| 3 | `estimate_cost` | 단가표로 비용 추정, 모르는 모델은 `KeyError` | 모델 선택, 월 예산 |
| 4 | `chunk_by_tokens` | 토큰 예산 + 오버랩으로 청크, 글자 쪼개지 않기 | RAG 인덱싱(Part 3) |
| 5 | `fit_to_context` | 출력 몫을 뺀 예산만큼 앞부분 남기기 | 컨텍스트 초과 방지 |

제공 코드: `train_local_tokenizer`(로컬 BPE 학습), `is_char_boundary`(토큰 i 앞에서 잘라도 글자가 안전한가), `PRICES_PER_MTOK`(단가표).

## 8. 예상 결과

아래는 클라우드 작업 환경(CPU, Python 3.12.13, tokenizers 0.23.3)에서 `run_demo.py --impl solution`을 **실제로 실행한 출력**입니다.

```
[1] 말뭉치: chars=2492, tokens=1682 (local BPE, vocab 500)
[2] 언어별 비교
    ko    chars= 10 tokens= 15 chars/token=0.67
    en    chars= 26 tokens= 15 chars/token=1.73
    emoji chars=  5 tokens=  9 chars/token=0.56
[3] 비용 추정: 입력 = 말뭉치 전체, 출력 300 토큰, 요청 1,000 건 (로컬 토큰 수 기준 근사, 단가 작성 시점 2026-10)
    claude-opus-5-5    $   12.73
    claude-sonnet-5-5  $    6.36
    claude-haiku-5-5   $    0.32
[4] 청크 (max_tokens=200)
    overlap=  0: 9 조각, 조각 토큰 합 1682 (원문 1682, 1.00배)
    overlap= 50: 11 조각, 조각 토큰 합 2182 (원문 1682, 1.30배)
    조각 0 끝: '인다.\n학습이 잘 되면 손실 값이 조금씩 줄어'
    조각 1 앞: '\n작은 모델이라도 직접 학습해 보면 원리가 잘'
[5] 컨텍스트 창 1,000 토큰, 예약 300 토큰
    남긴 글자 622/2492, 토큰 698, 끝: '으로 비교해야 공정한 실험이 된다.\n'
```

**해석**
- [2] 같은 15토큰에 영어는 26자, 한국어는 10자가 들어갑니다. 이 **작은** 토크나이저에서 한국어는 글자당 토큰이 영어의 약 2.6배(1.73 / 0.67)입니다. 어휘가 크고 다국어로 학습된 실제 모델 토크나이저에서는 차이가 줄어들 수 있으므로, 숫자는 **그 모델로 다시 재야** 합니다(5·6번 선택 스크립트).
- [3] 같은 트래픽에 모델만 바꿔도 비용이 약 40배(12.73 / 0.32) 차이 납니다. 계산 확인: opus 1건 = 1,682 × 4 / 10⁶ + 300 × 20 / 10⁶ = $0.012728.
- [4] 오버랩 50(step 150)이면 이론상 200/150 ≈ 1.33배, 실제 1.30배입니다(마지막 조각이 짧아서). 조각 0의 끝 문장이 조각 1 앞에 다시 들어가 문맥이 이어집니다.
- [5] 예산 700토큰 중 698토큰을 썼습니다. 700번째 토큰 앞이 글자 중간이라 그 글자를 통째로 뺐기 때문입니다.
- 선택 스크립트 5·6은 이 환경에서 API 키·인터넷이 없어 `건너뜀`으로 끝났습니다. **사용자 PC에서 실행 필요**(결과는 실행한 값만 기록).

## 9. 자주 나는 오류

| 증상 | 원인 | 해결 |
|---|---|---|
| 청크에 `�` 가 보임 | 토큰 번호 일부를 `decode` 함 | offsets로 **원문**을 자른다 |
| 오버랩 청크에서 무한 반복 | `overlap >= max_tokens` → 시작점이 앞으로 안 감 | `ValueError`로 막고, 다음 시작이 안 나아가면 `end`에서 이어 간다 |
| 겹침 0인데 이어 붙이면 원문보다 김 | 글자 중간에서 잘라 같은 글자가 두 조각에 들어감 | `is_char_boundary`로 경계까지 당긴다 |
| 조각을 다시 세면 `max_tokens`보다 큼 | 토큰 수는 **원문 전체 인코딩** 기준. 조각만 따로 인코딩하면 경계 병합이 달라짐 | 임베딩 모델 한도에 딱 맞추지 말고 여유를 둔다 |
| 비용이 0으로 나옴 | 모델 ID 오타 | 모르는 모델은 `KeyError` |
| `count_with_claude.py`가 `건너뜀` | `.env`에 `ANTHROPIC_API_KEY` 없음 | 저장소 루트 `.env`에 추가(커밋 금지, `.gitignore` 등록됨) |
| PowerShell에서 한글이 깨짐 | 콘솔 인코딩 | `$env:PYTHONUTF8 = "1"` |

## 10. 과제

1. `train_local_tokenizer`의 `vocab_size`를 300 / 500 / 1000으로 바꿔 [2]의 ko·en `chars/token`을 표로 기록하세요. 어휘가 커지면 어느 언어가 더 많이 좋아지나요?
2. (API 키가 있으면) `count_with_claude.py`로 같은 문장의 Claude 토큰 수를 재고, 로컬 BPE와 비교하세요. 로컬 수치로 비용을 잡으면 몇 % 틀리나요?
3. 하루 1만 건, 질문 50토큰 + 검색 청크 5개 × 300토큰 + 답 300토큰인 RAG 서비스의 월 비용을 세 모델로 추정하세요. 청크를 200토큰으로 줄이면?
4. `fit_to_context`는 앞부분만 남깁니다. 뒷부분이 중요한 문서(예: 결론이 끝에 있는 보고서)라면 어떻게 바꾸겠습니까? 코드 대신 설계로 답해 보세요.

## 11. 체크리스트

- [ ] `IMPL=starter pytest part1_ai_fundamentals/tokenizer/tests` 가 모두 통과한다
- [ ] 같은 글자 수라도 언어·토크나이저에 따라 토큰 수가 다른 이유를 설명할 수 있다
- [ ] 입력·출력 단가로 요청 1건과 월 비용을 계산할 수 있다
- [ ] 오버랩이 검색 품질과 저장 비용에 주는 영향을 말할 수 있다
- [ ] 컨텍스트 창 = 입력 + 출력이라는 것과 출력 몫을 예약하는 이유를 안다
- [ ] 실제 요금은 그 모델의 토큰 세기·`usage`로 확인해야 한다는 것을 안다

## 12. 다음 단계 연결

- **P1-02 Embedding**: 청크를 벡터로 바꿔 의미 검색을 합니다. 임베딩 모델에도 입력 토큰 한도가 있어 TODO 4의 청크 크기가 그대로 쓰입니다.
- **Part 2 P2-02/03**: Claude API의 토큰 세기·`usage`·프롬프트 캐싱으로 실제 비용을 측정합니다.
- **Part 3 P3-02**: 청크 크기·오버랩을 바꿔 검색 품질과 비용을 수치로 비교합니다.
- 원리를 더 파고 싶다면 선택 과제 [advanced/](./advanced/README.md) (BPE 직접 구현, P1-01A).

## REF

- Sennrich, Haddow, Birch, 2016, "Neural Machine Translation of Rare Words with Subword Units", §3.2 — BPE
- Hugging Face `tokenizers` 문서: https://huggingface.co/docs/tokenizers
- Claude 토큰 세기: https://platform.claude.com/docs/en/build-with-claude/token-counting
- Claude 가격: https://platform.claude.com/docs/en/about-claude/pricing (작성 시점 2026-10 기준)
- Qwen2.5-0.5B 모델 카드: https://huggingface.co/Qwen/Qwen2.5-0.5B

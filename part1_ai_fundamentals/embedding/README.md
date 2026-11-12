# P1-02 Embedding — 문장을 벡터로, 가까움을 숫자로

| 항목 | 값 |
|---|---|
| 실습 ID | P1-02 (메인) |
| 주차 · 진행 순서 | W3 · P1-01(Tokenizer) 다음 |
| 예상 시간 | 2~3시간 |
| 파일 | `starter.py`(직접 채우기), `solution.py`(정답), `tests/test_p1_02_embedding.py`, `run_demo.py`, 선택: `embed_with_openai.py`(API 키 필요) |
| 소속 | [Part 1 — AI 원리](../README.md) |
| 선택 과제 | [advanced/](./advanced/README.md) — 임베딩 층·위치 임베딩 직접 구현 (P1-02A) |

> 방향: **개념은 깊게, 구현은 얕게.** 임베딩 모델은 API(OpenAI)에 맡기고, 정규화·코사인·top-k 검색과 "임베딩을 서비스에 쓸 때의 비용·운영 판단"을 익힙니다.

## 1. 목표

- L2 정규화가 "길이를 버리고 방향만 남기기"임을 손계산으로 설명한다.
- 정규화한 벡터의 내적이 코사인 유사도임을 이해하고, 행렬곱 `@`으로 유사도 표를 한 번에 만든다.
- 임베딩 함수를 **주입**받아 배치로 임베딩한다(서비스: OpenAI API, 테스트: 가짜 함수).
- 질의와 가장 가까운 문서 k개를 찾는다 — **Part 3 Dense 검색의 원형**.
- (선택) 같은 텍스트를 다시 임베딩하지 않는 캐시로 비용을 줄인다.

## 2. 선수지식

- P1-04의 shape·브로드캐스팅(이 실습은 numpy를 쓰지만 규칙은 같습니다. torch `keepdim` = numpy `keepdims`)
- P1-01의 토큰·비용 개념(임베딩 API도 입력 토큰으로 과금)
- 벡터의 내적: `[a, b] · [c, d] = a×c + b×d`

## 3. 핵심 개념

| 개념 | 한 줄 설명 |
|---|---|
| 임베딩 | 문장(또는 단어)을 고정 길이 숫자 목록(벡터)으로 바꾼 것. 뜻이 비슷하면 방향이 비슷하도록 학습된 모델이 만든다 |
| 차원(D) | 벡터의 숫자 개수. 클수록 표현력·저장 비용이 함께 커진다 |
| L2 노름 | 벡터의 길이 = √(성분 제곱합) |
| 정규화 | 길이로 나눠 길이를 1로. 방향만 남는다 |
| 코사인 유사도 | 두 벡터 사이 각도의 코사인(1 같은 방향, 0 직각, −1 반대). 정규화 벡터의 내적과 같다 |
| top-k 검색 | 유사도가 가장 큰 k개 문서를 고르기 |
| 주입(injection) | 임베딩 함수를 코드 안에서 정하지 않고 바깥에서 넣어 주기 → 같은 코드로 API·가짜 함수 모두 사용 |

## 4. 내부 원리

### 4-1. 정규화와 브로드캐스팅 (손계산)

`[3, 4]`의 길이는 √(3² + 4²) = √25 = **5**. 각 성분을 5로 나누면 `[0.6, 0.8]` (길이 1). `[6, 8]`도 10으로 나누면 `[0.6, 0.8]` — **방향이 같으면 정규화 결과가 같습니다.**

여러 벡터를 한 번에 정규화할 때 `keepdims`가 결과를 가릅니다.

```
x = [[3, 4],          norm(x, axis=-1, keepdims=True)  → [[ 5],     모양 (2, 1)
     [6, 8]]                                               [10]]
                      norm(x, axis=-1)                 → [5, 10]    모양 (2,)

x / [[5],[10]]  → 가로로 복사 [[5, 5],[10,10]] 로 맞춰 나눔 → [[0.6, 0.8], [0.6, 0.8]]   ✔ 행마다 자기 길이
x / [5, 10]     → 세로로 복사 [[5,10],[5, 10]] 로 맞춰 나눔 → [[0.6, 0.4], [1.2, 0.8]]   ✘ 오류 없이 틀린 값
```

### 4-2. 코사인 = 정규화 후 내적, 행렬곱으로 한 번에

`[0.6, 0.8] · [1, 0] = 0.6×1 + 0.8×0 = 0.6` → 두 벡터 사이 각도의 코사인이 0.6.

질의 N개 × 문서 M개의 모든 쌍은 `A @ B.T` 한 번으로 구합니다. A가 (N, D), B가 (M, D)이면 B.T는 (D, M)이고 결과는 (N, M)입니다. `[i, j]` 칸은 A의 i번째와 B의 j번째 벡터의 내적입니다.

```
A = [[0.6, 0.8]]                 B = [[1, 0],      A @ B.T = [[0.6×1 + 0.8×0,  0.6×0 + 0.8×1]]
                                      [0, 1]]               = [[0.6, 0.8]]
```

### 4-3. 이 실습의 가짜 임베딩(`fake_embed`)

이웃한 두 글자(2-gram)를 해시해 256칸 중 하나에 세는 방식입니다(feature hashing). **글자 겹침**만 보므로 뜻은 모릅니다. 테스트를 키 없이 결정적으로 돌리기 위한 도구이고, 실제 임베딩 모델과의 차이는 §8에서 결과로 확인합니다.

## 서비스 관점

| 질문 | 판단 |
|---|---|
| 저장 비용은? | 문서 수 × 차원 × 4바이트(float32). 예) 100만 개 × 1,536차원 ≈ 6.1GB. 차원을 줄이면(모델 선택·차원 축소) 저장·검색 비용이 줄지만 품질을 측정해야 한다 |
| 왜 정규화해서 저장하나? | 내적만으로 코사인이 되어 계산이 단순해지고, 벡터 DB의 inner product 와 cosine 결과가 같아진다 |
| 임베딩 모델을 바꾸면? | 벡터 공간이 달라 기존 벡터와 비교할 수 없다 → **문서 전체 재색인**. 벡터와 함께 모델 이름·차원을 저장한다 |
| 비용은 어디서 나오나? | 문서: 색인할 때 1회(문서가 바뀔 때만 다시). 질의: **요청마다** → 트래픽에 비례. 캐시(TODO 5)로 반복 질의 비용을 줄인다 |
| 한 번에 몇 개씩 보내나? | 배치로 보낸다. 1개씩 보내면 왕복 지연이 쌓이고 호출 한도에 걸린다 |
| 한국어도 잘 되나? | **가정하지 말고 측정한다.** 내 도메인의 한국어 질의·정답 문서 쌍을 모아 top-k 적중률을 잰다(Part 3 평가). 한국어 질의 → 영어 문서 같은 교차 언어도 따로 확인 |
| 가격은? | 공식 가격 페이지에서 확인: https://platform.openai.com/docs/pricing (작성 시점 2026-11, 작성 환경에서 접속 불가 → 수치 미기재·미확인). 실제 사용량은 응답의 `usage.prompt_tokens` |

## 5. 수식

- L2 노름: ‖x‖ = √(x₁² + … + x_D²), 정규화: x̂ = x / max(‖x‖, ε)
- 코사인: cos(a, b) = (a · b) / (‖a‖ ‖b‖) = â · b̂
- 유사도 표: S = Â B̂ᵀ, S ∈ ℝ^{N×M}
- 저장 바이트 = N × D × 4 (float32)
- 배치 호출 수 = ⌈N / batch_size⌉

## 6. 실행법

저장소 루트에서 실행합니다.

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"
pip install -r requirements/part1.txt

# 1) 남은 TODO 확인
python part1_ai_fundamentals/embedding/starter.py

# 2) 직접 구현 검사 (TODO 가 남은 테스트는 skip)
$env:IMPL = "starter"; pytest part1_ai_fundamentals/embedding/tests; Remove-Item Env:IMPL

# 3) 정답 코드 검사
pytest part1_ai_fundamentals/embedding/tests

# 4) 데모 (가짜 임베딩, API 키 불필요)
python part1_ai_fundamentals/embedding/run_demo.py --impl starter

# 5) (선택) OpenAI 임베딩으로 같은 검색 — .env 에 OPENAI_API_KEY 입력 (커밋 금지)
Copy-Item .env.example .env   # 처음 한 번. 이후 .env 에 키 입력
python part1_ai_fundamentals/embedding/embed_with_openai.py
```

**bash (Linux/macOS)**

```bash
source .venv/bin/activate
python part1_ai_fundamentals/embedding/starter.py
IMPL=starter pytest part1_ai_fundamentals/embedding/tests
pytest part1_ai_fundamentals/embedding/tests
python part1_ai_fundamentals/embedding/run_demo.py --impl solution
cp .env.example .env    # (선택 스크립트용) 이후 .env 에 OPENAI_API_KEY 입력
python part1_ai_fundamentals/embedding/embed_with_openai.py
```

- 테스트는 API를 부르지 않습니다. 임베딩은 가짜 함수, OpenAI 클라이언트는 가짜 객체로 검사합니다. 반복은 모두 길이가 정해진 `for`라 멈추지 않는 일이 없습니다.
- 모델은 `.env`의 `EMBEDDING_MODEL`(기본 `text-embedding-3-small`)로 바꿉니다.

## 7. TODO 목록

| TODO | 함수 | 내용 | 서비스에서 쓰는 곳 |
|---|---|---|---|
| 1 | `l2_normalize` | 행마다 길이 1로, 0 벡터는 0으로 | 문서 벡터 저장 전 |
| 2 | `cosine_similarity` | 정규화 후 `A @ B.T` → (N, M) | 검색 점수 |
| 3 | `embed_texts` | `embed_fn` 주입, `batch_size`씩 호출, 개수 검사 | 색인·질의 임베딩 |
| 4 | `top_k` | 상위 k개 (번호, 점수), 동점은 번호 순 | Dense 검색(Part 3) |
| 5 (선택) | `embed_with_cache` | 캐시에 없는 텍스트만, 중복 없이 임베딩 | 반복 질의 비용 절감 |

제공 코드: `fake_embed`(가짜 임베딩), `make_openai_embed_fn`(OpenAI `client.embeddings.create`를 `embed_fn` 모양으로 감싸고 `usage`를 더함).

## 8. 예상 결과

아래는 클라우드 작업 환경(CPU, Python 3.12.13, numpy 2.5.4)에서 `run_demo.py --impl solution`을 **실제로 실행한 출력**입니다.

```
[1] L2 정규화와 코사인
    normalize([3,4]) = [0.6, 0.8], normalize([6,8]) = [0.6, 0.8]
    cos([3,4], [6,8] / [-3,-4] / [4,-3]) = [1.0, -1.0, 0.0]
[2] 문서 임베딩: 48개 × 256차원 (fake_embed), embed_fn 호출 3번(batch_size=16)
    float32 저장 크기: 49,152 바이트 = 문서 수 × 차원 × 4
[3] top-3 검색 (가짜 임베딩 = 글자 2-gram 겹침)
    질의: 손실 값이 줄지 않을 때는 무엇을 확인하나
      0.502  #07 손실 값이 줄지 않으면 학습률과 데이터부터 다시 확인한다.
      0.328  #06 학습이 잘 되면 손실 값이 조금씩 줄어든다.
      0.292  #18 실험 기록을 남기면 나중에 무엇이 달라졌는지 비교할 수 있다.
    질의: what to check if the loss does not go down
      0.781  #29 If the loss does not go down, check the 
      0.587  #28 When training goes well, the loss goes d
      0.523  #45 the cat sat on the mat and the dog sat o
    질의: 강아지와 고양이
      0.208  #48 the lower the learning rate, the slower 
      0.178  #47 a new word is made of smaller pieces suc
      0.170  #32 Mixing the value vectors with those weig
[4] 캐시: 같은 질의 목록을 두 번 임베딩
    입력 3개(중복 1) → 1회차 임베딩한 텍스트 2개, 2회차 추가 0개, 캐시 크기 2
```

**해석**
- [1] 같은 방향은 1, 반대 방향은 −1, 직각은 0. 길이(5와 10)는 결과에 영향이 없습니다.
- [2] 48개 문장을 16개씩 3번 호출했습니다. 저장 크기 49,152 = 48 × 256 × 4.
- [3] 글자가 많이 겹치는 질의는 정답 문장(#07, #29)이 1위입니다. 하지만 `강아지와 고양이`는 뜻으로는 #45·#46(cat, dog)이 정답인데 **글자가 하나도 겹치지 않아** 엉뚱한 문장이 나왔고, 점수 0.2 안팎은 해시 칸이 우연히 겹친 잡음입니다. 이것이 "글자 일치 검색의 한계"이며, 학습된 임베딩 모델(의미 검색)과 BM25(단어 일치 검색)를 함께 쓰는 Hybrid 검색을 Part 3에서 비교하는 이유입니다. 영어 질의의 3위 #45도 `the` 같은 흔한 글자 조각이 겹쳐 올라온 것으로, 뜻과는 무관합니다.
- [4] 중복 질의는 한 번만, 두 번째 호출에서는 0개를 임베딩했습니다 = API 비용 0.
- 선택 스크립트 `embed_with_openai.py`는 이 환경에 API 키가 없어 `건너뜀`으로 끝났습니다. **사용자 PC에서 실행 필요**(결과는 실행한 값만 기록). 특히 `강아지와 고양이` 질의가 #45·#46을 찾는지 확인해 보세요.

## 9. 자주 나는 오류

| 증상 | 원인 | 해결 |
|---|---|---|
| 정사각 행렬에서만 정규화 값이 이상함 | `keepdims` 누락 → 열 방향으로 나눔 | `np.linalg.norm(x, axis=-1, keepdims=True)` |
| 유사도에 `nan` | 0 벡터를 0으로 나눔 | `np.maximum(norm, eps)` |
| `matmul` 모양 오류 `(1,D) @ (M,D)` | 전치 누락 | `b.T` |
| top-k 결과가 실행마다 순서가 다름(동점) | 정렬이 안정적이지 않음 | `np.argsort(-scores, kind="stable")` |
| 문서와 벡터가 어긋남 | 배치 응답 개수·순서 확인 안 함 | 개수 검사 `ValueError`, OpenAI 응답은 `index`로 정렬(제공 코드) |
| 모델을 바꿨더니 검색 품질이 무너짐 | 다른 모델의 벡터끼리 비교 | 전체 재색인, 벡터와 모델 이름을 함께 저장 |
| `embed_with_openai.py`가 `건너뜀` | `.env`에 `OPENAI_API_KEY` 없음 | `cp .env.example .env`(PowerShell: `Copy-Item .env.example .env`) 후 입력(커밋 금지) |

## 10. 과제

1. `fake_embed`의 `dim`을 32 / 256 / 4096으로 바꿔 [3]의 `강아지와 고양이` 점수가 어떻게 변하는지 기록하세요. 해시 충돌과 차원의 관계를 설명해 보세요.
2. (API 키가 있으면) `embed_with_openai.py`로 [3]의 세 질의 결과를 기록하고 가짜 임베딩과 비교하세요. 출력되는 `prompt_tokens`로 비용을 계산해 보세요(단가는 공식 페이지).
3. 문서 50만 개, 하루 질의 2만 건인 검색 서비스에서 (a) 1,536차원 float32 저장 크기, (b) 문서 색인 1회와 질의 한 달의 임베딩 토큰 수를 추정하세요(문서 평균 300토큰, 질의 평균 20토큰). 어느 쪽이 비용을 지배하나요?
4. 캐시 키에 모델 이름을 넣지 않으면 어떤 사고가 날 수 있는지, 그리고 캐시를 언제 비워야 하는지 설계로 답하세요.

## 11. 체크리스트

- [ ] `IMPL=starter pytest part1_ai_fundamentals/embedding/tests` 가 모두 통과한다
- [ ] `[3, 4] → [0.6, 0.8]` 을 손으로 계산하고, `keepdims`가 빠지면 무엇이 틀리는지 설명할 수 있다
- [ ] 정규화 벡터의 내적 = 코사인임을 안다
- [ ] 임베딩 모델을 바꾸면 재색인이 필요한 이유를 안다
- [ ] 문서 임베딩 비용과 질의 임베딩 비용이 어떻게 다르게 쌓이는지 설명할 수 있다
- [ ] 한국어 검색 품질은 내 데이터로 측정해야 한다는 것을 안다

## 12. 다음 단계 연결

- **P1-03 Word2Vec**: "비슷한 문맥의 단어는 비슷한 벡터"라는 임베딩 학습의 출발점을 짧게 실행해 봅니다.
- **Part 3 P3-03/04**: 이 실습의 `top_k`가 Dense 검색이 되고, BM25·Hybrid·Rerank와 품질·지연·비용을 비교합니다.
- **Part 5**: 임베딩 fine-tuning 전후를 OpenAI 기준선과 비교해 "언제 값어치가 있는가"를 판단합니다.
- 원리를 더 파고 싶다면 선택 과제 [advanced/](./advanced/README.md) (임베딩 층·위치 임베딩 직접 구현, P1-02A).

## REF

- numpy `linalg.norm`: https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html (작성 환경에서 접속 불가 → 미확인)
- numpy 브로드캐스팅: https://numpy.org/doc/stable/user/basics.broadcasting.html (미확인)
- OpenAI 임베딩 가이드: https://platform.openai.com/docs/guides/embeddings (작성 환경에서 접속 불가 → 미확인)
- OpenAI 가격: https://platform.openai.com/docs/pricing (미확인, 사용 전 확인)
- OpenAI Python SDK `client.embeddings.create(model, input)` — 설치된 openai 3.49.0 패키지의 시그니처로 확인

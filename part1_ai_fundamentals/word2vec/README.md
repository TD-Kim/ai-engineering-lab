# P1-03 Word2Vec — "비슷한 문맥의 단어는 비슷한 벡터"

| 항목 | 값 |
|---|---|
| 실습 ID | P1-03 (메인) |
| 주차 · 진행 순서 | W3 · P1-02(Embedding) 다음 |
| 예상 시간 | 1~2시간 (짧은 실습) |
| 파일 | `starter.py`(직접 채우기), `solution.py`(정답), `tests/test_p1_03_word2vec.py`, `run_demo.py` |
| 소속 | [Part 1 — AI 원리](../README.md) |
| 선택 과제 | [advanced/](./advanced/README.md) — Skip-gram/CBOW·Negative Sampling 직접 구현 (P1-03A) |

> 방향: **짧게 — 개념 설명 + gensim 짧은 실행.** 알고리즘 구현은 gensim에 맡기고, 임베딩 학습의 출발점인 분포 가설과 "단어 하나에 벡터 하나"의 한계를 서비스 판단으로 연결합니다.

## 1. 목표

- 분포 가설(같은 주변 단어 → 비슷한 뜻)을 예문으로 설명한다.
- Skip-gram과 CBOW의 차이, Negative Sampling이 빠른 이유를 그림으로 설명한다.
- gensim Word2Vec을 재현 가능하게 학습하고, 학습 전후 유사도를 **대조군과 비교해** 해석한다.
- 정적 단어 임베딩의 한계(다의어, OOV, 평균 문장 벡터)를 알고, 실무 RAG가 문장 임베딩 모델을 쓰는 이유를 말한다.

## 2. 선수지식

- P1-02의 코사인 유사도·top-k 검색
- P1-01의 토큰 개념(이 실습은 공백 분리 단어를 토큰으로 씁니다)

## 3. 핵심 개념

| 개념 | 한 줄 설명 |
|---|---|
| 분포 가설 | 같은 주변 단어와 함께 나오는 단어는 뜻이 비슷하다 |
| Word2Vec | "주변 단어 맞히기"를 풀며 부산물로 단어 벡터를 얻는 방법(정답 라벨 불필요) |
| Skip-gram (`sg=1`) | 가운데 단어로 주변 단어 각각을 맞힌다. 드문 단어에 강함 |
| CBOW (`sg=0`) | 주변 단어들의 평균으로 가운데 단어를 맞힌다. 빠름 |
| Negative Sampling | 전체 어휘 소프트맥스 대신 "진짜 이웃 1개 + 무작위 오답 k개"의 이진 분류 |
| 정적 임베딩 | 단어 하나에 벡터 하나. 문맥이 달라도 같은 벡터 |
| OOV | Out-Of-Vocabulary. 학습 때 없던 단어 → 벡터 없음 |

하이퍼파라미터(gensim `Word2Vec`):

| 인자 | 이 실습 값 | 뜻 | 품질·비용 영향 |
|---|---|---|---|
| `vector_size` | 32 | 벡터 차원 | 크면 표현력·저장↑, 데이터가 적으면 과적합 |
| `window` | 2 | 양쪽 몇 단어까지 주변으로 보나 | 작으면 문법적 이웃(sat–likes), 크면 주제적 이웃 |
| `min_count` | 1 | 이보다 드문 단어는 버림 | 대규모 말뭉치는 5 이상으로 잡음 제거 |
| `sg` | 1 / 0 | Skip-gram / CBOW | 드문 단어 품질 vs 속도 |
| `negative` | 5 | 오답 단어 수 k | 작은 데이터 5~20, 큰 데이터 2~5 (Mikolov et al., 2013b §2.2) |
| `epochs` | 200 | 말뭉치 반복 횟수 | 데이터가 작을수록 많이 필요 |
| `seed`, `workers=1` | 0, 1 | 난수 고정 + 단일 스레드 | 같은 환경에서 같은 결과(재현성) |
| `sample` | 0 | 흔한 단어 다운샘플링 | 작은 말뭉치라 끔. 대규모에서는 1e-3 정도 |

## 4. 내부 원리

### 4-1. 분포 가설

```
the cat sat on the mat
the dog sat on the log      →  cat 과 dog 의 주변(the, sat, on)이 같다  →  벡터가 가까워진다
```

### 4-2. Skip-gram vs CBOW (window = 2, 가운데 단어 `sat`)

```
Skip-gram:   sat ──▶ the        CBOW:   the ─┐
             sat ──▶ cat                cat ─┤
             sat ──▶ on                  on ─┼─(평균)──▶ sat
             sat ──▶ the                the ─┘
  (가운데 → 주변 각각)                  (주변 묶음 → 가운데)
```

### 4-3. Negative Sampling이 왜 빠른가

어휘가 5만 개면 "주변 단어가 무엇인가"를 소프트맥스로 풀 때 **매번 5만 개 점수**를 계산해야 합니다.
Negative Sampling은 질문을 "(sat, cat)은 진짜 이웃인가? (sat, coffee)는? (sat, rain)은?"처럼 **정답 1개 + 무작위 오답 k개**의 예/아니오로 바꿉니다. 한 번에 k+1개(여기서는 6개)만 계산하므로 어휘가 커져도 빠릅니다. 직접 구현은 [advanced/](./advanced/README.md)(P1-03A)를 보세요.

### 4-4. 한계: 단어 하나에 벡터 하나

`배`(먹는 배 · 타는 배 · 신체의 배)는 문장마다 뜻이 다르지만 Word2Vec에서는 **벡터가 하나**입니다. 세 뜻의 주변 단어가 섞여 학습되어 어느 뜻에도 정확히 맞지 않습니다. P1-02의 문장 임베딩 모델(트랜스포머 기반)은 문장 전체를 보고 벡터를 만들어 이 문제를 줄입니다(P1-05에서 어텐션으로 원리를 봅니다).

## 서비스 관점

| 질문 | 판단 |
|---|---|
| 실무 RAG는 왜 Word2Vec 대신 문장 임베딩 모델을 쓰나? | 순서·문맥·다의어를 반영하고, 질문과 답처럼 **단어가 달라도 뜻이 같으면** 가깝게 학습되어 있다. Word2Vec 평균은 순서를 버리고 흔한 단어에 끌려간다(§8 [5]) |
| 그래도 Word2Vec을 쓰는 경우는? | **도메인 용어 유사어 확장**(사내 문서로 학습 → '환불' 근처에 '반품'), API 비용 0·CPU 수 초의 **가벼운 오프라인 처리**, 외부로 데이터를 보낼 수 없는 환경 |
| 유사어 확장을 그대로 믿어도 되나? | 아니다. 반의어도 문맥이 같아 가깝게 나올 수 있다. 사람이 검토해 사전에 넣는다 |
| OOV는? | 학습 때 없던 단어(신조어·오타·새 상품명)는 벡터가 없다. 서브워드를 쓰는 모델(FastText, 문장 임베딩 모델)은 조각으로라도 벡터를 만든다 |
| 한국어는? | 공백 분리는 `값이`·`값을`·`값`을 다른 단어로 만든다. 형태소 분석기·서브워드 토크나이저 선택이 품질을 바꾼다 |
| 결과를 어떻게 판단하나? | 정확한 숫자 대신 **대상 쌍 > 대조군** 같은 상대 비교. gensim 버전·플랫폼에 따라 값은 조금 달라질 수 있다 |

## 5. 수식

- Skip-gram 목표: 가운데 단어 w_t 에 대해 Σ_{−c ≤ j ≤ c, j ≠ 0} log p(w_{t+j} | w_t) 최대화 (Mikolov et al., 2013b 식 1; 모델 제안은 2013a)
- Negative Sampling 손실(한 쌍): −log σ(v'_{o} · v_{w}) − Σ_{i=1..k} log σ(−v'_{n_i} · v_{w}) (Mikolov et al., 2013b §2.2 식 4)
  - σ(x) = 1 / (1 + e^{−x}), v_w: 가운데 단어 벡터, v'_o: 진짜 주변 단어 벡터, n_i: 오답 단어
- 유사도: cos(a, b) = (a · b) / (‖a‖ ‖b‖) — P1-02와 같다

## 6. 실행법

저장소 루트에서 실행합니다.

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"
pip install -r requirements/part1.txt

# 1) 남은 TODO 확인
python part1_ai_fundamentals/word2vec/starter.py

# 2) 직접 구현 검사 (TODO 가 남은 테스트는 skip)
$env:IMPL = "starter"; pytest part1_ai_fundamentals/word2vec/tests; Remove-Item Env:IMPL

# 3) 정답 코드 검사
pytest part1_ai_fundamentals/word2vec/tests

# 4) 데모 (다운로드·API 키 불필요, CPU 수 초)
python part1_ai_fundamentals/word2vec/run_demo.py --impl starter
```

**bash (Linux/macOS)**

```bash
source .venv/bin/activate
python part1_ai_fundamentals/word2vec/starter.py
IMPL=starter pytest part1_ai_fundamentals/word2vec/tests
pytest part1_ai_fundamentals/word2vec/tests
python part1_ai_fundamentals/word2vec/run_demo.py --impl solution
```

- 테스트는 정확한 값 대신 상대 비교(cat–dog > cat–coffee 등)로 검사합니다. 반복은 모두 길이가 정해진 `for`라 멈추지 않는 일이 없습니다.
- 데모의 [5]는 P1-02의 `embedding/solution.py`에 있는 `fake_embed`를 불러와 비교합니다.

## 7. TODO 목록

| TODO | 함수 | 내용 |
|---|---|---|
| 1 | `preprocess` | 소문자 → 구두점 제거 → 공백 분리, 빈 줄 버리기 |
| 2 | `train_word2vec` | `Word2Vec(...)` → `build_vocab` → (`epochs > 0`이면) `train`, `workers=1`·`sample=0` 고정 |
| 3 | `pair_similarities`, `nearest` | 쌍 코사인(OOV면 `None`), `most_similar` 최근접(OOV면 `[]`) |
| 4 (선택) | `sentence_vector` | 아는 단어 벡터의 평균, 모두 OOV면 0 벡터 |

제공 코드: `load_lines`(말뭉치 읽기).

## 8. 예상 결과

아래는 클라우드 작업 환경(CPU, Python 3.12.13, gensim 4.4.0, numpy 2.5.4)에서 `run_demo.py --impl solution`을 **실제로 실행한 출력**입니다. 학습 시간은 실행마다 조금 다릅니다.

```
[1] 전처리: 문장 48개, 토큰 504개, 고유 단어 345개
    예) 'the cat sat on the mat and the dog sat on the log.' → ['the', 'cat', 'sat', 'on', 'the', 'mat', 'and', 'the', 'dog', 'sat', 'on', 'the', 'log']
    예) '학습이 잘 되면 손실 값이 조금씩 줄어든다.' → ['학습이', '잘', '되면', '손실', '값이', '조금씩', '줄어든다']
[2] 학습 (vector_size=32, window=2, negative=5, epochs=200, seed=0, workers=1)
    skip-gram 0.20초
    cbow      0.16초
[3] 단어 쌍 코사인: 학습 전(무작위 초기값) → 학습 후 skip-gram / cbow
     cat–dog     -0.32 →  0.97 /  0.95
     mat–log     -0.04 →  0.92 /  0.91
     cat–coffee   0.20 →  0.53 /  0.63
     mat–coffee   0.06 →  0.65 /  0.61
     cat–강아지       OOV →   OOV /   OOV
[4] 최근접 단어 top-5 (skip-gram)
    cat: dog(0.97), sat(0.94), size(0.94), log(0.93), likes(0.93)
    손실: 줄어든다(0.97), 조금씩(0.96), 줄지(0.96), 되면(0.93), 값이(0.91)
    강아지: [] (OOV — 학습 데이터에 없는 단어)
[5] (선택) 문장 검색: Word2Vec 평균 vs P1-02 가짜 임베딩(fake_embed, 글자 2-gram)
    질의: a dog and a cat  (토큰 ['a', 'dog', 'and', 'a', 'cat'], 어휘에 있는 것 ['a', 'dog', 'and', 'a', 'cat'])
      word2vec 평균: #39(0.89), #41(0.88), #32(0.86)
      fake_embed   : #33(0.44), #24(0.43), #41(0.41)
    질의: dog cat  (토큰 ['dog', 'cat'], 어휘에 있는 것 ['dog', 'cat'])
      word2vec 평균: #45(0.94), #46(0.91), #29(0.87)
      fake_embed   : #45(0.45), #33(0.39), #40(0.34)
    질의: 강아지와 고양이  (토큰 ['강아지와', '고양이'], 어휘에 있는 것 [])
      word2vec 평균: 검색 불가 (모든 토큰이 OOV → 0 벡터)
      fake_embed   : #48(0.21), #47(0.18), #32(0.17)
    참고 #45: the cat sat on the mat and the dog sat on the log.
    참고 #46: the cat likes the warm sun and the dog likes the long walk.
```

**해석**
- [3] 학습 전에는 무작위라 cat–dog가 −0.32였지만, 학습 후 0.97(Skip-gram)이 되었습니다. 대조군 cat–coffee(0.53)보다 확실히 높습니다. 다만 **대조군도 0.5~0.65로 꽤 높습니다.** 말뭉치가 504토큰뿐이라 모든 단어가 서로 비슷해지기 때문이며, 그래서 숫자 하나가 아니라 대조군과의 **차이**를 봅니다.
- [4] `cat`의 최근접에 `dog`·`log`(같은 자리에 오는 단어)와 함께 `sat`·`size`가 섞였습니다. 작은 말뭉치에서는 "같이 자주 나온 단어"와 "같은 자리에 오는 단어"가 함께 가까워집니다. `손실`의 이웃 `줄어든다`·`줄지`는 반대 방향의 말이 같은 문맥이라 가까워진 예입니다. 유사어 확장에 그대로 쓰면 위험한 이유입니다.
- [5] Word2Vec 평균으로 `a dog and a cat`을 검색하면 정답 #45·#46이 상위 3에 없습니다. 흔한 단어 `a`·`and`가 평균을 끌고 가기 때문입니다. 흔한 단어를 뺀 `dog cat`은 #45·#46이 1·2위입니다. `강아지와 고양이`는 모든 단어가 OOV라 검색 자체가 안 됩니다(`fake_embed`의 0.2 안팎은 해시 잡음). 실제 문장 임베딩 모델의 결과는 P1-02 `embed_with_openai.py`로 사용자 PC에서 확인하세요.

## 9. 자주 나는 오류

| 증상 | 원인 | 해결 |
|---|---|---|
| `KeyError: "Key 'xxx' not present"` | OOV 단어를 바로 조회 | `word in model.wv`로 먼저 확인 |
| 실행할 때마다 결과가 다름 | 여러 스레드 또는 시드 미지정 | `seed` 고정 + `workers=1` |
| `mat`과 `mat.`이 다른 단어 | 구두점 미제거 | `re.sub(r"[^\w\s]", " ", s)` |
| 학습해도 유사도가 거의 그대로 | `train` 누락 또는 `epochs`가 너무 작음 | `build_vocab` 후 `train(..., epochs=epochs)` |
| `train` 호출 오류 `total_examples` | 문장 수를 안 넘김 | `total_examples=len(sentences)` |
| 다른 PC에서 숫자가 조금 다름 | gensim 버전·플랫폼 차이 | 상대 비교로 판단(테스트도 그렇게 작성됨) |

## 10. 과제

1. `window`를 1 / 2 / 5로 바꿔 `cat`의 최근접 5개가 어떻게 바뀌는지 기록하세요. 문법적 이웃과 주제적 이웃 중 어느 쪽이 늘어나나요?
2. `epochs`를 5 / 50 / 200으로 바꿔 cat–dog와 대조군 cat–coffee의 차이를 표로 만드세요.
3. `sentence_vector`에서 흔한 단어(the, a, and, on)를 빼고 [5]의 `a dog and a cat`을 다시 검색해 보세요. 결과가 어떻게 바뀌나요? 실무에서 이 방식의 한계는 무엇인가요?
4. 사내 FAQ 검색에 "유사어 확장"을 넣는다고 할 때, Word2Vec 최근접 단어를 사전에 넣기 전에 무엇을 검토해야 할지 3가지 적어 보세요.

## 11. 체크리스트

- [ ] `IMPL=starter pytest part1_ai_fundamentals/word2vec/tests` 가 모두 통과한다
- [ ] 분포 가설을 예문으로 설명할 수 있다
- [ ] Skip-gram과 CBOW의 입력·출력을 그림으로 그릴 수 있다
- [ ] Negative Sampling이 빠른 이유를 "계산하는 단어 수"로 설명할 수 있다
- [ ] 정적 임베딩이 다의어·OOV에 약한 이유를 안다
- [ ] 실무 RAG가 문장 임베딩 모델을 쓰는 이유와, 그래도 Word2Vec을 쓰는 경우를 말할 수 있다

## 12. 다음 단계 연결

- **P1-05 Transformer**: 어텐션이 문맥에 따라 단어 표현을 바꾸는 원리 → "단어 하나에 벡터 하나" 한계의 해법.
- **Part 3**: 문장 임베딩(Dense) 검색과 BM25를 비교합니다. 이 실습의 유사어 확장은 BM25 질의 확장 아이디어로 이어집니다.
- 원리를 직접 구현하고 싶다면 선택 과제 [advanced/](./advanced/README.md) (Skip-gram·CBOW·Negative Sampling, P1-03A).

## REF

- Mikolov, Chen, Corrado, Dean, 2013a, "Efficient Estimation of Word Representations in Vector Space" — CBOW·Skip-gram. https://arxiv.org/abs/1301.3781 (작성 환경 접속 불가 → 미확인)
- Mikolov, Sutskever, Chen, Corrado, Dean, 2013b, "Distributed Representations of Words and Phrases and their Compositionality", §2.2 Negative Sampling. https://arxiv.org/abs/1310.4546 (미확인)
- gensim Word2Vec API: https://radimrehurek.com/gensim/models/word2vec.html (미확인). 인자 이름은 설치된 gensim 4.4.0에서 실행으로 확인

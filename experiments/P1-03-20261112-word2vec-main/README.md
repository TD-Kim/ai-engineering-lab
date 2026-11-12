# P1-03 Word2Vec (메인) — 실행 기록

- 실습 ID: P1-03 (메인)
- 날짜: 2026-11-12
- 실행 위치: 클라우드 작업 환경(CPU)

## 목적
작은 말뭉치에서 gensim Word2Vec이 (1) 같은 문맥 구조의 단어 쌍을 대조군보다 가깝게 만드는지, (2) 학습 전후가 어떻게 다른지, (3) 최근접 단어·OOV·평균 문장 벡터의 한계가 어떻게 드러나는지 확인한다.

## 데이터셋
- `part1_ai_fundamentals/data/corpus_ko_en.txt` 48줄 → 소문자·구두점 제거·공백 분리: 토큰 504개, 고유 단어 345개

## 모델
- gensim 4.4.0 `Word2Vec` (Skip-gram `sg=1`, CBOW `sg=0`)

## 하이퍼파라미터
| 이름 | 값 |
|---|---|
| vector_size | 32 |
| window | 2 |
| min_count | 1 |
| negative / hs | 5 / 0 |
| sample | 0 |
| epochs | 200 |
| seed / workers | 0 / 1 |

## 환경
| 항목 | 값 |
|---|---|
| Python | 3.12.13 |
| gensim | 4.4.0 |
| numpy | 2.5.4 |

## GPU 사용량
해당 없음(CPU, 모델 하나 학습 약 0.2초).

## 토큰 사용량·비용
API 호출 없음 → $0.

## 결과
명령: `python part1_ai_fundamentals/word2vec/run_demo.py --impl solution` (실제 출력, 학습 시간은 실행마다 조금 다름)

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

추가 측정(작성 중 탐색, 같은 설정): 시드 0~9 각각으로 학습했을 때 `cat–dog > cat–coffee`, `mat–log > mat–coffee`가 Skip-gram epochs 50·200, CBOW epochs 200에서 10/10번 성립. CBOW epochs 50에서는 `cat–dog > cat–coffee`가 8/10번. 같은 시드를 다른 프로세스에서 다시 실행해도 cat–dog 0.967357로 같았다.

## 비교
- 학습 전 cat–dog −0.32 → 학습 후 0.97(Skip-gram)/0.95(CBOW). 대조군 cat–coffee 0.53/0.63.
- 대조군도 0.5 이상: 작은 말뭉치에서는 전반적으로 유사도가 높다 → 상대 비교로 판단.
- 평균 문장 벡터: `a dog and a cat`은 흔한 단어에 끌려 정답(#45·#46) 실패, `dog cat`은 1·2위. 한국어 질의 `강아지와 고양이`는 전부 OOV.

## 개선점 / 다음에 확인할 것
- README 과제 1·2(window, epochs 변화) 결과 기록.
- 문장 임베딩 모델과의 비교는 P1-02 `embed_with_openai.py`를 사용자 PC에서 실행한 뒤 기록(예상치를 미리 쓰지 않음).

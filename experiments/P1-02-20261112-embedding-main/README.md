# P1-02 Embedding (메인) — 실행 기록

- 실습 ID: P1-02 (메인)
- 날짜: 2026-11-12
- 실행 위치: 클라우드 작업 환경(CPU). OpenAI API 키 없음 → `embed_with_openai.py`는 **사용자 PC에서 실행 필요**

## 목적
(1) 정규화·코사인의 손계산 값이 코드와 같은지, (2) 배치 호출 수와 저장 크기가 수식대로인지, (3) 글자 겹침 기반 가짜 임베딩으로 top-k 검색이 어디까지 되고 어디서 실패하는지, (4) 캐시가 중복·재요청 임베딩을 없애는지 확인한다.

## 데이터셋
- 문서: `part1_ai_fundamentals/data/corpus_ko_en.txt` 48줄(한국어 22, 영어 26), 한 줄 = 문서 하나
- 질의 3개: `손실 값이 줄지 않을 때는 무엇을 확인하나`, `what to check if the loss does not go down`, `강아지와 고양이`

## 모델
- `fake_embed`: 글자 2-gram feature hashing(md5), 256차원. 학습된 모델이 아님(뜻을 모름)
- 실제 임베딩 API 호출 없음

## 하이퍼파라미터
| 이름 | 값 |
|---|---|
| 차원 | 256 |
| batch_size | 16 |
| k | 3 |

## 환경
| 항목 | 값 |
|---|---|
| Python | 3.12.13 |
| numpy | 2.5.4 |
| openai | 3.49.0 (설치만, 키 없음) |

## GPU 사용량
해당 없음(CPU, 1초 이내).

## 토큰 사용량·비용
- 실제 API 호출 없음 → 실제 비용 $0, 토큰 0.
- OpenAI 임베딩 단가는 작성 환경에서 공식 페이지(https://platform.openai.com/docs/pricing)에 접속할 수 없어 기록하지 않음(미확인).

## 결과
명령: `python part1_ai_fundamentals/embedding/run_demo.py --impl solution` (실제 출력)

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

선택 스크립트:
| 스크립트 | 이 환경 결과 |
|---|---|
| `embed_with_openai.py` | `건너뜀: OPENAI_API_KEY 가 없습니다` — 사용자 PC에서 실행 필요 |

## 비교
- 글자가 겹치는 질의 2개는 정답 문서가 1위(#07 0.502, #29 0.781).
- 뜻은 같지만 글자가 겹치지 않는 `강아지와 고양이`는 정답(#45·#46)이 상위 3에 없음. 점수 0.17~0.21은 해시 충돌 잡음.
- 배치 3회 = ⌈48 / 16⌉, 저장 49,152 B = 48 × 256 × 4 — 수식과 일치.

## 개선점 / 다음에 확인할 것
- 사용자 PC에서 `embed_with_openai.py`로 같은 세 질의의 top-3과 `prompt_tokens`를 기록(예상치를 미리 쓰지 않음). 특히 교차 언어(한국어 질의 → 영어 문서) 결과.
- README 과제 1: `dim` 32/256/4096에서 잡음 점수 변화.

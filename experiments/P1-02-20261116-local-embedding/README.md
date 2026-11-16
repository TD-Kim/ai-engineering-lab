# P1-02 Embedding — 로컬 임베딩 공급자 (EMBEDDING_PROVIDER) 실행 기록

- 실습 ID: P1-02 (메인) 확장 — 로컬 임베딩 옵션
- 날짜: 2026-11-16
- 실행 위치: 클라우드 작업 환경(CPU). sentence-transformers·Ollama 서버·임베딩 모델이 없어 **실제 로컬 임베딩은 실행하지 못했다**. 아래 "결과"는 안내 메시지·가짜 임베딩 경로의 실제 출력이고, **로컬 모델 측정 표는 사용자 PC에서 채울 칸**이다.

## 목적
(1) `EMBEDDING_PROVIDER`로 공급자를 바꿔도 같은 `embed_texts`·`top_k` 코드가 동작하는지, (2) 로컬 모델(sentence-transformers e5/bge-m3, Ollama)이 글자가 겹치지 않는 질의(`강아지와 고양이` → #45·#46)를 찾는지, (3) 문서 임베딩 시간·차원·질의 지연이 모델·장치마다 어떻게 다른지 확인한다. 가설을 정하지 않는다(개선 가정 금지).

## 데이터셋
- 문서: `part1_ai_fundamentals/data/corpus_ko_en.txt` 48줄, 질의 3개(`run_demo.QUERIES`와 같음)

## 모델
- 작성 환경: `fake`(글자 2-gram 해시, 256차원)만 실행
- 사용자 PC 후보: `intfloat/multilingual-e5-small`(기본), `BAAI/bge-m3`, Ollama 임베딩 모델(태그는 ollama.com 라이브러리에서 확인 — 이 저장소는 정하지 않음)

## 하이퍼파라미터
| 이름 | 값 |
|---|---|
| batch_size | 32(문서), 질의는 1개씩 |
| k | 3 |
| 접두어 | e5: 질문 `query: `, 문서 `passage: ` / bge-m3: 없음 |
| 정규화 | sentence_transformers `normalize_embeddings=True`, Ollama `/api/embed`는 L2 정규화 벡터(공식 문서) |

## 환경
| 항목 | 작성 환경 | 사용자 PC |
|---|---|---|
| Python | 3.12.13 | |
| sentence-transformers | 미설치 | (측정 시 버전 기록) |
| Ollama | 없음 | (버전·모델 태그 기록) |
| 장치 | CPU | (GPU 이름·VRAM 또는 CPU) |

## GPU 사용량
작성 환경: 해당 없음. 사용자 PC: 측정 필요(`nvidia-smi`로 모델 적재 후 VRAM 기록).

## 토큰 사용량·비용
로컬 공급자는 호출 비용 0(전기·장비 비용 제외). OpenAI 비교는 `embed_with_openai.py`의 `usage.prompt_tokens`(P1-02 기록 참고). 단가는 공식 페이지 확인.

## 로컬 임베딩 측정 — 사용자 PC 측정 필요

`python part1_ai_fundamentals/embedding/embed_with_provider.py --provider <공급자>` 출력을 옮겨 적는다(같은 문서·질의·장치로 비교).

| 공급자 / 모델 | 차원 | 문서 48개 임베딩 시간(첫 실행 / 두 번째) | 질의 지연(ms, 평균) | `강아지와 고양이` top-3 에 #45·#46 포함? | 한국어 질의 → 영어 문서 | 메모 |
|---|---|---|---|---|---|---|
| fake | 256 | (작성 환경 실행: 0.0x초) | ~0 | 아니오(위 결과) | — | 글자 겹침만 봄 |
| sentence_transformers / multilingual-e5-small | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | |
| sentence_transformers / bge-m3 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | |
| ollama / (태그) | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | |
| openai / text-embedding-3-small(선택) | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 유료 |

접두어 실험(README 과제 6): e5에서 접두어를 빼면 top-3·점수 변화 — 측정 필요.

## 결과 (작성 환경 실제 출력)
```
$ python embed_with_provider.py --provider fake
provider=fake model=(기본값)  문서 48개 × 256차원, 문서 임베딩 0.0x초(처음 실행은 모델 적재 포함)
질의: 손실 값이 줄지 않을 때는 무엇을 확인하나  (0 ms)
  0.502  #07 손실 값이 줄지 않으면 학습률과 데이터부터 다시 확인한다.
  0.328  #06 학습이 잘 되면 손실 값이 조금씩 줄어든다.
  0.292  #18 실험 기록을 남기면 나중에 무엇이 달라졌는지 비교할 수 있다.
질의: what to check if the loss does not go down  (0 ms)
  0.781  #29 If the loss does not go down, check the 
  0.587  #28 When training goes well, the loss goes d
  0.523  #45 the cat sat on the mat and the dog sat o
질의: 강아지와 고양이  (0 ms)
  0.208  #48 the lower the learning rate, the slower 
  0.178  #47 a new word is made of smaller pieces suc
  0.170  #32 Mixing the value vectors with those weig

$ EMBEDDING_PROVIDER=sentence_transformers python embed_with_provider.py
건너뜀: sentence_transformers 패키지가 없습니다. 이 공급자를 쓰려면 설치하세요(requirements/part1.txt 의 선택 의존성 주석).

$ EMBEDDING_PROVIDER=ollama python embed_with_provider.py   (EMBEDDING_MODEL 없음)
건너뜀: EMBEDDING_PROVIDER=ollama 에는 EMBEDDING_MODEL(Ollama 임베딩 모델 태그)이 필요합니다. ollama.com 라이브러리에서 임베딩 모델을 고르고 `ollama pull <태그>` 후 .env 에 적으세요.

$ EMBEDDING_PROVIDER=ollama EMBEDDING_MODEL=my-model python embed_with_provider.py   (서버 꺼짐)
[오류] Ollama 서버에 연결할 수 없습니다(http://localhost:11434/api/embed). `ollama serve` 가 켜져 있는지 확인하세요: <urlopen error [Errno 111] Connection refused>
```
- `fake`는 P1-02 데모 [3]과 같은 값(재현). `sentence_transformers`·`ollama`는 모델·서버가 없어 안내 메시지로 끝났다(실패가 아니라 의도한 `건너뜀`/`[오류]` 안내).

## 관찰
- 공급자를 바꿔도 `embed_texts`·`top_k`는 한 줄도 바뀌지 않았다(`make_embed_fn`이 같은 모양의 `embed_fn`을 돌려줌). 요청 형식(`/api/embed`, `{model, input}`)과 e5 접두어 부착은 **가짜 서버·가짜 모델로 단위 테스트**했다(실제 서버 응답은 아님).
- 로컬 모델의 품질·속도에 대한 결론은 아직 없다(측정 전).

## 다음 실험
- 사용자 PC: 위 표 채우기. 로컬 두 모델을 같은 조건으로 비교하고, 내 도메인 한국어 질의·정답 쌍으로 top-k 적중률 측정(Part 3 평가의 예행연습).

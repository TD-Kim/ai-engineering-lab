# P1-02 Embedding — 실행 기록

- 실습 ID: P1-02
- 날짜: 2026-11-09
- 실행 위치: 클라우드 작업 환경(CPU). GPU·사전학습 비교 수치는 **사용자 PC에서 측정 필요**

## 목적
1. 직접 구현한 임베딩이 `nn.Embedding`과 출력·기울기가 같은지 확인한다(`padding_idx` 포함).
2. "다음 단어 맞히기"만으로 학습해도, 같은 자리에 오는 단어의 벡터가 가까워지는지 학습 전후 최근접 이웃으로 확인한다.

## 데이터셋
`part1_ai_fundamentals/data/corpus_ko_en.txt`: 저장소에 포함되어 있고 다운로드가 필요 없다. 공백 기준 단어를 쓰며, 같은 줄 안의 (현재 단어 → 다음 단어) 쌍을 만든다.

| 항목 | 값 |
|---|---|
| 어휘 크기 | 365 |
| 학습 쌍 | 456 |
| 관찰 단어 등장 횟수 | the 34, model 4, cat 2, dog 2, loss 2 |

## 모델
`EmbeddingLayer(365, 16)` (직접 구현, N(0,1) 초기화) → `nn.Linear(16, 365)` → cross entropy

## 하이퍼파라미터
| 이름 | 값 |
|---|---|
| 임베딩 차원 | 16 |
| optimizer | Adam, lr 0.05 |
| steps | 300 (전체 배치) |
| seed | 0 |

## 환경
| 항목 | 값 |
|---|---|
| Python | 3.12.13 |
| torch | 2.15.0+cu132 (PyPI 휠, CUDA 사용 불가 → device=cpu) |
| matplotlib | 3.11.2 |
| sentence-transformers | 미설치 → 사전학습 비교 건너뜀 |

## GPU 사용량
| 환경 | 최대 할당 메모리 |
|---|---|
| CPU (이 기록) | 해당 없음 |
| RTX 3070 | 사용자 PC에서 측정 필요 (모델이 작아 수 MB 수준일 것으로 예상, 측정 전까지 미기록) |

## 학습 시간
| 항목 | CPU (이 기록) | RTX 3070 |
|---|---|---|
| `run_demo.py` 학습 300 steps | 0.20s | 사용자 PC에서 측정 필요 |
| `pytest part1_ai_fundamentals/embedding` | 1.55s | 사용자 PC에서 측정 필요 |

## 결과

pytest: solution `12 passed, 1 skipped`, starter(빈 상태) `13 skipped`. skip 1건은 사전학습 비교이며, sentence-transformers가 없어서 건너뛰었다.

| 단어 | 학습 전 1순위 이웃 (코사인) | 학습 후 1순위 이웃 (코사인) |
|---|---|---|
| cat | 각 (0.723) | **dog (0.770)** |
| dog | Mixing (0.657) | **cat (0.770)** |
| the | help (0.647) | training, (0.610) |
| loss | different (0.633) | 설정으로 (0.613) |
| model | gradients (0.689) | turns (0.598) |

- loss: 6.0347 → 0.5645. 시작값은 무작위 예측 수준인 `ln 365 ≈ 5.90` 근처다.
- 학습 후 PCA 좌표 (PC1, PC2): cat (-2.57, 0.46), dog (-1.14, 0.61), the (2.13, 1.64), loss (-1.54, 0.23), model (-2.22, -1.27)
- 그림 `part1_ai_fundamentals/embedding/outputs/pca_{before,after}.png`는 커밋하지 않는다(`.gitignore`).

사전학습 임베딩 비교(`compare_pretrained.py`): **이 환경에서는 실행하지 않음**(건너뜀). 사용자 PC 결과를 아래에 채운다.

| 문장 쌍 | 사전학습 코사인 | 무작위 임베딩 평균 코사인 |
|---|---|---|
| (사용자 PC에서 측정 필요) | | |

## 비교
- `nn.Embedding(padding_idx=0)`과 출력·기울기가 일치한다(테스트 `test_matches_nn_embedding_outputs_and_grads`). 등장하지 않은 ID와 패딩 행의 기울기는 0이다.
- 학습 후 `cat`과 `dog`가 서로 1순위 이웃이 되었다. 말뭉치에서 둘 다 `the` 뒤, `sat`/`likes` 앞에 온다.
- 나머지 단어는 이웃에 뜻이 없다. 등장이 2~4회뿐이고, loss 0.56은 말뭉치를 외운(과적합) 결과다.

## 개선점 / 다음에 해 볼 것
- README 과제 1: 차원 2, 8, 64에서 이웃과 loss를 비교한다.
- README 과제 2: 내적 기준 이웃과 코사인 기준 이웃을 비교한다.
- 사용자 PC에서 `compare_pretrained.py`를 실행해 위 표를 채운다(로컬 모델, 라이선스 확인).
- 실제 단어 벡터 품질은 P1-03 Word2Vec(더 큰 생성 말뭉치, 주변 단어 전체 예측)에서 다시 본다.

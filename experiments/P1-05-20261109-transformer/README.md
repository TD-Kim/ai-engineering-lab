# P1-05 Transformer — 실행 기록

- 실습 ID: P1-05 (P1-05a 부품, P1-05b 미니 언어모델)
- 날짜: 2026-11-09
- 실행 위치: 클라우드 작업 환경(CPU). GPU 수치는 **사용자 PC에서 측정 필요**

## 목적
1. 직접 구현한 어텐션이 PyTorch 내장 구현과 같은 가중치에서 같은 출력을 내는지 확인한다.
2. 작은 Transformer 문자 단위 언어모델로 학습 손실이 줄어드는지 확인하고, 검증 손실로 과적합 여부를 본다.
3. 어텐션 가중치를 그림으로 보고 causal 마스크가 지켜지는지 확인한다.

## 데이터셋
`part1_ai_fundamentals/data/corpus_ko_en.txt`: 저장소에 포함되어 있고 다운로드가 필요 없다.

| 항목 | 값 |
|---|---|
| 문자 수 | 2,492 |
| 어휘(고유 문자) | 225 |
| 학습 / 검증 분할 | 앞 90% (2,242자) / 뒤 10% (250자, 마지막 영어 문장들) |

## 모델
`MiniTransformerLM`: 문자 임베딩 ×√D, 사인/코사인 위치 인코딩, Pre-LN TransformerBlock × 2, `ln_f`, `lm_head`.

| 이름 | 값 |
|---|---|
| d_model / heads / d_k | 64 / 4 / 16 |
| layers | 2 |
| FFN | 64 → 256 → 64, GELU |
| block size (max_len) | 64 |
| 학습 파라미터 수 | 128,384 (위치 인코딩 buffer 제외) |

## 하이퍼파라미터
| 이름 | 값 |
|---|---|
| optimizer | AdamW, lr 3e-3 (기본 weight decay) |
| batch size | 32 |
| steps | 300 |
| dropout | 0 |
| 평가 | 50 step 마다, 고정 위치 32개 묶음(seed 1234)으로 train/val 손실 |
| seed | 0 |

## 환경
| 항목 | 값 |
|---|---|
| Python | 3.12.13 |
| torch | 2.15.0+cu132 (PyPI 휠, CUDA 사용 불가 → device=cpu) |
| matplotlib | 3.11.2 |

## GPU 사용량
| 환경 | 최대 할당 메모리 |
|---|---|
| CPU (이 기록) | 해당 없음 |
| RTX 3070 | 사용자 PC에서 측정 필요 (`torch.cuda.max_memory_allocated()`) |

## 학습 시간
| 항목 | CPU (이 기록) | RTX 3070 |
|---|---|---|
| `train_tiny.py` 300 steps | 5.31s | 사용자 PC에서 측정 필요 |
| `pytest part1_ai_fundamentals/transformer` | 2.62s (19 passed) | 사용자 PC에서 측정 필요 |

## 결과

### 손실 곡선 (`train_tiny.py --impl solution`)
| step | 0 | 50 | 100 | 150 | 200 | 250 | 300 |
|---|---|---|---|---|---|---|---|
| train loss | 5.5856 | 2.4571 | 2.0545 | 1.8206 | 1.6163 | 1.4533 | 1.3197 |
| val loss | 5.5983 | 2.4659 | **2.4568** | 2.5139 | 2.6141 | 2.6836 | 2.7377 |

- 시작값은 무작위 예측 수준인 `ln 225 ≈ 5.42` 근처다.
- 검증 손실은 step 100에서 가장 낮고 이후 오른다. 작은 말뭉치에서 생긴 과적합이다.
- 600 step으로 따로 실행하면 train 0.49, val 4.42까지 벌어졌다(같은 설정, 참고용).
- 생성 예시(프롬프트 `"the "`, temperature 0.8): `'the ttaler ny.\nWes s deend dand landang a s anday theelano veani'`

### 내장 구현 비교 (`compare_with_torch.py`, d_model 16, heads 4, B 2, T 6)
| 비교 | 최대 절대 오차 |
|---|---|
| SelfAttention vs `F.scaled_dot_product_attention` (마스크 없음 / causal) | 4.47e−08 / 4.47e−08 |
| MultiHeadAttention vs `nn.MultiheadAttention` 출력 (없음 / causal / causal+padding) | 0 / 0 / 0 |
| 〃 head별 가중치 | 0 / 0 / 0 |

### 어텐션 가중치 (`visualize_attention.py`, 문장 `the cat sat on the mat`)
- 두 층 모두 미래 위치 가중치 합이 0.0, 행 합 평균이 1.000이다(causal 마스크와 softmax 정상).
- 각 query가 가장 많이 본 key가 "바로 앞 글자"인 비율은 head별로 0.10~0.29다. 뚜렷한 "앞 글자 head"는 생기지 않았다.
- layer 1 head 2는 거의 모든 query가 첫 단어의 `e`(위치 2)를 강하게 본다. 이 패턴의 의미는 이 실험만으로 단정하지 않는다.
- 그림 `part1_ai_fundamentals/transformer/outputs/attention_layer{0,1}.png`는 커밋하지 않는다.

### 테스트
pytest: solution `19 passed`, starter(빈 상태) `19 skipped`.
스크래치 변이 확인 결과, 아래 변이가 모두 하나 이상의 테스트에서 실패한다.
- √d_k 누락
- softmax dim 오류
- padding 마스크를 query 축에 적용
- LM causal 누락
- MHA를 D로 스케일
- 잔차 누락
- 위치 인코딩 sin/cos 뒤바뀜

## 비교
- 직접 구현은 PyTorch 내장 구현과 수식이 같다. 마스크 규약만 반대다(직접 구현 True = 볼 수 있음, PyTorch bool 마스크 True = 가림).
- 학습 손실 감소는 확인했다. 일반화(검증 손실)는 데이터가 작아 step 100 이후 나빠진다.

## 개선점 / 다음에 해 볼 것
- README 과제 2: head 수별 검증 손실과 시간을 비교한다.
- README 과제 3: Post-LN과 Pre-LN을 비교한다(층 수 6).
- README 과제 4: dropout, 조기 종료, 더 큰 공개 말뭉치로 과적합을 완화한다 → Part 2와 연결된다.
- 사용자 PC(RTX 3070)에서 같은 명령으로 실행해 시간·메모리를 위 표에 추가한다.

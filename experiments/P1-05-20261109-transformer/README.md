# P1-05 Transformer — 실행 기록

> 이 기록은 **심화 실습(선택 과제 `part1_ai_fundamentals/transformer/advanced/`)** 의 실행 결과다. 2026-11-09 방향 전환으로 기존 실습이 `advanced/`로 옮겨졌고, 메인 실습 기록은 별도 디렉터리에 남긴다.

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
`MiniTransformerLM`은 다음 순서로 구성된다.
- 문자 임베딩(**std = D^−0.5 초기화** 후 ×√D)
- 사인/코사인 위치 인코딩
- Pre-LN TransformerBlock × 2
- `ln_f`
- `lm_head`

| 이름 | 값 |
|---|---|
| d_model / heads / d_k | 64 / 4 / 16 |
| layers | 2 |
| FFN | 64 → 256 → 64, GELU |
| block size (max_len) | 64 |
| 학습 파라미터 수 | 128,384 (위치 인코딩 buffer 제외) |
| 초기 토큰 벡터 노름 (×√D 후) / 위치 인코딩 노름 | ≈ 7.9 / 5.66 |

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
| `train_tiny.py` 300 steps | 4.40s | 사용자 PC에서 측정 필요 |
| `pytest part1_ai_fundamentals/transformer/advanced` | 2.32s (22 passed) | 사용자 PC에서 측정 필요 |

## 결과

### 손실 곡선 (`train_tiny.py --impl solution`, 정답 기준: std = D^−0.5 초기화 × √D)
| step | 0 | 50 | 100 | 150 | 200 | 250 | 300 |
|---|---|---|---|---|---|---|---|
| train loss | 5.5758 | 2.2227 | 1.7111 | 1.0550 | 0.4787 | 0.2554 | 0.1919 |
| val loss | 5.6084 | **2.4005** | 2.4925 | 2.9526 | 3.6825 | 4.0081 | 4.3465 |

- 시작값은 무작위 예측 수준인 `ln 225 ≈ 5.42` 근처다.
- 검증 손실은 step 50에서 가장 낮고 이후 가파르게 오른다. 작은 말뭉치를 외운 과적합이다.
- 생성 예시(프롬프트 `"the "`, temperature 0.8): `'the data a owekends I tak ume a at다.\nResidual ply tipries ape wh'`

### 임베딩 크기 대조 실험 (같은 seed·설정, 300 step)
| 임베딩 설정 | 초기 토큰 노름 | train loss 300 | 최저 val loss |
|---|---|---|---|
| std = D^−0.5 × √D (**정답 기준**) | ≈ 7.9 | 0.1919 | 2.4005 @ 50 |
| std = D^−0.5, ×√D 없음 | ≈ 1.0 | 0.5396 | 2.4350 @ 100 |
| N(0, 1) × √D (이전 버전, 커밋 f923a2b) | ≈ 64 | 1.2688 | 2.4502 @ 100 |

- 이전 버전은 토큰 벡터 노름(실측 64.1, 검증 세션 측정 65.4)이 위치 인코딩 노름(5.66)을 압도했다. 그래서 위치 정보가 거의 쓰이지 않았다.
- 크기를 맞추자 학습 손실이 가장 빨리 줄었다. 최저 검증 손실 차이는 작아(2.40 vs 2.44~2.45) 일반화 개선이라고 단정하지 않는다.
- 이전 버전 단독 실행의 기록은 train 1.3197, val 최저 2.4568 @ 100이었다. 대조 실험에서는 초기화 뒤 난수 상태가 달라 값이 약간 다르다.

### 내장 구현 비교 (`compare_with_torch.py`, d_model 16, heads 4, B 2, T 6)
| 비교 | 최대 절대 오차 |
|---|---|
| SelfAttention vs `F.scaled_dot_product_attention` (마스크 없음 / causal) | 4.47e−08 / 4.47e−08 |
| MultiHeadAttention vs `nn.MultiheadAttention` 출력 (없음 / causal / causal+padding) | 0 / 0 / 0 |
| 〃 head별 가중치 | 0 / 0 / 0 |

### 어텐션 가중치 (`visualize_attention.py`, 문장 `the cat sat on the mat`)
- 두 층 모두 미래 위치 가중치 합이 0.0, 행 합 평균이 1.000이다(causal 마스크와 softmax 정상).
- "바로 앞 글자"를 가장 많이 본 비율:
  - layer 0: head 0 0.19, head 1 **0.38**, head 2 **0.43**, head 3 0.14
  - layer 1: 0.14~0.29
  - 이전 버전(0.10~0.29)보다 layer 0 head 1·2가 높다. 그림에서도 대각선 바로 아래가 밝다.
- layer 0 head 0은 여러 query가 `c`(위치 4)에 몰린다. 의미는 단정하지 않는다.
- 그림 `part1_ai_fundamentals/transformer/advanced/outputs/attention_layer{0,1}.png`는 커밋하지 않는다.

### 테스트
pytest: solution `22 passed`, starter(빈 상태) `22 skipped`. 위치 인코딩 사용, 임베딩 크기, finfo.min 마스킹(NaN 없음) 테스트가 추가되었다.
스크래치 변이 확인 결과, 아래 변이가 모두 하나 이상의 테스트에서 실패한다.
- √d_k 누락
- softmax dim 오류
- padding 마스크를 query 축에 적용
- LM causal 누락
- MHA를 D로 스케일
- 잔차 누락
- 위치 인코딩 sin/cos 뒤바뀜
- 위치 인코딩 누락(`test_lm_uses_position`)
- 기본 N(0,1) 초기화(`test_token_embedding_scale_matches_positional_encoding`)

## 비교
- 직접 구현은 PyTorch 내장 구현과 수식이 같다. 마스크 규약만 반대다(직접 구현 True = 볼 수 있음, PyTorch bool 마스크 True = 가림).
- 학습 손실 감소는 확인했다. 일반화(검증 손실)는 데이터가 작아 step 100 이후 나빠진다.

## 개선점 / 다음에 해 볼 것
- README 과제 2: head 수별 검증 손실과 시간을 비교한다.
- README 과제 3: Post-LN과 Pre-LN을 비교한다(층 수 6).
- README 과제 4: dropout, 조기 종료, 더 큰 공개 말뭉치로 과적합을 완화한다 → Part 2와 연결된다.
- 사용자 PC(RTX 3070)에서 같은 명령으로 실행해 시간·메모리를 위 표에 추가한다.

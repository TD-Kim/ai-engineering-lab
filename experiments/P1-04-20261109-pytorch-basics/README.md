# P1-04 PyTorch 기초 — 실행 기록

- 실습 ID: P1-04
- 날짜: 2026-11-09
- 실행 위치: 클라우드 작업 환경(CPU). GPU 수치는 **사용자 PC에서 측정 필요**

## 목적
`solution.py`가 모든 테스트를 통과하는지, `MyLinear`가 `nn.Linear`와 같은 학습 결과를 내는지, 직접 짠 경사하강법이 `y = 2x − 1`을 복원하는지 확인한다.

## 데이터셋
- GD: `x = linspace(-1, 1, 50)`, `y = 2x − 1` (잡음 없음)
- DataLoader: `ToyRegressionDataset(n=512, in=3, out=2, seed=1)`, 잡음 표준편차 0.01 (코드로 생성, 다운로드 없음)

## 모델
- 스칼라 `w, b` (GD)
- `MyLinear(3, 2)` vs `nn.Linear(3, 2)` (같은 초기값 복사)

## 하이퍼파라미터
| 이름 | GD | DataLoader 학습 |
|---|---|---|
| optimizer | 없음(직접 갱신) | SGD |
| lr | 0.5 | 0.1 |
| steps / epochs | 200 steps | 30 epochs |
| batch size | 전체(50) | 32 |
| seed | — | 0 |

## 환경
| 항목 | 값 |
|---|---|
| OS | Linux (클라우드 컨테이너, 4 vCPU) |
| Python | 3.12.13 |
| torch | 2.15.0+cu132 (PyPI 휠, CUDA 사용 불가 → device=cpu) |
| numpy | 2.5.4 |

## GPU 사용량
| 환경 | 최대 할당 메모리 |
|---|---|
| CPU (이 기록) | 해당 없음 |
| RTX 3070 | 사용자 PC에서 측정 필요 |

## 학습 시간
| 항목 | CPU (이 기록) | RTX 3070 |
|---|---|---|
| `run_demo.py` DataLoader 학습 (30 epoch × 2 모델) | 1.71s | 사용자 PC에서 측정 필요 |
| `pytest part1_ai_fundamentals/pytorch` | 2.72s | 사용자 PC에서 측정 필요 |

## 결과
`run_demo.py --impl solution` 출력:

```text
[device] cpu
[GD] w=2.0000 (정답 2.0), b=-1.0000 (정답 -1.0), loss 2.3878 -> 0.000000
[MyLinear]  loss 0.6794 -> 0.000106
[nn.Linear] loss 0.6794 -> 0.000106
[weight] learned=
tensor([[ 0.6610,  0.2675,  0.0621],
        [ 0.6208, -0.4519, -0.1656]])
 true=
tensor([[ 0.6614,  0.2669,  0.0617],
        [ 0.6213, -0.4519, -0.1661]])
[time] 1.71s
```

pytest: solution `10 passed in 2.72s`, starter(빈 상태) `10 skipped in 1.62s`

## 비교
- `MyLinear`와 `nn.Linear`의 epoch별 loss가 일치한다 (테스트 허용오차 `rtol=1e-4`).
- 학습된 weight와 데이터를 만든 true weight의 차이는 잡음 수준이다 (테스트 허용오차 0.02).

## 개선점 / 다음에 해 볼 것
- README 과제 1: 학습률별 loss 곡선 (이론상 lr < 1.0 이어야 수렴: b 방향 곡률 ∂²L/∂b² = 2 → 안정 조건 lr < 2/2)
- 사용자 PC(RTX 3070)에서 같은 명령으로 실행해 device=cuda 결과와 시간을 이 표에 추가

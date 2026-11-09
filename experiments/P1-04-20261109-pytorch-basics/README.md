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
ENV_INFO

## GPU 사용량
| 환경 | 최대 할당 메모리 |
|---|---|
| CPU (이 기록) | 해당 없음 |
| RTX 3070 | 사용자 PC에서 측정 필요 |

## 학습 시간
TIME_INFO

## 결과
RESULT_INFO

## 비교
- `MyLinear`와 `nn.Linear`의 epoch별 loss가 일치한다 (테스트 허용오차 `rtol=1e-4`).
- 학습된 weight와 데이터를 만든 true weight의 차이는 잡음 수준이다 (테스트 허용오차 0.02).

## 개선점 / 다음에 해 볼 것
- README 과제 1: 학습률별 loss 곡선 (이론상 lr ≥ 1.0 이면 b 방향에서 발산)
- 사용자 PC(RTX 3070)에서 같은 명령으로 실행해 device=cuda 결과와 시간을 이 표에 추가

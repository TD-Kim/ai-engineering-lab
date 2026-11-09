# experiments

> 상태: **자리표시자** — 실험이 생기면 아래 템플릿으로 기록한다.

## 디렉터리 규칙

`experiments/<실습ID>-<YYYYMMDD>-<짧은이름>/`

```text
experiments/P3-07-20270301-hybrid-compare/
├── README.md               # 아래 템플릿
├── results.csv             # 지표 표 (커밋)
├── figures/                # 그래프 (커밋)
├── requirements.lock.txt   # uv pip freeze 결과 (재현용)
└── artifacts/              # 가중치·인덱스 등 대용량 (커밋 안 함)
```

## 실험 기록 템플릿

```markdown
# <실험 이름>

- 실습 ID:
- 날짜:

## 목적
무엇을 확인하려는가? 가설은 무엇인가? (개선을 가정하지 말고 "확인"할 것)

## 데이터셋
이름·버전·분할(학습/검증/테스트)·크기. datasets/README.md 링크.

## 모델
이름·크기·라이선스·양자화 여부.

## 하이퍼파라미터
| 이름 | 값 |
|---|---|

## GPU 사용량
최대 할당 메모리(torch.cuda.max_memory_allocated), nvidia-smi 관찰값.

## 학습 시간
전체 / 에폭당 / 질의당 Latency(p50, p95).

## 결과
| 방식 | 지표1 | 지표2 | ... |
|---|---|---|---|

## 비교
베이스라인 또는 다른 방식과의 차이. 차이가 작으면 작다고 적는다.

## 개선점
다음에 바꿔 볼 것, 실패 원인 분석.
```

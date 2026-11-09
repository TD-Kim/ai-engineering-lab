# next_token_prediction (P2-01)

> 상태: **자리표시자** — 실습 내용은 해당 주차에 작성합니다.

| 항목 | 값 |
|---|---|
| 실습 ID | P2-01 |
| 구분 | 메인 |
| 주차 | W5 |
| 소속 | [Part 2 — LLM 활용 원리](../README.md) |

## 학습 목표

- 로컬 소형 모델(Qwen2.5-0.5B, Apache-2.0)로 다음 토큰 확률(logits → softmax)을 직접 출력해 본다
- temperature·top-k·top-p가 출력 다양성에 주는 영향을 비교한다
- 컨텍스트 길이가 생성 시간에 주는 영향을 측정한다

## 서비스 관점

샘플링 설정은 답변의 일관성(재현성)과 다양성을 바꾼다. 서비스 기본값을 고르는 근거가 된다.

## 작성 예정

- `README.md` 12항목 + 서비스 관점: 목표 · 선수지식 · 핵심개념 · 내부원리 · 서비스 관점 · 수식/그림 · 실행법 · TODO 목록 · 예상 결과 · 자주 나는 오류 · 과제 · 체크리스트 · 다음 단계 연결
- `starter.py` (TODO 3~6개 + 개념/HINT/WHY/서비스 관점/REF 주석), `solution.py`, `tests/`
- 규약: [ARCHITECTURE §3–4](../../docs/ARCHITECTURE.md#3-실습-파일-규약-startersolutiontest)

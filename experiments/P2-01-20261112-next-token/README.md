# P2-01 다음 토큰 예측과 샘플링 (메인) — 실행 기록

- 실습 ID: P2-01 (메인)
- 날짜: 2026-11-12
- 실행 위치: 클라우드 작업 환경(CPU). transformers·모델 다운로드 불가 → `run_qwen.py`는 **사용자 PC에서 실행 필요**

## 목적
(1) temperature·top-k·top-p 손계산 값이 코드와 같은지, (2) greedy와 sampling의 차이(되풀이, 재현성), (3) 스무딩(모델 확신도)이 sampling 품질에 주는 영향, (4) 컨텍스트 창 밖 토큰이 잘리는지 확인한다.

## 데이터셋
- `part1_ai_fundamentals/data/corpus_ko_en.txt` 48줄 → 소문자·구두점 제거·공백 분리, 어휘 345 + `<eos>` = 346

## 모델
- `BigramLM` (직전 단어 → 다음 단어 횟수, logits = log(count + alpha)), alpha = 0.001 (비교 0.1)
- Qwen/Qwen2.5-0.5B: 이 환경에서 미실행

## 하이퍼파라미터
| 이름 | 값 |
|---|---|
| temperature | 0(greedy), 0.5, 1.0, 2.0 |
| top-k / top-p | 2 / 0.7, 0.9 |
| max_new_tokens | 12 |
| context_window | 4 |
| seed | 0, 1, 2 |

## 환경
| 항목 | 값 |
|---|---|
| Python | 3.12.13 |
| torch | 2.15.0+cu132 (CPU) |
| transformers | 미설치 |

## GPU 사용량
해당 없음(CPU, 1초 이내). Qwen fp16 추론의 VRAM은 사용자 PC(RTX 3070)에서 측정 필요.

## 토큰 사용량·비용
API 호출 없음 → $0.

## 결과
명령: `python part2_llm/next_token_prediction/run_demo.py --impl solution` (실제 출력, 시드 고정으로 실행마다 같음)

```
[1] temperature: 후보 [mat, log, sun] 의 점수 [2, 1, 0]
    T=0.5: mat 0.867, log 0.117, sun 0.016   엔트로피 0.64 비트
    T=1.0: mat 0.665, log 0.245, sun 0.090   엔트로피 1.20 비트
    T=2.0: mat 0.506, log 0.307, sun 0.186   엔트로피 1.47 비트
[2] top-k / top-p: 확률 [0.50, 0.30, 0.15, 0.05] 의 후보 a, b, c, d
    top-k k=2  : a 0.625, b 0.375, c 0.000, d 0.000
    top-p p=0.7: a 0.625, b 0.375, c 0.000, d 0.000
    top-p p=0.9: a 0.526, b 0.316, c 0.158, d 0.000
    1 등이 0.95 일 때 top-p p=0.9: a 1.000, b 0.000, c 0.000, d 0.000  (후보 1 개)
    1 등이 0.95 일 때 top-k k=2  : a 0.969, b 0.031, c 0.000, d 0.000  (후보 2 개 고정)
[3] bigram LM (어휘 346개, alpha=0.001): 'the cat' 다음 단어 확률 상위 3
    likes 0.427, sat 0.427, alone 0.000
[4] 생성 12 토큰, 컨텍스트 창 4: 'the cat ...'
    greedy (T=0)        : likes the same data and the same data and the same data
    T=1, top-p 0.9, s=0: sat on weekends i stay home and read some code <eos>
    T=1, top-p 0.9, s=1: likes the slower the learning rate the loss goes down check the
    T=1, top-p 0.9, s=2: sat on rainy days i drink coffee every morning and the model
    같은 시드 0 다시     : sat on weekends i stay home and read some code <eos>  (재현됨)
    (비교) alpha=0.1, s=0  : home different 논문을 원리가 sat scores similar context cannot does 모일 안정시킨다
           'the cat' 다음 엔트로피 alpha=0.001 2.69 비트 vs alpha=0.1 8.31 비트
[5] 컨텍스트 창: 모델이 실제로 받은 입력 길이 (프롬프트 6 토큰, 창 4, 3 토큰 생성)
    단계별 입력 길이 = [4, 4, 4]  (창보다 긴 앞부분은 잘려서 모델이 보지 못함)
```

선택 스크립트:
| 스크립트 | 이 환경 결과 |
|---|---|
| `run_qwen.py` | `건너뜀: transformers 가 없습니다(pip install -r requirements/part2.txt).` — 사용자 PC에서 실행 필요 |

## 비교
- temperature 0.5 → 2.0: 엔트로피 0.64 → 1.47비트, 1등은 그대로.
- greedy: `and the same data` 고리 되풀이. sampling(T=1, top-p 0.9): 시드마다 다른 문장, 같은 시드는 재현.
- alpha 0.001 → 0.1: `the cat` 다음 엔트로피 2.69 → 8.31비트, 생성 문장이 무의미해짐.
- 컨텍스트 창 4: 6토큰 프롬프트에서도 모델 입력은 매 단계 4토큰.

## 개선점 / 다음에 확인할 것
- 사용자 PC에서 `run_qwen.py`: 상위 5 토큰·T별 엔트로피·greedy/sampling 문장·프롬프트 64/256/1024토큰 시간(GPU는 `torch.cuda.synchronize()` 포함) 기록(예상치를 미리 쓰지 않음).
- README 과제 1·2(temperature별 비교, 되풀이 억제) 결과.

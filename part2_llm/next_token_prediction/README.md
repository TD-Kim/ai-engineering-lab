# P2-01 다음 토큰 예측과 샘플링 — LLM은 "다음 토큰 점수"를 내는 함수

| 항목 | 값 |
|---|---|
| 실습 ID | P2-01 (메인) |
| 주차 · 진행 순서 | W5 · Part 2 첫 실습 |
| 예상 시간 | 2~3시간 |
| 파일 | `starter.py`(직접 채우기), `solution.py`(정답), `tests/test_p2_01_next_token.py`, `run_demo.py`, 선택: `run_qwen.py`(모델 다운로드 필요) |
| 소속 | [Part 2 — LLM 활용 원리](../README.md) |

> 방향: **개념은 깊게, 구현은 얕게.** 다음 토큰 예측·temperature·top-k·top-p·greedy·컨텍스트 창을 작은 숫자와 작은 모델로 확인합니다. 실제 LLM(Qwen2.5-0.5B, Apache-2.0)은 같은 함수로 돌려 보는 선택 스크립트입니다.

## 1. 목표

- LLM의 출력이 "다음 토큰 점수(logits)"이고, 문장은 그것을 반복해 만든다는 것(자기회귀 생성)을 설명한다.
- temperature가 확률 분포를 뾰족하게·평평하게 바꾸는 것을 손계산으로 확인한다.
- top-k와 top-p(nucleus)의 차이를 예시로 설명한다.
- greedy와 sampling의 차이(재현성 vs 다양성, 되풀이)를 실제 생성으로 본다.
- 컨텍스트 창을 넘는 앞부분을 모델이 보지 못한다는 것을 확인한다.

## 2. 선수지식

- P1-05의 softmax·마스크(-inf → 확률 0), KV 캐시·TTFT
- P1-01의 토큰·`max_tokens`·비용
- P1-04의 `torch.tensor`, 인덱싱

## 3. 핵심 개념

| 개념 | 한 줄 설명 |
|---|---|
| logits | 어휘의 모든 토큰에 대한 "다음에 올 점수". 아직 확률이 아니다 |
| softmax | 점수를 합이 1인 확률로: e^(z_i) / Σ e^(z_j) |
| temperature (T) | 점수를 T로 나눈 뒤 softmax. T<1 뾰족, T>1 평평. 1등 순서는 그대로 |
| greedy | 매번 가장 큰 점수만 고르기. 결정적이지만 되풀이하기 쉽다 |
| sampling | 확률대로 뽑기. 다양하지만 매번 다를 수 있다(시드로 재현) |
| top-k | 상위 k개 후보만 남기기(후보 수 고정) |
| top-p (nucleus) | 누적 확률이 p에 도달할 때까지의 후보만 남기기(후보 수가 분포에 따라 변함) |
| 자기회귀 생성 | 다음 토큰을 골라 입력 끝에 붙이고 다시 예측하기를 반복 |
| 컨텍스트 창 | 모델이 한 번에 볼 수 있는 토큰 수 상한. 넘는 앞부분은 잘린다 |
| 엔트로피 | 분포가 얼마나 퍼졌는지(비트). 0이면 한 후보에 확실 |

## 4. 내부 원리

### 4-1. 한 단계: 점수 → 확률 → 고르기

```
입력 "the cat"  ──모델──▶  logits (V개)  ──÷T──▶  [top-k] ──▶ [top-p] ──▶ softmax ──▶ 확률 ──뽑기──▶ 다음 토큰 "sat"
                                                                                   (T=0 이면 argmax = greedy)
```

**적용 순서가 결과를 바꾼다.** 이 실습은 ① ÷T → ② top-k → ③ top-p 순서입니다(테스트 `test_sample_next_filter_order_temperature_topk_topp`).
- 확률 `[0.50, 0.30, 0.15, 0.05]`, T=2, top-p 0.7: T를 먼저 적용해 평평해진 뒤 자르면 후보 **3개**, top-p를 먼저 하면 **2개**.
- 확률 `[0.4, 0.3, 0.2, 0.1]`, top-k 2, top-p 0.5: top-k 뒤 `[0.571, 0.429]`에서 자르면 후보 **1개**, top-p를 먼저 하면 **2개**.

이 순서는 흔히 쓰이는 방식 중 하나일 뿐이고, **라이브러리·서빙 엔진마다 순서와 세부 처리가 다를 수 있습니다.** 같은 설정값이라도 엔진을 바꾸면 결과가 달라질 수 있으니 문서를 확인하세요.

### 4-2. temperature 손계산

후보 `[mat, log, sun]`의 점수 `[2, 1, 0]`:

| T | 점수 ÷ T | e^(·) | 확률 |
|---|---|---|---|
| 0.5 | [4, 2, 0] | [54.60, 7.39, 1.00] (합 62.99) | **[0.867, 0.117, 0.016]** |
| 1 | [2, 1, 0] | [7.39, 2.72, 1.00] (합 11.11) | **[0.665, 0.245, 0.090]** |
| 2 | [1, 0.5, 0] | [2.72, 1.65, 1.00] (합 5.37) | **[0.506, 0.307, 0.186]** |

T가 작으면 1등에 몰리고(엔트로피 0.64비트), 크면 고르게 퍼집니다(1.47비트). 1등이 `mat`인 것은 바뀌지 않습니다.

### 4-3. top-k vs top-p

확률 `[0.50, 0.30, 0.15, 0.05]` (이미 큰 순서):

- **top-k, k=2**: 앞 2개 → 다시 정규화하면 `[0.625, 0.375, 0, 0]`
- **top-p, p=0.7**: "더하기 전 누적" `[0, 0.50, 0.80, 0.95]` 중 0.7보다 작은 것은 앞 2개 → `[0.625, 0.375, 0, 0]`
- **top-p, p=0.9**: 앞 3개(0.80 < 0.9) → `[0.526, 0.316, 0.158, 0]`

1등이 0.95로 확실할 때, top-p(p=0.9)는 후보를 **1개**만 남깁니다. top-k(k=2)는 여전히 2개를 남겨 0.03짜리도 가끔 뽑힙니다. 분포 모양에 맞춰 후보 수가 바뀌는 것이 top-p의 장점입니다(Holtzman et al., 2020, §3.1).

### 4-4. 이 실습의 모델: bigram LM

`BigramLM`은 "직전 단어 → 다음 단어"를 세어 logits = log(횟수 + alpha)로 둡니다. 직전 단어 **하나**만 봅니다. 실제 LLM은 컨텍스트 창 안의 모든 토큰을 어텐션(P1-05)으로 보고 점수를 냅니다. 하지만 인터페이스는 같습니다. **모델 = 토큰 목록 → 다음 토큰 점수 함수**. 그래서 같은 `generate`가 Qwen에서도 그대로 동작합니다(`run_qwen.py`).

## 서비스 관점

| 질문 | 판단 |
|---|---|
| 서비스 기본값은? | 분류·추출·코드처럼 **정답이 하나**면 낮은 다양성(일관성). 아이디어·문장 다듬기처럼 **여러 답이 괜찮으면** 높은 다양성 |
| 같은 질문에 답이 매번 달라도 되나? | 고객 응대·평가 자동화·회귀 테스트는 재현성이 중요하다. 무작위성이 있으면 "고쳤더니 좋아졌다"를 한 번의 실행으로 판단할 수 없다 → 여러 번 실행해 비율로 본다 |
| Claude API에서도 temperature를 조절하나? | **현재 API 버전(`2026-10-21`)에서는 `temperature`·`top_p`·`top_k`를 보내면 모든 모델에서 400 오류**다(작성 시점 2026-11, claude-api 스킬 문서 기준). 샘플링은 모델 쪽이 관리하고, 출력의 일관성·형식은 **프롬프트와 구조화 출력**으로 조절한다(P2-02). 이 실습의 원리는 로컬·오픈 모델을 직접 서빙할 때 그대로 쓰인다 |
| greedy면 안전한가? | 결정적이지만 같은 구절을 되풀이하기 쉽다(§8 [4]). 실무 오픈 모델 서빙은 보통 낮은 temperature + top-p 또는 반복 억제 설정을 함께 쓴다 |
| 출력 길이 상한은 왜 꼭 두나? | 멈춤 토큰이 안 나와도 반드시 끝나게(무한 생성 방지). 출력 토큰 = 비용·지연이므로 `max_tokens`는 **비용 상한**이다 |
| 긴 대화에서 처음 지시를 잊는 이유는? | 컨텍스트 창을 넘는 앞부분이 잘리면 모델은 그것을 볼 수 없다(§8 [5]). 서비스는 요약·중요 지시 재배치·RAG로 "무엇을 창 안에 넣을지"를 관리한다 |
| 프롬프트 vs RAG vs Fine-tuning은? | 이 실습은 "모델이 무엇을 하는가"의 기초다. 세 가지 선택 기준은 P2-04에서 비용·지연·품질로 정리한다 |

## 5. 수식

- softmax: p_i = e^(z_i / T) / Σ_j e^(z_j / T)
- top-k: z_i ← −∞ (z_i < k번째로 큰 점수)
- top-p: 확률을 내림차순 p_(1) ≥ p_(2) ≥ …로 정렬해, Σ_{j<i} p_(j) < p 인 후보 i만 남김
- 엔트로피: H = −Σ p_i log₂ p_i (비트)
- bigram LM: z_{다음} = log(count(직전 → 다음) + α)
- 생성: x_{t+1} ~ p(· | x_{t−W+1..t}) (W = 컨텍스트 창), 최대 `max_new_tokens`번

## 6. 실행법

저장소 루트에서 실행합니다.

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"

# 1) 남은 TODO 확인
python part2_llm/next_token_prediction/starter.py

# 2) 직접 구현 검사 (TODO 가 남은 테스트는 skip)
$env:IMPL = "starter"; pytest part2_llm/next_token_prediction/tests; Remove-Item Env:IMPL

# 3) 정답 코드 검사
pytest part2_llm/next_token_prediction/tests

# 4) 데모 (bigram LM, 다운로드·API 키 불필요)
python part2_llm/next_token_prediction/run_demo.py --impl starter

# 5) (선택) 실제 LLM — transformers 설치·모델 다운로드 필요, RTX 3070 에서 fp16
pip install -r requirements/part2.txt
python part2_llm/next_token_prediction/run_qwen.py
```

**bash (Linux/macOS)**

```bash
source .venv/bin/activate
python part2_llm/next_token_prediction/starter.py
IMPL=starter pytest part2_llm/next_token_prediction/tests
pytest part2_llm/next_token_prediction/tests
python part2_llm/next_token_prediction/run_demo.py --impl solution
python part2_llm/next_token_prediction/run_qwen.py     # 선택
```

- 테스트는 다운로드·API 호출이 없습니다. 난수는 `torch.Generator` 시드로 고정해 결정적입니다. 생성은 `max_new_tokens`로 반드시 끝납니다.
- `part2_llm/conftest.py`가 Part 1과 같은 규약(`IMPL`, TODO 미구현 → skip)을 제공합니다.

## 7. TODO 목록

| TODO | 함수 | 내용 |
|---|---|---|
| 1 | `softmax_with_temperature` | `softmax(logits / T)`, T ≤ 0이면 `ValueError` |
| 2 | `top_k_filter` | k번째 점수보다 작은 칸을 −∞(동점 남김), k ≥ V면 그대로 |
| 3 | `top_p_filter` | 정렬 → "더하기 전 누적 < p"만 남김 → 원래 위치로(scatter) |
| 4 | `sample_next` | T=0이면 argmax, 아니면 ÷T → top-k → top-p → softmax → `multinomial` |
| 5 | `generate` | `for` 최대 `max_new_tokens`번, 마지막 `context_window`개만 모델에, eos에서 멈춤 |

제공 코드: `BigramLM`(작은 언어 모델), `load_bigram_lm`(Part 1 말뭉치로 생성), `entropy_bits`(엔트로피).

## 8. 예상 결과

아래는 클라우드 작업 환경(CPU, Python 3.12.13, torch 2.15.0)에서 `run_demo.py --impl solution`을 **실제로 실행한 출력**입니다(bigram LM). `run_qwen.py`는 이 환경에 transformers와 모델이 없어 `건너뜀: transformers 가 없습니다`로 끝났습니다. Qwen 결과는 **사용자 PC에서 실행 필요**하며, 실행한 값만 기록하세요.

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

**해석**
- [2] 같은 p=0.9라도 분포가 뾰족하면 후보 1개, 퍼져 있으면 3개입니다. top-k는 상황과 관계없이 2개입니다.
- [3] `cat` 다음은 `likes`와 `sat`이 0.427로 동점입니다(말뭉치에 각 1번). greedy의 argmax는 동점이면 앞 번호(사전순으로 앞선 `likes`)를 고릅니다.
- [4] **greedy는 "the same data and the same data…"를 되풀이합니다.** bigram LM은 직전 단어만 보므로 `and → the → same → data → and`의 고리에 갇힙니다. 큰 LLM도 greedy에서 되풀이가 생기기 쉽다는 것이 nucleus sampling 논문의 동기입니다(Holtzman et al., 2020). sampling은 시드마다 다른 문장을 만들고, 같은 시드는 같은 문장을 재현합니다.
- [4] (비교) 스무딩 `alpha`를 0.1로 키우면 말뭉치에서 한 번도 안 나온 수백 개의 조합에 확률이 퍼져(엔트로피 2.69 → 8.31비트) 엉뚱한 단어가 뽑힙니다. "모델이 확신이 없으면 sampling은 잡음을 뽑는다"는 예입니다.
- [5] 프롬프트가 6토큰이어도 모델은 마지막 4토큰만 받았습니다.

## 9. 자주 나는 오류

| 증상 | 원인 | 해결 |
|---|---|---|
| `ZeroDivisionError`/`inf` | temperature 0으로 나눔 | T=0은 greedy로 따로 처리 |
| softmax 결과에 `nan` | 직접 `exp`를 써서 넘침 | `torch.softmax`(내부에서 최댓값을 뺌) |
| top-p 뒤 후보가 0개 | 누적에 자기 자신을 포함해 판정 | "더하기 전 누적 < p"로 판정 → 1등은 항상 남음 |
| top-p가 엉뚱한 토큰을 남김 | 정렬된 순서의 결과를 원래 위치로 안 되돌림 | `scatter(-1, order, keep)` |
| 같은 시드인데 결과가 다름 | `generator`를 안 넘김(전역 난수 사용) | `torch.Generator().manual_seed(s)`를 넘김 |
| 생성이 끝나지 않음 | `while`로 eos만 기다림 | `for _ in range(max_new_tokens)` |
| `KeyError: 어휘에 없는 단어(OOV)` | bigram LM 어휘에 없는 단어 | 말뭉치에 있는 단어로 프롬프트 작성(실제 LLM은 서브워드라 OOV가 없다 — P1-01) |

## 10. 과제

1. [4]의 sampling을 T=0.3 / 1.0 / 1.5로 바꿔 각각 시드 5개의 문장을 비교하세요. 되풀이·엉뚱함이 어떻게 변하나요?
2. greedy의 되풀이를 줄이는 방법을 하나 생각해 구현해 보세요(예: 이미 나온 토큰의 점수를 조금 낮추기). 무엇을 잃나요?
3. (모델을 받을 수 있으면) `run_qwen.py`로 `The capital of France is` 다음 토큰 상위 5와 T별 엔트로피를 기록하세요. bigram LM과 엔트로피를 비교하고, 프롬프트 64/256/1024토큰의 시간을 기록하세요.
4. 고객 문의를 "환불/배송/기타"로 분류하는 기능과 마케팅 문구 5개를 제안하는 기능에 각각 어떤 다양성 설정이 맞는지, 그리고 Claude API에서는 그것을 무엇으로 조절할지 적어 보세요.

## 11. 체크리스트

- [ ] `IMPL=starter pytest part2_llm/next_token_prediction/tests` 가 모두 통과한다
- [ ] 점수 [2, 1, 0]의 T=1 확률을 손으로 계산할 수 있다
- [ ] top-k와 top-p의 차이를 "후보 수"로 설명할 수 있다
- [ ] greedy의 장단점(재현성 vs 되풀이)을 안다
- [ ] 생성 루프에 최대 길이가 반드시 필요한 이유를 비용과 함께 설명할 수 있다
- [ ] 컨텍스트 창을 넘는 내용이 잘린다는 것을 안다

## 12. 다음 단계 연결

- **P2-02 Claude API 기초**: 토큰 세기, 메시지·시스템 프롬프트 구조, 구조화 출력, 스트리밍(첫 토큰 지연을 체감).
- **P2-03 비용·지연**: 프롬프트 캐싱(P1-05 KV 캐시의 요청 간 재사용), 배치, 토큰 사용량 기록.
- **P2-04**: 프롬프트 vs RAG vs Fine-tuning 선택 기준.
- 손실·역전파·학습 루프를 직접 구현하고 싶다면 선택 과제 P2-S1.

## REF

- Holtzman, Buys, Du, Forbes, Choi, 2020, "The Curious Case of Neural Text Degeneration" (ICLR 2020), §3.1 Nucleus Sampling — greedy·beam의 되풀이 문제와 top-p. (작성 환경에서 원문 접속 불가 → 절 번호 미확인)
- Claude API 샘플링 파라미터 제거(API 버전 `2026-10-21`): claude-api 스킬 `shared/api-version-migration.md` "Sampling parameters are removed" (작성 시점 2026-11)
- Qwen2.5-0.5B 모델 카드·라이선스(Apache-2.0): https://huggingface.co/Qwen/Qwen2.5-0.5B (작성 환경 접속 불가 → 미확인)
- PyTorch `torch.multinomial`, `torch.topk`, `torch.sort`, `Tensor.scatter` — 설치된 torch 2.15.0에서 실행으로 확인

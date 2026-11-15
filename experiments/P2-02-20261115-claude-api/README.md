# P2-02 Claude API 기초 (메인) — 실행 기록

- 실습 ID: P2-02 (메인)
- 날짜: 2026-11-15
- 실행 위치: 클라우드 작업 환경(API 키 없음). 아래 "결과(데모)"는 **가짜 SDK의 미리 정한 응답·가짜 시계**이며 Claude 실제 결과가 아니다. **실제 호출 표는 사용자 PC에서 채울 칸**이다.

## 목적
(1) 요청 본문이 현재 SDK 규칙(effort 명시, 샘플링·prefill·강제 tool_choice 없음, 서버 측 폴백)을 지키는지, (2) 응답 검사 순서(stop_reason → text 블록 → 검증 → 폴백 여부)가 동작하는지, (3) 구조화 출력 triple 추출, (4) 실제 TTFT·usage·비용(사용자 PC)을 확인한다.

## 데이터셋
- 추출 문장: "세종대왕은 1443년에 훈민정음을 창제했고, 집현전 학자들이 이를 도왔다."
- 스트리밍 질문: "훈민정음을 두 문장으로 소개해 주세요."

## 모델
- claude-opus-5-5 (LLM_MODEL 로 변경 가능), effort medium, `fallbacks: "default"` (베타 server-side-fallback-2026-07-01)

## 하이퍼파라미터
| 이름 | 데모 | call_claude.py |
|---|---|---|
| max_tokens | 16000 | 구조화 4000, 스트리밍 2000 |
| effort | medium | medium |
| timeout / max_retries | — | 60초 / 2 |

## 환경
| 항목 | 작성 환경 | 사용자 PC |
|---|---|---|
| Python | 3.12.13 | |
| anthropic | 2.6.0 | (측정 시 기록) |
| API 키 | 없음 | 필요(유료) |

## GPU 사용량
해당 없음(API 호출).

## 토큰 사용량·비용 — 사용자 PC 측정 필요

`python part2_llm/llm_api/basics/call_claude.py` 출력(그리고 outputs/usage_real.csv)을 옮겨 적는다.

| 호출 | count_tokens 추정 입력 | 실제 input_tokens | 실제 output_tokens | stop_reason | fell_back | 지연(ms) | TTFT(ms, thinking 포함) | 비용(USD, 단가 기준일 명시) |
|---|---|---|---|---|---|---|---|---|
| 구조화 출력(triple) | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | — | 측정 필요 |
| 스트리밍 | — | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 |

추출된 triple(실제): 측정 필요

effort 비교(README 과제 2, 선택):

| effort | TTFT(ms) | output_tokens | 답의 질(메모) |
|---|---|---|---|
| low | 측정 필요 | 측정 필요 | |
| medium | 측정 필요 | 측정 필요 | |
| high | 측정 필요 | 측정 필요 | |

## 결과(데모)
명령: `python part2_llm/llm_api/basics/run_demo.py --impl solution` (작성 환경 실제 출력, 가짜 SDK)

```
[1] 요청 본문 (TODO 1) — 멀티턴, system, effort 명시, 서버 측 폴백
    키: ['betas', 'fallbacks', 'max_tokens', 'messages', 'model', 'output_config', 'system']
    model=claude-opus-5-5 max_tokens=16000 output_config={'effort': 'medium'} fallbacks='default' betas=['server-side-fallback-2026-07-01']
    마지막 메시지 역할: user (prefill 없음)
[2] 구조화 출력 (TODO 3) — 가짜 SDK 의 미리 정한 응답
    스키마 required: ['triples'] additionalProperties: False
    (세종대왕, 창제했다, 훈민정음)
    (집현전 학자들, 도왔다, 훈민정음 창제)
[3] 스트리밍 (TODO 4) — 가짜 스트림·가짜 시계(초)
    text='훈민정음은 백성을 가르치는 바른 소리입니다.'  TTFT 1200 ms (thinking 포함)  전체 1250 ms  stop_reason=end_turn
[4] 응답 검사 (TODO 2) — 거절·잘림·폴백
    거절: RefusalError — refused (category=None)
    잘림: TruncatedError — response truncated at max_tokens
    폴백: fell_back=True, served_by=claude-opus-5, 건너뛴 블록 1개
[5] 호출 전 비용 추정 (TODO 5) — 가짜 count_tokens(입력 1,200 토큰으로 고정)
    입력 1200 토큰, 예상(출력 300) $0.0108, 최대(출력 16000) $0.3248  (claude-opus-5-5 $4/$20, 작성 시점 2026-10)
[6] (선택) 오류 분류 (TODO 6)
    RateLimitError       → ErrorDecision(kind='rate_limit', retry=True, retry_after_s=7.0)
    OverloadedError      → ErrorDecision(kind='server', retry=True, retry_after_s=None)
    BadRequestError      → ErrorDecision(kind='client', retry=False, retry_after_s=None)
    APITimeoutError      → ErrorDecision(kind='connection', retry=True, retry_after_s=None)
[7] 사용량 기록 — P2-01B log_usage 재사용: part2_llm/llm_api/basics/outputs/usage.csv
    claude,claude-opus-5-5,310,95,1250.0,1200.0,end_turn
```

`call_claude.py` 작성 환경 실행: `건너뜀: ANTHROPIC_API_KEY 가 없습니다(.env 에 넣고 다시 실행).`

## 비교
- 사용자 PC 측정 후 작성: count_tokens 추정 vs 실제 input_tokens, effort별 TTFT, P2-01B Ollama 와의 비교(README §4-4 공정 비교 조건 명시).

## 개선점 / 다음에 확인할 것
- 같은 시스템 프롬프트를 반복할 때 프롬프트 캐싱으로 입력 비용·TTFT가 얼마나 줄어드는지(P2-03).

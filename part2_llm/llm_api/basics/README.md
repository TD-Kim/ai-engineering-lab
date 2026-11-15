# P2-02 Claude API 기초 — 요청·응답을 "서비스 코드"로 다루기

| 항목 | 값 |
|---|---|
| 실습 ID | P2-02 (메인) |
| 주차 · 진행 순서 | W5 · P2-01B 다음 |
| 예상 시간 | 3~4시간 |
| 파일 | `starter.py`(직접 채우기), `solution.py`(정답), `tests/test_p2_02_claude_api.py`, `run_demo.py`, 선택: `call_claude.py`(API 키 필요, 유료) |
| 소속 | [Part 2 — LLM 활용 원리](../../README.md) · [llm_api](../README.md) |
| 재사용 | [P2-01B](../../local_llm_serving/README.md)의 `Usage`·`ProviderError`·`log_usage`를 그대로 불러 씁니다(복사하지 않음) |

> 방향: **개념은 깊게, 구현은 얕게.** 요청 본문(system·멀티턴), 응답 검사(stop_reason·블록·폴백), 구조화 출력, 스트리밍, 토큰 세기, 오류 분류를 "잘못 쓰면 무엇이 깨지는가"와 함께 익힙니다. 테스트와 데모는 가짜 SDK로 돌아가고, 실제 호출은 선택 스크립트입니다.

## 1. 목표

- `system`과 `messages`의 역할을 구분하고, 멀티턴 대화를 매 요청에 다시 보내는 이유를 설명한다.
- 응답을 읽을 때 **stop_reason을 먼저** 보고, **text 블록만** 모으고, 서버 측 폴백 여부를 판단한다.
- 구조화 출력(JSON 스키마)으로 문장에서 (주어, 관계, 목적어) triple을 받아 Pydantic 모델로 검증한다(Part 4 연결).
- 스트리밍을 소비해 TTFT와 최종 usage·stop_reason을 얻는다.
- `count_tokens`로 호출 전에 예상·최대 비용을 계산한다(P1-01 연결).
- (선택) 오류를 종류별로 나눠 재시도할지 즉시 실패할지 정한다.

## 2. 선수지식

- P1-01: 토큰·단가·`max_tokens`, P1-05: TTFT·프롬프트 캐싱의 접두어 원리
- P2-01B: 공통 인터페이스 `generate -> (text, Usage)`, `ProviderError`, 폴백
- Python `with` 문, 예외 계층(`isinstance`), Pydantic `BaseModel`

## 3. 핵심 개념

| 개념 | 한 줄 설명 |
|---|---|
| `system` | 역할·규칙·출력 형식. 모든 턴에 공통인 지시 |
| `messages` | 대화 내용. user로 시작해 user/assistant가 번갈고, **마지막은 user** |
| stateless | 서버는 대화를 기억하지 않는다. 매 요청에 대화 전체를 보낸다 |
| content 블록 | 응답은 블록 목록(thinking·text·fallback 표시…). 사용자에게 보일 답은 text 블록 |
| `stop_reason` | 왜 멈췄나: `end_turn`·`max_tokens`·`refusal`·`stop_sequence`·`tool_use`·`pause_turn` |
| effort | thinking 깊이(`low`~`max`). 이 모델은 thinking을 끌 수 없다 |
| 구조화 출력 | `output_config.format`의 JSON 스키마대로 답하게 하는 기능 |
| 서버 측 폴백 | 정책 거절 시 같은 호출 안에서 다른 모델로 재실행(`fallbacks: "default"`) |
| `count_tokens` | 모델을 실행하지 않고 입력 토큰 수만 세는 엔드포인트 |

## 4. 내부 원리

### 4-1. 근거 — claude-api 스킬로 확인한 현재 SDK 형태 (작성 시점 2026-11)

| 항목 | 이 실습의 선택 | 근거 |
|---|---|---|
| 모델 ID | `claude-opus-5-5`(날짜 접미사 없음), `LLM_MODEL`로 변경 | 스킬 `shared/models.md` |
| thinking | 끌 수 없음(생략 = adaptive). `thinking: {"type": "disabled"}`·budget은 400 | 스킬 `python/claude-api/README.md` Extended Thinking |
| effort | `output_config={"effort": "medium"}` **명시**(이 모델 기본 medium) | 같은 절. 기본값이 모델마다 달라 명시하면 모델 교체 시 동작이 덜 바뀐다 |
| 금지 파라미터 | `temperature`·`top_p`·`top_k`·`budget_tokens`·assistant prefill·강제 `tool_choice`(`any`/`tool`) 사용 안 함 | 스킬 `shared/error-codes.md`, `shared/api-version-migration.md` (모두 400). 테스트가 요청 키를 검사 |
| 구조화 출력 | `output_config={"format": {"type": "json_schema", "schema": anthropic.transform_schema(...)}}` | 스킬 `shared/model-migration.md` "Prefill replacement", 설치된 SDK가 `output_format`을 deprecated로 안내 |
| 스트리밍 | `client.beta.messages.stream(...)` → `text_stream` → `get_final_message()` | 스킬 `python/claude-api/streaming.md` |
| 거절 | **HTTP 200 + `stop_reason: "refusal"`**, `stop_details.category`는 참고용(없을 수 있음) | 스킬 `shared/model-migration.md` refusal 절 |
| 서버 측 폴백 | `fallbacks="default"` + `betas=["server-side-fallback-2026-07-01"]`, **기본으로 켬** | §4-3 |
| `max_tokens` | 기본 16000(thinking 포함 출력 상한). 구조화 추출은 `call_claude.py`에서 4000 | 스킬 예시가 16000 사용, "max_tokens는 thinking + 답변 전체의 상한" |
| 토큰 세기 | `client.messages.count_tokens(model, messages, system, output_config)` | 스킬 `shared/token-counting.md` (tiktoken 등은 Claude 토큰을 잘못 셈) |
| 오류·재시도 | SDK가 연결 오류·408·409·429·5xx를 `max_retries`(기본 2)만큼 자동 재시도. 예외 계층은 설치된 anthropic 2.6.0에서 확인 | 스킬 README Retries·Error Handling |

### 4-2. 응답을 읽는 순서

```
HTTP 200 응답
   │
   ├─ ① stop_reason 확인 ──── refusal ───▶ RefusalError (category 는 참고만)
   │                     └── max_tokens ─▶ TruncatedError (잘린 텍스트 보관, 정상 답처럼 쓰지 않음)
   ├─ ② content 에서 type == "text" 블록만 이어 붙이기 (thinking·fallback 블록 건너뜀)
   ├─ ③ (구조화 출력) JSON → Pydantic 검증 ── 실패 ─▶ StructuredOutputError
   └─ ④ usage.iterations 에 fallback_message 가 있고 refusal 이 아니면 → 폴백 모델이 답함(served_by = message.model)
```

SDK의 `messages.parse(output_format=Model)`는 받자마자 텍스트를 검증합니다(설치된 SDK의 `lib/_parse` 확인). 그래서 거절이나 잘림으로 텍스트가 비었거나 JSON이 끊겼으면 **진짜 원인을 보기 전에** 검증 오류가 납니다. 이 실습은 `output_config.format`으로 요청하고, ①→③ 순서로 직접 검사합니다.

### 4-3. 서버 측 폴백을 기본으로 켠 이유

- **무엇인가:** 안전 분류기가 요청을 거절하면, API가 같은 호출 안에서 Anthropic이 권장하는 다른 모델로 다시 실행합니다(`"default"`는 거절 분류별로 대상 모델을 고른다).
- **왜 기본으로 켜나:** 스킬 지침이 claude-opus-5-5에 대해 "Ship the opt-in from day one"(처음부터 켠다)입니다. 폴백이 없으면 거절된 요청은 그냥 멈춥니다. 무해한 요청도 분류기에 걸릴 수 있습니다.
- **무엇은 하지 않나:** **정책 거절에만** 작동합니다. rate limit·과부하·서버 오류는 폴백하지 않고 그대로 돌려줍니다. 그건 TODO 6의 분류와 P2-01B의 공급자 폴백이 맡습니다.
- **헤더 주의:** `"default"` 형식은 `server-side-fallback-2026-07-01`, 배열 형식(`[{"model": ...}]`)은 `server-side-fallback-2026-06-01`을 씁니다. 바꿔 쓰면 400입니다.
- **사용할 수 없는 곳:** Batches API에서는 거부되고, Amazon Bedrock·Vertex AI·Microsoft Foundry에서는 쓸 수 없습니다(SDK 미들웨어가 대안). 그런 플랫폼에서는 `build_request(..., fallbacks=False)`로 끕니다.
- **응답 읽기:** 폴백이 일어났는지는 `usage.iterations`의 `fallback_message`로 판단합니다. sticky routing으로 이어진 턴에는 `fallback` 블록이 없기 때문입니다. 답한 모델은 `message.model`입니다. 폴백 모델은 원래 모델의 thinking 블록을 읽지 못합니다.
- **과금:** 시도별 사용량은 `usage.iterations`에 있습니다. 출력이 나오기 전에 거절된 경우의 과금은 공식 문서 "How refusals are billed"를 확인하세요(https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback). 스트림 중간 거절은 정상 요금이고, 이미 받은 부분은 버립니다.

### 4-4. Claude의 TTFT에는 thinking이 들어간다 — 로컬 모델과 공정하게 비교하기

`text_stream`의 첫 텍스트 조각은 thinking이 끝난 **뒤**에 옵니다. 그래서 이 실습의 TTFT는 네트워크 + 입력 처리(prefill) + thinking 시간이고, effort가 높을수록 길어집니다. thinking 토큰도 `usage.output_tokens`로 과금됩니다(응답에 있으면 `usage.output_tokens_details.thinking_tokens`가 따로 알려 줌). P2-01B의 Ollama TTFT(thinking 없는 소형 모델)와 숫자만 나란히 놓으면 공정하지 않습니다.

공정 비교 방법:
1. **같은 지표 이름을 쓴다**: 양쪽 모두 "요청 → 첫 **보이는 텍스트**"로 재고, effort(Claude)·모델 크기·양자화(로컬)를 함께 기록한다.
2. **같은 조건**: 같은 프롬프트·같은 출력 상한, 둘 다 스트리밍, 첫 요청(모델 적재·연결 준비)은 버리고 N회 중앙값.
3. **끝까지 본다**: TTFT만이 아니라 같은 작업을 끝내는 전체 시간과 **품질**(정답률 등)을 함께 비교한다. 빨라도 틀리면 서비스에서 쓸 수 없다.
4. **effort를 바꿔 본다**: Claude는 `low`/`medium`/`high`에서 TTFT·품질·출력 토큰이 어떻게 바뀌는지 따로 잰다(과제 2).

## 서비스 관점

| 질문 | 판단 |
|---|---|
| 시스템 프롬프트는 어떻게 쓰나? | 역할·규칙·금지·출력 형식처럼 **바뀌지 않는 것**만 둔다. 현재 시각·사용자 이름·검색 결과처럼 바뀌는 값은 messages 쪽에 둔다. 그래야 프롬프트 캐싱이 깨지지 않는다(P1-05·P2-03) |
| 구조화 출력은 왜 쓰나? | "JSON으로 답해"라는 부탁만으로는 가끔 설명 문장이 섞이거나 괄호가 빠져 다음 단계가 멈춘다. 스키마를 강제하면 그래프 적재(Part 4) 같은 파이프라인이 안정적이다. 그래도 거절·잘림은 생기므로 검사 순서(§4-2)를 지킨다 |
| 스트리밍은 언제? | 사람이 읽는 화면(채팅)과 긴 답·큰 `max_tokens`. 체감 대기가 줄고, 오랫동안 조용한 연결로 인한 타임아웃 위험이 준다. 배치 처리처럼 사람이 기다리지 않으면 비스트리밍도 된다 |
| 타임아웃·재시도는? | SDK 기본 타임아웃은 10분이라 서비스에 맞게 줄인다(이 실습 60초). SDK가 일시 오류를 `max_retries`만큼 재시도한 **뒤**에 오는 예외를 분류해 안내·공급자 폴백(P2-01B)·즉시 실패를 고른다(TODO 6) |
| 로그에는 무엇을? | 모델·토큰·지연·TTFT·stop_reason·폴백 여부(`Usage` + `fell_back`). **프롬프트·답 원문은 기본적으로 남기지 않는다**(개인정보·영업비밀). 필요하면 마스킹·샘플링·보존 기간을 정한다 |
| 비용 상한은? | `max_tokens`가 한 요청의 출력 비용 상한이다. 데모 [5]에서 예상 $0.0108에 비해 상한은 $0.3248(약 30배). 작업에 맞게 `max_tokens`를 줄이고, 요청 전 `count_tokens`로 큰 입력을 막는다 |
| 거절을 어떻게 다루나? | HTTP 200이라도 `stop_reason == "refusal"`이면 실패로 처리한다. 서버 측 폴백을 켜고, 그래도 거절이면 사용자에게 정중히 안내한다. category는 참고로만 기록한다 |

## 5. 수식

- 예상 비용 = (입력 토큰 × 입력 단가 + 예상 출력 토큰 × 출력 단가) / 10⁶
- 최대 비용 = (입력 토큰 × 입력 단가 + `max_tokens` × 출력 단가) / 10⁶ (출력은 thinking 포함)
- 예) 입력 2,000, 예상 출력 500, `max_tokens` 1,000, $4/$20 → 예상 $0.018, 최대 $0.028
- TTFT(Claude) = 네트워크 + prefill + thinking + 첫 텍스트 조각

단가: claude-opus-5-5 입력 $4·출력 $20 / 100만 토큰(작성 시점 2026-10, P1-01 표 / claude-api 스킬 모델표 2026-11과 일치). 사용 전 [공식 가격 페이지](https://platform.claude.com/docs/en/about-claude/pricing)에서 재확인하세요.

## 6. 실행법

저장소 루트에서 실행합니다.

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"
pip install -r requirements/part2.txt

# 1) 남은 TODO 확인
python part2_llm/llm_api/basics/starter.py

# 2) 직접 구현 검사 (TODO 가 남은 테스트는 skip)
$env:IMPL = "starter"; pytest part2_llm/llm_api/basics/tests; Remove-Item Env:IMPL

# 3) 정답 코드 검사
pytest part2_llm/llm_api/basics/tests

# 4) 데모 (가짜 SDK, 키 불필요)
python part2_llm/llm_api/basics/run_demo.py --impl starter

# 5) (선택) 실제 호출 — 유료. .env 에 ANTHROPIC_API_KEY (커밋 금지)
Copy-Item .env.example .env   # 처음 한 번
python part2_llm/llm_api/basics/call_claude.py
```

**bash (Linux/macOS)**

```bash
source .venv/bin/activate
IMPL=starter pytest part2_llm/llm_api/basics/tests
pytest part2_llm/llm_api/basics/tests
python part2_llm/llm_api/basics/run_demo.py
python part2_llm/llm_api/basics/call_claude.py      # 선택, 유료
```

- 테스트는 키·네트워크가 필요 없습니다. 응답은 가짜 Message, SDK는 가짜 객체, 시계는 가짜 시계입니다. 스레드·대기 반복이 없습니다.
- `IMPL=starter`에서도 제공 코드만 검사하는 테스트 1개(`to_usage` + P2-01B `log_usage` 연결)는 통과합니다. 나머지 13개가 TODO 미구현으로 skip입니다.
- 오류 분류 테스트는 설치된 `anthropic`·`httpx2`로 실제 예외 객체를 만들어 검사합니다(네트워크 없음).

## 7. TODO 목록

| TODO | 함수 | 내용 |
|---|---|---|
| 1 | `build_messages`, `build_request` | 역할 교대·마지막 user 검사, `model`·`max_tokens`·`messages`·`output_config.effort`·(`system`)·(`betas`+`fallbacks="default"`) |
| 2 | `read_response`, `ensure_complete` | text 블록만, `ok`·`served_by`·`fell_back`, refusal → `RefusalError`, max_tokens → `TruncatedError` |
| 3 | `with_json_schema`, `parse_structured` | `output_config.format`(transform_schema) 합치기, 거절·잘림 먼저 → Pydantic 검증 → `StructuredOutputError` |
| 4 | `consume_stream` | `with` 안에서 `text_stream` 소비·TTFT·`get_final_message()` |
| 5 | `estimate_cost` | `count_tokens`(system·output_config 포함, betas 제외) → 예상·최대 비용 |
| 6 (선택) | `classify_error` | RateLimit → InternalServer → APIStatus → APIConnection 순서, retry-after |

제공 코드: `RefusalError`·`TruncatedError`·`StructuredOutputError`, `Triple`·`Extraction`(Pydantic), `ResponseView`·`StreamOutcome`·`CostEstimate`·`ErrorDecision`, `to_usage`, P2-01B에서 불러온 `Usage`·`ProviderError`·`log_usage`.

## 8. 예상 결과

아래는 클라우드 작업 환경(Python 3.12.13, anthropic 2.6.0)에서 `run_demo.py --impl solution`을 **실제로 실행한 출력**입니다. 다만 **가짜 SDK가 미리 정해 둔 응답·토큰 수(입력 1,200 등)·가짜 시계**를 쓴 것이라, Claude의 실제 답·토큰 수·시간이 아닙니다(형식 예시). 실제 호출 결과는 `call_claude.py`로 사용자 PC에서 얻어 experiments 기록에 적으세요. 이 환경에서는 키가 없어 `건너뜀: ANTHROPIC_API_KEY 가 없습니다`로 끝났습니다.

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

**해석**
- [1] 요청에는 샘플링 파라미터·thinking 끄기·강제 tool_choice가 없습니다. 마지막 메시지는 user입니다.
- [2] 스키마가 `additionalProperties: false`로 바뀌어 모델이 정의되지 않은 필드를 만들지 못합니다. 결과는 Pydantic 객체라 다음 단계에서 `t.subject`처럼 바로 씁니다.
- [3] 가짜 시계에서 첫 텍스트가 1.2초에 왔다면, 실제 Claude에서는 그 안에 thinking 시간이 들어 있습니다(§4-4).
- [4] 거절과 잘림은 예외로 분리됩니다. 폴백은 블록이 아니라 `usage.iterations`로 판단합니다.
- [5] 최대 비용이 예상의 약 30배입니다. `max_tokens`를 작업에 맞게 정하는 것이 비용 통제입니다.
- [6] `RateLimitError`는 `APIStatusError`의 하위 클래스라, 상위 클래스를 먼저 검사하면 "429 + 7초 뒤"라는 정보를 잃습니다.

## 9. 자주 나는 오류

| 증상 | 원인 | 해결 |
|---|---|---|
| `400 ... temperature` / `top_p` | 샘플링 파라미터 전송 | 빼기. 일관성은 프롬프트·구조화 출력으로 |
| `400 ... thinking` | `{"type": "disabled"}` 또는 `budget_tokens` | 생략, 깊이는 `output_config.effort` |
| `400` (마지막 메시지 assistant) | prefill | 마지막은 user, 형식은 구조화 출력 |
| `AttributeError: ... has no attribute 'text'` | `content[0].text` (첫 블록이 thinking) | `type == "text"` 블록만 |
| 거절인데 "JSON 오류"로 보임 | 검증을 stop_reason보다 먼저 | `ensure_complete` → 검증 |
| 답이 중간에 끊김 | `max_tokens`가 thinking + 답에 부족 | 늘리기, `stop_reason == "max_tokens"` 처리 |
| `400` (fallbacks) | 헤더와 형식 불일치(`"default"` ↔ `-2026-07-01`, 배열 ↔ `-2026-06-01`) 또는 Batches·Bedrock 등 | 짝 맞추기, 지원하지 않는 곳은 `fallbacks=False` |
| `client.messages.create(... fallbacks=...)` TypeError | stable 네임스페이스에는 `fallbacks`·`betas`가 없음 | `client.beta.messages.create`/`stream` |
| 429만 오면 서비스가 멈춤 | 재시도 정책 없음 | SDK `max_retries` + `classify_error` → 안내·공급자 폴백 |
| 토큰 수가 추정과 다름 | tiktoken 등 다른 토크나이저 | `count_tokens`에 실제 모델 ID |

## 10. 과제

1. (키가 있으면) `call_claude.py`를 실행해 triple 추출 결과·usage·stop_reason·지연을 experiments 표에 기록하세요. `count_tokens` 추정과 실제 `usage.input_tokens`를 비교하세요.
2. (키가 있으면) 같은 질문을 effort `low`/`medium`/`high`로 보내 TTFT·출력 토큰·답의 질을 비교하세요(§4-4). P2-01B Ollama 결과와 비교할 때 무엇을 같은 조건으로 맞췄는지 적으세요.
3. `Extraction` 스키마에 `confidence: float`(0~1) 필드를 추가하고 테스트를 고쳐 보세요. 스키마가 바뀌면 파이프라인의 어디가 영향을 받나요?
4. 사내 상담 봇의 시스템 프롬프트를 작성해 보세요. 캐싱을 깨지 않으려면 무엇을 system에 두고 무엇을 messages에 두어야 하나요?

## 11. 체크리스트

- [ ] `IMPL=starter pytest part2_llm/llm_api/basics/tests` 가 모두 통과한다
- [ ] system과 messages의 차이, 멀티턴을 매번 다시 보내는 이유를 설명할 수 있다
- [ ] 응답을 읽는 순서(stop_reason → text 블록 → 검증 → 폴백 여부)를 안다
- [ ] 거절이 HTTP 200으로 온다는 것과 서버 측 폴백이 다루는/다루지 않는 오류를 안다
- [ ] Claude의 TTFT에 thinking이 포함되는 이유와 로컬 모델과의 공정 비교 방법을 말할 수 있다
- [ ] `max_tokens`가 비용 상한이라는 것과 `count_tokens`로 호출 전 비용을 계산하는 법을 안다

## 12. 다음 단계 연결

- **P2-03 비용·지연**: 프롬프트 캐싱(`cache_control`, `usage.cache_read_input_tokens`)·배치·사용량 기록으로 이 실습의 비용을 줄입니다.
- **P2-04**: 프롬프트 vs RAG vs Fine-tuning 선택 기준.
- **Part 4 P4-02/03**: 이 실습의 triple 추출(구조화 출력)을 엔티티·관계 추출로 확장해 그래프에 적재합니다.
- **Part 6**: 이 요청 빌더·응답 검사·오류 분류를 FastAPI AI 서버로 옮깁니다(P2-01B 인터페이스와 합침).

## REF

- claude-api 스킬(작성 시점 2026-11): `python/claude-api/README.md`(Client·Timeouts·Retries·Error Handling·Extended Thinking·Stop Reasons), `python/claude-api/streaming.md`, `shared/models.md`, `shared/model-migration.md`(Prefill replacement, Migrating to Claude Opus 5 / 5.5, refusal·fallbacks), `shared/token-counting.md`, `shared/error-codes.md`, `shared/platform-availability.md`
- 설치된 anthropic 2.6.0 확인: `client.beta.messages.create/stream/parse`의 `fallbacks`(`Literal["default"]` 허용)·`betas`, `OutputConfigParam`(effort·format), `anthropic.transform_schema`, `lib/_parse`(parse 즉시 검증), 예외 계층
- 공식 문서: 구조화 출력 https://platform.claude.com/docs/en/build-with-claude/structured-outputs , 거절·폴백 과금 https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback , 가격 https://platform.claude.com/docs/en/about-claude/pricing (작성 환경에서 직접 열어 보지는 않음 — 스킬이 인용한 주소)

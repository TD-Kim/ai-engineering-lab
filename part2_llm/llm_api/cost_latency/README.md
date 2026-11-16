# P2-03 비용·지연 — 프롬프트 캐싱, 배치, 토큰 사용량 기록

| 항목 | 값 |
|---|---|
| 실습 ID | P2-03 (메인) |
| 주차 · 진행 순서 | W6 · P2-02 다음 |
| 예상 시간 | 3~4시간 |
| 파일 | `starter.py`(직접 채우기), `solution.py`(정답), `tests/test_p2_03_cost_latency.py`, `run_demo.py`, 선택: `measure_caching.py`(API 키 필요, 유료) |
| 소속 | [Part 2 — LLM 활용 원리](../../README.md) · [llm_api](../README.md) |
| 재사용 | [P2-02](../basics/README.md)의 `build_request`·`read_response`·`Usage`(P2-01B)·로그 열(`USAGE_FIELDS`)을 경로로 불러 씁니다(복사하지 않음) |

> 방향: **개념은 깊게, 구현은 얕게.** "같은 품질에서 비용·지연을 줄이는 설계"를 캐싱·배치·사용량 기록으로 익힙니다. 무엇을 앞에 두면 캐시가 읽히는지, 무엇이 캐시를 깨는지, 언제 배치를 쓸 수 있는지를 코드와 숫자로 확인합니다. 테스트와 데모는 가짜 SDK로 돌아가고, 실제 측정은 선택 스크립트입니다.

## 1. 목표

- 프롬프트 캐싱이 **접두어 일치**라는 것을 설명하고, 고정 부분을 앞에·바뀌는 질문을 뒤에 두는 요청을 만든다.
- 응답 `usage`의 세 입력 칸(정가·캐시 쓰기·캐시 읽기)을 읽어 적중을 판정하고 **실제 비용과 절감액**을 계산한다.
- 두 요청 본문을 비교해 **캐시가 깨진 위치**(현재 시각·키 순서·도구 목록·모델·effort)를 찾는다.
- Batches API 요청을 `custom_id`로 구성하고, 순서가 섞여 오는 결과를 `custom_id`로 맞춘다. 50% 할인과 "최대 24시간"의 맞바꿈을 설명한다.
- 사용량 로그(CSV)에서 요청 수·토큰·캐시 적중률·평균/p95 지연·총비용을 집계한다.
- (선택) 작업 성격에 따라 캐싱·배치·effort·`max_tokens` 중 무엇을 쓸지 고른다.

## 2. 선수지식

- P1-01: 토큰·단가, P1-05: KV 캐시와 "같은 앞부분은 다시 계산하지 않는다"는 원리
- P2-01B: `Usage`·`log_usage`(사용량 한 줄 기록), P2-02: `build_request`(effort·폴백), `read_response`(text 블록만), `max_tokens` = 비용 상한
- Python `csv`, `dataclass`, 정규 표현식 `re.match`

## 3. 핵심 개념

| 개념 | 한 줄 설명 |
|---|---|
| 프롬프트 캐싱 | 요청 앞부분(접두어)의 처리 결과를 서버가 잠시 보관했다가 다음 요청에 다시 쓴다 |
| 접두어 일치 | 맨 앞부터 breakpoint까지 **바이트가 똑같아야** 읽힌다. 한 글자라도 바뀌면 그 뒤는 전부 무효 |
| 그려지는 순서 | `tools` → `system` → `messages` (그리고 모델이 다르면 캐시가 완전히 따로) |
| `cache_control` (breakpoint) | "여기까지 캐시하라"는 표시. 요청당 최대 4개. 기본 수명 5분, 선택 1시간 |
| 자동 캐싱 | 요청 맨 위의 `cache_control` 하나로 마지막 캐시 가능 블록에 자동으로 표시 |
| 최소 캐시 길이 | 이보다 짧으면 표시가 있어도 **조용히** 캐시되지 않는다(모델마다 다름, `claude-opus-5-5` 512 토큰) |
| `usage` 세 칸 | `input_tokens`(정가, **캐시 제외 나머지**) + `cache_creation_input_tokens`(쓰기) + `cache_read_input_tokens`(읽기) = 입력 전체 |
| 캐시 적중률 | 캐시 읽기 토큰 ÷ 입력 전체 토큰 |
| Batches API | 요청 묶음을 맡기고 나중에 결과를 받는 비동기 처리. 입력·출력 50% 할인, 대부분 1시간 안·최대 24시간 |
| `custom_id` | 배치 요청마다 붙이는 이름표. 결과는 순서 무관이라 이것으로 맞춘다 |
| p95 지연 | 요청 100개 중 느린 5개를 뺀 가장 느린 값. 평균이 가리는 꼬리를 본다 |

## 4. 내부 원리

### 4-1. 근거 — 확인한 출처 (작성 시점 2026-11-16)

| 항목 | 이 실습의 선택 | 근거 |
|---|---|---|
| `cache_control` 형식 | `{"type": "ephemeral"}`(5분), `{"type": "ephemeral", "ttl": "1h"}`(1시간) | 스킬 `python/claude-api/README.md` Prompt Caching, 공식 [prompt-caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) |
| breakpoint 수 | 요청당 최대 4개(`count_breakpoints`로 확인) | 스킬 `shared/prompt-caching.md` API reference, 공식 문서 "up to 4 cache breakpoints" |
| 최소 캐시 길이 | `claude-opus-5-5` 512 토큰 (Opus 4.6 등 4096, Opus 4.8 1024 — 세대마다 오르내린다) | 공식 prompt-caching 문서(2026-11-16 열람). 스킬은 "값을 쓰기 전에 공식 문서 확인"이라고 적어 두어 공식 문서로 확인함 |
| 캐시 수명 | 기본 5분, 선택 1시간. 읽으면 수명이 다시 시작(무료 갱신). 수명은 요청 **시작**부터 센다 | 스킬 `shared/prompt-caching.md` Choosing the TTL |
| 단가(claude-opus-5-5) | 입력 $4, 5분 쓰기 $5(1.25×), 1시간 쓰기 $8(2×), 캐시 읽기 $0.20(**0.05×**), 출력 $20 / 100만 토큰 | 공식 [가격 페이지](https://platform.claude.com/docs/en/about-claude/pricing) 2026-11-16 열람. 캐시 배수는 배치 할인과 겹쳐 적용 |
| Sonnet 5.5 단가 | 입력 $2, 5분 쓰기 $2.50, 1시간 쓰기 $4, 캐시 읽기 **$0.10(0.05×)**, 출력 $10 — `PRICES["claude-sonnet-5-5"]` | 공식 가격 페이지(2026-11-16)를 따른다. 작성 환경의 스킬 `shared/models.md`·`shared/prompt-caching.md`도 $0.10으로 일치(검증 세션은 스킬 모델표의 $0.20 표기를 보고함 — 스킬 판본 차이로 보이며, 어느 쪽이든 이 실습은 공식 가격표를 따른다) |
| 배치 | `client.messages.batches.create(requests=[{custom_id, params}])` → `retrieve(id).processing_status == "ended"` → `results(id)`. 50% 할인, 최대 100,000 요청·256 MB, 대부분 1시간 안·최대 24시간, 결과 29일 보관 | 스킬 `python/claude-api/batches.md`, 공식 [batch-processing](https://platform.claude.com/docs/en/build-with-claude/batch-processing) |
| 배치에서 못 쓰는 것 | `stream`·`speed`·`fallbacks`·`max_tokens: 0`. 서버 측 폴백은 배치에 적용되지 않음 → P2-02 `build_request(..., fallbacks=False)` | 스킬 batches.md Key Facts, 공식 batch-processing 문서 표 |
| `custom_id` 형식 | `^[a-zA-Z0-9_-]{1,64}$`를 문자열 **전체**에 적용(`fullmatch`). `re.match` + `$`는 끝 줄바꿈 앞에서도 맞아 `"abc\n"`을 통과시킨다 | 공식 batch-processing 문서, Python `re` 동작은 작성 환경에서 실행해 확인 |
| errored 결과의 원인 | `result.result.error.error.type == "invalid_request_error"`면 요청 수정, 그 밖은 재전송 | 설치된 anthropic 2.6.0 `MessageBatchErroredResult.error: ErrorResponse{type: "error", error: ErrorObject}` + 공식 문서 예시. **스킬 예시의 `result.result.error.type == "invalid_request"`는 SDK 타입과 맞지 않아 쓰지 않음** |
| effort·thinking 변경 | 바꾸면 messages 캐시가 무효(모델에 따라 그 앞까지) → 경로마다 고정. 기본 effort를 명시하는 것은 생략과 같아 비용 없음 | 스킬 `shared/prompt-caching.md` Invalidation hierarchy |
| 캐시 범위 | Claude API에서는 workspace별로 분리(조직 간 공유 없음). 모델별로 따로 | 스킬 `shared/prompt-caching.md` API reference |
| 요청 형태 검증 | 설치된 SDK가 이 실습의 요청을 받는지 **가짜 HTTP 전송(MockTransport)** 으로 확인: `client.beta.messages.create(**req)`의 본문에 `system` 블록·`cache_control`·`fallbacks`가 그대로 실리고, `batches.create`의 params 에 `fallbacks`·`betas`가 없음 | 작성 환경에서 실행(네트워크 없음) |

### 4-2. 캐시가 읽히는 요청, 깨지는 요청

```
요청 1: [지시문][규정 문서 10,000 토큰]★ │ "연차는 며칠 전까지?"      → 쓰기 10,000 (1.25×)
요청 2: [지시문][규정 문서 10,000 토큰]★ │ "반차도 같아?"            → 읽기 10,000 (0.05×)
요청 3: [지시문 + "현재 시각 09:05"][규정 문서]★ │ "팀장이 거절하면?"  → 다시 쓰기 (★ 앞이 달라짐)
                                    ★ = cache_control
```

- **breakpoint는 "모든 요청에 공통인 마지막 블록"에.** 질문 블록에 두면 질문마다 다른 캐시를 써서 아무것도 읽히지 않는다(쓰기 값만 낸다).
- **흔한 범인**(스킬 Silent invalidators): 시스템 프롬프트의 현재 시각·요청 ID, 정렬하지 않은 JSON 직렬화(키 순서), 사용자마다 다른 도구 목록, 조건부 시스템 문단, 모델 교체, effort·thinking 변경.
- **멀티턴은 정상**: 다음 요청이 "앞 요청 + 새 턴"이면 앞 요청 전체가 접두어라 읽힌다. `first_divergence`가 `location=None`을 돌려주는 경우다.
- 찾는 방법: 직전·직후 요청 본문 두 개를 `first_divergence`로 비교한다(TODO 3). Claude API에는 같은 비교를 서버가 해 주는 **cache diagnostics**(베타)도 있다 — 요청마다 `diagnostics` 객체를 보내야 하므로 처음부터 켜 둬야 한다(스킬 `shared/prompt-caching.md`, 현재 형태는 공식 Cache diagnostics 페이지 확인).

### 4-3. 캐시가 지연에 주는 영향

캐시를 읽으면 그 앞부분의 입력 처리(prefill)를 다시 하지 않아 **TTFT가 줄 수 있다**. 줄어드는 양은 앞부분 길이·서버 상태에 따라 다르므로 이 저장소에는 숫자를 적지 않았다 — `measure_caching.py`로 사용자 PC에서 잰다. Claude의 TTFT에는 thinking 시간도 들어 있으므로(P2-02 §4-4), 캐시 효과를 보려면 같은 effort로 비교한다.

### 4-4. 배치는 "공짜 할인"이 아니다

| | 일반 호출 | 배치 |
|---|---|---|
| 비용 | 정가 | 입력·출력 50% (캐시 배수와 겹쳐 적용) |
| 지연 | 초 단위 | 대부분 1시간 안, **최대 24시간**(못 끝나면 expired) |
| 결과 순서 | 요청한 것 하나 | **순서 무관** → `custom_id`로 매칭 |
| 서버 측 폴백 | 켤 수 있음 | 적용 안 됨(거절된 것은 따로 다시 보냄) |
| 도구 반복 | 가능 | 요청 하나가 한 번에 끝나야 함(중간 도구 반복 없음) |
| 배치 안 캐시 | — | 같은 앞부분을 써도 적중은 "최선 노력"(동시에 처리되면 못 읽을 수 있음) |

## 서비스 관점

| 질문 | 판단 |
|---|---|
| RAG에서 무엇을 캐시하나? (Part 3) | **모든 질문에 공통인 것** — 시스템 프롬프트, 고정 규정집·제품 설명서, few-shot 예시. 질문마다 달라지는 검색 결과는 캐시해도 소용없으니 질문 쪽(뒤)에 둔다. 같은 문서 묶음을 여러 질문이 연달아 쓰는 화면(문서 하나를 두고 대화)은 문서까지 캐시한다 |
| 캐싱이 잘 되는지 어떻게 아나? | 코드가 아니라 **`usage`의 숫자로만** 안다. 캐시 적중률(토큰 기준)을 대시보드 지표로 두고, 프롬프트 조립 코드를 바꿀 때마다 확인한다. 가장 비싼 실패는 조용하다 — 에러 없이 청구서만 커진다 |
| 적중률이 떨어지면? | 직전·직후 요청 본문을 `first_divergence`로 비교해 범인(시각·키 순서·도구·effort)을 찾는다. 캐시 diagnostics(베타)를 미리 켜 두면 서버가 알려 준다 |
| 배치는 어디에? | 사람이 기다리지 않는 대량 작업: 평가 세트 채점(P3-07), 색인 시 엔티티·관계 추출(P4-02/03), 야간 요약. **지연에 민감한 경로(채팅·검색 화면)에는 쓰지 않는다** |
| 비용 상한은? | 요청 하나: `max_tokens`(P2-02). 하루: 로그의 `total_usd`로 예산 알림. 긴 입력은 호출 전에 `count_tokens`로 막는다 |
| effort·모델은 언제 낮추나? | 캐싱·배치는 품질이 그대로인 "공짜 수단"이라 먼저 쓴다. effort·모델 낮추기는 품질이 바뀔 수 있으니 평가 세트(P3-07)로 확인한 뒤 경로별로 정한다. effort는 경로마다 고정한다(바꾸면 캐시가 깨진다) |
| 로그에 무엇을 남기나? | 모델·토큰 4칸(정가·쓰기·읽기·출력)·지연·stop_reason·배치 여부. **프롬프트·답 원문은 남기지 않는다**(개인정보·영업비밀). 원인 분석용 본문 비교가 필요하면 마스킹·샘플링·짧은 보존 기간을 정한다 |

## 5. 수식

- 입력 전체 = `input_tokens` + `cache_creation_input_tokens` + `cache_read_input_tokens`
- 실제 비용 = (정가 × $4 + 5분 쓰기 × $5 + 1시간 쓰기 × $8 + 읽기 × $0.20 + 출력 × $20) / 10⁶ (claude-opus-5-5, 2026-11-16)
- 캐시 없었다면 = (입력 전체 × $4 + 출력 × $20) / 10⁶, 절감 = 캐시 없었다면 − 실제
- 배치 = 위 두 값 × 0.5
- 캐시 적중률 = Σ읽기 / Σ입력 전체
- p95(nearest-rank) = 정렬한 값의 ⌈0.95 × n⌉ 번째

손으로 따라 하기(문서 10,000 + 질문 50 토큰, 출력 200 토큰):

| | 실제 | 캐시 없었다면 |
|---|---|---|
| 1번째(쓰기) | 50×4 + 10,000×5 + 200×20 = 54,200 → **$0.0542** | 10,050×4 + 200×20 = 44,200 → $0.0442 |
| 2번째(읽기) | 50×4 + 10,000×0.20 + 200×20 = 6,200 → **$0.0062** | $0.0442 |
| 합계 | $0.0604 | $0.0884 |

1번째는 쓰기 값(1.25×) 때문에 오히려 비싸고, 2번째부터 이득이다(5분 수명의 손익분기 = 2회). 1시간 수명은 쓰기가 2×라 3회 이상 읽어야 이득이다. 단가는 바뀌므로 사용 전 [공식 가격 페이지](https://platform.claude.com/docs/en/about-claude/pricing)에서 다시 확인하세요.

## 6. 실행법

저장소 루트에서 실행합니다.

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"
pip install -r requirements/part2.txt

# 1) 남은 TODO 확인
python part2_llm/llm_api/cost_latency/starter.py

# 2) 직접 구현 검사 (TODO 가 남은 테스트는 skip)
$env:IMPL = "starter"; pytest part2_llm/llm_api/cost_latency/tests; Remove-Item Env:IMPL

# 3) 정답 코드 검사
pytest part2_llm/llm_api/cost_latency/tests

# 4) 데모 (가짜 SDK, 키 불필요)
python part2_llm/llm_api/cost_latency/run_demo.py --impl starter

# 5) (선택) 실제 측정 — 유료(호출 3회). .env 에 ANTHROPIC_API_KEY (커밋 금지)
Copy-Item .env.example .env   # 처음 한 번
python part2_llm/llm_api/cost_latency/measure_caching.py
python part2_llm/llm_api/cost_latency/measure_caching.py --batch            # 배치 2건 추가로 보냄(결과는 나중에)
python part2_llm/llm_api/cost_latency/measure_caching.py --batch-id msgbatch_...   # 배치 결과 확인
```

**bash (Linux/macOS)**

```bash
source .venv/bin/activate
IMPL=starter pytest part2_llm/llm_api/cost_latency/tests
pytest part2_llm/llm_api/cost_latency/tests
python part2_llm/llm_api/cost_latency/run_demo.py
python part2_llm/llm_api/cost_latency/measure_caching.py      # 선택, 유료
```

- 테스트는 키·네트워크가 필요 없습니다. usage·배치 결과는 가짜 객체, 폴링의 sleep도 가짜 함수입니다. 스레드·대기 반복이 없습니다.
- `IMPL=starter`에서도 제공 코드만 검사하는 테스트 2개(`wait_for_batch`, 단가·상수 확인)는 통과합니다. 나머지 27개가 TODO 미구현으로 skip입니다.
- 데모는 `--log` 경로(기본 `part2_llm/llm_api/cost_latency/outputs/usage_demo.csv`, git 제외)를 매번 새로 씁니다. `measure_caching.py`의 `outputs/usage_real.csv`는 누적됩니다.

## 7. TODO 목록

| TODO | 함수 | 내용 |
|---|---|---|
| 1 | `cache_marker`, `build_cached_request` | TTL 표시(5m/1h), system = [지시문, 문서들]의 **마지막 블록에만** `cache_control`, 질문은 messages(P2-02 `build_request` 재사용) |
| 2 | `read_cache_usage`, `cache_cost` | usage 세 칸 + TTL 구분(`cache_creation`) → `CacheUsage`, 칸별 단가로 실제·캐시 없었다면·절감, 배치 할인 |
| 3 | `first_divergence` | `render_segments` 두 목록을 앞에서부터 비교 → 첫 차이 위치·글자 위치·공통 길이, 바뀐 설정(effort 등) |
| 4 | `build_batch`, `collect_batch_results` | `custom_id` 형식(`fullmatch`)·중복·배치 불가 키 검사, 결과를 `custom_id`로 분류(성공·재전송·수정·취소·누락·보내지 않은 id) |
| 5 | `p95`, `summarize_usage_log` | nearest-rank p95, CSV 집계(토큰·적중률·평균/p95 지연·총비용·stop_reason, P2-01B 형식 로그도 읽기) |
| 6 (선택) | `choose_levers` | 캐싱(5m/1h/없음)·배치·effort_low·max_tokens_cap 규칙 |

제공 코드: `count_breakpoints`, `CacheUsage`(`prompt_tokens`·`status`), `CostBreakdown`, `render_segments`, `PrefixDiff`, `BatchOutcome`, `wait_for_batch`(횟수 상한 폴링), `log_cached_usage`(P2-01B 로그 열 + 캐시·배치 열), `UsageReport`, `Workload`, 단가표 `PRICES`. P2-02에서 불러온 `build_request`·`read_response`·`Usage`.

## 8. 예상 결과

아래는 클라우드 작업 환경(Python 3.12.13, anthropic 2.6.0)에서 `run_demo.py --impl solution`을 **실제로 실행한 출력**입니다. 다만 **가짜 SDK가 캐시를 흉내 낸 것**이라 토큰 수는 "글자 2개 = 1토큰"으로 센 가짜 값이고, 지연도 미리 정한 가짜 값입니다. Claude의 실제 토큰 수·지연이 아닙니다(형식 예시). 실제 값은 `measure_caching.py`로 사용자 PC에서 얻어 experiments 기록에 적으세요. 이 환경에서는 키가 없어 `건너뜀: ANTHROPIC_API_KEY 가 없습니다`로 끝났습니다.

```
[1] 캐시 친화적 요청 (TODO 1) — 고정 지시문·문서는 system 앞쪽, 질문은 messages 맨 뒤
    system[0] '당신은 사내 규정 안내원입니다. 아래 규정 '... 가짜 토큰 28  cache_control=None
    system[1] '제1조 (휴가 규정 1) 직원은 연차를 사용'... 가짜 토큰 1430  cache_control={'type': 'ephemeral'}
    messages[-1] = {'role': 'user', 'content': '연차는 며칠 전까지 신청해야 해?'}   breakpoint 1개 (최대 4)
[2] 같은 앞부분 + 다른 질문 → 캐시 쓰기 → 읽기 → (시스템 프롬프트에 시각을 넣으면) 다시 쓰기 (TODO 2)
    요청                           상태         정가입력     캐시쓰기     캐시읽기      실제 $     캐시없음 $      절감 $
    연차는 며칠 전까지 신청해야 해?           write         9     1458        0   0.00973    0.00827  -0.00146
    반차도 같은 규칙이야?                 hit           6        0     1458   0.00272    0.00826   0.00554
    [시각 포함] 팀장이 승인을 안 하면?        write         6     1470        0   0.00977    0.00830  -0.00147
[3] 왜 깨졌나 (TODO 3) — 두 요청 본문을 앞에서부터 비교
    첫 차이: system[0], 세그먼트 안 0번째 글자, 공통 15글자
    a: '당신은 사내 규정 안내원입니다. 아래 규정 문서에 근거'
    b: '현재 시각 2026-11-16 09:05. 당신은 사내'
    (정상) 질문만 다른 두 요청: 첫 차이 messages[0].user → system 까지는 캐시를 읽는다
    effort 만 바꾼 요청: 본문 차이 None, 바뀐 설정 ['output_config'] → 캐시 영향 있음(경로마다 고정)
[4] 배치 (TODO 4) — custom_id 로 결과 맞추기, 50% 할인 (가짜 결과, 도착 순서 뒤섞임)
    요청 3개, params 키: ['max_tokens', 'messages', 'model', 'output_config', 'system']  (fallbacks·betas 없음)
    성공 ['policy-q0', 'policy-q1']  다시 보낼 것 ['policy-q2']  고칠 것 []  빠진 것 []
    policy-q0: 일반 $0.00273 → 배치 $0.00136
    policy-q1: 일반 $0.00272 → 배치 $0.00136
[5] 사용량 로그 집계 (TODO 5) — part2_llm/llm_api/cost_latency/outputs/usage_demo.csv
    요청 5, 정가입력 36, 캐시쓰기 2928, 캐시읽기 4374, 출력 600
    캐시 적중률(토큰) 59.6%, 평균 지연 1720 ms, p95 2400 ms (가짜 지연), 총비용 $0.02494, stop_reason {'end_turn': 5}
[6] 언제 무엇을 쓰나 (TODO 6, 선택)
    사내 규정 챗봇(사용자 대기, 1분 간격)    → ['cache_5m', 'max_tokens_cap']
    야간 엔티티 추출(색인)              → ['cache_5m', 'batch', 'effort_low', 'max_tokens_cap']
    하루 한 번 보고서                 → ['batch', 'max_tokens_cap']
```

**해석**
- [1] breakpoint는 마지막 system 블록(규정 문서)에 하나. 질문에는 표시가 없습니다.
- [2] 1번째는 쓰기라 캐시 없을 때보다 비싸고(절감이 음수), 2번째는 같은 앞부분을 읽어 약 1/3 값입니다. 3번째는 지시문 앞에 시각을 붙여서 앞부분 전체를 다시 썼습니다.
- [3] `first_divergence`가 `system[0]`의 0번째 글자를 가리킵니다 — 시각을 맨 앞에 붙였기 때문입니다(공통 15글자는 앞의 model 값). 질문만 다른 두 요청은 `messages`에서 갈려 system까지는 캐시를 읽습니다. effort만 바꾸면 본문 차이는 없지만 `settings_changed`가 알려 줍니다.
- [4] 결과가 섞여 와도 `custom_id`로 맞춥니다. `overloaded_error`는 다시 보낼 목록으로 갑니다. 배치 비용은 일반의 절반입니다.
- [5] 적중률은 **토큰 기준**입니다. 지연은 가짜 값이라 평균·p95의 숫자 자체에는 의미가 없고, 계산 형식만 보세요.
- [6] 챗봇은 캐싱만, 야간 추출은 캐싱 + 배치 + 낮은 effort, 하루 한 번 보고서는 간격이 1시간을 넘어 캐싱 없이 배치만 나옵니다.

## 9. 자주 나는 오류

| 증상 | 원인 | 해결 |
|---|---|---|
| `cache_read_input_tokens`가 계속 0 | 앞부분에 시각·ID, 정렬 안 된 JSON, 도구 목록 변동 | `first_divergence`로 비교, 바뀌는 값은 마지막 breakpoint 뒤로 |
| 쓰기도 읽기도 0 | 앞부분이 최소 길이(512 토큰, 모델마다 다름) 미만 | 앞부분을 키우거나 캐싱을 빼기(에러 없이 조용히 안 됨) |
| 매번 쓰기만 함 | breakpoint를 질문 블록(매번 바뀌는 곳)에 둠 | 공통 부분의 마지막 블록으로 옮기기 |
| `400` cache_control 개수 | breakpoint가 4개를 넘음(맨 위 자동 캐싱도 1개로 셈) | `count_breakpoints` ≤ 4 |
| TTL을 섞었더니 오류·예상과 다른 캐시 | 1시간 표시가 5분 표시보다 뒤에 옴(규칙: 긴 TTL이 앞) | 긴 TTL을 앞에(스킬 `shared/prompt-caching.md`, 정확한 오류 형태는 공식 문서 확인) |
| `input_tokens`가 너무 작아 보임 | 캐시 제외 나머지라서 | 입력 전체 = 세 칸 합 |
| 배치 요청 오류(`fallbacks`·`stream` 등) | 배치가 받지 않는 키 | `build_request(..., fallbacks=False)`, `build_batch`가 미리 막음 |
| 배치 결과가 엉뚱한 질문에 붙음 | 결과를 순서(인덱스)로 맞춤 | `custom_id`로 맞추기 |
| 배치가 안 끝남 | 대부분 1시간이지만 최대 24시간 | 무한 대기 대신 횟수 상한 폴링, 나중에 `--batch-id`로 확인 |
| effort를 바꿨더니 비용이 뜀 | effort 변경 = messages 캐시 무효 | 경로마다 effort 고정 |

## 10. 과제

1. (키가 있으면) `measure_caching.py`를 실행해 세 요청의 usage·지연·비용을 experiments 표에 기록하세요. 1) 쓰기 → 2) 읽기 → 3) 다시 쓰기가 실제로 나오나요? 2)의 지연이 1)보다 짧았나요(같은 effort)?
2. `run_demo.py`의 `stamped`를 "지시문 뒤에 시각을 붙이기"로 바꿔 보세요. `first_divergence`의 결과와 캐시 쓰기 토큰이 어떻게 바뀌나요? 시각을 아예 질문(messages) 쪽으로 옮기면?
3. 사내 규정 챗봇(규정집 2만 토큰, 하루 2,000 질문, 평균 간격 30초)의 하루 입력 비용을 캐싱 있음/없음으로 계산해 보세요(출력은 같다고 가정). 간격이 20분이면 TTL을 무엇으로 고르겠어요?
4. (선택, 키가 있으면) `--batch`로 2건을 보내고 결과가 오는 데 걸린 시간을 기록하세요. 이 시간을 보고 어떤 작업에 배치를 쓰겠어요?

## 11. 체크리스트

- [ ] `IMPL=starter pytest part2_llm/llm_api/cost_latency/tests` 가 모두 통과한다
- [ ] 캐싱이 접두어 일치라는 것과 breakpoint를 어디에 두어야 하는지 설명할 수 있다
- [ ] usage의 세 입력 칸과 "입력 전체 = 세 칸 합"을 안다
- [ ] 캐시를 깨는 흔한 원인 4가지 이상과 찾는 방법을 말할 수 있다
- [ ] 5분/1시간 TTL의 손익분기(2회/3회)를 계산할 수 있다
- [ ] 배치의 50% 할인과 맞바꾸는 것(최대 24시간·순서 무관·폴백 없음)을 안다
- [ ] p95가 평균보다 서비스 지표로 나은 이유를 말할 수 있다

## 12. 다음 단계 연결

- **P2-04**: 프롬프트 vs RAG vs Fine-tuning 선택 기준 — 비용·지연 표를 판단 근거로 씁니다.
- **Part 3**: RAG 시스템 프롬프트·고정 문서 캐싱, P3-07 평가 세트 채점을 배치로.
- **Part 4 P4-02/03**: 색인 시 엔티티·관계 추출을 배치 + 캐싱으로.
- **Part 6 P6-04**: 이 로그 집계(적중률·p95·총비용)를 모니터링·예산 알림으로 옮깁니다.

## REF

- claude-api 스킬(작성 시점 2026-11-16): `shared/prompt-caching.md`(invariant, placement patterns, silent invalidators, API reference, Choosing the TTL, Verifying cache hits, Invalidation hierarchy), `python/claude-api/README.md`(Prompt Caching), `python/claude-api/batches.md`, `shared/cost-optimization.md`(공짜 수단 먼저, 배치 2.5절)
- 공식 문서(2026-11-16 작성 환경에서 열람): 가격 https://platform.claude.com/docs/en/about-claude/pricing , 프롬프트 캐싱 https://platform.claude.com/docs/en/build-with-claude/prompt-caching , 배치 https://platform.claude.com/docs/en/build-with-claude/batch-processing
- 설치된 anthropic 2.6.0: `types/usage.py`(`cache_creation_input_tokens`·`cache_read_input_tokens`·`cache_creation`), `types/cache_creation.py`, `types/messages/message_batch_*_result.py`, `types/messages/batch_create_params.py`
- Python 문서 `csv.DictReader` https://docs.python.org/3/library/csv.html#csv.DictReader (작성 환경에서 열람 불가 — 미확인, 동작은 설치된 Python 3.12 로 실행해 확인)

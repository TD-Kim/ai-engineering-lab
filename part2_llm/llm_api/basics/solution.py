"""P2-02 Claude API 기초 — 정답 코드 (메인 실습: 개념은 깊게, 구현은 얕게).

요청 본문을 만들고(TODO 1: system·messages·멀티턴), 응답을 검사하고(TODO 2: stop_reason·content 블록·폴백 여부),
구조화 출력으로 엔티티·관계 triple 을 받고(TODO 3), 스트리밍을 소비해 TTFT·최종 usage 를 얻고(TODO 4),
호출 전에 토큰을 세어 비용을 추정한다(TODO 5). (선택) 오류를 종류별로 나눠 재시도 여부를 정한다(TODO 6).

SDK 형태는 claude-api 스킬로 확인했다(작성 시점 2026-11, README §4 "근거" 표):
  - 모델 claude-opus-5-5 (환경 변수 LLM_MODEL 로 변경, 날짜 접미사 없음)
  - thinking 은 끌 수 없다(생략 = adaptive). 깊이는 output_config.effort (기본 medium — 명시한다)
  - temperature·top_p·top_k·budget_tokens·assistant prefill·강제 tool_choice 는 쓰지 않는다(400)
  - 거절은 HTTP 200 + stop_reason "refusal". 서버 측 폴백 fallbacks="default" (베타 server-side-fallback-2026-07-01)를 기본으로 켠다
P2-01B 의 Usage·ProviderError·log_usage 를 그대로 쓴다(part2_llm/local_llm_serving/solution.py).

테스트·데모는 가짜 SDK 객체로 돈다(키·네트워크 불필요). 실제 호출은 call_claude.py (선택, .env 의 ANTHROPIC_API_KEY).
진행 확인: python part2_llm/llm_api/basics/solution.py
"""
from __future__ import annotations

import importlib.util
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ValidationError

# ---------------------------------------------------------------------------
# (제공) P2-01B 재사용: Usage(사용량 기록 모양), ProviderError, log_usage
# ---------------------------------------------------------------------------
_P2_01B = Path(__file__).resolve().parents[2] / "local_llm_serving" / "solution.py"


def _load_p2_01b():
    name = "p2_01b_local_llm_serving_for_p2_02"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _P2_01B)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclass 가 자기 모듈을 찾을 수 있게 먼저 등록
    spec.loader.exec_module(mod)
    return mod


_p2_01b = _load_p2_01b()
Usage = _p2_01b.Usage
ProviderError = _p2_01b.ProviderError
log_usage = _p2_01b.log_usage

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_MAX_TOKENS = 16000          # thinking + 답변을 합친 출력 상한(아래 TODO 1 WHY)
EFFORTS = ("low", "medium", "high", "xhigh", "max")
FALLBACK_BETA = "server-side-fallback-2026-07-01"   # fallbacks="default" 형식 전용 베타(배열 형식은 다른 헤더)
# 단가(USD / 100만 토큰, 입력·출력). 작성 시점 2026-10 (P1-01 표), claude-api 스킬 모델표(2026-11)와 일치 확인.
# REF: https://platform.claude.com/docs/en/about-claude/pricing (사용 전 재확인)
PRICES_PER_MTOK = {"claude-opus-5-5": (4.00, 20.00), "claude-sonnet-5-5": (2.00, 10.00)}


class RefusalError(Exception):
    """stop_reason == "refusal". category 는 stop_details 의 참고 정보(없을 수 있음)."""

    def __init__(self, category: str | None = None):
        super().__init__(f"refused (category={category})")
        self.category = category


class TruncatedError(Exception):
    """stop_reason == "max_tokens": 출력 상한에 걸려 답이 잘렸다. partial 에 잘린 텍스트."""

    def __init__(self, partial: str):
        super().__init__("response truncated at max_tokens")
        self.partial = partial


class StructuredOutputError(Exception):
    """구조화 출력이 스키마 검증을 통과하지 못했다."""


# ---------------------------------------------------------------------------
# TODO 1 — 요청 본문: system 과 messages, 멀티턴
# ---------------------------------------------------------------------------
def build_messages(history: Sequence[tuple[str, str]], user_text: str) -> list[dict]:
    """history = [("user", "..."), ("assistant", "..."), ...] 뒤에 user_text 를 붙인 messages 목록.

    역할은 user 로 시작해 user/assistant 가 번갈아야 하고, 마지막은 반드시 user 다. 어기면 ValueError.
    """
    # TODO 1
    # REF: claude-api 스킬 python/claude-api/README.md "Multi-Turn Conversations", https://platform.claude.com/docs/en/api/messages/create
    # 개념: Claude API 의 대화는 상태가 없다(stateless). 서버가 이전 대화를 기억하지 않으므로,
    #       매 요청에 지금까지의 대화 전체를 messages 로 다시 보낸다.
    #       [{"role": "user", "content": "연차 신청은 며칠 전까지 해야 해?"},
    #        {"role": "assistant", "content": "(사내 규정에 따른 답)"},
    #        {"role": "user", "content": "반차도 같아?"}]     ← "반차도 같아?" 만으로는 뜻이 없다. 앞 대화가 문맥이다.
    # WHY(마지막은 user): 마지막을 assistant 로 두고 답의 앞부분을 미리 채우는 "prefill" 은 이 모델에서 400 오류다.
    #       답 형식을 강제하고 싶으면 구조화 출력(TODO 3)을 쓴다.
    # 서비스 관점: 대화가 길어질수록 매 요청의 입력 토큰(=비용·TTFT)이 는다 → 오래된 턴 요약·잘라내기·프롬프트 캐싱(P2-03).
    messages = [{"role": role, "content": text} for role, text in history]
    messages.append({"role": "user", "content": user_text})
    for i, m in enumerate(messages):
        expected = "user" if i % 2 == 0 else "assistant"
        if m["role"] != expected:
            raise ValueError(f"messages[{i}] role must be {expected!r}, got {m['role']!r}")
    return messages


def build_request(
    messages: list[dict],
    system: str | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    effort: str = "medium",
    fallbacks: bool = True,
) -> dict:
    """client.beta.messages.create(**req) / .stream(**req) 에 그대로 넘길 인자 사전."""
    # TODO 1 (이어서)
    # REF: claude-api 스킬 python/claude-api/README.md "System Prompts"·"Extended Thinking"(effort), shared/model-migration.md (fallbacks "default")
    # 개념: system = "역할·규칙·출력 형식" (모든 턴에 공통), messages = "대화 내용" (턴마다 바뀜).
    #       예) system="당신은 사내 규정 안내원입니다. 근거 문서에 없는 내용은 모른다고 답하세요."
    # 개념(effort): 이 모델은 thinking(답 전에 생각하기)을 끌 수 없고, 얼마나 깊이 생각할지를 effort 로 고른다.
    #       low → 빠르고 싸다, max → 느리고 비싸지만 어려운 문제에 강하다. 기본값(medium)도 명시해 두면 모델이 바뀌어도 동작이 같다.
    # WHY(max_tokens 16000): max_tokens 는 thinking + 답변을 합친 출력 상한이다. 너무 작으면 thinking 만 하다 잘린다(stop_reason "max_tokens").
    #       동시에 출력 토큰 = 비용이므로 "한 요청이 쓸 수 있는 최대 비용"이기도 하다(TODO 5).
    # WHY(fallbacks="default"): 안전 분류기가 요청을 거절하면 같은 호출 안에서 Anthropic 이 권장하는 다른 모델로 다시 실행한다.
    #       정책 거절에만 작동하고, rate limit·과부하·서버 오류는 폴백하지 않는다(그건 TODO 6 의 몫).
    #       베타 기능이라 betas=[FALLBACK_BETA] 와 client.beta.messages 를 쓴다(README §4-3).
    # 서비스 관점(system 설계): 바뀌지 않는 지시만 system 에 둔다. 현재 시각·사용자 이름처럼 매번 바뀌는 값을 넣으면
    #       프롬프트 캐싱이 깨진다(P1-05·P2-03). 사용자 입력은 messages 쪽에 둔다.
    if effort not in EFFORTS:
        raise ValueError(f"effort must be one of {EFFORTS}")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("messages must end with a user turn (no assistant prefill)")
    req: dict = {"model": model, "max_tokens": max_tokens, "messages": messages, "output_config": {"effort": effort}}
    if system:
        req["system"] = system
    if fallbacks:
        req["betas"] = [FALLBACK_BETA]
        req["fallbacks"] = "default"
    return req


# ---------------------------------------------------------------------------
# TODO 2 — 응답 검사: stop_reason 먼저, text 블록만, 폴백 여부
# ---------------------------------------------------------------------------
@dataclass
class ResponseView:
    text: str                  # text 블록만 이어 붙인 답
    stop_reason: str | None
    ok: bool                   # 정상 종료(end_turn, stop_sequence)인가
    served_by: str | None      # 실제로 답한 모델(message.model)
    fell_back: bool            # 서버 측 폴백 모델이 답했는가
    skipped_blocks: int        # 건너뛴 비-text 블록 수(thinking, fallback 표시 등)


def read_response(message) -> ResponseView:
    """Message 객체를 읽어 ResponseView 로. 예외를 내지 않는다(판단은 ensure_complete)."""
    # TODO 2
    # REF: claude-api 스킬 python/claude-api/README.md "Stop Reasons"·"Refusal Fallbacks", https://platform.claude.com/docs/en/api/messages/create
    # 개념: 응답 content 는 블록 목록이다. [thinking 블록, text 블록, ...]. 이 모델은 thinking 원문을 돌려주지 않아(빈 블록일 수 있음)
    #       사용자에게 보여 줄 답은 type == "text" 인 블록뿐이다. content[0].text 를 바로 읽으면 첫 블록이 thinking 일 때 깨진다.
    # 개념(stop_reason): 왜 멈췄나. end_turn(정상) · max_tokens(상한에서 잘림) · refusal(거절) · stop_sequence · tool_use · pause_turn.
    #       거절도 HTTP 200 으로 온다 → HTTP 성공만 보고 "답이 왔다"고 판단하면 안 된다. stop_reason 을 먼저 본다.
    # 개념(폴백 여부): usage.iterations 에 type == "fallback_message" 항목이 있고 최종 stop_reason 이 refusal 이 아니면
    #       폴백 모델이 답한 것이다. 답한 모델은 message.model.
    # 서비스 관점: 폴백·잘림·거절은 로그와 지표로 남긴다(Part 6 P6-04). 폴백 모델은 품질·스타일이 다를 수 있다.
    texts, skipped = [], 0
    for block in message.content:
        if getattr(block, "type", None) == "text":
            texts.append(block.text)
        else:
            skipped += 1
    iterations = getattr(message.usage, "iterations", None) or []
    fell_back = any(getattr(it, "type", None) == "fallback_message" for it in iterations) and message.stop_reason != "refusal"
    return ResponseView(
        text="".join(texts),
        stop_reason=message.stop_reason,
        ok=message.stop_reason in ("end_turn", "stop_sequence"),
        served_by=getattr(message, "model", None),
        fell_back=fell_back,
        skipped_blocks=skipped,
    )


def ensure_complete(message) -> ResponseView:
    """read_response 후 refusal 이면 RefusalError, max_tokens 이면 TruncatedError. 아니면 view 반환."""
    # TODO 2 (이어서)
    # WHY(category 는 참고만): stop_details.category("cyber", "bio", ...)는 없을 수도 있다. 분기는 stop_reason 으로만 한다.
    # WHY(잘린 답을 버리지 말고 담아 두기): 사용자에게 "답이 잘렸다"를 알리거나 이어 쓰기를 요청할 때 쓴다. 정상 답처럼 쓰면 안 된다.
    view = read_response(message)
    if view.stop_reason == "refusal":
        details = getattr(message, "stop_details", None)
        raise RefusalError(getattr(details, "category", None) if details else None)
    if view.stop_reason == "max_tokens":
        raise TruncatedError(view.text)
    return view


# ---------------------------------------------------------------------------
# TODO 3 — 구조화 출력: JSON 스키마로 triple 추출 (Part 4 로 연결)
# ---------------------------------------------------------------------------
class Triple(BaseModel):
    subject: str
    relation: str
    object: str


class Extraction(BaseModel):
    triples: list[Triple]


def with_json_schema(request: dict, schema_model: type[BaseModel]) -> dict:
    """request 의 output_config 에 format(json_schema)을 더한 새 사전. 기존 effort 는 유지한다."""
    # TODO 3
    # 개념: 구조화 출력 = "답을 이 JSON 스키마에 맞춰라"를 API 가 보장하게 하는 기능. 자유 문장 대신 프로그램이 바로 읽을 데이터를 받는다.
    #       {"triples": [{"subject": "세종대왕", "relation": "창제했다", "object": "훈민정음"}]}
    #       스키마는 Pydantic 모델에서 만들고, anthropic.transform_schema 로 API 가 받는 형태(additionalProperties: false 등)로 바꾼다.
    # WHY(새 사전): 원래 요청을 고치면 같은 요청 객체를 다른 곳에서 쓸 때 예기치 않게 바뀐다.
    # 서비스 관점: "JSON 으로 답해 줘"라고 프롬프트로만 부탁하면 가끔 설명 문장이 섞이거나 괄호가 빠져 파이프라인이 멈춘다.
    #       스키마를 강제하면 다음 단계(그래프 적재 — Part 4 P4-02/03)가 안정적이다. 그래도 거절·잘림은 생길 수 있어 TODO 2 검사가 먼저다.
    # REF: claude-api 스킬 — output_config={"format": {"type": "json_schema", "schema": ...}} (python/claude-api/sdk-upgrade.md,
    #      shared/model-migration.md "Prefill replacement"), 공식 문서 https://platform.claude.com/docs/en/build-with-claude/structured-outputs
    import anthropic

    out = dict(request)
    out["output_config"] = {
        **request.get("output_config", {}),
        "format": {"type": "json_schema", "schema": anthropic.transform_schema(schema_model.model_json_schema())},
    }
    return out


def parse_structured(message, schema_model: type[BaseModel]) -> BaseModel:
    """stop_reason 을 먼저 검사(ensure_complete)한 뒤 text 를 schema_model 로 검증한다. 검증 실패는 StructuredOutputError."""
    # TODO 3 (이어서)
    # REF: Pydantic 문서 "Models — model_validate_json" https://docs.pydantic.dev/latest/concepts/models/
    # WHY(순서): 거절·잘림이면 text 가 비었거나 JSON 이 중간에 끊겨 있다. 검증부터 하면 "JSON 오류"로 보여 진짜 원인(거절·잘림)을 놓친다.
    #       (SDK 의 messages.parse(output_format=...) 는 받자마자 검증해 이 순서를 지키기 어렵다 — README §4-2)
    # 서비스 관점: 검증 실패는 조용히 넘기지 말고 원문 일부(개인정보 주의)·오류를 기록하고, 재요청 또는 사람 검토로 보낸다.
    view = ensure_complete(message)
    try:
        return schema_model.model_validate_json(view.text)
    except ValidationError as e:
        raise StructuredOutputError(str(e)) from e


# ---------------------------------------------------------------------------
# TODO 4 — 스트리밍: text_stream 소비, TTFT, get_final_message
# ---------------------------------------------------------------------------
@dataclass
class StreamOutcome:
    text: str
    final: object            # get_final_message() 결과(usage·stop_reason·model)
    ttft_ms: float | None    # 요청 → 첫 텍스트 조각 (thinking 시간 포함!)
    total_ms: float


def consume_stream(stream_cm, start: float, clock: Callable[[], float] = time.perf_counter, on_text: Callable[[str], None] | None = None) -> StreamOutcome:
    """client.beta.messages.stream(**req) 가 돌려준 컨텍스트 매니저를 with 로 열어 텍스트를 모으고, 최종 메시지를 받는다."""
    # TODO 4
    # REF: claude-api 스킬 python/claude-api/streaming.md, https://platform.claude.com/docs/en/build-with-claude/streaming
    # 개념: 스트리밍은 답을 조각(text_delta)마다 받는다. stream.text_stream 은 텍스트 조각만 차례로 준다(thinking 조각은 빠진다).
    #       다 받은 뒤 stream.get_final_message() 로 완성된 메시지(usage·stop_reason·model)를 얻는다.
    #       with 블록을 써야 중간에 오류가 나도 연결이 닫힌다.
    # 개념(TTFT 에 thinking 포함): 이 모델은 답하기 전에 생각한다. 첫 "텍스트" 조각은 생각이 끝난 뒤에 오므로
    #       여기서 잰 TTFT = 네트워크 + 입력 처리(prefill) + thinking 시간. effort 가 높을수록 길다.
    # WHY(on_text): 받은 조각을 바로 화면에 내보내는 콜백(UI). 테스트에서는 None.
    # 서비스 관점: 긴 답일수록 스트리밍이 체감 대기를 줄인다. 타임아웃이 긴 요청(큰 max_tokens)은 스트리밍이 안전하다(연결이 오래 조용하지 않음).
    #       스트림 중간에 거절될 수도 있다 → 최종 메시지의 stop_reason 으로 다시 확인한다(TODO 2).
    parts: list[str] = []
    ttft_ms = None
    with stream_cm as stream:
        for piece in stream.text_stream:
            if ttft_ms is None and piece:
                ttft_ms = (clock() - start) * 1000
            parts.append(piece)
            if on_text is not None:
                on_text(piece)
        final = stream.get_final_message()
    return StreamOutcome("".join(parts), final, ttft_ms, (clock() - start) * 1000)


def to_usage(final, latency_ms: float, ttft_ms: float | None = None) -> Usage:
    """(제공) 최종 메시지 → P2-01B 의 Usage (log_usage 로 CSV 에 남기기 위해)."""
    return Usage("claude", getattr(final, "model", DEFAULT_MODEL), final.usage.input_tokens, final.usage.output_tokens,
                 latency_ms, ttft_ms, stop_reason=final.stop_reason)


# ---------------------------------------------------------------------------
# TODO 5 — 호출 전 토큰 세기로 비용 추정
# ---------------------------------------------------------------------------
@dataclass
class CostEstimate:
    input_tokens: int
    expected_usd: float | None   # expected_output_tokens 를 줬을 때
    max_usd: float               # 출력이 max_tokens 를 다 썼을 때(상한)


def estimate_cost(sdk, request: dict, expected_output_tokens: int | None = None, prices: dict = PRICES_PER_MTOK) -> CostEstimate:
    """sdk.messages.count_tokens 로 입력 토큰을 센 뒤 예상·최대 비용(USD)을 계산한다."""
    # TODO 5
    # REF: claude-api 스킬 shared/token-counting.md, https://platform.claude.com/docs/en/build-with-claude/token-counting ,
    #      가격 https://platform.claude.com/docs/en/about-claude/pricing (작성 시점 2026-10)
    # 개념: count_tokens 는 모델을 실행하지 않고 "이 요청의 입력이 몇 토큰인가"만 돌려준다(P1-01 count_with_claude.py).
    #       토큰 수는 모델(토크나이저)마다 다르므로 실제로 부를 모델 ID 를 넣는다. 다른 회사 토크나이저(tiktoken 등)로 세면 틀린다.
    #       system·output_config(스키마)도 입력 토큰에 들어가므로 같이 넘긴다. betas·fallbacks 는 세기와 무관해 넘기지 않는다.
    # 개념(상한): 출력은 실행 전에 알 수 없지만 max_tokens 를 넘지 않는다 → 최대 비용 = 입력 × 입력 단가 + max_tokens × 출력 단가.
    #       thinking 토큰도 출력 토큰으로 과금된다(usage.output_tokens 에 포함).
    # 서비스 관점: 요청 전에 상한을 계산해 두면 "한 요청이 예산을 넘지 않게" 막을 수 있다(긴 문서는 거절·요약·분할).
    model = request["model"]
    if model not in prices:
        raise KeyError(f"unknown model price: {model!r}")
    kwargs = {"model": model, "messages": request["messages"]}
    for key in ("system", "output_config"):
        if key in request:
            kwargs[key] = request[key]
    n_in = sdk.messages.count_tokens(**kwargs).input_tokens
    p_in, p_out = prices[model]
    expected = None if expected_output_tokens is None else (n_in * p_in + expected_output_tokens * p_out) / 1e6
    return CostEstimate(n_in, expected, (n_in * p_in + request["max_tokens"] * p_out) / 1e6)


# ---------------------------------------------------------------------------
# TODO 6 (선택) — 오류 처리 체인: 구체적인 예외부터
# ---------------------------------------------------------------------------
@dataclass
class ErrorDecision:
    kind: str                     # rate_limit | server | transient | client | connection | unknown
    retry: bool                   # 잠시 뒤 다시(또는 다른 공급자로) 시도할 가치가 있는가
    retry_after_s: float | None = None


def classify_error(exc: BaseException) -> ErrorDecision:
    """SDK 예외를 종류별로 나눈다. 하위 클래스가 상위 클래스보다 먼저 검사되어야 한다."""
    # TODO 6
    # REF: claude-api 스킬 python/claude-api/README.md "Error Handling"·"Retries", https://platform.claude.com/docs/en/api/errors
    # 개념(예외 계층): RateLimitError(429)·InternalServerError(5xx, 과부하 529 포함)·BadRequestError(400) ... ⊂ APIStatusError ⊂ APIError
    #       APITimeoutError ⊂ APIConnectionError ⊂ APIError   (설치된 anthropic 2.6.0 에서 확인)
    #       isinstance 로 위에서부터 검사하므로, APIStatusError 를 먼저 보면 RateLimitError 도 거기 걸려 "429 인지" 를 놓친다.
    # 개념(재시도 vs 즉시 실패): 429·5xx·연결 오류는 시간이 지나면 나을 수 있다(재시도·폴백). 400·401·403·404 는 요청·설정 문제라 다시 보내도 같다.
    # WHY(retry-after): 429 응답 헤더가 "몇 초 뒤에 다시"를 알려 주면 그만큼 기다린다(무작정 바로 재시도하면 또 걸린다).
    # 서비스 관점: SDK 는 연결 오류·408·409·429·5xx 를 max_retries 만큼 스스로 재시도한 뒤 예외를 올린다.
    #       여기서의 판단은 "그 뒤" — 사용자에게 잠시 후 안내, 다른 공급자로 폴백(P2-01B), 또는 즉시 오류 보고.
    import anthropic

    if isinstance(exc, anthropic.RateLimitError):
        header = exc.response.headers.get("retry-after") if getattr(exc, "response", None) is not None else None
        try:
            after = float(header) if header is not None else None
        except ValueError:
            after = None
        return ErrorDecision("rate_limit", True, after)
    if isinstance(exc, anthropic.InternalServerError):
        return ErrorDecision("server", True)
    if isinstance(exc, anthropic.APIStatusError) and exc.status_code in (408, 409):
        # 408(요청 시간 초과)·409(충돌)도 SDK 가 자동 재시도하는 일시 오류다 → 위 주석과 같게 재시도 가능으로 본다.
        # (여기 도착했다면 SDK 재시도는 이미 실패한 것 — 잠시 뒤 다시 하거나 다른 공급자로 넘기는 것은 호출부가 정한다)
        return ErrorDecision("transient", True)
    if isinstance(exc, anthropic.APIStatusError):
        return ErrorDecision("client", False)
    if isinstance(exc, anthropic.APIConnectionError):
        return ErrorDecision("connection", True)
    return ErrorDecision("unknown", False)


# ---------------------------------------------------------------------------
# 단독 실행: 어느 TODO 가 남았는지 알려 준다.
# ---------------------------------------------------------------------------
def _report_progress() -> None:
    from types import SimpleNamespace as NS

    msg = NS(content=[NS(type="text", text='{"triples": []}')], stop_reason="end_turn", model=DEFAULT_MODEL,
             usage=NS(input_tokens=1, output_tokens=1, iterations=None))

    class _S:
        text_stream = iter(["a"])

        def __enter__(self):
            return self

        def __exit__(self, *e):
            return False

        def get_final_message(self):
            return msg

    sdk = NS(messages=NS(count_tokens=lambda **k: NS(input_tokens=10)))
    checks = [
        ("TODO 1 build_messages / build_request", lambda: build_request(build_messages([], "hi"))),
        ("TODO 2 read_response / ensure_complete", lambda: ensure_complete(msg)),
        ("TODO 3 with_json_schema / parse_structured", lambda: (with_json_schema({"model": "m"}, Extraction), parse_structured(msg, Extraction))),
        ("TODO 4 consume_stream", lambda: consume_stream(_S(), 0.0, lambda: 0.1)),
        ("TODO 5 estimate_cost", lambda: estimate_cost(sdk, {"model": DEFAULT_MODEL, "messages": [], "max_tokens": 10})),
        ("TODO 6 classify_error (선택)", lambda: classify_error(ValueError("x"))),
    ]
    remaining = 0
    for name, fn in checks:
        try:
            fn()
            print(f"[완료]   {name}")
        except NotImplementedError:
            remaining += 1
            print(f"[미구현] {name}")
    if remaining:
        print(f"\n남은 TODO {remaining}개 — 위에서부터 순서대로 채우세요(TODO 3 은 TODO 2 를 사용합니다). 채운 뒤 검사:")
        print("  (bash)       IMPL=starter pytest part2_llm/llm_api/basics/tests")
        print('  (PowerShell) $env:IMPL="starter"; pytest part2_llm/llm_api/basics/tests; Remove-Item Env:IMPL')
    else:
        print("\n모든 TODO 가 구현되었습니다. 이제 pytest 로 정답 여부를 확인하세요.")


if __name__ == "__main__":
    _report_progress()

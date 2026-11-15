"""P2-02 Claude API 기초 — 실습 코드 (TODO 를 채우세요). 메인 실습: 개념은 깊게, 구현은 얕게.

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
검사: IMPL=starter pytest part2_llm/llm_api/basics/tests
진행 확인: python part2_llm/llm_api/basics/starter.py   (남은 TODO 목록 출력)
"""
from __future__ import annotations

import importlib.util
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ValidationError  # noqa: F401  (ValidationError 는 TODO 3 에서 사용)

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
    # TODO 1: history 의 (role, text) 를 {"role", "content"} 사전으로 바꾸고 끝에 {"role": "user", "content": user_text} 를 붙이세요.
    #         i 번째 메시지의 역할이 짝수면 "user", 홀수면 "assistant" 가 아니면 ValueError.
    # 개념: Claude API 의 대화는 상태가 없다. 매 요청에 지금까지의 대화 전체를 다시 보낸다.
    #       [user "연차 신청은 며칠 전까지?", assistant "(규정 답)", user "반차도 같아?"] ← 앞 대화가 마지막 질문의 문맥.
    # HINT 1: 먼저 목록을 만들고, 그다음 한 번 훑으며 역할 순서를 검사하세요.
    # HINT 2: enumerate 로 각 메시지의 위치(i)를 함께 받고, 위치의 짝·홀로 기대하는 역할을 정해 비교하세요.
    # WHY(마지막은 user): assistant 로 끝내 답의 앞부분을 채우는 prefill 은 이 모델에서 400. 형식 강제는 구조화 출력(TODO 3).
    # 서비스 관점: 대화가 길수록 입력 토큰(비용·TTFT)이 는다 → 요약·잘라내기·캐싱(P2-03).
    # REF: claude-api 스킬 python/claude-api/README.md "Multi-Turn Conversations", https://platform.claude.com/docs/en/api/messages/create
    raise NotImplementedError("TODO 1")


def build_request(
    messages: list[dict],
    system: str | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    effort: str = "medium",
    fallbacks: bool = True,
) -> dict:
    """client.beta.messages.create(**req) / .stream(**req) 에 그대로 넘길 인자 사전."""
    # TODO 1 (이어서): effort 가 EFFORTS 에 없거나 max_tokens <= 0 이거나 messages 가 비었거나 마지막이 user 가 아니면 ValueError.
    #         {"model", "max_tokens", "messages", "output_config": {"effort": effort}} 를 만들고,
    #         system 이 있으면 "system" 을, fallbacks 가 True 면 "betas": [FALLBACK_BETA] 와 "fallbacks": "default" 를 더하세요.
    # 개념: system = 역할·규칙·출력 형식(모든 턴 공통), messages = 대화 내용. thinking 은 끌 수 없고 깊이는 effort(기본 medium, 명시).
    # HINT 1: 필수 키로 사전을 먼저 만들고, 조건에 따라 키를 추가하세요.
    # HINT 2: 폴백은 두 키가 짝입니다 — 베타 목록 키(상수 FALLBACK_BETA)와 폴백 모드 키(문자열 값). 형식은 README §4-3.
    # WHY(max_tokens 16000): thinking + 답변을 합친 출력 상한. 작으면 생각만 하다 잘린다. 동시에 한 요청의 최대 비용(TODO 5).
    # WHY(fallbacks="default"): 정책 거절 시 같은 호출 안에서 권장 모델로 재실행. rate limit·과부하·서버 오류는 폴백하지 않는다.
    # 서비스 관점: 바뀌지 않는 지시만 system 에. 현재 시각·사용자 이름을 넣으면 캐싱이 깨진다(P2-03).
    # REF: claude-api 스킬 python/claude-api/README.md "System Prompts"·"Extended Thinking"(effort), shared/model-migration.md (fallbacks "default")
    raise NotImplementedError("TODO 1")


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
    # TODO 2: message.content 에서 type == "text" 인 블록의 text 만 이어 붙이고, 나머지 블록 수를 세세요.
    #         ok = stop_reason 이 "end_turn" 또는 "stop_sequence", served_by = message.model,
    #         fell_back = usage.iterations(없으면 빈 목록) 중 type == "fallback_message" 가 있고 stop_reason != "refusal".
    #         ResponseView 로 반환(예외 없음).
    # 개념: content 는 블록 목록 [thinking, text, ...]. 사용자에게 보일 답은 text 블록뿐 — content[0].text 를 바로 읽으면 깨진다.
    #       거절도 HTTP 200 이다 → stop_reason 을 먼저 본다.
    # HINT 1: 블록·iteration 은 속성이 없을 수도 있다고 가정하고 안전하게 읽으세요.
    # HINT 2: getattr(객체, "이름", 기본값) 은 속성이 없을 때 기본값을 돌려줍니다. iterations 는 None 일 수 있습니다.
    # WHY(iterations 로 판단): sticky routing 으로 폴백 모델이 바로 답한 턴에는 fallback 블록이 없다. usage.iterations 가 기준이다.
    # 서비스 관점: 폴백·잘림·거절은 지표로 남긴다(Part 6 P6-04).
    # REF: claude-api 스킬 python/claude-api/README.md "Stop Reasons"·"Refusal Fallbacks", https://platform.claude.com/docs/en/api/messages/create
    raise NotImplementedError("TODO 2")


def ensure_complete(message) -> ResponseView:
    """read_response 후 refusal 이면 RefusalError, max_tokens 이면 TruncatedError. 아니면 view 반환."""
    # TODO 2 (이어서): read_response 결과의 stop_reason 이 "refusal" 이면 RefusalError(stop_details.category 또는 None),
    #         "max_tokens" 이면 TruncatedError(view.text). 아니면 view 를 반환하세요.
    # HINT: stop_details 자체가 없을 수도 있으니(위 getattr 방법) 안전하게 읽은 뒤 category 를 꺼내세요.
    # WHY(category 는 참고만): stop_details 는 없을 수 있다. 분기는 stop_reason 으로만.
    raise NotImplementedError("TODO 2")


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
    # TODO 3: request 를 얕게 복사하고, output_config 를 {기존 output_config 내용, "format": {"type": "json_schema", "schema": ...}} 로 바꾸세요.
    #         schema = anthropic.transform_schema(schema_model.model_json_schema())  (원래 request 는 바꾸지 않음)
    # 개념: 구조화 출력 = 답이 JSON 스키마를 따르도록 API 가 보장. {"triples": [{"subject": .., "relation": .., "object": ..}]}
    # HINT 1: 사전 펼치기 {**a, "key": v} 로 effort 를 잃지 않게 합치세요.
    # HINT 2: 스키마 변환 함수는 anthropic 패키지 최상위에 있습니다(transform_schema). 얕은 복사는 dict(...).
    # 서비스 관점: "JSON 으로 답해"라는 부탁만으로는 가끔 문장이 섞여 파이프라인이 멈춘다. 스키마 강제가 다음 단계(Part 4)를 안정시킨다.
    # REF: claude-api 스킬 output_config.format (python/claude-api/sdk-upgrade.md), https://platform.claude.com/docs/en/build-with-claude/structured-outputs
    raise NotImplementedError("TODO 3")


def parse_structured(message, schema_model: type[BaseModel]) -> BaseModel:
    """stop_reason 을 먼저 검사(ensure_complete)한 뒤 text 를 schema_model 로 검증한다. 검증 실패는 StructuredOutputError."""
    # TODO 3 (이어서): ensure_complete(message) 로 먼저 거절·잘림을 걸러 내고, view.text 를 schema_model.model_validate_json 으로 검증하세요.
    #         pydantic ValidationError 는 StructuredOutputError 로 바꿔 올립니다(원인 연결: raise ... from e).
    # WHY(순서): 거절·잘림이면 JSON 이 비었거나 끊겨 있다. 검증부터 하면 진짜 원인을 놓친다.
    # HINT: Pydantic 모델의 JSON 문자열 검증 메서드는 model_validate_json 입니다. 예외 연결은 raise ... from e.
    # 서비스 관점: 검증 실패는 기록하고 재요청 또는 사람 검토로 보낸다.
    # REF: Pydantic 문서 "Models — model_validate_json" https://docs.pydantic.dev/latest/concepts/models/
    raise NotImplementedError("TODO 3")


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
    # TODO 4: with stream_cm as stream: 으로 열고 stream.text_stream 의 조각을 모으세요. 내용이 있는 첫 조각에서
    #         ttft_ms = (clock() - start) × 1000, on_text 가 있으면 조각마다 호출. 반복이 끝나면 final = stream.get_final_message().
    #         StreamOutcome(text, final, ttft_ms, total_ms=(clock() - start) × 1000) 반환.
    # 개념: text_stream 은 텍스트 조각만 준다(thinking 제외). 최종 usage·stop_reason 은 get_final_message() 에서.
    #       TTFT 에는 thinking 시간이 포함된다(첫 텍스트는 생각이 끝난 뒤).
    # HINT 1: with 블록 안에서 반복과 get_final_message 를 모두 하세요(블록을 나오면 연결이 닫힘).
    # HINT 2: with 로 연 stream 객체에는 반복 가능한 text_stream 속성과 get_final_message() 메서드가 있습니다.
    # WHY(with 블록 안에서): 블록을 나오면 연결이 닫혀 최종 메시지를 받을 수 없다. 오류가 나도 with 가 연결을 정리한다.
    # WHY(빈 조각은 TTFT 아님): 내용 없는 조각이 먼저 올 수 있다. 사용자가 "보기 시작한" 시각을 재야 한다.
    # 서비스 관점: 긴 답·큰 max_tokens 는 스트리밍이 체감 대기와 타임아웃 위험을 줄인다. 중간 거절은 최종 stop_reason 으로 확인.
    # REF: claude-api 스킬 python/claude-api/streaming.md, https://platform.claude.com/docs/en/build-with-claude/streaming
    raise NotImplementedError("TODO 4")


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
    # TODO 5: model 이 prices 에 없으면 KeyError. sdk.messages.count_tokens(model=..., messages=..., [system], [output_config])
    #         의 input_tokens 로 예상 비용(expected_output_tokens 가 있을 때만)과 최대 비용(출력 = max_tokens)을 USD 로 계산하세요.
    # 개념: count_tokens 는 모델을 실행하지 않고 입력 토큰 수만 센다. 토큰 수는 모델마다 다르니 실제 모델 ID 로.
    #       최대 비용 = (입력 × 입력 단가 + max_tokens × 출력 단가) / 1,000,000. thinking 도 출력 토큰으로 과금된다.
    # HINT 1: request 에서 세기에 필요한 키만 골라 넘기세요(betas·fallbacks 는 넣지 않음).
    # HINT 2: system·output_config 는 요청에 "있을 때만" 넘깁니다(in 으로 확인). 결과 객체의 input_tokens 를 읽습니다.
    # WHY(실제 모델 ID): 토큰 수는 모델(토크나이저)마다 다르다. 다른 토크나이저로 세면 Claude 토큰을 잘못 센다.
    # WHY(상한 = max_tokens): 출력은 실행 전에 모르지만 max_tokens 를 넘지 않는다 → 최대 비용을 미리 막을 수 있다.
    # 서비스 관점: 요청 전에 상한을 계산해 두면 한 요청이 예산을 넘지 않게 막을 수 있다.
    # REF: claude-api 스킬 shared/token-counting.md, https://platform.claude.com/docs/en/build-with-claude/token-counting ,
    #      가격 https://platform.claude.com/docs/en/about-claude/pricing (작성 시점 2026-10)
    raise NotImplementedError("TODO 5")


# ---------------------------------------------------------------------------
# TODO 6 (선택) — 오류 처리 체인: 구체적인 예외부터
# ---------------------------------------------------------------------------
@dataclass
class ErrorDecision:
    kind: str                     # rate_limit | server | client | connection | unknown
    retry: bool                   # 잠시 뒤 다시(또는 다른 공급자로) 시도할 가치가 있는가
    retry_after_s: float | None = None


def classify_error(exc: BaseException) -> ErrorDecision:
    """SDK 예외를 종류별로 나눈다. 하위 클래스가 상위 클래스보다 먼저 검사되어야 한다."""
    # TODO 6 (선택): 아래 순서로 isinstance 검사해 ErrorDecision 을 반환하세요.
    #         RateLimitError → ("rate_limit", True, retry-after 헤더를 float 로, 없거나 숫자가 아니면 None)
    #         InternalServerError → ("server", True) / APIStatusError 이고 status_code 가 408·409 → ("transient", True)
    #         / APIStatusError(그 밖의 4xx) → ("client", False)
    #         APIConnectionError(시간 초과 포함) → ("connection", True) / 그 밖 → ("unknown", False)
    # 개념: RateLimitError·InternalServerError ⊂ APIStatusError 이다. 상위 클래스를 먼저 검사하면 하위 클래스 분기에 도달하지 못한다.
    # HINT 1: 가장 구체적인(하위) 예외부터 검사하세요.
    # HINT 2: 429 예외 객체의 response.headers 에서 "retry-after" 를 읽습니다(없거나 숫자가 아닐 수 있음).
    # WHY(순서): 상위 클래스(APIStatusError)를 먼저 검사하면 RateLimitError 의 "몇 초 뒤" 정보를 잃는다.
    # WHY(408·409 재시도 가능): SDK 도 자동 재시도하는 일시 오류다. 여기 왔다면 SDK 재시도는 이미 실패 — 다음 행동은 호출부가 정한다.
    # 서비스 관점: SDK 가 이미 max_retries 만큼 재시도한 "뒤"의 판단 — 안내·폴백(P2-01B)·즉시 오류.
    # REF: claude-api 스킬 python/claude-api/README.md "Error Handling"·"Retries", https://platform.claude.com/docs/en/api/errors
    raise NotImplementedError("TODO 6")


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

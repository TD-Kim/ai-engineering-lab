"""P2-03 비용·지연 — 직접 채우는 코드(starter) (메인 실습: 개념은 깊게, 구현은 얕게).

"같은 품질에서 비용·지연을 줄이는 설계"를 코드로 확인한다.
  TODO 1 캐시 친화적 요청 구성(안정적인 앞부분 → 뒤에 바뀌는 질문, cache_control 위치, breakpoint 최대 4개)
  TODO 2 응답 usage 에서 캐시 쓰기/읽기 토큰을 읽고 캐시 적중 판정·비용·절감액 계산
  TODO 3 두 요청을 비교해 공통 접두어 길이와 첫 차이 위치 찾기(캐시가 깨지는 원인 점검)
  TODO 4 Batches API 요청 구성(custom_id) + 결과를 custom_id 로 맞추기(순서 무관)
  TODO 5 사용량 로그(CSV) 집계: 요청 수·토큰·캐시 적중률·평균/p95 지연·총비용
  TODO 6 (선택) "언제 무엇을 쓰나" — 캐싱·배치·effort·max_tokens 의사결정 규칙

SDK 형태·단가는 claude-api 스킬과 공식 문서로 확인했다(작성 시점 2026-11-16, README §4-1 "근거" 표).
P2-02 의 build_request·read_response·Usage·log_usage 관련 이름을 경로 로딩으로 재사용한다(복사하지 않음).

테스트·데모는 가짜 SDK 객체로 돈다(키·네트워크 불필요). 실제 호출은 measure_caching.py (선택, 유료).
진행 확인: python part2_llm/llm_api/cost_latency/solution.py
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math  # noqa: F401  (TODO 5 에서 math.ceil 사용)
import re
import sys
from collections import Counter  # noqa: F401  (TODO 5 에서 합계·개수 세기에 사용)
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# (제공) P2-02 재사용: build_request, read_response, Usage, USAGE_FIELDS(P2-01B 로그 열)
# ---------------------------------------------------------------------------
_P2_02 = Path(__file__).resolve().parents[1] / "basics" / "solution.py"


def _load_p2_02():
    name = "p2_02_llm_api_basics_for_p2_03"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _P2_02)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclass 가 자기 모듈을 찾을 수 있게 먼저 등록
    spec.loader.exec_module(mod)
    return mod


_p2_02 = _load_p2_02()
build_request = _p2_02.build_request
read_response = _p2_02.read_response
Usage = _p2_02.Usage
DEFAULT_MODEL = _p2_02.DEFAULT_MODEL
USAGE_FIELDS = _p2_02._p2_01b.USAGE_FIELDS     # P2-01B log_usage 의 열(timestamp, provider, model, ...)

MAX_BREAKPOINTS = 4                 # 한 요청에 cache_control 은 최대 4개(스킬 shared/prompt-caching.md API reference)
MIN_CACHEABLE_TOKENS = 512          # claude-opus-5-5 의 최소 캐시 길이(공식 prompt-caching 문서, 2026-11-16 확인). 모델마다 다르다
BATCH_DISCOUNT = 0.5                # Batches API: 입력·출력 모두 50%
CUSTOM_ID_RE = re.compile(r"[a-zA-Z0-9_-]{1,64}")     # 공식 문서 ^[a-zA-Z0-9_-]{1,64}$ — fullmatch 로 검사(아래 WHY)
BATCH_FORBIDDEN_KEYS = ("stream", "speed", "fallbacks", "betas")

# 단가(USD / 100만 토큰). 공식 가격 페이지에서 확인: 2026-11-16.
# REF: https://platform.claude.com/docs/en/about-claude/pricing (사용 전 재확인 — 단가는 바뀐다)
#   input = 기본 입력, write_5m = 5분 캐시 쓰기(1.25×), write_1h = 1시간 캐시 쓰기(2×),
#   read = 캐시 읽기(이 두 모델은 0.05×), output = 출력
PRICES = {
    "claude-opus-5-5": {"input": 4.00, "write_5m": 5.00, "write_1h": 8.00, "read": 0.20, "output": 20.00},
    "claude-sonnet-5-5": {"input": 2.00, "write_5m": 2.50, "write_1h": 4.00, "read": 0.10, "output": 10.00},
}


# ---------------------------------------------------------------------------
# TODO 1 — 캐시 친화적 요청: 안정적인 앞부분 → 뒤에 바뀌는 질문
# ---------------------------------------------------------------------------
def cache_marker(ttl: str = "5m") -> dict:
    """cache_control 값. "5m"(기본) → {"type": "ephemeral"}, "1h" → {"type": "ephemeral", "ttl": "1h"}. 그 밖은 ValueError."""
    # TODO 1: ttl 이 "5m" 이면 {"type": "ephemeral"}, "1h" 이면 {"type": "ephemeral", "ttl": "1h"} 를 돌려주세요. 그 밖은 ValueError.
    # HINT 1: 기본 5분은 ttl 키를 쓰지 않습니다(생략 = 5분). 1시간만 ttl 을 적습니다.
    # HINT 2: if/if/raise 세 줄이면 됩니다.
    # REF: claude-api 스킬 python/claude-api/README.md "Manual Cache Control", shared/prompt-caching.md "API reference",
    #      https://platform.claude.com/docs/en/build-with-claude/prompt-caching
    # 개념: 캐시 수명(TTL)은 기본 5분, 선택 1시간. 캐시를 읽으면 수명이 다시 처음부터 센다(무료 갱신).
    #       1시간은 쓰기 단가가 2배(5분은 1.25배)라 "요청 간격이 5~60분"일 때만 이득이다.
    raise NotImplementedError("TODO 1")


def build_cached_request(instructions: str, documents: Sequence[str], question: str, ttl: str = "5m", **request_kwargs) -> dict:
    """system = [지시문, 문서1, 문서2, ...] (마지막 system 블록에 cache_control), messages = [질문].

    request_kwargs 는 P2-02 build_request 로 그대로 넘긴다(model, max_tokens, effort, fallbacks).
    instructions 가 비었거나 documents 가 없으면 ValueError.
    """
    # TODO 1 (이어서): system 을 [지시문 블록, 문서 블록들] 로 만들고 "마지막" system 블록에만 cache_marker(ttl) 를 붙이세요.
    #         질문은 P2-02 build_request([{"role": "user", "content": question}], **request_kwargs) 의 messages 로 보내고,
    #         그 결과의 system 을 위 블록 목록으로 바꿔 돌려줍니다. instructions 가 비었거나 documents 가 없으면 ValueError.
    # HINT 1: 텍스트 블록 하나는 {"type": "text", "text": ...} 입니다. 입력 목록(documents)을 고치지 말고 새 목록을 만드세요.
    # HINT 2: 블록 목록을 다 만든 뒤 system[-1]["cache_control"] = ... 처럼 마지막 블록에만 표시합니다.
    # 개념(접두어 일치): 캐시는 "요청 맨 앞부터 breakpoint 까지의 바이트"가 전과 똑같을 때만 읽힌다.
    #       그려지는 순서는 tools → system → messages. 그래서 바뀌지 않는 것(지시문·문서)을 앞에, 바뀌는 것(질문)을 뒤에 둔다.
    #       요청 A: [지시문][문서들]★ + "연차는 며칠 전까지?"    요청 B: [지시문][문서들]★ + "반차도 같아?"
    #       ★ = cache_control. ★ 까지 같으므로 B 는 A 가 써 둔 캐시를 읽는다.
    # WHY(마지막 system 블록에 하나): breakpoint 에서 "그 앞 전부"가 캐시된다. 질문 블록에 두면 질문마다 다른 캐시를 써서
    #       아무것도 읽히지 않는다(쓰기 비용만 1.25배로 낸다).
    # WHY(질문은 messages): system 에 질문을 섞으면 앞부분이 매번 달라진다. 사용자 입력은 항상 맨 뒤.
    # 서비스 관점(RAG, Part 3): 검색된 문서가 질문마다 다르면 그 문서는 캐시해도 소용없다. 캐시할 것은
    #       "모든 질문에 공통인 것" — 시스템 프롬프트, 고정 규정집, few-shot 예시. 질문별 문서는 질문 쪽(뒤)에 둔다.
    raise NotImplementedError("TODO 1")


def count_breakpoints(request: dict) -> int:
    """(제공) 요청 안의 cache_control 개수(top-level 자동 캐싱 포함). 4개를 넘으면 API 가 400 을 낸다."""
    n = 1 if "cache_control" in request else 0
    for key in ("tools", "system"):
        value = request.get(key)
        if isinstance(value, list):
            n += sum("cache_control" in b for b in value if isinstance(b, dict))
    for m in request.get("messages", []):
        if isinstance(m.get("content"), list):
            n += sum("cache_control" in b for b in m["content"] if isinstance(b, dict))
    return n


# ---------------------------------------------------------------------------
# TODO 2 — usage 의 캐시 토큰 읽기 · 적중 판정 · 비용과 절감액
# ---------------------------------------------------------------------------
@dataclass
class CacheUsage:
    input_tokens: int          # 캐시와 무관하게 정가로 처리한 입력(마지막 breakpoint 뒤 + 캐시 안 된 부분)
    cache_write_5m: int        # 이번 요청에 5분 캐시로 쓴 토큰
    cache_write_1h: int        # 이번 요청에 1시간 캐시로 쓴 토큰
    cache_read: int            # 캐시에서 읽은 토큰
    output_tokens: int

    @property
    def prompt_tokens(self) -> int:
        """입력 전체 = 정가 입력 + 캐시 쓰기 + 캐시 읽기."""
        return self.input_tokens + self.cache_write_5m + self.cache_write_1h + self.cache_read

    @property
    def status(self) -> str:
        """"hit"(읽음) · "write"(안 읽고 쓰기만) · "none"(캐시 미사용 또는 최소 길이 미달)."""
        if self.cache_read > 0:
            return "hit"
        if self.cache_write_5m + self.cache_write_1h > 0:
            return "write"
        return "none"


def read_cache_usage(usage, default_ttl: str = "5m") -> CacheUsage:
    """SDK usage 객체 → CacheUsage. None 인 필드는 0. 쓰기의 TTL 구분은 usage.cache_creation 이 있으면 그것을, 없으면 default_ttl."""
    # TODO 2: usage 의 input_tokens·output_tokens·cache_creation_input_tokens·cache_read_input_tokens 를 읽어 CacheUsage 로.
    #         None 이거나 없는 필드는 0. 쓰기 토큰의 TTL 구분은 usage.cache_creation(있으면)의 ephemeral_5m_input_tokens /
    #         ephemeral_1h_input_tokens 를 쓰고, 없으면 쓰기 전부를 default_ttl 쪽에 넣습니다. default_ttl 이 5m·1h 가 아니면 ValueError.
    # HINT 1: 없을 수도 있는 속성은 getattr(obj, "이름", None) 으로 읽고, `값 or 0` 으로 None 을 0 으로 바꿉니다.
    # HINT 2: CacheUsage(input, write_5m, write_1h, read, output) 순서입니다.
    # REF: claude-api 스킬 shared/prompt-caching.md "Verifying cache hits", 설치된 anthropic 2.6.0 의 types/usage.py·cache_creation.py
    # 개념: usage 의 입력 토큰은 세 칸으로 나뉜다.
    #       input_tokens                = 정가로 처리한 나머지(캐시 제외!)
    #       cache_creation_input_tokens = 이번에 캐시에 "쓴" 토큰 (5분 1.25배, 1시간 2배)
    #       cache_read_input_tokens     = 캐시에서 "읽은" 토큰 (이 모델은 0.05배)
    #       입력 전체 = 셋의 합. input_tokens 만 보고 "입력이 4천 토큰뿐이네"라고 하면 틀린다.
    # 개념(TTL 구분): usage.cache_creation.ephemeral_5m_input_tokens / ephemeral_1h_input_tokens 가 쓰기를 TTL 별로 나눠 준다.
    # WHY(None → 0): 캐시를 쓰지 않은 요청·가짜 객체·오래된 로그에서는 이 필드가 None 이거나 아예 없다.
    # 서비스 관점: 캐시 적중 여부는 코드가 아니라 이 숫자로만 알 수 있다. 적중률을 지표로 계속 본다(TODO 5).
    raise NotImplementedError("TODO 2")


@dataclass
class CostBreakdown:
    actual_usd: float            # 캐시를 쓴 실제 비용
    no_cache_usd: float          # 같은 토큰을 캐시 없이 냈다면(입력 전체 × 기본 입력 단가)
    saved_usd: float             # no_cache - actual (음수면 손해: 쓰기만 하고 못 읽음)


def cache_cost(cu: CacheUsage, model: str = DEFAULT_MODEL, batch: bool = False, prices: dict = PRICES) -> CostBreakdown:
    """토큰 칸별로 단가를 곱해 실제 비용과 '캐시 없었다면' 비용을 계산한다. batch=True 면 둘 다 × BATCH_DISCOUNT."""
    # TODO 2 (이어서): 칸마다 단가를 곱해 actual(캐시 있음)과 no_cache(입력 전체 × 기본 입력 단가)를 USD 로 계산하고,
    #         saved = no_cache - actual 을 담아 CostBreakdown 으로. batch=True 면 actual·no_cache 둘 다 × BATCH_DISCOUNT.
    #         model 이 prices 에 없으면 KeyError.
    # HINT 1: prices[model] 은 {"input", "write_5m", "write_1h", "read", "output"} 단가 사전입니다(USD / 100만 토큰 → 마지막에 / 1e6).
    # HINT 2: 캐시가 없었다면 cu.prompt_tokens 전부가 기본 입력 단가였을 것입니다.
    # REF: https://platform.claude.com/docs/en/about-claude/pricing (2026-11-16 확인: 캐시 배수는 배치 할인과 겹쳐 적용된다)
    # 개념(작은 숫자 예, claude-opus-5-5): 문서 10,000 토큰 + 질문 50 토큰, 출력 200 토큰
    #       1번째 요청(쓰기): 50×$4 + 10,000×$5 + 200×$20 = 54,200 → $0.0542  (캐시 없으면 10,050×$4 + 200×$20 → $0.0442)
    #       2번째 요청(읽기): 50×$4 + 10,000×$0.20 + 200×$20 = 6,200 → $0.0062  (캐시 없으면 $0.0442)
    #       두 번 합계 $0.0604 vs $0.0884 — 2번째 요청부터 이득(5분 TTL 의 손익분기 = 2회).  (단위: 토큰×$/100만)
    # WHY(no_cache 를 같이): "캐시 덕에 얼마 아꼈나"는 기준이 있어야 말할 수 있다. 쓰기만 하고 못 읽으면 saved 가 음수다.
    # 서비스 관점: saved 가 계속 음수면 캐시 설정이 손해다(앞부분이 매번 바뀌거나 요청 간격 > TTL). TODO 3 으로 원인을 찾는다.
    raise NotImplementedError("TODO 2")


# ---------------------------------------------------------------------------
# TODO 3 — 캐시가 깨지는 곳 찾기: 두 요청의 공통 접두어와 첫 차이
# ---------------------------------------------------------------------------
def render_segments(request: dict) -> list[tuple[str, str]]:
    """(제공) 요청을 API 가 그리는 순서(model → tools → system → messages)의 (위치, 문자열) 목록으로 펼친다.

    cache_control 표시는 뺀다(표시 위치가 달라도 캐시 무효의 원인이 아니다). dict 는 들어온 키 순서 그대로 직렬화한다
    — SDK 도 그렇게 보내므로, 키 순서가 다르면 바이트가 다르다(정렬하면 이 원인을 숨긴다).
    """
    def strip(block):
        return {k: v for k, v in block.items() if k != "cache_control"} if isinstance(block, dict) else block

    def text_of(block) -> str:
        block = strip(block)
        if isinstance(block, dict) and block.get("type") == "text" and set(block) <= {"type", "text"}:
            return block["text"]
        return json.dumps(block, ensure_ascii=False)

    segs = [("model", request.get("model", ""))]
    for i, tool in enumerate(request.get("tools", []) or []):
        segs.append((f"tools[{i}]", text_of(tool)))
    system = request.get("system")
    if isinstance(system, str):
        segs.append(("system", system))
    elif system:
        segs += [(f"system[{i}]", text_of(b)) for i, b in enumerate(system)]
    for i, m in enumerate(request.get("messages", [])):
        content = m.get("content")
        if isinstance(content, str):
            segs.append((f"messages[{i}].{m.get('role')}", content))
        else:
            segs += [(f"messages[{i}].{m.get('role')}[{j}]", text_of(b)) for j, b in enumerate(content or [])]
    return segs


SETTINGS_KEYS = ("output_config", "thinking", "tool_choice", "speed")


@dataclass
class PrefixDiff:
    common_chars: int               # 앞에서부터 같은 글자 수(위치 표시 문자열은 빼고 내용만 센다)
    location: str | None            # 처음 달라진 세그먼트 위치(예: "system[0]"). 끝까지 같으면 None
    offset: int | None              # 그 세그먼트 안에서 몇 번째 글자부터 다른가
    a_snippet: str = ""             # 차이 지점부터 a 쪽 최대 30자
    b_snippet: str = ""
    settings_changed: list[str] = field(default_factory=list)   # 값이 달라진 설정 키(SETTINGS_KEYS 중)


def first_divergence(a: dict, b: dict, context: int = 30) -> PrefixDiff:
    """render_segments 로 펼친 두 요청을 앞에서부터 비교해 첫 차이를 찾는다.

    세그먼트 위치(이름)가 다르거나 개수가 달라 한쪽이 먼저 끝나면, 그 세그먼트가 차이 지점이다(offset 0).
    한쪽 목록이 다른 쪽의 앞부분과 완전히 같으면(대화가 뒤에 덧붙은 경우) location 은 None.
    """
    # TODO 3: sa, sb = render_segments(a), render_segments(b) 를 앞에서부터 짝지어 비교하세요.
    #         위치(이름)가 다르면 → PrefixDiff(지금까지 글자 수, 그 위치, 0, 각 문자열 앞 context 자, ...).
    #         문자열이 다르면 → 같은 글자 수 k 를 세어 PrefixDiff(누적 + k, 위치, k, a[k:k+context], b[k:k+context], ...).
    #         같으면 누적 글자 수에 더하고 다음으로. 끝까지 같으면 PrefixDiff(누적, None, None, "", "", ...).
    #         settings_changed 는 SETTINGS_KEYS 중 a.get(k) != b.get(k) 인 키 목록(모든 경우에 채움).
    # HINT 1: zip(sa, sb) 는 짧은 쪽에서 멈춥니다 — 한쪽이 다른 쪽의 앞부분이면(대화가 덧붙음) 루프를 빠져나와 location None.
    # HINT 2: 두 문자열이 몇 번째 글자에서 처음 갈리는지는 0 부터 하나씩 늘리며 비교해 셉니다(짧은 쪽 길이를 넘지 않게).
    # REF: claude-api 스킬 shared/prompt-caching.md "Silent invalidators"·"Finding the invalidator"
    # 개념: 캐시는 접두어 일치다. 두 요청을 같은 순서로 펼쳐 앞에서부터 비교하면, 처음 달라진 지점 "뒤"는 전부 캐시가 깨진다.
    #       a: system[0] = "당신은 안내원입니다. 오늘은 2026-11-16 09:00"     ← 현재 시각을 넣었다
    #       b: system[0] = "당신은 안내원입니다. 오늘은 2026-11-16 09:05"
    #       → location="system[0]", offset=31 (이 세그먼트의 31글자까지 같고 "0"/"5"에서 갈림). 뒤의 문서 10,000 토큰도 캐시를 못 읽는다.
    #       common_chars 는 앞 세그먼트(model 값 등)의 글자 수까지 더한 누적값이다.
    # 개념(흔한 원인): 시스템 프롬프트의 현재 시각·요청 ID(uuid), 정렬 안 된 JSON 직렬화(키 순서), 사용자별로 다른 도구 목록,
    #       모델 교체(model 이 다르면 캐시가 완전히 따로다 — location="model").
    # 개념(설정 변경): effort·thinking(output_config, thinking) 이 바뀌면 messages 쪽 캐시가 깨진다(모델에 따라 그 앞까지).
    #       이 값들은 세그먼트에는 안 보이므로 settings_changed 로 따로 알린다. → 경로(route)마다 effort 를 고정하라.
    # WHY(대화가 뒤에 붙는 것은 정상): 멀티턴에서 b = a + 새 턴이면 a 의 전체가 b 의 접두어다 → 캐시를 읽을 수 있다(location None).
    # 서비스 관점: 캐시 적중률이 갑자기 떨어지면(TODO 5) 직전·직후 요청 본문 두 개를 이 함수로 비교해 범인을 찾는다.
    #       (Claude API 에는 같은 일을 서버가 해 주는 cache diagnostics 베타도 있다 — README §4-3)
    raise NotImplementedError("TODO 3")


# ---------------------------------------------------------------------------
# TODO 4 — Batches API: 요청 구성과 결과 맞추기
# ---------------------------------------------------------------------------
def build_batch(items: Sequence[tuple[str, dict]]) -> list[dict]:
    """items = [(custom_id, 요청 사전), ...] → client.messages.batches.create(requests=...) 에 넘길 목록.

    custom_id 가 형식(^[a-zA-Z0-9_-]{1,64}$)에 안 맞거나 중복이면 ValueError.
    요청에 배치가 받지 않는 키(stream·speed·fallbacks·betas)나 max_tokens 0 이 있으면 ValueError.
    각 원소는 {"custom_id": id, "params": 요청 사전의 복사본}.
    """
    # TODO 4: items 를 돌며 custom_id 형식(CUSTOM_ID_RE)·중복을 검사하고, 요청에 BATCH_FORBIDDEN_KEYS 중 하나가 있거나
    #         max_tokens 가 0 이면 ValueError. 통과하면 {"custom_id": id, "params": dict(요청)} 을 목록에 담아 돌려주세요.
    # HINT 1: 문자열 "전체"가 형식에 맞는지는 CUSTOM_ID_RE.fullmatch(...) 로 봅니다(문자열에만 — isinstance 먼저).
    # HINT 2: 이미 본 id 는 set 에 모아 두면 중복을 바로 알 수 있습니다.
    # REF: claude-api 스킬 python/claude-api/batches.md, https://platform.claude.com/docs/en/build-with-claude/batch-processing
    # 개념: 배치 = 요청 묶음을 한 번에 맡기고 나중에 결과를 받는 비동기 처리. 입력·출력 모두 50% 할인.
    #       대신 언제 끝날지 보장이 없다 — 대부분 1시간 안, 최대 24시간(넘으면 expired). 사용자가 기다리는 경로에는 못 쓴다.
    # 개념(custom_id): 결과는 보낸 순서와 다르게 온다. 각 요청에 이름표(custom_id)를 붙여 두고 그것으로 맞춘다.
    # WHY(fallbacks 거부): 서버 측 폴백은 배치에 적용되지 않는다(공식 문서). P2-02 build_request(..., fallbacks=False) 로 만든다.
    #       betas 는 client.beta 전용 인자라 일반 batches.create 의 params 에 넣지 않는다. stream·speed·max_tokens 0 도 배치 불가.
    # 서비스 관점: 평가 세트 채점(Part 3 P3-07), 문서 색인 시 엔티티 추출(Part 4 P4-02) 같은 "밤새 돌려도 되는" 대량 작업이 배치 대상이다.
    raise NotImplementedError("TODO 4")


@dataclass
class BatchOutcome:
    texts: dict[str, str]                 # 성공: custom_id → 답(text 블록만)
    usages: dict[str, object]             # 성공: custom_id → usage(비용 계산용)
    retry: list[str]                      # 다시 보낼 것: 서버 오류(errored 중 invalid_request_error 가 아닌 것), expired
    fix: list[str]                        # 고쳐야 할 것: invalid_request_error
    canceled: list[str]
    missing: list[str]                    # 보냈는데 결과에 없는 custom_id
    unexpected: list[str]                 # 보내지 않은 custom_id 의 결과(다른 배치·오타) — 다른 칸에 섞지 않는다


def collect_batch_results(results: Iterable, sent_ids: Sequence[str]) -> BatchOutcome:
    """client.messages.batches.results(batch_id) 가 주는 결과들을 custom_id 로 분류한다(순서에 의존하지 않음)."""
    # TODO 4 (이어서): 결과마다 r.custom_id 와 r.result.type 으로 나누세요.
    #         succeeded → texts[id] = read_response(r.result.message).text, usages[id] = r.result.message.usage
    #         errored → r.result.error.error.type 이 "invalid_request_error" 면 fix, 아니면 retry / expired → retry / canceled → canceled
    #         sent_ids 에 없는 id 의 결과는 다른 칸에 넣지 말고 unexpected 로. sent_ids 중 결과에 없던 id 는 missing(보낸 순서).
    #         retry·fix·canceled·unexpected 는 정렬해서 BatchOutcome 으로.
    # HINT 1: 결과 순서는 보낸 순서와 다릅니다. 위치(인덱스)를 쓰지 말고 custom_id 를 열쇠로 씁니다.
    # HINT 2: 받은 id 를 set 에 모아 두고, 끝난 뒤 sent_ids 를 차례로 보며 그 set 에 없는 것을 고릅니다.
    # REF: 설치된 anthropic 2.6.0 types/messages/message_batch_*_result.py, 공식 batch-processing 문서 "Retrieving batch results"
    # 개념(result.type): succeeded(답 있음) · errored(요청 오류 또는 서버 오류, 과금 안 됨) · canceled · expired(24시간 안에 못 끝냄).
    #       errored 의 원인은 result.result.error.error.type — "invalid_request_error" 면 요청을 고쳐야 하고, 그 밖은 다시 보내면 된다.
    #       (error 가 두 번인 이유: result.error 는 ErrorResponse{type:"error", error: ErrorObject{type, message}} 이다)
    # WHY(succeeded 도 stop_reason 확인): 성공 = "메시지가 만들어졌다"일 뿐, 거절·잘림일 수 있다. 여기서는 text 만 모으고
    #       판정은 P2-02 ensure_complete 에 맡긴다(데모에서 사용).
    # WHY(missing): 결과가 빠졌는데 모르고 넘어가면 평가 점수·색인이 조용히 비어 있게 된다. 보낸 목록과 대조한다.
    # WHY(unexpected): 보내지 않은 id 의 결과(다른 배치 id 로 조회, id 생성 버그)를 조용히 받으면 엉뚱한 답이 섞인다. 따로 모아 알린다.
    raise NotImplementedError("TODO 4")


def wait_for_batch(retrieve: Callable[[str], object], batch_id: str, sleep: Callable[[float], None],
                   max_polls: int = 120, interval_s: float = 60.0):
    """(제공) processing_status 가 "ended" 가 될 때까지 최대 max_polls 번 확인한다. 끝나면 batch 객체, 못 끝나면 None.

    WHY(횟수 상한): "끝날 때까지 무한 반복"은 배치가 24시간 걸리면 프로그램도 24시간 붙잡힌다. 상한을 두고 나중에 다시 확인한다.
    WHY(sleep 주입): 테스트에서 실제로 기다리지 않게 가짜 sleep 을 넣는다.
    """
    for _ in range(max_polls):
        batch = retrieve(batch_id)
        if batch.processing_status == "ended":
            return batch
        sleep(interval_s)
    return None


# ---------------------------------------------------------------------------
# TODO 5 — 사용량 로그 집계: 비용·지연 리포트
# ---------------------------------------------------------------------------
CACHE_LOG_FIELDS = [*USAGE_FIELDS, "cache_write_5m", "cache_write_1h", "cache_read", "batch"]


def log_cached_usage(path: Path, usage: Usage, cu: CacheUsage, timestamp: str, batch: bool = False) -> None:
    """(제공) P2-01B log_usage 의 열 뒤에 캐시·배치 열을 덧붙인 CSV 한 줄.

    WHY(별도 함수): P2-01B 로그에는 캐시 칸이 없다. 앞쪽 열을 같게 두어 summarize_usage_log 가 두 형식을 모두 읽는다
    (캐시 칸이 없는 P2-01B·P2-02 로그는 캐시 0, batch False 로 본다). 프롬프트 원문은 남기지 않는다.
    """
    path = Path(path)
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"timestamp": timestamp}
    for name in USAGE_FIELDS[1:]:
        value = getattr(usage, name)
        row[name] = "" if value is None else value
    row.update(cache_write_5m=cu.cache_write_5m, cache_write_1h=cu.cache_write_1h, cache_read=cu.cache_read, batch=int(batch))
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CACHE_LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


@dataclass
class UsageReport:
    requests: int
    input_tokens: int               # 정가 입력 합(캐시 제외)
    cache_write_tokens: int         # 5분 + 1시간 쓰기 합
    cache_read_tokens: int
    output_tokens: int
    cache_hit_rate: float           # 캐시 읽기 / 입력 전체(정가 + 쓰기 + 읽기). 입력 0 이면 0.0
    avg_latency_ms: float
    p95_latency_ms: float           # nearest-rank: 정렬 후 ceil(0.95 × n) 번째 값
    total_usd: float                # 행마다 cache_cost 의 actual 을 더한 값(batch 행은 할인)
    stop_reasons: dict[str, int]    # stop_reason 별 개수(빈 칸은 "")


def p95(values: Sequence[float]) -> float:
    """nearest-rank 95 백분위수. 값이 없으면 ValueError."""
    # TODO 5: values 가 비었으면 ValueError. 정렬한 뒤 nearest-rank 로 ceil(0.95 × n) 번째(1부터 셈) 값을 돌려주세요.
    # HINT 1: math.ceil 을 씁니다. "n 번째"를 리스트 인덱스로 바꾸면 n - 1 입니다.
    # HINT 2: [3, 1, 2] → 정렬 [1, 2, 3], ceil(2.85) = 3 → 3.
    # 개념: p95 = "요청 100개 중 느린 쪽 5개를 빼면 가장 느린 값". 평균은 몇 개의 아주 느린 요청을 가린다.
    #       예) 지연 [100, 110, 120, ..., 290](20개) → 평균 195, p95 = 정렬 후 ceil(0.95×20)=19 번째 = 280
    # WHY(nearest-rank): 실제 관측값 중 하나를 고르므로 손으로 확인하기 쉽다(보간법은 도구마다 값이 조금씩 다르다).
    # 서비스 관점: SLO 는 보통 p95·p99 로 정한다("95% 요청이 2초 안"). 사용자 불만은 평균이 아니라 꼬리에서 나온다.
    raise NotImplementedError("TODO 5")


def _int(row: dict, key: str) -> int:
    value = row.get(key)
    return int(float(value)) if value not in (None, "") else 0


def summarize_usage_log(path: Path, prices: dict = PRICES) -> UsageReport:
    """log_cached_usage(또는 P2-01B log_usage) 로 쌓은 CSV 를 읽어 UsageReport 를 만든다. 행이 없으면 ValueError.

    단가표에 없는 모델(예: 로컬 ollama)의 행은 비용 0 으로 더한다(토큰·지연은 집계한다).
    """
    # TODO 5 (이어서): csv.DictReader 로 행을 모두 읽고(없으면 ValueError), 행마다 CacheUsage 를 만들어
    #         토큰 합계·지연 목록·stop_reason 개수를 모으세요. 비용은 단가표에 있는 모델만 cache_cost(..., batch=...).actual_usd 로 더합니다
    #         (batch 칸이 "1" 이면 배치). 캐시 적중률 = 읽기 합 / (정가 + 쓰기 + 읽기 합), 분모 0 이면 0.0.
    # HINT 1: 빈 칸·없는 열은 위의 _int(row, "열 이름") 이 0 으로 바꿔 줍니다. stop_reason 빈 칸은 "" 로 셉니다.
    # HINT 2: 지연은 float(row["latency_ms"]) 목록을 모아 평균과 p95(...) 를 계산합니다.
    # REF: Python 문서 csv.DictReader https://docs.python.org/3/library/csv.html#csv.DictReader
    # 개념(캐시 적중률): 입력 토큰 중 캐시에서 읽은 비율. 요청 단위("몇 번 적중")보다 토큰 단위가 비용과 바로 이어진다.
    #       예) 요청 3개: (정가 50, 쓰기 10,000, 읽기 0), (50, 0, 10,000), (50, 0, 10,000) → 20,000 / 30,150 ≈ 0.663
    # WHY(빈 칸 → 0): ttft_ms 처럼 없는 값은 빈 칸으로 기록돼 있다. P2-01B 형식 로그에는 캐시 열 자체가 없다(row.get).
    # 서비스 관점: 이 집계를 대시보드로 매일 본다(Part 6 P6-04). 적중률 급락 = 프롬프트 조립 코드의 회귀(TODO 3 으로 추적).
    #       예산 알림(일일 총비용 > 한도)도 total_usd 로 건다.
    raise NotImplementedError("TODO 5")


# ---------------------------------------------------------------------------
# TODO 6 (선택) — 언제 무엇을 쓰나: 비용·지연 수단 고르기
# ---------------------------------------------------------------------------
@dataclass
class Workload:
    user_waiting: bool               # 사용자가 응답을 기다리는가(채팅·검색 화면) / 아니면 오프라인 작업
    shared_prefix_tokens: int        # 요청들이 공유하는 앞부분 길이(시스템 프롬프트·고정 문서)
    requests_per_prefix: int         # 그 앞부분을 공유하는 요청 수(캐시 수명 안에서)
    gap_minutes: float               # 같은 앞부분 요청 사이 간격(시작~시작)
    simple_task: bool                # 분류·짧은 추출처럼 깊은 추론이 필요 없는가
    typical_output_tokens: int       # 보통 답의 길이(thinking 포함)


def choose_levers(w: Workload, min_cacheable: int = MIN_CACHEABLE_TOKENS) -> list[str]:
    """적용할 수단 이름 목록을 아래 순서로 돌려준다.

    1. "cache_5m"  : 공유 앞부분 ≥ min_cacheable 이고 요청 ≥ 2 이고 간격 < 5분
       "cache_1h"  : 같은 조건에서 간격 5분 이상 60분 미만 (간격 ≥ 60분이면 캐싱 없음)
    2. "batch"     : 사용자가 기다리지 않는다
    3. "effort_low": 단순 작업
    4. "max_tokens_cap" : 항상(보통 답 길이의 몇 배로 상한을 둔다 — 값은 측정으로 정한다)
    """
    # TODO 6 (선택): docstring 의 규칙 순서대로 수단 이름을 목록에 담아 돌려주세요.
    # HINT 1: 캐시 조건(공유 앞부분 ≥ min_cacheable 이고 요청 ≥ 2)을 먼저 보고, 그 안에서 간격으로 5m / 1h / 없음을 고릅니다.
    # HINT 2: "max_tokens_cap" 은 조건 없이 마지막에 항상 붙습니다.
    # REF: claude-api 스킬 shared/cost-optimization.md (공짜 수단 먼저: 캐싱·배치 → 그다음 품질과 바꾸는 수단: effort·모델),
    #      shared/prompt-caching.md "Choosing the TTL"
    # 개념(공짜 수단 먼저): 캐싱·배치는 같은 모델·같은 프롬프트라 품질이 그대로다. effort·모델 낮추기는 품질이 바뀔 수 있어
    #       평가 세트(P3-07)로 확인한 뒤에 쓴다. 모델 교체는 이 목록에 넣지 않았다 — "같은 모델 + 낮은 effort"를 먼저 재 본다.
    # 개념(TTL 고르기): 간격 < 5분이면 요청이 캐시를 계속 갱신하므로 5분이 항상 싸다. 5~60분이면 1시간(쓰기 2배)이 이득일 수 있다.
    # WHY(max_tokens 는 항상): max_tokens 는 품질 수단이 아니라 "한 요청이 쓸 수 있는 최대 비용"의 안전장치다(P2-02 TODO 5).
    # 서비스 관점: 같은 서비스 안에서도 경로(route)마다 답이 다르다. 채팅 경로 = 캐싱 + 스트리밍, 야간 색인 = 배치 + 캐싱.
    raise NotImplementedError("TODO 6")


# ---------------------------------------------------------------------------
# 단독 실행: 어느 TODO 가 남았는지 알려 준다.
# ---------------------------------------------------------------------------
def _report_progress() -> None:
    import tempfile
    from types import SimpleNamespace as NS

    usage = NS(input_tokens=1, output_tokens=1, cache_creation_input_tokens=0, cache_read_input_tokens=0)

    def todo5():
        p95([1.0])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "u.csv"
            log_cached_usage(path, Usage("claude", DEFAULT_MODEL, 1, 1, 1.0), CacheUsage(1, 0, 0, 0, 1), "t")
            summarize_usage_log(path)

    checks = [
        ("TODO 1 cache_marker / build_cached_request", lambda: build_cached_request("규칙", ["문서"], "질문")),
        ("TODO 2 read_cache_usage / cache_cost", lambda: cache_cost(read_cache_usage(usage))),
        ("TODO 3 first_divergence", lambda: first_divergence({"model": "m"}, {"model": "m"})),
        ("TODO 4 build_batch / collect_batch_results", lambda: (build_batch([("a", {"model": "m", "max_tokens": 1})]),
                                                              collect_batch_results([], ["a"]))),
        ("TODO 5 p95 / summarize_usage_log", todo5),
        ("TODO 6 choose_levers (선택)", lambda: choose_levers(Workload(True, 0, 1, 0, False, 100))),
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
        print(f"\n남은 TODO {remaining}개 — 위에서부터 순서대로 채우세요(TODO 5 는 TODO 2 의 cache_cost 를 사용합니다). 채운 뒤 검사:")
        print("  (bash)       IMPL=starter pytest part2_llm/llm_api/cost_latency/tests")
        print('  (PowerShell) $env:IMPL="starter"; pytest part2_llm/llm_api/cost_latency/tests; Remove-Item Env:IMPL')
    else:
        print("\n모든 TODO 가 구현되었습니다. 이제 pytest 로 정답 여부를 확인하세요.")


if __name__ == "__main__":
    _report_progress()

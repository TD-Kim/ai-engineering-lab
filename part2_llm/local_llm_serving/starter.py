"""P2-01B 로컬 LLM 서빙과 공급자 교체 — 실습 코드 (TODO 를 채우세요). 메인 실습: 개념은 깊게, 구현은 얕게.

로컬 LLM 서버(Ollama)를 HTTP API 처럼 호출하고(TODO 1), 스트리밍 응답에서 첫 토큰 시간·토큰/초를 재고(TODO 2),
Claude·로컬·가짜 공급자를 같은 인터페이스로 설정 교체하고(TODO 3), 주 공급자 실패 시 다른 공급자로 넘어가고(TODO 4),
사용량·지연을 CSV 로 남긴다(TODO 5). (선택) 공급자별 비용을 같은 단위로 비교한다(TODO 6).

공통 인터페이스:  client.generate(prompt) -> (text, Usage)
  - FakeClient   : 네트워크 없이 결정적으로 답한다(테스트·데모·기본값).
  - OllamaClient : http://localhost:11434/api/chat (사용자 PC 에서 Ollama 실행 필요).
  - ClaudeClient : Anthropic Python SDK, 모델은 LLM_MODEL(기본 claude-opus-5-5), 키는 .env 의 ANTHROPIC_API_KEY.
선택: 환경 변수 LLM_PROVIDER = fake(기본) | ollama | claude

transformers 로 모델을 직접 불러오는 방법은 P2-01 run_qwen.py 에 있다(여기서는 서버 방식만).
검사: IMPL=starter pytest part2_llm/local_llm_serving/tests
진행 확인: python part2_llm/local_llm_serving/starter.py   (남은 TODO 목록 출력)
"""
from __future__ import annotations

import csv  # noqa: F401  (TODO 5 에서 사용)
import json
import os  # noqa: F401  (TODO 3 에서 사용)
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, fields  # noqa: F401  (asdict 는 TODO 5 에서 사용)
from pathlib import Path

DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"
DEFAULT_OLLAMA_MODEL = "qwen2.5:0.5b"   # 태그 이름은 ollama.com/library/qwen2.5 에서 확인(작성 환경 접속 불가 → 미확인)
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
# 클라이언트 주소는 OLLAMA_BASE_URL 로 바꾼다. (OLLAMA_HOST 는 Ollama "서버"가 바인드할 주소(예: 0.0.0.0:11434, 스킴 없음)라
#  같은 이름을 쓰면 서버 설정이 클라이언트 URL 로 잘못 읽힌다 — Ollama FAQ, 2026-11-13 확인)


@dataclass
class Usage:
    """요청 1 건의 사용량·지연 기록. 공급자가 달라도 같은 모양으로 남겨야 비교할 수 있다."""

    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float              # 요청 시작 → 마지막 토큰
    ttft_ms: float | None = None   # 요청 시작 → 첫 토큰 (스트리밍일 때)
    stop_reason: str | None = None  # 왜 멈췄나(end_turn, max_tokens, refusal, stop ...)


class ProviderError(RuntimeError):
    """공급자 쪽 일시 장애(연결 실패·시간 초과·과부하·5xx). 폴백(TODO 4)은 이 오류에서만 다음 공급자로 넘어간다.

    WHY: 잘못된 요청(400)·인증 실패(401) 같은 오류는 다른 공급자로 넘겨도 해결되지 않거나(설정 오류) 문제를 숨긴다 → 그대로 올린다.
    """


# 전송 함수의 모양: (url, payload) → 응답 본문 줄들. 테스트에서는 가짜 함수로 바꿔 끼운다(네트워크·서버 불필요).
Transport = Callable[[str, dict], Iterable[bytes]]


def http_post_lines(url: str, payload: dict, timeout: float = 60.0) -> Iterator[bytes]:
    """(제공) JSON 을 POST 하고 응답을 줄 단위로 돌려준다(표준 라이브러리 urllib). 연결 실패·시간 초과는 ProviderError."""
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # localhost 의 사용자 서버
            yield from resp
    except urllib.error.HTTPError as e:
        if e.code >= 500:
            raise ProviderError(f"server error {e.code}") from e
        raise  # 4xx(모델 이름 오타 등)는 설정 문제 → 폴백하지 않고 그대로
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise ProviderError(f"cannot reach {url}: {e}") from e


# ---------------------------------------------------------------------------
# TODO 1 — 로컬 서버 chat 요청 만들기·응답 파싱 (비스트리밍)
# ---------------------------------------------------------------------------
def build_chat_payload(model: str, prompt: str, stream: bool = False, max_tokens: int = 256) -> dict:
    """Ollama /api/chat 요청 본문."""
    # TODO 1: {"model", "messages": [{"role": "user", "content": prompt}], "stream", "options": {"num_predict": max_tokens}} 를 반환하세요.
    # 개념: 로컬 LLM 서버는 "내 PC 안의 API"다. 요청 = 모델 이름 + 메시지 목록, 응답 = 답 + 사용량 → Claude API 와 같은 모양으로 쓸 수 있다.
    # HINT 1: 메시지는 역할(role)과 내용(content)을 가진 사전 하나짜리 목록입니다.
    # HINT 2: Ollama 는 출력 토큰 상한을 options 안의 "num_predict" 로 받습니다. stream 은 생략하면 true 이므로 꼭 넣으세요.
    # WHY(num_predict): 출력 상한 = 지연·자원 상한(P2-01 max_new_tokens 와 같은 역할).
    # REF: Ollama API 문서 "Generate a chat completion" (github.com/ollama/ollama docs/api.md, 2026-11-13 확인)
    raise NotImplementedError("TODO 1")


def parse_chat_response(obj: Mapping) -> tuple[str, int, int]:
    """비스트리밍 응답(JSON 객체 하나) → (답 텍스트, 입력 토큰 수, 출력 토큰 수)."""
    # TODO 1 (이어서): obj["message"]["content"], obj["prompt_eval_count"], obj["eval_count"] 를 (text, 입력 토큰, 출력 토큰)으로 반환하세요.
    #         필드가 없으면 "" / 0 으로 둡니다(message 자체가 없을 수도 있음).
    # 개념: prompt_eval_count = 입력 토큰 수, eval_count = 출력 토큰 수, *_duration 은 나노초.
    # HINT 1: 없는 키에서 멈추지 않도록 기본값을 주는 조회 방법을 쓰세요.
    # HINT 2: (obj.get("message") or {}).get("content", ""), int(obj.get("eval_count", 0))
    # 서비스 관점: 토큰 수는 그 모델의 토크나이저 기준 — Claude 의 input_tokens 와 숫자를 그대로 비교하면 안 된다(P1-01).
    raise NotImplementedError("TODO 1")


# ---------------------------------------------------------------------------
# TODO 2 — 스트리밍 응답 파싱: 첫 토큰 시간(TTFT)·토큰/초
# ---------------------------------------------------------------------------
@dataclass
class StreamResult:
    text: str
    input_tokens: int
    output_tokens: int
    ttft_ms: float | None
    total_ms: float
    tokens_per_sec: float | None


def parse_chat_stream(lines: Iterable[bytes | str], start: float, clock: Callable[[], float] = time.perf_counter) -> StreamResult:
    """줄 단위 JSON 스트림을 읽어 답을 이어 붙이고, 첫 토큰 시간·전체 시간·토큰/초를 계산한다. start 는 요청을 보낸 시각(clock 기준)."""
    # TODO 2: lines 의 각 줄(bytes 면 utf-8 로 decode, 빈 줄은 건너뜀)을 JSON 으로 읽고, 읽을 때마다 now = clock() 을 기록하세요.
    #         - message.content 가 비어 있지 않으면 이어 붙이고, 처음이면 ttft_ms = (now - start) × 1000
    #         - done 이 true 인 줄에서 prompt_eval_count, eval_count, eval_duration(나노초)을 읽음
    #         - total_ms = (마지막 줄 시각 - start) × 1000
    #         - tokens_per_sec = eval_count / (eval_duration / 1e9), eval_duration 이 없으면 eval_count / (마지막 시각 - 첫 토큰 시각),
    #           둘 다 안 되면 None
    #         StreamResult(text, input_tokens, output_tokens, ttft_ms, total_ms, tokens_per_sec) 를 반환하세요.
    # 개념: 스트리밍은 조각마다 한 줄씩 보낸다. TTFT = 요청 → 내용이 있는 첫 조각. 예) eval 282 토큰 / 4.54 초 ≈ 62.2 토큰/초
    # HINT 1: 첫 조각 시각과 마지막 줄 시각을 따로 기억해 두면 두 가지 시간을 모두 계산할 수 있습니다.
    # HINT 2: json.loads(line), raw.decode("utf-8") if isinstance(raw, bytes) else raw
    # WHY(clock 주입): 가짜 시계를 넣어 시간 계산을 결정적으로 검사한다.
    # 서비스 관점: 채팅 UI 의 체감 속도는 TTFT, 긴 답의 대기는 토큰/초가 정한다. 프롬프트가 길수록 TTFT 가 길다(P1-05).
    raise NotImplementedError("TODO 2")


# ---------------------------------------------------------------------------
# (제공) 세 공급자 — 모두 generate(prompt) -> (text, Usage)
# ---------------------------------------------------------------------------
class FakeClient:
    """네트워크 없는 결정적 가짜 공급자. 토큰 수는 공백 기준 단어 수로 흉내 낸다(실제 토크나이저 아님)."""

    provider = "fake"

    def __init__(self, model: str = "fake-echo", fail: bool = False):
        self.model = model
        self.fail = fail  # True 면 ProviderError — 폴백 실험용

    def generate(self, prompt: str, max_tokens: int = 256) -> tuple[str, Usage]:
        if self.fail:
            raise ProviderError("fake provider is down")
        words = prompt.split()
        text = "가짜 답변: " + " ".join(words[:max_tokens])
        return text, Usage(self.provider, self.model, len(words), len(text.split()), latency_ms=0.0, ttft_ms=0.0, stop_reason="end_turn")


class OllamaClient:
    """Ollama /api/chat 을 스트리밍으로 호출. transport 를 바꿔 끼우면 서버 없이 테스트할 수 있다."""

    provider = "ollama"

    def __init__(self, model: str = DEFAULT_OLLAMA_MODEL, base_url: str = DEFAULT_OLLAMA_HOST,
                 transport: Transport | None = None, clock: Callable[[], float] = time.perf_counter):
        self.model = model
        self.url = base_url.rstrip("/") + "/api/chat"
        self.transport = transport or http_post_lines
        self.clock = clock

    def generate(self, prompt: str, max_tokens: int = 256) -> tuple[str, Usage]:
        start = self.clock()
        lines = self.transport(self.url, build_chat_payload(self.model, prompt, stream=True, max_tokens=max_tokens))
        r = parse_chat_stream(lines, start, self.clock)
        return r.text, Usage(self.provider, self.model, r.input_tokens, r.output_tokens, r.total_ms, r.ttft_ms, stop_reason="stop")


class ClaudeClient:
    """Anthropic Python SDK 로 스트리밍 호출. sdk_client 는 anthropic.Anthropic() (테스트에서는 같은 모양의 가짜 객체).

    REF: claude-api 스킬 python/claude-api/streaming.md (messages.stream → text_stream → get_final_message),
         README.md Error Handling·Stop Reasons (작성 시점 2026-11 기준)
    """

    provider = "claude"

    def __init__(self, sdk_client, model: str = DEFAULT_CLAUDE_MODEL, clock: Callable[[], float] = time.perf_counter):
        self.client = sdk_client
        self.model = model
        self.clock = clock

    def generate(self, prompt: str, max_tokens: int = 1024) -> tuple[str, Usage]:
        import anthropic  # SDK 는 이 공급자를 쓸 때만 필요

        # 일시 장애만 ProviderError 로 바꾼다. BadRequest(400)·Authentication(401) 등은 설정 문제라 그대로 올린다.
        transient = (anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.InternalServerError)
        start = self.clock()
        ttft_ms = None
        parts: list[str] = []
        try:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for text in stream.text_stream:  # 텍스트 조각만 나온다(thinking 등 다른 블록은 제외)
                    if ttft_ms is None and text:
                        ttft_ms = (self.clock() - start) * 1000
                    parts.append(text)
                final = stream.get_final_message()
        except transient as e:
            # 설치된 SDK(anthropic 2.6.0)에서 확인: APITimeoutError ⊂ APIConnectionError,
            # OverloadedError·ServiceUnavailableError·GatewayTimeoutError ⊂ InternalServerError.
            # SDK 는 이런 오류를 max_retries 만큼 자동 재시도한 뒤에 올린다 → 여기 도착했다면 재시도도 실패한 것(이제 폴백).
            raise ProviderError(f"claude transient error: {type(e).__name__}") from e
        # stop_reason 확인: refusal(안전상 거절)·max_tokens(잘림)는 답이 불완전하다는 신호 → 기록해서 호출부가 판단하게 한다.
        usage = Usage(self.provider, self.model, final.usage.input_tokens, final.usage.output_tokens,
                      (self.clock() - start) * 1000, ttft_ms, stop_reason=final.stop_reason)
        return "".join(parts), usage


# ---------------------------------------------------------------------------
# TODO 3 — 설정(LLM_PROVIDER)으로 공급자 고르기
# ---------------------------------------------------------------------------
def make_client(provider: str | None = None, env: Mapping[str, str] | None = None, claude_sdk_factory: Callable[[], object] | None = None):
    """provider(없으면 env 의 LLM_PROVIDER, 그것도 없으면 "fake")에 맞는 클라이언트를 만든다. 모르는 이름은 ValueError.

    - "fake"   → FakeClient()
    - "ollama" → OllamaClient(model=env OLLAMA_MODEL 또는 기본값, base_url=env OLLAMA_BASE_URL 또는 기본값)
    - "claude" → env 에 ANTHROPIC_API_KEY 가 없으면 ValueError(안내 메시지). 있으면 ClaudeClient(SDK 객체, model=env LLM_MODEL 또는 기본값)
      SDK 객체는 claude_sdk_factory() 로 만든다(기본: anthropic.Anthropic(timeout=60, max_retries=2)).
    """
    # TODO 3: name = (provider 또는 env["LLM_PROVIDER"] 또는 "fake").lower() 로 정하고
    #         "fake" → FakeClient(), "ollama" → OllamaClient(model=env OLLAMA_MODEL 또는 DEFAULT_OLLAMA_MODEL, base_url=env OLLAMA_BASE_URL 또는 DEFAULT_OLLAMA_HOST),
    #         "claude" → ANTHROPIC_API_KEY 가 없으면 ValueError, 있으면 ClaudeClient(claude_sdk_factory(), model=env LLM_MODEL 또는 DEFAULT_CLAUDE_MODEL)
    #         (claude_sdk_factory 가 None 이면 anthropic.Anthropic(timeout=60.0, max_retries=2) 를 만드는 함수로), 그 밖은 ValueError.
    #         env 가 None 이면 os.environ 을 씁니다.
    # 개념: 호출부는 client.generate(prompt) 만 안다. 어떤 공급자인지는 설정이 정한다(인터페이스·전략 패턴, Spring DI 와 같은 발상).
    # HINT 1: 이름을 먼저 정규화(소문자)한 뒤 if 로 나누세요. claude 는 키 확인을 SDK 생성보다 먼저.
    # HINT 2: env.get("LLM_PROVIDER"), env.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    # WHY(기본 fake): 키·서버 없는 환경에서도 실행되고, 실수로 유료 API 를 부르지 않는다.
    # 서비스 관점: 공급자를 바꾸면 토큰 세는 방식·스트리밍 형식·오류·재시도·품질·한국어 능력이 달라진다 → 평가를 다시 한다.
    raise NotImplementedError("TODO 3")


# ---------------------------------------------------------------------------
# TODO 4 — 폴백: 주 공급자 실패 시 다음 공급자
# ---------------------------------------------------------------------------
def generate_with_fallback(clients: Sequence, prompt: str) -> tuple[str, Usage, list[str]]:
    """clients 를 순서대로 시도한다. ProviderError 면 기록하고 다음으로, 성공하면 (text, usage, 실패 기록) 반환.

    모두 실패하면 ProviderError(실패 기록을 모두 담은 메시지). ProviderError 가 아닌 오류는 바로 올린다.
    """
    # TODO 4: clients 가 비면 ValueError. 순서대로 client.generate(prompt) 를 시도해 성공하면 (text, usage, errors) 반환.
    #         ProviderError 면 errors 에 f"{client.provider}: {e}" 를 넣고 다음으로. 모두 실패하면
    #         ProviderError("all providers failed: " + "; ".join(errors)). ProviderError 가 아닌 오류는 잡지 마세요.
    # 개념: 주 공급자가 일시 장애면 보조 공급자로 답한다 → 서비스가 멈추지 않는다.
    # HINT 1: try/except 를 for 안에 두고, 잡는 예외 종류를 좁히세요.
    # HINT 2: except ProviderError as e: errors.append(...); continue
    # WHY(ProviderError 만): 잘못된 요청·버그는 다른 공급자에서도 실패하거나 문제를 숨긴다.
    # 서비스 관점: 폴백 응답은 품질·형식이 다를 수 있다 → 누가 답했는지(usage.provider) 기록, 폴백 비율 모니터링(Part 6).
    raise NotImplementedError("TODO 4")


# ---------------------------------------------------------------------------
# TODO 5 — 사용량·지연 기록 (CSV 한 줄)
# ---------------------------------------------------------------------------
USAGE_FIELDS = ["timestamp", *[f.name for f in fields(Usage)]]


def log_usage(path: Path, usage: Usage, timestamp: str) -> None:
    """usage 를 CSV 한 줄로 덧붙인다. 파일이 없으면 머리글(USAGE_FIELDS)부터 쓴다. 없는 값(None)은 빈 칸."""
    # TODO 5: path 가 없으면 머리글(USAGE_FIELDS)부터 쓰고, {"timestamp": timestamp, **asdict(usage)} 를 한 줄 덧붙이세요.
    #         None 값은 빈 칸(""). 상위 폴더가 없으면 만듭니다.
    # 개념: 요청마다 "누가·얼마나(토큰)·얼마나 걸려(지연)" 를 한 줄로 남기면 비용·지연·폴백 비율을 집계할 수 있다.
    # HINT 1: 파일 존재 여부를 "열기 전에" 확인해야 머리글을 한 번만 씁니다. 덧붙이기 모드("a")로 여세요.
    # HINT 2: csv.DictWriter(f, fieldnames=USAGE_FIELDS), w.writeheader(), w.writerow(...), open(..., newline="", encoding="utf-8")
    # WHY(csv 모듈): 쉼표·따옴표가 든 값을 직접 이어 붙이면 칸이 어긋난다.
    # 서비스 관점: 프롬프트 원문은 개인정보일 수 있어 기본적으로 남기지 않는다. 실제 서비스는 로그 수집기·DB 로 보낸다(Part 6 P6-04).
    raise NotImplementedError("TODO 5")


# ---------------------------------------------------------------------------
# TODO 6 (선택) — 공급자별 비용을 같은 단위로
# ---------------------------------------------------------------------------
def api_cost_per_mtok(input_price: float, output_price: float, output_ratio: float) -> float:
    """API 단가(USD / 100만 토큰)로, 전체 토큰 중 출력 비율이 output_ratio 일 때 100만 토큰당 비용."""
    # TODO 6 (선택): output_ratio 가 [0, 1] 밖이면 ValueError. input_price × (1 - output_ratio) + output_price × output_ratio 를 반환하세요.
    # 개념: 입력·출력 단가가 달라 섞인 비율로 가중 평균한다. 예) 입력 $4, 출력 $20, 출력 비율 0.2 → $7.2
    raise NotImplementedError("TODO 6")


def local_cost_per_mtok(tokens_per_sec: float, watts: float, price_per_kwh: float, hardware_per_hour: float = 0.0) -> float:
    """로컬 모델이 출력 100만 토큰을 만드는 데 드는 비용 = 전기료 + 장비 감가(시간당 금액 × 시간).

    입력(프롬프트) 처리 시간은 생략한 근사다(입력은 병렬 처리라 보통 출력 생성보다 훨씬 빠르다 — P1-01). API 의 "출력만" 단가와 비교한다.
    """
    # TODO 6 (이어서): tokens_per_sec <= 0 이면 ValueError. 100만 토큰 생성 시간(시간 단위) hours 를 구해
    #         전기료(와트 × 시간 / 1000 × kWh 단가) + 장비 감가(hardware_per_hour × 시간)를 반환하세요.
    # 개념: 예) 60 토큰/초, 250 W → 16,667 초 ≈ 4.63 시간 → 1.16 kWh. 로컬도 0 원이 아니다(전기·장비).
    # HINT: hours = 1_000_000 / tokens_per_sec / 3600
    # 서비스 관점: 로컬 = 프라이버시·오프라인·고정 비용, API = 동시 처리량·품질·운영 부담 없음. 트래픽 양과 패턴으로 고른다.
    raise NotImplementedError("TODO 6")


# ---------------------------------------------------------------------------
# (제공) nvidia-smi 출력 파싱 — 측정 스크립트에서 사용
# ---------------------------------------------------------------------------
def parse_nvidia_smi_memory(text: str) -> list[int]:
    """`nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits` 출력 → GPU 별 사용 메모리(MiB) 목록."""
    return [int(m) for m in re.findall(r"^\s*(\d+)\s*$", text, flags=re.MULTILINE)]


# ---------------------------------------------------------------------------
# 단독 실행: 어느 TODO 가 남았는지 알려 준다.
# ---------------------------------------------------------------------------
def _report_progress() -> None:
    import tempfile

    done_line = b'{"message": {"content": "hi"}, "done": true, "eval_count": 1}'
    checks = [
        ("TODO 1 build_chat_payload / parse_chat_response",
         lambda: (build_chat_payload("m", "p"), parse_chat_response({"message": {"content": "x"}}))),
        ("TODO 2 parse_chat_stream", lambda: parse_chat_stream([done_line], 0.0, lambda: 0.1)),
        ("TODO 3 make_client", lambda: make_client("fake", env={})),
        ("TODO 4 generate_with_fallback", lambda: generate_with_fallback([FakeClient()], "hi")),
        ("TODO 5 log_usage", lambda: log_usage(Path(tempfile.mkdtemp()) / "u.csv", Usage("fake", "m", 1, 1, 0.0), "t")),
        ("TODO 6 api_cost_per_mtok / local_cost_per_mtok (선택)",
         lambda: (api_cost_per_mtok(4, 20, 0.2), local_cost_per_mtok(60, 250, 0.2))),
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
        print(f"\n남은 TODO {remaining}개 — 위에서부터 순서대로 채우세요(OllamaClient 는 TODO 1·2, 폴백은 TODO 3 의 클라이언트를 씁니다). 채운 뒤 검사:")
        print("  (bash)       IMPL=starter pytest part2_llm/local_llm_serving/tests")
        print('  (PowerShell) $env:IMPL="starter"; pytest part2_llm/local_llm_serving/tests; Remove-Item Env:IMPL')
    else:
        print("\n모든 TODO 가 구현되었습니다. 이제 pytest 로 정답 여부를 확인하세요.")


if __name__ == "__main__":
    _report_progress()

"""P2-02 테스트. 기본은 solution 검사, `IMPL=starter pytest ...` 로 직접 구현을 검사한다.

API 키·네트워크 불필요: 응답은 SimpleNamespace 로 만든 가짜 Message, SDK 는 가짜 객체, 시계는 가짜 시계.
스레드·대기 반복 없음(모든 반복은 정해진 목록을 도는 for).
"""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

LAB_DIR = Path(__file__).resolve().parents[1]
ALLOWED_KEYS = {"model", "max_tokens", "messages", "output_config", "system", "betas", "fallbacks"}
FORBIDDEN_KEYS = {"temperature", "top_p", "top_k", "thinking", "tool_choice"}


def msg(blocks, stop_reason="end_turn", model="claude-opus-5-5", iterations=None, stop_details=None, in_tok=12, out_tok=34):
    content = [NS(type=t, text=x) if t == "text" else NS(type=t, thinking="") for t, x in blocks]
    return NS(content=content, stop_reason=stop_reason, model=model, stop_details=stop_details,
              usage=NS(input_tokens=in_tok, output_tokens=out_tok, iterations=iterations))


def fake_clock(*times):
    it = iter(times)
    return lambda: next(it)


# --- TODO 1 ---------------------------------------------------------------
def test_build_messages_multiturn(impl):
    out = impl.build_messages([("user", "연차는?"), ("assistant", "(답)")], "반차는?")
    assert out == [{"role": "user", "content": "연차는?"}, {"role": "assistant", "content": "(답)"},
                   {"role": "user", "content": "반차는?"}]
    assert impl.build_messages([], "hi") == [{"role": "user", "content": "hi"}]
    for bad in ([("assistant", "x")], [("user", "a"), ("user", "b")], [("user", "a"), ("assistant", "b"), ("assistant", "c")]):
        with pytest.raises(ValueError):
            impl.build_messages(bad, "q")


def test_build_request_shape_and_forbidden_params(impl):
    msgs = impl.build_messages([], "질문")
    req = impl.build_request(msgs, system="당신은 사내 규정 안내원입니다.")
    assert req == {"model": "claude-opus-5-5", "max_tokens": 16000, "messages": msgs, "output_config": {"effort": "medium"},
                   "system": "당신은 사내 규정 안내원입니다.", "betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
    assert set(req) <= ALLOWED_KEYS and not (set(req) & FORBIDDEN_KEYS)   # 샘플링·thinking 끄기·강제 tool_choice 없음
    plain = impl.build_request(msgs, model="claude-sonnet-5-5", max_tokens=500, effort="low", fallbacks=False)
    assert "system" not in plain and "betas" not in plain and "fallbacks" not in plain
    assert (plain["model"], plain["max_tokens"], plain["output_config"]) == ("claude-sonnet-5-5", 500, {"effort": "low"})


def test_build_request_validation(impl):
    msgs = impl.build_messages([], "q")
    for kwargs in ({"effort": "extreme"}, {"max_tokens": 0}):
        with pytest.raises(ValueError):
            impl.build_request(msgs, **kwargs)
    with pytest.raises(ValueError):
        impl.build_request([{"role": "user", "content": "q"}, {"role": "assistant", "content": "답의 앞부분"}])  # prefill 금지
    with pytest.raises(ValueError):
        impl.build_request([])


# --- TODO 2 ---------------------------------------------------------------
def test_read_response_text_blocks_only(impl):
    v = impl.read_response(msg([("thinking", ""), ("text", "안녕"), ("text", "하세요")]))
    assert (v.text, v.ok, v.stop_reason, v.served_by, v.fell_back, v.skipped_blocks) == ("안녕하세요", True, "end_turn", "claude-opus-5-5", False, 1)


def test_read_response_fallback_detection(impl):
    its = [NS(type="message"), NS(type="fallback_message")]
    v = impl.read_response(msg([("fallback", None), ("text", "답")], model="claude-opus-5", iterations=its))
    assert v.fell_back and v.served_by == "claude-opus-5" and v.skipped_blocks == 1
    v = impl.read_response(msg([], stop_reason="refusal", iterations=its))     # 폴백 모델도 거절 → 폴백 성공 아님
    assert not v.fell_back and not v.ok


def test_ensure_complete(impl):
    assert impl.ensure_complete(msg([("text", "ok")])).text == "ok"
    with pytest.raises(impl.RefusalError) as e:
        impl.ensure_complete(msg([], stop_reason="refusal", stop_details=NS(category="cyber", explanation=None)))
    assert e.value.category == "cyber"
    with pytest.raises(impl.RefusalError) as e:
        impl.ensure_complete(msg([], stop_reason="refusal", stop_details=None))  # stop_details 없음도 정상 상태
    assert e.value.category is None
    with pytest.raises(impl.TruncatedError) as e:
        impl.ensure_complete(msg([("thinking", ""), ("text", '{"triples": [{"sub')], stop_reason="max_tokens"))
    assert e.value.partial == '{"triples": [{"sub'


# --- TODO 3 ---------------------------------------------------------------
def test_with_json_schema(impl):
    req = impl.build_request(impl.build_messages([], "q"), effort="high")
    out = impl.with_json_schema(req, impl.Extraction)
    fmt = out["output_config"]["format"]
    assert out["output_config"]["effort"] == "high" and fmt["type"] == "json_schema"
    assert fmt["schema"]["additionalProperties"] is False and fmt["schema"]["required"] == ["triples"]
    assert "format" not in req["output_config"]                         # 원래 요청은 그대로


def test_parse_structured_success_and_errors(impl):
    good = json.dumps({"triples": [{"subject": "세종대왕", "relation": "창제했다", "object": "훈민정음"}]}, ensure_ascii=False)
    out = impl.parse_structured(msg([("thinking", ""), ("text", good)]), impl.Extraction)
    assert out.triples[0].object == "훈민정음"
    with pytest.raises(impl.StructuredOutputError):
        impl.parse_structured(msg([("text", '{"triples": [{"subject": "a"}]}')]), impl.Extraction)  # 필드 누락
    with pytest.raises(impl.TruncatedError):                             # 검증보다 stop_reason 이 먼저
        impl.parse_structured(msg([("text", '{"triples": [')], stop_reason="max_tokens"), impl.Extraction)
    with pytest.raises(impl.RefusalError):
        impl.parse_structured(msg([], stop_reason="refusal"), impl.Extraction)


# --- TODO 4 ---------------------------------------------------------------
class FakeStream:
    def __init__(self, pieces, final):
        self.text_stream = iter(pieces)
        self.final, self.closed = final, False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True
        return False

    def get_final_message(self):
        assert not self.closed, "get_final_message 는 with 블록 안에서"
        return self.final


def test_consume_stream(impl):
    final = msg([("thinking", ""), ("text", "안녕")])
    seen = []
    s = FakeStream(["", "안", "녕"], final)
    out = impl.consume_stream(s, start=0.0, clock=fake_clock(0.9, 1.5), on_text=seen.append)
    assert out.text == "안녕" and out.final is final and s.closed
    assert out.ttft_ms == pytest.approx(900.0) and out.total_ms == pytest.approx(1500.0)   # 빈 조각은 TTFT 아님
    assert seen == ["", "안", "녕"]


def test_consume_stream_then_check(impl):
    out = impl.consume_stream(FakeStream([], msg([], stop_reason="refusal")), start=0.0, clock=fake_clock(0.4))
    assert out.ttft_ms is None and not impl.read_response(out.final).ok


def test_to_usage_links_p2_01b(impl, tmp_path):
    u = impl.to_usage(msg([("text", "x")], in_tok=5, out_tok=7), latency_ms=100.0, ttft_ms=40.0)
    assert (u.provider, u.model, u.input_tokens, u.output_tokens, u.stop_reason) == ("claude", "claude-opus-5-5", 5, 7, "end_turn")
    impl.log_usage(tmp_path / "u.csv", u, "2026-11-15T00:00:00")          # P2-01B 의 기록 함수 재사용
    assert "claude,claude-opus-5-5,5,7" in (tmp_path / "u.csv").read_text(encoding="utf-8")


# --- TODO 5 ---------------------------------------------------------------
def test_estimate_cost(impl):
    calls = []
    sdk = NS(messages=NS(count_tokens=lambda **k: calls.append(k) or NS(input_tokens=2000)))
    req = impl.with_json_schema(impl.build_request(impl.build_messages([], "q"), system="S", max_tokens=1000), impl.Extraction)
    est = impl.estimate_cost(sdk, req, expected_output_tokens=500)
    assert est.input_tokens == 2000
    assert est.expected_usd == pytest.approx((2000 * 4 + 500 * 20) / 1e6)   # $0.018
    assert est.max_usd == pytest.approx((2000 * 4 + 1000 * 20) / 1e6)       # $0.028
    assert set(calls[0]) == {"model", "messages", "system", "output_config"}  # betas·fallbacks 는 넘기지 않음
    assert impl.estimate_cost(sdk, impl.build_request(impl.build_messages([], "q"), max_tokens=10)).expected_usd is None
    with pytest.raises(KeyError):
        impl.estimate_cost(sdk, {"model": "unknown", "messages": [], "max_tokens": 1})


# --- TODO 6 (선택) --------------------------------------------------------
def _errors():
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")

    def resp(code, headers=None):
        return httpx2.Response(code, request=req, headers=headers or {})

    return anthropic, {
        "rate": anthropic.RateLimitError("slow down", response=resp(429, {"retry-after": "7"}), body=None),
        "rate_nohdr": anthropic.RateLimitError("slow down", response=resp(429), body=None),
        "overloaded": anthropic.OverloadedError("busy", response=resp(529), body=None),
        "bad": anthropic.BadRequestError("bad", response=resp(400), body=None),
        "auth": anthropic.AuthenticationError("auth", response=resp(401), body=None),
        "timeout": anthropic.APITimeoutError(request=req),
        "conn": anthropic.APIConnectionError(request=req),
    }


def test_classify_error_order(impl):
    _, e = _errors()
    d = impl.classify_error(e["rate"])
    assert (d.kind, d.retry, d.retry_after_s) == ("rate_limit", True, 7.0)   # APIStatusError 보다 먼저 검사해야 함
    assert impl.classify_error(e["rate_nohdr"]).retry_after_s is None
    assert (impl.classify_error(e["overloaded"]).kind, impl.classify_error(e["overloaded"]).retry) == ("server", True)
    for k in ("bad", "auth"):
        assert (impl.classify_error(e[k]).kind, impl.classify_error(e[k]).retry) == ("client", False)
    for k in ("timeout", "conn"):
        assert (impl.classify_error(e[k]).kind, impl.classify_error(e[k]).retry) == ("connection", True)
    assert impl.classify_error(ValueError("bug")).kind == "unknown"


# --- 데모 스크립트 ----------------------------------------------------------
def test_run_demo_runs(impl, capsys, tmp_path):
    spec = importlib.util.spec_from_file_location("p2_02_run_demo_test", LAB_DIR / "run_demo.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    mod.load = lambda _name: impl
    argv, sys.argv = sys.argv, ["run_demo.py", "--log", str(tmp_path / "u.csv")]
    try:
        mod.main()
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert "훈민정음" in out and "RefusalError" in out and (tmp_path / "u.csv").exists()

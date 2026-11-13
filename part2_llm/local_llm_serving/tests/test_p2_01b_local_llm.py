"""P2-01B 테스트. 기본은 solution 검사, `IMPL=starter pytest ...` 로 직접 구현을 검사한다.

네트워크·서버·API 키 불필요: Ollama 전송은 가짜 함수, Claude 는 가짜 SDK 객체, 시계는 가짜 시계.
서버를 띄우는 스레드나 기다리는 반복이 없다(모든 반복은 정해진 목록을 도는 for).
"""
import csv
import importlib.util
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

LAB_DIR = Path(__file__).resolve().parents[1]


def fake_clock(*times):
    it = iter(times)
    return lambda: next(it)


def jl(content, done=False, **extra):
    import json

    return json.dumps({"message": {"role": "assistant", "content": content}, "done": done, **extra}).encode("utf-8")


# --- TODO 1 ---------------------------------------------------------------
def test_build_chat_payload(impl):
    assert impl.build_chat_payload("qwen2.5:0.5b", "안녕", stream=False, max_tokens=64) == {
        "model": "qwen2.5:0.5b",
        "messages": [{"role": "user", "content": "안녕"}],
        "stream": False,
        "options": {"num_predict": 64},
    }
    assert impl.build_chat_payload("m", "p")["stream"] is False          # 기본은 비스트리밍(명시)


def test_parse_chat_response(impl):
    obj = {"message": {"role": "assistant", "content": "안녕하세요"}, "done": True, "prompt_eval_count": 26, "eval_count": 282}
    assert impl.parse_chat_response(obj) == ("안녕하세요", 26, 282)
    assert impl.parse_chat_response({"done": True}) == ("", 0, 0)       # 필드가 빠져도 멈추지 않음


# --- TODO 2 ---------------------------------------------------------------
def test_parse_chat_stream_ttft_and_server_speed(impl):
    lines = [jl(""), jl("안"), b"\n", jl("녕"),
             jl("", done=True, prompt_eval_count=5, eval_count=2, eval_duration=40_000_000)]   # 0.04 초에 2 토큰
    r = impl.parse_chat_stream(lines, start=0.0, clock=fake_clock(0.05, 0.1, 0.2, 0.5))      # 빈 줄은 clock 안 씀
    assert r.text == "안녕"
    assert r.ttft_ms == pytest.approx(100.0)                            # 내용 있는 첫 조각(0.1 초)
    assert r.total_ms == pytest.approx(500.0)
    assert (r.input_tokens, r.output_tokens) == (5, 2)
    assert r.tokens_per_sec == pytest.approx(50.0)                      # 2 / 0.04


def test_parse_chat_stream_wallclock_speed_and_str_lines(impl):
    lines = [jl("a").decode(), jl("b").decode(), jl("", done=True, eval_count=8).decode()]
    r = impl.parse_chat_stream(lines, start=1.0, clock=fake_clock(1.2, 1.4, 2.2))
    assert r.text == "ab" and r.ttft_ms == pytest.approx(200.0)
    assert r.tokens_per_sec == pytest.approx(8 / (2.2 - 1.2))          # eval_duration 없음 → 시계 시간


def test_parse_chat_stream_no_tokens(impl):
    r = impl.parse_chat_stream([jl("", done=True)], start=0.0, clock=fake_clock(0.3))
    assert r.text == "" and r.ttft_ms is None and r.tokens_per_sec is None


def test_ollama_client_with_fake_transport(impl):
    calls = []

    def transport(url, payload):
        calls.append((url, payload))
        return [jl("hi"), jl("", done=True, prompt_eval_count=3, eval_count=1, eval_duration=10_000_000)]

    c = impl.OllamaClient(model="qwen2.5:1.5b", base_url="http://localhost:11434/", transport=transport,
                          clock=fake_clock(0.0, 0.1, 0.3))
    text, usage = c.generate("안녕", max_tokens=32)
    assert text == "hi"
    assert calls == [("http://localhost:11434/api/chat", impl.build_chat_payload("qwen2.5:1.5b", "안녕", stream=True, max_tokens=32))]
    assert (usage.provider, usage.model, usage.input_tokens, usage.output_tokens) == ("ollama", "qwen2.5:1.5b", 3, 1)
    assert usage.ttft_ms == pytest.approx(100.0) and usage.latency_ms == pytest.approx(300.0)


def test_http_post_lines_maps_errors(impl, monkeypatch):
    def refuse(*_a, **_k):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(impl.urllib.request, "urlopen", refuse)
    with pytest.raises(impl.ProviderError):
        list(impl.http_post_lines("http://localhost:11434/api/chat", {}))

    def http_error(code):
        def f(*_a, **_k):
            raise urllib.error.HTTPError("u", code, "x", {}, None)
        return f

    monkeypatch.setattr(impl.urllib.request, "urlopen", http_error(503))
    with pytest.raises(impl.ProviderError):
        list(impl.http_post_lines("http://localhost:11434/api/chat", {}))
    monkeypatch.setattr(impl.urllib.request, "urlopen", http_error(404))  # 모델 이름 오타 등 → 폴백 대상 아님
    with pytest.raises(urllib.error.HTTPError):
        list(impl.http_post_lines("http://localhost:11434/api/chat", {}))


# --- (제공) Claude 클라이언트: 가짜 SDK ------------------------------------
class FakeStream:
    def __init__(self, pieces, final, error=None):
        self.text_stream = iter(pieces)
        self._final, self._error = final, error

    def __enter__(self):
        if self._error:
            raise self._error
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._final


def fake_sdk(pieces=("안", "녕"), stop_reason="end_turn", error=None):
    calls = []
    final = SimpleNamespace(usage=SimpleNamespace(input_tokens=10, output_tokens=2), stop_reason=stop_reason)

    def stream(**kwargs):
        calls.append(kwargs)
        return FakeStream(list(pieces), final, error)

    return SimpleNamespace(messages=SimpleNamespace(stream=stream)), calls


def test_claude_client_stream_usage_and_params(impl):
    sdk, calls = fake_sdk()
    c = impl.ClaudeClient(sdk, clock=fake_clock(0.0, 0.25, 1.0))
    text, usage = c.generate("hi", max_tokens=100)
    assert text == "안녕"
    assert calls == [{"model": "claude-opus-5-5", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]}]  # 샘플링 파라미터 없음
    assert (usage.provider, usage.input_tokens, usage.output_tokens, usage.stop_reason) == ("claude", 10, 2, "end_turn")
    assert usage.ttft_ms == pytest.approx(250.0) and usage.latency_ms == pytest.approx(1000.0)


def test_claude_client_records_refusal(impl):
    sdk, _ = fake_sdk(pieces=(), stop_reason="refusal")
    text, usage = impl.ClaudeClient(sdk, clock=fake_clock(0.0, 0.5)).generate("x")
    assert text == "" and usage.stop_reason == "refusal" and usage.ttft_ms is None


def test_claude_client_error_mapping(impl):
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    sdk, _ = fake_sdk(error=anthropic.APIConnectionError(request=req))
    with pytest.raises(impl.ProviderError):
        impl.ClaudeClient(sdk, clock=fake_clock(0.0)).generate("x")
    sdk, _ = fake_sdk(error=anthropic.OverloadedError("busy", response=httpx2.Response(529, request=req), body=None))
    with pytest.raises(impl.ProviderError):
        impl.ClaudeClient(sdk, clock=fake_clock(0.0)).generate("x")
    sdk, _ = fake_sdk(error=anthropic.BadRequestError("bad", response=httpx2.Response(400, request=req), body=None))
    with pytest.raises(anthropic.BadRequestError):                    # 설정 문제는 폴백 대상이 아님
        impl.ClaudeClient(sdk, clock=fake_clock(0.0)).generate("x")


# --- TODO 3 ---------------------------------------------------------------
def test_make_client_selection(impl):
    assert isinstance(impl.make_client(env={}), impl.FakeClient)
    c = impl.make_client(env={"LLM_PROVIDER": "OLLAMA", "OLLAMA_MODEL": "qwen2.5:1.5b", "OLLAMA_BASE_URL": "http://pc:11434"})
    assert isinstance(c, impl.OllamaClient) and c.model == "qwen2.5:1.5b" and c.url == "http://pc:11434/api/chat"
    assert impl.make_client("ollama", env={}).model == impl.DEFAULT_OLLAMA_MODEL
    assert isinstance(impl.make_client("fake", env={"LLM_PROVIDER": "claude"}), impl.FakeClient)   # 인자가 env 보다 우선
    with pytest.raises(ValueError):
        impl.make_client("gpt", env={})


def test_make_client_claude(impl):
    with pytest.raises(ValueError):
        impl.make_client("claude", env={})                             # 키 없음 → 시작할 때 안내
    made = []
    c = impl.make_client("claude", env={"ANTHROPIC_API_KEY": "x"}, claude_sdk_factory=lambda: made.append(1) or "SDK")
    assert isinstance(c, impl.ClaudeClient) and c.model == "claude-opus-5-5" and c.client == "SDK" and made == [1]
    c = impl.make_client("claude", env={"ANTHROPIC_API_KEY": "x", "LLM_MODEL": "claude-sonnet-5-5"}, claude_sdk_factory=lambda: "SDK")
    assert c.model == "claude-sonnet-5-5"


# --- TODO 4 ---------------------------------------------------------------
def test_fallback_uses_next_provider(impl):
    text, usage, errors = impl.generate_with_fallback([impl.FakeClient(model="primary", fail=True), impl.FakeClient(model="backup")], "hi")
    assert usage.model == "backup" and text.startswith("가짜 답변")
    assert errors == ["fake: fake provider is down"]


def test_fallback_all_fail_and_non_provider_errors(impl):
    with pytest.raises(impl.ProviderError, match="all providers failed"):
        impl.generate_with_fallback([impl.FakeClient(fail=True), impl.FakeClient(fail=True)], "hi")

    called = []

    class Broken:
        provider = "broken"

        def generate(self, prompt):
            raise ValueError("bug")

    class Spy(impl.FakeClient):
        def generate(self, prompt, max_tokens=256):
            called.append(1)
            return super().generate(prompt, max_tokens)

    with pytest.raises(ValueError):
        impl.generate_with_fallback([Broken(), Spy()], "hi")
    assert called == []                                                # 버그는 다른 공급자로 넘기지 않음
    with pytest.raises(ValueError):
        impl.generate_with_fallback([], "hi")


# --- TODO 5 ---------------------------------------------------------------
def test_log_usage_csv(impl, tmp_path):
    path = tmp_path / "logs" / "usage.csv"
    impl.log_usage(path, impl.Usage("ollama", "qwen2.5:0.5b", 26, 282, 4883.6, 120.0, "stop"), "2026-11-13T10:00:00")
    impl.log_usage(path, impl.Usage("claude", "claude-opus-5-5", 10, 2, 1000.0, None, "end_turn"), "2026-11-13T10:01:00")
    rows = list(csv.reader(path.open(encoding="utf-8")))
    assert rows[0] == impl.USAGE_FIELDS == ["timestamp", "provider", "model", "input_tokens", "output_tokens",
                                             "latency_ms", "ttft_ms", "stop_reason"]
    assert rows[1] == ["2026-11-13T10:00:00", "ollama", "qwen2.5:0.5b", "26", "282", "4883.6", "120.0", "stop"]
    assert rows[2][6] == "" and len(rows) == 3                         # None → 빈 칸, 머리글은 한 번


# --- TODO 6 (선택) --------------------------------------------------------
def test_costs(impl):
    assert impl.api_cost_per_mtok(4.0, 20.0, 0.2) == pytest.approx(7.2)
    hours = 1_000_000 / 60 / 3600
    assert impl.local_cost_per_mtok(60, 250, 0.2) == pytest.approx(250 * hours / 1000 * 0.2)
    assert impl.local_cost_per_mtok(60, 250, 0.2, hardware_per_hour=0.1) == pytest.approx(250 * hours / 1000 * 0.2 + 0.1 * hours)
    for bad in ((4, 20, -0.1), (4, 20, 1.1)):
        with pytest.raises(ValueError):
            impl.api_cost_per_mtok(*bad)
    with pytest.raises(ValueError):
        impl.local_cost_per_mtok(0, 250, 0.2)


# --- (제공) nvidia-smi 파싱 · 데모 ------------------------------------------
def test_parse_nvidia_smi_memory(impl):
    assert impl.parse_nvidia_smi_memory("1234\n 567 \n") == [1234, 567]
    assert impl.parse_nvidia_smi_memory("NVIDIA-SMI has failed\n") == []


def test_run_demo_runs(impl, capsys, tmp_path):
    spec = importlib.util.spec_from_file_location("p2_01b_run_demo_test", LAB_DIR / "run_demo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.load = lambda _name: impl
    argv, sys.argv = sys.argv, ["run_demo.py", "--log", str(tmp_path / "u.csv")]
    try:
        mod.main()
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert "폴백" in out and (tmp_path / "u.csv").exists()

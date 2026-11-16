"""P2-03 테스트. 기본은 solution 검사, `IMPL=starter pytest ...` 로 직접 구현을 검사한다.

API 키·네트워크 불필요: usage·배치 결과는 SimpleNamespace 로 만든 가짜 객체, sleep 은 가짜 함수.
스레드·대기 반복 없음(모든 반복은 정해진 목록을 도는 for, 폴링은 횟수 상한).
"""
import copy
import math
from types import SimpleNamespace as NS

import pytest

FORBIDDEN_KEYS = {"temperature", "top_p", "top_k", "thinking", "tool_choice"}


def usage(inp, out, write=None, read=None, w5=None, w1=None):
    cc = None if w5 is None and w1 is None else NS(ephemeral_5m_input_tokens=w5 or 0, ephemeral_1h_input_tokens=w1 or 0)
    return NS(input_tokens=inp, output_tokens=out, cache_creation_input_tokens=write, cache_read_input_tokens=read, cache_creation=cc)


def text_message(text, u=None, stop_reason="end_turn"):
    return NS(content=[NS(type="thinking", thinking=""), NS(type="text", text=text)], stop_reason=stop_reason,
              model="claude-opus-5-5", stop_details=None, usage=u or usage(10, 5))


# --- TODO 1 ---------------------------------------------------------------
def test_cache_marker(impl):
    assert impl.cache_marker() == {"type": "ephemeral"}
    assert impl.cache_marker("5m") == {"type": "ephemeral"}
    assert impl.cache_marker("1h") == {"type": "ephemeral", "ttl": "1h"}
    with pytest.raises(ValueError):
        impl.cache_marker("10m")


def test_build_cached_request_layout(impl):
    req = impl.build_cached_request("당신은 사내 규정 안내원입니다.", ["규정 1장", "규정 2장"], "연차는 며칠 전까지?")
    assert req["system"] == [
        {"type": "text", "text": "당신은 사내 규정 안내원입니다."},
        {"type": "text", "text": "규정 1장"},
        {"type": "text", "text": "규정 2장", "cache_control": {"type": "ephemeral"}},   # 마지막 고정 블록에만
    ]
    assert req["messages"] == [{"role": "user", "content": "연차는 며칠 전까지?"}]          # 질문은 맨 뒤, 표시 없음
    # P2-02 build_request 의 나머지 형태를 그대로 쓴다
    assert (req["model"], req["max_tokens"], req["output_config"]) == ("claude-opus-5-5", 16000, {"effort": "medium"})
    assert req["fallbacks"] == "default" and req["betas"] == ["server-side-fallback-2026-07-01"]
    assert not (set(req) & FORBIDDEN_KEYS) and "cache_control" not in req
    assert impl.count_breakpoints(req) == 1


def test_build_cached_request_options_and_stability(impl):
    a = impl.build_cached_request("규칙", ["문서"], "질문 A", ttl="1h", effort="low", fallbacks=False, max_tokens=800)
    assert a["system"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert "fallbacks" not in a and a["output_config"] == {"effort": "low"} and a["max_tokens"] == 800
    b = impl.build_cached_request("규칙", ["문서"], "질문 B", ttl="1h", effort="low", fallbacks=False, max_tokens=800)
    assert a["system"] == b["system"] and a["messages"] != b["messages"]     # 질문만 다르고 앞부분은 같다
    docs = ["문서"]
    impl.build_cached_request("규칙", docs, "q")
    assert docs == ["문서"]                                                    # 입력 목록을 바꾸지 않음
    for bad in (("", ["문서"]), ("규칙", [])):
        with pytest.raises(ValueError):
            impl.build_cached_request(*bad, "q")


# --- TODO 2 ---------------------------------------------------------------
def test_read_cache_usage_fields(impl):
    cu = impl.read_cache_usage(usage(50, 200, write=10000, read=0))
    assert (cu.input_tokens, cu.cache_write_5m, cu.cache_write_1h, cu.cache_read, cu.output_tokens) == (50, 10000, 0, 0, 200)
    assert cu.prompt_tokens == 10050 and cu.status == "write"
    cu = impl.read_cache_usage(usage(50, 200, write=0, read=10000))
    assert cu.status == "hit" and cu.prompt_tokens == 10050
    cu = impl.read_cache_usage(NS(input_tokens=7, output_tokens=3))                 # 캐시 필드가 아예 없는 객체
    assert (cu.cache_write_5m, cu.cache_write_1h, cu.cache_read, cu.status) == (0, 0, 0, "none")
    cu = impl.read_cache_usage(usage(7, 3, write=None, read=None))                  # None 도 0
    assert cu.prompt_tokens == 7


def test_read_cache_usage_ttl_breakdown(impl):
    cu = impl.read_cache_usage(usage(5, 1, write=300, read=0, w5=100, w1=200))      # cache_creation 이 있으면 그것을 따른다
    assert (cu.cache_write_5m, cu.cache_write_1h) == (100, 200)
    cu = impl.read_cache_usage(usage(5, 1, write=300, read=0), default_ttl="1h")    # 없으면 default_ttl
    assert (cu.cache_write_5m, cu.cache_write_1h) == (0, 300)
    with pytest.raises(ValueError):
        impl.read_cache_usage(usage(1, 1), default_ttl="1d")


def test_cache_cost_worked_example(impl):
    write = impl.cache_cost(impl.read_cache_usage(usage(50, 200, write=10000, read=0)))
    read = impl.cache_cost(impl.read_cache_usage(usage(50, 200, write=0, read=10000)))
    assert write.actual_usd == pytest.approx(0.0542) and write.no_cache_usd == pytest.approx(0.0442)
    assert write.saved_usd == pytest.approx(-0.0100)                               # 쓰기만 하면 손해
    assert read.actual_usd == pytest.approx(0.0062) and read.saved_usd == pytest.approx(0.0380)
    assert write.actual_usd + read.actual_usd < write.no_cache_usd + read.no_cache_usd   # 2번째 요청부터 이득


def test_cache_cost_ttl_batch_and_model(impl):
    one_hour = impl.cache_cost(impl.read_cache_usage(usage(0, 0, write=1_000_000, read=0), default_ttl="1h"))
    assert one_hour.actual_usd == pytest.approx(8.00)                              # 1시간 쓰기 = 2× 입력 단가
    cu = impl.read_cache_usage(usage(1_000_000, 1_000_000))
    assert impl.cache_cost(cu, batch=True).actual_usd == pytest.approx(12.00)      # (4 + 20) × 0.5
    hit = impl.read_cache_usage(usage(0, 0, write=0, read=1_000_000))
    b = impl.cache_cost(hit, batch=True)                                           # 할인은 실제·기준 비용 둘 다에
    assert (b.actual_usd, b.no_cache_usd, b.saved_usd) == pytest.approx((0.10, 2.00, 1.90))
    assert impl.cache_cost(cu, model="claude-sonnet-5-5").actual_usd == pytest.approx(12.00)   # 2 + 10
    with pytest.raises(KeyError):
        impl.cache_cost(cu, model="gpt-x")


# --- TODO 3 ---------------------------------------------------------------
def _req(system_text, question="q", **extra):
    return {"model": "claude-opus-5-5", "system": [{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": question}], **extra}


def test_first_divergence_timestamp_in_system(impl):
    a = _req("당신은 안내원입니다. 오늘은 2026-11-16 09:00")
    b = _req("당신은 안내원입니다. 오늘은 2026-11-16 09:05")
    d = impl.first_divergence(a, b)
    assert (d.location, d.offset) == ("system[0]", 31)
    assert d.common_chars == len("claude-opus-5-5") + 31
    assert (d.a_snippet, d.b_snippet) == ("0", "5") and d.settings_changed == []


def test_first_divergence_same_prefix_and_appended_turn(impl):
    a, b = _req("규칙", "연차는?"), _req("규칙", "반차는?")
    d = impl.first_divergence(a, b)
    assert d.location == "messages[0].user" and d.offset == 0                       # 질문만 다름 → system 까지는 캐시 가능
    assert d.common_chars == len("claude-opus-5-5") + len("규칙")
    grown = copy.deepcopy(a)
    grown["messages"] += [{"role": "assistant", "content": "답"}, {"role": "user", "content": "또?"}]
    d = impl.first_divergence(a, grown)
    assert d.location is None and d.offset is None                                 # 뒤에 덧붙은 대화는 깨짐 아님
    assert d.common_chars == len("claude-opus-5-5") + len("규칙") + len("연차는?")


def test_first_divergence_cache_control_is_ignored(impl):
    a = _req("규칙")
    b = copy.deepcopy(a)
    del b["system"][0]["cache_control"]
    assert impl.first_divergence(a, b).location is None


def test_first_divergence_model_tools_and_key_order(impl):
    d = impl.first_divergence(_req("규칙"), {**_req("규칙"), "model": "claude-sonnet-5-5"})
    assert d.location == "model" and d.common_chars == len("claude-")               # 모델이 다르면 맨 앞에서 갈린다
    tool_a = {"name": "search", "description": "검색", "input_schema": {"type": "object", "properties": {}}}
    tool_b = {"description": "검색", "name": "search", "input_schema": {"type": "object", "properties": {}}}
    d = impl.first_divergence(_req("규칙", tools=[tool_a]), _req("규칙", tools=[tool_b]))
    assert d.location == "tools[0]" and d.offset == 2                               # 같은 내용, 키 순서만 다름 → 바이트가 다르다
    d = impl.first_divergence(_req("규칙", tools=[tool_a]), _req("규칙"))
    assert d.location == "tools[0]" and d.offset == 0                               # 도구가 빠지면 그 자리부터


def test_first_divergence_settings_changed(impl):
    a = _req("규칙", output_config={"effort": "medium"})
    b = _req("규칙", output_config={"effort": "high"})
    d = impl.first_divergence(a, b)
    assert d.location is None and d.settings_changed == ["output_config"]           # 본문은 같아도 effort 변경을 알린다
    c = _req("규칙 v2", output_config={"effort": "high"})                           # 본문과 effort 를 함께 바꿈
    d = impl.first_divergence(a, c)
    assert d.location == "system[0]" and d.settings_changed == ["output_config"]     # 일찍 반환하는 분기에서도 알린다


# --- TODO 4 ---------------------------------------------------------------
def test_build_batch_shape(impl):
    p = {"model": "claude-opus-5-5", "max_tokens": 1000, "messages": [{"role": "user", "content": "q"}]}
    out = impl.build_batch([("doc-1", p), ("doc_2", p)])
    assert out == [{"custom_id": "doc-1", "params": p}, {"custom_id": "doc_2", "params": p}]
    out[0]["params"]["max_tokens"] = 1
    assert p["max_tokens"] == 1000                                                  # 원래 요청을 공유하지 않음(복사)


@pytest.mark.parametrize("cid", ["", "a b", "한글", "x" * 65, "a/b", 7, "abc\n", "a" * 64 + "\n"])   # 끝 줄바꿈도 거부
def test_build_batch_rejects_bad_ids(impl, cid):
    with pytest.raises(ValueError):
        impl.build_batch([(cid, {"model": "m", "max_tokens": 1})])


def test_build_batch_rejects_duplicates_and_forbidden(impl):
    ok = {"model": "m", "max_tokens": 1}
    assert len(impl.build_batch([("x" * 64, ok)])) == 1
    with pytest.raises(ValueError):
        impl.build_batch([("a", ok), ("a", ok)])
    for extra in ({"fallbacks": "default"}, {"betas": ["b"]}, {"stream": True}, {"speed": "fast"}, {"max_tokens": 0}):
        with pytest.raises(ValueError):
            impl.build_batch([("a", {**ok, **extra})])


def test_collect_batch_results_by_custom_id(impl):
    def err(t):
        return NS(type="errored", error=NS(type="error", error=NS(type=t, message="...")))

    results = [   # 보낸 순서(a, b, c, d, e, f)와 다르게 도착
        NS(custom_id="c", result=err("api_error")),
        NS(custom_id="b", result=NS(type="succeeded", message=text_message("B 답", usage(3, 4)))),
        NS(custom_id="d", result=err("invalid_request_error")),
        NS(custom_id="a", result=NS(type="succeeded", message=text_message("A 답"))),
        NS(custom_id="e", result=NS(type="expired")),
        NS(custom_id="g", result=NS(type="canceled")),
        NS(custom_id="zz", result=NS(type="succeeded", message=text_message("엉뚱한 답"))),   # 보내지 않은 id
        NS(custom_id="yy", result=err("api_error")),
    ]
    out = impl.collect_batch_results(results, ["a", "b", "c", "d", "e", "f", "g"])
    assert out.texts == {"a": "A 답", "b": "B 답"}                                   # text 블록만, custom_id 로 매칭
    assert out.usages["b"].output_tokens == 4 and "zz" not in out.usages
    assert (out.retry, out.fix, out.canceled, out.missing) == (["c", "e"], ["d"], ["g"], ["f"])
    assert out.unexpected == ["yy", "zz"]                                             # 조용히 받지 않고 따로 알린다


def test_wait_for_batch_bounded(impl):
    states = iter(["in_progress", "in_progress", "ended"])
    slept = []
    batch = impl.wait_for_batch(lambda bid: NS(id=bid, processing_status=next(states)), "b1", slept.append, interval_s=5)
    assert batch.id == "b1" and slept == [5, 5]
    calls = []
    never = impl.wait_for_batch(lambda bid: calls.append(bid) or NS(processing_status="in_progress"), "b2", lambda s: None, max_polls=3)
    assert never is None and len(calls) == 3


# --- TODO 5 ---------------------------------------------------------------
def test_p95_nearest_rank(impl):
    values = list(range(100, 300, 10))          # 20개: 100, 110, ..., 290
    assert impl.p95(values) == 280
    assert impl.p95([5.0]) == 5.0
    assert impl.p95([3, 1, 2]) == 3            # ceil(0.95 × 3) = 3 번째
    assert impl.p95(list(range(1, 101))) == 95
    with pytest.raises(ValueError):
        impl.p95([])


def test_summarize_usage_log(impl, tmp_path):
    path = tmp_path / "usage.csv"
    U, CU = impl.Usage, impl.CacheUsage
    impl.log_cached_usage(path, U("claude", "claude-opus-5-5", 50, 200, 900.0, 300.0, "end_turn"), CU(50, 10000, 0, 0, 200), "t1")
    impl.log_cached_usage(path, U("claude", "claude-opus-5-5", 50, 200, 500.0, None, "end_turn"), CU(50, 0, 0, 10000, 200), "t2")
    impl.log_cached_usage(path, U("claude", "claude-opus-5-5", 50, 200, 400.0, None, "max_tokens"), CU(50, 0, 0, 10000, 200), "t3",
                          batch=True)
    r = impl.summarize_usage_log(path)
    assert (r.requests, r.input_tokens, r.cache_write_tokens, r.cache_read_tokens, r.output_tokens) == (3, 150, 10000, 20000, 600)
    assert r.cache_hit_rate == pytest.approx(20000 / 30150)
    assert r.avg_latency_ms == pytest.approx(600.0) and r.p95_latency_ms == 900.0
    assert r.total_usd == pytest.approx(0.0542 + 0.0062 + 0.0062 * 0.5)               # 배치 행은 할인
    assert r.stop_reasons == {"end_turn": 2, "max_tokens": 1}


def test_summarize_reads_p2_01b_log_format(impl, tmp_path):
    path = tmp_path / "old.csv"                 # P2-01B log_usage 형식: 캐시·batch 열 없음, ttft 빈 칸, 로컬 모델 포함
    path.write_text("timestamp,provider,model,input_tokens,output_tokens,latency_ms,ttft_ms,stop_reason\n"
                    "t1,claude,claude-opus-5-5,1000000,0,100.0,,end_turn\n"
                    "t2,ollama,qwen2.5:0.5b,26,282,4883.6,120.0,stop\n", encoding="utf-8")
    r = impl.summarize_usage_log(path)
    assert (r.requests, r.cache_read_tokens, r.cache_hit_rate) == (2, 0, 0.0)
    assert r.total_usd == pytest.approx(4.00)                                        # 단가표에 없는 로컬 모델은 비용 0
    assert r.stop_reasons == {"end_turn": 1, "stop": 1}
    empty = tmp_path / "empty.csv"
    empty.write_text("timestamp,provider,model,input_tokens,output_tokens,latency_ms,ttft_ms,stop_reason\n", encoding="utf-8")
    with pytest.raises(ValueError):
        impl.summarize_usage_log(empty)


# --- TODO 6 (선택) ---------------------------------------------------------
def test_choose_levers(impl):
    W = impl.Workload
    assert impl.choose_levers(W(True, 8000, 50, 1.0, False, 600)) == ["cache_5m", "max_tokens_cap"]          # 채팅 경로
    assert impl.choose_levers(W(False, 8000, 1000, 0.1, True, 300)) == ["cache_5m", "batch", "effort_low", "max_tokens_cap"]  # 야간 추출
    assert impl.choose_levers(W(True, 8000, 10, 20.0, False, 600)) == ["cache_1h", "max_tokens_cap"]         # 20분 간격
    assert impl.choose_levers(W(True, 8000, 10, 60.0, False, 600)) == ["max_tokens_cap"]                     # 1시간 이상 간격
    assert impl.choose_levers(W(True, 511, 10, 1.0, False, 600)) == ["max_tokens_cap"]                       # 최소 길이 미달
    assert impl.choose_levers(W(True, 512, 2, 4.9, False, 600)) == ["cache_5m", "max_tokens_cap"]            # 경계값
    assert impl.choose_levers(W(True, 512, 2, 5.0, False, 600)) == ["cache_1h", "max_tokens_cap"]            # 정확히 5분 → 1시간
    assert impl.choose_levers(W(True, 8000, 1, 1.0, False, 600)) == ["max_tokens_cap"]                       # 한 번만 쓰면 캐시 무의미
    assert impl.choose_levers(W(True, 600, 5, 1.0, False, 100), min_cacheable=1024) == ["max_tokens_cap"]   # 모델별 최소 길이


def test_constants_match_official_sources(impl):
    p = impl.PRICES["claude-opus-5-5"]
    assert (p["input"], p["write_5m"], p["write_1h"], p["read"], p["output"]) == (4.00, 5.00, 8.00, 0.20, 20.00)
    assert math.isclose(p["write_5m"], p["input"] * 1.25) and math.isclose(p["write_1h"], p["input"] * 2)
    assert (impl.MAX_BREAKPOINTS, impl.MIN_CACHEABLE_TOKENS, impl.BATCH_DISCOUNT) == (4, 512, 0.5)

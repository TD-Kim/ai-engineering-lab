"""P2-03 데모: 캐시 친화적 요청 → 캐시 쓰기·읽기·깨짐 → 원인 찾기 → 배치 → 사용량 로그 집계 → 수단 고르기.

실행: python part2_llm/llm_api/cost_latency/run_demo.py [--impl starter] [--log 경로]
가짜 SDK 가 캐시를 흉내 낸다(형식 예시). 토큰 수는 "글자 2개 = 1토큰"으로 센 가짜 값이고, 지연도 미리 정한 가짜 값이다.
Claude 의 실제 토큰 수·지연·답이 아니다. 실제 측정은 measure_caching.py (선택, API 키 필요, 유료).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

LAB_DIR = Path(__file__).resolve().parent

INSTRUCTIONS = "당신은 사내 규정 안내원입니다. 아래 규정 문서에 근거해서만 답하고, 문서에 없으면 모른다고 답하세요."
# 캐시 최소 길이(512 토큰)를 넘기려고 만든 가짜 규정 문서. 내용은 예시다.
POLICY = "\n".join(
    f"제{i}조 (휴가 규정 {i}) 직원은 연차를 사용하려면 사용일 {i % 5 + 1}일 전까지 신청서를 제출한다. "
    f"팀장은 업무 일정을 고려해 승인하며, 반차는 오전·오후로 나누어 쓸 수 있다."
    for i in range(1, 31)
)
QUESTIONS = ["연차는 며칠 전까지 신청해야 해?", "반차도 같은 규칙이야?", "팀장이 승인을 안 하면?"]
FAKE_LATENCY_MS = [2400.0, 1300.0, 2300.0, 1250.0, 1350.0, 1200.0]   # 가짜 값(형식 예시, 측정 아님)


def _load_sibling(filename: str):
    """같은 폴더의 파일을 "실습별 고유 이름"(lab_<상위>_<실습>_<파일>)으로 불러온다. sys.path 는 건드리지 않는다."""
    name = f"lab_{LAB_DIR.parent.name}_{LAB_DIR.name}_{Path(filename).stem}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, LAB_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclass 가 자기 모듈을 찾을 수 있게 먼저 등록
    spec.loader.exec_module(mod)
    return mod


def fake_tokens(text: str) -> int:
    return max(1, len(text) // 2)


class FakeCachingSDK:
    """client.beta.messages.create 흉내. 마지막 cache_control 까지의 접두어(+ model·output_config)가 전에 쓴 것과 같으면 읽기.

    실제 서버 규칙(20블록 lookback, TTL, 동시 요청 등)은 단순화했다. 원리(접두어 일치)만 보여 준다.
    """

    def __init__(self, m, min_cacheable: int):
        self.m, self.min_cacheable, self.store = m, min_cacheable, set()
        self.beta = NS(messages=NS(create=self.create))

    def create(self, **req):
        segs = self.m.render_segments(req)
        cut = 1 + len(req.get("system", []))   # model + system 블록들(이 데모는 마지막 system 블록에만 표시)
        prefix, rest = segs[:cut], segs[cut:]
        prefix_tokens = sum(fake_tokens(t) for _, t in prefix[1:])
        rest_tokens = sum(fake_tokens(t) for _, t in rest)
        key = json.dumps([prefix, req.get("output_config")], ensure_ascii=False)
        write = read = 0
        if prefix_tokens < self.min_cacheable:
            rest_tokens += prefix_tokens               # 최소 길이 미달: 표시가 있어도 캐시 안 됨
        elif key in self.store:
            read = prefix_tokens
        else:
            write = prefix_tokens
            self.store.add(key)
        usage = NS(input_tokens=rest_tokens, output_tokens=120, cache_creation_input_tokens=write,
                   cache_read_input_tokens=read, cache_creation=None)
        return NS(content=[NS(type="thinking", thinking=""), NS(type="text", text="(가짜 답)")], stop_reason="end_turn",
                  model=req["model"], stop_details=None, usage=usage)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["solution", "starter"], default="solution")
    ap.add_argument("--log", default="part2_llm/llm_api/cost_latency/outputs/usage_demo.csv")  # 저장소 루트 기준(git 에 안 올라감)
    args = ap.parse_args()
    m = _load_sibling(f"{args.impl}.py")
    log = Path(args.log)
    if log.exists():
        log.unlink()                                       # 데모 전용 로그: 매번 새로 쓴다
    latencies = iter(FAKE_LATENCY_MS)

    print("[1] 캐시 친화적 요청 (TODO 1) — 고정 지시문·문서는 system 앞쪽, 질문은 messages 맨 뒤")
    req = m.build_cached_request(INSTRUCTIONS, [POLICY], QUESTIONS[0])
    for i, b in enumerate(req["system"]):
        print(f"    system[{i}] {b['text'][:24]!r}... 가짜 토큰 {fake_tokens(b['text'])}  cache_control={b.get('cache_control')}")
    print(f"    messages[-1] = {req['messages'][-1]}   breakpoint {m.count_breakpoints(req)}개 (최대 {m.MAX_BREAKPOINTS})")

    print("[2] 같은 앞부분 + 다른 질문 → 캐시 쓰기 → 읽기 → (시스템 프롬프트에 시각을 넣으면) 다시 쓰기 (TODO 2)")
    sdk = FakeCachingSDK(m, m.MIN_CACHEABLE_TOKENS)
    stamped = "현재 시각 2026-11-16 09:05. " + INSTRUCTIONS
    reqs = [m.build_cached_request(INSTRUCTIONS, [POLICY], q) for q in QUESTIONS[:2]]
    reqs.append(m.build_cached_request(stamped, [POLICY], QUESTIONS[2]))
    print(f"    {'요청':<28} {'상태':<6} {'정가입력':>8} {'캐시쓰기':>8} {'캐시읽기':>8} {'실제 $':>9} {'캐시없음 $':>10} {'절감 $':>9}")
    for i, r in enumerate(reqs):
        msg = sdk.beta.messages.create(**r)
        cu = m.read_cache_usage(msg.usage)
        cost = m.cache_cost(cu, r["model"])
        label = ("[시각 포함] " if i == 2 else "") + r["messages"][-1]["content"]
        print(f"    {label[:26]:<28} {cu.status:<6} {cu.input_tokens:>8} {cu.cache_write_5m:>8} {cu.cache_read:>8} "
              f"{cost.actual_usd:>9.5f} {cost.no_cache_usd:>10.5f} {cost.saved_usd:>9.5f}")
        usage = m.Usage("claude", msg.model, msg.usage.input_tokens, msg.usage.output_tokens, next(latencies), None, msg.stop_reason)
        m.log_cached_usage(log, usage, cu, f"2026-11-16T09:0{i}:00")

    print("[3] 왜 깨졌나 (TODO 3) — 두 요청 본문을 앞에서부터 비교")
    d = m.first_divergence(reqs[1], reqs[2])
    print(f"    첫 차이: {d.location}, 세그먼트 안 {d.offset}번째 글자, 공통 {d.common_chars}글자")
    print(f"    a: {d.a_snippet!r}\n    b: {d.b_snippet!r}")
    d = m.first_divergence(reqs[0], reqs[1])
    print(f"    (정상) 질문만 다른 두 요청: 첫 차이 {d.location} → system 까지는 캐시를 읽는다")
    hi = m.build_cached_request(INSTRUCTIONS, [POLICY], QUESTIONS[1], effort="high")
    d = m.first_divergence(reqs[1], hi)
    print(f"    effort 만 바꾼 요청: 본문 차이 {d.location}, 바뀐 설정 {d.settings_changed} → 캐시 영향 있음(경로마다 고정)")

    print("[4] 배치 (TODO 4) — custom_id 로 결과 맞추기, 50% 할인 (가짜 결과, 도착 순서 뒤섞임)")
    items = [(f"policy-q{i}", m.build_cached_request(INSTRUCTIONS, [POLICY], q, fallbacks=False)) for i, q in enumerate(QUESTIONS)]
    batch = m.build_batch(items)
    print(f"    요청 {len(batch)}개, params 키: {sorted(batch[0]['params'])}  (fallbacks·betas 없음)")
    canned = {cid: sdk.beta.messages.create(**p) for cid, p in items}
    results = [NS(custom_id="policy-q2", result=NS(type="errored", error=NS(type="error", error=NS(type="overloaded_error", message="")))),
               NS(custom_id="policy-q0", result=NS(type="succeeded", message=canned["policy-q0"])),
               NS(custom_id="policy-q1", result=NS(type="succeeded", message=canned["policy-q1"]))]
    out = m.collect_batch_results(results, [cid for cid, _ in items])
    print(f"    성공 {sorted(out.texts)}  다시 보낼 것 {out.retry}  고칠 것 {out.fix}  빠진 것 {out.missing}")
    for j, cid in enumerate(sorted(out.usages)):
        cu = m.read_cache_usage(out.usages[cid])
        std, bat = m.cache_cost(cu).actual_usd, m.cache_cost(cu, batch=True).actual_usd
        print(f"    {cid}: 일반 ${std:.5f} → 배치 ${bat:.5f}")
        usage = m.Usage("claude", m.DEFAULT_MODEL, cu.input_tokens, cu.output_tokens, next(latencies), None, "end_turn")
        m.log_cached_usage(log, usage, cu, f"2026-11-16T10:0{j}:00", batch=True)

    print(f"[5] 사용량 로그 집계 (TODO 5) — {log}")
    rep = m.summarize_usage_log(log)
    print(f"    요청 {rep.requests}, 정가입력 {rep.input_tokens}, 캐시쓰기 {rep.cache_write_tokens}, 캐시읽기 {rep.cache_read_tokens}, "
          f"출력 {rep.output_tokens}")
    print(f"    캐시 적중률(토큰) {rep.cache_hit_rate:.1%}, 평균 지연 {rep.avg_latency_ms:.0f} ms, p95 {rep.p95_latency_ms:.0f} ms (가짜 지연), "
          f"총비용 ${rep.total_usd:.5f}, stop_reason {rep.stop_reasons}")

    print("[6] 언제 무엇을 쓰나 (TODO 6, 선택)")
    W = m.Workload
    for name, w in [("사내 규정 챗봇(사용자 대기, 1분 간격)", W(True, 8000, 200, 1.0, False, 600)),
                    ("야간 엔티티 추출(색인)", W(False, 3000, 5000, 0.1, True, 400)),
                    ("하루 한 번 보고서", W(False, 2000, 1, 1440.0, False, 3000))]:
        print(f"    {name:<26} → {m.choose_levers(w)}")


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as e:
        print(f"\n[멈춤] {e} 가 아직 구현되지 않았습니다. 남은 TODO: python part2_llm/llm_api/cost_latency/starter.py")

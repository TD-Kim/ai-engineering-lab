"""P2-02 데모: 요청 본문 → 구조화 출력 → 스트리밍 → 응답 검사(거절·잘림·폴백) → 비용 추정 → 오류 분류 → 사용량 기록.

실행: python part2_llm/llm_api/basics/run_demo.py [--impl starter] [--log 경로]
가짜 SDK 가 "미리 정해 둔 응답"을 돌려준다(형식 예시). Claude 의 실제 답·토큰 수·시간이 아니다.
실제 호출은 call_claude.py (선택, API 키 필요).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace as NS

LAB_DIR = Path(__file__).resolve().parent
SENTENCE = "세종대왕은 1443년에 훈민정음을 창제했고, 집현전 학자들이 이를 도왔다."
CANNED_JSON = json.dumps({"triples": [
    {"subject": "세종대왕", "relation": "창제했다", "object": "훈민정음"},
    {"subject": "집현전 학자들", "relation": "도왔다", "object": "훈민정음 창제"},
]}, ensure_ascii=False)


def load(impl: str):
    spec = importlib.util.spec_from_file_location(f"p2_02_{impl}", LAB_DIR / f"{impl}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def fake_message(text, stop_reason="end_turn", model="claude-opus-5-5", iterations=None, in_tok=310, out_tok=95):
    return NS(content=[NS(type="thinking", thinking=""), NS(type="text", text=text)] if text else [],
              stop_reason=stop_reason, model=model, stop_details=None,
              usage=NS(input_tokens=in_tok, output_tokens=out_tok, iterations=iterations))


class FakeStream:
    def __init__(self, pieces, final):
        self.text_stream, self.final = iter(pieces), final

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.final


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["solution", "starter"], default="solution")
    ap.add_argument("--log", default="part2_llm/llm_api/basics/outputs/usage.csv")  # 저장소 루트 기준(git 에 안 올라감)
    args = ap.parse_args()
    m = load(args.impl)

    print("[1] 요청 본문 (TODO 1) — 멀티턴, system, effort 명시, 서버 측 폴백")
    messages = m.build_messages([("user", "문장에서 관계를 뽑아 줄래?"), ("assistant", "네, 문장을 주세요.")], SENTENCE)
    req = m.build_request(messages, system="당신은 문장에서 (주어, 관계, 목적어) triple 을 추출합니다. 문장에 없는 사실은 만들지 마세요.")
    print("    키:", sorted(req))
    print(f"    model={req['model']} max_tokens={req['max_tokens']} output_config={req['output_config']} fallbacks={req['fallbacks']!r} betas={req['betas']}")
    print(f"    마지막 메시지 역할: {req['messages'][-1]['role']} (prefill 없음)")

    print("[2] 구조화 출력 (TODO 3) — 가짜 SDK 의 미리 정한 응답")
    sreq = m.with_json_schema(req, m.Extraction)
    print("    스키마 required:", sreq["output_config"]["format"]["schema"]["required"],
          "additionalProperties:", sreq["output_config"]["format"]["schema"]["additionalProperties"])
    sdk = NS(beta=NS(messages=NS(create=lambda **k: fake_message(CANNED_JSON))),
             messages=NS(count_tokens=lambda **k: NS(input_tokens=1200)))
    result = m.parse_structured(sdk.beta.messages.create(**sreq), m.Extraction)
    for t in result.triples:
        print(f"    ({t.subject}, {t.relation}, {t.object})")

    print("[3] 스트리밍 (TODO 4) — 가짜 스트림·가짜 시계(초)")
    ticks = iter([1.20, 1.25, 1.30, 1.90])
    out = m.consume_stream(FakeStream(["훈민정음은 ", "백성을 가르치는 ", "바른 소리입니다."], fake_message("훈민정음은 백성을 가르치는 바른 소리입니다.")),
                           start=0.0, clock=lambda: next(ticks))
    print(f"    text={out.text!r}  TTFT {out.ttft_ms:.0f} ms (thinking 포함)  전체 {out.total_ms:.0f} ms  stop_reason={out.final.stop_reason}")

    print("[4] 응답 검사 (TODO 2) — 거절·잘림·폴백")
    for label, message in (
        ("거절", fake_message("", stop_reason="refusal")),
        ("잘림", fake_message('{"triples": [{"subj', stop_reason="max_tokens")),
    ):
        try:
            m.ensure_complete(message)
        except (m.RefusalError, m.TruncatedError) as e:
            print(f"    {label}: {type(e).__name__} — {e}")
    fb = m.read_response(fake_message("답", model="claude-opus-5", iterations=[NS(type="message"), NS(type="fallback_message")]))
    print(f"    폴백: fell_back={fb.fell_back}, served_by={fb.served_by}, 건너뛴 블록 {fb.skipped_blocks}개")

    print("[5] 호출 전 비용 추정 (TODO 5) — 가짜 count_tokens(입력 1,200 토큰으로 고정)")
    est = m.estimate_cost(sdk, sreq, expected_output_tokens=300)
    print(f"    입력 {est.input_tokens} 토큰, 예상(출력 300) ${est.expected_usd:.4f}, 최대(출력 {sreq['max_tokens']}) ${est.max_usd:.4f}"
          "  (claude-opus-5-5 $4/$20, 작성 시점 2026-10)")

    print("[6] (선택) 오류 분류 (TODO 6)")
    try:
        import anthropic
        import httpx2

        r = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
        for e in (anthropic.RateLimitError("x", response=httpx2.Response(429, request=r, headers={"retry-after": "7"}), body=None),
                  anthropic.OverloadedError("x", response=httpx2.Response(529, request=r), body=None),
                  anthropic.BadRequestError("x", response=httpx2.Response(400, request=r), body=None),
                  anthropic.APITimeoutError(request=r)):
            print(f"    {type(e).__name__:20s} → {m.classify_error(e)}")
    except ImportError:
        print("    건너뜀: anthropic 미설치")

    print(f"[7] 사용량 기록 — P2-01B log_usage 재사용: {args.log}")
    m.log_usage(Path(args.log), m.to_usage(out.final, latency_ms=out.total_ms, ttft_ms=out.ttft_ms), datetime.now().isoformat(timespec="seconds"))
    print("    " + Path(args.log).read_text(encoding="utf-8").splitlines()[-1].split(",", 1)[1])


if __name__ == "__main__":
    main()

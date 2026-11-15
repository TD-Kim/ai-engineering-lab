"""(선택) 실제 Claude API 호출 — 사용자 PC 에서 API 키가 있을 때 실행. 유료(토큰 과금).

이 실습 함수로 요청을 만들고(build_request·with_json_schema), 호출 전에 토큰을 세고(estimate_cost),
구조화 출력 1 회 + 스트리밍 1 회를 실제로 보낸 뒤, 결과·usage·TTFT 를 출력하고 사용량을 CSV 에 남긴다.

준비: pip install -r requirements/part2.txt
      cp .env.example .env (PowerShell: Copy-Item .env.example .env) 후 ANTHROPIC_API_KEY 입력 (커밋 금지)
실행: python part2_llm/llm_api/basics/call_claude.py
모델: .env 또는 환경 변수 LLM_MODEL (기본 claude-opus-5-5)
REF: claude-api 스킬 python/claude-api/README.md·streaming.md (작성 시점 2026-11)
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB_DIR))

import solution as m
from run_demo import SENTENCE

LOG = LAB_DIR / "outputs" / "usage_real.csv"


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(LAB_DIR.parents[2] / ".env")
    except ImportError:
        pass
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("건너뜀: ANTHROPIC_API_KEY 가 없습니다(.env 에 넣고 다시 실행).")
        return
    import anthropic

    model = os.environ.get("LLM_MODEL", m.DEFAULT_MODEL)
    client = anthropic.Anthropic(timeout=60.0, max_retries=2)
    system = "당신은 문장에서 (주어, 관계, 목적어) triple 을 추출합니다. 문장에 없는 사실은 만들지 마세요."
    req = m.with_json_schema(m.build_request(m.build_messages([], SENTENCE), system=system, model=model, max_tokens=4000), m.Extraction)
    if model in m.PRICES_PER_MTOK:
        est = m.estimate_cost(client, req, expected_output_tokens=300)
        print(f"[추정] 입력 {est.input_tokens} 토큰, 최대 ${est.max_usd:.4f}")
    try:
        start = time.perf_counter()
        message = client.beta.messages.create(**req)
        latency = (time.perf_counter() - start) * 1000
        try:
            result = m.parse_structured(message, m.Extraction)
            print("[구조화 출력]", [t.model_dump() for t in result.triples])
        except (m.RefusalError, m.TruncatedError, m.StructuredOutputError) as e:
            print(f"[구조화 출력] {type(e).__name__}: {e}")
        view = m.read_response(message)
        print(f"  stop_reason={view.stop_reason} served_by={view.served_by} fell_back={view.fell_back} usage=({message.usage.input_tokens}, {message.usage.output_tokens}) {latency:.0f} ms")
        m.log_usage(LOG, m.to_usage(message, latency), datetime.now().isoformat(timespec="seconds"))

        sreq = m.build_request(m.build_messages([], "훈민정음을 두 문장으로 소개해 주세요."), model=model, max_tokens=2000)
        start = time.perf_counter()
        out = m.consume_stream(client.beta.messages.stream(**sreq), start, on_text=lambda t: print(t, end="", flush=True))
        print(f"\n[스트리밍] TTFT {out.ttft_ms:.0f} ms (thinking 포함), 전체 {out.total_ms:.0f} ms, stop_reason={out.final.stop_reason}, "
              f"usage=({out.final.usage.input_tokens}, {out.final.usage.output_tokens})")
        m.log_usage(LOG, m.to_usage(out.final, out.total_ms, out.ttft_ms), datetime.now().isoformat(timespec="seconds"))
    except anthropic.APIError as e:
        print(f"[오류] {type(e).__name__}: {m.classify_error(e)}")
        return
    print(f"사용량 기록: {LOG}  — 이 출력(실제 값)만 experiments 기록에 옮겨 적으세요.")


if __name__ == "__main__":
    main()

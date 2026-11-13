"""P2-01B 데모: 공급자 인터페이스·스트림 파싱·폴백·사용량 기록·비용 비교 (네트워크·서버·키 불필요).

실행: python part2_llm/local_llm_serving/run_demo.py [--impl starter] [--log 경로]
- 공급자는 LLM_PROVIDER(기본 fake). ollama/claude 로 바꾸면 실제 서버·API 를 부른다(사용자 PC).
- [2] 의 스트림은 Ollama 문서의 응답 예시 숫자로 만든 "예시 데이터"이며 측정값이 아니다.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent


def load(impl: str):
    spec = importlib.util.spec_from_file_location(f"p2_01b_{impl}", LAB_DIR / f"{impl}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclass 가 자기 모듈을 찾을 수 있게 먼저 등록
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["solution", "starter"], default="solution")
    ap.add_argument("--log", default="part2_llm/local_llm_serving/outputs/usage.csv")  # 저장소 루트 기준(git 에 안 올라감)
    args = ap.parse_args()
    m = load(args.impl)

    print("[1] Ollama /api/chat 요청 본문 (TODO 1)")
    print("    " + json.dumps(m.build_chat_payload("qwen2.5:0.5b", "RAG 를 한 문장으로", stream=True, max_tokens=128), ensure_ascii=False))

    print("[2] 스트림 파싱 (TODO 2) — 예시 데이터: 조각 3개 + 마지막 줄(Ollama 문서 예시 숫자 eval_count 282, eval_duration 4,535,599,000ns)")
    lines = [json.dumps({"message": {"content": c}, "done": False}).encode() for c in ("검색", "한 문서로", " 답한다")]
    lines.append(json.dumps({"message": {"content": ""}, "done": True, "prompt_eval_count": 26,
                             "eval_count": 282, "eval_duration": 4535599000}).encode())
    ticks = iter([0.35, 0.40, 0.45, 4.89])  # 가짜 시계(초): 요청 시각 0
    r = m.parse_chat_stream(lines, start=0.0, clock=lambda: next(ticks))
    print(f"    text={r.text!r}  TTFT {r.ttft_ms:.0f} ms  전체 {r.total_ms:.0f} ms  출력 {r.output_tokens} 토큰  {r.tokens_per_sec:.1f} 토큰/초")

    provider = os.environ.get("LLM_PROVIDER", "fake")
    print(f"[3] 공통 인터페이스 (TODO 3): LLM_PROVIDER={provider}")
    client = m.make_client()
    text, usage = client.generate("로컬 LLM 과 API 의 차이를 한 줄로")
    print(f"    {usage.provider}/{usage.model}: {text[:60]!r}  (입력 {usage.input_tokens}, 출력 {usage.output_tokens} 토큰)")

    print("[4] 폴백 (TODO 4): 주 공급자(장애) → 보조 공급자")
    clients = [m.FakeClient(model="primary", fail=True), m.FakeClient(model="backup")]
    text, used, errors = m.generate_with_fallback(clients, "주문 상태 알려줘")
    print(f"    답한 공급자: {used.provider}/{used.model}, 실패 기록: {errors}")

    print(f"[5] 사용량 기록 (TODO 5): {args.log}")
    now = datetime.now().isoformat(timespec="seconds")
    for u in (usage, used):
        m.log_usage(Path(args.log), u, now)
    print("    " + Path(args.log).read_text(encoding="utf-8").splitlines()[0])

    print("[6] (선택) 100만 토큰당 비용 (TODO 6) — 모두 가정값, 측정 아님")
    # claude-opus-5-5 단가(입력 $4·출력 $20 / 1M), 작성 시점 2026-10 (P1-01 표, 공식 가격 페이지 재확인)
    print(f"    API claude-opus-5-5, 출력만 100만 토큰           : ${m.api_cost_per_mtok(4.00, 20.00, output_ratio=1.0):.2f}")
    print(f"    API claude-opus-5-5, 입력 80%·출력 20% 섞인 100만 : ${m.api_cost_per_mtok(4.00, 20.00, output_ratio=0.2):.2f}")
    print("    로컬은 '출력 100만 토큰을 생성하는 시간' 기준(입력 처리 시간은 생략) → 위 첫 줄(출력만)과 비교:")
    for tps in (30, 60, 120):
        local = m.local_cost_per_mtok(tps, watts=250, price_per_kwh=0.15)
        with_hw = m.local_cost_per_mtok(tps, watts=250, price_per_kwh=0.15, hardware_per_hour=0.05)
        print(f"    로컬 {tps:3d} 토큰/초 가정, 250W, $0.15/kWh: 전기만 ${local:.2f}, 장비 감가 $0.05/시간 포함 ${with_hw:.2f}")
    print("    ※ 로컬 품질은 큰 API 모델과 다르다 — 같은 토큰 수라도 같은 결과가 아니다(비용만으로 고르지 않는다).")


if __name__ == "__main__":
    main()

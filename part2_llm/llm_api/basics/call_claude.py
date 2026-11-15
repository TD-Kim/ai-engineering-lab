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

import importlib.util
import os
import sys
import time
from datetime import datetime
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent


def _load_sibling(filename: str):
    """같은 폴더의 파일을 "실습별 고유 이름"으로 불러온다.

    `from solution import ...` 처럼 흔한 이름으로 불러오면, 다른 실습의 solution 모듈이 이미 sys.modules 에 있을 때
    그것을 잘못 받는다(한 프로세스에서 여러 실습 스크립트를 쓸 때, 테스트에서 실제로 겪은 문제). sys.path 도 건드리지 않는다.
    """
    name = f"lab_{LAB_DIR.parent.name}_{LAB_DIR.name}_{Path(filename).stem}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, LAB_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclass 가 자기 모듈을 찾을 수 있게 먼저 등록
    spec.loader.exec_module(mod)
    return mod


m = _load_sibling("solution.py")
SENTENCE = _load_sibling("run_demo.py").SENTENCE

LOG = LAB_DIR / "outputs" / "usage_real.csv"


def fmt_ms(value: float | None) -> str:
    """TTFT 등 없을 수 있는 시간 값을 표시. 텍스트 조각 없이 끝난 스트림(즉시 거절 등)은 TTFT 가 None 이다."""
    return "측정 없음" if value is None else f"{value:.0f} ms"


def report_stream(out) -> tuple[str, bool]:
    """스트리밍 결과 요약 한 줄과, 화면에 찍힌 텍스트를 정상 답으로 써도 되는지(ok)를 돌려준다.

    스트림은 텍스트를 받는 대로 화면에 찍으므로, 끝난 뒤 최종 메시지로 거절·잘림을 확인해
    "위 텍스트는 정상 답이 아님"을 분명히 알린다(폴백 모델까지 거절하면 받은 부분을 답으로 쓰지 않는다).
    """
    usage = out.final.usage
    base = (f"TTFT {fmt_ms(out.ttft_ms)} (thinking 포함), 전체 {fmt_ms(out.total_ms)}, "
            f"stop_reason={out.final.stop_reason}, usage=({usage.input_tokens}, {usage.output_tokens})")
    try:
        view = m.ensure_complete(out.final)
    except m.RefusalError as e:
        return f"[스트리밍] 거절됨({e.category}) — 위에 찍힌 텍스트는 정상 답이 아닙니다. {base}", False
    except m.TruncatedError:
        return f"[스트리밍] max_tokens 에서 잘림 — 위 텍스트는 미완성입니다. {base}", False
    note = f", 폴백 모델 {view.served_by} 가 답함" if view.fell_back else ""
    return f"[스트리밍] {base}{note}", True


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
        line, _ok = report_stream(out)
        print("\n" + line)
        m.log_usage(LOG, m.to_usage(out.final, out.total_ms, out.ttft_ms), datetime.now().isoformat(timespec="seconds"))
    except anthropic.APIError as e:
        print(f"[오류] {type(e).__name__}: {m.classify_error(e)}")
        return
    print(f"사용량 기록: {LOG}  — 이 출력(실제 값)만 experiments 기록에 옮겨 적으세요.")


if __name__ == "__main__":
    main()

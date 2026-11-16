"""(선택) 실제 프롬프트 캐싱 측정 — 사용자 PC 에서 API 키가 있을 때 실행. 유료(토큰 과금, 호출 수를 적게 잡았다).

같은 앞부분(지시문 + 규정 문서)에 질문만 바꿔 3번 호출한다.
  1) 첫 요청 → 캐시 쓰기 기대   2) 질문만 다른 요청 → 캐시 읽기 기대   3) 시스템 프롬프트에 현재 시각 → 다시 쓰기 기대
각 요청의 usage(정가 입력·캐시 쓰기·읽기)·지연·비용을 출력하고 CSV 에 남긴 뒤 집계한다.
--batch 를 주면 질문 2개를 Batches API 로도 보낸다(최대 24시간 걸릴 수 있음 → 기다리지 않고 id 를 출력, --batch-id 로 나중에 확인).

준비: pip install -r requirements/part2.txt
      cp .env.example .env (PowerShell: Copy-Item .env.example .env) 후 ANTHROPIC_API_KEY 입력 (커밋 금지)
실행: python part2_llm/llm_api/cost_latency/measure_caching.py [--batch] [--batch-id msgbatch_...]
모델: .env 또는 환경 변수 LLM_MODEL (기본 claude-opus-5-5)
REF: claude-api 스킬 python/claude-api/README.md "Prompt Caching", python/claude-api/batches.md (작성 시점 2026-11-16)
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from datetime import datetime
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent


def _load_sibling(filename: str):
    """같은 폴더의 파일을 "실습별 고유 이름"(lab_<상위>_<실습>_<파일>)으로 불러온다.

    `from solution import ...` 처럼 흔한 이름으로 불러오면 다른 실습의 solution 이 이미 sys.modules 에 있을 때
    그것을 잘못 받는다. sys.path 도 건드리지 않는다.
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
_demo = _load_sibling("run_demo.py")
INSTRUCTIONS, POLICY, QUESTIONS = _demo.INSTRUCTIONS, _demo.POLICY, _demo.QUESTIONS

LOG = LAB_DIR / "outputs" / "usage_real.csv"
MAX_TOKENS = 2000      # 짧은 답 + thinking. 비용 상한을 작게(P2-02 TODO 5)
EFFORT = "low"         # 이 측정은 답의 품질이 아니라 usage 를 보는 것이 목적


def show_batch_results(client, batch_id: str, sent_ids: list[str], model: str) -> None:
    batch = client.messages.batches.retrieve(batch_id)
    print(f"[배치] {batch_id} 상태 {batch.processing_status}, counts={batch.request_counts}")
    if batch.processing_status != "ended":
        print("  아직 끝나지 않았습니다. 나중에 --batch-id 로 다시 확인하세요(최대 24시간).")
        return
    out = m.collect_batch_results(client.messages.batches.results(batch_id), sent_ids)
    for cid, usage in sorted(out.usages.items()):
        cu = m.read_cache_usage(usage)
        money = f"${m.cache_cost(cu, model, batch=True).actual_usd:.5f}" if model in m.PRICES else "단가표에 없는 모델"
        print(f"  {cid}: {cu.status} 정가 {cu.input_tokens} 쓰기 {cu.cache_write_5m + cu.cache_write_1h} 읽기 {cu.cache_read} "
              f"출력 {cu.output_tokens} → 배치 비용 {money}")
    print(f"  다시 보낼 것 {out.retry}, 고칠 것 {out.fix}, 취소 {out.canceled}, 빠진 것 {out.missing}, 보내지 않은 id {out.unexpected}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", action="store_true", help="질문 2개를 Batches API 로 보낸다(유료, 결과는 나중에)")
    ap.add_argument("--batch-id", help="이전에 보낸 배치의 결과 확인")
    args = ap.parse_args()
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
    client = anthropic.Anthropic(timeout=120.0, max_retries=2)
    batch_ids = [f"policy-q{i}" for i in range(2)]
    if args.batch_id:
        show_batch_results(client, args.batch_id, batch_ids, model)
        return

    kwargs = {"model": model, "max_tokens": MAX_TOKENS, "effort": EFFORT}
    reqs = [m.build_cached_request(INSTRUCTIONS, [POLICY], QUESTIONS[0], **kwargs),
            m.build_cached_request(INSTRUCTIONS, [POLICY], QUESTIONS[1], **kwargs),
            m.build_cached_request(f"현재 시각 {datetime.now().isoformat(timespec='seconds')}. " + INSTRUCTIONS, [POLICY],
                                   QUESTIONS[2], **kwargs)]
    try:
        n_in = client.messages.count_tokens(model=model, system=reqs[0]["system"], messages=reqs[0]["messages"],
                                            output_config=reqs[0]["output_config"]).input_tokens   # P2-02 estimate_cost 와 같게
        print(f"[추정] 요청 1 입력 {n_in} 토큰 (캐시 최소 길이 {m.MIN_CACHEABLE_TOKENS} 토큰 — 미달이면 캐시가 조용히 안 된다)")
        labels = ["1) 첫 요청", "2) 질문만 다름", "3) 시스템에 현재 시각"]
        for label, req in zip(labels, reqs, strict=True):
            start = time.perf_counter()
            msg = client.beta.messages.create(**req)
            latency = (time.perf_counter() - start) * 1000
            view = m.read_response(msg)
            cu = m.read_cache_usage(msg.usage)
            cost = m.cache_cost(cu, model) if model in m.PRICES else None
            money = f"${cost.actual_usd:.5f} (캐시 없었다면 ${cost.no_cache_usd:.5f})" if cost else "단가표에 없는 모델"
            print(f"{label}: {cu.status} 정가 {cu.input_tokens} 쓰기 {cu.cache_write_5m}+{cu.cache_write_1h}(1h) 읽기 {cu.cache_read} "
                  f"출력 {cu.output_tokens} | {latency:.0f} ms | {money} | stop_reason={view.stop_reason} served_by={view.served_by}")
            usage = m.Usage("claude", msg.model, msg.usage.input_tokens, msg.usage.output_tokens, latency, None, msg.stop_reason)
            m.log_cached_usage(LOG, usage, cu, datetime.now().isoformat(timespec="seconds"))
        d = m.first_divergence(reqs[1], reqs[2])
        print(f"[원인] 2) vs 3) 첫 차이 {d.location} (세그먼트 안 {d.offset}번째 글자): {d.a_snippet!r} vs {d.b_snippet!r}")

        if args.batch:
            items = [(cid, m.build_cached_request(INSTRUCTIONS, [POLICY], q, fallbacks=False, **kwargs))
                     for cid, q in zip(batch_ids, QUESTIONS, strict=False)]
            batch = client.messages.batches.create(requests=m.build_batch(items))
            print(f"[배치] 보냄: {batch.id} (상태 {batch.processing_status}). 결과 확인: --batch-id {batch.id}")
    except anthropic.APIError as e:
        print(f"[오류] {type(e).__name__}: {e}")
        return
    rep = m.summarize_usage_log(LOG)
    print(f"[집계] {LOG}: 요청 {rep.requests}, 캐시 적중률 {rep.cache_hit_rate:.1%}, 평균 {rep.avg_latency_ms:.0f} ms, "
          f"p95 {rep.p95_latency_ms:.0f} ms, 총비용 ${rep.total_usd:.5f}")
    print("이 출력(실제 값)만 experiments 기록에 옮겨 적으세요. (로그 파일은 누적되므로 새로 재려면 지우고 실행)")


if __name__ == "__main__":
    main()

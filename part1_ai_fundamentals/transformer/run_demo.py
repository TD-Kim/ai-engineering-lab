"""P1-05 데모: 작은 예시 어텐션 손계산 표 → √d_k 효과 → 길이 T 별 점수 행렬·시간 → KV 캐시.

실행: python part1_ai_fundamentals/transformer/run_demo.py [--impl starter] [--max-t 2048]
CPU 수 초. 시간 수치는 PC·실행마다 다르다(모양·원소 수는 항상 같다).
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import time
from pathlib import Path

import torch

LAB_DIR = Path(__file__).resolve().parent
TOKENS = ["나는", "커피를", "마신다"]


def load(impl: str):
    spec = importlib.util.spec_from_file_location(f"p1_05_{impl}", LAB_DIR / f"{impl}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def table(name: str, m: torch.Tensor) -> None:
    print(f"    {name}")
    print("      " + "".join(f"{t:>8s}" for t in TOKENS))
    for t, row in zip(TOKENS, m.tolist(), strict=True):
        print(f"      {t:<6s}" + "".join(f"{v:8.3f}" for v in row))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["solution", "starter"], default="solution")
    ap.add_argument("--max-t", type=int, default=2048)
    args = ap.parse_args()
    m = load(args.impl)
    torch.set_num_threads(1)  # 시간 비교를 덜 흔들리게: 스레드 하나

    print("[1] 손계산 예시: 토큰 3개, D = d_k = 2, 가중치 = 단위행렬 → Q = K = V = x")
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    eye = torch.eye(2)
    q, k, v = m.make_qkv(x, eye, eye, eye)
    print(f"    x = {x.tolist()}")
    table("Q @ Kᵀ (원점수)", q @ k.T)
    table("/ √2", q @ k.T / math.sqrt(2))
    out, w = m.scaled_dot_product_attention(q, k, v)
    table("softmax → 가중치 (행 합 = 1)", w)
    print(f"    출력 = 가중치 @ V = {[[round(a, 3) for a in r] for r in out.tolist()]}")
    out_c, w_c = m.scaled_dot_product_attention(q, k, v, m.causal_mask(3))
    table("causal 마스크 적용 가중치 (미래 = 0)", w_c)
    print(f"    출력(causal) = {[[round(a, 3) for a in r] for r in out_c.tolist()]}")

    print("[2] √d_k 로 나누는 효과: d_k = 64, 무작위 q·k 16개 (행마다 가장 큰 가중치의 평균)")
    g = torch.Generator().manual_seed(0)
    qq, kk = torch.randn(16, 64, generator=g), torch.randn(16, 64, generator=g)
    raw = qq @ kk.T
    print(f"    나누지 않음: 점수 표준편차 {raw.std():.2f}, 최대 가중치 평균 {torch.softmax(raw, -1).max(-1).values.mean():.3f}")
    print(f"    / √64 = /8 : 점수 표준편차 {(raw / 8).std():.2f}, 최대 가중치 평균 {torch.softmax(raw / 8, -1).max(-1).values.mean():.3f}")

    print("[3] 길이 T 별 causal 어텐션 1회 (d_k = 64, float32, 스레드 1, 5회 중앙값)")
    lengths = [t for t in (128, 256, 512, 1024, 2048, 4096) if t <= args.max_t]
    rows = m.measure_attention_cost(lengths)
    print("         T   점수 행렬 모양   원소 수       MB       ms   ms 비율(앞 T 대비)")
    prev = None
    for r in rows:
        ratio = f"{r['ms'] / prev:6.1f}배" if prev else "     —"
        print(f"    {r['T']:6d}   {r['scores_shape']!s:14s} {r['scores_elements']:10,d} {r['scores_mb']:8.1f} {r['ms']:8.2f}   {ratio}")
        prev = r["ms"]

    print("[4] KV 캐시: 토큰 256개를 한 개씩 생성하는 흉내 (D = d_k = 512)")
    g = torch.Generator().manual_seed(1)
    x = torch.randn(256, 512, generator=g)
    wq, wk, wv = (torch.randn(512, 512, generator=g) / math.sqrt(512) for _ in range(3))
    for use_cache in (False, True):
        start = time.perf_counter()
        out, n_rows = m.decode_with_kv_cache(x, wq, wk, wv, use_cache=use_cache)
        ms = (time.perf_counter() - start) * 1000
        print(f"    use_cache={use_cache!s:5s}: K·V 계산 행 {n_rows:6,d}, {ms:7.1f} ms")
    q, k, v = m.make_qkv(x, wq, wk, wv)
    full, _ = m.scaled_dot_product_attention(q, k, v, m.causal_mask(256))
    print(f"    한 번에 계산한 causal 어텐션과 최대 차이: {(out - full).abs().max():.2e}")


if __name__ == "__main__":
    main()

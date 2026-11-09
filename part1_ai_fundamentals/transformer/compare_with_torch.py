"""P1-05 Level 3: 직접 구현한 어텐션을 PyTorch 내장 구현과 같은 가중치로 비교한다.

사용법:
    python part1_ai_fundamentals/transformer/compare_with_torch.py --impl solution

비교 항목
1. SelfAttention(단일 헤드) vs F.scaled_dot_product_attention
   - 같은 W_q/W_k/W_v 로 만든 q, k, v 에 내장 함수를 적용하고 같은 W_o 를 곱해 출력 비교 (마스크 없음 / causal)
2. MultiHeadAttention vs nn.MultiheadAttention(batch_first=True, bias=False)
   - 직접 구현의 qkv.weight (3D, D) 를 in_proj_weight 로, out_proj.weight 를 그대로 복사
   - 출력과 head 별 어텐션 가중치(average_attn_weights=False) 비교 (마스크 없음 / causal / causal + padding)
   - 마스크 규약 차이: 직접 구현은 True = 볼 수 있음, nn.MultiheadAttention 의 bool attn_mask·key_padding_mask 는 True = 가림
     → 비교할 때 ~ 로 뒤집어 넘긴다.

REF:
- https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
- https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
(작성 환경에서 pytorch.org 접속이 막혀 링크는 미확인)
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

LAB_DIR = Path(__file__).resolve().parent


def compare(impl, d_model: int = 16, num_heads: int = 4, batch: int = 2, seq_len: int = 6, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    x = torch.randn(batch, seq_len, d_model)
    results: dict[str, float] = {}

    # 1) 단일 헤드 vs F.scaled_dot_product_attention
    sa = impl.SelfAttention(d_model).eval()
    for causal in (False, True):
        mine, _ = sa(x, causal=causal)
        q, k, v = sa.W_q(x), sa.W_k(x), sa.W_v(x)
        ref = sa.W_o(F.scaled_dot_product_attention(q, k, v, is_causal=causal))
        results[f"self_attention_vs_sdpa{'_causal' if causal else ''}"] = (mine - ref).abs().max().item()

    # 2) MultiHeadAttention vs nn.MultiheadAttention
    mha = impl.MultiHeadAttention(d_model, num_heads).eval()
    ref_mha = nn.MultiheadAttention(d_model, num_heads, bias=False, batch_first=True).eval()
    with torch.no_grad():
        ref_mha.in_proj_weight.copy_(mha.qkv.weight)  # (3D, D): [W_q; W_k; W_v] 같은 배치
        ref_mha.out_proj.weight.copy_(mha.out_proj.weight)

    causal_block = ~torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))  # True = 가림 (PyTorch 규약)
    padding = torch.ones(batch, seq_len, dtype=torch.bool)
    padding[1, -2:] = False  # 두 번째 문장의 마지막 2 개는 패딩 (직접 구현 규약: False = 가림)
    cases = {
        "none": ({}, {}),
        "causal": ({"causal": True}, {"attn_mask": causal_block}),
        "causal_padding": ({"causal": True, "padding_mask": padding},
                           {"attn_mask": causal_block, "key_padding_mask": ~padding}),
    }
    for name, (mine_kw, ref_kw) in cases.items():
        mine_out, mine_w = mha(x, **mine_kw)
        ref_out, ref_w = ref_mha(x, x, x, need_weights=True, average_attn_weights=False, **ref_kw)
        valid = padding.unsqueeze(-1) if "padding" in name else torch.ones(batch, seq_len, 1, dtype=torch.bool)
        # 패딩 query 위치의 출력은 의미가 없으므로(어차피 loss 에서 제외) 실제 토큰 위치만 비교한다.
        results[f"mha_output_{name}"] = ((mine_out - ref_out).abs() * valid).max().item()
        results[f"mha_weights_{name}"] = (mine_w - ref_w).abs().max().item()
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=["solution", "starter"], default="solution")
    args = parser.parse_args()
    sys.path.insert(0, str(LAB_DIR))
    impl = importlib.import_module(args.impl)
    try:
        results = compare(impl)
    except NotImplementedError as exc:
        print(f"[미구현] {exc} — {args.impl}.py 의 TODO 를 먼저 채우세요")
        return
    print(f"[torch] {torch.__version__}")
    for name, diff in results.items():
        status = "일치" if diff < 1e-5 else "불일치"
        print(f"  {name:<32} 최대 절대 오차 {diff:.2e}  {status}")


if __name__ == "__main__":
    main()

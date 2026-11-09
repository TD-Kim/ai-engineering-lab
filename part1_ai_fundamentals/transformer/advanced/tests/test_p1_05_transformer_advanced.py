"""P1-05 테스트. 기본은 solution 검사, `IMPL=starter pytest ...` 로 직접 구현을 검사한다.

작고 결정적인 입력(seed 고정)만 쓰며 CPU 에서 수 초 안에 끝난다. 외부 다운로드 없음.
마스크 규약: True(1) = 볼 수 있음, False(0) = 가림.
`impl` 픽스처는 part1_ai_fundamentals/conftest.py 가 파일 경로 기준으로 불러온다.
"""
import math
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

LAB_DIR = Path(__file__).resolve().parents[1]
CORPUS = LAB_DIR.parents[1] / "data" / "corpus_ko_en.txt"
CPU = torch.device("cpu")


def _x(b=2, t=5, d=8, seed=0):
    torch.manual_seed(seed)
    return torch.randn(b, t, d)


# ----- Level 1: Q/K/V, 스케일드 닷-프로덕트 어텐션, 마스크 -----
def test_qkv_projections_shapes(impl):
    sa = impl.SelfAttention(8, d_k=4)
    for name in ("W_q", "W_k", "W_v"):
        layer = getattr(sa, name)
        assert isinstance(layer, nn.Linear) and layer.weight.shape == (4, 8) and layer.bias is None
    assert sa.W_o.weight.shape == (8, 4)
    out, w = sa(_x())
    assert out.shape == (2, 5, 8) and w.shape == (2, 5, 5)
    torch.testing.assert_close(w.sum(-1), torch.ones(2, 5))  # 각 query 행은 확률 분포
    assert sa.last_attn_weights is not None and not sa.last_attn_weights.requires_grad


@pytest.mark.parametrize("causal", [False, True])
def test_self_attention_matches_sdpa(impl, causal):
    """TODO 7~9: sqrt(d_k) 스케일·softmax·가중합이 PyTorch 내장 함수와 같아야 한다."""
    sa = impl.SelfAttention(8, d_k=4).eval()
    x = _x()
    out, _ = sa(x, causal=causal)
    q, k, v = sa.W_q(x), sa.W_k(x), sa.W_v(x)
    ref = sa.W_o(F.scaled_dot_product_attention(q, k, v, is_causal=causal))
    torch.testing.assert_close(out, ref)


def test_scores_are_scaled_by_sqrt_dk(impl):
    """스케일을 빼먹으면 softmax 가 더 뾰족해진다: 직접 계산한 가중치와 비교."""
    sa = impl.SelfAttention(16, d_k=16).eval()
    x = _x(1, 4, 16)
    _, w = sa(x)
    q, k = sa.W_q(x), sa.W_k(x)
    torch.testing.assert_close(w, torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(16), dim=-1))


def test_causal_mask_blocks_future(impl):
    sa = impl.SelfAttention(8).eval()
    x = _x()
    out, w = sa(x, causal=True)
    assert torch.triu(w, diagonal=1).abs().sum() == 0  # 대각선 위(미래)는 0
    x2 = x.clone()
    x2[:, 3:] = torch.randn(2, 2, 8)  # 미래(3, 4 번)만 바꿔도
    out2, _ = sa(x2, causal=True)
    torch.testing.assert_close(out[:, :3], out2[:, :3])  # 0~2 번 출력은 그대로


def test_padding_mask_hides_keys(impl):
    sa = impl.SelfAttention(8).eval()
    x = _x()
    pad = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    out, w = sa(x, padding_mask=pad)
    assert w[1, :, 3:].abs().sum() == 0  # 두 번째 문장의 패딩 key 는 아무도 보지 않음
    x2 = x.clone()
    x2[1, 3:] = 100.0  # 패딩 위치 내용을 바꿔도
    out2, _ = sa(x2, padding_mask=pad)
    torch.testing.assert_close(out[1, :3], out2[1, :3])  # 실제 토큰 출력은 그대로


def test_fully_masked_row_gives_nan(impl):
    """README 9절: 한 query 행의 key 가 전부 가려지면 softmax(-inf, ..., -inf) = NaN (재현 테스트)."""
    sa = impl.SelfAttention(8).eval()
    pad = torch.tensor([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]])  # 두 번째 문장은 전부 패딩
    out, w = sa(_x(), padding_mask=pad)
    assert torch.isnan(w[1]).all() and torch.isnan(out[1]).all()
    assert not torch.isnan(out[0]).any()  # 다른 문장은 영향 없음


def test_finfo_min_fill_avoids_nan(impl):
    """README 9절 해결법 ②: -inf 대신 dtype 의 가장 작은 유한값으로 채우면 전부 가려진 행도 NaN 이 아니다(균등 분포)."""
    sa = impl.SelfAttention(8).eval()
    x = _x()
    q, k = sa.W_q(x), sa.W_k(x)
    scores = q @ k.transpose(-2, -1) / math.sqrt(sa.d_k)
    mask = torch.tensor([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]], dtype=torch.bool)[:, None, :]  # (B, 1, T)
    filled = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(filled, dim=-1)
    assert not torch.isnan(weights).any()
    torch.testing.assert_close(weights[1], torch.full((5, 5), 0.2))  # 전부 가려진 문장: 균등 분포
    _, ref = sa(x, padding_mask=mask[:, 0].long())
    torch.testing.assert_close(weights[0], ref[0])  # 일부만 가려진 문장은 -inf 방식과 같은 결과


# ----- Level 2: Multi-Head, FFN, 위치 인코딩, Residual + LayerNorm -----
def test_mha_split_heads_and_divisibility(impl):
    mha = impl.MultiHeadAttention(8, 2)
    assert mha.d_k == 4 and mha.qkv.weight.shape == (24, 8) and mha.out_proj.weight.shape == (8, 8)
    assert mha._split_heads(_x()).shape == (2, 2, 5, 4)  # (B, H, T, d_k)
    out, w = mha(_x())
    assert out.shape == (2, 5, 8) and w.shape == (2, 2, 5, 5)
    with pytest.raises(ValueError):
        impl.MultiHeadAttention(10, 3)


@pytest.mark.parametrize("case", ["none", "causal", "causal_padding"])
def test_mha_matches_nn_multihead_attention(impl, case):
    """Level 3 비교: 같은 가중치면 nn.MultiheadAttention 과 출력·head 별 가중치가 같아야 한다."""
    torch.manual_seed(0)
    d, h, t = 16, 4, 6
    x = torch.randn(2, t, d)
    mha = impl.MultiHeadAttention(d, h).eval()
    ref = nn.MultiheadAttention(d, h, bias=False, batch_first=True).eval()
    with torch.no_grad():
        ref.in_proj_weight.copy_(mha.qkv.weight)
        ref.out_proj.weight.copy_(mha.out_proj.weight)
    pad = torch.ones(2, t, dtype=torch.bool)
    pad[1, -2:] = False
    block = ~torch.tril(torch.ones(t, t, dtype=torch.bool))  # PyTorch: True = 가림
    mine_kw, ref_kw = {
        "none": ({}, {}),
        "causal": ({"causal": True}, {"attn_mask": block}),
        "causal_padding": ({"causal": True, "padding_mask": pad}, {"attn_mask": block, "key_padding_mask": ~pad}),
    }[case]
    out, w = mha(x, **mine_kw)
    ref_out, ref_w = ref(x, x, x, need_weights=True, average_attn_weights=False, **ref_kw)
    torch.testing.assert_close(out, ref_out)
    torch.testing.assert_close(w, ref_w)


def test_attention_without_position_is_permutation_equivariant(impl):
    """위치 정보가 없으면 어텐션은 순서를 모른다: 입력 순서를 섞으면 출력도 똑같이 섞일 뿐이다."""
    mha = impl.MultiHeadAttention(8, 2).eval()
    x = _x(1, 5, 8)
    perm = torch.tensor([3, 0, 4, 1, 2])
    out, _ = mha(x)
    out_perm, _ = mha(x[:, perm])
    torch.testing.assert_close(out_perm, out[:, perm])


def test_feed_forward_structure(impl):
    ffn = impl.FeedForward(8)
    linears = [m for m in ffn.modules() if isinstance(m, nn.Linear)]
    assert [tuple(m.weight.shape) for m in linears] == [(32, 8), (8, 32)]  # D → 4D → D
    assert any(isinstance(m, (nn.GELU, nn.ReLU)) for m in ffn.modules())
    assert ffn(_x()).shape == (2, 5, 8)


def test_sinusoidal_positional_encoding(impl):
    pe = impl.PositionalEncoding(8, max_len=20)
    assert "pe" in dict(pe.named_buffers()) and len(list(pe.parameters())) == 0  # 학습되지 않는 buffer
    table = pe.pe
    assert table.shape == (20, 8)
    torch.testing.assert_close(table[0], torch.tensor([0.0, 1.0] * 4))  # sin(0)=0, cos(0)=1
    torch.testing.assert_close(table[5, 0], torch.tensor(math.sin(5.0)))  # i=0: 주파수 1
    torch.testing.assert_close(table[5, 3], torch.tensor(math.cos(5.0 / 10000 ** (2 / 8))))  # i=1
    x = torch.zeros(1, 4, 8)
    torch.testing.assert_close(pe(x)[0], table[:4])


def test_block_residual_is_identity_when_sublayers_output_zero(impl):
    block = impl.TransformerBlock(8, 2).eval()
    assert sum(isinstance(m, nn.LayerNorm) for m in block.modules()) == 2
    with torch.no_grad():
        block.attn.out_proj.weight.zero_()
        last_linear = [m for m in block.ffn.modules() if isinstance(m, nn.Linear)][-1]
        last_linear.weight.zero_()
        last_linear.bias.zero_()
    x = _x()
    torch.testing.assert_close(block(x), x)  # x + 0 + 0 = x (Pre-LN 잔차 연결)
    assert impl.EncoderBlock is impl.TransformerBlock


# ----- Level 3: 미니 언어모델 -----
def _tiny_lm(impl, vocab=12):
    torch.manual_seed(0)
    return impl.MiniTransformerLM(vocab, d_model=16, num_heads=2, num_layers=2, max_len=16).eval()


def test_lm_shapes_initial_loss_and_causality(impl):
    lm = _tiny_lm(impl)
    ids = torch.randint(0, 12, (3, 10), generator=torch.Generator().manual_seed(0))
    logits = lm(ids)
    assert logits.shape == (3, 10, 12)
    loss = lm.loss(ids, ids)
    assert abs(loss.item() - math.log(12)) < 1.5  # 초기 loss ≈ ln V
    ids2 = ids.clone()
    ids2[:, 6:] = (ids2[:, 6:] + 1) % 12  # 미래 토큰만 바꿔도
    torch.testing.assert_close(lm(ids2)[:, :6], logits[:, :6])  # 앞 위치 logits 는 그대로
    with pytest.raises(ValueError):
        lm(torch.zeros(1, 17, dtype=torch.long))  # max_len 초과


def test_lm_uses_position(impl):
    """위치 인코딩이 실제로 쓰이는지: (a) 같은 토큰 반복 입력에서도 위치마다 logits 가 달라야 하고,
    (b) 위치 인코딩 표를 0 으로 바꾸면 출력이 바뀌어야 한다. 위치 인코딩을 빼면 (a)·(b) 모두 실패한다."""
    lm = _tiny_lm(impl)
    same = torch.full((1, 4), 5)  # [[a, a, a, a]]
    logits = lm(same)[0]  # (4, V)
    for t in range(1, 4):
        assert not torch.allclose(logits[t], logits[0], atol=1e-5), f"위치 {t} 와 0 의 logits 가 같다 → 위치 정보가 없음"
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    before = lm(ids)
    with torch.no_grad():
        lm.pos_enc.pe.zero_()
    assert not torch.allclose(lm(ids), before, atol=1e-5)  # (b)
    # 위치 표가 0 이면 같은 토큰 반복 입력은 위치를 구분하지 못한다 (위치 인코딩이 없을 때의 한계)
    no_pos = lm(same)[0]
    torch.testing.assert_close(no_pos[3], no_pos[0])


def test_token_embedding_scale_matches_positional_encoding(impl):
    """TODO 20·21: std = D^-0.5 초기화 × √D → 토큰 벡터 노름 ≈ √D, 위치 인코딩 노름 ≈ √(D/2) 와 같은 크기."""
    torch.manual_seed(0)
    d = 64
    lm = impl.MiniTransformerLM(225, d_model=d, num_heads=4, num_layers=1, max_len=32)
    tok_norm = (lm.tok_embed.weight * math.sqrt(d)).norm(dim=-1).mean().item()
    pe_norm = lm.pos_enc.pe.norm(dim=-1).mean().item()
    assert 0.5 * math.sqrt(d) < tok_norm < 1.5 * math.sqrt(d)  # N(0, 1) 그대로면 ≈ d = 64 로 실패
    assert tok_norm < 3 * pe_norm


def test_generate_length_and_reproducible(impl):
    lm = _tiny_lm(impl)
    start = torch.tensor([[1, 2, 3]])
    a = lm.generate(start, 20, generator=torch.Generator().manual_seed(0))
    b = lm.generate(start, 20, generator=torch.Generator().manual_seed(0))
    assert a.shape == (1, 23) and torch.equal(a, b) and torch.equal(a[:, :3], start)
    assert a.min() >= 0 and a.max() < 12  # max_len(16) 보다 길어져도 동작


def test_training_reduces_loss(impl):
    text = CORPUS.read_text(encoding="utf-8")
    r = impl.train_lm(text, steps=100, d_model=32, num_heads=2, num_layers=1, block_size=32,
                      eval_every=50, seed=0, device=CPU)
    first, last = r["history"][0], r["history"][-1]
    assert last[1] < 0.6 * first[1], r["history"]  # 학습 손실이 크게 줄어야 한다
    assert r["params"] == impl.count_parameters(r["model"]) > 0


def test_compare_script_reports_match(impl):
    spec = spec_from_file_location("p1_05_compare_with_torch", LAB_DIR / "compare_with_torch.py")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert all(diff < 1e-5 for diff in mod.compare(impl).values())

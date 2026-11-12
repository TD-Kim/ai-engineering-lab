"""P1-05 테스트. 기본은 solution 검사, `IMPL=starter pytest ...` 로 직접 구현을 검사한다.

시간 측정은 환경에 따라 흔들리므로 모양·원소 수 같은 결정적인 값만 단언한다.
반복은 모두 길이가 정해진 for 문이고 텐서 크기는 작다(무한 반복·메모리 위험 없음).
"""
import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

LAB_DIR = Path(__file__).resolve().parents[1]


def _rand(*shape, seed=0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


# --- TODO 1 ---------------------------------------------------------------
def test_make_qkv_shapes_and_values(impl):
    x, wq, wk, wv = _rand(5, 8), _rand(8, 4, seed=1), _rand(8, 4, seed=2), _rand(8, 6, seed=3)
    q, k, v = impl.make_qkv(x, wq, wk, wv)
    assert q.shape == (5, 4) and k.shape == (5, 4) and v.shape == (5, 6)
    torch.testing.assert_close(q, x @ wq)
    torch.testing.assert_close(v, x @ wv)
    qb, _, _ = impl.make_qkv(_rand(2, 5, 8), wq, wk, wv)            # 배치 (B, T, D)
    assert qb.shape == (2, 5, 4)


# --- TODO 2 ---------------------------------------------------------------
def test_attention_hand_example(impl):
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    out, w = impl.scaled_dot_product_attention(x, x, x)
    torch.testing.assert_close(w.sum(-1), torch.ones(3))
    e = math.exp(1 / math.sqrt(2))                                  # 점수 [0.707, 0, 0.707] → softmax
    torch.testing.assert_close(w[0], torch.tensor([e, 1.0, e]) / (2 * e + 1))
    torch.testing.assert_close(out, w @ x)


def test_attention_matches_torch_sdpa(impl):
    q, k, v = _rand(6, 8), _rand(6, 8, seed=1), _rand(6, 5, seed=2)
    out, _ = impl.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(out, F.scaled_dot_product_attention(q, k, v), atol=1e-5, rtol=1e-5)


def test_attention_mask_convention_matches_torch_bool_mask(impl):
    # PyTorch SDPA 의 bool attn_mask 도 True = 참여. 모든 행에 True 가 하나 이상 있게(대각선) 만든다.
    q, k, v = _rand(2, 6, 8), _rand(2, 6, 8, seed=1), _rand(2, 6, 4, seed=2)
    mask = _rand(6, 6, seed=3) > 0
    mask |= torch.eye(6, dtype=torch.bool)
    out, w = impl.scaled_dot_product_attention(q, k, v, mask)
    torch.testing.assert_close(out, F.scaled_dot_product_attention(q, k, v, attn_mask=mask), atol=1e-5, rtol=1e-5)
    assert torch.all(w[:, ~mask] == 0)                              # 가린 칸의 가중치는 정확히 0


def test_fully_masked_row_gives_nan(impl):
    # 한 행 전체를 가리면 softmax(-inf, ..., -inf) = 0/0 = NaN. (이 환경의 torch SDPA 는 0 을 돌려주므로 비교하지 않는다.)
    q = k = v = _rand(3, 4)
    mask = torch.ones(3, 3, dtype=torch.bool)
    mask[1] = False
    _, w = impl.scaled_dot_product_attention(q, k, v, mask)
    assert torch.isnan(w[1]).all() and not torch.isnan(w[0]).any()


# --- TODO 3 ---------------------------------------------------------------
def test_causal_mask(impl):
    m = impl.causal_mask(4)
    assert m.dtype == torch.bool and m.shape == (4, 4)
    assert m.tolist() == [[True, False, False, False], [True, True, False, False],
                          [True, True, True, False], [True, True, True, True]]


def test_causal_attention_matches_is_causal(impl):
    q, k, v = _rand(7, 8), _rand(7, 8, seed=1), _rand(7, 8, seed=2)
    out, w = impl.scaled_dot_product_attention(q, k, v, impl.causal_mask(7))
    torch.testing.assert_close(out, F.scaled_dot_product_attention(q, k, v, is_causal=True), atol=1e-5, rtol=1e-5)
    assert torch.all(torch.triu(w, diagonal=1) == 0)                # 미래 가중치 0
    torch.testing.assert_close(w[0], torch.tensor([1.0, 0, 0, 0, 0, 0, 0]))  # 첫 토큰은 자기 자신만


# --- TODO 4 ---------------------------------------------------------------
def test_measure_attention_cost_deterministic_fields(impl):
    rows = impl.measure_attention_cost([8, 16, 32], d_k=4, repeats=2)
    assert [r["T"] for r in rows] == [8, 16, 32]
    assert [r["scores_shape"] for r in rows] == [(8, 8), (16, 16), (32, 32)]
    assert [r["scores_elements"] for r in rows] == [64, 256, 1024]   # T 가 2 배면 4 배
    assert rows[2]["scores_mb"] == pytest.approx(1024 * 4 / 1e6)     # float32 = 4 바이트
    assert all(isinstance(r["ms"], float) and r["ms"] >= 0 for r in rows)
    with pytest.raises(ValueError):
        impl.measure_attention_cost([8], repeats=0)


# --- TODO 5 (선택) --------------------------------------------------------
@pytest.mark.parametrize("use_cache", [True, False])
def test_kv_cache_same_result_fewer_rows(impl, use_cache):
    t = 12
    x, wq, wk, wv = _rand(t, 8), _rand(8, 8, seed=1), _rand(8, 8, seed=2), _rand(8, 6, seed=3)
    out, rows = impl.decode_with_kv_cache(x, wq, wk, wv, use_cache=use_cache)
    q, k, v = x @ wq, x @ wk, x @ wv
    full = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(out, full, atol=1e-5, rtol=1e-5)
    assert rows == (t if use_cache else t * (t + 1) // 2)


# --- 데모 스크립트 ----------------------------------------------------------
def test_run_demo_runs(impl, capsys):
    spec = importlib.util.spec_from_file_location("p1_05_run_demo_test", LAB_DIR / "run_demo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.load = lambda _name: impl                                    # 현재 IMPL 로 실행
    argv, threads = sys.argv, torch.get_num_threads()
    sys.argv = ["run_demo.py", "--max-t", "256"]
    try:
        mod.main()
    finally:
        sys.argv = argv
        torch.set_num_threads(threads)                               # 데모가 바꾼 스레드 수 되돌리기
    out = capsys.readouterr().out
    assert "(256, 256)" in out and "K·V 계산 행    256" in out

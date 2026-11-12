"""P2-01 테스트. 기본은 solution 검사, `IMPL=starter pytest ...` 로 직접 구현을 검사한다.

다운로드·API 없음: 모델은 저장소 말뭉치로 만든 bigram LM 과 가짜 모델 함수. 난수는 torch.Generator 시드로 고정.
반복은 모두 길이가 정해진 for 문(생성은 max_new_tokens 로 끝남) — 무한 반복·메모리 위험 없음.
"""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

LAB_DIR = Path(__file__).resolve().parents[1]
Z = torch.tensor([2.0, 1.0, 0.0])


@pytest.fixture(scope="module")
def lm(impl):
    return impl.load_bigram_lm()


# --- TODO 1 ---------------------------------------------------------------
def test_softmax_with_temperature_hand_values(impl):
    torch.testing.assert_close(impl.softmax_with_temperature(Z, 1.0), torch.tensor([0.6652, 0.2447, 0.0900]), atol=1e-4, rtol=0)
    torch.testing.assert_close(impl.softmax_with_temperature(Z, 0.5), torch.tensor([0.8668, 0.1173, 0.0159]), atol=1e-4, rtol=0)
    torch.testing.assert_close(impl.softmax_with_temperature(Z, 2.0), torch.tensor([0.5065, 0.3072, 0.1863]), atol=1e-4, rtol=0)


def test_softmax_with_temperature_properties(impl):
    big = torch.tensor([1000.0, 999.0, -1000.0])                       # 넘침 없이 계산되어야 함
    p = impl.softmax_with_temperature(big, 1.0)
    assert not torch.isnan(p).any() and float(p.sum()) == pytest.approx(1.0)
    assert impl.entropy_bits(impl.softmax_with_temperature(Z, 0.5)) < impl.entropy_bits(impl.softmax_with_temperature(Z, 2.0))
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            impl.softmax_with_temperature(Z, bad)


# --- TODO 2 ---------------------------------------------------------------
def test_top_k_filter(impl):
    out = impl.top_k_filter(torch.tensor([2.0, 1.0, 0.0, -1.0]), 2)
    assert out.tolist() == [2.0, 1.0, float("-inf"), float("-inf")]
    tie = impl.top_k_filter(torch.tensor([1.0, 3.0, 1.0, 0.0]), 2)      # 2 번째 점수(1.0)와 동점은 남김
    assert tie.tolist() == [1.0, 3.0, 1.0, float("-inf")]
    original = torch.tensor([2.0, 1.0])
    assert impl.top_k_filter(original, 5).tolist() == [2.0, 1.0]
    assert original.tolist() == [2.0, 1.0]                             # 입력을 바꾸지 않음
    with pytest.raises(ValueError):
        impl.top_k_filter(Z, 0)


# --- TODO 3 ---------------------------------------------------------------
def test_top_p_filter_hand_example(impl):
    # 정렬 순서와 원래 순서가 다르게: 확률 [0.15, 0.50, 0.05, 0.30]
    logits = torch.log(torch.tensor([0.15, 0.50, 0.05, 0.30]))
    keep = lambda p: (impl.top_p_filter(logits, p) > float("-inf")).tolist()
    assert keep(0.7) == [False, True, False, True]                    # 0.50 + 0.30 = 0.80 ≥ 0.7
    assert keep(0.9) == [True, True, False, True]                     # + 0.15 = 0.95
    assert keep(0.01) == [False, True, False, False]                  # 최소 1 개(1 등)
    assert keep(1.0) == [True, True, True, True]
    torch.testing.assert_close(impl.top_p_filter(logits, 0.7)[1], logits[1])  # 남긴 칸의 값은 그대로
    for bad in (0.0, 1.5):
        with pytest.raises(ValueError):
            impl.top_p_filter(logits, bad)


# --- TODO 4 ---------------------------------------------------------------
def test_sample_next_greedy_and_validation(impl):
    assert impl.sample_next(torch.tensor([0.1, 3.0, 2.0]), temperature=0) == 1
    with pytest.raises(ValueError):
        impl.sample_next(Z, temperature=-0.1)


def test_sample_next_reproducible_and_filtered(impl):
    draws = lambda seed: [impl.sample_next(Z, 1.0, generator=g) for g in [torch.Generator().manual_seed(seed)] for _ in range(50)]
    assert draws(0) == draws(0)                                        # 같은 시드 → 같은 결과
    g = torch.Generator().manual_seed(1)
    picks = [impl.sample_next(torch.tensor([2.0, 1.0, 0.0, -1.0]), 1.0, top_k=2, generator=g) for _ in range(200)]
    assert set(picks) <= {0, 1} and {0, 1} <= set(picks)              # top-k 밖은 절대 안 뽑힘
    g = torch.Generator().manual_seed(2)
    picks = [impl.sample_next(torch.log(torch.tensor([0.15, 0.50, 0.05, 0.30])), 1.0, top_p=0.7, generator=g) for _ in range(200)]
    assert set(picks) == {1, 3}


def test_sample_next_frequency_roughly_matches_probability(impl):
    g = torch.Generator().manual_seed(0)
    picks = [impl.sample_next(Z, 1.0, generator=g) for _ in range(2000)]
    assert 0.60 < picks.count(0) / 2000 < 0.73                        # 기대 0.665 (시드 고정이라 결정적)


# --- TODO 5 ---------------------------------------------------------------
def test_generate_greedy_and_eos(impl, lm):
    eos = lm.index["<eos>"]
    out = impl.generate(lm, lm.encode("the cat"), 12, context_window=4, eos_id=eos)
    assert len(out) == 12
    assert lm.decode(out).startswith("likes the same data and the same data")  # greedy 되풀이
    g = torch.Generator().manual_seed(0)
    out = impl.generate(lm, lm.encode("the cat"), 30, 4, eos_id=eos, temperature=1.0, top_p=0.9, generator=g)
    assert out[-1] == eos and eos not in out[:-1] and len(out) < 30   # eos 에서 멈춤(포함)


def test_generate_context_window_and_limits(impl, lm):
    seen = []

    def spy(ids):
        seen.append(list(ids))
        return lm(ids)

    prompt = lm.encode("the cat sat on the mat")
    out = impl.generate(spy, prompt, 3, context_window=4)
    assert [len(s) for s in seen] == [4, 4, 4]
    assert seen[0] == prompt[-4:] and seen[1] == [*prompt[-3:], out[0]]  # 마지막 4 개만
    assert impl.generate(lm, prompt, 0, 4) == []
    for args in ((prompt, -1, 4), (prompt, 3, 0), ([], 3, 4)):
        with pytest.raises(ValueError):
            impl.generate(lm, *args)


def test_generate_never_exceeds_max_new_tokens(impl):
    calls = []

    def never_eos(ids):                                               # eos 를 절대 안 내는 가짜 모델
        calls.append(1)
        return torch.tensor([5.0, 0.0, 0.0])

    assert impl.generate(never_eos, [1], 7, context_window=2, eos_id=2) == [0] * 7
    assert len(calls) == 7


# --- 데모 스크립트 ----------------------------------------------------------
def test_run_demo_runs(impl, capsys):
    spec = importlib.util.spec_from_file_location("p2_01_run_demo_test", LAB_DIR / "run_demo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.load = lambda _name: impl
    argv, sys.argv = sys.argv, ["run_demo.py"]
    try:
        mod.main()
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert "(재현됨)" in out and "단계별 입력 길이 = [4, 4, 4]" in out

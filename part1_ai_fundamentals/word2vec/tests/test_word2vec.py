"""P1-03 테스트. 기본은 solution 검사, `IMPL=starter pytest ...` 로 직접 구현을 검사한다."""
import pytest
import torch


# ----- Level 1 -----
def test_build_vocab_sorted_by_frequency(impl):
    sents = [["b", "a", "a"], ["c", "a", "b"], ["d"]]
    w2i, i2w, counts = impl.build_vocab(sents, min_count=1)
    assert i2w == ["a", "b", "c", "d"]  # a(3) b(2) c(1) d(1, 사전순)
    assert counts == [3, 2, 1, 1]
    assert w2i["c"] == 2
    _, i2w2, _ = impl.build_vocab(sents, min_count=2)
    assert i2w2 == ["a", "b"]


def test_skipgram_pairs_respect_window_and_boundary(impl):
    pairs = impl.skipgram_pairs([10, 11, 12, 13], window=1)
    assert pairs == [(10, 11), (11, 10), (11, 12), (12, 11), (12, 13), (13, 12)]
    assert len(impl.skipgram_pairs([1, 2, 3, 4, 5], window=2)) == 14


def test_cbow_examples_fixed_context(impl):
    ex = impl.cbow_examples([1, 2, 3, 4, 5], window=2)
    assert ex == [([1, 2, 4, 5], 3)]
    assert impl.cbow_examples([1, 2, 3], window=1) == [([1, 3], 2)]


# ----- Level 2 -----
def test_noise_distribution_three_quarter_power(impl):
    noise = impl.noise_distribution([16, 1], power=0.75)
    assert noise.shape == (2,)
    torch.testing.assert_close(noise.sum(), torch.tensor(1.0))
    torch.testing.assert_close(noise[0] / noise[1], torch.tensor(8.0))  # 16^0.75 = 8


def test_sample_negatives_shape_and_reproducible(impl):
    noise = impl.noise_distribution([5, 3, 2, 1])
    a = impl.sample_negatives(noise, 4, 6, generator=torch.Generator().manual_seed(0))
    b = impl.sample_negatives(noise, 4, 6, generator=torch.Generator().manual_seed(0))
    assert a.shape == (4, 6) and a.dtype == torch.int64
    assert torch.equal(a, b)
    assert a.min() >= 0 and a.max() < 4


def test_skipgram_ns_loss_matches_formula(impl):
    torch.manual_seed(0)
    model = impl.SkipGramNegSampling(5, 3)
    assert model.out_embed.weight.abs().sum() == 0  # 출력 테이블 0 초기화
    with torch.no_grad():
        model.in_embed.weight.normal_()
        model.out_embed.weight.normal_()
    center, context = torch.tensor([0, 1]), torch.tensor([2, 3])
    neg = torch.tensor([[4, 1], [0, 2]])
    loss = model(center, context, neg)
    v, U = model.in_embed.weight, model.out_embed.weight
    expected = 0.0
    for b in range(2):
        term = torch.log(torch.sigmoid(U[context[b]] @ v[center[b]]))
        term += sum(torch.log(torch.sigmoid(-(U[k] @ v[center[b]]))) for k in neg[b])
        expected += -term
    torch.testing.assert_close(loss, expected / 2)


def test_cbow_loss_shape(impl):
    model = impl.CBOWSoftmax(7, 4)
    loss = model(torch.tensor([[1, 2, 3, 4], [0, 1, 5, 6]]), torch.tensor([0, 2]))
    assert loss.shape == ()
    # 초기 loss 는 균등 예측 근처 log(V) 정도
    assert 0.5 * torch.log(torch.tensor(7.0)) < loss < 2 * torch.log(torch.tensor(7.0))


@pytest.mark.parametrize("model_type", ["skipgram", "cbow"])
def test_training_reduces_loss_and_groups_words(impl, model_type):
    sents = impl.make_toy_corpus(n_sentences=2000, seed=0)
    result = impl.train_word2vec(sents, model_type=model_type, epochs=5, seed=0,
                                 device=torch.device("cpu"))
    assert result["losses"][-1] < result["losses"][0]
    purity = impl.group_purity(lambda w, k: impl.nearest(w, result, k))
    assert purity >= 0.6, f"같은 의미 그룹 단어끼리 가까워져야 합니다 (purity={purity:.2f})"


# ----- Level 3 -----
def test_compare_with_gensim(impl):
    pytest.importorskip("gensim")
    from importlib.util import module_from_spec, spec_from_file_location
    from pathlib import Path

    spec = spec_from_file_location("p1_w2v_compare", Path(__file__).resolve().parents[1] / "compare_with_gensim.py")
    compare = module_from_spec(spec)
    spec.loader.exec_module(compare)
    result = compare.compare(impl, epochs=5, n_sentences=2000)
    assert result["my_purity"] >= 0.6 and result["gensim_purity"] >= 0.6

"""P1-03 테스트. 기본은 solution 검사, `IMPL=starter pytest ...` 로 직접 구현을 검사한다.

작은 고정 입력과 저장소 안의 말뭉치(data/corpus_ko_en.txt), 코드로 만드는 생성 말뭉치만 쓴다(다운로드 없음).
학습 테스트는 결과 재현을 위해 seed 0, CPU 로 고정한다.
`impl` 픽스처는 part1_ai_fundamentals/conftest.py 가 파일 경로 기준으로 불러온다.
"""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch

LAB_DIR = Path(__file__).resolve().parents[1]
CPU = torch.device("cpu")


# ----- Level 1 -----
def test_build_vocab_sorted_by_frequency(impl):
    sents = [["b", "a", "a"], ["c", "a", "b"], ["d"]]
    w2i, i2w, counts = impl.build_vocab(sents, min_count=1)
    assert i2w == ["a", "b", "c", "d"]  # a(3) b(2) c(1) d(1, 사전순)
    assert counts == [3, 2, 1, 1]
    assert w2i["c"] == 2
    assert impl.build_vocab(sents, min_count=2)[1] == ["a", "b"]


def test_build_vocab_tie_break_is_lexicographic_not_insertion_order(impl):
    # 등장 순서(d → c)와 사전순(c → d)이 다른 입력: 동점이면 사전순이어야 한다
    _, i2w, counts = impl.build_vocab([["d", "c"]])
    assert i2w == ["c", "d"] and counts == [1, 1]


def test_skipgram_pairs_respect_window_and_boundary(impl):
    pairs = impl.skipgram_pairs([10, 11, 12, 13], window=1)
    assert pairs == [(10, 11), (11, 10), (11, 12), (12, 11), (12, 13), (13, 12)]
    assert len(impl.skipgram_pairs([1, 2, 3, 4, 5], window=2)) == 14


def test_cbow_examples_use_every_position_with_padding(impl):
    ex = impl.cbow_examples([1, 2, 3], window=2, pad_id=9)
    assert ex == [([2, 3, 9, 9], 1), ([1, 3, 9, 9], 2), ([1, 2, 9, 9], 3)]
    full = impl.cbow_examples([1, 2, 3, 4, 5], window=2, pad_id=9)
    assert len(full) == 5 and full[2] == ([1, 2, 4, 5], 3)


# ----- Level 2 -----
def test_noise_distribution_three_quarter_power(impl):
    noise = impl.noise_distribution([16, 1], power=0.75)
    assert noise.shape == (2,) and noise.dtype == torch.float32
    torch.testing.assert_close(noise.sum(), torch.tensor(1.0))
    torch.testing.assert_close(noise[0] / noise[1], torch.tensor(8.0))  # 16^0.75 = 8 (빈도 그대로면 16)


def test_sample_negatives_shape_and_reproducible(impl):
    noise = impl.noise_distribution([5, 3, 2, 1])
    a = impl.sample_negatives(noise, 4, 6, generator=torch.Generator().manual_seed(0))
    b = impl.sample_negatives(noise, 4, 6, generator=torch.Generator().manual_seed(0))
    assert a.shape == (4, 6) and a.dtype == torch.int64
    assert torch.equal(a, b)
    assert a.min() >= 0 and a.max() < 4


def test_skipgram_ns_init_and_loss_formula(impl):
    torch.manual_seed(0)
    model = impl.SkipGramNegSampling(5, 3)
    assert model.out_embed.weight.abs().sum() == 0  # 출력 테이블 0 초기화
    assert model.in_embed.weight.abs().max() <= 0.5 / 3
    with torch.no_grad():
        model.in_embed.weight.normal_()
        model.out_embed.weight.normal_()
    center, context = torch.tensor([0, 1]), torch.tensor([2, 3])
    neg = torch.tensor([[4, 1], [0, 2]])
    loss = model(center, context, neg)
    v, u = model.in_embed.weight, model.out_embed.weight
    expected = 0.0
    for b in range(2):
        term = torch.log(torch.sigmoid(u[context[b]] @ v[center[b]]))
        term += sum(torch.log(torch.sigmoid(-(u[k] @ v[center[b]]))) for k in neg[b])
        expected += -term
    torch.testing.assert_close(loss, expected / 2)


def test_cbow_ignores_padding(impl):
    torch.manual_seed(0)
    model = impl.CBOWSoftmax(6, 4)
    assert model.pad_id == 6 and model.embed.weight.shape == (7, 4)
    assert model.embed.weight[6].abs().sum() == 0  # 패딩 행은 0
    target = torch.tensor([0])
    # 패딩을 뺀 평균이면 [2, pad] 와 [2, 2] 의 평균 벡터가 같다 → loss 도 같다
    torch.testing.assert_close(model(torch.tensor([[2, 6]]), target), model(torch.tensor([[2, 2]]), target))
    loss = model(torch.tensor([[1, 2, 3, 4], [0, 1, 6, 6]]), torch.tensor([0, 2]))
    assert loss.shape == ()
    assert 0.5 * torch.log(torch.tensor(6.0)) < loss < 2 * torch.log(torch.tensor(6.0))  # 초기 loss ≈ log(V)


def _trained_like_skipgram(impl, vocab_size=8, dim=4):
    """초기 상태(out_embed = 0)는 음성 샘플과 무관하게 loss 가 (K+1)·log 2 로 같다 → 무작위 값으로 바꿔 검사한다."""
    torch.manual_seed(0)
    model = impl.SkipGramNegSampling(vocab_size, dim)
    with torch.no_grad():
        model.in_embed.weight.normal_()
        model.out_embed.weight.normal_()
    return model


def test_skipgram_batch_loss_uses_generator(impl):
    model = _trained_like_skipgram(impl)
    noise = impl.noise_distribution([8, 7, 6, 5, 4, 3, 2, 1])
    batch = [(0, 1), (2, 3), (4, 5), (6, 7)]

    def loss(seed):
        return impl.skipgram_batch_loss(model, batch, noise, 3, torch.Generator().manual_seed(seed), CPU)

    torch.manual_seed(123)  # 전역 난수 상태를 바꿔도
    a = loss(1)
    torch.manual_seed(456)
    b = loss(1)
    torch.testing.assert_close(a, b)  # (a) 같은 generator 시드 → 같은 loss (generator 를 실제로 써야 통과)
    assert not torch.allclose(a, loss(2))  # (a) 다른 시드 → 다른 음성 샘플 → 다른 loss
    # 같은 시드로 뽑은 음성 샘플을 직접 넣으면 같은 값 (배치 → 텐서 변환 확인)
    neg = impl.sample_negatives(noise, len(batch), 3, generator=torch.Generator().manual_seed(1))
    torch.testing.assert_close(a, model(torch.tensor([0, 2, 4, 6]), torch.tensor([1, 3, 5, 7]), neg))


def test_skipgram_loss_depends_on_negatives(impl):
    model = _trained_like_skipgram(impl)
    center, context = torch.tensor([0, 1]), torch.tensor([2, 3])
    a = model(center, context, torch.tensor([[4, 5], [6, 7]]))
    b = model(center, context, torch.tensor([[5, 6], [7, 4]]))
    assert not torch.allclose(a, b)  # (b) 음성 샘플이 바뀌면 loss 도 바뀐다


def test_skipgram_loss_is_finite_for_large_scores(impl):
    """log(sigmoid(x)) 를 나눠 계산하면 큰 점수에서 log(0) = -inf → loss = inf. logsigmoid 는 유한하다."""
    model = _trained_like_skipgram(impl)
    with torch.no_grad():
        model.in_embed.weight.mul_(100)
        model.out_embed.weight.mul_(100)
    loss = model(torch.tensor([0, 1]), torch.tensor([2, 3]), torch.tensor([[4, 5], [6, 7]]))
    assert torch.isfinite(loss)


# ----- Level 3 -----
def test_nearest_excludes_self(impl):
    result = {"vectors": torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]]),
              "word_to_id": {"a": 0, "b": 1, "c": 2, "d": 3}, "id_to_word": ["a", "b", "c", "d"]}
    out = impl.nearest("a", result, k=2)
    assert [w for w, _ in out] == ["b", "c"]
    assert out[0][1] > out[1][1]


@pytest.mark.parametrize(("model_type", "epochs", "min_purity"), [("skipgram", 5, 0.9), ("cbow", 10, 0.7)])
def test_training_groups_words_on_toy_corpus(impl, model_type, epochs, min_purity):
    sents = impl.make_toy_corpus(n_sentences=2000, seed=0)
    r = impl.train_word2vec(sents, model_type=model_type, epochs=epochs, seed=0, device=CPU)
    assert r["losses"][-1] < r["losses"][0]

    def purity(vectors=None):
        return impl.group_purity(lambda w, k: impl.nearest(w, r, k, vectors=vectors))

    assert purity(r["vectors_before"]) < 0.3  # 학습 전: 무작위 수준
    assert purity() >= min_purity, f"같은 의미 그룹 단어끼리 가까워져야 합니다 ({model_type}, purity={purity():.2f})"


@pytest.mark.parametrize(("model_type", "min_gain"), [("skipgram", 0.5), ("cbow", 0.2)])
def test_same_structure_words_get_closer_on_corpus(impl, model_type, min_gain):
    """data/corpus_ko_en.txt: cat/dog 는 같은 문장 구조(the _ sat on / the _ likes)에 나온다."""
    r = impl.train_word2vec(impl.load_corpus(), model_type=model_type, epochs=100, seed=0, device=CPU)
    before = impl.cosine("cat", "dog", r, r["vectors_before"])
    after = impl.cosine("cat", "dog", r)
    assert after > before + min_gain
    assert after > impl.cosine("cat", "coffee", r)  # 구조가 다른 대조군보다 가깝다


def test_compare_with_gensim_same_trend(impl):
    pytest.importorskip("gensim", reason="gensim 미설치 — 비교 건너뜀")
    spec = spec_from_file_location("p1_03_compare_with_gensim", LAB_DIR / "compare_with_gensim.py")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    result = mod.compare(impl, epochs=100)
    pairs = result["skipgram"]["pairs"]
    # 값은 달라도 경향은 같아야 한다: 같은 구조 쌍 > 대조군
    for who in ("mine", "gensim"):
        assert pairs["cat-dog"][who] > pairs["cat-coffee"][who], who
    assert result["toy_purity"]["mine"] >= 0.9 and result["toy_purity"]["gensim"] >= 0.9

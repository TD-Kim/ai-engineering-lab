"""P1-03 테스트. 기본은 solution 검사, `IMPL=starter pytest ...` 로 직접 구현을 검사한다.

gensim 결과는 버전·플랫폼에 따라 조금 달라질 수 있으므로 정확한 값 대신 "대상 쌍 > 대조군" 같은 상대 비교로 검사한다.
학습은 저장소 말뭉치(48줄)로 CPU 1초 이내. 반복은 모두 길이가 정해진 for 문(무한 반복·메모리 위험 없음).
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

LAB_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sents(impl):
    return impl.preprocess(impl.load_lines())


@pytest.fixture(scope="module")
def models(impl, sents):
    return {sg: impl.train_word2vec(sents, sg=sg) for sg in (1, 0)}


# --- TODO 1 ---------------------------------------------------------------
def test_preprocess(impl):
    out = impl.preprocess(["The cat sat on the Mat.", "", "   ", "학습이 잘 되면, 손실 값이 줄어든다!"])
    assert out == [["the", "cat", "sat", "on", "the", "mat"], ["학습이", "잘", "되면", "손실", "값이", "줄어든다"]]


def test_preprocess_corpus(sents):
    assert len(sents) == 48
    assert sents[44] == ["the", "cat", "sat", "on", "the", "mat", "and", "the", "dog", "sat", "on", "the", "log"]


# --- TODO 2 ---------------------------------------------------------------
def test_train_word2vec_settings(impl, sents, models):
    m = models[1]
    assert (m.vector_size, m.window, m.sg, m.negative, m.epochs, m.workers, m.min_count) == (32, 2, 1, 5, 200, 1, 1)
    assert m.sample == 0 and m.hs == 0                                  # 다운샘플링·계층적 소프트맥스 끔
    assert models[0].sg == 0
    assert len(m.wv) == len({t for s in sents for t in s})


def test_train_word2vec_reproducible_and_epochs0(impl, sents, models):
    again = impl.train_word2vec(sents, sg=1)
    np.testing.assert_allclose(again.wv["cat"], models[1].wv["cat"], rtol=1e-5, atol=1e-6)
    before = impl.train_word2vec(sents, sg=1, epochs=0)
    assert len(before.wv) == len(models[1].wv)
    assert not np.allclose(before.wv["cat"], models[1].wv["cat"])          # 학습하면 벡터가 바뀐다


# --- TODO 3 ---------------------------------------------------------------
@pytest.mark.parametrize("sg", [1, 0])
def test_target_pairs_beat_controls(impl, models, sg):
    sim = {r["pair"]: r["similarity"] for r in impl.pair_similarities(
        models[sg], [("cat", "dog"), ("cat", "coffee"), ("mat", "log"), ("mat", "coffee")])}
    assert sim[("cat", "dog")] > sim[("cat", "coffee")]
    assert sim[("mat", "log")] > sim[("mat", "coffee")]


def test_training_raises_target_similarity(impl, sents, models):
    before = impl.train_word2vec(sents, sg=1, epochs=0)
    b = impl.pair_similarities(before, [("cat", "dog")])[0]["similarity"]
    a = impl.pair_similarities(models[1], [("cat", "dog")])[0]["similarity"]
    assert a > b


def test_pair_similarities_oov_and_types(impl, models):
    rows = impl.pair_similarities(models[1], [("cat", "강아지"), ("cat", "dog")])
    assert rows[0] == {"pair": ("cat", "강아지"), "similarity": None}
    assert isinstance(rows[1]["similarity"], float)


def test_nearest(impl, models):
    nn = impl.nearest(models[1], "cat", 5)
    assert len(nn) == 5 and "cat" not in [w for w, _ in nn]
    assert "dog" in [w for w, _ in nn[:3]]
    assert all(isinstance(w, str) and isinstance(s, float) for w, s in nn)
    assert [s for _, s in nn] == sorted((s for _, s in nn), reverse=True)
    assert impl.nearest(models[1], "강아지", 5) == []


# --- TODO 4 (선택) --------------------------------------------------------
def test_sentence_vector(impl, models):
    m = models[1]
    v = impl.sentence_vector(m, ["cat", "강아지", "dog"])               # OOV 는 건너뛴다
    np.testing.assert_allclose(v, (m.wv["cat"] + m.wv["dog"]) / 2, rtol=1e-5, atol=1e-6)
    assert v.dtype == np.float32 and v.shape == (32,)
    z = impl.sentence_vector(m, ["강아지", "고양이"])
    assert z.shape == (32,) and np.all(z == 0)


# --- 데모 스크립트 ----------------------------------------------------------
def test_run_demo_runs(impl, capsys):
    spec = importlib.util.spec_from_file_location("p1_03_run_demo_test", LAB_DIR / "run_demo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    original = mod.load
    mod.load = lambda path, name: impl if Path(path).parent == LAB_DIR else original(path, name)  # 현재 IMPL 로 실행
    argv, sys.argv = sys.argv, ["run_demo.py"]
    try:
        mod.main()
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert "강아지: [] (OOV" in out and "검색 불가" in out

"""P1-01 테스트. 기본은 solution 검사, `IMPL=starter pytest ...` 로 직접 구현을 검사한다.

모든 테스트는 작은 문자열과 저장소 안의 말뭉치(data/corpus_ko_en.txt)만 쓰며 외부 다운로드가 없다.
`impl` 픽스처는 part1_ai_fundamentals/conftest.py 가 파일 경로 기준으로 불러온다.
"""
from collections import Counter
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

LAB_DIR = Path(__file__).resolve().parents[1]
CORPUS = LAB_DIR.parent / "data" / "corpus_ko_en.txt"
# Sennrich et al. (2016) 논문 §3.2 의 예시 어휘
TOY = "low " * 5 + "lower " * 2 + "newest " * 6 + "widest " * 3


# ----- Level 1: 토큰화 · Vocabulary · 특수 토큰 · padding -----
def test_tokenize_word_and_char(impl):
    assert impl.tokenize("hello   world\n토큰", mode="word") == ["hello", "world", "토큰"]
    assert impl.tokenize("가 b", mode="char") == ["가", " ", "b"]
    with pytest.raises(ValueError):
        impl.tokenize("x", mode="byte?")


def test_vocab_special_tokens_first_and_frequency_order(impl):
    vocab = impl.Vocab.build(["b", "a", "c", "a", "b", "a", "<unk>"])
    assert vocab.id_to_token[:4] == ["<pad>", "<unk>", "<bos>", "<eos>"]
    assert vocab.id_to_token[4:] == ["a", "b", "c"]  # a(3) b(2) c(1), 특수 토큰 중복 없음
    assert vocab.pad_id == 0 and vocab.unk_id == 1
    assert len(vocab) == 7
    assert vocab.token_to_id["c"] == 6


def test_vocab_min_freq_and_deterministic(impl):
    a = impl.Vocab.build(["y", "x", "x", "z", "y"], min_freq=2)
    b = impl.Vocab.build(["x", "y", "z", "y", "x"], min_freq=2)
    assert a.id_to_token == b.id_to_token == ["<pad>", "<unk>", "<bos>", "<eos>", "x", "y"]


def test_vocab_encode_unknown_and_special(impl):
    vocab = impl.Vocab.build(["나는", "학생", "이다"])
    ids = vocab.encode(["나는", "선생", "이다"], add_special=True)
    assert ids[0] == vocab.token_to_id["<bos>"] and ids[-1] == vocab.token_to_id["<eos>"]
    assert ids[2] == vocab.unk_id  # 모르는 단어 → <unk>
    assert vocab.decode(ids) == ["나는", "<unk>", "이다"]  # bos/eos 제거, unk 는 남김
    assert vocab.decode(ids, skip_special=False)[0] == "<bos>"


def test_pad_and_truncate(impl):
    ids, mask = impl.pad_and_truncate([[5, 6, 7, 8], [9], []], max_len=3, pad_id=0)
    assert ids == [[5, 6, 7], [9, 0, 0], [0, 0, 0]]
    assert mask == [[1, 1, 1], [1, 0, 0], [0, 0, 0]]


# ----- Level 2: BPE -----
def test_pair_counts_weighted_by_frequency(impl):
    pairs = impl.get_pair_counts(impl.build_word_vocab(TOY))
    assert isinstance(pairs, Counter)
    assert pairs[("e", "s")] == 9  # newest(6) + widest(3)
    assert pairs[("l", "o")] == 7  # low(5) + lower(2)
    assert ("w", "n") not in pairs  # 단어 경계를 넘는 쌍은 없어야 한다


def test_merge_pair_non_overlapping(impl):
    merged = impl.merge_pair(("a", "a"), {("a", "a", "a", "</w>"): 2})
    assert merged == {("aa", "a", "</w>"): 2}


def test_train_bpe_matches_paper_example(impl):
    merges = impl.train_bpe(TOY, num_merges=4)
    # 논문 예시와 같은 첫 병합들: es, est, est</w>, lo (동점은 사전순)
    assert merges == [("e", "s"), ("es", "t"), ("est", "</w>"), ("l", "o")]


def test_bpe_tokenize_uses_merges(impl):
    tok = impl.BPETokenizer().fit(TOY, num_merges=10)
    assert tok.tokenize_word("newest") == ["newest</w>"]  # 자주 나온 단어는 통째로 하나의 토큰
    # 학습 때 없던 단어도 서브워드로 분절된다 (OOV 해결)
    assert tok.tokenize_word("lowest") == ["low", "est</w>"]


def test_bpe_vocab_has_special_tokens(impl):
    tok = impl.BPETokenizer().fit(TOY, num_merges=10)
    assert tok.vocab.id_to_token[:4] == ["<pad>", "<unk>", "<bos>", "<eos>"]
    ids = tok.encode("lowest", add_special=True)
    assert ids[0] == 2 and ids[-1] == 3
    assert tok.decode(ids) == "lowest"


def test_bpe_roundtrip_on_corpus(impl):
    text = CORPUS.read_text(encoding="utf-8")
    tok = impl.BPETokenizer().fit(text, num_merges=200)
    sample = "작은 모델이라도 직접 학습해 보면 원리가 잘 보인다. the newest model"
    assert tok.decode(tok.encode(sample)) == sample


def test_bpe_compresses_better_than_chars(impl):
    text = CORPUS.read_text(encoding="utf-8")
    bpe = impl.BPETokenizer().fit(text, num_merges=300)
    n_symbols = sum(len(w) + 1 for w in text.split())  # 문자 + 단어 경계
    assert len(bpe.encode(text)) < 0.6 * n_symbols


# ----- Level 3: HF tokenizers 와 비교 -----
def test_compare_with_hf_tokenizers(impl):
    pytest.importorskip("tokenizers", reason="tokenizers 미설치 — HF 비교 건너뜀")
    spec = spec_from_file_location("p1_01_compare_with_hf", LAB_DIR / "compare_with_hf.py")
    compare = module_from_spec(spec)
    spec.loader.exec_module(compare)
    result = compare.compare(impl, CORPUS.read_text(encoding="utf-8"), num_merges=200)
    assert result["my_vocab_size"] == result["hf_vocab_size"]
    # 같은 알고리즘이므로 결과가 크게 다르지 않아야 한다 (동점 처리·단어 끝 표시 방식 차이로 완전히 같지는 않음)
    assert result["first_merges_overlap"] >= 0.4  # 측정값 0.52 (README 8절)
    assert 0.8 <= result["token_count_ratio"] <= 1.25  # 측정값 0.97
    assert result["my_decode_ok"] and result["hf_decode_ok"]

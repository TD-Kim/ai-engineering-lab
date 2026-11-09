"""P1-01 테스트. 기본은 solution 검사, `IMPL=starter pytest ...` 로 직접 구현을 검사한다."""
from collections import Counter
from pathlib import Path

import pytest

CORPUS = Path(__file__).resolve().parents[2] / "data" / "corpus_ko_en.txt"
# Sennrich et al. (2016) 논문의 예시 어휘
TOY = "low " * 5 + "lower " * 2 + "newest " * 6 + "widest " * 3


# ----- Level 1 -----
def test_char_roundtrip_korean_english(impl):
    text = "토크나이저 tokenizer!"
    tok = impl.CharTokenizer().fit(text)
    assert tok.token_to_id[impl.UNK] == 0
    assert tok.decode(tok.encode(text)) == text


def test_char_ids_are_deterministic(impl):
    a = impl.CharTokenizer().fit("banana")
    b = impl.CharTokenizer().fit("nabana")
    assert a.token_to_id == b.token_to_id  # 같은 문자 집합 → 같은 id


def test_char_unknown_maps_to_unk(impl):
    tok = impl.CharTokenizer().fit("abc")
    assert tok.encode("abz") == [tok.token_to_id["a"], tok.token_to_id["b"], 0]


# ----- Level 2 -----
def test_pair_counts_weighted_by_frequency(impl):
    vocab = impl.build_word_vocab(TOY)
    pairs = impl.get_pair_counts(vocab)
    assert isinstance(pairs, Counter)
    assert pairs[("e", "s")] == 9  # newest(6) + widest(3)
    assert pairs[("l", "o")] == 7  # low(5) + lower(2)
    assert ("w", "n") not in pairs  # 단어 경계를 넘는 쌍은 없어야 한다


def test_merge_pair_non_overlapping(impl):
    vocab = {("a", "a", "a", "</w>"): 2}
    merged = impl.merge_pair(("a", "a"), vocab)
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


def test_bpe_roundtrip_on_corpus(impl):
    text = CORPUS.read_text(encoding="utf-8")
    tok = impl.BPETokenizer().fit(text, num_merges=200)
    sample = "작은 모델이라도 직접 학습해 보면 원리가 잘 보인다. the newest model"
    assert tok.decode(tok.encode(sample)) == sample


def test_bpe_compresses_better_than_chars(impl):
    text = CORPUS.read_text(encoding="utf-8")
    bpe = impl.BPETokenizer().fit(text, num_merges=300)
    n_chars = sum(len(w) + 1 for w in text.split())  # 문자 + 단어 경계
    assert len(bpe.encode(text)) < 0.6 * n_chars


# ----- Level 3: HF tokenizers 와 비교 -----
def test_compare_with_hf_tokenizers(impl):
    pytest.importorskip("tokenizers")
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("p1_tok_compare", Path(__file__).resolve().parents[1] / "compare_with_hf.py")
    compare = module_from_spec(spec)
    spec.loader.exec_module(compare)
    text = CORPUS.read_text(encoding="utf-8")
    result = compare.compare(impl, text, num_merges=200)
    # 같은 알고리즘이므로 결과가 크게 다르지 않아야 한다 (동점 처리·특수 규칙 차이로 완전히 같지는 않음)
    assert result["first_merges_overlap"] >= 0.4  # 측정값 약 0.5 (HF 는 "</w>" 를 마지막 글자에 처음부터 붙임)
    assert 0.8 <= result["token_count_ratio"] <= 1.25

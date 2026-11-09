"""P1-03 Level 3: 직접 구현한 Skip-gram(Negative Sampling)과 gensim Word2Vec 을 같은 말뭉치에서 비교한다.

사용법:
    python part1_ai_fundamentals/word2vec/compare_with_gensim.py --impl solution

비교 항목
- 그룹 순도(group purity): 각 단어의 최근접 3개 중 같은 의미 그룹(동물/탈것/과일) 단어 비율
- 대표 단어의 최근접 이웃
- 학습 시간

REF: https://radimrehurek.com/gensim/models/word2vec.html
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path


def train_gensim(sentences, dim=32, window=2, negatives=5, epochs=5, seed=0):
    from gensim.models import Word2Vec

    start = time.perf_counter()
    # sg=1: Skip-gram, negative=5: 음성 샘플 5개, workers=1: 결과 재현을 위해 단일 스레드
    model = Word2Vec(
        sentences, vector_size=dim, window=window, sg=1, negative=negatives, hs=0,
        min_count=1, epochs=epochs, seed=seed, workers=1,
    )
    return model, time.perf_counter() - start


def compare(impl, epochs: int = 5, n_sentences: int = 3000) -> dict:
    sentences = impl.make_toy_corpus(n_sentences=n_sentences, seed=0)
    mine = impl.train_word2vec(sentences, model_type="skipgram", epochs=epochs)
    gmodel, gsec = train_gensim(sentences, epochs=epochs)

    def mine_nn(w, k):
        return impl.nearest(w, mine, k)

    def gensim_nn(w, k):
        return gmodel.wv.most_similar(w, topn=k)

    return {
        "my_purity": impl.group_purity(mine_nn),
        "gensim_purity": impl.group_purity(gensim_nn),
        "my_seconds": mine["seconds"],
        "gensim_seconds": gsec,
        "my_loss": mine["losses"],
        "my_cat": mine_nn("cat", 3),
        "gensim_cat": gensim_nn("cat", 3),
        "my_car": mine_nn("car", 3),
        "gensim_car": gensim_nn("car", 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=["solution", "starter"], default="solution")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    impl = importlib.import_module(args.impl)
    for key, value in compare(impl, epochs=args.epochs).items():
        if isinstance(value, float):
            value = round(value, 4)
        elif isinstance(value, list) and value and isinstance(value[0], tuple):
            value = [(w, round(sim, 3)) for w, sim in value]
        elif isinstance(value, list):
            value = [round(v, 4) for v in value]
        print(f"{key:>15}: {value}")

if __name__ == "__main__":
    main()

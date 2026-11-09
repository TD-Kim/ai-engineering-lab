"""P1-03 데모: data/corpus_ko_en.txt 로 Skip-gram(Negative Sampling)과 CBOW 를 학습하고,
손실 곡선 요약과 학습 전후 단어 벡터(코사인 유사도, 최근접 이웃)를 비교한다.

사용법:
    python part1_ai_fundamentals/word2vec/advanced/run_demo.py --impl solution
    python part1_ai_fundamentals/word2vec/advanced/run_demo.py --impl starter --epochs 100

- 시드 고정(0), CPU 에서 수 초. GPU 가 있으면 자동으로 cuda 사용.
- 비교 쌍: (cat, dog), (mat, log), (sat, likes) 는 말뭉치에서 같은 문장 구조에 나오는 쌍,
  (cat, coffee) 는 구조가 다른 대조군이다. "같은 구조 쌍이 대조군보다 더 가까워지는가"를 본다.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent
PAIRS = [("cat", "dog"), ("mat", "log"), ("sat", "likes"), ("cat", "coffee")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=["solution", "starter"], default="solution")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    sys.path.insert(0, str(LAB_DIR))
    impl = importlib.import_module(args.impl)

    corpus = impl.load_corpus()
    w2i, _, counts = impl.build_vocab(corpus)
    print(f"[device] {impl.get_device()}")
    print(f"[data] sentences={len(corpus)} tokens={sum(map(len, corpus))} vocab={len(w2i)} "
          f"count(cat)={counts[w2i['cat']]} count(the)={counts[w2i['the']]}")
    for model_type in ("skipgram", "cbow"):
        r = impl.train_word2vec(corpus, model_type, dim=32, window=2, negatives=5,
                                epochs=args.epochs, batch_size=256, lr=0.01, seed=0)
        step = max(1, args.epochs // 5)
        curve = ", ".join(f"ep{i + 1}:{r['losses'][i]:.3f}" for i in range(0, args.epochs, step))
        print(f"\n[{model_type}] loss {curve}, ep{args.epochs}:{r['losses'][-1]:.3f} ({r['seconds']:.2f}s)")
        for a, b in PAIRS:
            before = impl.cosine(a, b, r, r["vectors_before"])
            after = impl.cosine(a, b, r)
            print(f"  cos({a}, {b}): {before:+.3f} -> {after:+.3f}")
        print(f"  cat 최근접 (학습 전): {[(w, round(v, 3)) for w, v in impl.nearest('cat', r, 3, vectors=r['vectors_before'])]}")
        print(f"  cat 최근접 (학습 후): {[(w, round(v, 3)) for w, v in impl.nearest('cat', r, 3)]}")


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as exc:
        print(f"[미구현] {exc} — starter.py 의 TODO 를 먼저 채우세요 (python part1_ai_fundamentals/word2vec/advanced/starter.py 로 확인)")

"""P1-03 Level 3: 직접 구현한 Word2Vec 과 gensim Word2Vec 을 같은 말뭉치·같은 하이퍼파라미터로 비교한다.

사용법:
    python part1_ai_fundamentals/word2vec/advanced/compare_with_gensim.py --impl solution

비교 대상
1. data/corpus_ko_en.txt (저장소 말뭉치): cat–dog 같은 "같은 문장 구조" 단어 쌍의 코사인, cat 의 최근접 이웃
2. 생성 말뭉치(동물/탈것/과일): 그룹 순도(group purity)

같게 맞춘 것: 말뭉치·토큰화, 벡터 차원 32, 윈도우 2, 음성 샘플 5개, epoch 수, min_count=1, 다운샘플링 끔(sample=0), 단일 스레드
다른 것(그래서 값이 똑같이 나오지 않는다):
- 최적화: 직접 구현은 Adam(lr 0.01, 미니배치 256), gensim 은 SGD(alpha 0.025 → 0.0001 선형 감소, 단어 단위 갱신)
- 윈도우: gensim 은 매 위치마다 실제 윈도우 크기를 1..window 에서 무작위로 줄인다(가까운 단어에 가중치를 주는 효과), 직접 구현은 고정
- CBOW: 직접 구현은 full softmax, gensim 은 CBOW 에도 음성 샘플링 사용
- 난수: 초기화·음성 샘플이 서로 다른 난수 생성기를 쓴다
따라서 "값의 일치"가 아니라 "경향(같은 구조 단어가 가까워지는가)"이 같은지를 본다.

REF: https://radimrehurek.com/gensim/models/word2vec.html (gensim Word2Vec 문서, 작성 환경에서 접속 미확인)
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

import torch

LAB_DIR = Path(__file__).resolve().parent
PAIRS = [("cat", "dog"), ("mat", "log"), ("sat", "likes"), ("cat", "coffee")]  # 마지막은 대조군(구조가 다른 쌍)
DIM, WINDOW, NEG, BATCH, LR = 32, 2, 5, 256, 0.01


def gensim_available() -> bool:
    try:
        import gensim  # noqa: F401
    except ImportError:
        return False
    return True


def train_gensim(sentences, sg: int, epochs: int, seed: int = 0):
    from gensim.models import Word2Vec

    start = time.perf_counter()
    model = Word2Vec(
        sentences, vector_size=DIM, window=WINDOW, sg=sg, negative=NEG, hs=0,
        min_count=1, sample=0, epochs=epochs, seed=seed, workers=1,
    )
    return model, time.perf_counter() - start


def gensim_cos(model, a: str, b: str) -> float:
    return float(model.wv.similarity(a, b))


def compare(impl, epochs: int = 100, toy_epochs: int = 10, toy_sentences: int = 2000) -> dict:
    cpu = torch.device("cpu")
    corpus = impl.load_corpus()
    out: dict = {"corpus_tokens": sum(map(len, corpus)), "corpus_vocab": len(impl.build_vocab(corpus)[1])}

    for name, sg in [("skipgram", 1), ("cbow", 0)]:
        mine = impl.train_word2vec(corpus, name, dim=DIM, window=WINDOW, negatives=NEG,
                                   epochs=epochs, batch_size=BATCH, lr=LR, device=cpu)
        gmodel, gsec = train_gensim(corpus, sg=sg, epochs=epochs)
        out[name] = {
            "my_loss_first_last": (round(mine["losses"][0], 4), round(mine["losses"][-1], 4)),
            "my_seconds": round(mine["seconds"], 2),
            "gensim_seconds": round(gsec, 2),
            "pairs": {
                f"{a}-{b}": {
                    "before": round(impl.cosine(a, b, mine, mine["vectors_before"]), 3),
                    "mine": round(impl.cosine(a, b, mine), 3),
                    "gensim": round(gensim_cos(gmodel, a, b), 3),
                }
                for a, b in PAIRS
            },
            "my_cat_nn": [(w, round(v, 3)) for w, v in impl.nearest("cat", mine, 3)],
            "gensim_cat_nn": [(w, round(v, 3)) for w, v in gmodel.wv.most_similar("cat", topn=3)],
        }

    toy = impl.make_toy_corpus(n_sentences=toy_sentences, seed=0)
    mine_toy = impl.train_word2vec(toy, "skipgram", dim=DIM, window=WINDOW, negatives=NEG,
                                   epochs=toy_epochs, batch_size=BATCH, lr=LR, device=cpu)
    g_toy, _ = train_gensim(toy, sg=1, epochs=toy_epochs)
    out["toy_purity"] = {
        "before": round(impl.group_purity(lambda w, k: impl.nearest(w, mine_toy, k, vectors=mine_toy["vectors_before"])), 3),
        "mine": round(impl.group_purity(lambda w, k: impl.nearest(w, mine_toy, k)), 3),
        "gensim": round(impl.group_purity(lambda w, k: g_toy.wv.most_similar(w, topn=k)), 3),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=["solution", "starter"], default="solution")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    if not gensim_available():
        print("[건너뜀] gensim 이 없어 비교를 건너뜁니다. 설치: uv pip install -r requirements/part1.txt")
        return
    sys.path.insert(0, str(LAB_DIR))
    impl = importlib.import_module(args.impl)
    try:
        result = compare(impl, epochs=args.epochs)
    except NotImplementedError as exc:
        print(f"[미구현] {exc} — {args.impl}.py 의 TODO 를 먼저 채우세요")
        return
    print(f"[corpus] tokens={result['corpus_tokens']} vocab={result['corpus_vocab']} epochs={args.epochs}")
    for name in ("skipgram", "cbow"):
        r = result[name]
        print(f"\n[{name}] my loss {r['my_loss_first_last'][0]} -> {r['my_loss_first_last'][1]}, "
              f"time mine {r['my_seconds']}s / gensim {r['gensim_seconds']}s")
        print(f"  {'pair':<12}{'before':>8}{'mine':>8}{'gensim':>8}")
        for pair, v in r["pairs"].items():
            print(f"  {pair:<12}{v['before']:>8}{v['mine']:>8}{v['gensim']:>8}")
        print(f"  cat 최근접 (mine):   {r['my_cat_nn']}")
        print(f"  cat 최근접 (gensim): {r['gensim_cat_nn']}")
    print(f"\n[toy skipgram] group purity before {result['toy_purity']['before']} / "
          f"mine {result['toy_purity']['mine']} / gensim {result['toy_purity']['gensim']} (무작위 기대값 약 0.077)")


if __name__ == "__main__":
    main()

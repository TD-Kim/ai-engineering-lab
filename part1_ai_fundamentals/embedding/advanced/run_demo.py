"""P1-02 데모: 작은 "다음 단어 맞히기" 모델로 임베딩을 학습하고, 학습 전후의 최근접 이웃과 PCA 2차원 그림을 비교한다.

사용법:
    python part1_ai_fundamentals/embedding/advanced/run_demo.py --impl solution
    python part1_ai_fundamentals/embedding/advanced/run_demo.py --impl starter --steps 300

- 데이터: part1_ai_fundamentals/data/corpus_ko_en.txt (공백 기준 단어, 같은 줄 안의 (현재 단어 → 다음 단어) 쌍)
- 모델: 직접 구현한 EmbeddingLayer(V, D) → nn.Linear(D, V) → cross entropy
- 그림: part1_ai_fundamentals/embedding/advanced/outputs/pca_before.png, pca_after.png (outputs/ 는 커밋하지 않음)
  matplotlib 이 없으면 그림만 건너뛴다.

주의: 말뭉치가 매우 작아(단어 504개) 의미 있는 단어 관계를 기대하기 어렵다. 여기서 보는 것은
"학습이 벡터를 움직여, 비슷한 자리(같은 다음 단어)에 오는 단어끼리 가까워지기 시작한다"는 현상이다.
제대로 된 단어 벡터는 P1-03 Word2Vec 에서 다룬다.
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from itertools import pairwise
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

LAB_DIR = Path(__file__).resolve().parent
CORPUS = LAB_DIR.parents[1] / "data" / "corpus_ko_en.txt"
PROBE_WORDS = ["cat", "dog", "the", "loss", "model"]


def build_data(text: str):
    lines = [line.split() for line in text.splitlines() if line.strip()]
    vocab = sorted({w for line in lines for w in line})
    stoi = {w: i for i, w in enumerate(vocab)}
    pairs = [(stoi[a], stoi[b]) for line in lines for a, b in pairwise(line)]
    x = torch.tensor([a for a, _ in pairs])
    y = torch.tensor([b for _, b in pairs])
    return vocab, stoi, x, y


def neighbors(impl, table, vocab, stoi, words, k=3):
    out = {}
    for w in words:
        if w in stoi:
            out[w] = [(vocab[i], round(s, 3)) for i, s in impl.most_similar(stoi[w], table, k=k)]
    return out


def save_pca(impl, table, vocab, path: Path, title: str) -> str:
    try:
        import matplotlib

        matplotlib.use("Agg")  # 화면 없이 파일로만 저장
        import matplotlib.pyplot as plt
    except ImportError:
        return "건너뜀 (matplotlib 미설치)"
    coords = impl.pca_2d(table).cpu()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(coords[:, 0], coords[:, 1], s=6, alpha=0.4)
    # 한글 폰트가 없는 환경이 많아 영어 단어만 이름을 붙인다.
    for w in PROBE_WORDS:
        i = vocab.index(w) if w in vocab else None
        if i is not None:
            ax.annotate(w, (coords[i, 0], coords[i, 1]), fontsize=10, color="crimson")
    ax.set_title(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(path.relative_to(LAB_DIR.parents[2]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=["solution", "starter"], default="solution")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, str(LAB_DIR))
    impl = importlib.import_module(args.impl)
    device = impl.get_device()
    torch.manual_seed(args.seed)

    vocab, stoi, x, y = build_data(CORPUS.read_text(encoding="utf-8"))
    x, y = x.to(device), y.to(device)
    emb = impl.EmbeddingLayer(len(vocab), args.dim).to(device)
    head = nn.Linear(args.dim, len(vocab)).to(device)
    opt = torch.optim.Adam(list(emb.parameters()) + list(head.parameters()), lr=0.05)

    print(f"[device] {device}")
    print(f"[data] vocab={len(vocab)} pairs={len(x)} dim={args.dim}")
    before = emb.weight.detach().clone()
    print(f"[before] neighbors: {neighbors(impl, before, vocab, stoi, PROBE_WORDS)}")
    pca_before = save_pca(impl, before, vocab, LAB_DIR / "outputs" / "pca_before.png", "before training (random)")

    start = time.perf_counter()
    losses = []
    for _ in range(args.steps):
        loss = F.cross_entropy(head(emb(x)), y)  # (N, V) vs (N,)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    elapsed = time.perf_counter() - start

    after = emb.weight.detach()
    print(f"[train] loss {losses[0]:.4f} -> {losses[-1]:.4f} ({args.steps} steps, {elapsed:.2f}s)")
    print(f"[after] neighbors: {neighbors(impl, after, vocab, stoi, PROBE_WORDS)}")
    pca_after = save_pca(impl, after, vocab, LAB_DIR / "outputs" / "pca_after.png", "after training")
    coords = impl.pca_2d(after).cpu()
    summary = {w: tuple(round(v, 2) for v in coords[stoi[w]].tolist()) for w in PROBE_WORDS if w in stoi}
    print(f"[pca] after coords (PC1, PC2): {summary}")
    print(f"[pca] images: before={pca_before}, after={pca_after}")


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as exc:
        print(f"[미구현] {exc} — starter.py 의 TODO 를 먼저 채우세요 (python part1_ai_fundamentals/embedding/advanced/starter.py 로 확인)")

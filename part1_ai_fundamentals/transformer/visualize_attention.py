"""P1-05 Level 3: 학습된 미니 언어모델의 어텐션 가중치를 그림으로 본다.

사용법 (먼저 train_tiny.py 로 체크포인트를 만든다):
    python part1_ai_fundamentals/transformer/train_tiny.py --impl solution
    python part1_ai_fundamentals/transformer/visualize_attention.py --impl solution --text "the cat sat on the mat"

- 각 층의 head 별 (T, T) 가중치를 heatmap 으로 저장: outputs/attention_layer{L}.png (커밋하지 않음)
- matplotlib 이 없으면 그림은 건너뛰고, 각 query 가 가장 많이 본 key(글자)를 텍스트로만 출력한다.
- 한글 폰트가 없는 환경이 많아 기본 문장은 영어다(한글 문장은 폰트가 있으면 사용 가능).
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import torch

LAB_DIR = Path(__file__).resolve().parent
CHECKPOINT = LAB_DIR / "outputs" / "tiny_lm.pt"


def load_model(impl, path: Path = CHECKPOINT):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    vocab = impl.CharVocab("".join(ckpt["chars"]))
    model = impl.MiniTransformerLM(len(vocab), cfg["d_model"], cfg["heads"], cfg["layers"], max_len=cfg["block_size"])
    # state_dict 로 저장·복원: 모듈 전체 저장보다 코드 변경에 안전하다(P1-02 README 9절 참고).
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), vocab


@torch.no_grad()
def attention_maps(model, vocab, text: str) -> list[torch.Tensor]:
    """각 층의 어텐션 가중치 [(H, T, T), ...]"""
    ids = torch.tensor([vocab.encode(text)])
    model(ids)  # forward 하면 각 블록의 attn.last_attn_weights 에 (1, H, T, T) 가 저장된다
    return [block.attn.last_attn_weights[0].cpu() for block in model.blocks]


def save_heatmaps(maps, chars: list[str], out_dir: Path) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    paths = []
    labels = ["␣" if c == " " else c for c in chars]
    for layer, weights in enumerate(maps):
        heads = weights.shape[0]
        fig, axes = plt.subplots(1, heads, figsize=(3.2 * heads, 3.4))
        for h, ax in enumerate(axes if heads > 1 else [axes]):
            ax.imshow(weights[h], cmap="viridis", vmin=0, vmax=1)
            ax.set_title(f"layer {layer} head {h}", fontsize=9)
            ax.set_xticks(range(len(labels)), labels, fontsize=7)
            ax.set_yticks(range(len(labels)), labels, fontsize=7)
            ax.set_xlabel("key", fontsize=7)
            ax.set_ylabel("query" if h == 0 else "", fontsize=7)
        fig.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"attention_layer{layer}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(str(path.relative_to(LAB_DIR.parent.parent)))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=["solution", "starter"], default="solution")
    parser.add_argument("--text", default="the cat sat on the mat")
    args = parser.parse_args()
    sys.path.insert(0, str(LAB_DIR))
    impl = importlib.import_module(args.impl)
    if not CHECKPOINT.exists():
        print(f"[없음] {CHECKPOINT} — 먼저 train_tiny.py 를 실행하세요")
        return
    model, vocab = load_model(impl)
    chars = [c for c in args.text if c in vocab.stoi]
    maps = attention_maps(model, vocab, "".join(chars))
    print(f"[text] {''.join(chars)!r} (T={len(chars)})")
    for layer, weights in enumerate(maps):
        # causal 마스크 확인: 대각선 위(미래)의 가중치 합은 0 이어야 한다
        future = torch.triu(weights, diagonal=1).sum().item()
        print(f"[layer {layer}] heads={weights.shape[0]} 미래 위치 가중치 합={future:.1e} 행 합 평균={weights.sum(-1).mean():.3f}")
        for h in range(weights.shape[0]):
            top = weights[h].argmax(dim=-1)  # 각 query 가 가장 많이 본 key 위치
            prev = sum(int(top[t]) == t - 1 for t in range(1, len(chars))) / max(len(chars) - 1, 1)
            self_ = sum(int(top[t]) == t for t in range(len(chars))) / len(chars)
            print(f"  head {h}: 바로 앞 글자를 가장 많이 본 비율 {prev:.2f}, 자기 자신 {self_:.2f}")
    paths = save_heatmaps(maps, chars, LAB_DIR / "outputs")
    print(f"[images] {paths if paths else '건너뜀 (matplotlib 미설치)'}")


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as exc:
        print(f"[미구현] {exc} — starter.py 의 TODO 를 먼저 채우세요")

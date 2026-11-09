"""P1-05 Level 3: 작은 Transformer 문자 단위 언어모델을 data/corpus_ko_en.txt 로 학습한다.

사용법:
    python part1_ai_fundamentals/transformer/train_tiny.py --impl solution
    python part1_ai_fundamentals/transformer/train_tiny.py --impl starter --steps 300

- 앞 90% 로 학습, 뒤 10% 로 검증 손실(같은 위치로 고정 평가). 시드 0 고정.
- 손실 곡선 요약, 파라미터 수, 학습 시간, 생성 예시를 출력하고,
  체크포인트를 part1_ai_fundamentals/transformer/outputs/tiny_lm.pt 로 저장한다(state_dict, 커밋하지 않음).
  visualize_attention.py 가 이 파일을 쓴다.
- GPU 가 있으면 자동으로 cuda 를 쓴다. CPU 에서도 300 step 이 수 초 걸린다.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import torch

LAB_DIR = Path(__file__).resolve().parent
CORPUS = LAB_DIR.parent / "data" / "corpus_ko_en.txt"
CHECKPOINT = LAB_DIR / "outputs" / "tiny_lm.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=["solution", "starter"], default="solution")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, str(LAB_DIR))
    impl = importlib.import_module(args.impl)
    text = CORPUS.read_text(encoding="utf-8")
    r = impl.train_lm(
        text, steps=args.steps, d_model=args.d_model, num_heads=args.heads, num_layers=args.layers,
        block_size=args.block_size, seed=args.seed,
    )
    vocab, model = r["vocab"], r["model"]
    print(f"[device] {r['device']}")
    print(f"[data] chars={len(text)} vocab={len(vocab)} (문자 단위)")
    print(f"[model] d_model={args.d_model} heads={args.heads} layers={args.layers} block={args.block_size} "
          f"params={r['params']:,}")
    print(f"[train] {args.steps} steps, {r['seconds']:.2f}s")
    print("  step   train_loss   val_loss")
    for step, tr, va in r["history"]:
        print(f"  {step:>4}   {tr:>10.4f}   {va:>8.4f}")
    best = min(r["history"], key=lambda h: h[2])
    print(f"[val] 최저 검증 손실 {best[2]:.4f} @ step {best[0]} (이후 증가하면 과적합)")

    model.eval()
    prompt = "the "
    ids = torch.tensor([vocab.encode(prompt)], device=r["device"])
    out = model.generate(ids, 60, temperature=0.8, generator=torch.Generator(device=r["device"]).manual_seed(0))
    print(f"[sample] {vocab.decode(out[0].tolist())!r}")

    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "chars": vocab.chars, "config": vars(args)}, CHECKPOINT)
    print(f"[saved] {CHECKPOINT.relative_to(LAB_DIR.parent.parent)}")


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as exc:
        print(f"[미구현] {exc} — starter.py 의 TODO 를 먼저 채우세요 (python part1_ai_fundamentals/transformer/starter.py 로 확인)")

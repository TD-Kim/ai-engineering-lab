"""P1-04 데모: 경사하강법과 DataLoader 학습 루프를 실행하고 결과를 출력한다.

사용법:
    python part1_ai_fundamentals/pytorch/run_demo.py --impl solution
    python part1_ai_fundamentals/pytorch/run_demo.py --impl starter
"""
import argparse
import importlib
import sys
import time
from pathlib import Path

import torch
from torch import nn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=["solution", "starter"], default="solution")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    impl = importlib.import_module(args.impl)

    print(f"[device] {impl.get_device()}")

    # Level 2: y = 2x - 1 을 직접 경사하강법으로 학습
    x = torch.linspace(-1, 1, 50)
    y = 2.0 * x - 1.0
    w, b, hist = impl.train_linear_regression_gd(x, y, lr=0.5, steps=200)
    print(f"[GD] w={w:.4f} (정답 2.0), b={b:.4f} (정답 -1.0), loss {hist[0]:.4f} -> {hist[-1]:.6f}")

    # Level 3: MyLinear vs nn.Linear
    ds = impl.ToyRegressionDataset(n=512, in_features=3, out_features=2, seed=1)
    torch.manual_seed(0)
    mine = impl.MyLinear(3, 2)
    ref = nn.Linear(3, 2)
    with torch.no_grad():
        ref.weight.copy_(mine.weight)
        ref.bias.copy_(mine.bias)
    start = time.perf_counter()
    h_mine = impl.train_model(mine, ds, epochs=30, lr=0.1)
    h_ref = impl.train_model(ref, ds, epochs=30, lr=0.1)
    elapsed = time.perf_counter() - start
    print(f"[MyLinear]  loss {h_mine[0]:.4f} -> {h_mine[-1]:.6f}")
    print(f"[nn.Linear] loss {h_ref[0]:.4f} -> {h_ref[-1]:.6f}")
    print(f"[weight] learned=\n{mine.weight.detach().cpu()}\n true=\n{ds.true_weight}")
    print(f"[time] {elapsed:.2f}s")


# Windows: DataLoader(num_workers>0) 는 spawn 방식이라 메인 가드가 반드시 필요하다.
if __name__ == "__main__":
    main()

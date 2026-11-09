"""P1-04 PyTorch 기초 — 정답 코드.

Level 1: Tensor, shape, broadcasting, device         (TODO 1~3)
Level 2: autograd, 손으로 구한 기울기와 비교, 경사하강법 (TODO 4~6)
Level 3: nn.Module · Dataset · DataLoader 학습 루프, nn.Linear와 비교 (TODO 7~10)

실행: python part1_ai_fundamentals/pytorch/run_demo.py --impl solution
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Level 1 — Tensor 기초
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    # TODO 1  REF: https://pytorch.org/docs/stable/notes/cuda.html
    # 왜: 같은 코드를 GPU가 있는 PC(RTX 3070)와 CPU만 있는 환경에서 모두 돌리기 위해 자동 선택한다.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_rows(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """각 행을 L2 노름 1로 정규화한다. x: (N, D) -> (N, D)"""
    # TODO 2  REF: https://pytorch.org/docs/stable/notes/broadcasting.html
    # norm: (N, 1) — keepdim=True 로 차원을 남겨 두어야 (N, D) / (N, 1) 브로드캐스팅이 행 단위로 맞는다.
    # keepdim 없이 (N,) 로 나누면 (N, D) / (N,) 은 "열" 방향으로 맞춰져 N != D 면 오류, N == D 면 조용히 틀린 값.
    norm = x.norm(dim=-1, keepdim=True)
    # eps: 0 벡터일 때 0으로 나누어 NaN 이 생기는 것을 막는다.
    return x / (norm + eps)


def linear_forward(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """y = x W^T + b.  x: (N, in), weight: (out, in), bias: (out,) -> (N, out)"""
    # TODO 3  REF: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    # nn.Linear 와 같은 규약: weight 는 (out, in) 이므로 전치해서 곱한다.
    # (N, in) @ (in, out) -> (N, out),  bias (out,) 는 N 개 행에 브로드캐스팅된다.
    return x @ weight.T + bias


# ---------------------------------------------------------------------------
# Level 2 — autograd 와 경사하강법
# ---------------------------------------------------------------------------
def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """평균 제곱 오차 (스칼라)."""
    # TODO 4  REF: https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
    # 합(sum)이 아니라 평균(mean): 배치 크기가 바뀌어도 loss·기울기 크기가 일정해 학습률을 다시 고를 필요가 없다.
    return ((pred - target) ** 2).mean()


def manual_mse_grads(
    x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """1차원 선형회귀 pred = w*x + b 에 대한 MSE 의 해석적 기울기 (dL/dw, dL/db).

    x, y: (N,), w, b: 스칼라 텐서
    L = (1/N) Σ (w x_i + b - y_i)^2
    dL/dw = (2/N) Σ (w x_i + b - y_i) x_i
    dL/db = (2/N) Σ (w x_i + b - y_i)
    """
    # TODO 5
    # 왜: autograd 가 "마법"이 아니라 연쇄 법칙(chain rule)을 자동으로 적용한 것임을 확인하기 위해 손으로 구해 비교한다.
    n = x.shape[0]
    residual = w * x + b - y  # (N,)
    grad_w = 2.0 / n * (residual * x).sum()
    grad_b = 2.0 / n * residual.sum()
    return grad_w, grad_b


def train_linear_regression_gd(
    x: torch.Tensor, y: torch.Tensor, lr: float = 0.1, steps: int = 200
) -> tuple[float, float, list[float]]:
    """autograd 로 기울기를 구해 직접 경사하강법으로 w, b 를 학습한다. 반환: (w, b, loss 기록)"""
    w = torch.zeros((), requires_grad=True)
    b = torch.zeros((), requires_grad=True)
    history: list[float] = []
    for _ in range(steps):
        loss = mse_loss(w * x + b, y)
        history.append(loss.item())
        # TODO 6  REF: https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
        loss.backward()  # w.grad, b.grad 에 dL/dw, dL/db 가 "누적"된다.
        # 파라미터 갱신은 계산 그래프에 기록되면 안 되므로 no_grad 안에서 in-place 로 한다.
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
        # 왜 0으로 초기화: .grad 는 덮어쓰기가 아니라 누적이므로, 지우지 않으면 이전 step 의 기울기가 계속 더해져 발산한다.
        w.grad.zero_()
        b.grad.zero_()
    return w.item(), b.item(), history


# ---------------------------------------------------------------------------
# Level 3 — nn.Module · Dataset · DataLoader
# ---------------------------------------------------------------------------
class MyLinear(nn.Module):
    """nn.Linear 를 직접 구현한 것. weight: (out, in), bias: (out,)"""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # TODO 7  REF: https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html
        # nn.Parameter 로 감싸야 model.parameters() 에 등록되어 optimizer 가 갱신한다.
        # 일반 텐서로 두면 학습이 전혀 되지 않는데 오류도 나지 않는다(가장 흔한 실수).
        bound = 1.0 / math.sqrt(in_features)
        # nn.Linear 기본 초기화와 같은 범위 U(-1/sqrt(in), 1/sqrt(in)): 출력 분산이 입력 차원에 따라 폭주하지 않게 한다.
        self.weight = nn.Parameter(torch.empty(out_features, in_features).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(out_features).uniform_(-bound, bound))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 8
        # (N, in) -> (N, out). Level 1 의 linear_forward 를 그대로 재사용한다.
        return linear_forward(x, self.weight, self.bias)


class ToyRegressionDataset(Dataset):
    """y = x A^T + c + noise 형태의 다변량 회귀 데이터."""

    def __init__(self, n: int = 512, in_features: int = 3, out_features: int = 2, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.true_weight = torch.randn(out_features, in_features, generator=g)
        self.true_bias = torch.randn(out_features, generator=g)
        self.x = torch.randn(n, in_features, generator=g)
        noise = 0.01 * torch.randn(n, out_features, generator=g)
        self.y = self.x @ self.true_weight.T + self.true_bias + noise

    # TODO 9  REF: https://pytorch.org/docs/stable/data.html#map-style-datasets
    # DataLoader 는 len() 으로 전체 크기를 알고, [i] 로 샘플 하나씩 꺼내 배치로 묶는다(collate).
    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # 샘플 하나: x (in,), y (out,) — 배치 차원은 DataLoader 가 붙인다.
        return self.x[idx], self.y[idx]


def train_model(
    model: nn.Module,
    dataset: Dataset,
    epochs: int = 20,
    lr: float = 0.05,
    batch_size: int = 32,
    seed: int = 0,
) -> list[float]:
    """표준 PyTorch 학습 루프. 반환: epoch 별 평균 loss."""
    device = get_device()
    model.to(device)
    torch.manual_seed(seed)
    # Windows 에서 num_workers > 0 을 쓰려면 호출하는 스크립트가 `if __name__ == "__main__":` 아래에서 실행되어야 한다.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    history: list[float] = []
    for _ in range(epochs):
        total, count = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)  # 모델과 데이터가 같은 device 에 있어야 한다.
            # TODO 10  REF: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
            # 순서가 중요: zero_grad -> forward -> loss -> backward -> step
            optimizer.zero_grad()          # 이전 배치 기울기 제거 (누적 방지)
            loss = mse_loss(model(xb), yb) # (B, out) vs (B, out)
            loss.backward()                # 모든 Parameter 의 .grad 계산
            optimizer.step()               # p <- p - lr * p.grad
            total += loss.item() * xb.shape[0]
            count += xb.shape[0]
        history.append(total / count)
    return history


# ---------------------------------------------------------------------------
# 단독 실행: 어느 TODO 가 남았는지 알려 준다.  python part1_ai_fundamentals/pytorch/solution.py
# ---------------------------------------------------------------------------
def _report_progress() -> None:
    x = torch.linspace(-1, 1, 8)
    checks = [
        ("TODO 1  get_device (Level 1)", lambda: get_device()),
        ("TODO 2  normalize_rows (Level 1)", lambda: normalize_rows(torch.ones(2, 3))),
        ("TODO 3  linear_forward (Level 1)", lambda: linear_forward(torch.ones(2, 3), torch.ones(4, 3), torch.zeros(4))),
        ("TODO 4  mse_loss (Level 2)", lambda: mse_loss(x, x)),
        ("TODO 5  manual_mse_grads (Level 2)", lambda: manual_mse_grads(x, x, torch.tensor(1.0), torch.tensor(0.0))),
        ("TODO 6  train_linear_regression_gd (Level 2)", lambda: train_linear_regression_gd(x, 2 * x, steps=2)),
        ("TODO 7  MyLinear.__init__ (Level 3)", lambda: MyLinear(3, 2)),
        ("TODO 8  MyLinear.forward (Level 3)", lambda: MyLinear(3, 2)(torch.ones(1, 3))),
        ("TODO 9  ToyRegressionDataset (Level 3)", lambda: ToyRegressionDataset(n=4)[0]),
        ("TODO 10 train_model (Level 3)", lambda: train_model(nn.Linear(3, 2), ToyRegressionDataset(n=8), epochs=1)),
    ]
    remaining = 0
    for name, fn in checks:
        try:
            fn()
            print(f"[완료]   {name}")
        except NotImplementedError:
            remaining += 1
            print(f"[미구현] {name}")
    if remaining:
        print(f"\n남은 TODO {remaining}개 — 위에서부터 순서대로 채우세요. 채운 뒤 검사:")
        print("  (bash)       IMPL=starter pytest part1_ai_fundamentals/pytorch")
        print('  (PowerShell) $env:IMPL="starter"; pytest part1_ai_fundamentals/pytorch; Remove-Item Env:IMPL')
    else:
        print("\n모든 TODO 가 구현되었습니다. 이제 pytest 로 정답 여부를 확인하세요.")


if __name__ == "__main__":
    _report_progress()

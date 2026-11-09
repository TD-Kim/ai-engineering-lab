"""P1-04 PyTorch 기초 — 실습 코드 (TODO 를 채우세요).

Level 1: Tensor, shape, broadcasting, device         (TODO 1~3)
Level 2: autograd, 손으로 구한 기울기와 비교, 경사하강법 (TODO 4~6)
Level 3: nn.Module · Dataset · DataLoader 학습 루프, nn.Linear와 비교 (TODO 7~10)

검사: IMPL=starter pytest part1_ai_fundamentals/pytorch
진행 확인: python part1_ai_fundamentals/pytorch/starter.py   (남은 TODO 목록 출력)
실행: python part1_ai_fundamentals/pytorch/run_demo.py --impl starter
"""
from __future__ import annotations

import math  # noqa: F401  (TODO 7 에서 사용)

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Level 1 — Tensor 기초
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    # TODO 1: GPU 를 쓸 수 있으면 cuda, 아니면 cpu device 를 반환하세요.
    # HINT 1: "GPU 사용 가능 여부"를 묻는 함수가 torch.cuda 모듈에 있습니다.
    # HINT 2: torch.cuda.is_available(), torch.device("...")
    # WHY: 장치를 하드코딩하면 GPU 없는 환경에서 바로 오류가 납니다. 자동 선택이 이식성을 줍니다.
    # REF: https://pytorch.org/docs/stable/notes/cuda.html
    raise NotImplementedError("TODO 1")


def normalize_rows(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """각 행을 L2 노름 1로 정규화한다. x: (N, D) -> (N, D)"""
    # TODO 2: 각 행의 L2 노름으로 그 행을 나누세요.
    # HINT 1: 노름은 "마지막 차원" 방향으로 구하고, 나눌 때 브로드캐스팅이 행 단위로 맞아야 합니다.
    # HINT 2: Tensor.norm(dim=..., keepdim=...) → (N, 1).  0으로 나누기 방지용 eps 를 더하세요.
    # WHY: keepdim 을 빼면 (N, D) / (N,) 이 열 방향으로 맞춰져, N == D 일 때 오류 없이 틀린 값이 나옵니다.
    # REF: https://pytorch.org/docs/stable/notes/broadcasting.html
    raise NotImplementedError("TODO 2")


def linear_forward(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """y = x W^T + b.  x: (N, in), weight: (out, in), bias: (out,) -> (N, out)"""
    # TODO 3: 선형 변환을 계산하세요.
    # HINT 1: weight 의 shape 이 (out, in) 이라는 점에 주의하세요(nn.Linear 규약).
    # HINT 2: 행렬곱 연산자 @ 와 전치 .T, bias 는 브로드캐스팅으로 더해집니다 → (N, out)
    # WHY: nn.Linear 내부가 정확히 이 연산임을 알면 shape 오류를 스스로 고칠 수 있습니다.
    # REF: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    raise NotImplementedError("TODO 3")


# ---------------------------------------------------------------------------
# Level 2 — autograd 와 경사하강법
# ---------------------------------------------------------------------------
def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """평균 제곱 오차 (스칼라)."""
    # TODO 4: (pred - target)^2 의 평균을 반환하세요.
    # HINT 1: 원소별 차이 → 제곱 → 전체 평균
    # HINT 2: 결과는 shape () 인 스칼라 텐서여야 .backward() 를 바로 호출할 수 있습니다.
    # WHY: 합 대신 평균을 쓰면 배치 크기가 바뀌어도 기울기 크기가 비슷해 학습률을 다시 고르지 않아도 됩니다.
    # REF: https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
    raise NotImplementedError("TODO 4")


def manual_mse_grads(
    x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """1차원 선형회귀 pred = w*x + b 에 대한 MSE 의 해석적 기울기 (dL/dw, dL/db).

    x, y: (N,), w, b: 스칼라 텐서
    L = (1/N) Σ (w x_i + b - y_i)^2
    """
    # TODO 5: 연쇄 법칙으로 dL/dw, dL/db 를 직접 계산해 반환하세요 (autograd 사용 금지).
    # HINT 1: 잔차 r_i = w x_i + b - y_i 를 먼저 구하면 두 기울기 모두 r 로 표현됩니다.
    # HINT 2: d(r^2)/dr = 2r,  dr/dw = x_i,  dr/db = 1,  그리고 평균의 1/N
    # WHY: 손으로 구한 값과 autograd 결과가 같다는 것을 확인하면 backward 가 하는 일을 정확히 이해하게 됩니다.
    raise NotImplementedError("TODO 5")


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
        # TODO 6: 역전파 → 파라미터 갱신 → 기울기 초기화를 구현하세요 (optimizer 사용 금지).
        # HINT 1: 갱신 연산 자체가 계산 그래프에 기록되면 안 됩니다.
        # HINT 2: Tensor.backward(), torch.no_grad() 컨텍스트, in-place 연산(-=), Tensor.grad, Tensor.zero_()
        # WHY: .grad 는 누적됩니다. 매 step 초기화하지 않으면 이전 기울기가 더해져 발산합니다.
        # REF: https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
        raise NotImplementedError("TODO 6")
    return w.item(), b.item(), history


# ---------------------------------------------------------------------------
# Level 3 — nn.Module · Dataset · DataLoader
# ---------------------------------------------------------------------------
class MyLinear(nn.Module):
    """nn.Linear 를 직접 구현한 것. weight: (out, in), bias: (out,)"""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # TODO 7: self.weight (out, in), self.bias (out,) 를 학습 가능한 파라미터로 만드세요.
        # HINT 1: 모델의 parameters() 에 잡혀야 optimizer 가 갱신할 수 있습니다.
        # HINT 2: nn.Parameter, torch.empty(shape).uniform_(a, b) — 범위는 ±1/sqrt(in_features) (nn.Linear 기본값)
        # WHY: 일반 텐서로 두면 오류 없이 "학습이 전혀 안 되는" 버그가 생깁니다.
        # REF: https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html
        raise NotImplementedError("TODO 7")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 8: (N, in) -> (N, out) 순전파를 구현하세요.
        # HINT 1: Level 1 에서 만든 함수를 재사용할 수 있습니다.
        # HINT 2: (N, in) -> (N, out). 필요한 weight·bias 는 TODO 7 에서 self 에 만들어 두었습니다.
        # WHY: nn.Module 의 forward 는 "연산 정의"일 뿐이고, 호출은 model(x) 로 합니다(hook 등이 함께 동작).
        raise NotImplementedError("TODO 8")


class ToyRegressionDataset(Dataset):
    """y = x A^T + c + noise 형태의 다변량 회귀 데이터."""

    def __init__(self, n: int = 512, in_features: int = 3, out_features: int = 2, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.true_weight = torch.randn(out_features, in_features, generator=g)
        self.true_bias = torch.randn(out_features, generator=g)
        self.x = torch.randn(n, in_features, generator=g)
        noise = 0.01 * torch.randn(n, out_features, generator=g)
        self.y = self.x @ self.true_weight.T + self.true_bias + noise

    # TODO 9: __len__ 과 __getitem__ 을 구현하세요.
    # HINT 1: DataLoader 는 전체 길이와 "샘플 하나"를 꺼내는 방법만 알면 배치를 만들 수 있습니다.
    # HINT 2: __len__ -> int,  __getitem__(idx) -> (x (in,), y (out,))
    # WHY: 배치 차원은 DataLoader 가 붙입니다. __getitem__ 에서 배치를 만들면 shape 이 한 차원 더 생깁니다.
    # REF: https://pytorch.org/docs/stable/data.html#map-style-datasets
    def __len__(self) -> int:
        raise NotImplementedError("TODO 9")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("TODO 9")


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
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    history: list[float] = []
    for _ in range(epochs):
        total, count = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            # TODO 10: 한 배치에 대한 학습 단계를 구현하고 loss 변수를 만드세요.
            # HINT 1: 기울기 초기화 → 순전파 → loss → 역전파 → 파라미터 갱신 순서입니다.
            # HINT 2: optimizer 의 '초기화'·'갱신' 메서드, TODO 4 의 mse_loss, Tensor.backward(). 결과를 loss 변수에 담으세요.
            # WHY: zero_grad 를 빼면 기울기가 누적되고, step 을 backward 보다 먼저 하면 이전 배치 기울기로 갱신됩니다.
            # REF: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
            raise NotImplementedError("TODO 10")
            total += loss.item() * xb.shape[0]  # noqa: F821
            count += xb.shape[0]
        history.append(total / count)
    return history


# ---------------------------------------------------------------------------
# 단독 실행: 어느 TODO 가 남았는지 알려 준다.  python part1_ai_fundamentals/pytorch/starter.py
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

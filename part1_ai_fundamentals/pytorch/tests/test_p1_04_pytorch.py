"""P1-04 테스트. 기본은 solution 검사, `IMPL=starter pytest ...` 로 직접 구현을 검사한다."""
import torch
from torch import nn


def test_get_device(impl):
    device = impl.get_device()
    assert isinstance(device, torch.device)
    assert device.type == ("cuda" if torch.cuda.is_available() else "cpu")


def test_normalize_rows_square_matrix(impl):
    # N == D 인 경우: keepdim 을 빼먹으면 오류 없이 틀린 값이 나오는 함정
    x = torch.tensor([[3.0, 4.0, 0.0], [1.0, 0.0, 0.0], [2.0, 2.0, 1.0]])
    out = impl.normalize_rows(x)
    assert out.shape == x.shape
    torch.testing.assert_close(out.norm(dim=-1), torch.ones(3))
    torch.testing.assert_close(out[0], torch.tensor([0.6, 0.8, 0.0]))


def test_normalize_rows_zero_vector_no_nan(impl):
    out = impl.normalize_rows(torch.zeros(2, 4))
    assert not torch.isnan(out).any()


def test_linear_forward_matches_nn_linear(impl):
    torch.manual_seed(0)
    layer = nn.Linear(5, 3)
    x = torch.randn(7, 5)
    out = impl.linear_forward(x, layer.weight, layer.bias)
    assert out.shape == (7, 3)
    torch.testing.assert_close(out, layer(x))


def test_mse_loss(impl):
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.0, 0.0, 0.0])
    loss = impl.mse_loss(pred, target)
    assert loss.shape == ()
    torch.testing.assert_close(loss, torch.tensor((0 + 4 + 9) / 3))


def test_manual_grads_match_autograd(impl):
    torch.manual_seed(0)
    x, y = torch.randn(10), torch.randn(10)
    w = torch.tensor(0.7, requires_grad=True)
    b = torch.tensor(-0.3, requires_grad=True)
    loss = ((w * x + b - y) ** 2).mean()
    loss.backward()
    gw, gb = impl.manual_mse_grads(x, y, w.detach(), b.detach())
    torch.testing.assert_close(gw, w.grad)
    torch.testing.assert_close(gb, b.grad)


def test_gradient_descent_recovers_line(impl):
    torch.manual_seed(0)
    x = torch.linspace(-1, 1, 50)
    y = 2.0 * x - 1.0
    w, b, history = impl.train_linear_regression_gd(x, y, lr=0.5, steps=200)
    assert abs(w - 2.0) < 1e-2 and abs(b + 1.0) < 1e-2
    assert history[-1] < history[0]


def test_my_linear_registers_parameters(impl):
    layer = impl.MyLinear(4, 2)
    names = {name for name, _ in layer.named_parameters()}
    assert names == {"weight", "bias"}
    assert layer.weight.shape == (2, 4) and layer.bias.shape == (2,)
    assert layer(torch.randn(3, 4)).shape == (3, 2)


def test_dataset_protocol(impl):
    ds = impl.ToyRegressionDataset(n=20, in_features=3, out_features=2)
    assert len(ds) == 20
    x, y = ds[0]
    assert x.shape == (3,) and y.shape == (2,)


def test_train_model_my_linear_matches_nn_linear(impl):
    """Level 3: 직접 만든 MyLinear 와 nn.Linear 가 같은 초기값·같은 데이터에서 같은 학습 결과를 내야 한다."""
    ds = impl.ToyRegressionDataset(n=256, in_features=3, out_features=2, seed=1)
    torch.manual_seed(0)
    mine = impl.MyLinear(3, 2)
    ref = nn.Linear(3, 2)
    with torch.no_grad():
        ref.weight.copy_(mine.weight)
        ref.bias.copy_(mine.bias)
    h_mine = impl.train_model(mine, ds, epochs=30, lr=0.1, seed=0)
    h_ref = impl.train_model(ref, ds, epochs=30, lr=0.1, seed=0)
    assert h_mine[-1] < 1e-3, "loss 가 충분히 줄지 않았습니다"
    torch.testing.assert_close(torch.tensor(h_mine), torch.tensor(h_ref), rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(mine.weight.detach().cpu(), ds.true_weight, atol=0.02, rtol=0)

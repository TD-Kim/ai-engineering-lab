"""P1-02 테스트. 기본은 solution 검사, `IMPL=starter pytest ...` 로 직접 구현을 검사한다."""
import torch
import torch.nn.functional as F
from torch import nn


def test_one_hot_matches_torch(impl):
    ids = torch.tensor([0, 3, 1, 3])
    out = impl.one_hot(ids, 5)
    assert out.dtype == torch.float32 and out.shape == (4, 5)
    torch.testing.assert_close(out, F.one_hot(ids, 5).float())


def test_cosine_similarity_matrix(impl):
    a = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    b = torch.tensor([[3.0, 0.0], [1.0, 1.0], [-1.0, 0.0]])
    sims = impl.cosine_similarity(a, b)
    assert sims.shape == (2, 3)
    expected = F.cosine_similarity(a.unsqueeze(1), b.unsqueeze(0), dim=-1)
    torch.testing.assert_close(sims, expected)


def test_my_embedding_lookup_shapes(impl):
    emb = impl.MyEmbedding(10, 4)
    assert isinstance(emb.weight, nn.Parameter) and emb.weight.shape == (10, 4)
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    out = emb(ids)
    assert out.shape == (2, 3, 4)
    torch.testing.assert_close(out[1, 2], emb.weight[6])


def test_lookup_equals_one_hot_matmul(impl):
    torch.manual_seed(0)
    weight = torch.randn(8, 3)
    ids = torch.tensor([7, 0, 2, 2])
    torch.testing.assert_close(impl.embedding_via_one_hot(ids, weight), weight[ids])


def test_matches_nn_embedding_outputs_and_grads(impl):
    """Level 3: nn.Embedding 과 같은 가중치면 출력과 기울기가 같아야 한다."""
    torch.manual_seed(0)
    mine = impl.MyEmbedding(6, 3)
    ref = nn.Embedding(6, 3)
    with torch.no_grad():
        ref.weight.copy_(mine.weight)
    ids = torch.tensor([[0, 2, 2], [5, 0, 1]])
    out_m, out_r = mine(ids), ref(ids)
    torch.testing.assert_close(out_m, out_r)
    out_m.sum().backward()
    out_r.sum().backward()
    torch.testing.assert_close(mine.weight.grad, ref.weight.grad)
    # 등장하지 않은 id(3, 4)의 행은 기울기가 0 → 그 토큰은 이번 step 에 갱신되지 않는다
    assert mine.weight.grad[3].abs().sum() == 0
    # id 2 는 두 번 등장 → 기울기가 2 배로 누적
    torch.testing.assert_close(mine.weight.grad[2], torch.full((3,), 2.0))


def test_most_similar(impl):
    weight = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0], [0.7, 0.7]]
    )
    result = impl.most_similar(0, weight, k=2)
    assert [i for i, _ in result] == [1, 4]
    assert result[0][1] > result[1][1]

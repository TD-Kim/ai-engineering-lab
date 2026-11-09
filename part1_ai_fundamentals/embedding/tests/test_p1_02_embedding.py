"""P1-02 테스트. 기본은 solution 검사, `IMPL=starter pytest ...` 로 직접 구현을 검사한다.

모든 테스트는 작은 고정 텐서만 쓰며 외부 다운로드가 없다(사전학습 비교는 조건이 맞을 때만 실행, 아니면 skip).
`impl` 픽스처는 part1_ai_fundamentals/conftest.py 가 파일 경로 기준으로 불러온다.
"""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

LAB_DIR = Path(__file__).resolve().parents[1]


# ----- Level 1 -----
def test_lookup_any_shape(impl):
    table = torch.arange(12.0).view(4, 3)
    ids = torch.tensor([[1, 3], [0, 1]])
    out = impl.lookup(ids, table)
    assert out.shape == (2, 2, 3)
    torch.testing.assert_close(out[0, 1], torch.tensor([9.0, 10.0, 11.0]))


def test_one_hot_matches_torch(impl):
    ids = torch.tensor([0, 3, 1, 3])
    out = impl.one_hot(ids, 5)
    assert out.dtype == torch.float32 and out.shape == (4, 5)
    torch.testing.assert_close(out, F.one_hot(ids, 5).float())


def test_lookup_equals_one_hot_matmul(impl):
    torch.manual_seed(0)
    table = torch.randn(8, 3)
    ids = torch.tensor([7, 0, 2, 2])
    torch.testing.assert_close(impl.embedding_via_one_hot(ids, table), impl.lookup(ids, table))


def test_l2_normalize_square_and_zero(impl):
    x = torch.tensor([[3.0, 4.0], [0.0, 0.0]])  # N == D: keepdim 누락을 잡는다
    out = impl.l2_normalize(x)
    torch.testing.assert_close(out[0], torch.tensor([0.6, 0.8]))
    assert not torch.isnan(out).any()  # 0 벡터에서도 NaN 없음


def test_cosine_similarity_matrix_and_length_invariance(impl):
    a = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    b = torch.tensor([[3.0, 0.0], [1.0, 1.0], [-1.0, 0.0]])
    sims = impl.cosine_similarity(a, b)
    assert sims.shape == (2, 3)
    torch.testing.assert_close(sims, F.cosine_similarity(a.unsqueeze(1), b.unsqueeze(0), dim=-1))
    # 길이를 10 배 해도 코사인 유사도는 그대로 (방향만 비교)
    torch.testing.assert_close(impl.cosine_similarity(a * 10, b), sims)


def test_sentence_vector_ignores_padding(impl):
    table = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [9.0, 9.0]])
    ids = torch.tensor([[1, 2, 3], [1, 2, 0]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    out = impl.sentence_vector(ids, table, mask)
    assert out.shape == (2, 2)
    torch.testing.assert_close(out[0], torch.tensor([10 / 3, 10 / 3]))
    torch.testing.assert_close(out[1], torch.tensor([0.5, 0.5]))  # 패딩(id 0)은 평균에서 제외
    torch.testing.assert_close(impl.sentence_vector(ids[:1], table), out[:1])  # mask=None → 모두 실제 토큰


# ----- Level 2 -----
def test_embedding_layer_parameter_and_lookup(impl):
    torch.manual_seed(0)
    emb = impl.EmbeddingLayer(10, 4)
    assert isinstance(emb.weight, nn.Parameter) and emb.weight.shape == (10, 4)
    assert emb.weight.std() > 0.5  # N(0, 1) 근처, 0 초기화가 아님
    out = emb(torch.tensor([[1, 2, 3], [4, 5, 6]]))
    assert out.shape == (2, 3, 4)
    torch.testing.assert_close(out[1, 2], emb.weight[6])


def test_embedding_layer_padding_idx(impl):
    emb = impl.EmbeddingLayer(5, 3, padding_idx=0)
    assert emb.weight[0].abs().sum() == 0
    emb(torch.tensor([0, 0, 2])).sum().backward()
    assert emb.weight.grad[0].abs().sum() == 0  # 패딩 행은 학습되지 않음
    assert emb.weight.grad[2].abs().sum() > 0


def test_positional_embedding_adds_order(impl):
    torch.manual_seed(0)
    layer = impl.TokenAndPositionEmbedding(vocab_size=6, max_len=4, embedding_dim=8)
    assert isinstance(layer.pos.weight, nn.Parameter) and layer.pos.weight.shape == (4, 8)
    out = layer(torch.tensor([[1, 2, 3], [3, 2, 1]]))
    assert out.shape == (2, 3, 8)
    torch.testing.assert_close(out[0, 0], layer.tok.weight[1] + layer.pos.weight[0])
    # 토큰 임베딩만 쓰면 같은 토큰(id 1)은 어느 위치에 있든 같은 벡터다
    tok_only = layer.tok(torch.tensor([[1, 2, 3], [3, 2, 1]]))
    torch.testing.assert_close(tok_only[0, 0], tok_only[1, 2])
    # 위치를 더하면 같은 토큰도 자리(0 번 vs 2 번)에 따라 다른 벡터가 된다
    assert not torch.allclose(out[0, 0], out[1, 2])
    # 주의: 단순 평균은 위치를 더해도 순서를 구별하지 못한다 (평균(tok+pos) = 평균(tok) + 평균(pos))
    torch.testing.assert_close(out[0].mean(dim=0), out[1].mean(dim=0))
    with pytest.raises(ValueError):
        layer(torch.zeros(1, 5, dtype=torch.long))  # max_len 초과


def test_most_similar_top_k(impl):
    table = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0], [0.7, 0.7]], requires_grad=True)
    result = impl.most_similar(0, table, k=2)
    assert [i for i, _ in result] == [1, 4]  # 자기 자신(0) 제외, 유사도 내림차순
    assert result[0][1] > result[1][1]


# ----- Level 3 -----
def test_matches_nn_embedding_outputs_and_grads(impl):
    """같은 가중치면 nn.Embedding 과 출력·기울기가 같아야 한다 (padding_idx 포함)."""
    torch.manual_seed(0)
    mine = impl.EmbeddingLayer(6, 3, padding_idx=0)
    ref = nn.Embedding(6, 3, padding_idx=0)
    with torch.no_grad():
        ref.weight.copy_(mine.weight)
    ids = torch.tensor([[0, 2, 2], [5, 0, 1]])
    torch.testing.assert_close(mine(ids), ref(ids))
    mine(ids).pow(2).sum().backward()
    ref(ids).pow(2).sum().backward()
    torch.testing.assert_close(mine.weight.grad, ref.weight.grad)
    assert mine.weight.grad[3].abs().sum() == 0  # 등장하지 않은 id 는 기울기 0


def test_pca_2d_finds_main_direction(impl):
    torch.manual_seed(0)
    t = torch.linspace(-1, 1, 50)
    # 3차원 공간에서 거의 한 직선(방향 [1, 2, 2]/3) 위에 놓인 점들
    x = t.unsqueeze(1) * torch.tensor([1.0, 2.0, 2.0]) / 3 + 0.01 * torch.randn(50, 3) + 5.0
    coords = impl.pca_2d(x)
    assert coords.shape == (50, 2)
    torch.testing.assert_close(coords.mean(dim=0), torch.zeros(2), atol=1e-5, rtol=0)  # 중심화
    # 첫 번째 축이 분산 대부분을 설명하고 t 와 (부호 무관) 거의 일치
    assert coords[:, 0].var() > 50 * coords[:, 1].var()
    assert abs(torch.corrcoef(torch.stack([coords[:, 0], t]))[0, 1]) > 0.99


def test_compare_pretrained_optional(impl):
    pytest.importorskip("sentence_transformers", reason="sentence-transformers 미설치 — 사전학습 비교 건너뜀")
    spec = spec_from_file_location("p1_02_compare_pretrained", LAB_DIR / "compare_pretrained.py")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    model, reason = mod.load_model(mod.DEFAULT_MODEL)
    if model is None:
        pytest.skip(f"사전학습 비교 건너뜀: {reason}")
    rows = mod.compare(impl, model)
    # 사전학습 모델은 의미가 비슷한 쌍(0, 2)을 다른 쌍(1, 3)보다 더 가깝게 본다
    assert rows[0]["pretrained_cos"] > rows[1]["pretrained_cos"]
    assert rows[2]["pretrained_cos"] > rows[3]["pretrained_cos"]

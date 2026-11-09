"""P1-02 Embedding — 정답 코드.

Level 1: one-hot 인코딩과 코사인 유사도 (TODO 1~2)
Level 2: 임베딩 = 학습 가능한 룩업 테이블 직접 구현 (TODO 3~5)
Level 3: nn.Embedding 과 출력·기울기 비교, 최근접 이웃 검색 (TODO 6)

왜 Token ID 를 벡터로 바꾸나?
- Token ID 는 "이름표"일 뿐 크기 비교가 의미가 없다(id 7 이 id 3 보다 "더 크다"는 의미 없음).
- one-hot 은 모든 단어 사이 거리가 같아 "비슷함"을 표현할 수 없고, 어휘가 5만 개면 5만 차원이 된다.
- dense 임베딩은 d(예: 64) 차원 실수 벡터로, 학습을 통해 비슷한 쓰임의 토큰이 가까워진다.
"""
from __future__ import annotations

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Level 1
# ---------------------------------------------------------------------------
def one_hot(ids: torch.Tensor, num_classes: int) -> torch.Tensor:
    """ids: (N,) int64 -> (N, num_classes) float32"""
    # TODO 1  REF: https://pytorch.org/docs/stable/generated/torch.nn.functional.one_hot.html
    out = torch.zeros(ids.shape[0], num_classes, dtype=torch.float32, device=ids.device)
    # scatter_: 각 행 i 의 열 ids[i] 위치에 1 을 쓴다. index 는 (N, 1) 이어야 dim=1 방향으로 흩뿌려진다.
    out.scatter_(1, ids.unsqueeze(1), 1.0)
    return out


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """a: (N, D), b: (M, D) -> (N, M) 모든 쌍의 코사인 유사도"""
    # TODO 2
    # cos(a, b) = a·b / (|a||b|).  먼저 정규화하면 행렬곱 한 번으로 N×M 쌍을 모두 구할 수 있다.
    # 왜 코사인: 벡터 길이(빈도 등에 영향을 받음)가 아니라 "방향"만 비교하려고.
    a_n = a / (a.norm(dim=-1, keepdim=True) + eps)  # (N, D)
    b_n = b / (b.norm(dim=-1, keepdim=True) + eps)  # (M, D)
    return a_n @ b_n.T  # (N, D) @ (D, M) -> (N, M)


# ---------------------------------------------------------------------------
# Level 2
# ---------------------------------------------------------------------------
class MyEmbedding(nn.Module):
    """nn.Embedding 직접 구현: (num_embeddings, embedding_dim) 행렬에서 id 번째 행을 꺼낸다."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        # TODO 3  REF: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        # nn.Embedding 기본 초기화와 같은 N(0, 1). 0 으로 초기화하면 모든 토큰이 같은 벡터라 구별이 안 된다.
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # TODO 4
        # 인덱싱 = 룩업. ids 가 어떤 shape 이든 (..., ) -> (..., embedding_dim)
        # 예: (B, T) -> (B, T, D).  one-hot 행렬곱과 결과는 같지만 V 배 적은 연산·메모리로 끝난다.
        return self.weight[ids]


def embedding_via_one_hot(ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """one-hot(ids) @ W 로 임베딩을 구한다. ids: (N,), weight: (V, D) -> (N, D)"""
    # TODO 5
    # (N, V) @ (V, D) -> (N, D).  one-hot 은 W 의 한 행만 1 로 골라 내므로 결과가 W[ids] 와 같다.
    # 즉 "임베딩 층 = 입력이 one-hot 인 bias 없는 Linear 층" 이다.
    return one_hot(ids, weight.shape[0]).to(weight.dtype) @ weight


# ---------------------------------------------------------------------------
# Level 3
# ---------------------------------------------------------------------------
def most_similar(
    query_id: int, weight: torch.Tensor, k: int = 3
) -> list[tuple[int, float]]:
    """임베딩 행렬에서 query_id 와 코사인 유사도가 가장 높은 k 개 (자기 자신 제외)."""
    # TODO 6
    weight = weight.detach()  # 검색은 학습이 아니므로 계산 그래프가 필요 없다.
    sims = cosine_similarity(weight[query_id : query_id + 1], weight).squeeze(0)  # (V,)
    # 자기 자신은 항상 1.0 이라 결과를 가리므로 제외한다.
    sims[query_id] = float("-inf")
    values, indices = torch.topk(sims, k)
    return [(int(i), float(v)) for v, i in zip(values, indices)]

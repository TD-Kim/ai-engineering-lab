"""P1-02 Embedding — 실습 코드 (TODO 를 채우세요).

Level 1: one-hot 인코딩과 코사인 유사도 (TODO 1~2)
Level 2: 임베딩 = 학습 가능한 룩업 테이블 직접 구현 (TODO 3~5)
Level 3: nn.Embedding 과 출력·기울기 비교, 최근접 이웃 검색 (TODO 6)

검사: IMPL=starter pytest part1_ai_fundamentals/embedding
"""
from __future__ import annotations

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Level 1
# ---------------------------------------------------------------------------
def one_hot(ids: torch.Tensor, num_classes: int) -> torch.Tensor:
    """ids: (N,) int64 -> (N, num_classes) float32"""
    # TODO 1: 각 행 i 에서 ids[i] 열만 1 인 행렬을 만드세요 (F.one_hot 사용 금지).
    # HINT 1: 0 으로 채운 행렬을 만든 뒤 해당 위치에만 1 을 씁니다.
    # HINT 2: torch.zeros(N, C), Tensor.scatter_(dim=1, index=(N, 1), value=1.0)
    # WHY: one-hot 을 직접 만들어 보면 "어휘 크기 = 차원"이라 비효율적이라는 것이 바로 보입니다.
    # REF: https://pytorch.org/docs/stable/generated/torch.nn.functional.one_hot.html
    raise NotImplementedError("TODO 1")


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """a: (N, D), b: (M, D) -> (N, M) 모든 쌍의 코사인 유사도"""
    # TODO 2: 모든 (a_i, b_j) 쌍의 코사인 유사도 행렬을 반환하세요.
    # HINT 1: 각 행을 단위 벡터로 만든 뒤 내적하면 코사인 유사도입니다.
    # HINT 2: norm(dim=-1, keepdim=True), a_n @ b_n.T → (N, M)
    # WHY: 길이(크기)가 아니라 방향만 비교해야, 자주 나와서 벡터가 커진 단어가 모든 단어와 "비슷해" 보이지 않습니다.
    raise NotImplementedError("TODO 2")


# ---------------------------------------------------------------------------
# Level 2
# ---------------------------------------------------------------------------
class MyEmbedding(nn.Module):
    """nn.Embedding 직접 구현: (num_embeddings, embedding_dim) 행렬에서 id 번째 행을 꺼낸다."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        # TODO 3: self.weight 를 (num_embeddings, embedding_dim) 학습 파라미터로 만드세요.
        # HINT 1: nn.Embedding 의 기본 초기화는 표준정규분포입니다.
        # HINT 2: nn.Parameter(torch.randn(V, D))
        # WHY: 모두 0 으로 두면 모든 토큰 벡터가 같아 구별할 수 없고, 기울기도 대칭적으로만 흐릅니다.
        # REF: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        raise NotImplementedError("TODO 3")

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # TODO 4: ids 에 해당하는 행들을 꺼내세요. (B, T) -> (B, T, D) 처럼 임의 shape 을 지원해야 합니다.
        # HINT 1: 행렬곱이 필요 없습니다. 인덱스로 고르면 됩니다.
        # HINT 2: 텐서 인덱싱 self.weight[ids]
        # WHY: one-hot @ W 와 같은 결과를 V 배 적은 연산으로 얻습니다. 이것이 임베딩 층의 정체입니다.
        raise NotImplementedError("TODO 4")


def embedding_via_one_hot(ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """one-hot(ids) @ W 로 임베딩을 구한다. ids: (N,), weight: (V, D) -> (N, D)"""
    # TODO 5: TODO 1 의 one_hot 으로 행렬곱 방식 임베딩을 구하세요.
    # HINT 1: one-hot 행렬은 W 의 특정 행만 "선택"합니다.
    # HINT 2: (N, V) @ (V, D) → (N, D), dtype 을 weight 와 맞추세요.
    # WHY: "임베딩 = 입력이 one-hot 인 bias 없는 Linear" 라는 동치를 직접 확인합니다.
    raise NotImplementedError("TODO 5")


# ---------------------------------------------------------------------------
# Level 3
# ---------------------------------------------------------------------------
def most_similar(
    query_id: int, weight: torch.Tensor, k: int = 3
) -> list[tuple[int, float]]:
    """임베딩 행렬에서 query_id 와 코사인 유사도가 가장 높은 k 개 (자기 자신 제외)."""
    # TODO 6: [(id, 유사도), ...] 를 유사도 내림차순으로 반환하세요.
    # HINT 1: 자기 자신의 유사도는 항상 1 이므로 후보에서 빼야 합니다.
    # HINT 2: cosine_similarity(weight[q:q+1], weight).squeeze(0) → (V,), torch.topk
    # WHY: 검색(RAG)의 Dense retrieval 도 결국 이 "코사인 top-k" 입니다(Part 3 로 연결).
    raise NotImplementedError("TODO 6")

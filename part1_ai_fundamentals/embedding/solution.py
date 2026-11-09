"""P1-02 Embedding — 정답 코드.

Level 1: 토큰 ID → 벡터 조회, one-hot × 행렬과의 동치, L2 정규화, 코사인 유사도, 문장 벡터(평균) (TODO 1~6)
Level 2: EmbeddingLayer(nn.Module) 직접 구현, 학습형 위치 임베딩, 최근접 이웃 top-k (TODO 7~11)
Level 3: PCA 2차원 투영 (TODO 12), nn.Embedding 비교(tests), 벡터 공간 시각화(run_demo.py),
         사전학습 임베딩 비교(compare_pretrained.py, 선택)

진행 확인: python part1_ai_fundamentals/embedding/solution.py

왜 Token ID 를 벡터로 바꾸나?
- Token ID 는 이름표다. id 7 과 id 8 이 id 7 과 id 300 보다 "가깝다"는 뜻이 전혀 없다.
- 그런데 신경망은 입력에 곱하고 더한다. ID 를 숫자 그대로 넣으면 이런 의미 없는 크기·거리를 학습 신호로 쓰게 된다.
- 그래서 ID 마다 학습 가능한 실수 벡터를 하나씩 두고(임베딩 테이블), 학습이 그 벡터들의 위치를 정하게 한다.
REF: Mikolov et al., 2013, "Efficient Estimation of Word Representations in Vector Space" (단어의 분산 표현)
"""
from __future__ import annotations

import torch
from torch import nn


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Level 1 — 조회, one-hot, 정규화, 코사인, 문장 벡터
# ---------------------------------------------------------------------------
def lookup(ids: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """ids: 임의 shape (...,) int64, table: (V, D) -> (..., D)"""
    # TODO 1
    # 정수 텐서로 인덱싱하면 각 ID 에 해당하는 "행"을 꺼낸다. (B, T) 를 넣으면 (B, T, D) 가 나온다.
    # 행렬곱이 없으므로 계산량은 O(B·T·D) — 어휘 크기 V 와 무관하다.
    return table[ids]


def one_hot(ids: torch.Tensor, num_classes: int) -> torch.Tensor:
    """ids: (N,) int64 -> (N, num_classes) float32"""
    # TODO 2
    out = torch.zeros(ids.shape[0], num_classes, dtype=torch.float32, device=ids.device)  # (N, V)
    # scatter_(dim=1, index=(N, 1)): 행 i 의 열 ids[i] 에 1 을 쓴다.
    out.scatter_(1, ids.unsqueeze(1), 1.0)
    return out


def embedding_via_one_hot(ids: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """one_hot(ids) @ table. ids: (N,), table: (V, D) -> (N, D)"""
    # TODO 3
    # (N, V) @ (V, D) -> (N, D). one-hot 행은 1 이 하나뿐이라 table 의 한 행만 "선택"된다 → lookup 과 같은 결과.
    # 즉 임베딩 층 = 입력이 one-hot 인 bias 없는 Linear 층. 다만 V 배 많은 곱셈·메모리를 쓴다.
    return one_hot(ids, table.shape[0]).to(table.dtype) @ table


def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """마지막 차원 기준 L2 노름을 1로. x: (..., D) -> (..., D)"""
    # TODO 4
    # keepdim=True: (..., 1) 로 남겨야 (..., D) 와 브로드캐스팅된다.
    # eps: 0 벡터(예: 패딩 행)를 나눌 때 NaN 이 생기는 것을 막는다.
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a: (N, D), b: (M, D) -> (N, M) 모든 쌍의 코사인 유사도"""
    # TODO 5
    # cos(a, b) = a·b / (|a||b|). 먼저 정규화하면 행렬곱 한 번으로 N×M 쌍을 모두 구한다.
    # 왜 코사인인가: 내적은 벡터 길이에 비례한다. 길이는 빈도 등 의미와 무관한 요인으로 커지기 쉬워서,
    # 길이를 지우고 "방향"만 비교해야 의미가 비슷한지를 본다. 값의 범위도 [-1, 1] 로 고정되어 해석하기 쉽다.
    return l2_normalize(a) @ l2_normalize(b).T  # (N, D) @ (D, M) -> (N, M)


def sentence_vector(
    ids: torch.Tensor, table: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """토큰 벡터의 평균으로 문장 벡터를 만든다. ids: (B, T), mask: (B, T) 1=실제/0=패딩 -> (B, D)"""
    # TODO 6
    vecs = lookup(ids, table)  # (B, T, D)
    if mask is None:
        mask = torch.ones(ids.shape, dtype=vecs.dtype, device=ids.device)
    m = mask.to(vecs.dtype).unsqueeze(-1)  # (B, T, 1) — D 축으로 브로드캐스팅
    # 패딩 위치는 0 을 곱해 합에서 빼고, 실제 토큰 수로 나눈다. 패딩까지 평균하면 문장 길이에 따라 벡터가 달라진다.
    # clamp(min=1): 토큰이 하나도 없는 문장에서 0 으로 나누는 것을 막는다.
    return (vecs * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)  # (B, D)


# ---------------------------------------------------------------------------
# Level 2 — EmbeddingLayer, 위치 임베딩, 최근접 이웃
# ---------------------------------------------------------------------------
class EmbeddingLayer(nn.Module):
    """nn.Embedding 직접 구현. weight: (num_embeddings, embedding_dim)"""

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int | None = None):
        super().__init__()
        # TODO 7  REF: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        # 초기화 N(0, 1): nn.Embedding 의 기본값과 같다.
        # - 0 으로 두면 모든 토큰 벡터가 같아 구별이 안 되고, 기울기도 모든 행에 똑같이 흘러 대칭이 깨지지 않는다.
        # - 무작위 값이면 처음부터 토큰마다 다른 방향을 가져 학습이 서로 다른 위치로 밀어낼 수 있다.
        # (Transformer 처럼 sqrt(D) 를 곱하거나 std 를 작게 쓰는 등 모델마다 조정한다 → P1-05)
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        self.padding_idx = padding_idx
        if padding_idx is not None:
            with torch.no_grad():
                self.weight[padding_idx].zero_()  # 패딩은 "아무 정보 없음" → 0 벡터
            # 패딩 행은 학습되지 않게 기울기를 0 으로 만든다(nn.Embedding 의 padding_idx 와 같은 동작).
            # 이렇게 하지 않으면 패딩이 많은 배치에서 패딩 벡터가 의미를 갖도록 학습되어 버린다.
            self.weight.register_hook(lambda g: g.index_fill(0, torch.tensor([padding_idx], device=g.device), 0.0))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # TODO 8
        # (...,) -> (..., D). 인덱싱 결과는 autograd 로 추적되어, 등장한 ID 의 행에만 기울기가 흐른다.
        return self.weight[ids]


class PositionalEmbedding(nn.Module):
    """학습형(learned) 위치 임베딩: 위치 0..max_len-1 마다 벡터 하나."""

    def __init__(self, max_len: int, embedding_dim: int):
        super().__init__()
        # TODO 9  REF: Vaswani et al., 2017, "Attention Is All You Need", §3.5 (learned positional embeddings 언급)
        # 위치도 토큰처럼 "ID(0, 1, 2, ...)" 이므로 같은 방식의 테이블로 표현한다. 단 max_len 보다 긴 문장은 처리할 수 없다.
        # 표준편차 0.02 로 작게 시작: 처음에는 토큰 임베딩(std 1)을 크게 흔들지 않고, 학습하면서 위치 정보를 키운다.
        self.weight = nn.Parameter(torch.randn(max_len, embedding_dim) * 0.02)

    def forward(self, seq_len: int) -> torch.Tensor:
        if seq_len > self.weight.shape[0]:
            raise ValueError(f"seq_len {seq_len} > max_len {self.weight.shape[0]}")
        return self.weight[:seq_len]  # (T, D)


class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, vocab_size: int, max_len: int, embedding_dim: int, padding_idx: int | None = None):
        super().__init__()
        self.tok = EmbeddingLayer(vocab_size, embedding_dim, padding_idx)
        self.pos = PositionalEmbedding(max_len, embedding_dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: (B, T) -> (B, T, D)"""
        # TODO 10
        # (B, T, D) + (T, D): 위치 벡터는 배치 축으로 브로드캐스팅되어 모든 문장의 같은 위치에 같은 벡터가 더해진다.
        # 왜 필요한가: 토큰 벡터만 쓰면 "개가 사람을 문다"의 "개"와 "사람이 개를 문다"의 "개"가 완전히 같은 벡터다.
        # 어텐션처럼 위치를 보지 않는 연산은 입력 순서를 섞어도 출력이 똑같이 섞일 뿐이라 순서를 알 수 없다.
        # 위치 벡터를 더하면 같은 토큰도 자리마다 다른 벡터가 되어, 뒤의 층이 순서를 쓸 수 있다.
        # (주의: 단순 평균은 위치를 더해도 순서를 구별하지 못한다. 평균(tok+pos) = 평균(tok) + 평균(pos) 이기 때문.)
        return self.tok(ids) + self.pos(ids.shape[1])


def most_similar(query_id: int, table: torch.Tensor, k: int = 3) -> list[tuple[int, float]]:
    """table 에서 query_id 와 코사인 유사도가 가장 높은 k 개 (자기 자신 제외). [(id, 유사도), ...] 내림차순"""
    # TODO 11
    table = table.detach()  # 검색은 학습이 아니므로 계산 그래프가 필요 없다
    sims = cosine_similarity(table[query_id : query_id + 1], table).squeeze(0)  # (1, V) -> (V,)
    # 자기 자신은 항상 1.0 이라 결과를 가리므로 제외한다.
    sims[query_id] = float("-inf")
    values, indices = torch.topk(sims, k)  # 큰 값부터 k 개
    return [(int(i), float(v)) for v, i in zip(values, indices, strict=True)]


# ---------------------------------------------------------------------------
# Level 3 — PCA 2차원 투영
# ---------------------------------------------------------------------------
def pca_2d(x: torch.Tensor) -> torch.Tensor:
    """x: (N, D) -> (N, 2). 분산이 가장 큰 두 방향으로 투영한다."""
    # TODO 12
    x = x.detach().float()
    # 평균을 빼야(중심화) 주성분이 "원점에서 데이터까지"가 아니라 "데이터가 퍼진 방향"을 가리킨다.
    centered = x - x.mean(dim=0, keepdim=True)  # (N, D)
    # SVD: centered = U S Vᵀ. Vᵀ 의 앞 두 행이 분산이 가장 큰 두 방향(주성분)이다.
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)  # vh: (min(N, D), D)
    return centered @ vh[:2].T  # (N, D) @ (D, 2) -> (N, 2)


# ---------------------------------------------------------------------------
# 단독 실행: 어느 TODO 가 남았는지 알려 준다.
# ---------------------------------------------------------------------------
def _report_progress() -> None:
    table = torch.randn(5, 3)
    ids = torch.tensor([0, 2])
    checks = [
        ("TODO 1  lookup (Level 1)", lambda: lookup(ids, table)),
        ("TODO 2  one_hot (Level 1)", lambda: one_hot(ids, 5)),
        ("TODO 3  embedding_via_one_hot (Level 1)", lambda: embedding_via_one_hot(ids, table)),
        ("TODO 4  l2_normalize (Level 1)", lambda: l2_normalize(table)),
        ("TODO 5  cosine_similarity (Level 1)", lambda: cosine_similarity(table, table)),
        ("TODO 6  sentence_vector (Level 1)", lambda: sentence_vector(ids.view(1, 2), table)),
        ("TODO 7  EmbeddingLayer.__init__ (Level 2)", lambda: EmbeddingLayer(5, 3)),
        ("TODO 8  EmbeddingLayer.forward (Level 2)", lambda: EmbeddingLayer(5, 3)(ids)),
        ("TODO 9  PositionalEmbedding.__init__ (Level 2)", lambda: PositionalEmbedding(4, 3)),
        ("TODO 10 TokenAndPositionEmbedding.forward (Level 2)", lambda: TokenAndPositionEmbedding(5, 4, 3)(ids.view(1, 2))),
        ("TODO 11 most_similar (Level 2)", lambda: most_similar(0, table, k=2)),
        ("TODO 12 pca_2d (Level 3)", lambda: pca_2d(table)),
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
        print(f"\n남은 TODO {remaining}개 — 위에서부터 순서대로 채우세요(뒤 TODO 는 앞 TODO 를 사용합니다). 채운 뒤 검사:")
        print("  (bash)       IMPL=starter pytest part1_ai_fundamentals/embedding")
        print('  (PowerShell) $env:IMPL="starter"; pytest part1_ai_fundamentals/embedding; Remove-Item Env:IMPL')
    else:
        print("\n모든 TODO 가 구현되었습니다. 이제 pytest 로 정답 여부를 확인하세요.")


if __name__ == "__main__":
    _report_progress()

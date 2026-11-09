"""P1-02 Embedding — 실습 코드 (TODO 를 채우세요).

Level 1: 토큰 ID → 벡터 조회, one-hot × 행렬과의 동치, L2 정규화, 코사인 유사도, 문장 벡터(평균) (TODO 1~6)
Level 2: EmbeddingLayer(nn.Module) 직접 구현, 학습형 위치 임베딩, 최근접 이웃 top-k (TODO 7~11)
Level 3: PCA 2차원 투영 (TODO 12), nn.Embedding 비교(tests), 벡터 공간 시각화(run_demo.py),
         사전학습 임베딩 비교(compare_pretrained.py, 선택)

진행 확인: python part1_ai_fundamentals/embedding/advanced/starter.py   (남은 TODO 목록 출력)
검사:     IMPL=starter pytest part1_ai_fundamentals/embedding/advanced

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
    # TODO 1: ids 에 해당하는 table 의 행들을 꺼내세요. ids 가 (B, T) 이면 결과는 (B, T, D) 입니다.
    # HINT 1: 행렬곱이 필요 없습니다. "몇 번째 행"을 고르는 연산입니다.
    # HINT 2: 정수 텐서로 텐서를 인덱싱하면 index 의 shape 뒤에 나머지 차원이 붙습니다.
    # WHY: 조회 비용은 어휘 크기 V 와 무관합니다. 어휘가 5만 개여도 문장 길이만큼만 계산합니다.
    raise NotImplementedError("TODO 1")


def one_hot(ids: torch.Tensor, num_classes: int) -> torch.Tensor:
    """ids: (N,) int64 -> (N, num_classes) float32"""
    # TODO 2: 각 행 i 에서 ids[i] 열만 1 인 (N, num_classes) float32 행렬을 만드세요 (F.one_hot 사용 금지).
    # HINT 1: 0 으로 채운 행렬을 만든 뒤 해당 위치에만 1 을 씁니다.
    # HINT 2: torch.zeros, Tensor.scatter_(dim, index, value) — index 는 (N, 1) 이어야 합니다.
    # WHY: one-hot 은 "어휘 크기 = 차원"이고 모든 단어 쌍의 거리가 같습니다. 비슷함을 표현할 수 없다는 점을 직접 확인합니다.
    raise NotImplementedError("TODO 2")


def embedding_via_one_hot(ids: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """one_hot(ids) @ table. ids: (N,), table: (V, D) -> (N, D)"""
    # TODO 3: TODO 2 의 one-hot 행렬과 table 의 행렬곱으로 임베딩을 구하세요.
    # HINT 1: one-hot 행 하나는 table 의 행 하나만 "선택"합니다.
    # HINT 2: (N, V) @ (V, D) → (N, D). dtype 을 table 과 맞추세요.
    # WHY: 결과가 TODO 1 과 같다는 것 = "임베딩 층은 입력이 one-hot 인 bias 없는 Linear" 라는 뜻입니다.
    raise NotImplementedError("TODO 3")


def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """마지막 차원 기준 L2 노름을 1로. x: (..., D) -> (..., D)"""
    # TODO 4: 마지막 차원의 L2 노름으로 나눠 길이를 1 로 만드세요. 0 벡터에서 NaN 이 나오지 않게 eps 를 더합니다.
    # HINT 1: 노름은 마지막 차원 방향으로 구하고, 나눌 때 차원이 맞아야 합니다.
    # HINT 2: Tensor.norm(dim=..., keepdim=...)
    # WHY: keepdim 을 빼면 (N, D) / (N,) 이 열 방향으로 맞춰져 N == D 일 때 오류 없이 틀린 값이 나옵니다(P1-04 TODO 2 와 같은 함정).
    raise NotImplementedError("TODO 4")


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a: (N, D), b: (M, D) -> (N, M) 모든 쌍의 코사인 유사도"""
    # TODO 5: a 의 모든 행과 b 의 모든 행 사이의 코사인 유사도 (N, M) 행렬을 반환하세요.
    # HINT 1: 정규화한 벡터끼리의 내적이 곧 코사인 유사도입니다.
    # HINT 2: TODO 4 를 재사용하고, (N, D) 와 (M, D) 를 행렬곱하려면 한쪽을 전치해야 합니다.
    # WHY: 내적은 벡터 길이에 비례합니다. 길이는 빈도처럼 의미와 무관한 이유로 커지는 경향이 있어, 방향만 비교해야 의미가 비슷한지 봅니다.
    raise NotImplementedError("TODO 5")


def sentence_vector(
    ids: torch.Tensor, table: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """토큰 벡터의 평균으로 문장 벡터를 만든다. ids: (B, T), mask: (B, T) 1=실제/0=패딩 -> (B, D)"""
    # TODO 6: 각 문장의 실제 토큰(mask == 1) 벡터만 평균해 (B, D) 를 반환하세요. mask 가 None 이면 모두 실제 토큰입니다.
    # HINT 1: 패딩 위치에 0 을 곱해 합에서 빼고, 실제 토큰 수로 나눕니다.
    # HINT 2: lookup → (B, T, D), mask → (B, T, 1) 로 늘려 곱하고 dim=1 로 합합니다. 토큰 수가 0 이면 0 으로 나누지 않게 clamp.
    # WHY: 패딩까지 평균하면 같은 문장도 배치의 최대 길이에 따라 벡터가 달라집니다.
    raise NotImplementedError("TODO 6")


# ---------------------------------------------------------------------------
# Level 2 — EmbeddingLayer, 위치 임베딩, 최근접 이웃
# ---------------------------------------------------------------------------
class EmbeddingLayer(nn.Module):
    """nn.Embedding 직접 구현. weight: (num_embeddings, embedding_dim)"""

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int | None = None):
        super().__init__()
        # TODO 7: self.weight 를 (num_embeddings, embedding_dim) 학습 파라미터로 만들고 self.padding_idx 를 저장하세요.
        #         padding_idx 가 주어지면 그 행을 0 으로 만들고, 그 행에는 기울기가 흐르지 않게 하세요.
        # HINT 1: nn.Embedding 의 기본 초기화는 표준정규분포입니다. 패딩 행은 "학습 중에도" 0 이어야 합니다.
        # HINT 2: nn.Parameter, torch.randn, torch.no_grad() 안에서 zero_(), Tensor.register_hook (기울기를 바꿔 돌려주는 함수)
        # WHY: 0 으로만 초기화하면 처음엔 모든 토큰 벡터가 같아 구별할 수 없고, 뒤 층도 0 이면 학습으로도 깨지지 않습니다.
        #      패딩 행이 학습되면 "빈칸"이 의미를 갖게 됩니다.
        # REF: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        raise NotImplementedError("TODO 7")

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # TODO 8: ids (...,) 에 해당하는 행들을 반환하세요 → (..., D)
        # HINT 1: TODO 1 과 같은 연산을 self.weight 에 합니다.
        # HINT 2: 인덱싱 결과도 autograd 가 추적합니다(별도 처리 불필요).
        # WHY: 인덱싱 덕분에 이번 배치에 등장한 토큰의 행에만 기울기가 흐릅니다.
        raise NotImplementedError("TODO 8")


class PositionalEmbedding(nn.Module):
    """학습형(learned) 위치 임베딩: 위치 0..max_len-1 마다 벡터 하나."""

    def __init__(self, max_len: int, embedding_dim: int):
        super().__init__()
        # TODO 9: 위치 0..max_len-1 각각의 벡터를 담는 self.weight (max_len, D) 학습 파라미터를 만드세요.
        # HINT 1: 위치도 0, 1, 2, ... 라는 ID 이므로 토큰 임베딩과 같은 모양의 테이블입니다.
        # HINT 2: 작은 표준편차(예: 0.02)의 정규분포로 시작합니다. nn.Parameter, torch.randn
        # WHY: 처음부터 큰 값이면 토큰 임베딩을 덮어 버립니다. 작게 시작해 학습이 위치 정보의 크기를 정하게 합니다.
        # REF: Vaswani et al., 2017, "Attention Is All You Need", §3.5 (learned positional embeddings 언급)
        raise NotImplementedError("TODO 9")

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
        # TODO 10: 토큰 임베딩 (B, T, D) 에 위치 임베딩 (T, D) 을 더해 반환하세요.
        # HINT 1: 위치 벡터는 문장 길이 T 만큼만 필요합니다.
        # HINT 2: self.tok(ids), self.pos(T) — (B, T, D) + (T, D) 는 배치 축으로 브로드캐스팅됩니다.
        # WHY: 위치를 더하지 않으면 "개가 사람을 문다"와 "사람이 개를 문다"의 "개"가 같은 벡터라,
        #      위치를 보지 않는 어텐션은 두 문장의 순서 차이를 알 수 없습니다(단순 평균은 위치를 더해도 순서를 모릅니다).
        raise NotImplementedError("TODO 10")


def most_similar(query_id: int, table: torch.Tensor, k: int = 3) -> list[tuple[int, float]]:
    """table 에서 query_id 와 코사인 유사도가 가장 높은 k 개 (자기 자신 제외). [(id, 유사도), ...] 내림차순"""
    # TODO 11: query_id 와 코사인 유사도가 높은 순서로 k 개 [(id, 유사도), ...] 를 반환하세요. 자기 자신은 제외합니다.
    # HINT 1: 자기 자신의 유사도는 항상 1 이라 1 등을 차지합니다. 후보에서 빼야 합니다.
    # HINT 2: TODO 5 로 (1, V) 유사도를 구하고 squeeze, 자기 위치를 -inf 로, torch.topk. 계산 그래프는 필요 없습니다(detach).
    # WHY: Part 3 의 Dense 검색도 결국 "코사인 top-k" 입니다. 같은 함수가 문장 검색으로 이어집니다.
    raise NotImplementedError("TODO 11")


# ---------------------------------------------------------------------------
# Level 3 — PCA 2차원 투영
# ---------------------------------------------------------------------------
def pca_2d(x: torch.Tensor) -> torch.Tensor:
    """x: (N, D) -> (N, 2). 분산이 가장 큰 두 방향으로 투영한다."""
    # TODO 12: x 를 평균 0 으로 중심화한 뒤, 분산이 가장 큰 두 방향으로 투영한 (N, 2) 를 반환하세요.
    # HINT 1: 중심화한 행렬의 SVD 에서 오른쪽 특이벡터(Vᵀ 의 행)가 주성분 방향입니다.
    # HINT 2: torch.linalg.svd(..., full_matrices=False), 앞 두 개 방향과 행렬곱 → (N, D) @ (D, 2)
    # WHY: 중심화하지 않으면 첫 번째 "주성분"이 데이터의 평균 방향을 가리켜 퍼진 모양을 보여 주지 못합니다.
    raise NotImplementedError("TODO 12")


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
        print("  (bash)       IMPL=starter pytest part1_ai_fundamentals/embedding/advanced")
        print('  (PowerShell) $env:IMPL="starter"; pytest part1_ai_fundamentals/embedding/advanced; Remove-Item Env:IMPL')
    else:
        print("\n모든 TODO 가 구현되었습니다. 이제 pytest 로 정답 여부를 확인하세요.")


if __name__ == "__main__":
    _report_progress()

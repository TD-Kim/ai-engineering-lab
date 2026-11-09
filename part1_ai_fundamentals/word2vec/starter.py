"""P1-03 Word2Vec — 실습 코드 (TODO 를 채우세요).

Level 1: 어휘 구축, Skip-gram 쌍 / CBOW 예제 만들기 (TODO 1~3)
Level 2: Negative Sampling Skip-gram, Full-softmax CBOW 직접 구현 (TODO 4~9)
Level 3: gensim Word2Vec 과 비교 → compare_with_gensim.py

검사: IMPL=starter pytest part1_ai_fundamentals/word2vec

REF:
- Mikolov et al., 2013a, "Efficient Estimation of Word Representations in Vector Space", §3 (CBOW, Skip-gram)
- Mikolov et al., 2013b, "Distributed Representations of Words and Phrases and their Compositionality", §2.2 (Negative Sampling)
"""
from __future__ import annotations

import random
import time
from collections import Counter

import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------------------
# 데이터: 의미 그룹이 분명한 작은 영어 말뭉치를 코드로 생성한다(다운로드·라이선스 문제 없음).
# ---------------------------------------------------------------------------
GROUPS: dict[str, list[str]] = {
    "animal": ["cat", "dog", "rabbit", "horse"],
    "vehicle": ["car", "bus", "truck", "train"],
    "fruit": ["apple", "banana", "orange", "grape"],
}
_TEMPLATES = {
    "animal": [
        ("the", "{w}", ["eats", "sleeps", "plays", "runs"], "in", "the", ["garden", "field", "yard"]),
    ],
    "vehicle": [
        ("the", "{w}", ["drives", "stops", "moves", "parks"], "on", "the", ["road", "street", "highway"]),
    ],
    "fruit": [
        ("i", "eat", "a", "{w}", "for", ["breakfast", "lunch", "dinner"]),
        ("the", "{w}", "is", ["sweet", "fresh", "ripe"]),
    ],
}


def make_toy_corpus(n_sentences: int = 3000, seed: int = 0) -> list[list[str]]:
    rng = random.Random(seed)
    sentences = []
    for _ in range(n_sentences):
        group = rng.choice(list(GROUPS))
        template = rng.choice(_TEMPLATES[group])
        word = rng.choice(GROUPS[group])
        sent = []
        for slot in template:
            if slot == "{w}":
                sent.append(word)
            elif isinstance(slot, list):
                sent.append(rng.choice(slot))
            else:
                sent.append(slot)
        sentences.append(sent)
    return sentences


# ---------------------------------------------------------------------------
# Level 1
# ---------------------------------------------------------------------------
def build_vocab(
    sentences: list[list[str]], min_count: int = 1
) -> tuple[dict[str, int], list[str], list[int]]:
    """반환: (word_to_id, id_to_word, counts). id 는 빈도 내림차순(동점은 사전순)."""
    # TODO 1: 단어 빈도를 세고 min_count 미만을 버린 뒤, 빈도 내림차순(동점은 사전순)으로 id 를 매기세요.
    # HINT 1: 반환값은 (word_to_id, id_to_word, counts) 이며 counts[i] 는 id i 단어의 빈도입니다.
    # HINT 2: collections.Counter, list.sort(key=lambda wc: (-wc[1], wc[0]))
    # WHY: 정렬 기준이 없으면 실행마다 id 가 달라지고, 희귀 단어를 남기면 근거 없는 벡터가 어휘만 키웁니다.
    raise NotImplementedError("TODO 1")


def skipgram_pairs(ids: list[int], window: int) -> list[tuple[int, int]]:
    """(center, context) 쌍. 중심 단어 앞뒤 window 개 이내의 모든 단어가 context."""
    # TODO 2: 각 위치 i 의 중심 단어와, 앞뒤 window 개 이내(문장 안)의 모든 단어로 (center, context) 쌍을 만드세요.
    # HINT 1: 문장 경계에서 범위를 잘라 내고, 자기 자신(j == i)은 제외합니다.
    # HINT 2: lo = max(0, i - window), hi = min(len(ids), i + window + 1)
    # WHY: 경계를 넘기면 다른 문장의 단어가 문맥으로 섞이고, 자기 자신을 넣으면 "자기 예측"이라는 쓸모없는 신호가 생깁니다.
    # REF: Mikolov et al., 2013a, "Efficient Estimation of Word Representations in Vector Space", §3.2
    raise NotImplementedError("TODO 2")


def cbow_examples(ids: list[int], window: int) -> list[tuple[list[int], int]]:
    """(context 2*window 개, center). 양쪽 문맥이 모두 있는 위치만 사용한다."""
    # TODO 3: 양쪽 문맥이 모두 있는 위치 i 에 대해 (앞 window 개 + 뒤 window 개, 중심 단어) 를 만드세요.
    # HINT 1: i 의 범위는 window 부터 len(ids) - window - 1 까지입니다.
    # HINT 2: 리스트 슬라이싱 ids[i - window : i] + ids[i + 1 : i + window + 1]
    # WHY: 문맥 길이를 2*window 로 고정해야 배치를 (B, 2*window) 텐서로 쌓을 수 있습니다.
    # REF: Mikolov et al., 2013a, §3.1
    raise NotImplementedError("TODO 3")


# ---------------------------------------------------------------------------
# Level 2 — Negative Sampling
# ---------------------------------------------------------------------------
def noise_distribution(counts: list[int], power: float = 0.75) -> torch.Tensor:
    """P_n(w) ∝ count(w)^power  (합이 1 인 (V,) 텐서)"""
    # TODO 4: count^power 를 정규화한 확률 분포를 반환하세요.
    # HINT 1: 합이 1 이 되어야 합니다.
    # HINT 2: torch.tensor(counts, dtype=torch.float64) ** power, 마지막에 .float()
    # WHY: power=1 이면 "the" 같은 고빈도어만, power=0 이면 희귀어가 너무 자주 음성 샘플로 뽑힙니다.
    # REF: Mikolov et al., 2013b, "Distributed Representations of Words and Phrases and their Compositionality", §2.2
    raise NotImplementedError("TODO 4")


def sample_negatives(
    noise: torch.Tensor, batch_size: int, k: int, generator: torch.Generator | None = None
) -> torch.Tensor:
    """noise 분포에서 (batch_size, k) 개의 음성 단어 id 를 복원 추출한다."""
    # TODO 5: noise 분포에서 batch_size * k 개를 복원 추출해 (batch_size, k) 로 만드세요.
    # HINT 1: 같은 단어가 여러 번 뽑혀도 됩니다(복원 추출).
    # HINT 2: torch.multinomial(noise, n, replacement=True, generator=generator).view(...)
    # WHY: generator 를 넘겨야 같은 seed 에서 같은 음성 샘플이 나와 실험을 재현할 수 있습니다.
    raise NotImplementedError("TODO 5")


class SkipGramNegSampling(nn.Module):
    """중심 단어 벡터 v_c 와 문맥 단어 벡터 u_o 를 서로 다른 테이블로 학습한다."""

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        # TODO 6: self.in_embed (중심 단어용), self.out_embed (문맥 단어용) 두 nn.Embedding 을 만들고 초기화하세요.
        # HINT 1: 원 구현은 입력 테이블을 아주 작은 균등분포, 출력 테이블을 0 으로 초기화합니다.
        # HINT 2: nn.Embedding(V, D), nn.init.uniform_(w, -0.5 / dim, 0.5 / dim), nn.init.zeros_(w)
        # WHY: 테이블을 하나만 쓰면 자기 자신과의 내적 v_w·v_w 가 항상 커서 학습 신호가 왜곡됩니다.
        # REF: Mikolov et al., 2013b, §2.2
        raise NotImplementedError("TODO 6")

    def forward(self, center: torch.Tensor, context: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        """center, context: (B,), negatives: (B, K) -> 스칼라 loss

        loss = -log σ(u_o·v_c) - Σ_k log σ(-u_k·v_c)
        """
        # TODO 7: 양성 점수 u_o·v_c 와 음성 점수 u_k·v_c 를 구해 위 식의 배치 평균을 반환하세요.
        # HINT 1: 양성은 (B, D)·(B, D) 원소곱 후 합 → (B,),  음성은 (B, K, D) 와 (B, D) 의 배치 행렬곱 → (B, K)
        # HINT 2: torch.bmm(u_neg, v.unsqueeze(-1)).squeeze(-1),  F.logsigmoid
        # WHY: log(torch.sigmoid(x)) 로 나눠 계산하면 x 가 큰 음수일 때 log(0) = -inf 가 되어 학습이 깨집니다.
        # REF: Mikolov et al., 2013b, §2.2 (식 4)
        raise NotImplementedError("TODO 7")


class CBOWSoftmax(nn.Module):
    """문맥 단어 벡터의 평균으로 중심 단어를 맞힌다(작은 어휘라 full softmax 사용)."""

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        # TODO 8: self.embed (V, D) 와 bias 없는 출력층 self.out (D -> V) 을 만드세요.
        # HINT 1: 출력층 가중치가 곧 "출력용 단어 벡터" 테이블 역할을 합니다.
        # HINT 2: nn.Embedding(V, D), nn.Linear(D, V, bias=False)
        # WHY: 작은 어휘에서는 full softmax 가 가능하지만, V 가 수십만이면 이 층이 병목이라 음성 샘플링이 필요해집니다.
        # REF: Mikolov et al., 2013a, §3.1
        raise NotImplementedError("TODO 8")

    def forward(self, contexts: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """contexts: (B, 2w), target: (B,) -> 스칼라 loss"""
        # TODO 9: 문맥 벡터 평균 → 어휘 전체 logits → cross entropy 를 반환하세요.
        # HINT 1: 문맥 축(dim=1)으로 평균을 냅니다.
        # HINT 2: (B, 2w, D) -> (B, D) -> (B, V),  F.cross_entropy(logits, target)
        # WHY: 평균은 문맥 단어의 순서를 버립니다. 그래서 "Continuous Bag-of-Words" 입니다.
        # REF: Mikolov et al., 2013a, §3.1
        raise NotImplementedError("TODO 9")


# ---------------------------------------------------------------------------
# 학습 루프 (제공)
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_word2vec(
    sentences: list[list[str]],
    model_type: str = "skipgram",
    dim: int = 32,
    window: int = 2,
    negatives: int = 5,
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 0.01,
    seed: int = 0,
    device: torch.device | None = None,
) -> dict:
    """반환: {"model", "word_to_id", "id_to_word", "losses"(epoch 평균), "vectors"(V, D), "seconds"}"""
    torch.manual_seed(seed)
    device = device or get_device()
    word_to_id, id_to_word, counts = build_vocab(sentences)
    id_sents = [[word_to_id[w] for w in s if w in word_to_id] for s in sentences]
    V = len(id_to_word)

    if model_type == "skipgram":
        data = [p for s in id_sents for p in skipgram_pairs(s, window)]
        model = SkipGramNegSampling(V, dim)
        noise = noise_distribution(counts)
    elif model_type == "cbow":
        data = [e for s in id_sents for e in cbow_examples(s, window)]
        model = CBOWSoftmax(V, dim)
    else:
        raise ValueError(model_type)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    rng = random.Random(seed)

    start = time.perf_counter()
    losses = []
    for _ in range(epochs):
        rng.shuffle(data)
        total, n = 0.0, 0
        for b in range(0, len(data), batch_size):
            batch = data[b : b + batch_size]
            if model_type == "skipgram":
                center = torch.tensor([c for c, _ in batch], device=device)
                context = torch.tensor([o for _, o in batch], device=device)
                neg = sample_negatives(noise, len(batch), negatives, generator=gen).to(device)
                loss = model(center, context, neg)
            else:
                ctx = torch.tensor([c for c, _ in batch], device=device)
                tgt = torch.tensor([t for _, t in batch], device=device)
                loss = model(ctx, tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
            n += len(batch)
        losses.append(total / n)
    seconds = time.perf_counter() - start
    vectors = (model.in_embed.weight if model_type == "skipgram" else model.embed.weight).detach().cpu()
    return {
        "model": model,
        "word_to_id": word_to_id,
        "id_to_word": id_to_word,
        "losses": losses,
        "vectors": vectors,
        "seconds": seconds,
    }


def nearest(word: str, result: dict, k: int = 3) -> list[tuple[str, float]]:
    """학습된 벡터에서 코사인 최근접 단어 k 개."""
    vecs = F.normalize(result["vectors"], dim=-1)
    idx = result["word_to_id"][word]
    sims = vecs @ vecs[idx]
    sims[idx] = -2.0
    values, ids = torch.topk(sims, k)
    return [(result["id_to_word"][i], float(v)) for v, i in zip(values, ids)]


def group_purity(neighbors_fn, k: int = 3) -> float:
    """각 그룹 단어의 최근접 k 개 중 같은 그룹 단어 비율의 평균 (1.0 이 최고)."""
    scores = []
    for words in GROUPS.values():
        for w in words:
            nbrs = [n for n, _ in neighbors_fn(w, k)]
            scores.append(sum(n in words for n in nbrs) / k)
    return sum(scores) / len(scores)

"""P1-03 Word2Vec — 정답 코드.

Level 1: 어휘 구축, Skip-gram 쌍 / CBOW 예제 만들기 (TODO 1~3)
Level 2: Negative Sampling Skip-gram, Full-softmax CBOW 직접 구현 (TODO 4~9)
Level 3: gensim Word2Vec 과 비교 → compare_with_gensim.py

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
    # TODO 1
    counter = Counter(w for s in sentences for w in s)
    # min_count 미만 단어 제거: 한두 번 나온 단어는 벡터를 학습할 근거가 부족하고 어휘만 키운다.
    items = [(w, c) for w, c in counter.items() if c >= min_count]
    # 빈도순 정렬: 결정적인 id + 자주 나오는 단어가 작은 id (gensim 도 같은 관례)
    items.sort(key=lambda wc: (-wc[1], wc[0]))
    id_to_word = [w for w, _ in items]
    word_to_id = {w: i for i, w in enumerate(id_to_word)}
    counts = [c for _, c in items]
    return word_to_id, id_to_word, counts


def skipgram_pairs(ids: list[int], window: int) -> list[tuple[int, int]]:
    """(center, context) 쌍. 중심 단어 앞뒤 window 개 이내의 모든 단어가 context."""
    # TODO 2  REF: Mikolov et al., 2013a, §3.2
    pairs = []
    for i, center in enumerate(ids):
        # 문장 경계에서 범위를 잘라낸다(문장 밖 단어는 문맥이 아니다).
        lo, hi = max(0, i - window), min(len(ids), i + window + 1)
        for j in range(lo, hi):
            if j != i:  # 자기 자신은 문맥이 아니다
                pairs.append((center, ids[j]))
    return pairs


def cbow_examples(ids: list[int], window: int) -> list[tuple[list[int], int]]:
    """(context 2*window 개, center). 양쪽 문맥이 모두 있는 위치만 사용한다."""
    # TODO 3  REF: Mikolov et al., 2013a, §3.1
    examples = []
    for i in range(window, len(ids) - window):
        context = ids[i - window : i] + ids[i + 1 : i + window + 1]
        # 고정 길이(2*window)로 맞추면 배치를 (B, 2*window) 텐서로 쌓을 수 있다.
        # (실전 구현은 패딩/마스크로 문장 끝 위치도 쓴다 → 과제)
        examples.append((context, ids[i]))
    return examples


# ---------------------------------------------------------------------------
# Level 2 — Negative Sampling
# ---------------------------------------------------------------------------
def noise_distribution(counts: list[int], power: float = 0.75) -> torch.Tensor:
    """P_n(w) ∝ count(w)^power  (합이 1 인 (V,) 텐서)"""
    # TODO 4  REF: Mikolov et al., 2013b, §2.2
    # 3/4 제곱: 그냥 빈도(power=1)면 "the" 같은 단어만 음성 샘플로 뽑히고,
    # 균등(power=0)이면 희귀 단어가 너무 자주 뽑힌다. 논문에서 실험적으로 가장 좋았던 절충값.
    probs = torch.tensor(counts, dtype=torch.float64) ** power
    return (probs / probs.sum()).float()


def sample_negatives(
    noise: torch.Tensor, batch_size: int, k: int, generator: torch.Generator | None = None
) -> torch.Tensor:
    """noise 분포에서 (batch_size, k) 개의 음성 단어 id 를 복원 추출한다."""
    # TODO 5
    # multinomial(replacement=True): 같은 단어가 여러 번 뽑힐 수 있다(word2vec 원 구현도 같음).
    # (실제 정답 문맥 단어가 우연히 뽑히는 경우는 드물어 무시한다 — 원 구현과 같은 단순화)
    flat = torch.multinomial(noise, batch_size * k, replacement=True, generator=generator)
    return flat.view(batch_size, k)


class SkipGramNegSampling(nn.Module):
    """중심 단어 벡터 v_c 와 문맥 단어 벡터 u_o 를 서로 다른 테이블로 학습한다."""

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        # TODO 6  REF: Mikolov et al., 2013b, §2.2
        # 왜 테이블 두 개: 같은 테이블을 쓰면 v_w·v_w (자기 자신과의 내적)가 항상 커서
        # "단어는 자기 자신의 문맥이다"라는 잘못된 신호가 생긴다. 입력/출력 역할을 분리한다.
        self.in_embed = nn.Embedding(vocab_size, dim)   # v: 중심 단어(최종적으로 쓰는 단어 벡터)
        self.out_embed = nn.Embedding(vocab_size, dim)  # u: 문맥 단어
        # word2vec 원 구현(C 코드)과 같은 초기화: 입력은 작은 균등분포, 출력은 0.
        nn.init.uniform_(self.in_embed.weight, -0.5 / dim, 0.5 / dim)
        nn.init.zeros_(self.out_embed.weight)

    def forward(self, center: torch.Tensor, context: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        """center, context: (B,), negatives: (B, K) -> 스칼라 loss

        loss = -log σ(u_o·v_c) - Σ_k log σ(-u_k·v_c)
        """
        # TODO 7  REF: Mikolov et al., 2013b, §2.2 (식 4)
        v = self.in_embed(center)          # (B, D)
        u_pos = self.out_embed(context)    # (B, D)
        u_neg = self.out_embed(negatives)  # (B, K, D)
        pos_score = (v * u_pos).sum(-1)                         # (B,)
        # bmm: (B, K, D) @ (B, D, 1) -> (B, K, 1) -> (B, K)
        neg_score = torch.bmm(u_neg, v.unsqueeze(-1)).squeeze(-1)
        # 왜 logsigmoid: log(sigmoid(x)) 를 따로 계산하면 x 가 큰 음수일 때 log(0) = -inf 가 된다.
        # 왜 음성 샘플링: 전체 어휘 V 에 대한 softmax 분모 대신 K(5~20)개만 보고 이진 분류로 근사 → V/K 배 빠름.
        loss = -(F.logsigmoid(pos_score) + F.logsigmoid(-neg_score).sum(-1))
        return loss.mean()


class CBOWSoftmax(nn.Module):
    """문맥 단어 벡터의 평균으로 중심 단어를 맞힌다(작은 어휘라 full softmax 사용)."""

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        # TODO 8  REF: Mikolov et al., 2013a, §3.1
        self.embed = nn.Embedding(vocab_size, dim)
        # bias 없는 Linear: 출력 가중치 (V, D) 가 곧 "출력용 단어 벡터" 테이블이다.
        self.out = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, contexts: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """contexts: (B, 2w), target: (B,) -> 스칼라 loss"""
        # TODO 9  REF: Mikolov et al., 2013a, §3.1
        # 평균: 문맥 단어의 "순서"를 버린다 → 그래서 bag-of-words(CBOW) 라는 이름이 붙었다.
        h = self.embed(contexts).mean(dim=1)  # (B, 2w, D) -> (B, D)
        logits = self.out(h)                  # (B, V)
        # cross_entropy 가 log_softmax + NLL 을 수치적으로 안정되게 한 번에 계산한다.
        return F.cross_entropy(logits, target)


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

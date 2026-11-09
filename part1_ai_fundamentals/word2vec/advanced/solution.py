"""P1-03 Word2Vec — 정답 코드.

Level 1: 단어→ID 어휘, 윈도우 기반 Skip-gram 쌍 / CBOW 예제(패딩+마스크) 만들기 (TODO 1~3)
Level 2: 음성 샘플링 분포(unigram^3/4), 음성 샘플 추출, Skip-gram + Negative Sampling, CBOW(마스크 평균 + softmax),
         Skip-gram 학습 한 단계 (TODO 4~10)
Level 3: 코사인 최근접 이웃 (TODO 11), gensim 비교(compare_with_gensim.py), 학습 전후 비교(run_demo.py)

진행 확인: python part1_ai_fundamentals/word2vec/advanced/solution.py

분포 가설(Distributional Hypothesis): "비슷한 문맥에 나오는 단어는 비슷한 뜻을 갖는다."
Word2Vec 은 이 가설을 그대로 학습 목표로 만든다. 단어의 벡터로 주변(윈도우 안) 단어를 맞히게(Skip-gram) 하거나,
주변 단어로 가운데 단어를 맞히게(CBOW) 하면, 주변이 비슷한 단어끼리 벡터가 비슷해질 수밖에 없다.

REF:
- Mikolov et al., 2013a, "Efficient Estimation of Word Representations in Vector Space", §3 (CBOW §3.1, Skip-gram §3.2)
- Mikolov et al., 2013b, "Distributed Representations of Words and Phrases and their Compositionality", §2.2 (Negative Sampling)
"""
from __future__ import annotations

import random
import re
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

CORPUS = Path(__file__).resolve().parents[2] / "data" / "corpus_ko_en.txt"


# ---------------------------------------------------------------------------
# 데이터 (제공)
# ---------------------------------------------------------------------------
def tokenize_line(line: str) -> list[str]:
    """소문자로 바꾸고 단어(한글·영문·숫자)만 남긴다. "mat." → "mat" """
    return re.findall(r"\w+", line.lower())


def load_corpus(path: Path = CORPUS) -> list[list[str]]:
    """data/corpus_ko_en.txt → 문장(줄)별 단어 리스트"""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [toks for toks in (tokenize_line(line) for line in lines) if toks]


# 테스트·비교용 작은 생성 말뭉치: 의미 그룹(동물/탈것/과일)이 분명해 결과를 정량 평가하기 쉽다.
GROUPS: dict[str, list[str]] = {
    "animal": ["cat", "dog", "rabbit", "horse"],
    "vehicle": ["car", "bus", "truck", "train"],
    "fruit": ["apple", "banana", "orange", "grape"],
}
_TEMPLATES = {
    "animal": [("the", "{w}", ["eats", "sleeps", "plays", "runs"], "in", "the", ["garden", "field", "yard"])],
    "vehicle": [("the", "{w}", ["drives", "stops", "moves", "parks"], "on", "the", ["road", "street", "highway"])],
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
        sentences.append([
            word if slot == "{w}" else rng.choice(slot) if isinstance(slot, list) else slot
            for slot in template
        ])
    return sentences


# ---------------------------------------------------------------------------
# Level 1 — 어휘, 윈도우 쌍
# ---------------------------------------------------------------------------
def build_vocab(
    sentences: list[list[str]], min_count: int = 1
) -> tuple[dict[str, int], list[str], list[int]]:
    """반환: (word_to_id, id_to_word, counts). id 는 빈도 내림차순(동점은 사전순)."""
    # TODO 1
    counter = Counter(w for s in sentences for w in s)
    # min_count 미만 단어 제거: 한두 번 나온 단어는 벡터를 학습할 근거가 부족하고 어휘만 키운다.
    items = [(w, c) for w, c in counter.items() if c >= min_count]
    # 빈도순 정렬: 실행마다 같은 id(재현성) + 자주 나오는 단어가 작은 id (gensim 도 같은 관례)
    items.sort(key=lambda wc: (-wc[1], wc[0]))
    id_to_word = [w for w, _ in items]                         # list: id → 단어
    word_to_id = {w: i for i, w in enumerate(id_to_word)}      # dict: 단어 → id
    counts = [c for _, c in items]                             # counts[id] = 빈도 (음성 샘플링 분포에 사용)
    return word_to_id, id_to_word, counts


def skipgram_pairs(ids: list[int], window: int) -> list[tuple[int, int]]:
    """(center, context) 쌍. 중심 단어 앞뒤 window 개 이내(같은 문장)의 모든 단어가 context."""
    # TODO 2  REF: Mikolov et al., 2013a, §3.2
    pairs = []
    for i, center in enumerate(ids):
        # 문장 경계에서 범위를 자른다: 다른 문장의 단어는 문맥이 아니다.
        lo, hi = max(0, i - window), min(len(ids), i + window + 1)
        for j in range(lo, hi):
            if j != i:  # 자기 자신은 문맥이 아니다(자기 예측은 쓸모없는 신호)
                pairs.append((center, ids[j]))
    return pairs


def cbow_examples(ids: list[int], window: int, pad_id: int) -> list[tuple[list[int], int]]:
    """모든 위치 i 에 대해 (문맥 2*window 개, 중심 단어). 문맥이 모자라면 pad_id 로 채운다."""
    # TODO 3  REF: Mikolov et al., 2013a, §3.1
    examples = []
    for i in range(len(ids)):
        context = ids[max(0, i - window) : i] + ids[i + 1 : i + window + 1]
        # 길이를 2*window 로 고정해야 배치를 (B, 2*window) 텐서로 쌓을 수 있다. 빈 자리는 pad_id, 모델이 마스크로 뺀다.
        # 왜 모든 위치를 쓰나: "양쪽 문맥이 다 있는 위치"만 쓰면 짧은 문장의 단어(특히 문장 앞쪽 단어)가
        # 중심 단어로 거의 등장하지 않아 학습 신호를 잃는다(README 9절 CBOW 실패 기록).
        examples.append((context + [pad_id] * (2 * window - len(context)), ids[i]))
    return examples


# ---------------------------------------------------------------------------
# Level 2 — Negative Sampling, 모델, 학습 단계
# ---------------------------------------------------------------------------
def noise_distribution(counts: list[int], power: float = 0.75) -> torch.Tensor:
    """P_n(w) ∝ count(w)^power  (합이 1 인 (V,) 텐서)"""
    # TODO 4  REF: Mikolov et al., 2013b, §2.2
    # 3/4 제곱: 빈도 그대로(power=1)면 "the" 같은 고빈도어가 음성 샘플을 거의 독차지하고,
    # 균등(power=0)이면 희귀어가 너무 자주 뽑힌다. 논문은 unigram 분포의 3/4 제곱이 실험에서 가장 좋았다고 보고한다.
    # 예: 빈도 16 과 1 → 그대로면 16:1, 3/4 제곱이면 8:1 로 격차가 줄어 고빈도어 과대표집이 완화된다.
    probs = torch.tensor(counts, dtype=torch.float64) ** power  # float64: 큰 빈도에서도 정밀도 유지
    return (probs / probs.sum()).float()                       # (V,), 합 1


def sample_negatives(
    noise: torch.Tensor, batch_size: int, k: int, generator: torch.Generator | None = None
) -> torch.Tensor:
    """noise 분포에서 (batch_size, k) 개의 음성 단어 id 를 복원 추출한다."""
    # TODO 5
    # replacement=True: 같은 단어가 여러 번 뽑힐 수 있다(원 구현과 같음).
    # 정답 문맥 단어가 우연히 뽑히는 경우는 드물어 무시한다(원 구현과 같은 단순화).
    flat = torch.multinomial(noise, batch_size * k, replacement=True, generator=generator)  # (B*K,)
    return flat.view(batch_size, k)                                                          # (B, K)


class SkipGramNegSampling(nn.Module):
    """중심 단어 벡터 v (in_embed) 와 문맥 단어 벡터 u (out_embed) 를 서로 다른 테이블로 학습한다."""

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        # TODO 6  REF: Mikolov et al., 2013b, §2.2
        # 왜 테이블 두 개: 하나만 쓰면 점수가 v_w·v_w(자기 자신과의 내적)를 포함하게 되는데, 이 값은 항상 커서
        # "단어는 자기 자신의 문맥이다"라는 잘못된 신호가 된다. 입력(중심)/출력(문맥) 역할을 분리한다.
        self.in_embed = nn.Embedding(vocab_size, dim)   # v: 최종적으로 "단어 벡터"로 쓰는 쪽
        self.out_embed = nn.Embedding(vocab_size, dim)  # u: 문맥 쪽
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
        pos_score = (v * u_pos).sum(-1)                              # (B,)
        neg_score = torch.bmm(u_neg, v.unsqueeze(-1)).squeeze(-1)    # (B, K, D) @ (B, D, 1) -> (B, K)
        # 왜 음성 샘플링: softmax 는 분모에 어휘 V 개 전부의 exp(u_w·v_c) 합이 필요해 단어 하나마다 O(V) 계산이다.
        # 대신 "진짜 문맥(1) vs 무작위 단어 K 개(0)" 이진 분류로 바꾸면 O(K) (K=5~20) 로 줄어든다.
        # 왜 logsigmoid: log(sigmoid(x)) 를 나눠 계산하면 x 가 큰 음수일 때 log(0) = -inf 가 된다.
        loss = -(F.logsigmoid(pos_score) + F.logsigmoid(-neg_score).sum(-1))  # (B,)
        return loss.mean()


class CBOWSoftmax(nn.Module):
    """문맥 단어 벡터의 (패딩 제외) 평균으로 중심 단어를 맞힌다. 작은 어휘라 full softmax 를 쓴다."""

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.pad_id = vocab_size  # 실제 단어 id 는 0..V-1, 패딩은 V
        # TODO 8  REF: Mikolov et al., 2013a, §3.1
        # 패딩 자리까지 V+1 행. padding_idx: 패딩 행은 0 벡터로 고정되고 학습되지 않는다(P1-02 TODO 7 과 같은 개념).
        self.embed = nn.Embedding(vocab_size + 1, dim, padding_idx=self.pad_id)
        # bias 없는 Linear: 가중치 (V, D) 가 "출력용 단어 벡터" 테이블 역할을 한다. 패딩은 예측 대상이 아니므로 V 개.
        self.out = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, contexts: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """contexts: (B, 2w) (패딩 = pad_id), target: (B,) -> 스칼라 loss"""
        # TODO 9  REF: Mikolov et al., 2013a, §3.1
        mask = (contexts != self.pad_id).float().unsqueeze(-1)       # (B, 2w, 1)
        summed = (self.embed(contexts) * mask).sum(dim=1)            # (B, D)
        # 실제 문맥 수로 나눈 평균: 패딩까지 나누면 문장 끝 단어의 문맥 벡터가 작아져 크기가 들쭉날쭉해진다.
        # 평균은 문맥 단어의 "순서"를 버린다 → bag-of-words(CBOW) 라는 이름이 붙은 이유.
        h = summed / mask.sum(dim=1).clamp(min=1.0)                  # (B, D)
        logits = self.out(h)                                         # (B, V)
        # cross_entropy = log_softmax + NLL 을 수치적으로 안정되게 한 번에. V 가 수십만이면 이 줄이 병목 → 음성 샘플링.
        return F.cross_entropy(logits, target)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def skipgram_batch_loss(
    model: SkipGramNegSampling,
    batch: list[tuple[int, int]],
    noise: torch.Tensor,
    negatives: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """(center, context) 쌍 목록 하나를 텐서로 바꾸고 음성 샘플을 뽑아 loss 를 계산한다."""
    # TODO 10
    center = torch.tensor([c for c, _ in batch], device=device)   # (B,)
    context = torch.tensor([o for _, o in batch], device=device)  # (B,)
    # 음성 샘플은 배치마다 새로 뽑는다: 같은 음성 단어만 계속 쓰면 그 단어들만 밀어내는 편향이 생긴다.
    # generator 는 CPU 에서 뽑고 device 로 옮긴다(재현성: 같은 seed → 같은 음성 샘플).
    neg = sample_negatives(noise, len(batch), negatives, generator=generator).to(device)  # (B, K)
    return model(center, context, neg)


# ---------------------------------------------------------------------------
# 학습 루프 (제공)
# ---------------------------------------------------------------------------
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
    """반환: {"model", "word_to_id", "id_to_word", "counts", "losses"(epoch 평균), "vectors_before", "vectors", "seconds"}"""
    torch.manual_seed(seed)
    device = device or get_device()
    word_to_id, id_to_word, counts = build_vocab(sentences)
    id_sents = [[word_to_id[w] for w in s if w in word_to_id] for s in sentences]
    vocab_size = len(id_to_word)

    if model_type == "skipgram":
        data = [p for s in id_sents for p in skipgram_pairs(s, window)]
        model = SkipGramNegSampling(vocab_size, dim)
        noise = noise_distribution(counts)
    elif model_type == "cbow":
        model = CBOWSoftmax(vocab_size, dim)
        data = [e for s in id_sents for e in cbow_examples(s, window, model.pad_id)]
    else:
        raise ValueError(model_type)

    def word_vectors() -> torch.Tensor:
        # 두 모델 모두 "입력 쪽" 테이블을 단어 벡터로 쓴다(원 논문·gensim 의 wv 와 같은 쪽).
        w = model.in_embed.weight if model_type == "skipgram" else model.embed.weight[:vocab_size]
        return w.detach().cpu().clone()

    vectors_before = word_vectors()
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
                loss = skipgram_batch_loss(model, batch, noise, negatives, gen, device)
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
    return {
        "model": model,
        "word_to_id": word_to_id,
        "id_to_word": id_to_word,
        "counts": counts,
        "losses": losses,
        "vectors_before": vectors_before,
        "vectors": word_vectors(),
        "seconds": time.perf_counter() - start,
    }


# ---------------------------------------------------------------------------
# Level 3 — 최근접 이웃
# ---------------------------------------------------------------------------
def nearest(word: str, result: dict, k: int = 3, vectors: torch.Tensor | None = None) -> list[tuple[str, float]]:
    """학습된(또는 주어진) 단어 벡터에서 코사인 최근접 단어 k 개 (자기 자신 제외)."""
    # TODO 11
    vecs = F.normalize(result["vectors"] if vectors is None else vectors, dim=-1)  # (V, D), 길이 1
    idx = result["word_to_id"][word]
    sims = vecs @ vecs[idx]   # (V,) 코사인 유사도 — 길이를 지우고 방향만 비교(P1-02 TODO 5)
    sims[idx] = -2.0          # 자기 자신 제외(코사인 최솟값 -1 보다 작게)
    values, ids = torch.topk(sims, k)
    return [(result["id_to_word"][i], float(v)) for v, i in zip(values, ids, strict=True)]


def cosine(word_a: str, word_b: str, result: dict, vectors: torch.Tensor | None = None) -> float:
    """(제공) 두 단어 벡터의 코사인 유사도."""
    vecs = result["vectors"] if vectors is None else vectors
    a, b = vecs[result["word_to_id"][word_a]], vecs[result["word_to_id"][word_b]]
    return float(F.cosine_similarity(a, b, dim=0))


def group_purity(neighbors_fn, k: int = 3) -> float:
    """(제공) 각 그룹 단어의 최근접 k 개 중 같은 그룹 단어 비율의 평균 (1.0 이 최고. 무작위 기대값은 3/(V-1): 생성 말뭉치 V=40 이면 약 0.077)."""
    scores = []
    for words in GROUPS.values():
        for w in words:
            nbrs = [n for n, _ in neighbors_fn(w, k)]
            scores.append(sum(n in words for n in nbrs) / k)
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# 단독 실행: 어느 TODO 가 남았는지 알려 준다.
# ---------------------------------------------------------------------------
def _report_progress() -> None:
    sents = [["a", "b", "c"], ["b", "c"]]
    noise = torch.tensor([0.5, 0.5])
    checks = [
        ("TODO 1  build_vocab (Level 1)", lambda: build_vocab(sents)),
        ("TODO 2  skipgram_pairs (Level 1)", lambda: skipgram_pairs([0, 1, 2], 1)),
        ("TODO 3  cbow_examples (Level 1)", lambda: cbow_examples([0, 1, 2], 1, pad_id=3)),
        ("TODO 4  noise_distribution (Level 2)", lambda: noise_distribution([3, 1])),
        ("TODO 5  sample_negatives (Level 2)", lambda: sample_negatives(noise, 2, 3)),
        ("TODO 6  SkipGramNegSampling.__init__ (Level 2)", lambda: SkipGramNegSampling(3, 4)),
        ("TODO 7  SkipGramNegSampling.forward (Level 2)",
         lambda: SkipGramNegSampling(3, 4)(torch.tensor([0]), torch.tensor([1]), torch.tensor([[2]]))),
        ("TODO 8  CBOWSoftmax.__init__ (Level 2)", lambda: CBOWSoftmax(3, 4)),
        ("TODO 9  CBOWSoftmax.forward (Level 2)", lambda: CBOWSoftmax(3, 4)(torch.tensor([[1, 3]]), torch.tensor([0]))),
        ("TODO 10 skipgram_batch_loss (Level 2)",
         lambda: skipgram_batch_loss(SkipGramNegSampling(3, 4), [(0, 1)], torch.tensor([0.3, 0.3, 0.4]), 2,
                                     torch.Generator().manual_seed(0), torch.device("cpu"))),
        ("TODO 11 nearest (Level 3)",
         lambda: nearest("a", {"vectors": torch.eye(3), "word_to_id": {"a": 0}, "id_to_word": ["a", "b", "c"]}, k=1)),
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
        print("  (bash)       IMPL=starter pytest part1_ai_fundamentals/word2vec/advanced")
        print('  (PowerShell) $env:IMPL="starter"; pytest part1_ai_fundamentals/word2vec/advanced; Remove-Item Env:IMPL')
    else:
        print("\n모든 TODO 가 구현되었습니다. 이제 pytest 로 정답 여부를 확인하세요.")


if __name__ == "__main__":
    _report_progress()

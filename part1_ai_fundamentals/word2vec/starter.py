"""P1-03 Word2Vec — 실습 코드 (TODO 를 채우세요).

Level 1: 단어→ID 어휘, 윈도우 기반 Skip-gram 쌍 / CBOW 예제(패딩+마스크) 만들기 (TODO 1~3)
Level 2: 음성 샘플링 분포(unigram^3/4), 음성 샘플 추출, Skip-gram + Negative Sampling, CBOW(마스크 평균 + softmax),
         Skip-gram 학습 한 단계 (TODO 4~10)
Level 3: 코사인 최근접 이웃 (TODO 11), gensim 비교(compare_with_gensim.py), 학습 전후 비교(run_demo.py)

진행 확인: python part1_ai_fundamentals/word2vec/starter.py   (남은 TODO 목록 출력)
검사:     IMPL=starter pytest part1_ai_fundamentals/word2vec

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
from collections import Counter  # noqa: F401  (TODO 1 에서 사용)
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus_ko_en.txt"


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
    # TODO 1: 단어 빈도를 세고 min_count 미만을 버린 뒤, 빈도 내림차순(동점은 사전순)으로 id 를 매기세요.
    #         반환: (word_to_id, id_to_word, counts) — counts[i] 는 id i 단어의 빈도
    # HINT 1: 정렬 기준이 (빈도 큰 순, 같으면 사전순) 두 단계입니다.
    # HINT 2: collections.Counter, list.sort(key=...) 에 튜플 키 (빈도의 음수, 단어)
    # WHY: 정렬이 없으면 실행마다 id 가 달라지고, counts 순서가 id 와 어긋나면 음성 샘플링 분포가 엉뚱한 단어를 가리킵니다.
    raise NotImplementedError("TODO 1")


def skipgram_pairs(ids: list[int], window: int) -> list[tuple[int, int]]:
    """(center, context) 쌍. 중심 단어 앞뒤 window 개 이내(같은 문장)의 모든 단어가 context."""
    # TODO 2: 각 위치 i 의 중심 단어와, 앞뒤 window 개 이내(같은 문장)의 단어들로 (center, context) 쌍을 만드세요.
    # HINT 1: 문장 경계에서 범위를 잘라 내고, 자기 자신(j == i)은 제외합니다.
    # HINT 2: 범위 시작은 0 보다 작아질 수 없고, 끝은 len(ids) 를 넘을 수 없습니다(max, min).
    # WHY: 분포 가설 — "주변 단어"가 곧 학습 신호입니다. 경계를 넘기면 다른 문장 단어가 섞이고, 자기 자신을 넣으면 쓸모없는 신호가 생깁니다.
    # REF: Mikolov et al., 2013a, "Efficient Estimation of Word Representations in Vector Space", §3.2
    raise NotImplementedError("TODO 2")


def cbow_examples(ids: list[int], window: int, pad_id: int) -> list[tuple[list[int], int]]:
    """모든 위치 i 에 대해 (문맥 2*window 개, 중심 단어). 문맥이 모자라면 pad_id 로 채운다."""
    # TODO 3: 모든 위치 i 에 대해 (앞 window 개 + 뒤 window 개 문맥, 중심 단어) 를 만들고,
    #         문맥이 2*window 개보다 적으면 뒤를 pad_id 로 채우세요.
    # HINT 1: 문장 앞·뒤 끝에서는 한쪽 문맥이 모자랍니다. 그래도 예제를 버리지 않습니다.
    # HINT 2: 리스트 슬라이싱(시작이 음수가 되지 않게 max), [pad_id] * n
    # WHY: "양쪽 문맥이 다 있는 위치"만 쓰면 짧은 문장 앞쪽 단어가 중심 단어로 거의 나오지 않아 학습 신호를 잃습니다
    #      (이 저장소에서 실제로 CBOW 테스트가 실패했던 원인 — README 9절).
    # REF: Mikolov et al., 2013a, §3.1
    raise NotImplementedError("TODO 3")


# ---------------------------------------------------------------------------
# Level 2 — Negative Sampling, 모델, 학습 단계
# ---------------------------------------------------------------------------
def noise_distribution(counts: list[int], power: float = 0.75) -> torch.Tensor:
    """P_n(w) ∝ count(w)^power  (합이 1 인 (V,) 텐서)"""
    # TODO 4: count^power 를 정규화한 확률 분포 (V,) 를 반환하세요 (합 1).
    # HINT 1: 거듭제곱 후 전체 합으로 나눕니다.
    # HINT 2: float64 로 계산한 뒤 마지막에 float32 로 바꾸면 정밀도 문제가 없습니다.
    # WHY: power=1 이면 "the" 같은 고빈도어가 음성 샘플을 독차지하고, power=0 이면 희귀어가 과하게 뽑힙니다.
    #      3/4 제곱은 그 사이에서 고빈도어 과대표집을 완화하며, 논문에서 실험적으로 가장 좋았던 값입니다.
    # REF: Mikolov et al., 2013b, "Distributed Representations of Words and Phrases and their Compositionality", §2.2
    raise NotImplementedError("TODO 4")


def sample_negatives(
    noise: torch.Tensor, batch_size: int, k: int, generator: torch.Generator | None = None
) -> torch.Tensor:
    """noise 분포에서 (batch_size, k) 개의 음성 단어 id 를 복원 추출한다."""
    # TODO 5: noise 분포에서 batch_size * k 개를 복원 추출해 (batch_size, k) 로 만드세요.
    # HINT 1: 같은 단어가 여러 번 뽑혀도 됩니다(복원 추출).
    # HINT 2: torch.multinomial(..., replacement=True, generator=generator), Tensor.view
    # WHY: generator 를 넘겨야 같은 seed 에서 같은 음성 샘플이 나와 실험을 재현할 수 있습니다.
    raise NotImplementedError("TODO 5")


class SkipGramNegSampling(nn.Module):
    """중심 단어 벡터 v (in_embed) 와 문맥 단어 벡터 u (out_embed) 를 서로 다른 테이블로 학습한다."""

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        # TODO 6: self.in_embed (중심 단어용), self.out_embed (문맥 단어용) 두 nn.Embedding(V, D) 를 만들고 초기화하세요.
        # HINT 1: 원 구현(C 코드 word2vec.c)은 입력 테이블을 아주 작은 균등분포(±0.5/D), 출력 테이블을 0 으로 초기화합니다.
        # HINT 2: nn.Embedding, nn.init.uniform_, nn.init.zeros_
        # WHY: 테이블을 하나만 쓰면 자기 자신과의 내적이 항상 커서 학습 신호가 왜곡됩니다.
        # REF: Mikolov et al., 2013b, §2.2
        raise NotImplementedError("TODO 6")

    def forward(self, center: torch.Tensor, context: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        """center, context: (B,), negatives: (B, K) -> 스칼라 loss

        loss = -log σ(u_o·v_c) - Σ_k log σ(-u_k·v_c)
        """
        # TODO 7: 양성 점수 u_o·v_c (B,) 와 음성 점수 u_k·v_c (B, K) 를 구해 위 식의 배치 평균을 반환하세요.
        # HINT 1: 양성은 원소곱 후 마지막 차원 합, 음성은 (B, K, D) 와 (B, D) 의 배치 행렬곱입니다.
        # HINT 2: torch.bmm 에 v 를 (B, D, 1) 로 늘려 넣으면 (B, K, 1). F.logsigmoid
        # WHY: softmax 는 단어마다 어휘 V 개 전부를 계산해야 합니다(O(V)). 음성 샘플링은 "진짜 문맥 vs 무작위 K 개" 이진 분류라 O(K) 입니다.
        #      log(sigmoid(x)) 를 나눠 계산하면 x 가 큰 음수일 때 log(0) = -inf 가 됩니다.
        # REF: Mikolov et al., 2013b, §2.2 (식 4)
        raise NotImplementedError("TODO 7")


class CBOWSoftmax(nn.Module):
    """문맥 단어 벡터의 (패딩 제외) 평균으로 중심 단어를 맞힌다. 작은 어휘라 full softmax 를 쓴다."""

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.pad_id = vocab_size  # 실제 단어 id 는 0..V-1, 패딩은 V
        # TODO 8: self.embed (V+1 행, 마지막 행 = 패딩, padding_idx 지정) 와 bias 없는 출력층 self.out (D -> V) 을 만드세요.
        # HINT 1: 패딩 id 는 self.pad_id (= V) 입니다. 패딩은 예측 대상이 아니므로 출력은 V 개입니다.
        # HINT 2: nn.Embedding(..., padding_idx=...), nn.Linear(..., bias=False)
        # WHY: padding_idx 를 지정하면 패딩 벡터가 0 으로 고정되고 학습되지 않아, 평균에 섞여도 값을 오염시키지 않습니다.
        # REF: Mikolov et al., 2013a, §3.1
        raise NotImplementedError("TODO 8")

    def forward(self, contexts: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """contexts: (B, 2w) (패딩 = pad_id), target: (B,) -> 스칼라 loss"""
        # TODO 9: 패딩을 뺀 문맥 벡터 평균 → 어휘 전체 logits → cross entropy 를 반환하세요.
        # HINT 1: 패딩 위치를 0 으로 만드는 마스크를 곱해 합하고, 실제 문맥 수로 나눕니다(0 이면 1 로).
        # HINT 2: (B, 2w, D) * (B, 2w, 1) → sum(dim=1) → (B, D) → self.out → (B, V), F.cross_entropy
        # WHY: 패딩 수까지 포함해 나누면 문장 끝 단어의 문맥 벡터만 작아져 예측이 흔들립니다.
        # REF: Mikolov et al., 2013a, §3.1
        raise NotImplementedError("TODO 9")


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
    # TODO 10: batch 의 (center, context) 를 각각 (B,) 텐서로 만들고, 음성 샘플 (B, K) 를 뽑아 model 의 loss 를 반환하세요.
    # HINT 1: 텐서는 device 위에 만들고, 음성 샘플은 CPU generator 로 뽑은 뒤 device 로 옮깁니다.
    # HINT 2: torch.tensor([...], device=device), TODO 5 의 sample_negatives, Tensor.to(device)
    # WHY: 음성 샘플을 배치마다 새로 뽑지 않으면 같은 단어들만 계속 밀어내는 편향이 생깁니다.
    raise NotImplementedError("TODO 10")


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
    # TODO 11: (vectors 가 주어지면 그것, 아니면 result["vectors"]) 에서 word 와 코사인 유사도가 높은 순서로
    #          k 개 [(단어, 유사도), ...] 를 반환하세요. 자기 자신은 제외합니다.
    # HINT 1: 모든 벡터를 길이 1 로 정규화하면 내적이 곧 코사인 유사도입니다.
    # HINT 2: F.normalize(..., dim=-1), 행렬-벡터 곱 → (V,), 자기 위치를 -2 로, torch.topk
    # WHY: P1-02 와 같은 코사인 top-k 입니다. 학습 전후 이웃을 비교해 "주변이 비슷한 단어끼리 가까워졌는지" 확인합니다.
    raise NotImplementedError("TODO 11")


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
        print("  (bash)       IMPL=starter pytest part1_ai_fundamentals/word2vec")
        print('  (PowerShell) $env:IMPL="starter"; pytest part1_ai_fundamentals/word2vec; Remove-Item Env:IMPL')
    else:
        print("\n모든 TODO 가 구현되었습니다. 이제 pytest 로 정답 여부를 확인하세요.")


if __name__ == "__main__":
    _report_progress()

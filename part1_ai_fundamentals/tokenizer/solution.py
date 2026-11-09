"""P1-01 Tokenizer · BPE — 정답 코드.

Level 1: 문자 단위 토크나이저 (TODO 1~3)
Level 2: BPE(Byte Pair Encoding) 학습·인코딩·디코딩 직접 구현 (TODO 4~8)
Level 3: HF `tokenizers` BPE 와 비교 → compare_with_hf.py

BPE 는 Sennrich et al. (2016) 방식을 따른다:
단어를 문자 단위로 쪼개고 끝에 단어 경계 표시 "</w>" 를 붙인 뒤,
가장 자주 붙어 나오는 기호 쌍을 하나로 병합하는 일을 num_merges 번 반복한다.
"""
from __future__ import annotations

from collections import Counter

UNK = "<unk>"
END_OF_WORD = "</w>"

Word = tuple[str, ...]  # 기호(symbol)의 튜플. 예: ("l", "o", "w", "</w>")
Pair = tuple[str, str]


# ---------------------------------------------------------------------------
# Level 1 — 문자 단위 토크나이저
# ---------------------------------------------------------------------------
class CharTokenizer:
    """문자 하나를 토큰 하나로 쓰는 가장 단순한 토크나이저."""

    def __init__(self) -> None:
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

    def fit(self, text: str) -> "CharTokenizer":
        # TODO 1
        # id 0 은 <unk> 로 예약: 학습 때 못 본 문자를 만나도 encode 가 실패하지 않게 한다.
        # sorted(set(...)): 실행할 때마다 같은 id 가 나오도록(재현성) 정렬한다. set 순서는 실행마다 다를 수 있다.
        vocab = [UNK] + sorted(set(text))
        self.token_to_id = {tok: i for i, tok in enumerate(vocab)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}
        return self

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str) -> list[int]:
        # TODO 2
        # 모르는 문자는 <unk> id 로. dict.get 의 기본값을 쓰면 KeyError 없이 처리된다.
        unk_id = self.token_to_id[UNK]
        return [self.token_to_id.get(ch, unk_id) for ch in text]

    def decode(self, ids: list[int]) -> str:
        # TODO 3
        # encode 의 역함수. <unk> 는 원래 문자를 복원할 수 없으므로 정보 손실이 생긴다(왜 서브워드가 필요한가의 출발점).
        return "".join(self.id_to_token[i] for i in ids)


# ---------------------------------------------------------------------------
# Level 2 — BPE
# ---------------------------------------------------------------------------
def build_word_vocab(text: str) -> dict[Word, int]:
    """공백으로 단어를 나누고, 각 단어를 '문자 + </w>' 기호 튜플로 바꿔 빈도를 센다."""
    counts = Counter(text.split())
    return {tuple(word) + (END_OF_WORD,): freq for word, freq in counts.items()}


def get_pair_counts(vocab: dict[Word, int]) -> Counter[Pair]:
    """인접한 기호 쌍의 등장 횟수(단어 빈도 가중)를 센다."""
    # TODO 4  REF: Sennrich et al., 2016, "Neural Machine Translation of Rare Words with Subword Units", §3.2
    pairs: Counter[Pair] = Counter()
    for symbols, freq in vocab.items():
        # 단어 안에서만 쌍을 만든다: 단어 경계를 넘는 병합을 막아 "단어 내부" 서브워드만 학습한다.
        for a, b in zip(symbols, symbols[1:]):
            # 단어가 freq 번 등장했으므로 그 안의 쌍도 freq 번 등장한 것으로 센다.
            pairs[(a, b)] += freq
    return pairs


def merge_pair(pair: Pair, vocab: dict[Word, int]) -> dict[Word, int]:
    """vocab 의 모든 단어에서 pair 를 왼쪽부터 겹치지 않게 하나의 기호로 합친다."""
    # TODO 5  REF: Sennrich et al., 2016, §3.2
    a, b = pair
    merged = a + b
    new_vocab: dict[Word, int] = {}
    for symbols, freq in vocab.items():
        out: list[str] = []
        i = 0
        while i < len(symbols):
            # 왼쪽부터 탐색하며 겹치지 않게 병합: ("a","a","a") 에서 ("a","a") 병합 → ("aa","a")
            if i + 1 < len(symbols) and symbols[i] == a and symbols[i + 1] == b:
                out.append(merged)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        # 문자열 replace 로 하면 "a b" 가 "xa b" 안에서도 잘못 매칭될 수 있어 기호 단위로 비교한다.
        new_vocab[tuple(out)] = new_vocab.get(tuple(out), 0) + freq
    return new_vocab


def train_bpe(text: str, num_merges: int) -> list[Pair]:
    """BPE 병합 규칙을 학습해 순서대로 반환한다."""
    vocab = build_word_vocab(text)
    merges: list[Pair] = []
    for _ in range(num_merges):
        # TODO 6  REF: Sennrich et al., 2016, §3.2 (Algorithm 1)
        pairs = get_pair_counts(vocab)
        if not pairs:
            break  # 모든 단어가 기호 하나가 되면 더 병합할 쌍이 없다.
        # 빈도 최대 쌍을 고르되, 동점이면 사전순으로 가장 앞선 쌍을 고른다.
        # 왜: max() 는 동점일 때 "먼저 본 것"을 고르는데, 그 순서는 dict 삽입 순서에 달려 결과가 흔들릴 수 있다.
        best = min(pairs.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        vocab = merge_pair(best, vocab)
        merges.append(best)
    return merges


class BPETokenizer:
    def __init__(self) -> None:
        self.merges: list[Pair] = []
        self.ranks: dict[Pair, int] = {}
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

    def fit(self, text: str, num_merges: int) -> "BPETokenizer":
        self.merges = train_bpe(text, num_merges)
        # rank 가 낮을수록 먼저 학습된(=더 자주 나온) 병합이다.
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        base = sorted({ch for word in text.split() for ch in word} | {END_OF_WORD})
        merged = [a + b for a, b in self.merges]
        vocab = [UNK] + base + [t for t in merged if t not in base]
        # 같은 문자열이 서로 다른 병합으로 두 번 만들어질 수 있어 중복을 제거한다(순서 유지).
        vocab = list(dict.fromkeys(vocab))
        self.token_to_id = {tok: i for i, tok in enumerate(vocab)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}
        return self

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def tokenize_word(self, word: str) -> list[str]:
        """단어 하나를 학습된 병합 규칙으로 서브워드 기호 목록으로 바꾼다."""
        # TODO 7  REF: Sennrich et al., 2016, §3.2
        symbols = list(word) + [END_OF_WORD]
        while len(symbols) > 1:
            # 현재 인접 쌍 중 rank 가 가장 낮은(가장 먼저 학습된) 쌍을 찾는다.
            candidates = [
                (self.ranks[pair], i)
                for i, pair in enumerate(zip(symbols, symbols[1:]))
                if pair in self.ranks
            ]
            if not candidates:
                break
            # 왜 rank 순서: 학습 때 병합이 일어난 순서를 그대로 재현해야 같은 분절 결과가 나온다.
            # 아무 쌍이나 먼저 병합하면 학습 때 없던 분절이 생길 수 있다.
            _, i = min(candidates)
            symbols = symbols[:i] + [symbols[i] + symbols[i + 1]] + symbols[i + 2 :]
        return symbols

    def tokenize(self, text: str) -> list[str]:
        return [tok for word in text.split() for tok in self.tokenize_word(word)]

    def encode(self, text: str) -> list[int]:
        unk_id = self.token_to_id[UNK]
        return [self.token_to_id.get(tok, unk_id) for tok in self.tokenize(text)]

    def decode(self, ids: list[int]) -> str:
        # TODO 8
        # "</w>" 는 단어 끝 표시이므로 공백으로 되돌린다. 이 표시 덕분에 "low" 와 "lowest" 의 "low" 가 구별된다.
        text = "".join(self.id_to_token[i] for i in ids)
        return text.replace(END_OF_WORD, " ").strip()

"""P1-01 Tokenizer · BPE — 실습 코드 (TODO 를 채우세요).

Level 1: 문자 단위 토크나이저 (TODO 1~3)
Level 2: BPE(Byte Pair Encoding) 학습·인코딩·디코딩 직접 구현 (TODO 4~8)
Level 3: HF `tokenizers` BPE 와 비교 → compare_with_hf.py

검사: IMPL=starter pytest part1_ai_fundamentals/tokenizer
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
        # TODO 1: 어휘(vocab)를 만들고 token_to_id / id_to_token 을 채운 뒤 self 를 반환하세요.
        # HINT 1: id 0 은 UNK 로 예약하고, 나머지는 text 에 나온 고유 문자들입니다.
        # HINT 2: sorted(set(text)), enumerate, dict comprehension
        # WHY: 정렬하지 않으면 실행할 때마다 id 가 달라져, 저장한 모델과 토크나이저가 어긋날 수 있습니다.
        raise NotImplementedError("TODO 1")

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str) -> list[int]:
        # TODO 2: 문자열을 id 리스트로 바꾸세요. 모르는 문자는 UNK 의 id 로.
        # HINT 1: 문자 하나씩 순회합니다.
        # HINT 2: dict.get(key, default)
        # WHY: 학습 때 못 본 문자(새 이모지, 희귀 한자 등)는 실제 서비스에서 반드시 나옵니다.
        raise NotImplementedError("TODO 2")

    def decode(self, ids: list[int]) -> str:
        # TODO 3: id 리스트를 문자열로 되돌리세요.
        # HINT 1: encode 의 역방향 사전을 씁니다.
        # HINT 2: "".join(...)
        # WHY: decode(encode(x)) == x 가 성립해야 모델 출력(id)을 사람이 읽을 수 있습니다.
        raise NotImplementedError("TODO 3")


# ---------------------------------------------------------------------------
# Level 2 — BPE
# ---------------------------------------------------------------------------
def build_word_vocab(text: str) -> dict[Word, int]:
    """공백으로 단어를 나누고, 각 단어를 '문자 + </w>' 기호 튜플로 바꿔 빈도를 센다. (제공)"""
    counts = Counter(text.split())
    return {tuple(word) + (END_OF_WORD,): freq for word, freq in counts.items()}


def get_pair_counts(vocab: dict[Word, int]) -> Counter[Pair]:
    """인접한 기호 쌍의 등장 횟수(단어 빈도 가중)를 센다."""
    # TODO 4: 모든 단어의 인접 기호 쌍을 세세요. 단어 빈도만큼 가중합니다.
    # HINT 1: 단어 경계를 넘는 쌍은 만들지 않습니다(단어 하나 안에서만).
    # HINT 2: zip(symbols, symbols[1:]) → (a, b) 쌍,  pairs[(a, b)] += freq
    # WHY: 빈도 가중을 빼면 "한 번 나온 긴 단어"의 쌍이 "천 번 나온 짧은 단어"와 같은 비중이 됩니다.
    # REF: Sennrich et al., 2016, "Neural Machine Translation of Rare Words with Subword Units", §3.2
    raise NotImplementedError("TODO 4")


def merge_pair(pair: Pair, vocab: dict[Word, int]) -> dict[Word, int]:
    """vocab 의 모든 단어에서 pair 를 왼쪽부터 겹치지 않게 하나의 기호로 합친다."""
    # TODO 5: 새 vocab 을 만들어 반환하세요.
    # HINT 1: 인덱스 i 를 직접 움직이며, (symbols[i], symbols[i+1]) == pair 이면 합치고 i 를 2 증가시킵니다.
    # HINT 2: 결과 기호 리스트를 tuple 로 바꿔 key 로 쓰고 빈도는 그대로(또는 합산) 유지합니다.
    # WHY: 문자열 replace 로 처리하면 기호 경계가 무시되어 엉뚱한 위치가 병합될 수 있습니다.
    # REF: Sennrich et al., 2016, §3.2
    raise NotImplementedError("TODO 5")


def train_bpe(text: str, num_merges: int) -> list[Pair]:
    """BPE 병합 규칙을 학습해 순서대로 반환한다."""
    vocab = build_word_vocab(text)
    merges: list[Pair] = []
    for _ in range(num_merges):
        # TODO 6: 가장 빈도가 높은 쌍을 골라 병합하고 merges 에 추가하세요. 쌍이 없으면 중단합니다.
        # HINT 1: 동점이면 사전순으로 가장 앞선 쌍을 고르도록 해서 결과를 결정적으로 만드세요.
        # HINT 2: min(pairs.items(), key=lambda kv: (-kv[1], kv[0]))
        # WHY: 동점 처리를 정하지 않으면 실행 환경에 따라 병합 순서가 달라져 재현이 안 됩니다.
        # REF: Sennrich et al., 2016, §3.2 (Algorithm 1)
        raise NotImplementedError("TODO 6")
    return merges


class BPETokenizer:
    def __init__(self) -> None:
        self.merges: list[Pair] = []
        self.ranks: dict[Pair, int] = {}
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

    def fit(self, text: str, num_merges: int) -> "BPETokenizer":
        """(제공) 병합 규칙 학습 후 어휘를 만든다."""
        self.merges = train_bpe(text, num_merges)
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        base = sorted({ch for word in text.split() for ch in word} | {END_OF_WORD})
        merged = [a + b for a, b in self.merges]
        vocab = list(dict.fromkeys([UNK] + base + [t for t in merged if t not in base]))
        self.token_to_id = {tok: i for i, tok in enumerate(vocab)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}
        return self

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def tokenize_word(self, word: str) -> list[str]:
        """단어 하나를 학습된 병합 규칙으로 서브워드 기호 목록으로 바꾼다."""
        # TODO 7: 문자 + </w> 에서 시작해, 병합 가능한 쌍 중 rank 가 가장 낮은 쌍을 반복해서 병합하세요.
        # HINT 1: 더 이상 self.ranks 에 있는 쌍이 없으면 멈춥니다.
        # HINT 2: symbols = list(word) + [END_OF_WORD];  후보 = [(self.ranks[p], i) ...];  min(후보)
        # WHY: 학습 때의 병합 순서를 그대로 따라야 학습 때와 같은 분절 결과가 나옵니다.
        # REF: Sennrich et al., 2016, §3.2
        raise NotImplementedError("TODO 7")

    def tokenize(self, text: str) -> list[str]:
        return [tok for word in text.split() for tok in self.tokenize_word(word)]

    def encode(self, text: str) -> list[int]:
        unk_id = self.token_to_id[UNK]
        return [self.token_to_id.get(tok, unk_id) for tok in self.tokenize(text)]

    def decode(self, ids: list[int]) -> str:
        # TODO 8: id 를 토큰으로 바꿔 이어 붙이고, END_OF_WORD 를 공백으로 바꾼 뒤 양끝 공백을 제거하세요.
        # HINT 1: 단어 경계 정보는 END_OF_WORD 에만 들어 있습니다.
        # HINT 2: str.replace, str.strip
        # WHY: 경계 표시가 없으면 "new est" 와 "newest" 를 구별해 복원할 수 없습니다.
        raise NotImplementedError("TODO 8")

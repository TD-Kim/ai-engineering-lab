"""P1-01 Tokenizer · BPE — 실습 코드 (TODO 를 채우세요).

Level 1: 문자열 → 토큰(공백/문자 단위) → ID, Vocabulary, 특수 토큰, padding/truncation (TODO 1~5)
Level 2: BPE(Byte Pair Encoding) 학습·인코딩·디코딩 직접 구현 (TODO 6~10)
Level 3: HF `tokenizers` BPE 와 비교 → compare_with_hf.py

진행 확인: python part1_ai_fundamentals/tokenizer/advanced/starter.py   (남은 TODO 목록 출력)
검사:     IMPL=starter pytest part1_ai_fundamentals/tokenizer/advanced
REF: Sennrich et al., 2016, "Neural Machine Translation of Rare Words with Subword Units", §3.2
"""
from __future__ import annotations

from collections import Counter
from itertools import pairwise  # noqa: F401  (TODO 6, 9 에서 사용)

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
SPECIAL_TOKENS = [PAD, UNK, BOS, EOS]  # id 0, 1, 2, 3 로 고정
END_OF_WORD = "</w>"

Word = tuple[str, ...]  # 기호(symbol)의 튜플. 예: ("l", "o", "w", "</w>")
Pair = tuple[str, str]


# ---------------------------------------------------------------------------
# Level 1 — 토큰화, Vocabulary, 특수 토큰, padding/truncation
# ---------------------------------------------------------------------------
def tokenize(text: str, mode: str = "word") -> list[str]:
    """mode="word": 공백 기준 단어 / mode="char": 문자 하나씩 (공백도 하나의 문자)"""
    # TODO 1: mode 에 따라 단어 리스트 또는 문자 리스트를 반환하세요. 그 밖의 mode 는 ValueError.
    # HINT 1: "word" 는 공백 경계로, "char" 는 문자 하나하나(공백 포함)로 나눕니다.
    # HINT 2: str.split() 의 인자 유무에 따른 차이를 확인해 보세요. 문자열은 list() 로 문자 리스트가 됩니다.
    # WHY: split(" ") 은 공백이 두 개 연속이면 빈 문자열 토큰을 만들고, 문자 단위에서 공백을 버리면 원문을 복원할 수 없습니다.
    raise NotImplementedError("TODO 1")


class Vocab:
    """토큰 ↔ ID 사전. 특수 토큰은 항상 앞쪽 고정 ID (pad=0, unk=1, bos=2, eos=3)."""

    def __init__(self) -> None:
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: list[str] = []  # 리스트: id 가 0 부터 빈틈없이 이어지므로 dict 보다 단순

    @classmethod
    def build(cls, tokens: list[str], min_freq: int = 1) -> Vocab:
        # TODO 2: 특수 토큰(SPECIAL_TOKENS)을 앞에, 그 뒤에 min_freq 이상 토큰을 "빈도 내림차순, 동점은 사전순"으로 놓아
        #         vocab.id_to_token / vocab.token_to_id 를 채운 Vocab 을 반환하세요.
        # HINT 1: 입력 tokens 에 특수 토큰 문자열이 섞여 있어도 중복으로 넣지 않아야 합니다.
        # HINT 2: collections.Counter, sorted(..., key=...) 에 튜플 키 (빈도의 음수, 토큰), enumerate
        # WHY: 정렬 기준이 없으면 실행마다 ID 가 달라져 저장한 모델과 어휘가 어긋나고, pad 가 0 이어야 0 으로 채운 텐서가 곧 패딩이 됩니다.
        raise NotImplementedError("TODO 2")

    def __len__(self) -> int:
        return len(self.id_to_token)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK]

    def encode(self, tokens: list[str], add_special: bool = False) -> list[int]:
        # TODO 3: 토큰 리스트를 ID 리스트로 바꾸세요. 모르는 토큰은 unk_id. add_special 이면 앞에 <bos>, 뒤에 <eos> ID 를 붙입니다.
        # HINT 1: 사전 조회에 실패하면 기본값을 쓰는 방법이 있습니다.
        # HINT 2: dict.get(key, default), self.token_to_id[BOS]
        # WHY: 학습 때 못 본 단어·오타는 실제 입력에 반드시 나옵니다. <eos> 는 생성 모델이 멈출 시점을 배우는 신호입니다.
        raise NotImplementedError("TODO 3")

    def decode(self, ids: list[int], skip_special: bool = True) -> list[str]:
        # TODO 4: ID 리스트를 토큰 리스트로 바꾸세요. skip_special 이면 <pad>, <bos>, <eos> 는 빼고 <unk> 는 남깁니다.
        # HINT 1: id_to_token 은 리스트이므로 ID 로 바로 인덱싱할 수 있습니다.
        # HINT 2: 숨길 토큰 집합(set)을 만들고 리스트 컴프리헨션으로 거르세요.
        # WHY: <unk> 를 남겨 두면 "원문 정보가 사라졌다"는 사실이 결과에 드러납니다(서브워드가 필요한 이유).
        raise NotImplementedError("TODO 4")


def pad_and_truncate(
    batch: list[list[int]], max_len: int, pad_id: int = 0
) -> tuple[list[list[int]], list[list[int]]]:
    """길이가 다른 ID 리스트들을 max_len 으로 맞춘다. 반환: (ids (B, max_len), attention_mask (B, max_len))"""
    # TODO 5: 각 리스트를 max_len 으로 자르고(오른쪽), 모자라면 pad_id 로 오른쪽을 채우세요.
    #         attention_mask 는 실제 토큰 1, 패딩 0 입니다.
    # HINT 1: 자르기를 먼저 하고, 남은 길이로 채울 개수를 계산합니다.
    # HINT 2: 슬라이싱 ids[:max_len], 리스트 곱셈 [x] * n
    # WHY: 텐서는 직사각형이어야 배치로 쌓을 수 있고, mask 가 없으면 모델이 패딩을 진짜 글자로 학습합니다.
    raise NotImplementedError("TODO 5")


# ---------------------------------------------------------------------------
# Level 2 — BPE
# ---------------------------------------------------------------------------
def build_word_vocab(text: str) -> dict[Word, int]:
    """공백으로 단어를 나누고, 각 단어를 '문자 + </w>' 기호 튜플로 바꿔 빈도를 센다."""
    counts = Counter(text.split())
    return {tuple(word) + (END_OF_WORD,): freq for word, freq in counts.items()}


def get_pair_counts(vocab: dict[Word, int]) -> Counter[Pair]:
    """인접한 기호 쌍의 등장 횟수(단어 빈도 가중)를 센다."""
    # TODO 6: 모든 단어의 인접 기호 쌍을 세세요. 단어 빈도만큼 가중합니다.
    # HINT 1: 단어 경계를 넘는 쌍은 만들지 않습니다(단어 하나 안에서만).
    # HINT 2: itertools.pairwise(symbols) 로 인접 쌍을 순회하고, Counter 에 freq 를 더합니다.
    # WHY: 빈도 가중을 빼면 "한 번 나온 긴 단어"의 쌍이 "천 번 나온 짧은 단어"와 같은 비중이 됩니다.
    # REF: Sennrich et al., 2016, "Neural Machine Translation of Rare Words with Subword Units", §3.2
    raise NotImplementedError("TODO 6")


def merge_pair(pair: Pair, vocab: dict[Word, int]) -> dict[Word, int]:
    """vocab 의 모든 단어에서 pair 를 왼쪽부터 겹치지 않게 하나의 기호로 합친다."""
    # TODO 7: pair 를 하나의 기호로 합친 새 vocab 을 만들어 반환하세요.
    # HINT 1: 인덱스를 직접 움직이며, 현재와 다음 기호가 pair 와 같으면 합치고 두 칸 건너뜁니다(왼쪽부터, 겹치지 않게).
    # HINT 2: 결과 기호 리스트를 tuple 로 바꿔 dict 키로 쓰고, 빈도는 유지(같은 키가 생기면 합산)합니다.
    # WHY: 문자열 replace 로 처리하면 기호 경계가 무시되어 엉뚱한 위치가 병합될 수 있습니다.
    # REF: Sennrich et al., 2016, §3.2
    raise NotImplementedError("TODO 7")


def train_bpe(text: str, num_merges: int) -> list[Pair]:
    """BPE 병합 규칙을 학습해 순서대로 반환한다."""
    vocab = build_word_vocab(text)  # noqa: F841  (TODO 8 에서 사용)
    merges: list[Pair] = []
    for _ in range(num_merges):
        # TODO 8: 가장 빈도가 높은 쌍을 골라 병합하고 merges 에 추가하세요. 쌍이 없으면 반복을 멈춥니다.
        # HINT 1: 동점이면 사전순으로 가장 앞선 쌍을 골라 결과를 결정적으로 만드세요.
        # HINT 2: min(pairs.items(), key=...) 에 (빈도의 음수, 쌍) 튜플 키를 쓰면 한 번에 고를 수 있습니다.
        # WHY: 동점 처리를 정하지 않으면 dict 삽입 순서(=말뭉치에서 단어가 나온 순서)에 따라 동점 선택이 달라져,
        #      같은 단어 집합이라도 문장 순서만 바꾸면 다른 병합 규칙이 나옵니다.
        # REF: Sennrich et al., 2016, §3.2 (Algorithm 1)
        raise NotImplementedError("TODO 8")
    return merges


class BPETokenizer:
    def __init__(self) -> None:
        self.merges: list[Pair] = []
        self.ranks: dict[Pair, int] = {}
        self.vocab = Vocab()

    def fit(self, text: str, num_merges: int) -> BPETokenizer:
        """(제공) 병합 규칙을 학습하고 어휘(특수 토큰 + 기본 문자 + 병합 결과)를 만든다."""
        self.merges = train_bpe(text, num_merges)
        # rank 가 낮을수록 먼저 학습된(=더 자주 나온) 병합이다.
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        base = sorted({ch for word in text.split() for ch in word} | {END_OF_WORD})
        merged = [a + b for a, b in self.merges]
        tokens = list(dict.fromkeys(base + merged))  # 같은 문자열이 두 번 생길 수 있어 순서 유지 중복 제거
        self.vocab = Vocab()
        self.vocab.id_to_token = SPECIAL_TOKENS + [t for t in tokens if t not in SPECIAL_TOKENS]
        self.vocab.token_to_id = {t: i for i, t in enumerate(self.vocab.id_to_token)}
        return self

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def tokenize_word(self, word: str) -> list[str]:
        """단어 하나를 학습된 병합 규칙으로 서브워드 기호 목록으로 바꾼다."""
        # TODO 9: 문자 + END_OF_WORD 에서 시작해, 인접 쌍 중 self.ranks 에 있고 rank 가 가장 낮은 쌍을 반복해서 병합하세요.
        # HINT 1: 병합할 수 있는 쌍이 하나도 없으면 멈춥니다.
        # HINT 2: (rank, 위치) 튜플 후보 목록을 만들고 min() 으로 고르면 위치까지 함께 얻습니다.
        # WHY: 학습 때의 병합 순서를 그대로 따라야 학습 때와 같은 분절 결과가 나옵니다.
        # REF: Sennrich et al., 2016, §3.2
        raise NotImplementedError("TODO 9")

    def tokenize(self, text: str) -> list[str]:
        return [tok for word in text.split() for tok in self.tokenize_word(word)]

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        return self.vocab.encode(self.tokenize(text), add_special=add_special)

    def decode(self, ids: list[int]) -> str:
        # TODO 10: ID 를 토큰으로 바꿔(특수 토큰 제외) 이어 붙이고, END_OF_WORD 를 공백으로 바꾼 뒤 양끝 공백을 제거하세요.
        # HINT 1: TODO 4 의 Vocab.decode 를 재사용할 수 있습니다.
        # HINT 2: str.join, str.replace, str.strip
        # WHY: 단어 경계 정보는 END_OF_WORD 에만 들어 있습니다. 이 표시가 없으면 "new est" 와 "newest" 를 구별할 수 없습니다.
        raise NotImplementedError("TODO 10")


# ---------------------------------------------------------------------------
# 단독 실행: 어느 TODO 가 남았는지 알려 준다.
# ---------------------------------------------------------------------------
_TOY = "low " * 5 + "lower " * 2 + "newest " * 6 + "widest " * 3


def _report_progress() -> None:
    def vocab_checks():
        return Vocab.build(["a", "b", "a"])

    checks = [
        ("TODO 1  tokenize (Level 1)", lambda: tokenize("a b")),
        ("TODO 2  Vocab.build (Level 1)", vocab_checks),
        ("TODO 3  Vocab.encode (Level 1)", lambda: vocab_checks().encode(["a"])),
        ("TODO 4  Vocab.decode (Level 1)", lambda: vocab_checks().decode([4])),
        ("TODO 5  pad_and_truncate (Level 1)", lambda: pad_and_truncate([[5]], 2)),
        ("TODO 6  get_pair_counts (Level 2)", lambda: get_pair_counts(build_word_vocab(_TOY))),
        ("TODO 7  merge_pair (Level 2)", lambda: merge_pair(("l", "o"), build_word_vocab(_TOY))),
        ("TODO 8  train_bpe (Level 2)", lambda: train_bpe(_TOY, 3)),
        ("TODO 9  BPETokenizer.tokenize_word (Level 2)", lambda: BPETokenizer().fit(_TOY, 3).tokenize_word("low")),
        ("TODO 10 BPETokenizer.decode (Level 2)", lambda: BPETokenizer().fit(_TOY, 3).decode([4])),
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
        print("  (bash)       IMPL=starter pytest part1_ai_fundamentals/tokenizer/advanced")
        print('  (PowerShell) $env:IMPL="starter"; pytest part1_ai_fundamentals/tokenizer/advanced; Remove-Item Env:IMPL')
    else:
        print("\n모든 TODO 가 구현되었습니다. 이제 pytest 로 정답 여부를 확인하세요.")


if __name__ == "__main__":
    _report_progress()

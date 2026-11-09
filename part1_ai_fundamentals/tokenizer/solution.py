"""P1-01 Tokenizer · BPE — 정답 코드.

Level 1: 문자열 → 토큰(공백/문자 단위) → ID, Vocabulary, 특수 토큰, padding/truncation (TODO 1~5)
Level 2: BPE(Byte Pair Encoding) 학습·인코딩·디코딩 직접 구현 (TODO 6~10)
Level 3: HF `tokenizers` BPE 와 비교 → compare_with_hf.py

진행 확인: python part1_ai_fundamentals/tokenizer/solution.py
REF: Sennrich et al., 2016, "Neural Machine Translation of Rare Words with Subword Units", §3.2
"""
from __future__ import annotations

from collections import Counter

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
    # TODO 1
    if mode == "word":
        # str.split() (인자 없음): 연속 공백·탭·줄바꿈을 모두 하나의 경계로 본다. split(" ") 는 빈 문자열 토큰을 만든다.
        return text.split()
    if mode == "char":
        # 문자 단위는 공백도 토큰이다. 공백을 버리면 decode 할 때 단어 경계를 복원할 수 없다.
        return list(text)
    raise ValueError(f"unknown mode: {mode!r}")


class Vocab:
    """토큰 ↔ ID 사전. 특수 토큰은 항상 앞쪽 고정 ID (pad=0, unk=1, bos=2, eos=3)."""

    def __init__(self) -> None:
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: list[str] = []  # 리스트: id 가 0 부터 빈틈없이 이어지므로 dict 보다 단순

    @classmethod
    def build(cls, tokens: list[str], min_freq: int = 1) -> "Vocab":
        # TODO 2
        counts = Counter(tokens)
        # 빈도 내림차순, 동점은 사전순 → 실행할 때마다 같은 ID (재현성).
        # 자주 나오는 토큰이 작은 ID 를 갖는 것은 관례(디버깅 시 읽기 쉬움).
        ordered = sorted(
            (t for t, c in counts.items() if c >= min_freq and t not in SPECIAL_TOKENS),
            key=lambda t: (-counts[t], t),
        )
        vocab = cls()
        # 특수 토큰을 먼저: pad=0 이면 0 으로 채운 텐서가 곧 패딩이 되어 다루기 쉽다.
        vocab.id_to_token = SPECIAL_TOKENS + ordered
        vocab.token_to_id = {t: i for i, t in enumerate(vocab.id_to_token)}
        return vocab

    def __len__(self) -> int:
        return len(self.id_to_token)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK]

    def encode(self, tokens: list[str], add_special: bool = False) -> list[int]:
        # TODO 3
        # 모르는 토큰 → <unk>. KeyError 로 죽으면 실제 서비스 입력(새 단어, 오타)을 처리할 수 없다.
        ids = [self.token_to_id.get(t, self.unk_id) for t in tokens]
        if add_special:
            # <bos>/<eos>: 모델에게 "문장이 여기서 시작/끝난다"를 알려 준다. 생성 모델은 <eos> 를 내면 멈춘다.
            ids = [self.token_to_id[BOS]] + ids + [self.token_to_id[EOS]]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> list[str]:
        # TODO 4
        # skip_special: 사람이 읽을 결과에는 <pad>/<bos>/<eos> 가 필요 없다. <unk> 는 정보 손실을 보여 주려고 남긴다.
        hidden = {PAD, BOS, EOS} if skip_special else set()
        return [tok for tok in (self.id_to_token[i] for i in ids) if tok not in hidden]


def pad_and_truncate(
    batch: list[list[int]], max_len: int, pad_id: int = 0
) -> tuple[list[list[int]], list[list[int]]]:
    """길이가 다른 ID 리스트들을 max_len 으로 맞춘다. 반환: (ids (B, max_len), attention_mask (B, max_len))"""
    # TODO 5
    padded, mask = [], []
    for ids in batch:
        # truncation: 오른쪽을 자른다. 모델의 최대 길이(위치 인코딩 표 크기 등)를 넘으면 처리할 수 없기 때문.
        ids = ids[:max_len]
        n_pad = max_len - len(ids)
        # padding: 배치를 직사각형 (B, max_len) 텐서로 쌓기 위해 짧은 문장 뒤를 pad_id 로 채운다.
        padded.append(ids + [pad_id] * n_pad)
        # attention_mask: 1 = 실제 토큰, 0 = 패딩. 패딩을 어텐션·loss 에서 빼는 데 쓴다(P1-05 transformer).
        mask.append([1] * len(ids) + [0] * n_pad)
    return padded, mask


# ---------------------------------------------------------------------------
# Level 2 — BPE
# ---------------------------------------------------------------------------
def build_word_vocab(text: str) -> dict[Word, int]:
    """공백으로 단어를 나누고, 각 단어를 '문자 + </w>' 기호 튜플로 바꿔 빈도를 센다."""
    counts = Counter(text.split())
    return {tuple(word) + (END_OF_WORD,): freq for word, freq in counts.items()}


def get_pair_counts(vocab: dict[Word, int]) -> Counter[Pair]:
    """인접한 기호 쌍의 등장 횟수(단어 빈도 가중)를 센다."""
    # TODO 6  REF: Sennrich et al., 2016, §3.2
    pairs: Counter[Pair] = Counter()
    for symbols, freq in vocab.items():
        # 단어 안에서만 쌍을 만든다: 단어 경계를 넘는 병합을 막아 "단어 내부" 서브워드만 학습한다.
        for a, b in zip(symbols, symbols[1:]):
            # 단어가 freq 번 등장했으므로 그 안의 쌍도 freq 번 등장한 것으로 센다.
            pairs[(a, b)] += freq
    return pairs


def merge_pair(pair: Pair, vocab: dict[Word, int]) -> dict[Word, int]:
    """vocab 의 모든 단어에서 pair 를 왼쪽부터 겹치지 않게 하나의 기호로 합친다."""
    # TODO 7  REF: Sennrich et al., 2016, §3.2
    a, b = pair
    merged = a + b
    new_vocab: dict[Word, int] = {}
    for symbols, freq in vocab.items():
        out: list[str] = []
        i = 0
        while i < len(symbols):
            # 왼쪽부터 겹치지 않게 병합: ("a","a","a") 에서 ("a","a") 병합 → ("aa","a")
            if i + 1 < len(symbols) and symbols[i] == a and symbols[i + 1] == b:
                out.append(merged)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        # 문자열 replace 로 하면 기호 경계가 무시되어 엉뚱한 위치가 합쳐질 수 있어 기호 단위로 비교한다.
        new_vocab[tuple(out)] = new_vocab.get(tuple(out), 0) + freq
    return new_vocab


def train_bpe(text: str, num_merges: int) -> list[Pair]:
    """BPE 병합 규칙을 학습해 순서대로 반환한다."""
    vocab = build_word_vocab(text)
    merges: list[Pair] = []
    for _ in range(num_merges):
        # TODO 8  REF: Sennrich et al., 2016, §3.2 (Algorithm 1)
        pairs = get_pair_counts(vocab)
        if not pairs:
            break  # 모든 단어가 기호 하나가 되면 더 병합할 쌍이 없다.
        # 빈도 최대 쌍, 동점이면 사전순으로 가장 앞선 쌍.
        # 왜: max() 는 동점일 때 "먼저 본 것"을 고르는데, 그 순서는 dict 삽입 순서에 달려 결과가 흔들릴 수 있다.
        best = min(pairs.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        vocab = merge_pair(best, vocab)
        merges.append(best)
    return merges


class BPETokenizer:
    def __init__(self) -> None:
        self.merges: list[Pair] = []
        self.ranks: dict[Pair, int] = {}
        self.vocab = Vocab()

    def fit(self, text: str, num_merges: int) -> "BPETokenizer":
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
        # TODO 9  REF: Sennrich et al., 2016, §3.2
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
            _, i = min(candidates)
            symbols = symbols[:i] + [symbols[i] + symbols[i + 1]] + symbols[i + 2 :]
        return symbols

    def tokenize(self, text: str) -> list[str]:
        return [tok for word in text.split() for tok in self.tokenize_word(word)]

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        return self.vocab.encode(self.tokenize(text), add_special=add_special)

    def decode(self, ids: list[int]) -> str:
        # TODO 10
        # 특수 토큰을 뺀 서브워드를 이어 붙이고 "</w>" 를 공백으로 되돌린다.
        # 이 표시 덕분에 "low"+"est</w>" (한 단어) 와 "low</w>"+"est</w>" (두 단어) 가 구별된다.
        text = "".join(self.vocab.decode(ids, skip_special=True))
        return text.replace(END_OF_WORD, " ").strip()


# ---------------------------------------------------------------------------
# 단독 실행: 어느 TODO 가 남았는지 알려 준다.
# ---------------------------------------------------------------------------
_TOY = "low " * 5 + "lower " * 2 + "newest " * 6 + "widest " * 3


def _report_progress() -> None:
    vocab_checks = lambda: Vocab.build(["a", "b", "a"])  # noqa: E731
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
        print("  (bash)       IMPL=starter pytest part1_ai_fundamentals/tokenizer")
        print('  (PowerShell) $env:IMPL="starter"; pytest part1_ai_fundamentals/tokenizer; Remove-Item Env:IMPL')
    else:
        print("\n모든 TODO 가 구현되었습니다. 이제 pytest 로 정답 여부를 확인하세요.")


if __name__ == "__main__":
    _report_progress()

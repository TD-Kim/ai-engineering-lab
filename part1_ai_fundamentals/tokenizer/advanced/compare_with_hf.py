"""P1-01 Level 3: 직접 구현한 BPE 와 Hugging Face `tokenizers` 의 BPE 를 같은 말뭉치에서 비교한다.

사용법:
    python part1_ai_fundamentals/tokenizer/advanced/compare_with_hf.py --impl solution --merges 200

비교 항목
- 처음 N 개 병합 규칙의 겹침 비율 (같은 알고리즘이면 대부분 겹친다)
- 말뭉치 전체를 인코딩했을 때의 토큰 수 비율 (압축률)
- 같은 문장의 분절 결과
- 디코딩 일치 여부: decode(encode(문장)) 가 원문과 같은가

`tokenizers` 가 설치되어 있지 않으면 "건너뜀" 메시지를 출력하고 종료한다(테스트는 skip).

완전히 같지 않은 이유: 동점 쌍 처리 순서, 최소 빈도 기준, 어휘 크기 계산 방식이 구현마다 다르다.
REF: https://huggingface.co/docs/tokenizers/api/models#tokenizers.models.BPE
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

CORPUS = Path(__file__).resolve().parents[2] / "data" / "corpus_ko_en.txt"


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]  # 직접 구현과 같은 순서·같은 개수


def hf_available() -> bool:
    try:
        import tokenizers  # noqa: F401
    except ImportError:
        return False
    return True


def train_hf_bpe(text: str, vocab_size: int):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    # 직접 구현과 조건을 맞춘다: 공백 분리만 하고(구두점 분리 없음), 단어 끝 표시 "</w>" 사용
    tok = Tokenizer(models.BPE(unk_token="<unk>", end_of_word_suffix="</w>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=1,
        special_tokens=SPECIAL_TOKENS,
        end_of_word_suffix="</w>",
        show_progress=False,
    )
    tok.train_from_iterator([text], trainer=trainer)
    # 디코더를 지정하지 않으면 HF 는 토큰 사이에 공백을 넣어 이어 붙인다 → "</w>" 를 단어 경계로 되돌리도록 설정
    tok.decoder = decoders.BPEDecoder(suffix="</w>")
    return tok


def hf_merges(tok) -> list[tuple[str, str]]:
    import json

    raw = json.loads(tok.to_str())["model"]["merges"]
    # 버전에 따라 merges 가 "a b" 문자열 또는 ["a", "b"] 리스트로 저장된다.
    return [tuple(m.split(" ", 1)) if isinstance(m, str) else tuple(m) for m in raw]


def compare(impl, text: str, num_merges: int = 200, sample: str | None = None) -> dict:
    mine = impl.BPETokenizer().fit(text, num_merges=num_merges)
    hf = train_hf_bpe(text, vocab_size=mine.vocab_size)

    mine_merges = mine.merges
    theirs = hf_merges(hf)
    n = min(50, len(mine_merges), len(theirs))
    overlap = len(set(mine_merges[:n]) & set(theirs[:n])) / max(n, 1)

    mine_count = len(mine.encode(text))
    hf_count = len(hf.encode(text).tokens)

    sample = sample or "작은 모델이라도 직접 학습해 보면 원리가 잘 보인다. the newest model"
    hf_ids = hf.encode(sample).ids
    return {
        "my_vocab_size": mine.vocab_size,
        "hf_vocab_size": hf.get_vocab_size(),
        "first_merges_overlap": overlap,
        "my_token_count": mine_count,
        "hf_token_count": hf_count,
        "token_count_ratio": mine_count / hf_count,
        "my_tokens": mine.tokenize(sample),
        "hf_tokens": hf.encode(sample).tokens,
        "sample": sample,
        "my_decode_ok": mine.decode(mine.encode(sample)) == sample,
        "hf_decode_ok": hf.decode(hf_ids) == sample,
        "hf_decoded": hf.decode(hf_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=["solution", "starter"], default="solution")
    parser.add_argument("--merges", type=int, default=200)
    args = parser.parse_args()
    if not hf_available():
        print("[건너뜀] `tokenizers` 패키지가 없어 HF 비교를 건너뜁니다. 설치: uv pip install -r requirements/part1.txt")
        return
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    impl = importlib.import_module(args.impl)

    try:
        result = compare(impl, CORPUS.read_text(encoding="utf-8"), num_merges=args.merges)
    except NotImplementedError as exc:
        print(f"[미구현] {exc} — {args.impl}.py 의 TODO 를 먼저 채우세요 (python part1_ai_fundamentals/tokenizer/advanced/{args.impl}.py 로 확인)")
        return
    for key, value in result.items():
        print(f"{key:>22}: {value}")


if __name__ == "__main__":
    main()

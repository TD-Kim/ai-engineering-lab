"""(선택) 공개 모델의 토크나이저로 토큰 수 비교 — 인터넷이 되는 사용자 PC 에서 실행.

Qwen2.5-0.5B(Apache-2.0)의 tokenizer.json 을 Hugging Face Hub 에서 내려받아(모델 가중치는 받지 않음)
로컬 BPE(어휘 500 개)와 같은 문장의 토큰 수를 비교한다. 어휘가 크고 다국어 데이터로 학습된 토크나이저일수록 한국어 토큰 수가 준다.

실행: python part1_ai_fundamentals/tokenizer/pretrained_tokenizer.py
REF: https://huggingface.co/Qwen/Qwen2.5-0.5B  (모델 카드·라이선스)
"""
from __future__ import annotations

import sys
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB_DIR))

from solution import compare_languages, train_local_tokenizer
from tokenizers import Tokenizer

PRETRAINED = "Qwen/Qwen2.5-0.5B"
CORPUS = LAB_DIR.parents[0] / "data" / "corpus_ko_en.txt"
SAMPLES = {"ko": "오늘은 날씨가 맑다", "en": "The weather is clear today"}


def main() -> None:
    local = train_local_tokenizer(CORPUS.read_text(encoding="utf-8"))
    try:
        hub = Tokenizer.from_pretrained(PRETRAINED)
    except Exception as e:  # noqa: BLE001  오프라인·차단 환경(네트워크 오류 종류가 다양함)
        print(f"건너뜀: {PRETRAINED} 토크나이저를 받을 수 없습니다 ({type(e).__name__}).")
        return
    for label, tok in [(f"local BPE (vocab {local.get_vocab_size()})", local), (f"{PRETRAINED} (vocab {hub.get_vocab_size()})", hub)]:
        print(label)
        for row in compare_languages(tok, SAMPLES):
            print(f"  {row['name']:3s} chars={row['chars']:3d} tokens={row['tokens']:3d} chars/token={row['chars_per_token']}")


if __name__ == "__main__":
    main()

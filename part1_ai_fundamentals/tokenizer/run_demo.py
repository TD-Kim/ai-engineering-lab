"""P1-01 데모: 로컬 BPE 로 토큰 수·언어 비교·비용 추정·청크·자르기를 한 번에 본다(다운로드·API 키 불필요).

실행: python part1_ai_fundamentals/tokenizer/run_demo.py [--impl starter]
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent
CORPUS = LAB_DIR.parents[0] / "data" / "corpus_ko_en.txt"
SAMPLES = {
    "ko": "오늘은 날씨가 맑다",
    "en": "The weather is clear today",
    "emoji": "이모지 😀",
}


def load(impl: str):
    spec = importlib.util.spec_from_file_location(f"p1_01_{impl}", LAB_DIR / f"{impl}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["solution", "starter"], default="solution")
    m = load(ap.parse_args().impl)

    text = CORPUS.read_text(encoding="utf-8")
    tok = m.train_local_tokenizer(text, vocab_size=500)
    n_tokens = m.count_tokens(tok, text)
    print(f"[1] 말뭉치: chars={len(text)}, tokens={n_tokens} (local BPE, vocab {tok.get_vocab_size()})")

    print("[2] 언어별 비교")
    for row in m.compare_languages(tok, SAMPLES):
        print(f"    {row['name']:5s} chars={row['chars']:3d} tokens={row['tokens']:3d} chars/token={row['chars_per_token']}")

    print("[3] 비용 추정: 입력 = 말뭉치 전체, 출력 300 토큰, 요청 1,000 건 (로컬 토큰 수 기준 근사, 단가 작성 시점 2026-10)")
    for model in m.PRICES_PER_MTOK:
        print(f"    {model:18s} ${m.estimate_cost(n_tokens, 300, model, requests=1000):8.2f}")

    print("[4] 청크 (max_tokens=200)")
    for overlap in (0, 50):
        chunks = m.chunk_by_tokens(tok, text, max_tokens=200, overlap=overlap)
        total = sum(m.count_tokens(tok, c) for c in chunks)
        print(f"    overlap={overlap:3d}: {len(chunks)} 조각, 조각 토큰 합 {total} (원문 {n_tokens}, {total / n_tokens:.2f}배)")
    first = m.chunk_by_tokens(tok, text, max_tokens=200, overlap=50)
    print(f"    조각 0 끝: {first[0][-25:]!r}")
    print(f"    조각 1 앞: {first[1][:25]!r}")

    print("[5] 컨텍스트 창 1,000 토큰, 예약 300 토큰")
    fitted = m.fit_to_context(tok, text, context_window=1000, reserved_tokens=300)
    print(f"    남긴 글자 {len(fitted)}/{len(text)}, 토큰 {m.count_tokens(tok, fitted)}, 끝: {fitted[-20:]!r}")


if __name__ == "__main__":
    main()

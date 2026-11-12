"""P1-03 데모: 전처리 → gensim Word2Vec 학습 → 학습 전후 유사도 → 최근접 단어 → (선택) 문장 벡터 비교.

실행: python part1_ai_fundamentals/word2vec/run_demo.py [--impl starter]
다운로드·API 키 불필요, CPU 수 초.
"""
from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np

LAB_DIR = Path(__file__).resolve().parent
PAIRS = [("cat", "dog"), ("mat", "log"), ("cat", "coffee"), ("mat", "coffee"), ("cat", "강아지")]  # 3·4번째는 대조군, 5번째는 OOV
PROBES = ["cat", "손실", "강아지"]
QUERIES = ["a dog and a cat", "dog cat", "강아지와 고양이"]  # 2번째: 흔한 단어(a, and)를 뺀 질의


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cosine_rank(q: np.ndarray, docs: np.ndarray, k: int = 3) -> list[tuple[int, float]]:
    qn = q / max(float(np.linalg.norm(q)), 1e-12)
    dn = docs / np.maximum(np.linalg.norm(docs, axis=1, keepdims=True), 1e-12)
    scores = dn @ qn
    return [(int(i), float(scores[i])) for i in np.argsort(-scores, kind="stable")[:k]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["solution", "starter"], default="solution")
    impl = ap.parse_args().impl
    m = load(LAB_DIR / f"{impl}.py", f"p1_03_{impl}")

    lines = [ln for ln in m.load_lines() if ln.strip()]
    sents = m.preprocess(lines)
    vocab = {t for s in sents for t in s}
    print(f"[1] 전처리: 문장 {len(sents)}개, 토큰 {sum(map(len, sents))}개, 고유 단어 {len(vocab)}개")
    print(f"    예) {lines[44]!r} → {sents[44]}")
    print(f"    예) {lines[5]!r} → {sents[5]}")

    print("[2] 학습 (vector_size=32, window=2, negative=5, epochs=200, seed=0, workers=1)")
    models = {}
    for name, sg in (("skip-gram", 1), ("cbow", 0)):
        t = time.perf_counter()
        models[name] = m.train_word2vec(sents, sg=sg)
        print(f"    {name:9s} {time.perf_counter() - t:.2f}초")
    before = m.train_word2vec(sents, sg=1, epochs=0)

    print("[3] 단어 쌍 코사인: 학습 전(무작위 초기값) → 학습 후 skip-gram / cbow")
    rows = zip(m.pair_similarities(before, PAIRS), m.pair_similarities(models["skip-gram"], PAIRS),
               m.pair_similarities(models["cbow"], PAIRS), strict=True)
    for b, s, c in rows:
        fmt = (lambda v: "  OOV" if v is None else f"{v:5.2f}")
        print(f"    {b['pair'][0]:>4s}–{b['pair'][1]:<7s} {fmt(b['similarity'])} → {fmt(s['similarity'])} / {fmt(c['similarity'])}")

    print("[4] 최근접 단어 top-5 (skip-gram)")
    for w in PROBES:
        nn = m.nearest(models["skip-gram"], w, 5)
        print(f"    {w}: " + (", ".join(f"{t}({v:.2f})" for t, v in nn) if nn else "[] (OOV — 학습 데이터에 없는 단어)"))

    print("[5] (선택) 문장 검색: Word2Vec 평균 vs P1-02 가짜 임베딩(fake_embed, 글자 2-gram)")
    emb = load(LAB_DIR.parents[0] / "embedding" / "solution.py", "p1_02_solution_for_p1_03_demo")
    w2v_docs = np.stack([m.sentence_vector(models["skip-gram"], s) for s in sents])
    fake_docs = np.asarray(emb.fake_embed(lines), dtype=np.float32)
    for q in QUERIES:
        qt = m.preprocess([q])[0]
        known = [t for t in qt if t in models["skip-gram"].wv]
        qv = m.sentence_vector(models["skip-gram"], qt)
        print(f"    질의: {q}  (토큰 {qt}, 어휘에 있는 것 {known})")
        w2v = cosine_rank(qv, w2v_docs) if known else []
        print("      word2vec 평균: " + (", ".join(f"#{i + 1:02d}({v:.2f})" for i, v in w2v) if w2v else "검색 불가 (모든 토큰이 OOV → 0 벡터)"))
        fq = np.asarray(emb.fake_embed([q])[0], dtype=np.float32)
        print("      fake_embed   : " + ", ".join(f"#{i + 1:02d}({v:.2f})" for i, v in cosine_rank(fq, fake_docs)))
    print(f"    참고 #45: {lines[44]}")
    print(f"    참고 #46: {lines[45]}")


if __name__ == "__main__":
    main()

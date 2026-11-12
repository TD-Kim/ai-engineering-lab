"""P1-02 데모: 가짜 임베딩(fake_embed)으로 정규화·코사인·배치 임베딩·top-k 검색·캐시를 한 번에 본다(API 키·다운로드 불필요).

실행: python part1_ai_fundamentals/embedding/run_demo.py [--impl starter]
실제 OpenAI 임베딩으로 같은 검색을 해 보려면: embed_with_openai.py (선택, API 키 필요)
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np

LAB_DIR = Path(__file__).resolve().parent
CORPUS = LAB_DIR.parents[0] / "data" / "corpus_ko_en.txt"
QUERIES = [
    "손실 값이 줄지 않을 때는 무엇을 확인하나",       # 문서와 글자가 많이 겹침
    "what to check if the loss does not go down",  # 영어 문서와 단어가 겹침
    "강아지와 고양이",                               # 뜻은 45·46번 줄(cat, dog)과 가깝지만 글자가 하나도 안 겹침
]


def load(impl: str):
    spec = importlib.util.spec_from_file_location(f"p1_02_{impl}", LAB_DIR / f"{impl}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["solution", "starter"], default="solution")
    m = load(ap.parse_args().impl)

    print("[1] L2 정규화와 코사인")
    v = np.array([[3.0, 4.0], [6.0, 8.0], [-3.0, -4.0], [4.0, -3.0]])
    def r(a):  # float32 를 보기 좋게: 소수 셋째 자리
        return [round(float(x), 3) for x in np.ravel(a)]

    print(f"    normalize([3,4]) = {r(m.l2_normalize(v[0]))}, normalize([6,8]) = {r(m.l2_normalize(v[1]))}")
    print(f"    cos([3,4], [6,8] / [-3,-4] / [4,-3]) = {r(m.cosine_similarity(v[:1], v[1:]))}")

    docs = [line for line in CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()]
    calls = {"n": 0, "texts": 0}

    def counted_embed(texts):
        calls["n"] += 1
        calls["texts"] += len(texts)
        return m.fake_embed(texts)

    doc_vecs = m.embed_texts(docs, counted_embed, batch_size=16)
    print(f"[2] 문서 임베딩: {doc_vecs.shape[0]}개 × {doc_vecs.shape[1]}차원 (fake_embed), embed_fn 호출 {calls['n']}번(batch_size=16)")
    print(f"    float32 저장 크기: {doc_vecs.nbytes:,} 바이트 = 문서 수 × 차원 × 4")

    print("[3] top-3 검색 (가짜 임베딩 = 글자 2-gram 겹침)")
    for q in QUERIES:
        qv = m.embed_texts([q], m.fake_embed)[0]
        print(f"    질의: {q}")
        for idx, score in m.top_k(qv, doc_vecs, 3):
            print(f"      {score:.3f}  #{idx + 1:02d} {docs[idx][:40]}")

    print("[4] 캐시: 같은 질의 목록을 두 번 임베딩")
    cache: dict[str, list[float]] = {}
    calls.update(n=0, texts=0)
    batch = [QUERIES[0], QUERIES[1], QUERIES[0]]
    m.embed_with_cache(batch, counted_embed, cache)
    first = calls["texts"]
    m.embed_with_cache(batch, counted_embed, cache)
    print(f"    입력 {len(batch)}개(중복 1) → 1회차 임베딩한 텍스트 {first}개, 2회차 추가 {calls['texts'] - first}개, 캐시 크기 {len(cache)}")


if __name__ == "__main__":
    main()

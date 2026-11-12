"""(선택) OpenAI 임베딩 API 로 같은 검색 해 보기 — 사용자 PC 에서 API 키가 있을 때 실행.

가짜 임베딩(fake_embed)은 글자 겹침만 보지만, 학습된 임베딩 모델은 뜻이 비슷하면 글자가 달라도 가깝게 만든다.
run_demo.py 와 같은 문서·질의로 결과를 비교해 보자(특히 '강아지와 고양이' 질의, 한국어 질의 → 영어 문서).

준비: pip install -r requirements/part1.txt
      cp .env.example .env (PowerShell: Copy-Item .env.example .env) 후 OPENAI_API_KEY 입력 (커밋 금지)
실행: python part1_ai_fundamentals/embedding/embed_with_openai.py
모델 변경: .env 또는 환경 변수 EMBEDDING_MODEL (기본 text-embedding-3-small)
비용: 응답의 usage.prompt_tokens 를 출력한다. 단가는 공식 가격 페이지에서 확인 https://platform.openai.com/docs/pricing (작성 환경에서 접속 불가 → 미확인)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB_DIR))

from run_demo import CORPUS, QUERIES
from solution import DEFAULT_EMBEDDING_MODEL, embed_texts, make_openai_embed_fn, top_k


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(LAB_DIR.parents[1] / ".env")
    except ImportError:
        pass
    if not os.environ.get("OPENAI_API_KEY"):
        print("건너뜀: OPENAI_API_KEY 가 없습니다(.env 에 넣고 다시 실행).")
        return
    import openai

    model = os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    client = openai.OpenAI(timeout=30.0, max_retries=2)
    usage: dict = {}
    embed_fn = make_openai_embed_fn(client, model=model, usage=usage)

    docs = [line for line in CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()]
    doc_vecs = embed_texts(docs, embed_fn, batch_size=64)
    print(f"model={model}  문서 {doc_vecs.shape[0]}개 × {doc_vecs.shape[1]}차원")
    for q in QUERIES:
        qv = embed_texts([q], embed_fn)[0]
        print(f"질의: {q}")
        for idx, score in top_k(qv, doc_vecs, 3):
            print(f"  {score:.3f}  #{idx + 1:02d} {docs[idx][:40]}")
    print(f"usage: 요청 {usage.get('requests', 0)}번, prompt_tokens {usage.get('prompt_tokens', 0)}")


if __name__ == "__main__":
    main()

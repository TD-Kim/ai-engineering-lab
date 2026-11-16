"""(선택) EMBEDDING_PROVIDER 로 고른 임베딩으로 같은 검색 해 보기 — 사용자 PC 용.

run_demo.py 와 같은 문서·질의를 쓴다. 공급자만 바꿔 가며 top-3 결과·차원·시간을 비교해 보자.
  fake                  : 키·다운로드 불필요(글자 겹침만 봄)
  sentence_transformers : 로컬 모델(처음 실행 때 모델을 내려받음, 수백 MB~수 GB). pip install sentence-transformers
  ollama                : 로컬 Ollama 서버(`ollama serve`) + 임베딩 모델(`ollama pull <태그>`, 태그는 ollama.com 라이브러리에서 확인)
  openai                : OPENAI_API_KEY 필요(유료)

설정: .env 의 EMBEDDING_PROVIDER, EMBEDDING_MODEL(공급자에 맞는 이름), OLLAMA_BASE_URL
      예) EMBEDDING_PROVIDER=sentence_transformers, EMBEDDING_MODEL=intfloat/multilingual-e5-small (또는 BAAI/bge-m3)
실행: python part1_ai_fundamentals/embedding/embed_with_provider.py [--provider sentence_transformers]
결과(점수·시간)는 이 PC 에서 실제로 나온 값만 experiments 기록에 옮겨 적는다.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent


def _load_sibling(filename: str):
    """같은 폴더의 파일을 "실습별 고유 이름"(lab_<상위 폴더>_<실습 폴더>_<파일>)으로 불러온다. sys.path 는 건드리지 않는다."""
    name = f"lab_{LAB_DIR.parent.name}_{LAB_DIR.name}_{Path(filename).stem}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, LAB_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclass 가 자기 모듈을 찾을 수 있게 먼저 등록
    spec.loader.exec_module(mod)
    return mod


_demo = _load_sibling("run_demo.py")
CORPUS = _demo.CORPUS
QUERIES = _demo.QUERIES
_sol = _load_sibling("solution.py")
make_embed_fn = _sol.make_embed_fn
embed_texts = _sol.embed_texts
top_k = _sol.top_k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=_sol.PROVIDERS, help="생략하면 .env 의 EMBEDDING_PROVIDER(기본 fake)")
    args = ap.parse_args()
    try:
        from dotenv import load_dotenv

        load_dotenv(LAB_DIR.parents[1] / ".env")
    except ImportError:
        pass
    provider = args.provider or os.environ.get("EMBEDDING_PROVIDER", "fake")
    try:
        doc_fn = make_embed_fn(provider, role="passage")     # 문서와 질문은 같은 모델, 역할(접두어)만 다르다
        query_fn = make_embed_fn(provider, role="query")
    except ValueError as e:
        print(f"건너뜀: {e}")
        return
    except ImportError as e:
        print(f"건너뜀: {e.name or e} 패키지가 없습니다. 이 공급자를 쓰려면 설치하세요(requirements/part1.txt 의 선택 의존성 주석).")
        return

    docs = [line for line in CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()]
    try:
        start = time.perf_counter()
        doc_vecs = embed_texts(docs, doc_fn, batch_size=32)
        index_s = time.perf_counter() - start
        print(f"provider={provider} model={os.environ.get('EMBEDDING_MODEL', '(기본값)')}  "
              f"문서 {doc_vecs.shape[0]}개 × {doc_vecs.shape[1]}차원, 문서 임베딩 {index_s:.2f}초(처음 실행은 모델 적재 포함)")
        for q in QUERIES:
            start = time.perf_counter()
            qv = embed_texts([q], query_fn)[0]
            q_ms = (time.perf_counter() - start) * 1000
            print(f"질의: {q}  ({q_ms:.0f} ms)")
            for idx, score in top_k(qv, doc_vecs, 3):
                print(f"  {score:.3f}  #{idx + 1:02d} {docs[idx][:40]}")
    except ConnectionError as e:
        print(f"[오류] {e}")


if __name__ == "__main__":
    main()

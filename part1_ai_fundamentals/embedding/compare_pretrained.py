"""P1-02 Level 3 (선택): 사전학습 문장 임베딩과 "토큰 벡터 평균" 문장 벡터를 비교한다.

사용법:
    python part1_ai_fundamentals/embedding/compare_pretrained.py --impl solution
    (모델 지정) $env:LOCAL_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"   # PowerShell

- `sentence-transformers` 가 설치되어 있고 모델 파일이 로컬(HF 캐시 또는 경로)에 있을 때만 실행한다.
  다운로드는 하지 않는다(local_files_only). 조건이 안 맞으면 "건너뜀" 을 출력하고 끝낸다.
- 접두어: multilingual-e5 계열은 문장 유사도 같은 대칭 작업에 양쪽 모두 "query: " 접두어를 쓰도록 모델 카드가 권장한다
  (작성 환경에서 huggingface.co 접속이 막혀 원문은 재확인하지 못함 — 사용 전 모델 카드 확인). 모델 이름에 "e5" 가 있으면
  기본으로 "query: " 를 붙이고, --prefix 로 바꿀 수 있다(--prefix "" 이면 접두어 없이 측정).
- 비교: 의미가 비슷한 문장 쌍 / 다른 문장 쌍의 코사인 유사도를
  (a) 사전학습 모델, (b) 무작위 EmbeddingLayer + sentence_vector(평균) 로 각각 계산한다.
  무작위 임베딩은 의미를 모르므로 (a) 와 같은 경향이 나오지 않는 것이 정상이다.

모델 라이선스는 사용 전 모델 카드에서 확인할 것 (docs/ENVIRONMENT.md 4절).
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import torch

LAB_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "intfloat/multilingual-e5-small"
PAIRS = [
    ("학습률이 너무 크면 손실이 줄지 않는다.", "If the learning rate is too high, the loss does not go down."),
    ("학습률이 너무 크면 손실이 줄지 않는다.", "주말에는 공원에서 산책을 한다."),
    ("the cat sat on the mat", "the dog sat on the log"),
    ("the cat sat on the mat", "a fair experiment uses the same data"),
]


def load_model(name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None, "sentence-transformers 미설치"
    try:
        return SentenceTransformer(name, local_files_only=True), None
    except Exception as exc:  # noqa: BLE001  로컬에 모델이 없거나 버전이 맞지 않음 → 어떤 이유든 건너뛴다
        return None, f"로컬 모델을 불러올 수 없음 ({type(exc).__name__})"


def default_prefix(name: str) -> str:
    # e5 계열: 대칭 작업(문장 유사도)은 양쪽 모두 "query: " (모델 카드 권장, 원문 재확인 필요)
    return "query: " if "e5" in name.lower() else ""


def compare(impl, model, prefix: str = "") -> list[dict]:
    texts = sorted({t for pair in PAIRS for t in pair})
    vocab = {w: i for i, w in enumerate(sorted({w for t in texts for w in t.split()}))}
    torch.manual_seed(0)
    emb = impl.EmbeddingLayer(len(vocab), 32)
    rows = []
    for a, b in PAIRS:
        # 접두어는 사전학습 모델 입력에만 붙인다(무작위 임베딩 평균은 같은 단어 집합으로 비교).
        pre = torch.tensor(model.encode([prefix + a, prefix + b], normalize_embeddings=True))
        ids = [torch.tensor([vocab[w] for w in t.split()]) for t in (a, b)]
        vecs = torch.cat([impl.sentence_vector(i.view(1, -1), emb.weight.detach()) for i in ids])
        rows.append({
            "a": a, "b": b,
            "pretrained_cos": float(impl.cosine_similarity(pre[:1], pre[1:])),
            "random_mean_cos": float(impl.cosine_similarity(vecs[:1], vecs[1:])),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=["solution", "starter"], default="solution")
    parser.add_argument("--prefix", default=None, help='사전학습 모델 입력 접두어. 생략 시 e5 계열은 "query: "')
    args = parser.parse_args()
    name = os.environ.get("LOCAL_EMBEDDING_MODEL", DEFAULT_MODEL)
    prefix = default_prefix(name) if args.prefix is None else args.prefix
    model, reason = load_model(name)
    if model is None:
        print(f"[건너뜀] 사전학습 임베딩 비교: {reason}. 모델: {name}")
        return
    sys.path.insert(0, str(LAB_DIR))
    impl = importlib.import_module(args.impl)
    print(f"[설정] 모델: {name}, 접두어: {prefix!r}")
    for row in compare(impl, model, prefix):
        print(f"{row['pretrained_cos']:+.3f} (pretrained) {row['random_mean_cos']:+.3f} (random mean) | {row['a']} || {row['b']}")


if __name__ == "__main__":
    main()

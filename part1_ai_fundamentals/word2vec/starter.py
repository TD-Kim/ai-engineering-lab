"""P1-03 Word2Vec — 실습 코드 (TODO 를 채우세요). 메인 실습: 짧게 — 개념 + gensim 짧은 실행.

문장을 토큰 목록으로 바꾸고(TODO 1), gensim 으로 Word2Vec 을 학습하고(TODO 2),
학습 전후 단어 쌍 유사도와 최근접 단어를 비교한다(TODO 3). (선택) 단어 벡터 평균으로 문장 벡터를 만들어
P1-02 문장 임베딩과의 차이를 본다(TODO 4).

검사: IMPL=starter pytest part1_ai_fundamentals/word2vec/tests
진행 확인: python part1_ai_fundamentals/word2vec/starter.py   (남은 TODO 목록 출력)
원리(Skip-gram·CBOW·Negative Sampling 직접 구현)는 선택 과제: part1_ai_fundamentals/word2vec/advanced/

# 개념(분포 가설): "같은 주변 단어와 함께 나오는 단어는 뜻이 비슷하다."
#   the cat sat on the mat / the dog sat on the log → cat 과 dog 은 주변(the, sat, on)이 같아 벡터가 가까워진다.
#   Word2Vec 은 "주변 단어 맞히기" 문제를 풀면서 그 부산물로 단어 벡터를 얻는다(정답 라벨이 필요 없다).
# 개념(Skip-gram vs CBOW): 창(window)=2, 가운데 단어가 sat 일 때
#   Skip-gram (sg=1):  sat  →  the, cat, on, the     (가운데 단어 하나로 주변 단어 각각을 맞힌다. 드문 단어에 강함)
#   CBOW      (sg=0):  the, cat, on, the  →  sat     (주변 단어들의 평균으로 가운데 단어를 맞힌다. 빠름)
# 개념(Negative Sampling): 어휘 전체(수만 개) 중 정답 하나를 고르는 소프트맥스는 매번 모든 단어 점수를 계산해야 해 느리다.
#   대신 "정답 주변 단어 1 개 + 무작위로 뽑은 오답 단어 k 개(negative)"만 보고 "진짜 이웃인가?" 를 맞히는 이진 분류로 바꾼다
#   → 한 번에 k+1 개 단어만 계산한다. 직접 구현은 advanced/ (P1-03A) 참고.
# 개념(한계): 단어 하나에 벡터 하나(정적 임베딩). '배'(먹는 배·타는 배·신체의 배)를 문맥에 따라 구분하지 못한다.
#   P1-02 의 문장 임베딩 모델(트랜스포머 기반)은 문맥 전체를 보고 벡터를 만들어 이 문제를 줄인다.
"""
from __future__ import annotations

import re  # noqa: F401  (TODO 1 에서 사용)
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec

CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus_ko_en.txt"


def load_lines(path: Path = CORPUS) -> list[str]:
    """(제공) 말뭉치 파일을 줄 단위로 읽는다."""
    return path.read_text(encoding="utf-8").splitlines()


# ---------------------------------------------------------------------------
# TODO 1 — 전처리: 문장 → 토큰 목록
# ---------------------------------------------------------------------------
def preprocess(lines: Iterable[str]) -> list[list[str]]:
    """각 줄을 소문자로 바꾸고 구두점을 공백으로 바꾼 뒤 공백으로 나눈다. 빈 줄은 버린다.

    예) "The cat sat on the mat." → ["the", "cat", "sat", "on", "the", "mat"]
    """
    # TODO 1: 각 줄을 소문자로 → 구두점을 공백으로 → 공백 기준으로 나눠 토큰 목록을 만드세요. 토큰이 없는 줄은 버립니다.
    # 개념: gensim Word2Vec 의 입력은 "문장 목록", 문장은 "토큰(문자열) 목록"이다. [["the", "cat", ...], ["나는", ...], ...]
    # HINT 1: "The" 와 "the", "mat." 과 "mat" 이 같은 단어로 세어져야 합니다.
    # HINT 2: line.lower(), re.sub(r"[^\w\s]", " ", s), s.split()
    # WHY: 대소문자·구두점이 다르면 같은 단어의 학습 데이터가 여러 단어로 쪼개진다.
    # 서비스 관점: 공백 분리는 한국어에서 "값이", "값을", "값" 을 서로 다른 단어로 만든다(조사). 실무는 형태소 분석기·서브워드를 쓴다.
    raise NotImplementedError("TODO 1")


# ---------------------------------------------------------------------------
# TODO 2 — gensim Word2Vec 학습
# ---------------------------------------------------------------------------
def train_word2vec(
    sentences: Sequence[Sequence[str]],
    sg: int = 1,
    vector_size: int = 32,
    window: int = 2,
    min_count: int = 1,
    negative: int = 5,
    epochs: int = 200,
    seed: int = 0,
) -> Word2Vec:
    """어휘를 만들고(build_vocab) epochs 만큼 학습한다(train). epochs=0 이면 학습 전(무작위 초기값) 모델을 돌려준다."""
    # TODO 2: Word2Vec(...) 객체를 인자 그대로 만들고(hs=0, sample=0, workers=1 고정), build_vocab 으로 어휘를 만든 뒤,
    #         epochs > 0 일 때만 train 으로 학습해 모델을 반환하세요.
    # 하이퍼파라미터: vector_size(차원), window(주변 범위), min_count(최소 빈도), sg(1=Skip-gram, 0=CBOW),
    #   negative(오답 단어 수), epochs(반복 횟수), seed + workers=1(재현성), sample=0(빈도 높은 단어 다운샘플링 끔).
    #   (각각의 뜻과 품질·비용 영향은 solution.py 주석과 README §3 참고)
    # HINT 1: 두 단계로 나눕니다. 1) 단어 목록 세기·벡터 무작위 초기화 2) 주변 단어 맞히기로 벡터 고치기.
    # HINT 2: model = Word2Vec(vector_size=..., ...); model.build_vocab(sentences);
    #         model.train(sentences, total_examples=len(sentences), epochs=epochs)
    # WHY(두 단계): 학습 전(epochs=0)과 학습 후를 같은 초기값에서 비교할 수 있다(TODO 3).
    # 서비스 관점: 같은 데이터·설정·시드여야 비교가 공정하다. 버전·플랫폼에 따라 값이 조금 다를 수 있어 결과는 상대 비교로 판단한다.
    # REF: gensim Word2Vec API https://radimrehurek.com/gensim/models/word2vec.html (작성 환경 접속 불가 → 미확인)
    raise NotImplementedError("TODO 2")


# ---------------------------------------------------------------------------
# TODO 3 — 학습 전후 유사도 비교, 최근접 단어
# ---------------------------------------------------------------------------
def pair_similarities(model: Word2Vec, pairs: Sequence[tuple[str, str]]) -> list[dict]:
    """단어 쌍마다 코사인 유사도. 어휘에 없는 단어(OOV)가 있으면 similarity=None. 반환: [{"pair": (a, b), "similarity": float|None}]"""
    # TODO 3: pairs 의 각 (a, b) 에 대해 {"pair": (a, b), "similarity": 코사인} 을 만드세요. 둘 중 하나라도 어휘에 없으면 None.
    # 개념: model.wv 는 단어 → 벡터 표. wv.similarity(a, b) = 두 단어 벡터의 코사인(P1-02 TODO 2 와 같은 계산).
    # HINT 1: 없는 단어를 바로 조회하면 KeyError 입니다. 먼저 어휘에 있는지 확인하세요.
    # HINT 2: `a in model.wv`, float(model.wv.similarity(a, b))
    # WHY(대조군): 작은 말뭉치에서는 모든 단어가 꽤 가깝다. 구조가 다른 쌍(cat–coffee)보다 "상대적으로" 높은지를 본다.
    # 서비스 관점(OOV): 학습 때 없던 단어(신조어·오타·새 상품명)는 벡터가 없다.
    raise NotImplementedError("TODO 3")


def nearest(model: Word2Vec, word: str, topn: int = 5) -> list[tuple[str, float]]:
    """word 와 가장 가까운 단어 topn 개 [(단어, 유사도), ...]. OOV 면 []."""
    # TODO 3 (이어서): word 와 가장 가까운 단어 topn 개를 [(단어, float 유사도), ...] 로 반환하세요. 어휘에 없으면 [].
    # 개념: most_similar 는 다른 모든 단어와의 코사인을 구해 큰 순서로 돌려준다(P1-02 TODO 4 top_k 와 같은 일).
    # HINT 1: OOV 확인은 위와 같습니다.
    # HINT 2: model.wv.most_similar(word, topn=topn)
    # 서비스 관점(유사어 확장): 도메인 문서로 학습한 최근접 단어는 검색어 확장 후보가 된다. 반의어도 가깝게 나올 수 있어 사람이 검토한다.
    raise NotImplementedError("TODO 3")


# ---------------------------------------------------------------------------
# TODO 4 (선택) — 단어 벡터 평균으로 만든 문장 벡터
# ---------------------------------------------------------------------------
def sentence_vector(model: Word2Vec, tokens: Sequence[str]) -> np.ndarray:
    """어휘에 있는 토큰들의 벡터 평균 (vector_size,). 아는 토큰이 하나도 없으면 0 벡터."""
    # TODO 4 (선택): tokens 중 어휘에 있는 단어 벡터들의 평균을 float32 (vector_size,) 로 반환하세요. 하나도 없으면 0 벡터.
    # 개념: 정적 단어 벡터로 문장 벡터를 만드는 가장 단순한 방법 = 평균. 단어 순서를 버린다.
    # HINT 1: 아는 단어의 벡터만 모은 목록을 먼저 만드세요.
    # HINT 2: model.wv[t], np.mean(vectors, axis=0), np.zeros(model.vector_size, dtype=np.float32)
    # 서비스 관점: 실무 RAG 가 문장 임베딩 모델(P1-02)을 쓰는 이유 — 순서·문맥·다의어 반영. Word2Vec 은 가볍고 오프라인·사내 용어 처리에 쓸 만하다.
    raise NotImplementedError("TODO 4")


# ---------------------------------------------------------------------------
# 단독 실행: 어느 TODO 가 남았는지 알려 준다.
# ---------------------------------------------------------------------------
def _report_progress() -> None:
    toy = [["the", "cat", "sat"], ["the", "dog", "sat"]]
    checks = [
        ("TODO 1 preprocess", lambda: preprocess(["The cat."])),
        ("TODO 2 train_word2vec", lambda: train_word2vec(toy, epochs=1)),
        ("TODO 3 pair_similarities / nearest", lambda: (pair_similarities(train_word2vec(toy, epochs=0), [("cat", "dog")]),
                                                       nearest(train_word2vec(toy, epochs=0), "cat", 1))),
        ("TODO 4 sentence_vector (선택)", lambda: sentence_vector(train_word2vec(toy, epochs=0), ["cat"])),
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
        print(f"\n남은 TODO {remaining}개 — 위에서부터 순서대로 채우세요(TODO 3·4 는 TODO 2 의 모델을 사용합니다). 채운 뒤 검사:")
        print("  (bash)       IMPL=starter pytest part1_ai_fundamentals/word2vec/tests")
        print('  (PowerShell) $env:IMPL="starter"; pytest part1_ai_fundamentals/word2vec/tests; Remove-Item Env:IMPL')
    else:
        print("\n모든 TODO 가 구현되었습니다. 이제 pytest 로 정답 여부를 확인하세요.")


if __name__ == "__main__":
    _report_progress()

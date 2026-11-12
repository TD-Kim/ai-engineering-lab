"""P1-03 Word2Vec — 정답 코드 (메인 실습: 짧게 — 개념 + gensim 짧은 실행).

문장을 토큰 목록으로 바꾸고(TODO 1), gensim 으로 Word2Vec 을 학습하고(TODO 2),
학습 전후 단어 쌍 유사도와 최근접 단어를 비교한다(TODO 3). (선택) 단어 벡터 평균으로 문장 벡터를 만들어
P1-02 문장 임베딩과의 차이를 본다(TODO 4).

진행 확인: python part1_ai_fundamentals/word2vec/solution.py
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

import re
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
    # TODO 1
    # 개념: gensim Word2Vec 의 입력은 "문장 목록", 문장은 "토큰(문자열) 목록"이다. [["the", "cat", ...], ["나는", ...], ...]
    #       문장 경계를 넘어서는 주변 단어로 보지 않으므로 줄(문장) 단위로 나눠 둔다.
    # WHY(소문자·구두점): "The" 와 "the", "mat." 과 "mat" 이 서로 다른 단어로 세어지면 같은 단어의 학습 데이터가 쪼개진다.
    #   re.sub(r"[^\w\s]", " ", s): \w(글자·숫자·_)와 \s(공백)가 아닌 것(구두점)을 공백으로. 한글은 \w 에 포함된다.
    # 서비스 관점: 공백 분리는 한국어에서 "값이", "값을", "값" 을 서로 다른 단어로 만든다(조사가 붙음).
    #   실무에서는 형태소 분석기·서브워드 토크나이저(P1-01)를 쓰고, 그 선택이 검색 품질을 바꾼다.
    sentences = []
    for line in lines:
        tokens = re.sub(r"[^\w\s]", " ", line.lower()).split()
        if tokens:
            sentences.append(tokens)
    return sentences


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
    # TODO 2
    # 하이퍼파라미터의 뜻과 영향:
    #   vector_size: 단어 벡터 차원. 크면 표현력↑ 저장·계산↑, 데이터가 적으면 과적합. 이 작은 말뭉치(504 토큰)에는 32 로 충분.
    #   window:      양쪽으로 몇 단어까지를 "주변"으로 보나. 작으면 문법적 이웃(sat–likes), 크면 주제적 이웃(cat–sun)에 가까워진다.
    #   min_count:   이보다 적게 나온 단어는 버린다. 실무 대규모 말뭉치는 5 이상으로 잡음 단어를 줄인다. 여기서는 1(전부 사용).
    #   sg:          1 = Skip-gram, 0 = CBOW (위 모듈 설명 그림).
    #   negative:    Negative Sampling 의 오답 단어 수 k. 작은 데이터는 5~20, 큰 데이터는 2~5 가 흔한 선택(Mikolov et al. 2013b).
    #   epochs:      말뭉치를 몇 번 반복 학습하나. 데이터가 작을수록 많이 필요. 0 이면 학습 전 상태.
    #   seed, workers=1: 초기값 난수 고정 + 단일 스레드 → 같은 환경에서 같은 결과(재현성). 여러 스레드는 순서가 달라져 결과가 흔들린다.
    #   sample=0:    자주 나오는 단어 다운샘플링 끔(작은 말뭉치에서 the 같은 단어까지 모두 쓰기 위해). 대규모에서는 1e-3 정도를 쓴다.
    # WHY(두 단계): build_vocab 은 단어 목록을 세고 벡터를 무작위로 채운다. train 이 그 벡터를 주변 단어 맞히기로 고친다.
    #   두 단계를 나누면 "학습 전 vs 학습 후" 를 같은 초기값에서 비교할 수 있다(TODO 3).
    # 서비스 관점: 같은 데이터·같은 설정·같은 시드여야 실험 비교가 공정하다. gensim 버전·플랫폼이 다르면 값이 조금 다를 수 있으니
    #   테스트·리포트는 "정확한 값" 대신 "대상 쌍이 대조군보다 높다" 같은 상대 비교로 판단한다.
    # REF: gensim Word2Vec API https://radimrehurek.com/gensim/models/word2vec.html (작성 환경 접속 불가 → 미확인)
    model = Word2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        negative=negative,
        hs=0,
        sample=0,
        seed=seed,
        workers=1,
    )
    model.build_vocab(sentences)
    if epochs > 0:
        model.train(sentences, total_examples=len(sentences), epochs=epochs)
    return model


# ---------------------------------------------------------------------------
# TODO 3 — 학습 전후 유사도 비교, 최근접 단어
# ---------------------------------------------------------------------------
def pair_similarities(model: Word2Vec, pairs: Sequence[tuple[str, str]]) -> list[dict]:
    """단어 쌍마다 코사인 유사도. 어휘에 없는 단어(OOV)가 있으면 similarity=None. 반환: [{"pair": (a, b), "similarity": float|None}]"""
    # TODO 3
    # 개념: model.wv 는 단어 → 벡터 표(KeyedVectors). wv.similarity(a, b) = 두 단어 벡터의 코사인(P1-02 TODO 2 와 같은 계산).
    #       `word in model.wv` 로 어휘에 있는지 확인한다. 없는 단어를 바로 조회하면 KeyError.
    # WHY(대조군): "cat–dog 0.97" 만 보면 높은지 알 수 없다. 이 작은 말뭉치에서는 모든 단어가 서로 꽤 가깝기 때문이다.
    #   구조가 다른 쌍(cat–coffee)과 비교해 "상대적으로" 높은지를 본다.
    # 서비스 관점(OOV): 학습 때 없던 단어는 벡터가 없다. 신조어·오타·새 상품명이 많으면 Word2Vec 은 답을 못 한다.
    #   서브워드를 쓰는 모델(FastText, 문장 임베딩 모델)은 조각으로라도 벡터를 만든다.
    rows = []
    for a, b in pairs:
        sim = float(model.wv.similarity(a, b)) if a in model.wv and b in model.wv else None
        rows.append({"pair": (a, b), "similarity": sim})
    return rows


def nearest(model: Word2Vec, word: str, topn: int = 5) -> list[tuple[str, float]]:
    """word 와 가장 가까운 단어 topn 개 [(단어, 유사도), ...]. OOV 면 []."""
    # TODO 3 (이어서)
    # 개념: most_similar 는 word 벡터와 다른 모든 단어 벡터의 코사인을 구해 큰 순서로 돌려준다(P1-02 TODO 4 top_k 와 같은 일).
    # 서비스 관점(유사어 확장): 도메인 문서로 학습한 Word2Vec 의 최근접 단어는 검색어 확장 후보가 된다(예: '환불' → '반품', '취소').
    #   다만 반의어도 문맥이 같아 가깝게 나올 수 있다(줄어든다 ↔ 늘어난다) → 사람이 검토한 뒤 사전에 넣는다.
    if word not in model.wv:
        return []
    return [(w, float(s)) for w, s in model.wv.most_similar(word, topn=topn)]


# ---------------------------------------------------------------------------
# TODO 4 (선택) — 단어 벡터 평균으로 만든 문장 벡터
# ---------------------------------------------------------------------------
def sentence_vector(model: Word2Vec, tokens: Sequence[str]) -> np.ndarray:
    """어휘에 있는 토큰들의 벡터 평균 (vector_size,). 아는 토큰이 하나도 없으면 0 벡터."""
    # TODO 4
    # 개념: 정적 단어 벡터로 문장 벡터를 만드는 가장 단순한 방법 = 평균. 단어 순서를 버린다.
    #       "개가 사람을 물었다" 와 "사람이 개를 물었다" 가 (조사 처리를 하면) 거의 같은 벡터가 된다.
    # WHY(0 벡터): 모르는 단어만 있으면 평균을 낼 수 없다(0 으로 나누기). 0 벡터는 P1-02 의 l2_normalize 가 안전하게 다룬다.
    # 서비스 관점: 실무 RAG 가 Word2Vec 평균 대신 문장 임베딩 모델(P1-02)을 쓰는 이유
    #   — 순서·문맥·다의어를 반영하고, 질문과 답 문장처럼 단어가 달라도 뜻이 같으면 가깝게 학습되어 있다.
    #   그래도 Word2Vec 은 가볍고(CPU 로 수 초 학습, API 비용 0) 오프라인·사내 전용 용어 처리에 쓸 만하다.
    known = [model.wv[t] for t in tokens if t in model.wv]
    if not known:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(known, axis=0).astype(np.float32)


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

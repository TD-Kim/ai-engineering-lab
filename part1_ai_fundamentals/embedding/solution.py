"""P1-02 Embedding — 정답 코드 (메인 실습: 서비스 관점).

벡터를 정규화하고(TODO 1), 코사인 유사도를 구하고(TODO 2), 임베딩 함수로 문장들을 배치 임베딩하고(TODO 3),
질의와 가장 가까운 문서 k 개를 찾는다(TODO 4). 같은 텍스트를 다시 임베딩하지 않는 캐시(TODO 5, 선택).

임베딩 함수(embed_fn)는 "문자열 목록 → 벡터 목록" 함수를 바깥에서 넣어 준다(주입).
- 기본 서비스 경로: OpenAI 임베딩 API (make_openai_embed_fn, 모델은 환경 변수 EMBEDDING_MODEL, 기본 text-embedding-3-small)
- 테스트·키 없는 환경: fake_embed (결정적인 가짜 임베딩, API 호출 없음)
같은 코드가 두 경로 모두에서 동작하는 것이 "주입"의 목적이다.

진행 확인: python part1_ai_fundamentals/embedding/solution.py
원리(임베딩 층·위치 임베딩 직접 구현)는 선택 과제: part1_ai_fundamentals/embedding/advanced/
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence

import numpy as np

# 임베딩 함수의 모양: 문자열 목록을 받아 같은 개수의 벡터(숫자 목록)를 돌려준다.
EmbedFn = Callable[[list[str]], list[list[float]]]

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


# ---------------------------------------------------------------------------
# (제공) 가짜 임베딩 — 테스트·데모용. API 키·다운로드 불필요
# ---------------------------------------------------------------------------
def fake_embed(texts: list[str], dim: int = 256) -> list[list[float]]:
    """글자 2-gram 을 해시해 dim 차원 칸에 세는 결정적인 가짜 임베딩(feature hashing).

    개념: "손실 값" → 2-gram ['손실', '실 ', ' 값'] 처럼 이웃한 두 글자를 잘라, 각 조각을 해시해 0 ~ dim-1 칸 중 하나에 +1 한다.
          글자 조각이 많이 겹치는 문장끼리 비슷한 벡터가 된다.
    한계: 뜻이 아니라 "글자 겹침"만 본다. '강아지' 와 '개' 처럼 글자가 다른 동의어는 전혀 가깝지 않다 → 실제 서비스는 학습된 임베딩 모델을 쓴다.
    WHY(hashlib): 파이썬 내장 hash() 는 실행할 때마다 값이 바뀌어(보안용 무작위화) 결과가 재현되지 않는다. md5 는 항상 같다.
    """
    out = []
    for text in texts:
        vec = [0.0] * dim
        s = f" {text.lower()} "  # 앞뒤 공백: 첫 글자·끝 글자도 2-gram 에 들어가게
        for i in range(len(s) - 1):
            h = int(hashlib.md5(s[i : i + 2].encode("utf-8")).hexdigest(), 16)
            vec[h % dim] += 1.0
        out.append(vec)
    return out


def make_openai_embed_fn(client, model: str = DEFAULT_EMBEDDING_MODEL, usage: dict | None = None) -> EmbedFn:
    """OpenAI 임베딩 API 를 EmbedFn 모양으로 감싼다. usage 사전을 주면 호출마다 사용 토큰 수를 더한다.

    client: openai.OpenAI() 객체(테스트에서는 같은 모양의 가짜 객체).
    REF: OpenAI Python SDK `client.embeddings.create(model=..., input=[...])` — 설치된 openai 패키지의 시그니처로 확인.
         공식 임베딩 가이드 https://platform.openai.com/docs/guides/embeddings (작성 환경에서 접속 불가 → 미확인)
    """

    def embed(texts: list[str]) -> list[list[float]]:
        resp = client.embeddings.create(model=model, input=texts)
        if usage is not None:
            usage["prompt_tokens"] = usage.get("prompt_tokens", 0) + resp.usage.prompt_tokens
            usage["requests"] = usage.get("requests", 0) + 1
        # 응답 data 는 입력 순서와 같은 index 를 갖는다. 순서를 index 로 확실히 맞춘다.
        return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]

    return embed


# ---------------------------------------------------------------------------
# TODO 1 — L2 정규화
# ---------------------------------------------------------------------------
def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """각 행(벡터)의 길이를 1 로 맞춘다. x: (N, D) → (N, D). 1차원 (D,) 도 받는다."""
    # TODO 1
    # 개념: 벡터에는 "방향"과 "길이"가 있다. 길이(L2 노름) = 각 성분 제곱합의 제곱근(피타고라스).
    #       [3, 4] 의 길이 = √(3² + 4²) = √25 = 5 → 각 성분을 5 로 나누면 [0.6, 0.8], 길이 1. 방향은 그대로.
    #       [6, 8] 도 길이 10 으로 나누면 [0.6, 0.8] — 방향이 같은 벡터는 정규화하면 똑같아진다.
    # 개념(keepdims): x = [[3, 4], [6, 8]] 이면 행마다 길이를 구해야 한다(axis=-1: 마지막 축 = 가로 방향).
    #       keepdims=True  → 길이 모양 (2, 1) = [[5], [10]]  (세로 한 줄짜리 표)
    #       keepdims=False → 길이 모양 (2,)   = [5, 10]
    #       (2, 2) / (2, 1) 은 "브로드캐스팅"으로 [[5],[10]] 을 가로로 복사해 [[5, 5],[10, 10]] 처럼 맞춘 뒤 나눈다 → 행마다 자기 길이로 나눔.
    #       (2, 2) / (2,) 은 [5, 10] 을 "세로로" 복사해 [[5, 10],[5, 10]] 로 맞춰 버려 열마다 나누게 된다 → 오류 없이 틀린 값.
    # WHY(eps): 0 벡터는 길이가 0 이라 그대로 나누면 0/0 = NaN. 길이를 최소 eps 로 두면 0 벡터는 0 벡터로 남는다.
    # 서비스 관점: 문서 벡터를 저장할 때 정규화해 두면, 검색할 때 내적(곱해서 더하기)만으로 코사인 유사도가 된다(TODO 2).
    #             벡터 DB 의 "inner product" 거리와 "cosine" 거리가 같은 결과를 내는 조건이 바로 정규화다.
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, eps)


# ---------------------------------------------------------------------------
# TODO 2 — 코사인 유사도
# ---------------------------------------------------------------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N, D), b: (M, D) → (N, M). [i, j] = a 의 i 번째와 b 의 j 번째 벡터의 코사인 유사도(-1 ~ 1)."""
    # TODO 2
    # 개념: 코사인 유사도 = 두 벡터 사이 각도의 코사인. 같은 방향 1, 직각 0, 반대 방향 -1. 길이는 무시하고 방향만 본다.
    #       길이 1 로 정규화한 두 벡터의 내적(성분끼리 곱해 더하기)이 곧 코사인이다.
    #       예) [0.6, 0.8] · [1, 0] = 0.6×1 + 0.8×0 = 0.6
    # 개념(@ 행렬곱): A (N, D) @ B.T (D, M) → (N, M). 결과의 [i, j] 칸 = A 의 i 행과 B 의 j 행의 내적.
    #       A = [[0.6, 0.8]],  B = [[1, 0], [0, 1]] 이면  A @ B.T = [[0.6×1 + 0.8×0, 0.6×0 + 0.8×1]] = [[0.6, 0.8]]
    #       반복문 없이 "모든 질의 × 모든 문서" 유사도 표를 한 번에 만든다.
    # WHY(.T): B 는 (M, D) 라서 그대로는 (N, D) @ (M, D) 가 모양이 안 맞는다. 전치(.T)로 (D, M) 을 만들어야 D 끼리 곱해진다.
    # 서비스 관점: 문서가 N 개면 질의 하나에 N 번의 내적. 수백만 개면 이 계산이 지연을 정한다 → 벡터 DB 는 근사 검색(ANN)으로 줄인다(Part 3).
    a = l2_normalize(np.atleast_2d(a))
    b = l2_normalize(np.atleast_2d(b))
    return a @ b.T


# ---------------------------------------------------------------------------
# TODO 3 — 문장 임베딩 얻기 (embed_fn 주입, 배치 호출)
# ---------------------------------------------------------------------------
def embed_texts(texts: Sequence[str], embed_fn: EmbedFn, batch_size: int = 64) -> np.ndarray:
    """texts 를 batch_size 개씩 묶어 embed_fn 으로 임베딩한다. 반환: (len(texts), D) float32 배열 (입력 순서 유지)."""
    # TODO 3
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    texts = list(texts)
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    vectors: list[list[float]] = []
    # range(시작, 끝, 간격): 0, batch_size, 2×batch_size, ... — 반드시 끝나는 반복(while 없음)
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        out = embed_fn(batch)
        if len(out) != len(batch):
            # 개수가 어긋나면 어떤 벡터가 어떤 문장 것인지 알 수 없다 → 조용히 넘어가지 말고 바로 알린다.
            raise ValueError(f"embed_fn returned {len(out)} vectors for {len(batch)} texts")
        vectors.extend(out)
    arr = np.asarray(vectors, dtype=np.float32)  # 벡터 길이가 서로 다르면 여기서 ValueError
    # 개념: 임베딩 = 문장을 고정 길이 숫자 목록(벡터)으로 바꾼 것. 뜻이 비슷한 문장일수록 벡터 방향이 비슷해지도록 학습된 모델이 만든다.
    # WHY(배치): API 는 요청 한 번에 여러 문장을 받는다. 1 문장씩 1,000 번 부르면 왕복 지연이 1,000 번 쌓이고 호출 한도에도 걸린다.
    # 서비스 관점: 저장 공간 ≈ 문서 수 × 차원 × 4 바이트(float32). 예) 문서 100만 개 × 1,536 차원이면 약 6.1 GB.
    #             임베딩 모델을 바꾸면 벡터 공간 자체가 달라져 기존 벡터와 비교할 수 없다 → 문서 전체를 다시 임베딩(재색인)해야 한다.
    #             모델 이름과 차원을 벡터와 함께 저장해 두어야 섞이는 사고를 막는다.
    return arr


# ---------------------------------------------------------------------------
# TODO 4 — 최근접 검색 (top-k)
# ---------------------------------------------------------------------------
def top_k(query_vec: np.ndarray, doc_vecs: np.ndarray, k: int) -> list[tuple[int, float]]:
    """질의 벡터 (D,) 와 문서 벡터 (N, D) 의 코사인 유사도 상위 k 개. 반환: [(문서 번호, 점수), ...] 점수 내림차순.

    점수가 같으면 문서 번호가 작은 것이 먼저. k 가 N 보다 크면 N 개.
    """
    # TODO 4
    if k <= 0:
        raise ValueError("k must be > 0")
    scores = cosine_similarity(query_vec, doc_vecs)[0]  # (1, N) → (N,)
    # 개념: argsort 는 "작은 값부터 정렬했을 때의 원래 위치" 목록. 점수에 - 를 붙이면 큰 점수부터가 된다.
    #       kind="stable": 같은 값끼리는 원래 순서(문서 번호 작은 것 먼저)를 지킨다 → 결과가 재현된다.
    order = np.argsort(-scores, kind="stable")[:k]
    # WHY(float(...)): numpy 숫자 대신 파이썬 float 로 돌려주면 JSON 응답·로그에 그대로 쓸 수 있다.
    # 서비스 관점: 이것이 Part 3 Dense 검색의 원형이다. 질의 임베딩 1 번 + 유사도 계산 → 상위 k 개 문서를 LLM 프롬프트에 넣는다(RAG).
    #             k 가 크면 재현율(놓치지 않음)은 오르지만 프롬프트 토큰(=비용·지연)이 는다(P1-01 TODO 3).
    #             문서는 미리 한 번 임베딩해 저장하고, 질의는 요청마다 임베딩한다 → 비용은 "문서: 색인 때 1 회", "질의: 요청 수만큼".
    return [(int(i), float(scores[i])) for i in order]


# ---------------------------------------------------------------------------
# TODO 5 (선택) — 임베딩 캐시
# ---------------------------------------------------------------------------
def embed_with_cache(
    texts: Sequence[str], embed_fn: EmbedFn, cache: dict[str, list[float]], batch_size: int = 64
) -> np.ndarray:
    """cache(텍스트 → 벡터)에 없는 텍스트만 embed_fn 으로 임베딩하고 cache 에 넣는다. 반환: (len(texts), D) 입력 순서."""
    # TODO 5
    # 개념: 같은 입력이면 같은 벡터가 나오므로(같은 모델 기준), 한 번 구한 벡터를 저장해 두고 재사용한다.
    #       dict.fromkeys(목록) 은 순서를 지키며 중복을 없앤다. 예) ['a', 'b', 'a'] → ['a', 'b']
    missing = [t for t in dict.fromkeys(texts) if t not in cache]
    if missing:
        new_vecs = embed_texts(missing, embed_fn, batch_size=batch_size)
        for t, v in zip(missing, new_vecs, strict=True):
            cache[t] = v.tolist()
    # WHY(중복 제거): 같은 배치 안에 같은 문장이 두 번 있어도 API 에는 한 번만 보낸다.
    # 서비스 관점: 자주 묻는 질문·변경되지 않은 문서를 다시 임베딩하지 않아 비용과 지연이 준다.
    #             캐시 키에는 텍스트만이 아니라 "모델 이름"도 넣어야 한다(모델을 바꾸면 벡터가 달라짐). 이 실습은 모델 하나만 가정한다.
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    return np.asarray([cache[t] for t in texts], dtype=np.float32)


# ---------------------------------------------------------------------------
# 단독 실행: 어느 TODO 가 남았는지 알려 준다.
# ---------------------------------------------------------------------------
def _report_progress() -> None:
    v = np.array([[3.0, 4.0], [1.0, 0.0]])
    checks = [
        ("TODO 1 l2_normalize", lambda: l2_normalize(v)),
        ("TODO 2 cosine_similarity", lambda: cosine_similarity(v, v)),
        ("TODO 3 embed_texts", lambda: embed_texts(["a", "b"], fake_embed, batch_size=1)),
        ("TODO 4 top_k", lambda: top_k(v[0], v, 1)),
        ("TODO 5 embed_with_cache (선택)", lambda: embed_with_cache(["a"], fake_embed, {})),
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
        print(f"\n남은 TODO {remaining}개 — 위에서부터 순서대로 채우세요(TODO 2 는 TODO 1, TODO 4 는 TODO 2, TODO 5 는 TODO 3 을 사용). 채운 뒤 검사:")
        print("  (bash)       IMPL=starter pytest part1_ai_fundamentals/embedding/tests")
        print('  (PowerShell) $env:IMPL="starter"; pytest part1_ai_fundamentals/embedding/tests; Remove-Item Env:IMPL')
    else:
        print("\n모든 TODO 가 구현되었습니다. 이제 pytest 로 정답 여부를 확인하세요.")


if __name__ == "__main__":
    _report_progress()

"""P1-02 Embedding — 실습 코드 (TODO 를 채우세요). 메인 실습: 서비스 관점.

벡터를 정규화하고(TODO 1), 코사인 유사도를 구하고(TODO 2), 임베딩 함수로 문장들을 배치 임베딩하고(TODO 3),
질의와 가장 가까운 문서 k 개를 찾는다(TODO 4). 같은 텍스트를 다시 임베딩하지 않는 캐시(TODO 5, 선택).

임베딩 함수(embed_fn)는 "문자열 목록 → 벡터 목록" 함수를 바깥에서 넣어 준다(주입).
- 기본 서비스 경로: OpenAI 임베딩 API (make_openai_embed_fn, 모델은 환경 변수 EMBEDDING_MODEL, 기본 text-embedding-3-small)
- 테스트·키 없는 환경: fake_embed (결정적인 가짜 임베딩, API 호출 없음)
같은 코드가 두 경로 모두에서 동작하는 것이 "주입"의 목적이다.

검사: IMPL=starter pytest part1_ai_fundamentals/embedding/tests
진행 확인: python part1_ai_fundamentals/embedding/starter.py   (남은 TODO 목록 출력)
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

    개념: "손실 값" → 앞뒤에 공백을 붙인 " 손실 값 " 에서 이웃한 두 글자씩 [' 손', '손실', '실 ', ' 값', '값 '] 을 잘라,
          각 조각을 해시해 0 ~ dim-1 칸 중 하나에 +1 한다.
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
    # TODO 1: x 의 각 행을 자기 길이(L2 노름)로 나눠 길이 1 로 만드세요. 길이가 eps 보다 작으면 eps 로 나눕니다(0 벡터 → 0 벡터).
    # 개념: 길이 = 각 성분 제곱합의 제곱근. [3, 4] → 길이 √(9 + 16) = 5 → [3/5, 4/5] = [0.6, 0.8].
    #       방향이 같은 [6, 8] 도 정규화하면 [0.6, 0.8] 이 된다 — 정규화는 "길이를 버리고 방향만 남기기"다.
    # 개념(keepdims): x = [[3, 4], [6, 8]] 의 행별 길이는 keepdims=True 면 모양 (2, 1) = [[5], [10]], False 면 (2,) = [5, 10].
    #       (2, 2) / (2, 1) 은 [[5],[10]] 을 가로로 복사해 행마다 자기 길이로 나눈다(브로드캐스팅).
    #       (2, 2) / (2,) 은 [5, 10] 을 세로로 복사해 열마다 나눠 버린다 → 오류 없이 틀린 값.
    # HINT 1: 마지막 축(axis=-1, 가로 방향)을 따라 길이를 구하고, 모양을 유지(keepdims)한 채 나누세요.
    # HINT 2: np.linalg.norm(x, axis=-1, keepdims=True), np.maximum(norm, eps)
    # WHY(eps): 0 벡터를 그대로 나누면 0/0 = NaN 이 생기고, NaN 은 이후 모든 유사도 계산을 망친다.
    # 서비스 관점: 문서 벡터를 정규화해 저장하면, 검색 때 내적만으로 코사인 유사도가 된다(벡터 DB 의 inner product = cosine).
    raise NotImplementedError("TODO 1")


# ---------------------------------------------------------------------------
# TODO 2 — 코사인 유사도
# ---------------------------------------------------------------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N, D), b: (M, D) → (N, M). [i, j] = a 의 i 번째와 b 의 j 번째 벡터의 코사인 유사도(-1 ~ 1)."""
    # TODO 2: a 와 b 를 TODO 1 로 정규화한 뒤, 모든 쌍의 내적 표 (N, M) 을 행렬곱으로 구하세요. 1차원 입력은 (1, D) 로 바꿔 씁니다.
    # 개념: 길이 1 인 두 벡터의 내적 = 사이 각도의 코사인. 같은 방향 1, 직각 0, 반대 -1.
    #       예) [0.6, 0.8] · [1, 0] = 0.6×1 + 0.8×0 = 0.6
    # 개념(@ 행렬곱): (N, D) @ (D, M) → (N, M). [i, j] 칸 = 앞 행렬의 i 행과 뒤 행렬의 j 열의 내적.
    #       A = [[0.6, 0.8]], B = [[1, 0], [0, 1]] 이면 A @ B.T = [[0.6, 0.8]].
    # HINT 1: b 는 (M, D) 이므로 그대로는 모양이 맞지 않습니다. 전치해서 (D, M) 으로 만드세요.
    # HINT 2: np.atleast_2d, l2_normalize, a @ b.T
    # 서비스 관점: 문서 N 개면 질의 하나에 내적 N 번. 수백만 개면 이 계산이 지연을 정한다 → 벡터 DB 는 근사 검색(ANN)으로 줄인다.
    raise NotImplementedError("TODO 2")


# ---------------------------------------------------------------------------
# TODO 3 — 문장 임베딩 얻기 (embed_fn 주입, 배치 호출)
# ---------------------------------------------------------------------------
def embed_texts(texts: Sequence[str], embed_fn: EmbedFn, batch_size: int = 64) -> np.ndarray:
    """texts 를 batch_size 개씩 묶어 embed_fn 으로 임베딩한다. 반환: (len(texts), D) float32 배열 (입력 순서 유지)."""
    # TODO 3: texts 를 batch_size 개씩 잘라 embed_fn 을 부르고, 결과를 이어 붙여 (N, D) float32 배열로 반환하세요.
    #         batch_size <= 0 이면 ValueError, 빈 목록이면 np.zeros((0, 0), dtype=np.float32),
    #         embed_fn 이 배치 크기와 다른 개수의 벡터를 돌려주면 ValueError.
    # 개념: 임베딩 = 문장을 고정 길이 숫자 목록(벡터)으로 바꾼 것. 뜻이 비슷한 문장일수록 방향이 비슷하도록 학습된 모델이 만든다.
    # HINT 1: 반복은 "0, batch_size, 2×batch_size, ..." 위치에서 시작하는 조각 단위로 하세요(반드시 끝나는 for 반복).
    # HINT 2: for i in range(0, len(texts), batch_size): batch = texts[i : i + batch_size] ...  np.asarray(vectors, dtype=np.float32)
    # WHY(배치): 1 문장씩 1,000 번 호출하면 왕복 지연이 1,000 번 쌓이고 호출 한도에도 걸린다.
    # 서비스 관점: 저장 공간 ≈ 문서 수 × 차원 × 4 바이트. 예) 100만 개 × 1,536 차원이면 약 6.1 GB.
    #             임베딩 모델을 바꾸면 벡터 공간이 달라져 기존 벡터와 비교할 수 없다 → 전체 재색인. 모델 이름·차원을 함께 저장한다.
    raise NotImplementedError("TODO 3")


# ---------------------------------------------------------------------------
# TODO 4 — 최근접 검색 (top-k)
# ---------------------------------------------------------------------------
def top_k(query_vec: np.ndarray, doc_vecs: np.ndarray, k: int) -> list[tuple[int, float]]:
    """질의 벡터 (D,) 와 문서 벡터 (N, D) 의 코사인 유사도 상위 k 개. 반환: [(문서 번호, 점수), ...] 점수 내림차순.

    점수가 같으면 문서 번호가 작은 것이 먼저. k 가 N 보다 크면 N 개.
    """
    # TODO 4: 질의와 모든 문서의 코사인 유사도(TODO 2)를 구해, 점수가 큰 순서로 k 개의 (문서 번호, 점수)를 반환하세요.
    #         k <= 0 이면 ValueError. 점수가 같으면 문서 번호가 작은 것 먼저. 번호는 int, 점수는 float 로.
    # 개념: argsort 는 "작은 값부터 정렬했을 때의 원래 위치" 목록이다. 점수에 - 를 붙이면 큰 점수부터가 된다.
    # HINT 1: cosine_similarity(query_vec, doc_vecs) 의 결과는 (1, N) 입니다. 첫 행만 쓰세요.
    # HINT 2: np.argsort(-scores, kind="stable")[:k]   (stable: 같은 값은 원래 순서 유지)
    # 서비스 관점: Part 3 Dense 검색의 원형이다. 상위 k 개 문서를 LLM 프롬프트에 넣는다(RAG). k 가 크면 놓치는 문서는 줄지만 토큰(비용)이 는다.
    #             문서는 색인 때 1 회 임베딩, 질의는 요청마다 임베딩 → 질의 임베딩 비용은 트래픽에 비례한다.
    raise NotImplementedError("TODO 4")


# ---------------------------------------------------------------------------
# TODO 5 (선택) — 임베딩 캐시
# ---------------------------------------------------------------------------
def embed_with_cache(
    texts: Sequence[str], embed_fn: EmbedFn, cache: dict[str, list[float]], batch_size: int = 64
) -> np.ndarray:
    """cache(텍스트 → 벡터)에 없는 텍스트만 embed_fn 으로 임베딩하고 cache 에 넣는다. 반환: (len(texts), D) 입력 순서."""
    # TODO 5 (선택): cache 에 없는 텍스트만(중복 없이, 처음 나온 순서로) TODO 3 의 embed_texts 로 임베딩해 cache 에 넣고,
    #         texts 순서대로 벡터를 모아 (N, D) 배열로 반환하세요. 빈 목록이면 np.zeros((0, 0), dtype=np.float32).
    # 개념: 같은 모델·같은 입력이면 같은 벡터가 나오므로, 한 번 구한 벡터를 저장해 두고 재사용한다.
    # HINT 1: 먼저 "임베딩해야 할 텍스트 목록"을 만들고(이미 cache 에 있거나 앞에서 나온 것은 제외), 한 번에 임베딩하세요.
    # HINT 2: dict.fromkeys(texts) 는 순서를 지키며 중복을 없앱니다. cache[t] = vec.tolist()
    # 서비스 관점: 자주 묻는 질문·바뀌지 않은 문서를 다시 임베딩하지 않아 비용과 지연이 준다. 캐시 키에는 모델 이름도 넣어야 한다.
    raise NotImplementedError("TODO 5")


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

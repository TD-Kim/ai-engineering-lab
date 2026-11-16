"""P1-02 Embedding — 실습 코드 (TODO 를 채우세요). 메인 실습: 서비스 관점.

벡터를 정규화하고(TODO 1), 코사인 유사도를 구하고(TODO 2), 임베딩 함수로 문장들을 배치 임베딩하고(TODO 3),
질의와 가장 가까운 문서 k 개를 찾는다(TODO 4). 같은 텍스트를 다시 임베딩하지 않는 캐시(TODO 5, 선택).

임베딩 함수(embed_fn)는 "문자열 목록 → 벡터 목록" 함수를 바깥에서 넣어 준다(주입).
- 기본 서비스 경로: OpenAI 임베딩 API (make_openai_embed_fn, 모델은 환경 변수 EMBEDDING_MODEL, 기본 text-embedding-3-small)
- 테스트·키 없는 환경: fake_embed (결정적인 가짜 임베딩, API 호출 없음)
- 로컬(키 없이): sentence-transformers(multilingual-e5-small·bge-m3) 또는 Ollama 서버 /api/embed
  → make_embed_fn() 이 EMBEDDING_PROVIDER(fake | openai | sentence_transformers | ollama)로 고른다(제공 코드).
같은 코드가 모든 경로에서 동작하는 것이 "주입"의 목적이다.

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
# (제공) 공급자 고르기 — EMBEDDING_PROVIDER: fake | openai | sentence_transformers | ollama
# ---------------------------------------------------------------------------
# 개념: 아래 함수들은 모두 같은 모양(EmbedFn: 문자열 목록 → 벡터 목록)을 돌려준다. 그래서 TODO 3~5 의 코드는
#       공급자가 바뀌어도 한 줄도 바뀌지 않는다(P2-01B 의 LLM_PROVIDER 와 같은 설계).
# 개념(임베딩 전용 모델): 채팅용 LLM(Qwen·Llama 등)은 "다음 토큰 맞히기"로 학습되어, 문장 하나를 벡터 하나로 요약하는 법을
#       배우지 않았다. 그 은닉 상태를 평균 내 쓰면 검색 품질이 나쁘다. 검색에는 "비슷한 문장은 가깝게, 다른 문장은 멀게"를
#       목표로 학습한 임베딩 전용 모델(e5·bge·OpenAI text-embedding 등)을 쓴다.
# 규칙 1(같은 모델): 문서와 질문은 반드시 같은 모델로 임베딩한다. 모델이 다르면 벡터 공간이 달라 비교 자체가 의미 없다.
# 규칙 2(재임베딩): 모델을 바꾸면 저장된 문서 벡터를 전부 다시 만든다(차원이 같아도 공간이 다르다).
# 규칙 3(차원·정규화): 모델마다 차원이 다르다(DB 칸 크기가 바뀜). 정규화 여부도 다르다 — 이 실습은 cosine_similarity 가
#       항상 다시 정규화하므로 어느 쪽이든 결과가 같지만, 벡터 DB 에 내적(inner product)으로 저장할 때는 정규화를 명시한다.
# 서비스 관점: API(openai) = 키·호출 비용·네트워크 지연·데이터가 외부로 나감 / 로컬(sentence_transformers·ollama) = 비용 0·
#       데이터가 PC 밖으로 안 나감(프라이버시)·대신 GPU/CPU 성능이 지연을 정함. 한국어 품질은 모델마다 다르므로 "내 데이터로" 확인한다.

PROVIDERS = ("fake", "openai", "sentence_transformers", "ollama")
DEFAULT_LOCAL_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"   # sentence_transformers 기본(다국어·소형). 다른 예: BAAI/bge-m3
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"                  # P2-01B 와 같은 변수 OLLAMA_BASE_URL 로 바꾼다
ROLES = ("query", "passage")


def embedding_prefix(model: str, role: str) -> str:
    """모델이 요구하는 입력 접두어. e5 계열 → "query: " / "passage: ", 그 밖(bge-m3 등) → "".

    개념: e5 는 질문과 문서에 서로 다른 접두어를 붙여 학습했다. 붙이지 않으면 학습 때와 입력 모양이 달라 검색 품질이 떨어진다.
          예) 질문 "query: 연차는 며칠 전까지?",  문서 "passage: 연차는 3일 전까지 신청한다."
    근거: bge-m3 는 공식 FlagEmbedding README 모델표의 "query instruction for retrieval" 칸이 비어 있다(2026-11-16 확인).
          e5 의 "query: "/"passage: " 규칙은 Hugging Face 모델 카드(intfloat/multilingual-e5-small)에 적혀 있다
          — 작성 환경에서 huggingface.co 접속 불가 → 미확인. 쓰기 전에 모델 카드에서 확인하자.
    한계: 이 함수는 기본 e5 규칙(query:/passage:)만 처리한다. e5-instruct 같은 변형이나 다른 임베딩 모델은 각 모델 카드의 형식을 따를 것(미확인).
    """
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    return f"{role}: " if "e5" in model.lower() else ""


_ST_MODELS: dict[tuple[str, str | None], object] = {}   # (모델, 장치) → 불러온 모델. 질문용·문서용이 같은 모델을 공유한다


def make_sentence_transformer_embed_fn(model: str = DEFAULT_LOCAL_EMBEDDING_MODEL, role: str = "passage", normalize: bool = True,
                                       device: str | None = None, factory: Callable | None = None, batch_size: int = 32) -> EmbedFn:
    """로컬 sentence-transformers 모델을 EmbedFn 으로. 접두어(embedding_prefix)를 붙이고 normalize 여부를 명시한다.

    device=None 이면 라이브러리가 GPU 를 쓸 수 있는지 스스로 확인해 고른다(sentence-transformers 6.3.0 docstring 확인).
    factory: 모델을 만드는 함수(기본 sentence_transformers.SentenceTransformer). 테스트에서는 가짜를 넣는다(다운로드 없음).
    REF: sentence-transformers 6.3.0 `SentenceTransformer(model_name_or_path, device=None)`,
         `encode(sentences, batch_size=..., normalize_embeddings=..., convert_to_numpy=True)` — 패키지 소스로 확인(2026-11-16)
    """
    prefix = embedding_prefix(model, role)
    if factory is None:
        # 선택 의존성: 이 공급자를 쓸 때만 필요(requirements/part1.txt 주석 참고)
        from sentence_transformers import SentenceTransformer

        factory = SentenceTransformer
    key = (model, device)
    if key not in _ST_MODELS:
        _ST_MODELS[key] = factory(model, device=device)   # 처음 한 번만 불러온다(수백 MB 다운로드·메모리 적재)
    st_model = _ST_MODELS[key]

    def embed(texts: list[str]) -> list[list[float]]:
        vecs = st_model.encode([prefix + t for t in texts], batch_size=batch_size,
                               normalize_embeddings=normalize, convert_to_numpy=True)
        return [list(map(float, v)) for v in vecs]

    return embed


def post_json(url: str, payload: dict, timeout: float = 60.0) -> dict:
    """(제공) JSON 을 POST 하고 JSON 응답을 돌려준다(표준 라이브러리 urllib). 연결 실패는 ConnectionError 로 안내."""
    import json
    import urllib.error
    import urllib.request

    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # localhost 의 사용자 서버
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise ConnectionError(f"Ollama 서버에 연결할 수 없습니다({url}). `ollama serve` 가 켜져 있는지 확인하세요: {e}") from e


def make_ollama_embed_fn(model: str, base_url: str = DEFAULT_OLLAMA_BASE_URL, post: Callable[[str, dict], dict] = post_json,
                         role: str = "passage") -> EmbedFn:
    """로컬 Ollama 서버의 POST /api/embed 를 EmbedFn 으로. 응답의 "embeddings" 목록을 그대로 돌려준다.

    REF: Ollama 공식 문서 github.com/ollama/ollama docs/api.md "Generate Embeddings", docs/capabilities/embeddings.mdx (2026-11-16 확인)
         요청 {"model": ..., "input": [문장, ...]} → 응답 {"model": ..., "embeddings": [[...], ...], ...}.
         /api/embed 는 L2 정규화된(길이 1) 벡터를 돌려준다. 예전 /api/embeddings(단수 prompt)는 /api/embed 로 대체됨.
    모델 태그는 ollama.com 라이브러리의 embedding 모델에서 고르고 `ollama pull <태그>` 로 받는다(이 저장소는 태그를 정하지 않는다).
    post: (url, payload) → 응답 JSON. 테스트에서는 가짜 함수를 넣는다(서버 불필요).
    접두어 규칙은 공급자와 무관하게 모델을 따른다(e5 계열 태그면 "query: "/"passage: ").
    """
    url = base_url.rstrip("/") + "/api/embed"
    prefix = embedding_prefix(model, role)

    def embed(texts: list[str]) -> list[list[float]]:
        body = post(url, {"model": model, "input": [prefix + t for t in texts]})
        if "embeddings" not in body:
            raise ValueError(f"Ollama 응답에 embeddings 가 없습니다: {body.get('error', body)}")
        return body["embeddings"]

    return embed


def make_embed_fn(provider: str | None = None, role: str = "passage", env: dict | None = None, openai_factory: Callable | None = None,
                  st_factory: Callable | None = None, ollama_post: Callable | None = None) -> EmbedFn:
    """환경 변수로 공급자를 골라 EmbedFn 을 만든다. provider 를 주지 않으면 env["EMBEDDING_PROVIDER"](기본 "fake").

    - fake                  → fake_embed (키·다운로드·서버 불필요)
    - openai                → OPENAI_API_KEY 필요. 모델 EMBEDDING_MODEL(기본 text-embedding-3-small)
    - sentence_transformers → 로컬. 모델 EMBEDDING_MODEL(기본 intfloat/multilingual-e5-small), role 에 맞는 접두어
    - ollama                → 로컬 서버. 모델 EMBEDDING_MODEL 필수(태그는 직접 확인), 주소 OLLAMA_BASE_URL
    role: "query"(질문) / "passage"(문서). e5 처럼 둘을 구분하는 모델에서만 결과가 달라진다.
    로컬 공급자에 OpenAI 모델 이름(text-embedding-...)이 남아 있으면 ValueError(.env 를 함께 바꾸라는 안내).
    """
    import os

    env = dict(os.environ) if env is None else env
    name = (provider or env.get("EMBEDDING_PROVIDER") or "fake").strip().lower()
    if name not in PROVIDERS:
        raise ValueError(f"EMBEDDING_PROVIDER must be one of {PROVIDERS}, got {name!r}")
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    model = env.get("EMBEDDING_MODEL", "").strip()
    if name == "fake":
        return fake_embed
    if name == "openai":
        if not env.get("OPENAI_API_KEY"):
            raise ValueError("EMBEDDING_PROVIDER=openai 에는 OPENAI_API_KEY 가 필요합니다(.env 에 입력, 커밋 금지).")
        if openai_factory is None:
            import openai  # 선택 의존성

            def openai_factory():
                return openai.OpenAI(timeout=30.0, max_retries=2)
        return make_openai_embed_fn(openai_factory(), model=model or DEFAULT_EMBEDDING_MODEL)
    if model.startswith("text-embedding-"):
        raise ValueError(f"EMBEDDING_MODEL={model!r} 은 OpenAI 모델입니다. {name} 공급자에 맞는 모델 이름으로 .env 를 바꾸세요.")
    if name == "sentence_transformers":
        return make_sentence_transformer_embed_fn(model or DEFAULT_LOCAL_EMBEDDING_MODEL, role=role, factory=st_factory)
    if not model:
        raise ValueError("EMBEDDING_PROVIDER=ollama 에는 EMBEDDING_MODEL(Ollama 임베딩 모델 태그)이 필요합니다. "
                         "ollama.com 라이브러리에서 임베딩 모델을 고르고 `ollama pull <태그>` 후 .env 에 적으세요.")
    return make_ollama_embed_fn(model, env.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL), post=ollama_post or post_json, role=role)

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
    # 개념(공급자 교체): embed_fn 은 fake_embed 든 make_embed_fn("sentence_transformers") 든 같은 모양이라 이 함수는 그대로다.
    #       단, 문서는 role="passage", 질문은 role="query" 로 만든 함수를 쓴다(e5 접두어). 모델은 둘이 같아야 한다.
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

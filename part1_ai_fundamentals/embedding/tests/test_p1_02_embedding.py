"""P1-02 테스트. 기본은 solution 검사, `IMPL=starter pytest ...` 로 직접 구현을 검사한다.

API 호출·다운로드 없음: 임베딩은 결정적인 가짜 함수, OpenAI 클라이언트는 가짜 객체로 검사한다.
반복은 모두 길이가 정해진 for 문이라 무한 반복·메모리 폭주 위험이 없다.
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

LAB_DIR = Path(__file__).resolve().parents[1]


class CountingEmbed:
    """호출 횟수와 받은 배치를 기록하는 가짜 embed_fn. 텍스트 길이·글자 코드로 결정적인 3차원 벡터를 만든다."""

    def __init__(self):
        self.batches = []

    def __call__(self, texts):
        self.batches.append(list(texts))
        return [[float(len(t)), float(sum(map(ord, t)) % 7), 1.0] for t in texts]


# --- TODO 1 ---------------------------------------------------------------
def test_l2_normalize_hand_example(impl):
    out = impl.l2_normalize(np.array([[3.0, 4.0], [6.0, 8.0]]))
    np.testing.assert_allclose(out, [[0.6, 0.8], [0.6, 0.8]], rtol=1e-6)


def test_l2_normalize_square_matrix_rows_not_columns(impl):
    # N == D: keepdims 를 빼먹으면 오류 없이 열 방향으로 나눠 틀린 값이 나오는 함정
    x = np.array([[3.0, 4.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    out = impl.l2_normalize(x)
    np.testing.assert_allclose(np.linalg.norm(out, axis=-1), np.ones(3), rtol=1e-6)
    np.testing.assert_allclose(out[0], [0.6, 0.8, 0.0], rtol=1e-6)


def test_l2_normalize_zero_vector_and_1d(impl):
    out = impl.l2_normalize(np.zeros((2, 4)))
    assert not np.isnan(out).any() and np.all(out == 0)
    np.testing.assert_allclose(impl.l2_normalize(np.array([0.0, 5.0])), [0.0, 1.0], rtol=1e-6)


# --- TODO 2 ---------------------------------------------------------------
def test_cosine_similarity_values_and_shape(impl):
    a = np.array([[3.0, 4.0]])
    b = np.array([[6.0, 8.0], [-3.0, -4.0], [4.0, -3.0], [1.0, 0.0]])
    sim = impl.cosine_similarity(a, b)
    assert sim.shape == (1, 4)
    np.testing.assert_allclose(sim[0], [1.0, -1.0, 0.0, 0.6], atol=1e-6)


def test_cosine_similarity_matches_formula(impl):
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(5, 8)), rng.normal(size=(3, 8))
    expected = (a @ b.T) / np.outer(np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1))
    np.testing.assert_allclose(impl.cosine_similarity(a, b), expected, atol=1e-5)
    np.testing.assert_allclose(impl.cosine_similarity(a[0], b).shape, (1, 3))  # 1차원 입력


# --- TODO 3 ---------------------------------------------------------------
def test_embed_texts_batches_and_order(impl):
    fn = CountingEmbed()
    texts = [f"문장{i}" * (i + 1) for i in range(7)]
    out = impl.embed_texts(texts, fn, batch_size=3)
    assert out.shape == (7, 3) and out.dtype == np.float32
    assert [len(b) for b in fn.batches] == [3, 3, 1]                  # 배치 크기대로 나눠 호출
    assert [t for b in fn.batches for t in b] == texts                # 순서 유지
    np.testing.assert_allclose(out, np.asarray(CountingEmbed()(texts), dtype=np.float32))


def test_embed_texts_edge_cases(impl):
    fn = CountingEmbed()
    assert impl.embed_texts([], fn).shape == (0, 0) and fn.batches == []
    with pytest.raises(ValueError):
        impl.embed_texts(["a"], fn, batch_size=0)
    with pytest.raises(ValueError):
        impl.embed_texts(["a", "b"], lambda texts: [[1.0, 2.0]])     # 개수가 어긋나면 알림


def test_embed_texts_with_fake_embed(impl):
    out = impl.embed_texts(["손실 값", "손실 값", "커피"], impl.fake_embed)
    assert out.shape == (3, 256)
    np.testing.assert_array_equal(out[0], out[1])                     # 결정적
    sim = impl.cosine_similarity(out[0], out[1:])[0]
    assert sim[0] == pytest.approx(1.0) and sim[1] < sim[0]


def test_openai_embed_fn_with_fake_client(impl):
    calls = []

    class FakeEmbeddings:
        def create(self, **kwargs):
            calls.append(kwargs)
            n = len(kwargs["input"])
            # 일부러 역순으로 돌려줘도 index 기준으로 순서를 맞추는지 확인
            data = [SimpleNamespace(index=i, embedding=[float(i), 1.0]) for i in reversed(range(n))]
            return SimpleNamespace(data=data, usage=SimpleNamespace(prompt_tokens=10 * n))

    client = SimpleNamespace(embeddings=FakeEmbeddings())
    usage = {}
    fn = impl.make_openai_embed_fn(client, model="text-embedding-3-small", usage=usage)
    out = impl.embed_texts(["a", "b", "c"], fn, batch_size=2)
    np.testing.assert_allclose(out, [[0.0, 1.0], [1.0, 1.0], [0.0, 1.0]])
    assert calls == [{"model": "text-embedding-3-small", "input": ["a", "b"]},
                     {"model": "text-embedding-3-small", "input": ["c"]}]
    assert usage == {"prompt_tokens": 30, "requests": 2}


# --- TODO 4 ---------------------------------------------------------------
def test_top_k_order_ties_and_types(impl):
    docs = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [1.0, 1.0], [-1.0, 0.0]])
    res = impl.top_k(np.array([1.0, 0.0]), docs, 3)
    assert [i for i, _ in res] == [0, 2, 3]                            # 동점(0, 2)은 번호 작은 것 먼저
    assert all(isinstance(i, int) and isinstance(s, float) for i, s in res)
    assert res[0][1] == pytest.approx(1.0) and res[2][1] == pytest.approx(2 ** -0.5, rel=1e-5)


def test_top_k_many_ties_ascending_index(impl):
    # 같은 벡터 25개가 섞인 문서: 동점은 반드시 번호 오름차순. 기본(불안정) 정렬은 이 순서를 지키지 않는다(kind="stable" 필요).
    rng = np.random.default_rng(0)
    docs = np.vstack([np.tile([1.0, 0.0], (25, 1)), rng.normal(size=(15, 2))])
    docs = docs[rng.permutation(len(docs))]
    same = sorted(int(i) for i in np.where((docs == [1.0, 0.0]).all(axis=1))[0])
    res = impl.top_k(np.array([1.0, 0.0]), docs, 25)
    assert [i for i, _ in res] == same

def test_top_k_k_larger_than_n_and_invalid(impl):
    docs = np.eye(3)
    assert len(impl.top_k(np.array([1.0, 0.0, 0.0]), docs, 10)) == 3
    with pytest.raises(ValueError):
        impl.top_k(np.array([1.0, 0.0, 0.0]), docs, 0)


def test_top_k_search_with_fake_embed(impl):
    docs = ["손실 값이 줄지 않으면 학습률을 확인한다.", "나는 아침에 커피를 마신다.", "the cat sat on the mat"]
    doc_vecs = impl.embed_texts(docs, impl.fake_embed)
    q = impl.embed_texts(["손실 값이 줄지 않을 때"], impl.fake_embed)[0]
    assert impl.top_k(q, doc_vecs, 1)[0][0] == 0


# --- TODO 5 (선택) --------------------------------------------------------
def test_embed_with_cache_skips_known_and_duplicates(impl):
    fn = CountingEmbed()
    cache = {}
    out1 = impl.embed_with_cache(["a", "bb", "a"], fn, cache)
    assert fn.batches == [["a", "bb"]]                                 # 중복은 한 번만
    out2 = impl.embed_with_cache(["bb", "ccc", "a"], fn, cache)
    assert fn.batches == [["a", "bb"], ["ccc"]]                        # 캐시에 있는 것은 다시 안 부름
    assert set(cache) == {"a", "bb", "ccc"}
    np.testing.assert_allclose(out1[0], out1[2])
    np.testing.assert_allclose(out2, np.asarray(CountingEmbed()(["bb", "ccc", "a"]), dtype=np.float32))
    assert impl.embed_with_cache([], fn, cache).shape == (0, 0)


# --- 데모 스크립트 ----------------------------------------------------------
def test_run_demo_runs(impl, capsys):
    spec = importlib.util.spec_from_file_location("p1_02_run_demo_test", LAB_DIR / "run_demo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.load = lambda _name: impl                                     # 현재 IMPL 로 실행
    import sys

    argv = sys.argv
    sys.argv = ["run_demo.py"]
    try:
        mod.main()
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert "#07 손실 값이 줄지 않으면" in out and "2회차 추가 0개" in out


# --- (제공 코드) 공급자 고르기: EMBEDDING_PROVIDER ---------------------------------
# 모두 가짜 객체: 모델 다운로드·Ollama 서버·API 키 불필요. starter 에서도 통과해야 한다(제공 코드).
class FakeST:
    """sentence_transformers.SentenceTransformer 흉내. 받은 인자를 기록하고 결정적인 2차원 벡터를 돌려준다."""

    def __init__(self, name, device=None):
        self.name, self.device, self.calls = name, device, []

    def encode(self, sentences, batch_size=32, normalize_embeddings=False, convert_to_numpy=True):
        self.calls.append({"sentences": list(sentences), "batch_size": batch_size,
                           "normalize": normalize_embeddings, "numpy": convert_to_numpy})
        return np.array([[float(len(s)), 1.0] for s in sentences], dtype=np.float32)


def st_factory(created):
    def make(name, device=None):
        created.append(FakeST(name, device))
        return created[-1]
    return make


def test_embedding_prefix_rules(impl):
    assert impl.embedding_prefix("intfloat/multilingual-e5-small", "query") == "query: "
    assert impl.embedding_prefix("intfloat/multilingual-e5-small", "passage") == "passage: "
    assert impl.embedding_prefix("BAAI/bge-m3", "query") == ""
    assert impl.embedding_prefix("intfloat/Multilingual-E5-small", "passage") == "passage: "   # 대소문자 무시
    with pytest.raises(ValueError):
        impl.embedding_prefix("BAAI/bge-m3", "doc")


def test_sentence_transformer_embed_fn_prefix_normalize_and_reuse(impl):
    impl._ST_MODELS.clear()
    created = []
    q = impl.make_embed_fn("sentence_transformers", role="query", env={}, st_factory=st_factory(created))
    d = impl.make_embed_fn("sentence_transformers", role="passage", env={}, st_factory=st_factory(created))
    assert len(created) == 1 and created[0].name == "intfloat/multilingual-e5-small" and created[0].device is None   # 기본 모델, 장치 자동
    assert q(["연차는?"]) == [[float(len("query: 연차는?")), 1.0]]                      # 파이썬 float 목록
    d(["연차는 3일 전"])
    calls = created[0].calls
    assert calls[0]["sentences"] == ["query: 연차는?"] and calls[1]["sentences"] == ["passage: 연차는 3일 전"]
    assert all(c["normalize"] is True and c["numpy"] is True for c in calls)          # 정규화 여부를 명시
    impl._ST_MODELS.clear()


def test_sentence_transformer_bge_no_prefix_and_text_search(impl):
    impl._ST_MODELS.clear()
    created = []
    fn = impl.make_embed_fn("sentence_transformers", env={"EMBEDDING_MODEL": "BAAI/bge-m3"}, st_factory=st_factory(created))
    vecs = impl.embed_texts(["가", "나다"], fn, batch_size=1)                           # TODO 3 과 그대로 연결(solution 에서만 계산)
    assert created[0].name == "BAAI/bge-m3" and created[0].calls[0]["sentences"] == ["가"]
    assert vecs.shape == (2, 2)
    impl._ST_MODELS.clear()


def test_ollama_embed_fn_request_and_response(impl):
    sent = []

    def post(url, payload):
        sent.append((url, payload))
        return {"model": payload["model"], "embeddings": [[0.6, 0.8] for _ in payload["input"]], "prompt_eval_count": 3}

    env = {"EMBEDDING_PROVIDER": "ollama", "EMBEDDING_MODEL": "my-embed-model", "OLLAMA_BASE_URL": "http://127.0.0.1:9999/"}
    fn = impl.make_embed_fn(env=env, ollama_post=post)
    assert fn(["a", "b"]) == [[0.6, 0.8], [0.6, 0.8]]
    assert sent == [("http://127.0.0.1:9999/api/embed", {"model": "my-embed-model", "input": ["a", "b"]})]   # 공식 형식: model + input 목록
    fn_e5 = impl.make_ollama_embed_fn("some-e5-tag", post=post, role="query")
    fn_e5(["q"])
    assert sent[-1] == ("http://localhost:11434/api/embed", {"model": "some-e5-tag", "input": ["query: q"]})   # 기본 주소, 접두어
    q = impl.make_embed_fn("ollama", role="query", env={"EMBEDDING_MODEL": "x-e5-y"}, ollama_post=post)
    q(["질문"])
    assert sent[-1][1]["input"] == ["query: 질문"]                                      # make_embed_fn 의 role 이 Ollama 까지 전달
    bad = impl.make_ollama_embed_fn("m", post=lambda url, payload: {"error": "model not found"})
    with pytest.raises(ValueError, match="model not found"):
        bad(["x"])


def test_make_embed_fn_selection_and_guards(impl):
    assert impl.make_embed_fn(env={}) is impl.fake_embed                                # 기본 = fake
    assert impl.make_embed_fn(env={"EMBEDDING_PROVIDER": " Fake "}) is impl.fake_embed
    with pytest.raises(ValueError):
        impl.make_embed_fn("huggingface", env={})
    with pytest.raises(ValueError):
        impl.make_embed_fn("fake", role="doc", env={})
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        impl.make_embed_fn("openai", env={})
    with pytest.raises(ValueError, match="OpenAI 모델"):                                 # .env 의 OpenAI 모델 이름이 남아 있음
        impl.make_embed_fn("sentence_transformers", env={"EMBEDDING_MODEL": "text-embedding-3-small"}, st_factory=st_factory([]))
    with pytest.raises(ValueError, match="EMBEDDING_MODEL"):                             # ollama 는 태그를 직접 정해야 함
        impl.make_embed_fn("ollama", env={}, ollama_post=lambda u, p: {})


def test_make_embed_fn_openai_with_fake_client(impl):
    def create(model, input):
        return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=[1.0, float(i)]) for i in range(len(input))],
                               usage=SimpleNamespace(prompt_tokens=5))

    client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    fn = impl.make_embed_fn("openai", env={"OPENAI_API_KEY": "test"}, openai_factory=lambda: client)
    assert fn(["a", "b"]) == [[1.0, 0.0], [1.0, 1.0]]

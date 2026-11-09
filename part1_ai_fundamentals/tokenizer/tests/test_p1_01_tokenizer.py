"""P1-01 테스트. 기본은 solution 검사, `IMPL=starter pytest ...` 로 직접 구현을 검사한다.

다운로드·API 호출 없음: 토크나이저는 저장소 말뭉치로 학습하고, Claude 토큰 세기는 가짜 클라이언트로 검사한다.
"""
import importlib.util
import itertools
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

LAB_DIR = Path(__file__).resolve().parents[1]
CORPUS = LAB_DIR.parents[0] / "data" / "corpus_ko_en.txt"


@pytest.fixture(scope="module")
def tok(impl):
    return impl.train_local_tokenizer(CORPUS.read_text(encoding="utf-8"), vocab_size=500)


@pytest.fixture(scope="module")
def corpus():
    return CORPUS.read_text(encoding="utf-8")


# --- TODO 1 ---------------------------------------------------------------
def test_count_tokens_matches_encoding(impl, tok):
    text = "토크나이저는 문장을 작은 조각으로 나눈다. Tokens cost money."
    assert impl.count_tokens(tok, text) == len(tok.encode(text).ids)
    assert impl.count_tokens(tok, "") == 0


# --- TODO 2 ---------------------------------------------------------------
def test_compare_languages_rows(impl, tok):
    samples = {"ko": "오늘은 날씨가 맑다", "en": "The weather is clear today", "empty": ""}
    rows = impl.compare_languages(tok, samples)
    assert [r["name"] for r in rows] == ["ko", "en", "empty"]
    ko, en, empty = rows
    assert ko["chars"] == 10 and en["chars"] == 26
    assert ko["tokens"] == len(tok.encode(samples["ko"]).ids)
    assert ko["chars_per_token"] == round(ko["chars"] / ko["tokens"], 2)
    assert empty == {"name": "empty", "chars": 0, "tokens": 0, "chars_per_token": 0.0}
    # 이 작은 바이트 단위 BPE 에서는 한글 한 글자가 여러 토큰이 되기 쉬워 영어보다 글자당 토큰이 많다.
    assert ko["chars_per_token"] < en["chars_per_token"]


# --- TODO 3 ---------------------------------------------------------------
def test_estimate_cost(impl):
    # opus: 입력 $4, 출력 $20 / 1M 토큰 → 2,000 입력 + 500 출력 = 0.008 + 0.010 = 0.018
    assert impl.estimate_cost(2_000, 500, "claude-opus-5-5") == pytest.approx(0.018)
    assert impl.estimate_cost(2_000, 500, "claude-opus-5-5", requests=1_000) == pytest.approx(18.0)
    custom = {"m": (1.0, 2.0)}
    assert impl.estimate_cost(1_000_000, 1_000_000, "m", prices=custom) == pytest.approx(3.0)
    with pytest.raises(KeyError):
        impl.estimate_cost(1, 1, "no-such-model")


# --- TODO 4 ---------------------------------------------------------------
def _chunk(impl, tok, text, max_tokens, overlap, timeout=10.0):
    """chunk_by_tokens 를 시간 제한을 두고 부른다. 시작점이 앞으로 안 나아가는 구현이면 멈추는 대신 실패 메시지를 낸다.

    (Windows 에서도 동작하도록 signal 대신 스레드 사용. 시간 초과 시 그 스레드는 daemon 으로 버려진다.)
    """
    box = {}

    def run():
        try:
            box["value"] = impl.chunk_by_tokens(tok, text, max_tokens=max_tokens, overlap=overlap)
        except BaseException as e:  # noqa: BLE001  호출한 쪽에서 그대로 다시 낸다
            box["error"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        pytest.fail(f"chunk_by_tokens 가 {timeout}초 안에 끝나지 않음 — 다음 조각 시작(start)이 앞으로 나아가는지 확인하세요")
    if "error" in box:
        raise box["error"]
    chunks = box["value"]
    n_tokens = len(tok.encode(text).ids)
    # 조각 수 상한: 시작점이 매번 최소 1 토큰은 나아가므로 조각 수는 토큰 수를 넘을 수 없다.
    assert len(chunks) <= max(1, n_tokens), f"조각 {len(chunks)}개 > 토큰 {n_tokens}개 — 같은 위치를 반복해서 자르고 있습니다"
    return chunks

def _token_spans(tok, text, chunk):
    """chunk 가 원문의 어느 토큰 구간인지 찾아 토큰 수를 센다(원문 전체 인코딩 기준)."""
    start = text.index(chunk)
    end = start + len(chunk)
    return sum(1 for a, b in tok.encode(text).offsets if a >= start and b <= end and b > a)


@pytest.mark.parametrize("max_tokens,overlap", [(200, 0), (50, 10), (7, 2), (3, 1)])
def test_chunk_by_tokens_budget_and_overlap(impl, tok, corpus, max_tokens, overlap):
    chunks = _chunk(impl, tok, corpus, max_tokens, overlap)
    assert chunks and all(c for c in chunks)
    assert chunks[0] == corpus[: len(chunks[0])]          # 처음부터 시작
    assert corpus.endswith(chunks[-1])                    # 끝까지 담음
    assert all("�" not in c for c in chunks)          # 깨진 글자 없음
    pos = 0
    for c in chunks:
        i = corpus.index(c, max(0, pos - len(c)))          # 원문의 연속 구간이고 앞으로만 나아감
        assert i >= pos - len(c)
        pos = i + len(c)
    # 예산: 원문 전체 인코딩 기준 토큰 수. 단, 글자를 쪼개지 않고는 자를 수 없는 최소 덩어리가 예산보다 길면 그 길이까지 허용.
    offsets = tok.encode(corpus).offsets
    cuts = [i for i in range(len(offsets) + 1) if impl.is_char_boundary(offsets, i)]
    limit = max(max_tokens, max(b - a for a, b in itertools.pairwise(cuts)))
    assert all(_token_spans(tok, corpus, c) <= limit for c in chunks)


def test_chunk_by_tokens_no_overlap_reconstructs(impl, tok, corpus):
    chunks = _chunk(impl, tok, corpus, 37, 0)
    assert "".join(chunks) == corpus                       # 겹침 0 이면 이어 붙여 원문 복원(글자 중복·누락 없음)


def test_chunk_by_tokens_overlap_increases_total(impl, tok, corpus):
    no = _chunk(impl, tok, corpus, 100, 0)
    ov = _chunk(impl, tok, corpus, 100, 25)
    assert len(ov) > len(no)
    assert sum(len(c) for c in ov) > len(corpus)


def test_chunk_by_tokens_edge_cases(impl, tok):
    assert _chunk(impl, tok, "", 10, 2) == []
    assert _chunk(impl, tok, "hello", 100, 0) == ["hello"]
    assert "".join(_chunk(impl, tok, "토토토 😀", 1, 0)) == "토토토 😀"  # 글자 하나가 예산보다 커도 멈추고 글자를 쪼개지 않음
    for bad in [(0, 0), (5, 5), (5, -1)]:
        with pytest.raises(ValueError):
            impl.chunk_by_tokens(tok, "hello world", *bad)


# --- TODO 5 ---------------------------------------------------------------
def test_fit_to_context(impl, tok, corpus):
    short = "짧은 글"
    assert impl.fit_to_context(tok, short, context_window=100, reserved_tokens=10) == short
    for budget in (1, 2, 3, 50, 700):
        out = impl.fit_to_context(tok, corpus, context_window=budget + 300, reserved_tokens=300)
        assert corpus.startswith(out)
        assert _token_spans(tok, corpus, out) <= budget if out else True
        assert "�" not in out
    out = impl.fit_to_context(tok, corpus, context_window=1000, reserved_tokens=300)
    assert 690 <= _token_spans(tok, corpus, out) <= 700     # 예산을 거의 다 쓴다(글자 경계 때문에 몇 토큰 덜 쓸 수 있음)
    with pytest.raises(ValueError):
        impl.fit_to_context(tok, corpus, context_window=100, reserved_tokens=100)


# --- (선택 스크립트) Claude 토큰 세기: 가짜 클라이언트 ---------------------------------
def test_count_claude_tokens_with_fake_client():
    spec = importlib.util.spec_from_file_location("p1_01_count_with_claude", LAB_DIR / "count_with_claude.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    calls = []

    class FakeMessages:
        def count_tokens(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(input_tokens=42)

    client = SimpleNamespace(messages=FakeMessages())
    assert mod.count_claude_tokens(client, "claude-opus-5-5", "안녕") == 42
    assert calls == [{"model": "claude-opus-5-5", "messages": [{"role": "user", "content": "안녕"}]}]

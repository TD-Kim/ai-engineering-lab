"""P1-01 Tokenizer — 정답 코드 (메인 실습: 서비스 관점).

토큰 수를 세고(TODO 1), 한국어와 영어를 비교하고(TODO 2), 비용을 추정하고(TODO 3),
토큰 예산으로 문서를 청크로 나누고(TODO 4), 컨텍스트 창에 맞게 자른다(TODO 5).

토크나이저는 다운로드 없이 저장소 말뭉치(data/corpus_ko_en.txt)로 작은 BPE 를 직접 학습해 쓴다(train_local_tokenizer, 제공).
실제 서비스에서는 모델마다 토크나이저가 다르므로, Claude 비용은 count_with_claude.py(messages.count_tokens)로,
오픈 모델은 pretrained_tokenizer.py 로 그 모델의 토크나이저를 써서 센다(둘 다 사용자 PC 실행용 선택 스크립트).

진행 확인: python part1_ai_fundamentals/tokenizer/solution.py
원리(BPE 직접 구현)는 선택 과제: part1_ai_fundamentals/tokenizer/advanced/
"""
from __future__ import annotations

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

# 개념: 토큰 단가(USD / 100만 토큰). API 는 "글자 수"가 아니라 "토큰 수"로 요금을 매기고, 입력과 출력 단가가 다르다.
#       출력 단가가 입력의 5 배다. 토큰 하나를 처리하는 연산량은 입력·출력이 비슷하지만,
#       입력은 한 번에 병렬로 처리하고(prefill) 출력은 앞 토큰이 나와야 다음 토큰을 만드는 식으로 한 토큰씩 순서대로 생성한다(decode).
#       그래서 같은 하드웨어에서 출력 쪽 처리량(초당 토큰)이 훨씬 낮고, 그만큼 토큰당 가격이 비싸다.
# 서비스 관점: 같은 질문이라도 모델 선택에 따라 비용이 수십 배 차이 난다. 단가는 바뀌므로 코드에 박지 말고 설정으로 관리한다.
# REF: https://platform.claude.com/docs/en/about-claude/pricing  (작성 시점 2026-10 기준, 사용 전 공식 페이지에서 재확인)
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.00, 20.00),    # (입력, 출력)
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),    # 프롬프트 10만 토큰 이하 기준(그 이상은 단가가 다름)
}


# ---------------------------------------------------------------------------
# (제공) 다운로드 없이 쓰는 작은 토크나이저
# ---------------------------------------------------------------------------
def train_local_tokenizer(text: str, vocab_size: int = 500) -> Tokenizer:
    """말뭉치로 바이트 단위 BPE 토크나이저를 학습한다(Hugging Face `tokenizers` 라이브러리).

    개념: BPE(Byte Pair Encoding)는 "자주 붙어 나오는 조각을 하나로 합치기"를 반복해 어휘를 만든다.
          바이트 단위(byte-level)로 시작하면 어떤 글자든(한글, 이모지) 바이트로는 항상 표현되므로 <unk>(모르는 토큰)가 생기지 않는다.
          대신 한글 한 글자는 UTF-8 로 3 바이트라서, 학습 데이터에 자주 나오지 않은 글자는 토큰 2~3 개로 쪼개진다.
    REF: Sennrich et al., 2016, "Neural Machine Translation of Rare Words with Subword Units", §3.2
         https://huggingface.co/docs/tokenizers (공식 문서)
    """
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # 256 개 바이트를 기본 어휘로
        show_progress=False,
    )
    tok.train_from_iterator([text], trainer=trainer)
    return tok


# ---------------------------------------------------------------------------
# TODO 1 — 토큰 수 세기
# ---------------------------------------------------------------------------
def count_tokens(tokenizer: Tokenizer, text: str) -> int:
    """text 의 토큰 수."""
    # TODO 1
    # 개념: tokenizer.encode(text) 는 Encoding 객체를 돌려준다. .ids 는 토큰 번호 목록, .tokens 는 토큰 문자열 목록,
    #       .offsets 는 각 토큰이 원문의 몇 번째 글자부터 몇 번째 글자 앞까지인지 (시작, 끝) 목록이다.
    #       토큰 수 = 번호 목록의 길이.
    # WHY: 사람은 글자·단어로 길이를 재지만, 모델과 요금은 토큰으로 잰다. 같은 길이의 글도 언어에 따라 토큰 수가 다르다(TODO 2).
    # 서비스 관점: 요청 전에 토큰 수를 알아야 비용 추정(TODO 3), 청크 크기 결정(TODO 4), 컨텍스트 초과 방지(TODO 5)가 가능하다.
    return len(tokenizer.encode(text).ids)


# ---------------------------------------------------------------------------
# TODO 2 — 한국어와 영어 비교
# ---------------------------------------------------------------------------
def compare_languages(tokenizer: Tokenizer, samples: dict[str, str]) -> list[dict]:
    """언어별 문자 수 / 토큰 수 비교. 반환: [{"name", "chars", "tokens", "chars_per_token"}, ...] (입력 순서 유지)"""
    # TODO 2
    rows = []
    for name, text in samples.items():
        chars = len(text)                       # 파이썬 len(문자열) = 글자(유니코드 문자) 수. 공백·구두점도 1 글자.
        tokens = count_tokens(tokenizer, text)
        rows.append({
            "name": name,
            "chars": chars,
            "tokens": tokens,
            # 개념: 토큰 하나가 평균 몇 글자를 담는가. 클수록 "압축이 잘 된다" = 같은 글을 적은 토큰(=적은 비용)으로 표현.
            # 0 으로 나누기 방지: 빈 문자열이면 토큰도 0 개라 0.0 으로 둔다.
            "chars_per_token": round(chars / tokens, 2) if tokens else 0.0,
        })
    # WHY: 토크나이저는 학습 데이터에 많이 나온 언어를 더 크게 묶는다. 영어 위주로 학습된 토크나이저에서는
    #      한국어가 같은 내용이라도 토큰이 더 많이 나오는 경우가 흔하다 — 측정해서 확인한다(가정하지 않는다).
    # 서비스 관점: 한국어 서비스는 영어 기준 예산표로 비용을 잡으면 과소 추정될 수 있다. 실제 쓰는 모델의 토크나이저로 재야 한다.
    return rows


# ---------------------------------------------------------------------------
# TODO 3 — 비용 추정
# ---------------------------------------------------------------------------
def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str,
    requests: int = 1,
    prices: dict[str, tuple[float, float]] = PRICES_PER_MTOK,
) -> float:
    """요청 requests 번의 예상 비용(USD). 단가는 100만(1,000,000) 토큰당 가격."""
    # TODO 3
    if model not in prices:
        # 모르는 모델로 0 원을 돌려주면 비용이 조용히 빠진 보고서가 된다 → 바로 알린다.
        raise KeyError(f"unknown model: {model!r}")
    price_in, price_out = prices[model]
    # 개념: 단가가 "100만 토큰당"이므로 토큰 수 / 1,000,000 × 단가. 예) 입력 2,000 토큰 × $4 / 1M = $0.008
    per_request = input_tokens / 1_000_000 * price_in + output_tokens / 1_000_000 * price_out
    # 서비스 관점: 1 건은 싸 보여도 하루 요청 수를 곱하면 커진다. RAG 는 검색한 문서를 입력에 넣으므로 입력 토큰이 크게 늘어난다.
    #             프롬프트 캐싱(반복되는 앞부분 재사용)·배치 처리로 줄일 수 있다(Part 2 P2-03).
    return per_request * requests


# ---------------------------------------------------------------------------
# (제공) 글자 경계 확인 — TODO 4, 5 에서 사용
# ---------------------------------------------------------------------------
def is_char_boundary(offsets: list[tuple[int, int]], i: int) -> bool:
    """토큰 i 앞에서 잘라도 글자가 쪼개지지 않으면 True. i == 0 또는 i == len(offsets) 는 항상 True.

    개념: 바이트 단위 BPE 에서는 한글 한 글자(3 바이트)가 토큰 2~3 개로 나뉠 수 있다. 이때 그 토큰들은
          offsets(원문 글자 위치)가 같은 글자를 가리킨다. 예) '토' 한 글자 → 토큰 3 개, offsets 모두 (0, 1).
          토큰 i 의 시작 위치가 앞 토큰의 끝 위치보다 앞이면(겹치면) 같은 글자의 일부이므로 그 사이에서 자르면 안 된다.
    """
    return i == 0 or i >= len(offsets) or offsets[i][0] >= offsets[i - 1][1]


def snap_chunk_end(offsets: list[tuple[int, int]], start: int, end: int) -> int:
    """조각 [start, end) 의 끝 end 를 글자 경계로 맞춘다. 반환값은 항상 start 보다 크다.

    개념: end 가 글자 중간이면 한 칸씩 앞으로 당긴다(예산을 넘지 않게).
          당기다가 start 까지 와 버리면 = 글자를 쪼개지 않고는 자를 곳이 예산 안에 없다
          (예: 한 글자가 토큰 3 개인데 예산 2). 이때는 뒤로 늘려 그 최소 덩어리를 통째로 담는다 → 이 조각만 예산을 넘는다.
    WHY: start 보다 큰 값을 보장해야 조각이 비지 않고 반복이 반드시 앞으로 나아간다.
    """
    end = min(end, len(offsets))
    cut = end
    while cut > start and not is_char_boundary(offsets, cut):
        cut -= 1
    if cut > start:
        return cut
    cut = start + 1
    while not is_char_boundary(offsets, cut):
        cut += 1
    return cut


def snap_next_start(offsets: list[tuple[int, int]], start: int, end: int, overlap: int) -> int:
    """다음 조각의 시작 = end - overlap 을 글자 경계로 맞춘 값. 반환값은 항상 start 보다 크다.

    개념: 겹칠 부분의 시작이 글자 중간이면 앞으로 당긴다(겹침이 조금 늘어날 수 있음).
          당기다가 이전 조각의 start 이하가 되면 앞으로 나아가지 못해 같은 조각을 영원히 반복하게 되므로,
          그때는 겹치지 않고 end 에서 이어 간다.
    """
    nxt = end - overlap
    while nxt > start and not is_char_boundary(offsets, nxt):
        nxt -= 1
    return nxt if nxt > start else end


# ---------------------------------------------------------------------------
# TODO 4 — 토큰 예산으로 청크 나누기 (오버랩 포함)
# ---------------------------------------------------------------------------
def chunk_by_tokens(tokenizer: Tokenizer, text: str, max_tokens: int, overlap: int = 0) -> list[str]:
    """text 를 토큰 max_tokens 개 이하의 조각으로 나눈다. 이웃 조각은 토큰 약 overlap 개를 겹친다.

    토큰 수는 "원문 전체를 한 번 인코딩한 결과" 기준이다. 조각만 따로 다시 인코딩하면 경계에서 몇 토큰 달라질 수 있다.
    글자를 쪼개지 않고는 자를 곳이 없을 때만 그 최소 덩어리 하나로 된 조각이 예산을 넘는다(snap_chunk_end).
    """
    if max_tokens <= 0 or not 0 <= overlap < max_tokens:
        # overlap >= max_tokens 면 다음 조각 시작점이 앞으로 가지 않는다.
        raise ValueError("need max_tokens > 0 and 0 <= overlap < max_tokens")
    offsets = tokenizer.encode(text).offsets  # [(시작 글자 위치, 끝 글자 위치), ...] 토큰마다 하나
    # TODO 4
    # 개념: 토큰 번호 start ~ end(포함 안 함)가 한 조각. 다음 조각은 end - overlap 에서 시작 → 끝 overlap 토큰이 다시 들어간다.
    chunks: list[str] = []
    start = 0
    while start < len(offsets):
        end = snap_chunk_end(offsets, start, start + max_tokens)  # 최대 max_tokens 개, 글자 경계로 맞춤
        # 원문을 글자 위치로 자른다. 글자 경계를 확인하지 않고 토큰 번호 일부를 decode 하면 한글이 깨질 수 있다(�).
        chunks.append(text[offsets[start][0] : offsets[end - 1][1]])
        if end == len(offsets):  # 마지막 토큰까지 담았으면 끝(꼬리만 겹치는 조각 방지)
            break
        start = snap_next_start(offsets, start, end, overlap)  # overlap 만큼 뒤로, 항상 앞으로 나아감
    # WHY(오버랩): 문장이 조각 경계에서 잘리면 앞뒤 문맥이 사라진다. 조금 겹치면 검색된 조각만으로도 뜻이 이어진다.
    # 서비스 관점: 조각이 작으면 검색은 정밀하지만 문맥이 부족하고, 크면 프롬프트 토큰(=비용·지연)이 늘어난다.
    #             오버랩은 저장·임베딩 비용을 대략 max_tokens / (max_tokens - overlap) 배 늘린다. Part 3 P3-02 에서 수치로 비교한다.
    #             조각을 따로 다시 세면 토큰 수가 조금 달라지므로, 임베딩 모델 한도에 딱 맞추지 말고 여유를 둔다.
    return chunks


# ---------------------------------------------------------------------------
# TODO 5 — 컨텍스트 창에 맞게 자르기
# ---------------------------------------------------------------------------
def fit_to_context(tokenizer: Tokenizer, text: str, context_window: int, reserved_tokens: int) -> str:
    """context_window 에서 reserved_tokens(시스템 프롬프트·질문·답변용 몫)를 뺀 예산만큼 text 의 앞부분을 남긴다."""
    # TODO 5
    # 개념: 컨텍스트 창(context window)은 모델이 한 번에 볼 수 있는 토큰 수의 상한이다. 입력 + 출력이 모두 이 안에 들어가야 한다.
    # WHY: 출력 몫을 남기지 않고 입력을 꽉 채우면, 답이 중간에 끊기거나 요청이 거부된다.
    # 서비스 관점: 무작정 자르면 중요한 내용이 잘릴 수 있다. 실제 서비스는 자르기 전에 "무엇을 넣을지"를 고른다
    #             (검색 상위 문서만 넣기 = RAG, 요약해서 넣기). 자른 사실은 로그로 남겨야 품질 문제를 추적할 수 있다.
    budget = context_window - reserved_tokens
    if budget <= 0:
        raise ValueError("reserved_tokens must be smaller than context_window")
    offsets = tokenizer.encode(text).offsets
    if len(offsets) <= budget:
        return text  # 이미 들어가면 그대로
    end = budget  # 앞에서부터 토큰 budget 개(번호 0 ~ budget-1)
    while end > 0 and not is_char_boundary(offsets, end):
        end -= 1  # 글자 중간이면 그 글자는 통째로 뺀다(예산을 넘지 않게)
    return text[: offsets[end - 1][1]] if end > 0 else ""


# ---------------------------------------------------------------------------
# 단독 실행: 어느 TODO 가 남았는지 알려 준다.
# ---------------------------------------------------------------------------
def _report_progress() -> None:
    tok = train_local_tokenizer("hello world 안녕하세요 " * 20, vocab_size=300)
    checks = [
        ("TODO 1 count_tokens", lambda: count_tokens(tok, "hello")),
        ("TODO 2 compare_languages", lambda: compare_languages(tok, {"en": "hello"})),
        ("TODO 3 estimate_cost", lambda: estimate_cost(100, 10, "claude-opus-5-5")),
        ("TODO 4 chunk_by_tokens", lambda: chunk_by_tokens(tok, "hello world " * 5, 4, 1)),
        ("TODO 5 fit_to_context", lambda: fit_to_context(tok, "hello world " * 5, 10, 4)),
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
        print(f"\n남은 TODO {remaining}개 — 위에서부터 순서대로 채우세요(TODO 2 는 TODO 1 을 사용합니다). 채운 뒤 검사:")
        print("  (bash)       IMPL=starter pytest part1_ai_fundamentals/tokenizer/tests")
        print('  (PowerShell) $env:IMPL="starter"; pytest part1_ai_fundamentals/tokenizer/tests; Remove-Item Env:IMPL')
    else:
        print("\n모든 TODO 가 구현되었습니다. 이제 pytest 로 정답 여부를 확인하세요.")


if __name__ == "__main__":
    _report_progress()

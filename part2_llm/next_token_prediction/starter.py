"""P2-01 다음 토큰 예측과 샘플링 — 실습 코드 (TODO 를 채우세요). 메인 실습: 개념은 깊게, 구현은 얕게.

LLM 은 "다음 토큰의 점수(logits)"를 내는 함수다. 그 점수를 확률로 바꾸고(TODO 1), 후보를 줄이고(TODO 2 top-k, TODO 3 top-p),
하나를 고르고(TODO 4), 이것을 반복해 문장을 만든다(TODO 5, 컨텍스트 창 안에서).

모델: 테스트·데모는 다운로드 없이 쓰는 작은 단어 bigram 언어 모델(BigramLM, 제공). "직전 단어 하나"만 보고 다음 단어 점수를 낸다.
실제 LLM(Qwen2.5-0.5B)으로 같은 함수를 돌려 보는 것은 선택 스크립트 run_qwen.py (사용자 PC, 모델 다운로드 필요).

검사: IMPL=starter pytest part2_llm/next_token_prediction/tests
진행 확인: python part2_llm/next_token_prediction/starter.py   (남은 TODO 목록 출력)
"""
from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from pathlib import Path

import torch

CORPUS = Path(__file__).resolve().parents[2] / "part1_ai_fundamentals" / "data" / "corpus_ko_en.txt"
EOS = "<eos>"

# 모델의 모양: 지금까지의 토큰 번호 목록 → 다음 토큰 점수 (V,)  (V = 어휘 크기)
ModelFn = Callable[[list[int]], torch.Tensor]


# ---------------------------------------------------------------------------
# (제공) 다운로드 없이 쓰는 작은 언어 모델
# ---------------------------------------------------------------------------
class BigramLM:
    """단어 bigram 언어 모델: logits[다음] = log(count(직전 → 다음) + alpha).

    개념: "the" 다음에 "cat" 이 3 번, "dog" 이 2 번 나왔다면 the → cat 점수가 더 높다. 학습 = 세기, 예측 = 센 값 보기.
    한계: 직전 단어 하나만 본다. 실제 LLM 은 컨텍스트 창 안의 모든 토큰을 어텐션(P1-05)으로 보고 점수를 낸다.
    alpha: 한 번도 안 나온 조합도 0 이 아닌 작은 점수를 받게 하는 스무딩(log(0) = -inf 방지).
           크게 잡으면(예: 0.1) 어휘 346 개 전체에 확률이 퍼져, 샘플링하면 엉뚱한 단어가 자주 뽑힌다(README §8 [4]).
    """

    def __init__(self, sentences: Sequence[Sequence[str]], alpha: float = 0.001):
        words = sorted({w for s in sentences for w in s})
        self.vocab: list[str] = [*words, EOS]
        self.index = {w: i for i, w in enumerate(self.vocab)}
        v = len(self.vocab)
        counts = torch.zeros(v, v)
        for s in sentences:
            ids = [self.index[w] for w in s] + [self.index[EOS]]
            for a, b in itertools.pairwise(ids):
                counts[a, b] += 1
        self.logits = torch.log(counts + alpha)

    def __call__(self, ids: list[int]) -> torch.Tensor:
        return self.logits[ids[-1]].clone()

    def encode(self, text: str) -> list[int]:
        unknown = [w for w in text.lower().split() if w not in self.index]
        if unknown:
            raise KeyError(f"어휘에 없는 단어(OOV): {unknown}")
        return [self.index[w] for w in text.lower().split()]

    def decode(self, ids: Sequence[int]) -> str:
        return " ".join(self.vocab[i] for i in ids)


def load_bigram_lm(path: Path = CORPUS, alpha: float = 0.001) -> BigramLM:
    """(제공) Part 1 말뭉치(48줄)를 소문자·구두점 제거·공백 분리해 BigramLM 을 만든다."""
    import re

    lines = path.read_text(encoding="utf-8").splitlines()
    sentences = [re.sub(r"[^\w\s]", " ", ln.lower()).split() for ln in lines if ln.strip()]
    return BigramLM(sentences, alpha=alpha)


# ---------------------------------------------------------------------------
# TODO 1 — temperature 를 적용한 softmax
# ---------------------------------------------------------------------------
def softmax_with_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """logits (V,) → 확률 (V,). temperature 는 0 보다 커야 한다."""
    # TODO 1: temperature <= 0 이면 ValueError. 아니면 logits 를 temperature 로 나눈 뒤 softmax(마지막 축) 확률을 반환하세요.
    # 개념: logits 는 아직 확률이 아닌 점수. softmax(z)_i = e^(z_i) / Σ_j e^(z_j) 로 합이 1 인 확률이 된다.
    #       예) 점수 [2, 1, 0]:  T = 1 → [0.665, 0.245, 0.090],  T = 0.5 → [0.867, 0.117, 0.016],  T = 2 → [0.506, 0.307, 0.186]
    #       순서(1 등)는 그대로, "얼마나 1 등에 몰리는지"만 바뀐다.
    # HINT 1: 나눗셈 한 번, softmax 한 번입니다. 직접 exp 를 쓰기보다 라이브러리 softmax 가 안전합니다(넘침 방지).
    # HINT 2: torch.softmax(logits / temperature, dim=-1)
    # WHY(T > 0): T = 0 은 0 으로 나누기. "항상 1 등"은 greedy(TODO 4)로 따로 처리한다.
    # 서비스 관점: 낮은 T = 일관된 답(분류·추출·코드), 높은 T = 다양한 답(아이디어·창작).
    raise NotImplementedError("TODO 1")


# ---------------------------------------------------------------------------
# TODO 2 — top-k 필터
# ---------------------------------------------------------------------------
def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """점수 상위 k 개만 남기고 나머지를 -inf 로 바꾼 새 logits. k 가 V 이상이면 그대로."""
    # TODO 2: k <= 0 이면 ValueError. k 가 후보 수(V) 이상이면 logits 복사본을 그대로, 아니면 k 번째로 큰 점수보다 작은 칸을 -inf 로 바꾸세요.
    # 개념: 확률이 아주 낮은 후보도 샘플링하면 가끔 뽑힌다. 상위 k 개 밖은 후보에서 뺀다.
    #       예) [2, 1, 0, -1], k = 2 → [2, 1, -inf, -inf] → softmax [0.73, 0.27, 0, 0]
    # HINT 1: 먼저 "k 번째로 큰 점수"를 구하고, 그보다 작은 칸을 지웁니다(동점은 남김).
    # HINT 2: torch.topk(logits, k).values[-1], logits.masked_fill(조건, float("-inf"))
    # WHY(-inf): e^(-inf) = 0 이라 softmax 뒤 확률이 정확히 0 (P1-05 마스크와 같은 방법).
    # 서비스 관점: k 가 고정이라 확실할 때도 애매할 때도 같은 수를 남긴다 → top-p(TODO 3)의 동기.
    raise NotImplementedError("TODO 2")


# ---------------------------------------------------------------------------
# TODO 3 — top-p (nucleus) 필터
# ---------------------------------------------------------------------------
def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """확률이 큰 순서로 더해 누적 확률이 p 에 처음 도달할 때까지의 후보만 남긴다(최소 1 개). 나머지는 -inf."""
    # TODO 3: p 가 (0, 1] 밖이면 ValueError. 확률을 큰 순서로 정렬해 "이 후보를 더하기 전까지의 누적 확률 < p" 인 후보만 남기고,
    #         나머지 칸을 -inf 로 바꾼 logits 를 반환하세요(원래 토큰 순서 유지).
    # 개념: 확률 큰 순서 [0.50, 0.30, 0.15, 0.05], p = 0.7 → 더하기 전 누적 [0, 0.50, 0.80, 0.95] → < 0.7 인 앞의 2 개만 남김.
    #       1 등은 더하기 전 누적이 0 이라 항상 남는다. 후보 수가 분포 모양에 따라 바뀐다(top-k 와 차이).
    # HINT 1: 판정은 정렬된 순서에서 하고, 결과(남김 여부)는 원래 위치로 되돌려 적어야 합니다.
    # HINT 2: torch.sort(probs, descending=True) → (값, 원래 위치), torch.cumsum, torch.zeros_like(keep).scatter(-1, order, keep)
    # REF: Holtzman et al., 2020, "The Curious Case of Neural Text Degeneration", §3.1 (nucleus sampling)
    # 서비스 관점: p 가 작으면 보수적·일관, 크면 다양. 보통 temperature 와 함께 쓴다.
    raise NotImplementedError("TODO 3")


# ---------------------------------------------------------------------------
# TODO 4 — 다음 토큰 하나 고르기
# ---------------------------------------------------------------------------
def sample_next(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> int:
    """temperature == 0 이면 greedy(가장 큰 점수). 아니면 temperature → top-k → top-p 순으로 거른 뒤 확률대로 하나 뽑는다."""
    # TODO 4: temperature < 0 이면 ValueError, == 0 이면 가장 큰 점수의 번호(greedy).
    #         아니면 ① logits / T ② top_k 가 있으면 TODO 2 ③ top_p 가 있으면 TODO 3 ④ softmax ⑤ 확률대로 1 개 뽑아 int 로 반환.
    # 개념: greedy 는 매번 1 등 → 같은 입력이면 같은 출력(결정적), 대신 되풀이하기 쉽다. sampling 은 확률대로 → 매번 다를 수 있다.
    # HINT 1: 뽑기는 "확률 목록에서 1 개"를 고르는 함수를 쓰고, generator 를 넘겨 재현 가능하게 합니다.
    # HINT 2: torch.argmax(logits), torch.multinomial(probs, 1, generator=generator)
    # 서비스 관점: 고객 응대·평가 자동화는 재현성, 브레인스토밍은 다양성이 중요하다.
    raise NotImplementedError("TODO 4")


# ---------------------------------------------------------------------------
# TODO 5 — 생성 루프 (컨텍스트 창 안에서)
# ---------------------------------------------------------------------------
def generate(
    model: ModelFn,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    context_window: int,
    eos_id: int | None = None,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> list[int]:
    """prompt_ids 뒤에 토큰을 최대 max_new_tokens 개 생성해 새 토큰만 돌려준다. eos_id 가 나오면 멈춘다(eos 포함).

    모델에는 매 단계 "마지막 context_window 개 토큰"만 넣는다.
    """
    # TODO 5: max_new_tokens < 0 이거나 context_window <= 0 이면 ValueError, prompt_ids 가 비면 ValueError.
    #         최대 max_new_tokens 번: 마지막 context_window 개 토큰만 모델에 넣어 logits → TODO 4 로 다음 토큰 → 입력 끝에 붙이기.
    #         eos_id 가 나오면(포함해서) 멈춥니다. 새로 만든 토큰 목록을 반환하세요.
    # 개념: "the cat" → sat → "the cat sat" → on → ... (자기회귀 생성). 창을 넘는 앞부분은 모델이 "잊는다".
    # HINT 1: 반복은 횟수가 정해진 for 로 하세요(eos 가 안 나와도 반드시 끝나게).
    # HINT 2: for _ in range(max_new_tokens): logits = model(ids[-context_window:]) ...
    # WHY(최대 길이): API 의 max_tokens 와 같은 역할. 출력 토큰 = 비용이므로 상한은 비용 상한이기도 하다.
    # 서비스 관점: 출력 길이에 비례해 시간이 든다(P1-05). 입력이 길수록 첫 토큰이 늦다(TTFT).
    raise NotImplementedError("TODO 5")


def entropy_bits(probs: torch.Tensor) -> float:
    """(제공) 확률 분포의 엔트로피(비트). 0 = 한 후보에 확실, 클수록 고르게 퍼짐. 데모에서 temperature 효과를 숫자로 본다."""
    p = probs[probs > 0]
    return float(-(p * torch.log2(p)).sum())


# ---------------------------------------------------------------------------
# 단독 실행: 어느 TODO 가 남았는지 알려 준다.
# ---------------------------------------------------------------------------
def _report_progress() -> None:
    z = torch.tensor([2.0, 1.0, 0.0])
    lm = BigramLM([["a", "b"], ["a", "c"]])
    checks = [
        ("TODO 1 softmax_with_temperature", lambda: softmax_with_temperature(z, 1.0)),
        ("TODO 2 top_k_filter", lambda: top_k_filter(z, 2)),
        ("TODO 3 top_p_filter", lambda: top_p_filter(z, 0.7)),
        ("TODO 4 sample_next", lambda: sample_next(z, temperature=0)),
        ("TODO 5 generate", lambda: generate(lm, [0], 2, context_window=4)),
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
        print(f"\n남은 TODO {remaining}개 — 위에서부터 순서대로 채우세요(TODO 4 는 TODO 2·3, TODO 5 는 TODO 4 를 사용합니다). 채운 뒤 검사:")
        print("  (bash)       IMPL=starter pytest part2_llm/next_token_prediction/tests")
        print('  (PowerShell) $env:IMPL="starter"; pytest part2_llm/next_token_prediction/tests; Remove-Item Env:IMPL')
    else:
        print("\n모든 TODO 가 구현되었습니다. 이제 pytest 로 정답 여부를 확인하세요.")


if __name__ == "__main__":
    _report_progress()

"""P1-05 Transformer — 실습 코드 (TODO 를 채우세요). 메인 실습: 어텐션 개념은 깊게, 구현은 한 번만.

Q·K·V 를 만들고(TODO 1), 스케일드 닷-프로덕트 어텐션을 한 번 계산하고(TODO 2), causal 마스크로 미래를 가리고(TODO 3),
시퀀스 길이 T 에 따라 점수 행렬(T×T)과 시간이 어떻게 늘어나는지 재고(TODO 4), (선택) KV 캐시로 생성 계산을 줄인다(TODO 5).
멀티헤드·FFN·Residual·LayerNorm·위치 인코딩·블록 쌓기는 README 의 구조 그림과 설명으로 다룬다(직접 구현은 advanced/).

마스크 규약(이 실습 전체): bool 텐서, **True = 볼 수 있음(참여)**, False = 가림.
  - torch.nn.functional.scaled_dot_product_attention 의 bool attn_mask 와 같은 규약이다.
  - torch.nn.MultiheadAttention 의 key_padding_mask 는 반대(True = 무시)이니 섞어 쓸 때 주의.

검사: IMPL=starter pytest part1_ai_fundamentals/transformer/tests
진행 확인: python part1_ai_fundamentals/transformer/starter.py   (남은 TODO 목록 출력)
"""
from __future__ import annotations

import math  # noqa: F401  (TODO 2 에서 사용)
import time  # noqa: F401  (TODO 4 에서 사용)
from collections.abc import Sequence

import torch


# ---------------------------------------------------------------------------
# TODO 1 — Q·K·V 만들기
# ---------------------------------------------------------------------------
def make_qkv(
    x: torch.Tensor, w_q: torch.Tensor, w_k: torch.Tensor, w_v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """x: (T, D), w_*: (D, d_k) → Q, K, V 각각 (T, d_k). 앞에 배치 차원이 있어도 된다((B, T, D))."""
    # TODO 1: x 에 세 가중치를 각각 곱해 Q, K, V 를 반환하세요.
    # 개념(비유): Q = "나는 무엇을 찾나"(질문), K = "나는 어떤 내용인가"(색인표), V = "나를 고르면 가져갈 내용".
    #   질문과 열쇠가 잘 맞을수록(내적이 클수록) 그 토큰의 내용을 많이 가져온다.
    # 개념(행렬곱): (T, D) @ (D, d_k) → (T, d_k). 토큰 T 개에 같은 가중치를 한 번에 곱한다(P1-04 linear_forward 와 같음).
    # HINT 1: 가중치 모양이 (D, d_k) 이므로 전치 없이 곱합니다(nn.Linear 의 (out, in) 규약과 다름).
    # HINT 2: x @ w_q
    # WHY(셋으로 나누기): 같은 벡터를 그대로 쓰면 "무엇을 찾나"와 "무엇을 주나"가 같아야 한다. 세 가중치로 따로 학습한다.
    # REF: Vaswani et al., 2017, "Attention Is All You Need", §3.2
    raise NotImplementedError("TODO 1")


# ---------------------------------------------------------------------------
# TODO 2 — 스케일드 닷-프로덕트 어텐션 1회
# ---------------------------------------------------------------------------
def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """q: (T_q, d_k), k: (T_k, d_k), v: (T_k, d_v), mask: (T_q, T_k) bool(True=볼 수 있음).

    반환: (출력 (T_q, d_v), 어텐션 가중치 (T_q, T_k)). 앞에 배치 차원이 있어도 된다.
    """
    # TODO 2: scores = Q @ Kᵀ / √d_k → mask 가 있으면 False 칸을 -inf 로 → 마지막 축 softmax → 가중치 @ V.
    #         (출력, 가중치) 를 반환하세요.
    # 개념: [i, j] 점수 = 토큰 i 의 질문과 토큰 j 의 열쇠의 내적. softmax 는 한 행을 "합이 1 인 비율"로 바꾼다.
    #   예) 점수 [2, 1, 0] → exp [7.39, 2.72, 1.00] / 합 11.11 → [0.67, 0.24, 0.09]
    #   출력 = 다른 토큰 내용(V)의 가중 평균.
    # HINT 1: Kᵀ 는 마지막 두 축을 바꾼 것입니다(배치 차원이 있어도 되게). d_k 는 q 의 마지막 축 크기.
    # HINT 2: k.transpose(-2, -1), math.sqrt(d_k), scores.masked_fill(~mask, float("-inf")), torch.softmax(scores, dim=-1)
    # WHY(√d_k): d_k 차원 내적은 분산이 d_k 라 값이 커진다 → softmax 가 한 칸에 몰려(포화) 기울기가 사라진다. √d_k 로 나눠 되돌린다.
    #   (Vaswani et al., 2017, §3.2.1 각주 4)
    # 서비스 관점: 점수 행렬이 (T, T) 라 컨텍스트 길이의 제곱만큼 메모리·계산이 든다(TODO 4).
    raise NotImplementedError("TODO 2")


# ---------------------------------------------------------------------------
# TODO 3 — causal 마스크
# ---------------------------------------------------------------------------
def causal_mask(t: int) -> torch.Tensor:
    """(t, t) bool. [i, j] = True ⇔ j <= i (토큰 i 는 자기 자신과 이전 토큰만 볼 수 있음)."""
    # TODO 3: [i, j] = True ⇔ j <= i 인 (t, t) bool 텐서를 반환하세요(아래 삼각형, 대각선 포함).
    # 개념: 생성 모델은 "다음 토큰 맞히기"로 학습한다. 토큰 i 가 미래(i+1 이후)를 보면 정답을 베끼므로 가린다.
    #        j: 0 1 2 3
    #     i=0 [ T F F F ]
    #     i=1 [ T T F F ]
    #     i=2 [ T T T F ]
    #     i=3 [ T T T T ]
    # HINT 1: 모두 True 인 정사각 행렬에서 "아래 삼각형"만 남기세요.
    # HINT 2: torch.ones(t, t, dtype=torch.bool), torch.tril
    # WHY(대각선 포함): 자기 자신은 항상 보여야 한 행이 전부 가려지는 일(NaN)이 없다.
    # 서비스 관점: 이미 만든 토큰의 K·V 는 이후에 바뀌지 않는다 → KV 캐시가 가능한 이유(TODO 5).
    # REF: Vaswani et al., 2017, §3.2.3
    raise NotImplementedError("TODO 3")


# ---------------------------------------------------------------------------
# TODO 4 — 컨텍스트 길이와 비용
# ---------------------------------------------------------------------------
def measure_attention_cost(lengths: Sequence[int], d_k: int = 64, repeats: int = 5, seed: int = 0) -> list[dict]:
    """길이 T 마다 causal 어텐션 1회의 점수 행렬 모양·원소 수·메모리·시간(중앙값 ms)을 잰다.

    반환: [{"T", "scores_shape", "scores_elements", "scores_mb", "ms"}, ...] (lengths 순서)
    """
    # TODO 4: lengths 의 각 T 에 대해 (T, d_k) 무작위 q, k, v 를 g 로 만들고, causal 마스크로 어텐션을 repeats 번 실행해
    #         시간을 재세요. 행마다 {"T", "scores_shape"(가중치 모양 튜플), "scores_elements"(원소 수),
    #         "scores_mb"(원소 수 × 원소 바이트 / 1e6), "ms"(중앙값)} 를 만듭니다. repeats <= 0 이면 ValueError.
    # 개념: 점수 행렬은 (T, T). T 가 2 배면 원소 수는 4 배(T²).
    # HINT 1: 시간은 시작·끝 시각의 차이. 여러 번 잰 값을 정렬해 가운데 값을 씁니다.
    # HINT 2: g = torch.Generator().manual_seed(seed); torch.randn(t, d_k, generator=g); time.perf_counter();
    #         w.numel(), w.element_size()
    # WHY(중앙값): 한 번 잰 시간은 흔들린다. 작은 T 는 고정 비용이 커서 T² 비율이 잘 안 보인다.
    # 서비스 관점: 컨텍스트가 길수록 첫 토큰 지연(TTFT)과 비용이 커진다. RAG 청크를 많이 넣을수록 T 가 커진다.
    raise NotImplementedError("TODO 4")


# ---------------------------------------------------------------------------
# TODO 5 (선택) — KV 캐시
# ---------------------------------------------------------------------------
def decode_with_kv_cache(
    x: torch.Tensor, w_q: torch.Tensor, w_k: torch.Tensor, w_v: torch.Tensor, use_cache: bool = True
) -> tuple[torch.Tensor, int]:
    """토큰을 하나씩 생성하는 상황을 흉내 내어, 매 단계 마지막 토큰의 어텐션 출력을 구한다. x: (T, D)

    use_cache=True : 이전 단계의 K·V 를 저장해 두고, 새 토큰 1 개의 K·V 만 계산해 붙인다.
    use_cache=False: 매 단계 지금까지의 모든 토큰에 대해 K·V 를 처음부터 다시 계산한다.
    반환: (단계별 출력 (T, d_v), K·V 를 계산한 토큰 행 수의 합)
    """
    # TODO 5 (선택): t = 0..T-1 단계마다 마지막 토큰(x[t])의 어텐션 출력을 구해 (T, d_v) 로 모으고,
    #         K·V 를 계산한 토큰 행 수의 합을 함께 반환하세요.
    #         use_cache=True  → 새 토큰 1 개의 K·V 만 계산해 미리 잡아 둔 캐시의 t 번째 행에 쓰고, 0..t 행을 사용(단계마다 1 행)
    #         use_cache=False → x[: t + 1] 전체의 K·V 를 다시 계산(단계마다 t + 1 행)
    # 개념: causal 이라 이전 토큰의 K·V 는 뒤에 토큰이 붙어도 바뀌지 않는다 → 저장해 재사용(KV 캐시).
    # HINT 1: 질문(Q)은 마지막 토큰 것만 필요합니다. 미래 토큰은 아직 없으니 마스크도 필요 없습니다.
    # HINT 2: k_cache = torch.empty(T, d_k) 를 미리 잡고 k_cache[t : t + 1] = x[t : t + 1] @ w_k, 사용은 k_cache[: t + 1].
    #         (목록에 붙여 매번 torch.cat 하면 복사 비용이 T² 로 늘어 캐시 이득이 사라진다)
    # WHY: 캐시는 계산을 줄이는 방법이지 결과를 바꾸면 안 된다 → 한 번에 계산한 causal 어텐션과 같아야 한다.
    # 서비스 관점: 캐시 없음 T(T+1)/2 행 vs 캐시 T 행. 프롬프트 캐싱(Part 2 P2-03)은 같은 앞부분 계산을 요청 사이에 재사용하는 발상이다.
    raise NotImplementedError("TODO 5")


# ---------------------------------------------------------------------------
# 단독 실행: 어느 TODO 가 남았는지 알려 준다.
# ---------------------------------------------------------------------------
def _report_progress() -> None:
    x = torch.ones(3, 4)
    w = torch.eye(4)
    checks = [
        ("TODO 1 make_qkv", lambda: make_qkv(x, w, w, w)),
        ("TODO 2 scaled_dot_product_attention", lambda: scaled_dot_product_attention(x, x, x)),
        ("TODO 3 causal_mask", lambda: causal_mask(3)),
        ("TODO 4 measure_attention_cost", lambda: measure_attention_cost([4], repeats=1)),
        ("TODO 5 decode_with_kv_cache (선택)", lambda: decode_with_kv_cache(x, w, w, w)),
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
        print(f"\n남은 TODO {remaining}개 — 위에서부터 순서대로 채우세요(TODO 4 는 TODO 2·3, TODO 5 는 TODO 2 를 사용합니다). 채운 뒤 검사:")
        print("  (bash)       IMPL=starter pytest part1_ai_fundamentals/transformer/tests")
        print('  (PowerShell) $env:IMPL="starter"; pytest part1_ai_fundamentals/transformer/tests; Remove-Item Env:IMPL')
    else:
        print("\n모든 TODO 가 구현되었습니다. 이제 pytest 로 정답 여부를 확인하세요.")


if __name__ == "__main__":
    _report_progress()

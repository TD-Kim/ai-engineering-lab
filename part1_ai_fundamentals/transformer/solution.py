"""P1-05 Transformer — 정답 코드 (메인 실습: 어텐션 개념은 깊게, 구현은 한 번만).

Q·K·V 를 만들고(TODO 1), 스케일드 닷-프로덕트 어텐션을 한 번 계산하고(TODO 2), causal 마스크로 미래를 가리고(TODO 3),
시퀀스 길이 T 에 따라 점수 행렬(T×T)과 시간이 어떻게 늘어나는지 재고(TODO 4), (선택) KV 캐시로 생성 계산을 줄인다(TODO 5).
멀티헤드·FFN·Residual·LayerNorm·위치 인코딩·블록 쌓기는 README 의 구조 그림과 설명으로 다룬다(직접 구현은 advanced/).

마스크 규약(이 실습 전체): bool 텐서, **True = 볼 수 있음(참여)**, False = 가림.
  - torch.nn.functional.scaled_dot_product_attention 의 bool attn_mask 와 같은 규약이다.
  - torch.nn.MultiheadAttention 의 key_padding_mask 는 반대(True = 무시)이니 섞어 쓸 때 주의.

진행 확인: python part1_ai_fundamentals/transformer/solution.py
"""
from __future__ import annotations

import math
import time
from collections.abc import Sequence

import torch


# ---------------------------------------------------------------------------
# TODO 1 — Q·K·V 만들기
# ---------------------------------------------------------------------------
def make_qkv(
    x: torch.Tensor, w_q: torch.Tensor, w_k: torch.Tensor, w_v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """x: (T, D), w_*: (D, d_k) → Q, K, V 각각 (T, d_k). 앞에 배치 차원이 있어도 된다((B, T, D))."""
    # TODO 1
    # 개념(비유): 도서관에서 책을 찾는 상황.
    #   Q(query, 질문)  — "나는 지금 무엇을 찾고 있나"   (각 토큰이 다른 토큰에게 던지는 질문)
    #   K(key, 열쇠)    — "나는 어떤 내용의 책인가"     (각 토큰이 붙이고 있는 색인표)
    #   V(value, 내용)  — "나를 고르면 가져갈 내용"      (실제로 섞여 들어갈 정보)
    #   질문과 열쇠가 잘 맞을수록(내적이 클수록) 그 토큰의 내용을 많이 가져온다.
    # WHY(셋으로 나누기): 같은 벡터 x 를 그대로 쓰면 "무엇을 찾나"와 "무엇을 주나"가 같아야 한다.
    #   세 개의 다른 가중치 행렬로 바꾸면, 'it' 이 'animal' 을 찾을 때의 기준(K)과 가져올 정보(V)를 따로 학습할 수 있다.
    # 개념(행렬곱 복습): (T, D) @ (D, d_k) → (T, d_k). 토큰 T 개 각각에 같은 가중치를 한 번에 곱한다(P1-04 linear_forward 와 같음).
    #   이 실습은 가중치를 (D, d_k) 모양으로 두어 x @ W 로 곱한다(nn.Linear 는 (out, in) 이라 x @ W.T — 규약 차이).
    # REF: Vaswani et al., 2017, "Attention Is All You Need", §3.2
    return x @ w_q, x @ w_k, x @ w_v


# ---------------------------------------------------------------------------
# TODO 2 — 스케일드 닷-프로덕트 어텐션 1회
# ---------------------------------------------------------------------------
def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """q: (T_q, d_k), k: (T_k, d_k), v: (T_k, d_v), mask: (T_q, T_k) bool(True=볼 수 있음).

    반환: (출력 (T_q, d_v), 어텐션 가중치 (T_q, T_k)). 앞에 배치 차원이 있어도 된다.
    """
    # TODO 2
    # 개념(4 단계): scores = Q @ Kᵀ → scores / √d_k → (마스크) → softmax(행마다) → 가중치 @ V
    #   1) Q @ Kᵀ: (T_q, d_k) @ (d_k, T_k) → (T_q, T_k). [i, j] = 토큰 i 의 질문과 토큰 j 의 열쇠의 내적 = "i 가 j 를 얼마나 보나"의 원점수.
    #      k.transpose(-2, -1): 마지막 두 축을 바꾼다(배치가 있어도 동작). 2차원이면 k.T 와 같다.
    #   2) / √d_k: 아래 WHY.
    #   3) softmax: 한 행의 점수를 "합이 1 인 비율"로 바꾼다. exp 를 쓰므로 큰 점수가 훨씬 큰 비율을 받는다.
    #      예) 점수 [2, 1, 0] → exp [7.39, 2.72, 1.00] / 합 11.11 → [0.67, 0.24, 0.09]
    #   4) 가중치 @ V: (T_q, T_k) @ (T_k, d_v) → (T_q, d_v). 각 토큰의 출력 = 다른 토큰 내용(V)의 가중 평균.
    # WHY(√d_k): 성분이 평균 0·분산 1 인 두 d_k 차원 벡터의 내적은 분산이 d_k 다(값이 √d_k 배 커진다).
    #   d_k=64 면 점수가 ±8 수준이 되어 softmax 가 거의 한 칸에만 1 을 주고(포화), 나머지 칸의 기울기가 0 에 가까워 학습이 멈춘다.
    #   √d_k 로 나누면 분산이 다시 1 수준이 된다. (Vaswani et al., 2017, §3.2.1 각주 4)
    # 서비스 관점: 점수 행렬이 (T_q, T_k) 라서 컨텍스트 길이 T 의 제곱만큼 메모리·계산이 든다(TODO 4).
    d_k = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        # 가린 칸(False)을 -inf 로: exp(-inf) = 0 이라 softmax 비율이 정확히 0 이 된다.
        # 한 행이 전부 False 면 그 행은 모두 -inf → exp 합이 0 → 0/0 = NaN (README §9). causal 마스크는 대각선(자기 자신)이 항상 True 라 안전하다.
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v, weights


# ---------------------------------------------------------------------------
# TODO 3 — causal 마스크
# ---------------------------------------------------------------------------
def causal_mask(t: int) -> torch.Tensor:
    """(t, t) bool. [i, j] = True ⇔ j <= i (토큰 i 는 자기 자신과 이전 토큰만 볼 수 있음)."""
    # TODO 3
    # 개념: GPT 같은 생성 모델은 "다음 토큰 맞히기"로 학습한다. 학습할 때는 문장 전체를 한 번에 넣는데,
    #   토큰 i 가 i+1 번째(정답)를 볼 수 있으면 답을 베껴 맞히게 된다 → 미래 칸을 가린다.
    #   t = 4 일 때 (행 = 보는 토큰 i, 열 = 보이는 토큰 j)
    #        j: 0 1 2 3
    #     i=0 [ T F F F ]
    #     i=1 [ T T F F ]
    #     i=2 [ T T T F ]
    #     i=3 [ T T T T ]     ← 아래 삼각형(대각선 포함) = torch.tril
    # WHY(대각선 포함): 자기 자신은 항상 볼 수 있어야 한 행이 전부 가려지는 일(NaN)이 없다.
    # 서비스 관점: 생성(추론) 때도 같은 규칙이라, 이미 만든 토큰의 K·V 는 이후에 바뀌지 않는다 → KV 캐시가 가능한 이유(TODO 5).
    # REF: Vaswani et al., 2017, §3.2.3 (decoder 의 미래 위치 마스킹)
    return torch.tril(torch.ones(t, t, dtype=torch.bool))


# ---------------------------------------------------------------------------
# TODO 4 — 컨텍스트 길이와 비용
# ---------------------------------------------------------------------------
def measure_attention_cost(lengths: Sequence[int], d_k: int = 64, repeats: int = 5, seed: int = 0) -> list[dict]:
    """길이 T 마다 causal 어텐션 1회의 점수 행렬 모양·원소 수·메모리·시간(중앙값 ms)을 잰다.

    반환: [{"T", "scores_shape", "scores_elements", "scores_mb", "ms"}, ...] (lengths 순서)
    """
    # TODO 4
    # 개념: 점수 행렬은 (T, T). T 가 2 배면 원소 수는 4 배(T²). 계산량도 Q @ Kᵀ 가 T × T × d_k 번 곱셈이라 T² 에 비례한다.
    # WHY(중앙값·반복): 한 번 잰 시간은 다른 프로그램·캐시 상태에 따라 흔들린다. 여러 번 재 중앙값을 쓴다.
    #   작은 T 에서는 함수 호출 자체의 고정 비용이 커서 T² 비율이 잘 안 보인다 → 큰 T 쪽에서 비율을 본다.
    # 서비스 관점: 컨텍스트(프롬프트)가 길수록 첫 토큰이 나오기까지(TTFT) 오래 걸리고 비싸다.
    #   RAG 에서 청크를 많이 넣으면 T 가 커져 지연·비용이 늘고, 관련 없는 청크가 섞여 품질이 떨어질 수도 있다 → 넣을 양을 측정으로 정한다.
    if repeats <= 0:
        raise ValueError("repeats must be > 0")
    g = torch.Generator().manual_seed(seed)
    rows = []
    for t in lengths:
        q, k, v = (torch.randn(t, d_k, generator=g) for _ in range(3))
        mask = causal_mask(t)
        times = []
        for _ in range(repeats):  # 반복 횟수가 정해진 for
            start = time.perf_counter()
            _, w = scaled_dot_product_attention(q, k, v, mask)
            times.append((time.perf_counter() - start) * 1000)
        times.sort()
        rows.append({
            "T": t,
            "scores_shape": tuple(w.shape),
            "scores_elements": w.numel(),
            "scores_mb": w.numel() * w.element_size() / 1e6,
            "ms": times[len(times) // 2],
        })
    return rows


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
    # TODO 5
    # 개념: 생성은 한 토큰씩 순서대로다. 단계 t 에서 새 토큰은 0..t 번 토큰의 K·V 가 필요하다.
    #   causal 마스크 때문에 이전 토큰의 K·V 는 뒤에 토큰이 붙어도 바뀌지 않는다 → 저장해 두고 재사용(KV 캐시).
    #   캐시 없음: 단계마다 1, 2, ..., T 개 행을 다시 계산 → 합 T(T+1)/2.   캐시 있음: 단계마다 1 개 → 합 T.
    # WHY(결과가 같아야 함): 캐시는 "계산을 줄이는 방법"이지 결과를 바꾸면 안 된다 → 한 번에 계산한 causal 어텐션과 같은지 테스트한다.
    # 서비스 관점: KV 캐시 덕분에 토큰당 생성 시간이 길이에 따라 폭증하지 않는다. 대신 캐시 메모리가 T 에 비례해 쌓인다.
    #   API 의 프롬프트 캐싱은 같은 앞부분(시스템 프롬프트·고정 문서)의 이 계산 결과를 요청 사이에 재사용하는 것과 같은 발상이다
    #   → 반복되는 앞부분의 비용·지연이 준다(Part 2 P2-03 에서 Claude API 로 측정).
    t_total = x.shape[0]
    # 캐시는 미리 (T, d) 크기로 잡아 두고 단계마다 한 행씩 채운다. (목록에 붙이며 매번 torch.cat 하면 그 복사 비용이 T² 로 늘어
    # 캐시의 이득을 지워 버린다 — 실제 추론 엔진도 미리 할당한 버퍼를 쓴다.)
    k_cache = torch.empty(t_total, w_k.shape[1], dtype=x.dtype)
    v_cache = torch.empty(t_total, w_v.shape[1], dtype=x.dtype)
    outputs = torch.empty(t_total, w_v.shape[1], dtype=x.dtype)
    rows_computed = 0
    for t in range(t_total):  # 생성 단계: 길이가 정해진 for
        if use_cache:
            new = x[t : t + 1]                           # 새 토큰 1 개만
            k_cache[t : t + 1] = new @ w_k
            v_cache[t : t + 1] = new @ w_v
            rows_computed += 1
            k_all, v_all = k_cache[: t + 1], v_cache[: t + 1]
        else:
            prefix = x[: t + 1]                          # 지금까지 전부 다시
            k_all, v_all = prefix @ w_k, prefix @ w_v
            rows_computed += t + 1
        q_t = x[t : t + 1] @ w_q                         # (1, d_k): 마지막 토큰의 질문만 필요
        out, _ = scaled_dot_product_attention(q_t, k_all, v_all)  # 마스크 불필요: 미래 토큰은 아직 없다
        outputs[t : t + 1] = out
    return outputs, rows_computed


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

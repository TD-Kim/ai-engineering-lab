"""P1-05 Transformer — 실습 코드 (TODO 를 채우세요).

Level 1: Q/K/V 선형 층, 스케일드 닷-프로덕트 어텐션, causal·padding 마스크 (TODO 1~10, SelfAttention)
Level 2: Multi-Head Attention(헤드 분할·합치기), Feed Forward, 위치 인코딩(sinusoidal),
         Residual + LayerNorm 으로 Encoder 블록 조립 (TODO 11~19)
Level 3: 문자 단위 미니 언어모델 조립·학습·생성 (TODO 20~23)
         → train_tiny.py(학습·손실 곡선), visualize_attention.py(어텐션 가중치 그림),
           compare_with_torch.py(nn.MultiheadAttention / F.scaled_dot_product_attention 과 같은 가중치로 비교)

진행 확인: python part1_ai_fundamentals/transformer/starter.py   (남은 TODO 목록 출력)
검사:     IMPL=starter pytest part1_ai_fundamentals/transformer

마스크 규약: bool 텐서, True(1) = "볼 수 있음(attend)", False(0) = "가림".
  - padding_mask: (B, T), 1 = 실제 토큰, 0 = 패딩 (P1-01 tokenizer 의 attention_mask 와 같은 뜻)
  - causal: i 번째 query 는 j <= i 인 key 만 본다.
  - 주의: 한 query 행의 key 가 전부 가려지면 그 행은 softmax(-inf, ..., -inf) = NaN 이 된다(README 9절, 테스트로 재현).
shape 기호: B = batch, T = 시퀀스 길이, D = d_model, H = head 수, d_k = D / H, V = 어휘 크기

REF: Vaswani et al., 2017, "Attention Is All You Need"
"""
from __future__ import annotations

import math  # noqa: F401  (TODO 7, 13, 16, 21 에서 사용)

import torch
import torch.nn.functional as F  # noqa: F401  (TODO 22 에서 사용)
from torch import nn


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Level 1 — Single-head Self-Attention (TODO 1~10)
# ---------------------------------------------------------------------------
class SelfAttention(nn.Module):
    def __init__(self, d_model: int, d_k: int | None = None, dropout: float = 0.0):
        super().__init__()
        d_k = d_k or d_model
        self.d_k = d_k
        # TODO 1: 입력 (…, d_model) 을 각각 (…, d_k) 로 바꾸는 Query·Key·Value 투영 self.W_q, self.W_k, self.W_v 를 만드세요(bias 없음).
        # HINT 1: 같은 입력 x 에서 세 가지 다른 선형 변환을 얻습니다.
        # HINT 2: nn.Linear(in_features, out_features, bias=False) 세 개
        # WHY: Q/K/V 를 하나로 쓰면 점수 qᵢ·qⱼ 가 대칭(i→j 와 j→i 가 같음)이 되고, "찾는 것·찾게 하는 표시·건네줄 내용"을 구분할 수 없습니다.
        # REF: Vaswani et al., 2017, "Attention Is All You Need", §3.2.1
        raise NotImplementedError("TODO 1")
        # TODO 2: (…, d_k) 를 다시 (…, d_model) 로 바꾸는 출력 투영 self.W_o 를 만드세요(bias 없음).
        # HINT 1: 어텐션 출력이 다음 층(잔차 연결)의 입력과 같은 크기여야 합니다.
        # HINT 2: nn.Linear(d_k, d_model, bias=False)
        # WHY: x + attn(x) 는 shape 이 같아야 더할 수 있습니다.
        # REF: Vaswani et al., 2017, §3.2.2
        # TODO 3: 어텐션 가중치에 적용할 self.dropout 을 만드세요.
        # HINT 1: 학습 때만 동작하고 평가(eval) 때는 꺼지는 층입니다.
        # HINT 2: nn.Dropout(p)
        # WHY: 특정 위치 하나에만 지나치게 의존하는 것을 막는 정규화입니다.
        self.last_attn_weights: torch.Tensor | None = None  # 시각화용

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, T, D), padding_mask: (B, T) bool(True=실제 토큰) -> (out (B, T, D), weights (B, T, T))"""
        T = x.shape[1]  # noqa: F841  (B, T, D) 중 시퀀스 길이 — TODO 5 의 causal 마스크 크기

        # TODO 4: x 를 Q, K, V 로 투영하세요. 각각 (B, T, d_k).
        # HINT 1: TODO 1 의 세 선형 층을 x 에 적용합니다.
        # HINT 2: 결과 shape 을 print(q.shape) 로 확인해 보세요 → (B, T, d_k)
        # WHY: 같은 x 에서 서로 다른 세 표현을 만들어, 점수 계산(q·k)과 전달할 내용(v)을 분리합니다.
        # REF: Vaswani et al., 2017, §3.2.1
        raise NotImplementedError("TODO 4")  # ← 이 줄을 지우고 q, k, v 를 만드세요

        mask = None  # (제공) bool, True = 볼 수 있음
        # TODO 5: causal 이면 i 번째 query 가 j <= i 인 key 만 보는 (1, T, T) bool 마스크를 mask 에 넣으세요.
        # HINT 1: 대각선 포함 아래쪽 삼각형만 True 인 행렬입니다.
        # HINT 2: torch.tril, torch.ones(..., dtype=torch.bool, device=x.device), unsqueeze(0)
        # WHY: 언어모델이 미래 글자를 보면 학습 때만 정답을 "엿보는" 누수가 생겨, 실제 생성에서는 성능이 무너집니다.
        # REF: Vaswani et al., 2017, §3.2.3
        if causal:
            raise NotImplementedError("TODO 5")

        # TODO 6: padding_mask (B, T) 가 있으면 key 축에 적용되도록 (B, 1, T) 로 바꿔 mask 와 AND 로 합치세요(mask 가 None 이면 그대로 사용).
        # HINT 1: 패딩 토큰은 "누구에게도 참고되면 안 되는" key 입니다. 모든 query 행에 같은 마스크를 씁니다.
        # HINT 2: Tensor.bool(), unsqueeze(1), & 연산자 (브로드캐스팅으로 (B, T, T))
        # WHY: 마스크를 query 축에 잘못 붙이면 패딩이 여전히 참고되고, 실제 토큰의 출력이 문장 길이에 따라 달라집니다.
        if padding_mask is not None:
            raise NotImplementedError("TODO 6")

        # TODO 7: 어텐션 점수 scores (B, T, T) 를 계산하세요.
        # HINT 1: 각 query 와 모든 key 의 유사도를 내적으로 구하고 크기를 보정합니다.
        # HINT 2: k 의 마지막 두 차원을 바꾼 뒤 행렬곱 → (B, T, T). math.sqrt, Tensor.transpose
        # WHY: d_k 가 크면 내적 값의 분산이 커져 softmax 가 한쪽으로 포화되고 기울기가 매우 작아집니다.
        # REF: Vaswani et al., 2017, "Attention Is All You Need", §3.2.1
        raise NotImplementedError("TODO 7")  # ← 이 줄을 지우고 scores 를 만드세요

        # TODO 8: mask 가 있으면 가려야 할 위치(False)의 점수를 -inf 로 바꾸세요.
        # HINT 1: softmax 이후 그 위치의 가중치가 정확히 0 이 되어야 합니다.
        # HINT 2: Tensor.masked_fill(조건, 값) — 조건은 "가릴 위치" 이므로 mask 를 뒤집어야 합니다(~).
        # WHY: 한 query 행 전체가 가려지면 softmax(-inf, …) = NaN 이 됩니다(README 9절). 마스크 설계 때 이 경우를 피해야 합니다.
        # REF: Vaswani et al., 2017, §3.2.3
        if mask is not None:
            raise NotImplementedError("TODO 8")

        # TODO 9: softmax 로 가중치 weights (B, T, T) 를 만들고 dropout 을 적용한 뒤, value 의 가중합 context (B, T, d_k) 를 구하세요.
        # HINT 1: 마지막 차원(key 축)에 대해 확률 분포가 되어야 합니다.
        # HINT 2: torch.softmax(..., dim=-1), self.dropout, 행렬곱 (B, T, T) @ (B, T, d_k)
        # WHY: 잘못된 dim 으로 softmax 를 하면 각 query 의 가중치 합이 1 이 되지 않아 출력 스케일이 깨집니다.
        # REF: Vaswani et al., 2017, §3.2.1
        raise NotImplementedError("TODO 9")  # ← 이 줄을 지우고 weights, context 를 만드세요

        # TODO 10: context 를 출력 투영해 out (B, T, D) 을 만들고, 시각화용으로 self.last_attn_weights 에 weights 를 (기울기 없이) 저장한 뒤
        #          (out, weights) 를 반환하세요.
        # HINT 1: TODO 2 의 출력 투영을 씁니다.
        # HINT 2: Tensor.detach()
        # WHY: detach 하지 않고 저장하면 계산 그래프가 계속 붙잡혀 메모리가 새고, 시각화 코드가 학습에 영향을 줄 수 있습니다.
        raise NotImplementedError("TODO 10")


# ---------------------------------------------------------------------------
# Level 2 — Multi-Head Attention (TODO 11~14)
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        # TODO 11: d_model 이 num_heads 로 나누어떨어지지 않으면 ValueError, 아니면 self.d_model, self.num_heads, self.d_k(= D / H),
        #          Q·K·V 를 한 번에 만드는 self.qkv (D -> 3D, bias 없음), self.out_proj (D -> D, bias 없음), self.dropout 을 만드세요.
        # HINT 1: 헤드마다 따로 Linear 를 만들지 않고, 큰 Linear 하나의 출력을 나중에 잘라 씁니다.
        # HINT 2: nn.Linear(d_model, 3 * d_model, bias=False). 출력의 앞·중간·뒤 1/3 이 Q·K·V 입니다(nn.MultiheadAttention 의 in_proj_weight 와 같은 배치).
        # WHY: H 개 헤드 × 3 개 투영을 행렬곱 한 번으로 끝내 빠르고, 전체 파라미터 수는 단일 헤드(D×D ×3)와 같습니다.
        # REF: Vaswani et al., 2017, §3.2.2
        raise NotImplementedError("TODO 11")
        self.last_attn_weights: torch.Tensor | None = None

    def _split_heads(self, t: torch.Tensor) -> torch.Tensor:
        # TODO 12: (B, T, D) 를 (B, H, T, d_k) 로 바꾸세요.
        # HINT 1: 마지막 차원 D 를 (H, d_k) 두 차원으로 쪼갠 뒤, head 축을 시퀀스 축 앞으로 보냅니다.
        # HINT 2: Tensor.view(B, T, H, d_k), Tensor.transpose(1, 2)
        # WHY: head 축이 앞에 있어야 이후 행렬곱이 head 마다 독립적으로 (T, T) 점수를 계산합니다.
        raise NotImplementedError("TODO 12")

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, T, D) -> (out (B, T, D), weights (B, H, T, T))"""
        B, T, D = x.shape  # noqa: RUF059  (B, D 는 TODO 14 에서 사용)
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # 각각 (B, T, D)
        q, k, v = self._split_heads(q), self._split_heads(k), self._split_heads(v)  # (B, H, T, d_k)

        mask = None
        if causal:
            mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))  # (T, T)
        if padding_mask is not None:
            pad = padding_mask.bool()[:, None, None, :]  # (B, 1, 1, T)
            mask = pad if mask is None else (mask & pad)  # (B, 1, T, T)

        # TODO 13: head 별 점수 (B, H, T, T) → 마스크 → softmax(+dropout) → value 가중합 context (B, H, T, d_k) 를 구하세요.
        # HINT 1: TODO 7~9 와 같은 계산이지만 head 축이 하나 더 있습니다. mask 는 H 축으로 브로드캐스팅됩니다.
        # HINT 2: 스케일은 head 하나의 차원 self.d_k 로 나눕니다(D 가 아님).
        # WHY: 각 head 의 q·k 는 d_k 차원 내적이므로 분산도 d_k 에 비례합니다.
        # REF: Vaswani et al., 2017, §3.2.1, §3.2.2
        raise NotImplementedError("TODO 13")
        # TODO 14: context 를 (B, T, D) 로 다시 합치고 self.out_proj 를 적용해, self.last_attn_weights 저장 후 (out, weights) 를 반환하세요.
        # HINT 1: TODO 12 의 반대 순서입니다(head 축을 뒤로 보낸 뒤 (H, d_k) 를 D 로 합치기).
        # HINT 2: transpose(1, 2) 다음 view 를 하려면 contiguous() 가 필요합니다.
        # WHY: 여러 head 가 서로 다른 관계(바로 앞 글자, 같은 단어 안 등)를 본 결과를 하나의 벡터로 모아 다음 층에 넘깁니다.
        # REF: Vaswani et al., 2017, §3.2.2


# ---------------------------------------------------------------------------
# Level 2 — FFN, Positional Encoding, Block (TODO 15~19)
# ---------------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, dropout: float = 0.0):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        # TODO 15: self.net = Linear(D → d_ff) → GELU → Linear(d_ff → D) → Dropout 을 순서대로 만드세요.
        # HINT 1: 위치마다 똑같이 적용되는 2층 MLP 입니다(위치끼리 섞지 않음).
        # HINT 2: nn.Sequential, nn.Linear, nn.GELU, nn.Dropout
        # WHY: 비선형 활성화가 없으면 두 Linear 는 하나의 Linear 와 같아집니다. 어텐션이 섞은 정보를 각 위치에서 비선형 변환합니다.
        # REF: Vaswani et al., 2017, §3.3 (원 논문은 ReLU, GPT 계열은 주로 GELU)
        raise NotImplementedError("TODO 15")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, T, D) -> (B, T, D)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        # TODO 16: (max_len, d_model) 사인/코사인 위치 표 pe 를 만들어 self.register_buffer("pe", pe, persistent=False) 로 등록하세요.
        #          짝수 열 = sin(pos / 10000^(2i/D)), 홀수 열 = cos(pos / 10000^(2i/D))
        # HINT 1: 위치 (max_len, 1) 와 주파수 (D/2,) 를 곱하면 브로드캐스팅으로 (max_len, D/2) 가 됩니다.
        # HINT 2: torch.arange, torch.exp(... * (-math.log(10000.0) / d_model)), 슬라이싱 pe[:, 0::2], pe[:, 1::2]
        # WHY: 어텐션은 순서를 모릅니다(입력 순서를 섞으면 출력도 똑같이 섞일 뿐). buffer 로 두면 학습되지 않으면서 .to(device) 에 함께 따라갑니다.
        # REF: Vaswani et al., 2017, §3.5
        raise NotImplementedError("TODO 16")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 17: x (B, T, D) 에 앞 T 개 위치의 인코딩을 더해 반환하세요.
        # HINT 1: 표 전체가 아니라 시퀀스 길이만큼만 잘라 씁니다.
        # HINT 2: self.pe[:T] 는 (T, D) → (B, T, D) 와 브로드캐스팅
        # WHY: 같은 글자라도 위치마다 다른 벡터가 되어, 뒤의 어텐션이 순서를 구별할 수 있습니다.
        raise NotImplementedError("TODO 17")


class TransformerBlock(nn.Module):
    """Encoder Block 구조(Self-Attention + FFN, 각각 Residual + LayerNorm).

    causal=True 로 쓰면 GPT 같은 decoder-only 언어모델의 블록이 된다.
    LayerNorm 위치는 Pre-LN(서브층 앞)을 쓴다: 원 논문(Post-LN)보다 작은 모델·짧은 학습에서 안정적이다.
    REF: Xiong et al., 2020, "On Layer Normalization in the Transformer Architecture"
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int | None = None, dropout: float = 0.0):
        super().__init__()
        # TODO 18: self.ln1, self.attn(MultiHeadAttention), self.ln2, self.ffn(FeedForward), self.dropout 을 만드세요.
        # HINT 1: 서브층(어텐션, FFN)마다 LayerNorm 이 하나씩 있습니다.
        # HINT 2: nn.LayerNorm(d_model), MultiHeadAttention(d_model, num_heads, dropout), FeedForward(d_model, d_ff, dropout)
        # WHY: LayerNorm 은 각 토큰 벡터를 평균 0·분산 1 로 맞춰 층을 거칠수록 값 크기가 폭주하지 않게 합니다(배치 크기와 무관).
        # REF: Ba et al., 2016, "Layer Normalization"
        raise NotImplementedError("TODO 18")

    def forward(self, x: torch.Tensor, padding_mask=None, causal: bool = False) -> torch.Tensor:
        # TODO 19: Pre-LN 순서로 두 번의 잔차 연결을 구현하세요: x = x + dropout(attn(ln1(x))), x = x + ffn(ln2(x))
        # HINT 1: 어텐션은 (out, weights) 를 반환합니다. out 만 씁니다. padding_mask·causal 을 그대로 넘기세요.
        # HINT 2: 입력과 출력 모두 (B, T, D)
        # WHY: 잔차 연결은 기울기가 덧셈을 통해 그대로 아래층으로 흐르는 지름길이라 깊어져도 학습됩니다. 서브층이 0 을 출력해도 항등 함수가 됩니다.
        #      Pre-LN(서브층 앞에 LN)은 원 논문의 Post-LN 보다 작은 모델·짧은 학습에서 학습률 warmup 없이도 안정적입니다(README 4절).
        # REF: He et al., 2016, "Deep Residual Learning for Image Recognition"; Vaswani et al., 2017, §3.1; Xiong et al., 2020
        raise NotImplementedError("TODO 19")


# 이 저장소의 Encoder 블록 = TransformerBlock (causal=False 면 인코더, True 면 GPT 식 디코더 전용 블록으로 쓴다)
EncoderBlock = TransformerBlock


# ---------------------------------------------------------------------------
# Level 3 — 문자 단위 미니 언어모델 (TODO 20~23)
# ---------------------------------------------------------------------------
class MiniTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        max_len: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        # TODO 20: self.tok_embed(V → D), self.pos_enc(PositionalEncoding), self.blocks(TransformerBlock num_layers 개, nn.ModuleList),
        #          self.ln_f(최종 LayerNorm), self.lm_head(D → V, bias 없음) 를 만드세요.
        # HINT 1: 블록 목록은 파이썬 리스트가 아니라 모듈 리스트여야 파라미터가 등록됩니다.
        #          tok_embed 는 std = d_model ** -0.5 의 정규분포로 다시 초기화하세요(TODO 21 의 ×√D 와 짝).
        # HINT 2: nn.Embedding, nn.init.normal_(weight, std=...), nn.ModuleList([...]), nn.LayerNorm, nn.Linear(..., bias=False)
        # WHY: Pre-LN 은 마지막 블록 출력이 정규화되지 않으므로 출력층 앞에 LayerNorm 을 한 번 더 둡니다.
        #      기본 초기화 N(0, 1) 에 ×√D 를 하면 토큰 벡터 노름(≈ D)이 위치 인코딩 노름(≈ √(D/2))을 압도해 위치 정보가 묻힙니다.
        raise NotImplementedError("TODO 20")

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: (B, T) -> logits (B, T, V)"""
        # TODO 21: 길이가 max_len 을 넘으면 ValueError. 토큰 임베딩 × sqrt(D) → 위치 인코딩 → 블록들(causal=True) → ln_f → lm_head 로
        #          logits (B, T, V) 를 반환하세요.
        # HINT 1: 언어모델은 다음 글자를 맞히므로 모든 블록에서 미래를 가려야 합니다.
        # HINT 2: math.sqrt(self.d_model), for block in self.blocks: x = block(x, causal=True)
        # WHY: causal 을 빼면 학습 loss 는 거의 0 까지 떨어지지만(정답을 보고 맞힘) 생성은 엉망이 됩니다.
        #      위치 인코딩을 빼도 causal LM 은 학습되지만, 같은 글자가 반복된 입력("aaaa")처럼 내용만으로 구별되지 않는 위치는
        #      구분하지 못합니다(위치마다 같은 logits). 임베딩을 키우는 ×√D 는 TODO 20 의 작은 초기화와 짝입니다.
        # REF: Vaswani et al., 2017, §3.4 (임베딩에 sqrt(d_model) 곱하기), §3.5 (위치 인코딩)
        raise NotImplementedError("TODO 21")

    def loss(self, ids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """ids, targets: (B, T). targets 는 ids 를 한 칸 왼쪽으로 민 것(다음 글자)."""
        # TODO 22: logits (B, T, V) 와 targets (B, T) 로 cross entropy 를 계산하세요.
        # HINT 1: cross_entropy 는 (N, V) 와 (N,) 를 기대합니다.
        # HINT 2: Tensor.reshape(-1, V), targets.reshape(-1)
        # WHY: B 와 T 축을 합치면 "모든 위치의 다음 글자 맞히기"를 한 번에 평균합니다.
        raise NotImplementedError("TODO 22")

    @torch.no_grad()
    def generate(
        self, ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, generator=None
    ) -> torch.Tensor:
        """ids: (B, T0) -> (B, T0 + max_new_tokens)"""
        # TODO 23: max_new_tokens 번 반복하며 (최근 max_len 개만 넣어) 마지막 위치의 logits 로 다음 토큰을 샘플링해 ids 뒤에 붙이세요.
        # HINT 1: logits / temperature 에 softmax → 확률 분포에서 1 개 뽑기.
        # HINT 2: ids[:, -self.max_len:], self(context)[:, -1, :], torch.multinomial(probs, 1, generator=generator), torch.cat(dim=1)
        # WHY: 위치 인코딩 표가 max_len 까지만 있어 더 긴 문맥은 넣을 수 없습니다. temperature 는 낮을수록 보수적, 높을수록 다양합니다.
        raise NotImplementedError("TODO 23")


# ---------------------------------------------------------------------------
# 데이터 유틸 (제공)
# ---------------------------------------------------------------------------
class CharVocab:
    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    def __len__(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids) -> str:
        return "".join(self.chars[int(i)] for i in ids)


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, generator=None):
    """1차원 토큰 텐서에서 무작위 위치의 (입력, 다음 토큰 타깃) 묶음을 만든다. 둘 다 (B, block_size)."""
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([data[s : s + block_size] for s in starts])
    y = torch.stack([data[s + 1 : s + block_size + 1] for s in starts])
    return x, y


def count_parameters(model: nn.Module) -> int:
    """학습되는 파라미터 수 (buffer 인 위치 인코딩 표는 제외)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_lm(
    text: str,
    steps: int = 300,
    d_model: int = 64,
    num_heads: int = 4,
    num_layers: int = 2,
    block_size: int = 64,
    batch_size: int = 32,
    lr: float = 3e-3,
    eval_every: int = 50,
    seed: int = 0,
    device: torch.device | None = None,
) -> dict:
    """(제공) 문자 단위 언어모델 학습. 앞 90% 로 학습하고 뒤 10% 로 검증 손실을 잰다.

    반환: {"model", "vocab", "history": [(step, train_loss, val_loss)], "params", "seconds", "device"}
    """
    import time

    torch.manual_seed(seed)
    device = device or get_device()
    vocab = CharVocab(text)
    data = torch.tensor(vocab.encode(text), dtype=torch.long)
    split = int(len(data) * 0.9)
    train_data, val_data = data[:split], data[split:]
    model = MiniTransformerLM(len(vocab), d_model, num_heads, num_layers, max_len=block_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)

    @torch.no_grad()
    def evaluate(split_data: torch.Tensor) -> float:
        model.eval()
        eval_gen = torch.Generator().manual_seed(1234)  # 같은 위치로 평가해 step 간 비교가 가능하게
        x, y = get_batch(split_data, min(block_size, len(split_data) - 2), 32, generator=eval_gen)
        loss = model.loss(x.to(device), y.to(device)).item()
        model.train()
        return loss

    history = [(0, evaluate(train_data), evaluate(val_data))]
    start = time.perf_counter()
    for step in range(1, steps + 1):
        x, y = get_batch(train_data, block_size, batch_size, generator=gen)
        loss = model.loss(x.to(device), y.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % eval_every == 0 or step == steps:
            history.append((step, evaluate(train_data), evaluate(val_data)))
    return {
        "model": model,
        "vocab": vocab,
        "history": history,
        "params": count_parameters(model),
        "seconds": time.perf_counter() - start,
        "device": device,
    }


# ---------------------------------------------------------------------------
# 단독 실행: 어느 TODO 가 남았는지 알려 준다.
# ---------------------------------------------------------------------------
def _report_progress() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 5, 8)
    ids = torch.randint(0, 10, (2, 5))

    def lm():
        return MiniTransformerLM(10, d_model=8, num_heads=2, num_layers=1, max_len=16)

    checks = [
        ("TODO 1~3   SelfAttention.__init__ — Q/K/V·출력 투영, dropout (Level 1)", lambda: SelfAttention(8)),
        ("TODO 4~10  SelfAttention.forward — 투영·마스크·점수·softmax·가중합 (Level 1)", lambda: SelfAttention(8)(x, causal=True)),
        ("TODO 11    MultiHeadAttention.__init__ (Level 2)", lambda: MultiHeadAttention(8, 2)),
        ("TODO 12~14 MultiHeadAttention.forward — 헤드 분할·어텐션·합치기 (Level 2)", lambda: MultiHeadAttention(8, 2)(x)),
        ("TODO 15    FeedForward (Level 2)", lambda: FeedForward(8)(x)),
        ("TODO 16    PositionalEncoding.__init__ (Level 2)", lambda: PositionalEncoding(8, 16)),
        ("TODO 17    PositionalEncoding.forward (Level 2)", lambda: PositionalEncoding(8, 16)(x)),
        ("TODO 18    TransformerBlock.__init__ (Level 2)", lambda: TransformerBlock(8, 2)),
        ("TODO 19    TransformerBlock.forward — Residual + LayerNorm (Level 2)", lambda: TransformerBlock(8, 2)(x)),
        ("TODO 20    MiniTransformerLM.__init__ (Level 3)", lm),
        ("TODO 21    MiniTransformerLM.forward (Level 3)", lambda: lm()(ids)),
        ("TODO 22    MiniTransformerLM.loss (Level 3)", lambda: lm().loss(ids, ids)),
        ("TODO 23    MiniTransformerLM.generate (Level 3)", lambda: lm().generate(ids[:, :2], 3)),
    ]
    remaining = 0
    for name, fn in checks:
        try:
            fn()
            print(f"[완료]   {name}")
        except NotImplementedError as exc:
            remaining += 1
            print(f"[미구현] {name}  ← 처음 막힌 곳: {exc}")
    if remaining:
        print(f"\n남은 항목 {remaining}개 — 위에서부터 순서대로 채우세요(뒤 TODO 는 앞 TODO 를 사용합니다). 채운 뒤 검사:")
        print("  (bash)       IMPL=starter pytest part1_ai_fundamentals/transformer")
        print('  (PowerShell) $env:IMPL="starter"; pytest part1_ai_fundamentals/transformer; Remove-Item Env:IMPL')
    else:
        print("\n모든 TODO 가 구현되었습니다. 이제 pytest 로 정답 여부를 확인하세요.")


if __name__ == "__main__":
    _report_progress()

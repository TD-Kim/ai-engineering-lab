"""P1-05 Transformer — 정답 코드.

Level 1: Single-head Self-Attention (TODO 1~10)
Level 2: Multi-Head Attention, FFN, Positional Encoding, Transformer Block(Residual + LayerNorm) (TODO 11~19)
Level 3: 문자 단위 미니 언어모델 조립·학습·생성 (TODO 20~23),
         nn.MultiheadAttention / F.scaled_dot_product_attention 과 비교 → compare_with_torch.py

마스크 규약: bool 텐서, True = "볼 수 있음(attend)", False = "가림".
shape 기호: B = batch, T = 시퀀스 길이, D = d_model, H = head 수, d_k = D / H, V = 어휘 크기

REF: Vaswani et al., 2017, "Attention Is All You Need"
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
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
        # TODO 1  REF: Vaswani et al., 2017, §3.2.1
        # 왜 Q/K/V 를 분리하나: 같은 토큰이라도 "내가 찾는 것(Query)", "나를 찾게 하는 표시(Key)",
        # "내가 실제로 건네줄 내용(Value)" 의 역할이 다르다. 하나의 벡터로 세 역할을 하면
        # 점수 qᵢ·kⱼ 가 항상 대칭(i→j 와 j→i 가 같음)이 되고, 전달할 내용도 점수 계산용 표현에 묶인다.
        self.W_q = nn.Linear(d_model, d_k, bias=False)  # (D) -> (d_k)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_k, bias=False)
        # TODO 2  REF: Vaswani et al., 2017, §3.2.2
        # 출력 차원을 다시 d_model 로: 잔차 연결(x + attn(x))을 하려면 입력과 shape 이 같아야 한다.
        self.W_o = nn.Linear(d_k, d_model, bias=False)
        # TODO 3
        # attention dropout: 특정 위치 하나에만 의존하는 것을 막는 정규화. 학습(train) 때만 동작한다.
        self.dropout = nn.Dropout(dropout)
        self.last_attn_weights: torch.Tensor | None = None  # 시각화용

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, T, D), padding_mask: (B, T) bool(True=실제 토큰) -> (out (B, T, D), weights (B, T, T))"""
        B, T, _ = x.shape

        # TODO 4  REF: Vaswani et al., 2017, §3.2.1
        q = self.W_q(x)  # (B, T, d_k)
        k = self.W_k(x)  # (B, T, d_k)
        v = self.W_v(x)  # (B, T, d_k)

        # TODO 5  REF: Vaswani et al., 2017, §3.2.3
        # 하삼각 행렬: i 번째 query 는 j <= i 인 key 만 본다 (미래 토큰 차단).
        mask = None
        if causal:
            mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device)).unsqueeze(0)  # (1, T, T)

        # TODO 6
        # padding_mask 는 "key" 축에 적용: 패딩 토큰은 누구에게도 참고 대상이 되면 안 된다.
        # (B, T) -> (B, 1, T): 모든 query 행에 같은 key 마스크를 브로드캐스팅.
        if padding_mask is not None:
            pad = padding_mask.bool().unsqueeze(1)  # (B, 1, T)
            mask = pad if mask is None else (mask & pad)  # (B, T, T)

        # TODO 7  REF: Vaswani et al., 2017, "Attention Is All You Need", §3.2.1
        # (B, T, d_k) @ (B, d_k, T) -> (B, T, T)
        # 왜 sqrt(d_k) 로 나누나: q, k 의 각 성분이 평균 0·분산 1 이면 내적의 분산은 d_k 이다.
        # d_k 가 크면 점수 차이가 너무 커져 softmax 가 one-hot 에 가깝게 포화되고 기울기가 거의 0 이 된다.
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)

        # TODO 8  REF: Vaswani et al., 2017, §3.2.3
        # -inf 는 softmax 후 정확히 0 → 가려진 위치는 가중합에 전혀 기여하지 않는다.
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        # TODO 9  REF: Vaswani et al., 2017, §3.2.1
        # softmax 의 역할: 점수를 "합이 1 인 양수 가중치"로 바꿔, 출력이 value 들의 볼록 결합(가중 평균)이 되게 한다.
        # dim=-1 (key 축): 각 query 행마다 확률 분포. dim 을 틀리면 행 합이 1 이 아니다.
        weights = torch.softmax(scores, dim=-1)  # (B, T, T)
        weights = self.dropout(weights)
        context = weights @ v  # (B, T, T) @ (B, T, d_k) -> (B, T, d_k)

        # TODO 10
        out = self.W_o(context)  # (B, T, d_k) -> (B, T, D)
        self.last_attn_weights = weights.detach()
        return out, weights


# ---------------------------------------------------------------------------
# Level 2 — Multi-Head Attention (TODO 11~14)
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        # TODO 11  REF: Vaswani et al., 2017, §3.2.2
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model, self.num_heads = d_model, num_heads
        self.d_k = d_model // num_heads
        # Q, K, V 투영을 하나의 Linear(D -> 3D) 로 합친다: 행렬곱 한 번이라 빠르고,
        # nn.MultiheadAttention 의 in_proj_weight 배치([W_q; W_k; W_v])와 같아 비교하기 쉽다.
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.last_attn_weights: torch.Tensor | None = None

    def _split_heads(self, t: torch.Tensor) -> torch.Tensor:
        # TODO 12
        # (B, T, D) -> (B, T, H, d_k) -> (B, H, T, d_k)
        # head 축을 batch 처럼 앞으로 보내면 이후 행렬곱이 head 별로 독립적으로 계산된다.
        B, T, _ = t.shape
        return t.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, T, D) -> (out (B, T, D), weights (B, H, T, T))"""
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # 각각 (B, T, D)
        q, k, v = self._split_heads(q), self._split_heads(k), self._split_heads(v)  # (B, H, T, d_k)

        mask = None
        if causal:
            mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))  # (T, T)
        if padding_mask is not None:
            pad = padding_mask.bool()[:, None, None, :]  # (B, 1, 1, T)
            mask = pad if mask is None else (mask & pad)  # (B, 1, T, T)

        # TODO 13  REF: Vaswani et al., 2017, §3.2.1, §3.2.2
        # TODO 7~9 와 같은 계산이지만 head 축이 하나 더 있다. 각 head 의 d_k 로 나눈다(D 가 아님).
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)  # (B, H, T, T)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))  # mask 는 H 축으로 브로드캐스팅
        weights = self.dropout(torch.softmax(scores, dim=-1))  # (B, H, T, T)
        context = weights @ v  # (B, H, T, d_k)

        # TODO 14  REF: Vaswani et al., 2017, §3.2.2
        # (B, H, T, d_k) -> (B, T, H, d_k) -> (B, T, D)
        # transpose 후 메모리가 연속적이지 않아 view 전에 contiguous() 가 필요하다.
        # 왜 여러 head: head 마다 서로 다른 투영 공간에서 서로 다른 관계(예: 바로 앞 글자, 같은 단어 안)를 본다.
        context = context.transpose(1, 2).contiguous().view(B, T, D)
        self.last_attn_weights = weights.detach()
        return self.out_proj(context), weights


# ---------------------------------------------------------------------------
# Level 2 — FFN, Positional Encoding, Block (TODO 15~19)
# ---------------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, dropout: float = 0.0):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        # TODO 15  REF: Vaswani et al., 2017, §3.3
        # 위치별(position-wise) 2층 MLP: 어텐션이 "섞은" 정보를 각 위치에서 비선형 변환한다.
        # 비선형 활성화가 없으면 두 Linear 는 하나의 Linear 와 같아진다.
        # 원 논문은 ReLU, GPT 계열은 GELU 를 주로 쓴다.
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, T, D) -> (B, T, D)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        # TODO 16  REF: Vaswani et al., 2017, §3.5
        # PE(pos, 2i) = sin(pos / 10000^(2i/D)),  PE(pos, 2i+1) = cos(pos / 10000^(2i/D))
        # 왜 필요한가: 어텐션은 순서를 모른다(입력 순서를 섞으면 출력도 똑같이 섞일 뿐). 위치 정보를 더해 줘야 한다.
        pos = torch.arange(max_len).unsqueeze(1).float()  # (max_len, 1)
        # exp(-log(10000) * 2i / D) = 1 / 10000^(2i/D) — 거듭제곱보다 수치적으로 안정적
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (D/2,)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)  # (max_len, D/2)
        pe[:, 1::2] = torch.cos(pos * div)
        # buffer: 학습되지 않지만 model.to(device), state_dict 에 함께 포함된다.
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 17
        # x: (B, T, D) + pe[:T]: (T, D) → batch 축으로 브로드캐스팅
        return x + self.pe[: x.size(1)]


class TransformerBlock(nn.Module):
    """Encoder Block 구조(Self-Attention + FFN, 각각 Residual + LayerNorm).

    causal=True 로 쓰면 GPT 같은 decoder-only 언어모델의 블록이 된다.
    LayerNorm 위치는 Pre-LN(서브층 앞)을 쓴다: 원 논문(Post-LN)보다 작은 모델·짧은 학습에서 안정적이다.
    REF: Xiong et al., 2020, "On Layer Normalization in the Transformer Architecture"
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int | None = None, dropout: float = 0.0):
        super().__init__()
        # TODO 18  REF: Ba et al., 2016, "Layer Normalization"
        # LayerNorm: 각 토큰 벡터를 평균 0·분산 1 로 맞춰(학습 가능한 scale/shift 포함) 층마다 값 크기가 폭주하지 않게 한다.
        # BatchNorm 과 달리 배치 크기·시퀀스 길이에 의존하지 않는다.
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, padding_mask=None, causal: bool = False) -> torch.Tensor:
        # TODO 19  REF: He et al., 2016, "Deep Residual Learning for Image Recognition"; Vaswani et al., 2017, §3.1
        # 잔차 연결 x + f(x): 기울기가 덧셈을 통해 그대로 아래층으로 흐르는 "지름길"이 생겨 깊어져도 학습된다.
        # 서브층이 아무것도 배우지 않아도(f ≈ 0) 최소한 항등 함수가 되어 성능이 망가지지 않는다.
        attn_out, _ = self.attn(self.ln1(x), padding_mask=padding_mask, causal=causal)
        x = x + self.dropout(attn_out)       # (B, T, D)
        x = x + self.ffn(self.ln2(x))        # (B, T, D)
        return x


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
        # TODO 20
        # 왜 Token ID 를 벡터로: id 는 크기 비교가 의미 없는 이름표다. 학습 가능한 벡터로 바꿔야
        # 비슷한 역할의 토큰이 가까워지고, 어텐션의 내적 점수가 의미를 갖는다.
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        # Pre-LN 은 마지막 블록 출력이 정규화되지 않으므로 출력층 앞에 LayerNorm 을 한 번 더 둔다.
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: (B, T) -> logits (B, T, V)"""
        # TODO 21  REF: Vaswani et al., 2017, §3.4
        if ids.size(1) > self.max_len:
            raise ValueError(f"sequence length {ids.size(1)} > max_len {self.max_len}")
        # §3.4: 임베딩에 sqrt(D) 를 곱한다. 흔한 해석: N(0,1) 초기화된 임베딩을 키워 위치 인코딩(±1)에 묻히지 않게 한다.
        x = self.tok_embed(ids) * math.sqrt(self.d_model)  # (B, T, D)
        x = self.pos_enc(x)
        for block in self.blocks:
            # causal=True: 언어모델은 다음 글자를 맞혀야 하므로 미래를 보면 "정답 엿보기"가 된다.
            x = block(x, causal=True)
        return self.lm_head(self.ln_f(x))  # (B, T, V)

    def loss(self, ids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """ids, targets: (B, T). targets 는 ids 를 한 칸 왼쪽으로 민 것(다음 글자)."""
        # TODO 22
        logits = self(ids)  # (B, T, V)
        # cross_entropy 는 (N, V) 와 (N,) 를 기대하므로 B, T 축을 합친다.
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

    @torch.no_grad()
    def generate(
        self, ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, generator=None
    ) -> torch.Tensor:
        """ids: (B, T0) -> (B, T0 + max_new_tokens)"""
        # TODO 23
        for _ in range(max_new_tokens):
            # max_len 보다 길어지면 최근 max_len 개만 본다(위치 인코딩 표 크기 한계).
            context = ids[:, -self.max_len :]
            logits = self(context)[:, -1, :]  # 마지막 위치의 다음 토큰 분포 (B, V)
            # temperature < 1: 분포를 뾰족하게(보수적), > 1: 평평하게(다양하게)
            probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
            next_id = torch.multinomial(probs, 1, generator=generator)  # (B, 1)
            ids = torch.cat([ids, next_id], dim=1)
        return ids


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

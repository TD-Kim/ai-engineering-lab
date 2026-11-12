"""(선택) 실제 LLM(Qwen/Qwen2.5-0.5B, Apache-2.0)으로 다음 토큰 확률·샘플링·컨텍스트 길이별 시간을 본다 — 사용자 PC 용.

이 실습의 함수(softmax_with_temperature, top_p_filter, generate ...)를 그대로 쓰고, 모델만 bigram 대신 Qwen 으로 바꾼다.
같은 코드가 두 모델에서 동작하는 것이 "모델 = 토큰 목록 → 다음 토큰 점수 함수"라는 관점의 핵심이다.

준비: pip install -r requirements/part2.txt   (transformers 포함. torch 는 docs/ENVIRONMENT.md 의 CUDA 설치 명령)
실행: python part2_llm/next_token_prediction/run_qwen.py
처음 실행할 때 Hugging Face Hub 에서 모델을 내려받는다(0.5B 파라미터 × 2 바이트 ≈ 1GB 로 추정 — 정확한 크기는 모델 카드에서 확인). 인터넷이 없으면 "건너뜀".
RTX 3070(8GB)에서 fp16 추론 가능 범위(docs/LEARNING_ROADMAP.md §3). GPU 가 없으면 CPU 로 느리게 동작한다.
REF: 모델 카드·라이선스 https://huggingface.co/Qwen/Qwen2.5-0.5B (작성 환경 접속 불가 → 미확인)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

LAB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB_DIR))

from solution import entropy_bits, generate, softmax_with_temperature

MODEL_ID = "Qwen/Qwen2.5-0.5B"
PROMPT = "The capital of France is"


def main() -> None:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("건너뜀: transformers 가 없습니다(pip install -r requirements/part2.txt).")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device).eval()
    except Exception as e:  # noqa: BLE001  오프라인·차단·디스크 부족 등 원인이 다양함
        print(f"건너뜀: {MODEL_ID} 를 불러올 수 없습니다 ({type(e).__name__}: {e}).")
        return
    if device == "cuda":
        model = model.half()  # fp16: VRAM 절반

    @torch.no_grad()
    def model_fn(ids: list[int]) -> torch.Tensor:
        out = model(torch.tensor([ids], device=device))
        return out.logits[0, -1].float().cpu()  # 마지막 위치의 다음 토큰 점수 (V,)

    ids = tok(PROMPT)["input_ids"]
    logits = model_fn(ids)
    print(f"device={device}  어휘 크기 V={logits.shape[0]:,}  프롬프트 토큰 {len(ids)}개: {PROMPT!r}")
    for t in (0.5, 1.0, 2.0):
        p = softmax_with_temperature(logits, t)
        top = torch.topk(p, 5)
        cands = ", ".join(f"{tok.decode([int(i)])!r} {float(v):.3f}" for v, i in zip(top.values, top.indices, strict=True))
        print(f"  T={t}: 엔트로피 {entropy_bits(p):.2f} 비트 | 상위 5: {cands}")

    eos = tok.eos_token_id
    print("생성 20 토큰:")
    print("  greedy        :", repr(tok.decode(generate(model_fn, ids, 20, context_window=2048, eos_id=eos))))
    for seed in range(2):
        g = torch.Generator().manual_seed(seed)
        out = generate(model_fn, ids, 20, context_window=2048, eos_id=eos, temperature=0.8, top_p=0.9, generator=g)
        print(f"  T=0.8 p=0.9 s={seed}:", repr(tok.decode(out)))

    print("프롬프트 길이별 다음 토큰 1개 계산 시간(첫 토큰 지연의 대부분 = prefill, 3회 중앙값):")
    base = tok("The quick brown fox jumps over the lazy dog. ")["input_ids"]
    for n in (64, 256, 1024):
        long_ids = (base * (n // len(base) + 1))[:n]
        times = []
        for _ in range(3):
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model_fn(long_ids)
            if device == "cuda":
                torch.cuda.synchronize()  # GPU 는 비동기라 끝날 때까지 기다려야 시간이 맞다
            times.append((time.perf_counter() - start) * 1000)
        print(f"  {n:5d} 토큰: {sorted(times)[1]:8.1f} ms")
    print("※ 이 출력은 사용자 PC 에서 실행한 값만 README/experiments 에 기록하세요.")


if __name__ == "__main__":
    main()

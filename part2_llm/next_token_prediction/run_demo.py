"""P2-01 데모: 작은 bigram 언어 모델로 temperature·top-k·top-p·greedy·컨텍스트 창을 본다(다운로드·API 키 불필요).

실행: python part2_llm/next_token_prediction/run_demo.py [--impl starter]
실제 LLM(Qwen2.5-0.5B)으로 같은 것을 보려면: run_qwen.py (선택, 모델 다운로드 필요)
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch

LAB_DIR = Path(__file__).resolve().parent


def load(impl: str):
    spec = importlib.util.spec_from_file_location(f"p2_01_{impl}", LAB_DIR / f"{impl}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fmt(probs: torch.Tensor, names: list[str]) -> str:
    return ", ".join(f"{n} {float(p):.3f}" for n, p in zip(names, probs, strict=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["solution", "starter"], default="solution")
    m = load(ap.parse_args().impl)

    print("[1] temperature: 후보 [mat, log, sun] 의 점수 [2, 1, 0]")
    z = torch.tensor([2.0, 1.0, 0.0])
    names = ["mat", "log", "sun"]
    for t in (0.5, 1.0, 2.0):
        p = m.softmax_with_temperature(z, t)
        print(f"    T={t:<3}: {fmt(p, names)}   엔트로피 {m.entropy_bits(p):.2f} 비트")

    print("[2] top-k / top-p: 확률 [0.50, 0.30, 0.15, 0.05] 의 후보 a, b, c, d")
    names4 = ["a", "b", "c", "d"]
    logits = torch.log(torch.tensor([0.50, 0.30, 0.15, 0.05]))
    for label, f in (("top-k k=2  ", lambda x: m.top_k_filter(x, 2)), ("top-p p=0.7", lambda x: m.top_p_filter(x, 0.7)),
                     ("top-p p=0.9", lambda x: m.top_p_filter(x, 0.9))):
        print(f"    {label}: {fmt(torch.softmax(f(logits), -1), names4)}")
    peaked = torch.log(torch.tensor([0.95, 0.03, 0.01, 0.01]))
    print(f"    1 등이 0.95 일 때 top-p p=0.9: {fmt(torch.softmax(m.top_p_filter(peaked, 0.9), -1), names4)}  (후보 1 개)")
    print(f"    1 등이 0.95 일 때 top-k k=2  : {fmt(torch.softmax(m.top_k_filter(peaked, 2), -1), names4)}  (후보 2 개 고정)")

    lm = m.load_bigram_lm()
    eos = lm.index[m.EOS]
    prompt = lm.encode("the cat")
    print(f"[3] bigram LM (어휘 {len(lm.vocab)}개, alpha=0.001): 'the cat' 다음 단어 확률 상위 3")
    p = m.softmax_with_temperature(lm(prompt), 1.0)
    top = torch.topk(p, 3)
    print("    " + ", ".join(f"{lm.vocab[int(i)]} {float(v):.3f}" for v, i in zip(top.values, top.indices, strict=True)))

    print("[4] 생성 12 토큰, 컨텍스트 창 4: 'the cat ...'")
    print(f"    greedy (T=0)        : {lm.decode(m.generate(lm, prompt, 12, 4, eos_id=eos))}")
    for seed in range(3):
        g = torch.Generator().manual_seed(seed)
        out = m.generate(lm, prompt, 12, 4, eos_id=eos, temperature=1.0, top_p=0.9, generator=g)
        print(f"    T=1, top-p 0.9, s={seed}: {lm.decode(out)}")
    g = torch.Generator().manual_seed(0)
    out = m.generate(lm, prompt, 12, 4, eos_id=eos, temperature=1.0, top_p=0.9, generator=g)
    print(f"    같은 시드 0 다시     : {lm.decode(out)}  (재현됨)")
    noisy = m.load_bigram_lm(alpha=0.1)  # 비교용: 스무딩만 크게
    p = m.softmax_with_temperature(noisy(prompt), 1.0)
    g = torch.Generator().manual_seed(0)
    out = m.generate(noisy, prompt, 12, 4, eos_id=eos, temperature=1.0, top_p=0.9, generator=g)
    print(f"    (비교) alpha=0.1, s=0  : {noisy.decode(out)}")
    print(f"           'the cat' 다음 엔트로피 alpha=0.001 {m.entropy_bits(m.softmax_with_temperature(lm(prompt), 1.0)):.2f} 비트"
          f" vs alpha=0.1 {m.entropy_bits(p):.2f} 비트")

    print("[5] 컨텍스트 창: 모델이 실제로 받은 입력 길이 (프롬프트 6 토큰, 창 4, 3 토큰 생성)")
    seen: list[int] = []

    def spy(ids: list[int]) -> torch.Tensor:
        seen.append(len(ids))
        return lm(ids)

    m.generate(spy, lm.encode("the cat sat on the mat"), 3, context_window=4)
    print(f"    단계별 입력 길이 = {seen}  (창보다 긴 앞부분은 잘려서 모델이 보지 못함)")


if __name__ == "__main__":
    main()

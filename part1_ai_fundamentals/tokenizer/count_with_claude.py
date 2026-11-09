"""(선택) Claude API 로 실제 토큰 수 세기 — 사용자 PC 에서 API 키가 있을 때 실행.

로컬 BPE(solution.py)는 원리를 보여 주는 작은 토크나이저다. 실제 요금은 Claude 의 토크나이저로 센 토큰으로 매겨지므로,
비용을 정확히 잡으려면 messages.count_tokens 로 그 모델 기준 토큰 수를 받아야 한다(모델을 실행하지 않고 개수만 돌려준다).

준비: pip install -r requirements/part1.txt   그리고 저장소 루트 .env 에  ANTHROPIC_API_KEY=...  (커밋 금지)
실행: python part1_ai_fundamentals/tokenizer/count_with_claude.py
모델 변경: .env 또는 환경 변수 LLM_MODEL (기본 claude-opus-5-5)

REF: https://platform.claude.com/docs/en/build-with-claude/token-counting
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB_DIR))

from solution import PRICES_PER_MTOK, estimate_cost

DEFAULT_MODEL = "claude-opus-5-5"
SAMPLES = {
    "ko": "오늘은 날씨가 맑다",
    "en": "The weather is clear today",
}


def count_claude_tokens(client, model: str, text: str) -> int:
    """text 를 user 메시지 하나로 보냈을 때의 입력 토큰 수(메시지 형식에 쓰이는 몇 토큰 포함)."""
    # 서비스 관점: 같은 글이라도 모델(토크나이저)이 바뀌면 토큰 수가 바뀐다 → 모델을 바꾸면 비용표도 다시 잰다.
    resp = client.messages.count_tokens(
        model=model,
        messages=[{"role": "user", "content": text}],
    )
    return resp.input_tokens


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(LAB_DIR.parents[1] / ".env")
    except ImportError:
        pass  # python-dotenv 가 없으면 환경 변수만 사용
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("건너뜀: ANTHROPIC_API_KEY 가 없습니다(.env 에 넣고 다시 실행).")
        return
    import anthropic

    model = os.environ.get("LLM_MODEL", DEFAULT_MODEL)
    client = anthropic.Anthropic(timeout=30.0, max_retries=2)
    print(f"model={model}")
    for name, text in SAMPLES.items():
        n = count_claude_tokens(client, model, text)
        print(f"{name:3s} chars={len(text):3d} input_tokens={n}")
    if model in PRICES_PER_MTOK:
        n = count_claude_tokens(client, model, SAMPLES["ko"] * 100)
        print(f"ko x100: input_tokens={n}, 1,000 요청(출력 200 토큰 가정) 예상 비용 ${estimate_cost(n, 200, model, 1000):.2f}")


if __name__ == "__main__":
    main()

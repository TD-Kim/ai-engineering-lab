"""(선택) 사용자 PC 측정: Ollama 모델 크기별 TTFT·토큰/초·VRAM 을 CSV 로 기록한다 — RTX 3070(8GB) 판단용.

이 저장소의 클라우드 작성 환경에는 GPU·Ollama 가 없어 실행하지 않았다. 결과는 사용자 PC 에서 실행한 값만 기록한다.

준비 (README §6):
  1) Ollama 설치 후 실행 중인지 확인:  ollama --version
  2) 모델 받기:  ollama pull qwen2.5:0.5b ; ollama pull qwen2.5:1.5b ; ollama pull qwen2.5:3b
     (태그·양자화 기본값은 ollama.com/library/qwen2.5 에서 확인 — 작성 환경 접속 불가로 미확인. 실제 양자화는 결과 CSV 의 quantization 칸)
실행:  python part2_llm/local_llm_serving/measure_local_models.py qwen2.5:0.5b qwen2.5:1.5b qwen2.5:3b
출력:  part2_llm/local_llm_serving/outputs/local_models.csv  (git 에 올라가지 않음 — 표는 experiments 기록에 옮겨 적기)

측정 방법:
  - TTFT·토큰/초: /api/chat 스트리밍(이 실습의 OllamaClient + parse_chat_stream). 첫 요청은 모델 적재 시간이 섞이므로 1 회 버리고 3 회 중앙값.
  - VRAM: /api/ps 의 size_vram(모델이 GPU 에 올린 바이트)과 nvidia-smi 전체 사용량(MiB, 다른 프로그램 포함).
REF: Ollama API 문서 /api/chat, /api/ps (github.com/ollama/ollama docs/api.md, 2026-11-13 확인)
"""
from __future__ import annotations

import csv
import json
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB_DIR))

from solution import (
    DEFAULT_OLLAMA_HOST,
    OllamaClient,
    ProviderError,
    build_chat_payload,
    http_post_lines,
    parse_chat_stream,
    parse_nvidia_smi_memory,
)

PROMPT = "대한민국의 수도와 그 도시의 특징을 세 문장으로 설명해 주세요."
OUT = LAB_DIR / "outputs" / "local_models.csv"


def get_json(path: str):
    with urllib.request.urlopen(DEFAULT_OLLAMA_HOST + path, timeout=5) as r:  # localhost 의 사용자 서버
        return json.load(r)


def gpu_used_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    used = parse_nvidia_smi_memory(out)
    return used[0] if used else None


def measure(model: str, runs: int = 3) -> dict:
    client = OllamaClient(model=model)
    lines = http_post_lines(client.url, build_chat_payload(model, "hi", stream=True, max_tokens=8))
    parse_chat_stream(lines, time.perf_counter())  # 1 회 버림(모델 적재)
    ttfts, speeds = [], []
    for _ in range(runs):
        start = time.perf_counter()
        r = parse_chat_stream(http_post_lines(client.url, build_chat_payload(model, PROMPT, stream=True, max_tokens=200)), start)
        ttfts.append(r.ttft_ms or 0.0)
        speeds.append(r.tokens_per_sec or 0.0)
    loaded = next((x for x in get_json("/api/ps").get("models", []) if x.get("name") == model or x.get("model") == model), {})
    details = loaded.get("details", {})
    return {
        "model": model,
        "parameter_size": details.get("parameter_size", ""),
        "quantization": details.get("quantization_level", ""),
        "size_vram_mib": round(loaded.get("size_vram", 0) / 2**20) if loaded else "",
        "gpu_used_mib_total": gpu_used_mib() or "",
        "ttft_ms_median": round(statistics.median(ttfts), 1),
        "tokens_per_sec_median": round(statistics.median(speeds), 1),
    }


def main() -> None:
    models = sys.argv[1:] or ["qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b"]
    try:
        version = get_json("/api/version").get("version")
    except OSError as e:
        print(f"건너뜀: Ollama 서버({DEFAULT_OLLAMA_HOST})에 연결할 수 없습니다 ({e}). Ollama 를 실행한 뒤 다시 시도하세요.")
        return
    print(f"Ollama {version}, GPU 사용량(측정 전) {gpu_used_mib()} MiB")
    rows = []
    for model in models:
        try:
            row = measure(model)
        except (ProviderError, OSError) as e:
            print(f"  {model}: 실패 ({e}) — `ollama pull {model}` 를 했는지 확인")
            continue
        rows.append(row)
        print(f"  {row}")
    if rows:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with OUT.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"저장: {OUT}  — experiments/P2-01B-20261113-local-llm/README.md 의 표에 옮겨 적으세요.")


if __name__ == "__main__":
    main()

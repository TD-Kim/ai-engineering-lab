"""PyTorch CUDA 설치와 RTX 3070 사용 가능 여부를 확인한다."""
import importlib
import platform
import sys
import time

import torch


def main() -> None:
    print(f"python          : {sys.version.split()[0]} ({platform.system()})")
    print(f"torch           : {torch.__version__}")
    print(f"cuda available  : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("!! CUDA를 쓸 수 없습니다. CPU판 torch가 설치되었는지 확인하세요 (트러블슈팅 참고).")
        return

    props = torch.cuda.get_device_properties(0)
    print(f"cuda (torch)    : {torch.version.cuda}")
    print(f"device          : {props.name}")
    print(f"vram total      : {props.total_memory / 1024**3:.1f} GB")
    print(f"compute cap.    : {props.major}.{props.minor}")
    print(f"bf16 supported  : {torch.cuda.is_bf16_supported()}")

    # 간단한 행렬곱으로 실제 연산과 속도 확인
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        y = x @ x
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    tflops = 10 * 2 * 4096**3 / elapsed / 1e12
    print(f"fp16 matmul     : {elapsed:.3f}s for 10 iters (~{tflops:.1f} TFLOPS)")
    print(f"peak allocated  : {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")
    del x, y

    # 선택 패키지 확인 (설치된 경우에만)
    for name in ("transformers", "sentence_transformers", "faiss", "bitsandbytes", "peft"):
        try:
            mod = importlib.import_module(name)
            print(f"{name:<16}: {getattr(mod, '__version__', 'ok')}")
        except ImportError:
            print(f"{name:<16}: (not installed)")


if __name__ == "__main__":
    main()

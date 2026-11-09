# 환경 구성 (ENVIRONMENT)

대상: Windows · RTX 3070 8GB · RAM 32GB · uv · **Python 3.12**

> 참고 저장소의 `part1\.venv`(Python 3.14.7)는 건드리지 않는다. 이 저장소는 자체 `.venv`를 쓴다.

## 1. 왜 Python 3.12인가

- gensim, faiss 계열 등 일부 패키지가 Python 3.14용 휠을 제공하지 않아 소스 빌드가 필요해진다(Windows에서 특히 어렵다).
- 3.12는 PyTorch·Transformers·Sentence Transformers 등 주요 라이브러리의 휠이 안정적으로 제공된다.

## 2. 설치 순서 (PowerShell)

```powershell
# 0) uv 설치 (이미 있으면 생략) — https://docs.astral.sh/uv/getting-started/installation/
winget install --id=astral-sh.uv -e

# 1) 저장소 루트에서 Python 3.12 가상환경 생성
cd ai-engineering-lab
uv python install 3.12
uv venv --python 3.12 .venv
.\.venv\Scripts\Activate.ps1

# 2) 한글 입출력 깨짐 방지 (현재 세션)
$env:PYTHONUTF8 = "1"
#    영구 적용: setx PYTHONUTF8 1   (새 터미널부터 적용)

# 3) PyTorch를 CUDA 12.8 인덱스에서 "먼저" 설치
#    PyPI의 Windows용 torch는 CPU 전용이므로 반드시 인덱스를 지정한다.
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 4) 공통·개발 의존성
uv pip install -r requirements/base.txt -r requirements/dev.txt

# 5) Part별 의존성은 해당 Part를 시작할 때 설치
uv pip install -r requirements/part1.txt
```

- 3단계를 4·5단계보다 먼저 해야 한다. 다른 패키지가 의존성으로 CPU판 torch를 끌어오는 것을 막기 위해서다.
- 설치 후 [GPU 검증](#5-gpu-검증-스크립트)을 실행해 `cuda available: True`를 확인한다.
- PyTorch 설치 명령은 [공식 선택기](https://pytorch.org/get-started/locally/)에서 최신 값을 확인한다.

## 3. 라이브러리 (도입 시점)

| Part | 라이브러리 | 용도 | 파일 |
|---|---|---|---|
| 공통 | numpy, pandas, matplotlib, tqdm | 계산·표·그래프 | [base.txt](../requirements/base.txt) |
| 공통 | pytest, ruff, ipykernel | 테스트·린트·노트북 | [dev.txt](../requirements/dev.txt) |
| 1 | PyTorch(별도 설치), tokenizers, gensim | 원리 구현, 비교 기준 | [part1.txt](../requirements/part1.txt) |
| 2 | transformers, datasets, accelerate, peft | 소형 LM, LoRA 원리 | [part2.txt](../requirements/part2.txt) |
| 3 | sentence-transformers, faiss-cpu, chromadb, rank-bm25, pypdf, LangChain·LlamaIndex·Haystack | 검색·비교 | [part3.txt](../requirements/part3.txt) |
| 4 | neo4j, neo4j-graphrag, networkx | GraphRAG | [part4.txt](../requirements/part4.txt) |
| 5 | peft, bitsandbytes, trl | Fine-tuning | [part5.txt](../requirements/part5.txt) |
| 6 | fastapi, uvicorn, httpx, sse-starlette | AI 서버 | [part6.txt](../requirements/part6.txt) |
| 6 | Java 21, Spring Boot 3.x, Gradle, Docker Desktop | Spring Boot 서버 | (Python 의존성 아님) |

참고 사항:
- **FAISS:** Windows에서는 `faiss-cpu`를 쓴다. 공식 `faiss-gpu` 휠은 Linux 위주라 Windows 3.12에서 쓰기 어렵다.
  학습용 코퍼스 규모(수만 청크 이하)에서는 CPU로 충분하다.
- **bitsandbytes:** Windows용 공식 휠이 PyPI에 있다(작성 시점 최신 `0.50.2`, `win_amd64` 휠 확인).
  설치 전 [PyPI 페이지](https://pypi.org/project/bitsandbytes/)에서 최신 버전을 다시 확인한다.
- **Neo4j:** Docker Desktop의 `neo4j` 이미지 또는 Neo4j Desktop을 쓴다.
  참고 저장소 스크립트가 DB를 지우므로 **이 저장소의 실험은 별도 DB/컨테이너**를 쓴다.
- **Ollama:** 서비스(Part 6)에서 로컬 LLM 서빙용. 학습(LoRA)은 HF Transformers + PEFT로 한다.

## 4. 모델 · VRAM · 라이선스

> 라이선스는 **사용 전에 반드시 모델 카드에서 다시 확인**한다. 아래는 계획 시점의 정리다.

### 4-1. VRAM 기준 (8GB)

| 규모 | 추론 | 학습 | 비고 |
|---|---|---|---|
| 0.5~1.5B | 가능(fp16/bf16) | LoRA 가능 | 1.5B LoRA는 gradient checkpointing 필수 |
| 3B | 4bit 가능 | QLoRA 가능 | |
| 7~8B | 4bit 가능(여유 적음) | QLoRA 조건부 | 짧은 시퀀스·배치 1, OOM 위험 |
| 13B 이상 | 사실상 불가 | 불가 | |

### 4-2. 추천 모델

| 용도 | 모델 | 라이선스(계획 시점) | 메모 |
|---|---|---|---|
| LLM | Qwen2.5-0.5B / 1.5B / 7B Instruct | Apache-2.0 | LoRA 실습 기본 |
| LLM | **Qwen2.5-3B** Instruct | **Qwen Research License(비상업)** | 학습용만, 서비스 기본값 금지 |
| LLM | Qwen3 계열(0.6B / 1.7B / 4B / 8B) | Apache-2.0 | |
| LLM | EXAONE 계열 | **비상업 라이선스** | 한국어 비교용, 서비스 기본값 금지 |
| 임베딩 | intfloat/multilingual-e5 (small/base/large) | MIT | `query: ` / `passage: ` 접두어 필요 |
| 임베딩 | BAAI/bge-m3 | MIT | 다국어, 긴 입력 |
| Reranker | BAAI/bge-reranker-v2-m3 | Apache-2.0 | Cross-Encoder |
| 서빙 | Ollama | MIT(런타임) | 모델별 라이선스는 별도 |

## 5. GPU 검증 스크립트

설치 직후 아래 코드를 `check_gpu.py`로 저장해 실행한다(`python check_gpu.py`).

```python
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
```

기대 결과: `cuda available: True`, `device: NVIDIA GeForce RTX 3070`, `vram total` 약 8GB.

## 6. 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| `torch.cuda.is_available()`가 `False` | PyPI의 CPU판 torch 설치 | `uv pip uninstall torch torchvision` 후 cu128 인덱스로 재설치 |
| `torch.__version__`에 `+cpu` | 위와 같음 | 위와 같음 |
| NVIDIA 드라이버 관련 오류 | 드라이버가 CUDA 12.8 런타임보다 오래됨 | `nvidia-smi`로 확인 후 최신 드라이버 설치 |
| DataLoader에서 `RuntimeError: ... bootstrapping phase` 또는 무한 재실행 | Windows는 `spawn` 방식이라 메인 가드 필요 | 스크립트 실행 코드를 `if __name__ == "__main__":` 아래로 이동, 또는 `num_workers=0` |
| `UnicodeDecodeError` / 한글 깨짐 | 기본 인코딩 cp949 | `PYTHONUTF8=1` 설정, 파일 열 때 `encoding="utf-8"` 명시 |
| gensim·faiss 설치 시 빌드 오류 | Python 3.13+ 휠 부재 | `.venv`가 3.12인지 `python --version`으로 확인 |
| `CUDA out of memory` | 배치·시퀀스·모델 크기 과다 | 배치 축소, gradient checkpointing, bf16/fp16, 4bit(QLoRA), `torch.cuda.empty_cache()` |
| bitsandbytes 로딩 오류 | 구버전 또는 비공식 빌드 | 최신 공식 휠로 재설치, 버전 재확인 |
| Neo4j 데이터가 사라짐 | 참고 저장소 스크립트가 DB 초기화 | 이 저장소용 별도 DB/컨테이너 사용 |
| HF 모델 다운로드가 C 드라이브를 채움 | 기본 캐시 위치 | `HF_HOME` 환경 변수로 캐시 위치 변경 |

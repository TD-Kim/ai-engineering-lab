# requirements

Part별로 의존성을 나눠, **필요한 단계에서만** 설치한다.

| 파일 | 내용 | 설치 시점 |
|---|---|---|
| [base.txt](base.txt) | numpy, pandas, matplotlib, tqdm | Part 0 |
| [dev.txt](dev.txt) | pytest, ruff, ipykernel | Part 0 |
| [part1.txt](part1.txt) | tokenizers, gensim (+ PyTorch 별도 설치) | Part 1 |
| [part2.txt](part2.txt) | transformers, datasets, accelerate, peft | Part 2 |
| [part3.txt](part3.txt) | sentence-transformers, faiss-cpu, chromadb, rank-bm25, pypdf, 하이브리드 비교용 프레임워크 | Part 3 |
| [part4.txt](part4.txt) | neo4j, neo4j-graphrag, networkx | Part 4 |
| [part5.txt](part5.txt) | peft, bitsandbytes, trl | Part 5 |
| [part6.txt](part6.txt) | fastapi, uvicorn, httpx, sse-starlette | Part 6 |

## 설치

```powershell
.\.venv\Scripts\Activate.ps1
# PyTorch는 CUDA 인덱스에서 먼저 설치 (PyPI Windows판은 CPU 전용)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements/base.txt -r requirements/dev.txt
uv pip install -r requirements/part1.txt   # 해당 Part 시작 시
```

## 규칙

- **torch는 이 파일들에 넣지 않는다.** `-r`로 설치할 때 PyPI의 CPU판이 설치되는 것을 막기 위해서다.
- 하한 버전(`>=`)만 적는다. 실험을 재현해야 할 때는 `uv pip freeze > experiments/<실험>/requirements.lock.txt`로 고정해 기록한다.
- 각 Part 파일은 `-r base.txt`를 포함한다.
- 자세한 배경은 [docs/ENVIRONMENT.md](../docs/ENVIRONMENT.md).

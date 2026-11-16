# Python AI 서버

최종 서비스 흐름: 사용자 → Spring Boot → **Python AI 서버** → RAG/GraphRAG → 벡터DB/Neo4j → LLM

## 구조

```
python_ai_server/
├─ pyproject.toml          # 패키지 정보·의존성 (필요할 때 추가)
├─ src/ai_server/
│  ├─ llm/                 # LLM 호출부: Claude API · 로컬 모델(Ollama)
│  ├─ embeddings/          # 임베딩: OpenAI · 로컬 임베딩 모델
│  ├─ chunking/            # 문서 로딩·청킹
│  ├─ retrieval/           # BM25 · Dense · Hybrid · Rerank
│  ├─ graph/               # GraphRAG: 엔티티·관계 추출, Neo4j, 그래프 탐색
│  ├─ api/                 # FastAPI 라우터 (/api/v1/*)
│  └─ observability/       # 토큰·지연·비용 기록
└─ tests/                  # 테스트 (필요할 때)
```

각 모듈의 역할은 그 폴더의 `__init__.py`에 적혀 있습니다.

## 설치 (저장소 루트의 `.venv` 사용)

```powershell
.\.venv\Scripts\Activate.ps1
uv pip install -e python_ai_server        # ai_server 패키지를 "편집 가능"하게 설치
python -c "import ai_server; print('ok')"
```

- 편집 가능 설치(`-e`)를 하면 코드를 고칠 때마다 다시 설치하지 않아도 됩니다.
- 라이브러리가 필요해지면 `pyproject.toml`의 `dependencies`에 추가하고 `uv pip install -e python_ai_server`를 다시 실행합니다.

## 실행 예

```powershell
python -m ai_server.llm.<모듈이름>         # 모듈에 if __name__ == "__main__": 블록을 두면 이렇게 실행
```

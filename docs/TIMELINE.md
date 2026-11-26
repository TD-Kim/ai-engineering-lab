# 타임라인

커밋할 때마다 `scripts/timeline.py`가 자동으로 기록합니다(언제 · 무엇을 · 어떤 함수를).
각 항목의 **메모**는 직접 채웁니다. 배운 점, 막혔던 점, 다음에 할 일을 적어 두세요.

## 2026-11-17

### 01:38 — 프로젝트 구조 만들기: python_ai_server 골격, 타임라인 자동 기록

- `.githooks/post-commit` (추가)
- `.gitignore` (수정)
- `CLAUDE.md` (수정)
- `README.md` (수정)
- `data/README.md` (추가)
- `docs/PROJECT_PLAN.md` (수정)
- `experiments/README.md` (추가)
- `python_ai_server/README.md` (추가)
- `python_ai_server/pyproject.toml` (추가)
- `python_ai_server/src/ai_server/__init__.py` (추가)
- `python_ai_server/src/ai_server/api/__init__.py` (추가)
- `python_ai_server/src/ai_server/chunking/__init__.py` (추가)
- `python_ai_server/src/ai_server/embeddings/__init__.py` (추가)
- `python_ai_server/src/ai_server/graph/__init__.py` (추가)
- `python_ai_server/src/ai_server/llm/__init__.py` (추가)
- `python_ai_server/src/ai_server/observability/__init__.py` (추가)
- `python_ai_server/src/ai_server/retrieval/__init__.py` (추가)
- `python_ai_server/tests/.gitkeep` (추가)
- `scripts/timeline.py` (추가)
  - 새로 작성: `git`, `file_at`, `definitions`, `changed_files`, `describe_functions`, `build_entry`, `append`, `main`
- `spring_boot_server/README.md` (추가)
- 메모: 

### 01:57 — 2026. 11. 17. 학습방법 개편 및 시작

- `python_ai_server/pyproject.toml` (수정)
- `python_ai_server/src/ai_server/llm/ollama_chat.py` (추가)
- 메모: 

### 08:37 — 힌트 제공 규칙 5단계로 변경 (CLAUDE.md §3)

- `CLAUDE.md` (수정)
- `docs/PROJECT_PLAN.md` (수정)
- 메모: 

### 17:12 — 총괄 세션 인계 메모 추가, 타임라인을 한국 시각(KST) 기준으로 기록

- `CLAUDE.md` (수정)
- `docs/HANDOFF.md` (추가)
- `scripts/timeline.py` (수정)
  - 수정: `build_entry`, `append`
- 메모: 

### 23:48 — 2026. 11. 17. 로컬 LLM 간단한 RAG 테스트

- `python_ai_server/src/ai_server/llm/ollama_chat.py` (수정)
  - 새로 작성: `chat`, `makeDedentStr`
- 메모: Ollama로 qwen2.5:1.5b에 문서를 시스템 프롬프트로 넣어 간단한 RAG 테스트. 문서가 1개일 때는 "문서에 있는 내용으로만 답하라" 규칙을 잘 따랐지만, 3개로 늘리자 규칙을 벗어난 답이 나옴. 다음 할 일: 규칙 강화, 근거(문서 번호) 인용 요구, 더 큰 모델로 바꿔 비교.

### 23:51 — 인계 메모: 로컬 RAG 테스트 결과와 다음 단계(규칙 강화) 기록

- `docs/HANDOFF.md` (수정)
- 메모: 

## 2026-11-18

### 13:38 — 학습 노트 스킬 추가: /note로 대화 속 개념 정리, /note 메모로 타임라인 메모 채우기

- `.claude/skills/note/SKILL.md` (추가)
- `CLAUDE.md` (수정)
- 메모: 

### 13:42 — 인계 메모: /note 학습 노트 스킬 기록

- `docs/HANDOFF.md` (수정)
- 메모: 

## 2026-11-26

### 09:42 — 인계 메모: 폰 학습(사이드 채팅) 방식과 주제 목록 추가

- `docs/HANDOFF.md` (수정)
- 메모: 

### 10:12 — 학습 노트: 규칙의 위치와 공정한 비교 조건 (폰 학습 1-2)

- `docs/HANDOFF.md` (수정)
- `docs/notes/2026-11-26.md` (추가)
- 메모: 

### 10:22 — 학습 노트: 작은 모델이 규칙을 놓치는 이유, 문서 수와 주의 몫 (폰 학습 1-1)

- `docs/HANDOFF.md` (수정)
- `docs/notes/2026-11-26.md` (수정)
- 메모: 

### 10:32 — 학습 노트: 문서 구분자와 프롬프트 인젝션 (폰 학습 1-3)

- `docs/HANDOFF.md` (수정)
- `docs/notes/2026-11-26.md` (수정)
- 메모: 

### 10:50 — 학습 노트: 근거 인용과 가짜 인용 검사 (폰 학습 1-4)

- `docs/HANDOFF.md` (수정)
- `docs/notes/2026-11-26.md` (수정)
- 메모: 

### 10:58 — 학습 노트: 토큰과 비용 계산, 입력이 비용의 대부분이 되는 이유 (폰 학습 2-1)

- `docs/HANDOFF.md` (수정)
- `docs/notes/2026-11-26.md` (수정)
- 메모: 

### 11:08 — 학습 노트: 임베딩, 검색 성공과 답변 품질을 나눠 보기 (폰 학습 2-2)

- `docs/HANDOFF.md` (수정)
- `docs/notes/2026-11-26.md` (수정)
- 메모: 

### 11:16 — 학습 노트: 코사인 유사도 손계산, 기준선이 못 거르는 것 (폰 학습 2-3)

- `docs/HANDOFF.md` (수정)
- `docs/notes/2026-11-26.md` (수정)
- 메모: 

### 11:29 — 학습 노트: 정규화와 내적·코사인 순위 차이 (폰 학습 2-4)

- `docs/HANDOFF.md` (수정)
- `docs/notes/2026-11-26.md` (수정)
- 메모: 

### 16:32 — 학습 노트: 타임아웃·재시도·폴백과 구간별 시간 예산 (폰 학습 3-1)

- `docs/HANDOFF.md` (수정)
- `docs/notes/2026-11-26.md` (수정)
- 메모: 

### 16:34 — 인계 메모: 규칙 강화 실험 후보와 다음 폰 학습 주제(4~6번 묶음) 추가

- `docs/HANDOFF.md` (수정)
- 메모: 

### 16:43 — 학습 노트: 청킹 크기의 맞바꿈과 겹침 (폰 학습 4-1)

- `docs/HANDOFF.md` (수정)
- `docs/notes/2026-11-26.md` (수정)
- 메모: 

### 16:52 — 학습 노트: BM25의 TF·IDF·길이 보정과 0점의 함정 (폰 학습 4-2)

- `docs/HANDOFF.md` (수정)
- `docs/notes/2026-11-26.md` (수정)
- 메모: 

### 16:59 — 학습 노트: Hybrid 검색, min-max 가중합과 RRF (폰 학습 4-3)

- `docs/HANDOFF.md` (수정)
- `docs/notes/2026-11-26.md` (수정)
- 메모: 

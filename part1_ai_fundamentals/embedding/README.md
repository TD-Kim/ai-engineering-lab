# P1-02 Embedding — 문장을 벡터로, 가까움을 숫자로

| 항목 | 값 |
|---|---|
| 실습 ID | P1-02 (메인) |
| 소속 | [Part 1 — AI 원리](../README.md) |

> **최종 서비스에서의 위치:** **Dense 검색**의 핵심 부품입니다. Python AI 서버가 문서 청크와 질의를 벡터로 바꿔 벡터DB(FAISS/ChromaDB)에서 top-k를 찾고(Part 3 P3-03), GraphRAG의 Vector 검색(Part 4)과 임베딩 fine-tuning 비교(Part 5)가 모두 이 위에 올라갑니다.

> 이 문서에 나오는 `함수·파일 이름`(예: `run_demo.py`)은 이전 실습 코드(커밋 `aa46d2f`)의 이름입니다. 그 코드는 지웠고 개념만 남겼습니다. 옛 코드가 필요하면 `git checkout aa46d2f`(복구 방법은 [루트 README](../../README.md#지금-어디를-보면-되나-학습-방식)).

> 방향: **개념은 깊게, 구현은 얕게.** 임베딩 모델은 API(OpenAI)나 **로컬 모델**(sentence-transformers, Ollama)에 맡기고, 정규화·코사인·top-k 검색과 "임베딩을 서비스에 쓸 때의 비용·운영 판단"을 익힙니다. 공급자는 `.env`의 `EMBEDDING_PROVIDER`로 고르며, 키가 없어도 로컬로 할 수 있습니다.

## 1. 목표

- L2 정규화가 "길이를 버리고 방향만 남기기"임을 손계산으로 설명한다.
- 정규화한 벡터의 내적이 코사인 유사도임을 이해하고, 행렬곱 `@`으로 유사도 표를 한 번에 만든다.
- 임베딩 함수를 **주입**받아 배치로 임베딩한다(서비스: OpenAI API, 테스트: 가짜 함수).
- 질의와 가장 가까운 문서 k개를 찾는다 — **Part 3 Dense 검색의 원형**.
- (선택) 같은 텍스트를 다시 임베딩하지 않는 캐시로 비용을 줄인다.

## 2. 선수지식

- P1-04의 shape·브로드캐스팅(이 실습은 numpy를 쓰지만 규칙은 같습니다. torch `keepdim` = numpy `keepdims`)
- P1-01의 토큰·비용 개념(임베딩 API도 입력 토큰으로 과금)
- 벡터의 내적: `[a, b] · [c, d] = a×c + b×d`

## 3. 핵심 개념

| 개념 | 한 줄 설명 |
|---|---|
| 임베딩 | 문장(또는 단어)을 고정 길이 숫자 목록(벡터)으로 바꾼 것. 뜻이 비슷하면 방향이 비슷하도록 학습된 모델이 만든다 |
| 차원(D) | 벡터의 숫자 개수. 클수록 표현력·저장 비용이 함께 커진다 |
| L2 노름 | 벡터의 길이 = √(성분 제곱합) |
| 정규화 | 길이로 나눠 길이를 1로. 방향만 남는다 |
| 코사인 유사도 | 두 벡터 사이 각도의 코사인(1 같은 방향, 0 직각, −1 반대). 정규화 벡터의 내적과 같다 |
| top-k 검색 | 유사도가 가장 큰 k개 문서를 고르기 |
| 주입(injection) | 임베딩 함수를 코드 안에서 정하지 않고 바깥에서 넣어 주기 → 같은 코드로 API·로컬·가짜 함수 모두 사용 |
| 임베딩 전용 모델 | "비슷한 문장은 가깝게, 다른 문장은 멀게" 학습한 모델(e5·bge·OpenAI text-embedding). 채팅용 LLM과 다르다(§4-4) |
| `EMBEDDING_PROVIDER` | `fake` / `openai` / `sentence_transformers` / `ollama`. 공급자 선택 함수가 골라 같은 모양의 `embed_fn`을 만든다 |
| `query: `·`passage: ` 접두어 | e5 계열이 질문과 문서를 구분하려고 입력 앞에 붙이는 문구. 모델마다 규칙이 다르다 |

## 4. 내부 원리

### 4-1. 정규화와 브로드캐스팅 (손계산)

`[3, 4]`의 길이는 √(3² + 4²) = √25 = **5**. 각 성분을 5로 나누면 `[0.6, 0.8]` (길이 1). `[6, 8]`도 10으로 나누면 `[0.6, 0.8]` — **방향이 같으면 정규화 결과가 같습니다.**

여러 벡터를 한 번에 정규화할 때 `keepdims`가 결과를 가릅니다.

```
x = [[3, 4],          norm(x, axis=-1, keepdims=True)  → [[ 5],     모양 (2, 1)
     [6, 8]]                                               [10]]
                      norm(x, axis=-1)                 → [5, 10]    모양 (2,)

x / [[5],[10]]  → 가로로 복사 [[5, 5],[10,10]] 로 맞춰 나눔 → [[0.6, 0.8], [0.6, 0.8]]   ✔ 행마다 자기 길이
x / [5, 10]     → 세로로 복사 [[5,10],[5, 10]] 로 맞춰 나눔 → [[0.6, 0.4], [1.2, 0.8]]   ✘ 오류 없이 틀린 값
```

### 4-2. 코사인 = 정규화 후 내적, 행렬곱으로 한 번에

`[0.6, 0.8] · [1, 0] = 0.6×1 + 0.8×0 = 0.6` → 두 벡터 사이 각도의 코사인이 0.6.

질의 N개 × 문서 M개의 모든 쌍은 `A @ B.T` 한 번으로 구합니다. A가 (N, D), B가 (M, D)이면 B.T는 (D, M)이고 결과는 (N, M)입니다. `[i, j]` 칸은 A의 i번째와 B의 j번째 벡터의 내적입니다.

```
A = [[0.6, 0.8]]                 B = [[1, 0],      A @ B.T = [[0.6×1 + 0.8×0,  0.6×0 + 0.8×1]]
                                      [0, 1]]               = [[0.6, 0.8]]
```

### 4-3. 글자 겹침만 보는 가짜 임베딩 (예시)

이웃한 두 글자(2-gram)를 해시해 256칸 중 하나에 세는 방식입니다(feature hashing). **글자 겹침**만 보므로 뜻은 모릅니다. 테스트를 키 없이 결정적으로 돌리기 위한 도구이고, 실제 임베딩 모델과의 차이는 §8에서 결과로 확인합니다.

### 4-4. 로컬 임베딩 공급자 (`EMBEDDING_PROVIDER` 설계)

임베딩 함수를 `embed_fn: list[str] -> list[list[float]]` 모양 하나로 정해 두면 배치 임베딩·top-k 검색 코드는 공급자를 바꿔도 **그대로**입니다(P2-01B의 `LLM_PROVIDER`와 같은 설계). 아래 설계는 이전 실습 코드(커밋 `aa46d2f`)에서 쓰던 것이고, 직접 구현할 때 같은 구조를 권장합니다.

| 공급자 | 어디서 계산하나 | 필요한 것 | 비고 |
|---|---|---|---|
| `fake` | 이 PC(글자 2-gram 해시) | 없음 | 뜻을 모른다. 테스트·기본값 |
| `openai` | OpenAI 서버 | `OPENAI_API_KEY`(유료) | 모델 `EMBEDDING_MODEL`, 기본 `text-embedding-3-small` |
| `sentence_transformers` | 이 PC(CPU/GPU) | `uv pip install sentence-transformers`, 첫 실행 때 모델 다운로드 | 기본 `intfloat/multilingual-e5-small`, 예: `BAAI/bge-m3`. 장치는 라이브러리가 자동 선택 |
| `ollama` | 이 PC의 Ollama 서버 | `ollama serve` + `ollama pull <임베딩 모델 태그>` | 주소 `OLLAMA_BASE_URL`(P2-01B와 같은 변수). 모델 태그는 [ollama.com 라이브러리](https://ollama.com/library)에서 확인 — 이 저장소는 태그를 정하지 않습니다 |

**왜 채팅용 LLM(Qwen 등)을 임베딩에 쓰면 안 되나?** 채팅용 LLM은 "다음 토큰 맞히기"로 학습되어 문장 하나를 벡터 하나로 요약하는 법을 배우지 않았습니다. 은닉 상태를 평균 내 써도 검색 품질이 나쁩니다. 검색에는 임베딩 전용 모델을 씁니다.

**`encode()`에 접두어를 직접 붙이는 이유와 주의**: sentence-transformers에는 `encode_query()`·`encode_document()`가 있어, 모델 설정의 `prompts`에 `"query"` 프롬프트가 정의돼 있으면 접두어를 자동으로 붙입니다(패키지 6.3.0 소스 확인). 이 실습은 원리를 보이려고 `encode()`에 `embedding_prefix()`로 **직접** 붙이므로, 두 방식을 섞으면 접두어가 두 번 붙을 수 있습니다 — 하나만 쓰세요.

**꼭 지킬 규칙**
1. **문서와 질문은 같은 모델로.** 모델이 다르면 벡터 공간이 달라 점수가 의미 없습니다.
2. **모델을 바꾸면 전부 재임베딩.** 차원이 같아도 공간이 다릅니다. 벡터와 함께 모델 이름·차원을 저장합니다.
3. **접두어 규칙을 모델에 맞춘다.** e5 계열은 질문 `query: …`, 문서 `passage: …`. 이전 실습 코드의 `embedding_prefix()`는 모델 이름에 `e5`가 있으면 붙이고, bge-m3 등은 붙이지 않습니다. 새 모델은 모델 카드에서 규칙을 확인하세요.
4. **차원·정규화가 모델마다 다르다.** 정규화 후 내적으로 구하는 코사인 유사도는 항상 다시 정규화하면 결과가 같지만, 벡터 DB에 내적(inner product)으로 저장하려면 정규화 여부를 명시해야 합니다. `sentence_transformers` 공급자는 `normalize_embeddings=True`로 부릅니다.

**근거 (확인 일자 2026-11-16)**

| 항목 | 내용 | 근거 |
|---|---|---|
| Ollama 임베딩 엔드포인트 | `POST /api/embed`, 요청 `{"model": …, "input": 문자열 또는 문자열 목록}`, 응답 `{"embeddings": [[…], …], …}` | 공식 저장소 `docs/api.md` "Generate Embeddings"·`docs/capabilities/embeddings.mdx`(raw.githubusercontent.com에서 직접 열람) |
| 정규화 | `/api/embed`는 L2 정규화된(길이 1) 벡터를 돌려준다 | 같은 `embeddings.mdx` Note |
| 구 엔드포인트 | `POST /api/embeddings`(단수 `prompt`)는 `/api/embed`로 대체됨 | `docs/api.md` "Generate Embedding" 절의 Note |
| 차원 | "모델에 따라 다르다(보통 384–1024)" | `embeddings.mdx`. 이 저장소는 특정 모델의 차원을 적지 않는다(측정 필요) |
| sentence-transformers | `SentenceTransformer(model_name_or_path, device=None)`(None이면 GPU 사용 가능 여부를 확인해 자동 선택), `encode(sentences, batch_size, normalize_embeddings, convert_to_numpy)` | PyPI에서 받은 `sentence-transformers 6.3.0` 패키지 소스·docstring(작성 환경에서 실행하지는 않음 — 설치하지 않음) |
| bge-m3 접두어 | 공식 FlagEmbedding README 모델표의 "query instruction for retrieval" 칸이 비어 있다 → 접두어 없음으로 처리 | `FlagOpen/FlagEmbedding` README(raw.githubusercontent.com) |
| e5 접두어 | `query: `/`passage: ` | Hugging Face 모델 카드 — 작성 환경에서 huggingface.co 접속 불가 → **미확인**. 공식 `microsoft/unilm` e5 README에는 접두어 설명이 없음을 확인했다. 쓰기 전에 모델 카드를 확인하세요 |
| 로컬 모델의 한국어 품질·속도 | — | **사용자 PC 측정 필요**(실습에 결과를 적지 않음) |

## 서비스 관점

| 질문 | 판단 |
|---|---|
| 저장 비용은? | 문서 수 × 차원 × 4바이트(float32). 예) 100만 개 × 1,536차원 ≈ 6.1GB. 차원을 줄이면(모델 선택·차원 축소) 저장·검색 비용이 줄지만 품질을 측정해야 한다 |
| 왜 정규화해서 저장하나? | 내적만으로 코사인이 되어 계산이 단순해지고, 벡터 DB의 inner product 와 cosine 결과가 같아진다 |
| 임베딩 모델을 바꾸면? | 벡터 공간이 달라 기존 벡터와 비교할 수 없다 → **문서 전체 재색인**. 벡터와 함께 모델 이름·차원을 저장한다 |
| 비용은 어디서 나오나? | 문서: 색인할 때 1회(문서가 바뀔 때만 다시). 질의: **요청마다** → 트래픽에 비례. 캐시로 반복 질의 비용을 줄인다 |
| 한 번에 몇 개씩 보내나? | 배치로 보낸다. 1개씩 보내면 왕복 지연이 쌓이고 호출 한도에 걸린다 |
| API냐 로컬이냐? | **API**: 키·호출 비용·네트워크 지연이 들고, 문서 내용이 외부 서버로 나간다. **로컬**: 호출 비용 0, 데이터가 PC(서버) 밖으로 나가지 않아 프라이버시에 유리하지만, 모델 내려받기·GPU/CPU 성능이 색인 시간과 질의 지연을 정하고 운영(버전·장애)을 직접 맡는다. 어느 쪽이든 품질은 내 데이터로 측정한다 |
| 한국어도 잘 되나? | **가정하지 말고 측정한다.** 내 도메인의 한국어 질의·정답 문서 쌍을 모아 top-k 적중률을 잰다(Part 3 평가). 한국어 질의 → 영어 문서 같은 교차 언어도 따로 확인 |
| 가격은? | 공식 가격 페이지에서 확인: https://platform.openai.com/docs/pricing (작성 시점 2026-11, 작성 환경에서 접속 불가 → 수치 미기재·미확인). 실제 사용량은 응답의 `usage.prompt_tokens` |

## 5. 수식

- L2 노름: ‖x‖ = √(x₁² + … + x_D²), 정규화: x̂ = x / max(‖x‖, ε)
- 코사인: cos(a, b) = (a · b) / (‖a‖ ‖b‖) = â · b̂
- 유사도 표: S = Â B̂ᵀ, S ∈ ℝ^{N×M}
- 저장 바이트 = N × D × 4 (float32)
- 배치 호출 수 = ⌈N / batch_size⌉

## 6. 자주 나는 오류

| 증상 | 원인 | 해결 |
|---|---|---|
| 정사각 행렬에서만 정규화 값이 이상함 | `keepdims` 누락 → 열 방향으로 나눔 | `np.linalg.norm(x, axis=-1, keepdims=True)` |
| 유사도에 `nan` | 0 벡터를 0으로 나눔 | `np.maximum(norm, eps)` |
| `matmul` 모양 오류 `(1,D) @ (M,D)` | 전치 누락 | `b.T` |
| top-k 결과가 실행마다 순서가 다름(동점) | 정렬이 안정적이지 않음 | `np.argsort(-scores, kind="stable")` |
| 문서와 벡터가 어긋남 | 배치 응답 개수·순서 확인 안 함 | 개수 검사, OpenAI 응답은 `index`로 정렬 |
| 모델을 바꿨더니 검색 품질이 무너짐 | 다른 모델의 벡터끼리 비교 | 전체 재색인, 벡터와 모델 이름을 함께 저장 |
| `ValueError: ... OpenAI 모델입니다` | 공급자는 바꿨는데 `EMBEDDING_MODEL`에 `text-embedding-3-small`이 남음 | `.env`의 모델 이름을 그 공급자 것으로 |
| 로컬 모델로 바꿨더니 검색이 이상함 | 문서는 예전 모델 벡터, 질문은 새 모델 벡터 | 문서 전체 재임베딩 |
| e5 모델인데 점수가 낮게 나옴 | `query: `/`passage: ` 접두어 누락 | 모델 이름에 `e5`가 있을 때 붙이도록 직접 구현했다면 변형 모델 이름을 확인하세요. **이런 함수는 기본 e5 규칙(`query:`/`passage:`)만 처리합니다. e5-instruct 같은 변형이나 다른 임베딩 모델은 각 모델 카드의 형식을 따르세요(미확인)** |
| `Ollama 서버에 연결할 수 없습니다` | 서버가 꺼져 있거나 주소 틀림 | `ollama serve`, `OLLAMA_BASE_URL`(클라이언트 주소 — 서버용 `OLLAMA_HOST`와 다른 변수) |
| `sentence_transformers 패키지가 없습니다` | 선택 의존성 미설치 | `uv pip install sentence-transformers` |

## 7. 과제 (직접 만들어 볼 것)

1. 4-3의 글자 2-gram 해시 가짜 임베딩을 직접 만들어 `dim`을 32 / 256 / 4096으로 바꿔 가며 `강아지와 고양이` 질의의 점수가 어떻게 변하는지 기록하세요. 해시 충돌과 차원의 관계를 설명해 보세요.
2. (API 키가 있으면) OpenAI 임베딩으로 세 질의의 top-3을 기록하고 가짜 임베딩과 비교하세요. 출력되는 `prompt_tokens`로 비용을 계산해 보세요(단가는 공식 페이지).
3. 문서 50만 개, 하루 질의 2만 건인 검색 서비스에서 (a) 1,536차원 float32 저장 크기, (b) 문서 색인 1회와 질의 한 달의 임베딩 토큰 수를 추정하세요(문서 평균 300토큰, 질의 평균 20토큰). 어느 쪽이 비용을 지배하나요?
4. 캐시 키에 모델 이름을 넣지 않으면 어떤 사고가 날 수 있는지, 그리고 캐시를 언제 비워야 하는지 설계로 답하세요.
5. (로컬 임베딩이 가능하면) 가짜 임베딩과 로컬 공급자로 세 질의의 top-3을 비교하세요. `강아지와 고양이`가 #45·#46을 찾았나요? 문서 임베딩 시간과 차원도 기록하세요. 로컬 모델 두 개(예: e5-small, bge-m3)를 비교한다면 같은 조건(같은 문서·질의·장치)으로.
6. e5 모델에서 접두어를 일부러 빼 보세요(접두어를 빈 문자열로). 점수와 순위가 어떻게 변하나요? 왜 모델마다 규칙이 다를까요?

## 8. 체크리스트

- [ ] `[3, 4] → [0.6, 0.8]` 을 손으로 계산하고, `keepdims`가 빠지면 무엇이 틀리는지 설명할 수 있다
- [ ] 정규화 벡터의 내적 = 코사인임을 안다
- [ ] 임베딩 모델을 바꾸면 재색인이 필요한 이유를 안다
- [ ] 문서 임베딩 비용과 질의 임베딩 비용이 어떻게 다르게 쌓이는지 설명할 수 있다
- [ ] 한국어 검색 품질은 내 데이터로 측정해야 한다는 것을 안다

## 9. 다음 단계 연결

- **P1-03 Word2Vec**: "비슷한 문맥의 단어는 비슷한 벡터"라는 임베딩 학습의 출발점을 짧게 실행해 봅니다.
- **Part 3 P3-03/04**: 이 실습의 `top_k`와 `make_embed_fn`(공급자 교체)이 Dense 검색이 되고, BM25·Hybrid·Rerank와 품질·지연·비용을 비교합니다.
- **Part 5**: 임베딩 fine-tuning 전후를 OpenAI 기준선과 비교해 "언제 값어치가 있는가"를 판단합니다.
- 원리를 더 파고 싶다면 선택 과제 [advanced/](./advanced/README.md) (임베딩 층·위치 임베딩 직접 구현, P1-02A).

## REF

- numpy `linalg.norm`: https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html (작성 환경에서 접속 불가 → 미확인)
- numpy 브로드캐스팅: https://numpy.org/doc/stable/user/basics.broadcasting.html (미확인)
- OpenAI 임베딩 가이드: https://platform.openai.com/docs/guides/embeddings (작성 환경에서 접속 불가 → 미확인)
- OpenAI 가격: https://platform.openai.com/docs/pricing (미확인, 사용 전 확인)
- OpenAI Python SDK `client.embeddings.create(model, input)` — 설치된 openai 3.49.0 패키지의 시그니처로 확인
- Ollama 임베딩 API(2026-11-16 열람): https://raw.githubusercontent.com/ollama/ollama/main/docs/api.md (Generate Embeddings), https://raw.githubusercontent.com/ollama/ollama/main/docs/capabilities/embeddings.mdx
- FlagEmbedding(bge-m3) 모델표(2026-11-16 열람): https://raw.githubusercontent.com/FlagOpen/FlagEmbedding/master/README.md
- sentence-transformers 6.3.0: PyPI에서 받은 패키지의 `SentenceTransformer`·`encode` docstring(2026-11-16, 설치·실행은 하지 않음)
- e5 접두어: Hugging Face 모델 카드 `intfloat/multilingual-e5-small`(작성 환경에서 접속 불가 → 미확인, 사용 전 확인)

# P1-01 Tokenizer (메인) — 실행 기록

- 실습 ID: P1-01 (메인)
- 날짜: 2026-11-09
- 실행 위치: 클라우드 작업 환경(CPU). API 키·인터넷(Hugging Face) 없음 → 선택 스크립트 2개는 **사용자 PC에서 실행 필요**

## 목적
로컬 바이트 단위 BPE로 (1) 한국어/영어 글자당 토큰 차이, (2) 모델별 비용 차이, (3) 청크 오버랩이 저장 토큰을 얼마나 늘리는지, (4) 컨텍스트 예산으로 자를 때 예산을 얼마나 쓰는지 확인한다.

## 데이터셋
- `part1_ai_fundamentals/data/corpus_ko_en.txt` (저장소 포함, 2,492자, 한국어 위주 + 영어 일부)
- 비교 문장: `오늘은 날씨가 맑다`(10자), `The weather is clear today`(26자), `이모지 😀`(5자)

## 모델
- 토크나이저: Hugging Face `tokenizers` BPE, `ByteLevel` 사전 분리, 위 말뭉치로 학습, vocab 500
- LLM 호출 없음

## 하이퍼파라미터
| 이름 | 값 |
|---|---|
| vocab_size | 500 |
| 청크 max_tokens | 200 |
| 청크 overlap | 0, 50 |
| 컨텍스트 창 / 예약 | 1,000 / 300 |
| 비용 가정 | 입력 = 말뭉치 전체(1,682 토큰), 출력 300 토큰, 요청 1,000 건 |

## 환경
| 항목 | 값 |
|---|---|
| Python | 3.12.13 |
| tokenizers | 0.23.3 |
| anthropic | 2.6.0 (설치만, 키 없음) |

## GPU 사용량
해당 없음(CPU만 사용, 수 초).

## 토큰 사용량·비용
- 실제 API 호출 없음 → 실제 비용 $0.
- 아래 비용은 **로컬 토큰 수 × 공식 단가(작성 시점 2026-10 기준)** 로 계산한 추정치다. Claude 토크나이저 기준 수치가 아니다.

## 결과
명령: `python part1_ai_fundamentals/tokenizer/run_demo.py --impl solution` (실제 출력)

```
[1] 말뭉치: chars=2492, tokens=1682 (local BPE, vocab 500)
[2] 언어별 비교
    ko    chars= 10 tokens= 15 chars/token=0.67
    en    chars= 26 tokens= 15 chars/token=1.73
    emoji chars=  5 tokens=  9 chars/token=0.56
[3] 비용 추정: 입력 = 말뭉치 전체, 출력 300 토큰, 요청 1,000 건 (로컬 토큰 수 기준 근사, 단가 작성 시점 2026-10)
    claude-opus-5-5    $   12.73
    claude-sonnet-5-5  $    6.36
    claude-haiku-5-5   $    0.32
[4] 청크 (max_tokens=200)
    overlap=  0: 9 조각, 조각 토큰 합 1682 (원문 1682, 1.00배)
    overlap= 50: 11 조각, 조각 토큰 합 2182 (원문 1682, 1.30배)
    조각 0 끝: '인다.\n학습이 잘 되면 손실 값이 조금씩 줄어'
    조각 1 앞: '\n작은 모델이라도 직접 학습해 보면 원리가 잘'
[5] 컨텍스트 창 1,000 토큰, 예약 300 토큰
    남긴 글자 622/2492, 토큰 698, 끝: '으로 비교해야 공정한 실험이 된다.\n'
```

추가 측정(같은 토크나이저):
- 말뭉치 1,682 토큰 중 글자를 쪼개지 않고 자를 수 있는 위치는 1,360곳(시작 포함), 자를 수 없는 최소 덩어리는 최대 5 토큰.
- 겹침 0 청크(max 37)를 이어 붙이면 원문과 정확히 같다(테스트 `test_chunk_by_tokens_no_overlap_reconstructs`).

선택 스크립트:
| 스크립트 | 이 환경 결과 |
|---|---|
| `count_with_claude.py` | `건너뜀: ANTHROPIC_API_KEY 가 없습니다` — 사용자 PC에서 실행 필요 |
| `pretrained_tokenizer.py` | `건너뜀: Qwen/Qwen2.5-0.5B 토크나이저를 받을 수 없습니다 (ProxyError)` — 사용자 PC에서 실행 필요 |

## 비교
- 같은 15 토큰에 영어 26자, 한국어 10자 → 이 작은 토크나이저에서 한국어 글자당 토큰이 약 2.6배.
- 모델만 바꿔도 같은 트래픽 비용이 약 40배 차이(opus $12.73 vs haiku $0.32).
- 오버랩 50/200: 이론 1.33배, 실제 1.30배(마지막 조각이 짧음).

## 개선점 / 다음에 확인할 것
- 사용자 PC에서 `count_with_claude.py`로 Claude 기준 토큰 수를 재어 로컬 추정과의 차이를 기록(예상치를 미리 쓰지 않음).
- `pretrained_tokenizer.py`로 다국어 대형 어휘 토크나이저의 한국어 글자당 토큰을 측정.
- vocab_size 300/500/1000 비교(README 과제 1).

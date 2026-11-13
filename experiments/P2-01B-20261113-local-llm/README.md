# P2-01B 로컬 LLM 서빙과 공급자 교체 (메인) — 실행 기록

- 실습 ID: P2-01B (메인)
- 날짜: 2026-11-13
- 실행 위치: 클라우드 작업 환경(CPU, **GPU·Ollama 없음**). 아래 "결과(데모)"는 가짜 공급자·예시 데이터이며, **측정 표는 사용자 PC(RTX 3070)에서 채울 칸**이다. 측정한 척 숫자를 넣지 않는다.

## 목적
(1) 공통 인터페이스로 공급자를 설정만으로 교체할 수 있는지, (2) 스트리밍 파싱으로 TTFT·토큰/초를 계산하는지, (3) 폴백·사용량 기록이 동작하는지(데모), (4) RTX 3070에서 Qwen2.5 크기별로 서비스 가능한 속도·메모리인지(사용자 PC 측정)를 확인한다.

## 데이터셋
- 측정 프롬프트(measure_local_models.py): "대한민국의 수도와 그 도시의 특징을 세 문장으로 설명해 주세요.", 출력 상한 200토큰

## 모델
| 모델(Ollama 태그) | 라이선스 | 비고 |
|---|---|---|
| qwen2.5:0.5b | Apache-2.0 | 서비스 기본값 가능 |
| qwen2.5:1.5b | Apache-2.0 | |
| qwen2.5:3b | Qwen Research License(비상업) | 측정·학습용만 |
| (여유가 있으면) qwen2.5:7b | Apache-2.0 | 8GB에서 여유 적음 |

라이선스 출처: 이 저장소 docs/ENVIRONMENT.md §4-2(계획 시점), 사용 전 모델 카드 재확인. 태그 이름·기본 양자화는 ollama.com/library/qwen2.5에서 확인(작성 환경 접속 불가 → 미확인).

## 하이퍼파라미터
| 이름 | 값 |
|---|---|
| num_predict | 200 |
| 반복 | 적재용 1회 버림 + 3회 중앙값 |
| OLLAMA_NUM_PARALLEL | 기본값(1) |

## 환경
| 항목 | 작성 환경 | 사용자 PC |
|---|---|---|
| OS | Linux (클라우드) | Windows (측정 필요) |
| GPU | 없음 | RTX 3070 8GB |
| Ollama | 미설치 | (측정 시 `ollama --version` 기록) |
| Python | 3.12.13 | |

## GPU 사용량 · 속도 — 사용자 PC 측정 필요

`python part2_llm/local_llm_serving/measure_local_models.py qwen2.5:0.5b qwen2.5:1.5b qwen2.5:3b` 의 출력(outputs/local_models.csv)을 옮겨 적는다.

| 모델 | parameter_size | quantization | size_vram (MiB) | nvidia-smi 전체 (MiB) | `ollama ps` PROCESSOR | TTFT 중앙값 (ms) | 토큰/초 중앙값 |
|---|---|---|---|---|---|---|---|
| qwen2.5:0.5b | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 |
| qwen2.5:1.5b | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 |
| qwen2.5:3b | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 |

판단(측정 후 작성): RTX 3070에서 서비스 기본값으로 쓸 크기 = ___ (근거: 100% GPU 여부, 토큰/초, 남는 VRAM, 라이선스)

## 토큰 사용량·비용
- 이 기록의 데모는 API 호출 없음 → $0.
- 로컬 비용은 README §8 [6]처럼 가정값으로만 계산했다. 측정한 토큰/초와 실제 전력(가능하면 측정)으로 다시 계산할 것.

## 결과(데모)
명령: `python part2_llm/local_llm_serving/run_demo.py --impl solution` (작성 환경 실제 출력, 가짜 공급자 · [2]는 Ollama 문서 예시 숫자 · [6]은 가정값)

```
[1] Ollama /api/chat 요청 본문 (TODO 1)
    {"model": "qwen2.5:0.5b", "messages": [{"role": "user", "content": "RAG 를 한 문장으로"}], "stream": true, "options": {"num_predict": 128}}
[2] 스트림 파싱 (TODO 2) — 예시 데이터: 조각 3개 + 마지막 줄(Ollama 문서 예시 숫자 eval_count 282, eval_duration 4,535,599,000ns)
    text='검색한 문서로 답한다'  TTFT 350 ms  전체 4890 ms  출력 282 토큰  62.2 토큰/초
[3] 공통 인터페이스 (TODO 3): LLM_PROVIDER=fake
    fake/fake-echo: '가짜 답변: 로컬 LLM 과 API 의 차이를 한 줄로'  (입력 8, 출력 10 토큰)
[4] 폴백 (TODO 4): 주 공급자(장애) → 보조 공급자
    답한 공급자: fake/backup, 실패 기록: ['fake: fake provider is down']
[5] 사용량 기록 (TODO 5): part2_llm/local_llm_serving/outputs/usage.csv
    timestamp,provider,model,input_tokens,output_tokens,latency_ms,ttft_ms,stop_reason
[6] (선택) 100만 토큰당 비용 (TODO 6) — 모두 가정값, 측정 아님
    API claude-opus-5-5, 출력만 100만 토큰           : $20.00
    API claude-opus-5-5, 입력 80%·출력 20% 섞인 100만 : $7.20
    로컬은 '출력 100만 토큰을 생성하는 시간' 기준(입력 처리 시간은 생략) → 위 첫 줄(출력만)과 비교:
    로컬  30 토큰/초 가정, 250W, $0.15/kWh: 전기만 $0.35, 장비 감가 $0.05/시간 포함 $0.81
    로컬  60 토큰/초 가정, 250W, $0.15/kWh: 전기만 $0.17, 장비 감가 $0.05/시간 포함 $0.41
    로컬 120 토큰/초 가정, 250W, $0.15/kWh: 전기만 $0.09, 장비 감가 $0.05/시간 포함 $0.20
    ※ 로컬 품질은 큰 API 모델과 다르다 — 같은 토큰 수라도 같은 결과가 아니다(비용만으로 고르지 않는다).
```

`measure_local_models.py` 작성 환경 실행: `건너뜀: Ollama 서버(http://localhost:11434)에 연결할 수 없습니다 (<urlopen error [Errno 111] Connection refused>).`

## 비교
- 사용자 PC 측정 후 작성(크기별 토큰/초·VRAM, API 대비 비용·품질).

## 개선점 / 다음에 확인할 것
- 동시 요청 2개에서 TTFT 변화(`OLLAMA_NUM_PARALLEL` 1 vs 2, README 과제 3).
- 같은 질문의 공급자별 품질·토큰 수 비교(README 과제 2).

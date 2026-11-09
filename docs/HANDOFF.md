# HANDOFF — Part 1 (main 에서 한 실습씩 순차 진행)

진행 방식: 한 번에 한 실습만 완성한 뒤 멈춘다. 다음 실습은 ①(총괄 세션)의 지시를 받고 시작한다.
순서: ① pytorch → ② tokenizer → ③ embedding → ④ word2vec → ⑤ transformer

## 실습별 상태

| 순서 | 실습 | 상태 | 비고 |
|---|---|---|---|
| ① | P1-04 pytorch | **완료(③ PASS_WITH_NOTES, 권장 수정 반영)** | 테스트 10개 통과(solution) / starter 10 skip. noqa, skip 사유 설명, 진행 순서 표기 반영. REF 링크는 pytorch.org 차단으로 미확인 |
| ② | P1-01 tokenizer | **완료(③ PASS_WITH_NOTES, 수정 반영·재검증 대기)** | Level 1(TODO 1~5: tokenize·Vocab·특수 토큰·pad/truncate), Level 2(TODO 6~10: BPE), Level 3(compare_with_hf.py). 테스트 15개 통과(solution) / starter 15 skip (TODO 8 동점·TODO 9 rank 순서 검사 포함). README 12항목, experiments/P1-01-20261109-bpe |
| ③ | P1-02 embedding | **완료(검증 대기)** | Level 1(TODO 1~6: 조회·one-hot 동치·L2·코사인·문장 벡터), Level 2(TODO 7~11: EmbeddingLayer+padding_idx, 학습형 위치 임베딩, top-k), Level 3(TODO 12 PCA, nn.Embedding 비교, run_demo.py, compare_pretrained.py 선택). solution 12 passed + 1 skipped(사전학습 비교, sentence-transformers 미설치) / starter 13 skip. README 12항목, experiments/P1-02-20261109-embedding |
| ④ | P1-03 word2vec | **완료(검증 대기)** | Level 1(TODO 1~3: 어휘, Skip-gram 쌍, CBOW 예제 패딩), Level 2(TODO 4~10: count^0.75 잡음 분포, 음성 샘플, Skip-gram NS, CBOW 마스크 평균, 배치 loss), Level 3(TODO 11 최근접, run_demo.py 전후 비교, compare_with_gensim.py). **CBOW 실패 해결**(양쪽 문맥만 쓰던 예제 → 모든 위치 + 패딩·마스크, 순도 0.33→0.86). solution 17 passed / starter 17 skipped (generator·사전순 동점·logsigmoid 변이 검사 포함). README 12항목, experiments/P1-03-20261109-word2vec |
| ⑤ | P1-05 transformer | **완료(검증 대기)** | Level 1(TODO 1~10: Q/K/V, 스케일드 닷-프로덕트, causal·padding 마스크), Level 2(TODO 11~19: MHA, FFN, 사인/코사인 PE, Pre-LN 블록=EncoderBlock), Level 3(TODO 20~23: 문자 LM, train_tiny.py·visualize_attention.py·compare_with_torch.py). solution 21 passed / starter 21 skip. 전체 NaN 재현, 위치 인코딩 사용, 임베딩 크기(std D^-0.5 × √D) 테스트 포함. README 12항목, experiments/P1-05-20261109-transformer |

`docs/PROGRESS.md`의 Part 1 항목은 모두 `NOT_STARTED`로 유지한다. 학습자가 직접 확인해야 `COMPLETED`가 된다.

## 공통 규약 (Part 1 전체에 적용)
- 테스트는 `part1_ai_fundamentals/conftest.py`의 `impl` 픽스처를 쓴다.
  - 실습 디렉터리의 `{IMPL}.py`를 파일 경로 기준으로, 고유한 모듈명(`part1_<디렉터리>_<impl>`)으로 불러온다.
  - starter에서 `NotImplementedError`가 나면 skip("TODO 미구현: TODO n")으로 보고한다.
- 루트 `pytest.ini`: `--import-mode=importlib`.
- 테스트 파일명은 실습별로 고유하게 짓는다. pytorch(`test_p1_04_pytorch.py`), tokenizer(`test_p1_01_tokenizer.py`), embedding(`test_p1_02_embedding.py`), word2vec(`test_p1_03_word2vec.py`), transformer(`test_p1_05_transformer.py`) — Part 1 전체 적용 완료.
- 루트에서 `pytest` 한 번으로 실습들이 섞이지 않고 실행된다(P1-05 반영 후 75 passed, 1 skipped — skip 은 P1-02 사전학습 비교, sentence-transformers 미설치).
- starter를 단독 실행하면(`python <실습>/starter.py`) 남은 TODO 목록이 나온다. Part 1 다섯 실습 모두 적용했다.

## 환경 메모
- 클라우드 환경에서는 download.pytorch.org가 막혀 있어 PyPI의 torch(2.15.0+cu132, CPU로 동작)를 설치했다. 설치에 수십 분이 걸린다.
- `.venv`: Python 3.12.13, numpy 2.5.4, gensim 4.4.0, tokenizers 0.23.3.

## 보류 (①의 지시가 있을 때 진행)
Part 0 문서 수정 — ENVIRONMENT 정정, REPOSITORY_ANALYSIS 실제 경로, API 기반 전환(.env.example, 비용 안내), 실험 디렉터리 이름 통일, ARCHITECTURE §4 예시를 실제 SelfAttention 코드에 맞추기 등.

## 다음에 재개할 첫 단계
Part 1 실습 자료(5개)는 모두 작성되었다. 학습자 확인 후 PROGRESS 상태를 바꾼다.
①의 지시에 따라 다음 중 하나를 진행한다: 보류 중인 Part 0 문서 수정(ENVIRONMENT, REPOSITORY_ANALYSIS, API 전환 등) 또는 Part 2.

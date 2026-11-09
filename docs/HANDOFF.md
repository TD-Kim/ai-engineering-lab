# HANDOFF — Part 1 (main 에서 한 실습씩 순차 진행)

진행 방식: 한 번에 한 실습만 완성한 뒤 멈춘다. 다음 실습은 ①(총괄 세션)의 지시를 받고 시작한다.
순서: ① pytorch → ② tokenizer → ③ embedding → ④ word2vec → ⑤ transformer

## 실습별 상태

| 순서 | 실습 | 상태 | 비고 |
|---|---|---|---|
| ① | P1-04 pytorch | **완료(③ PASS_WITH_NOTES, 권장 수정 반영)** | 테스트 10개 통과(solution) / starter 10 skip. noqa, skip 사유 설명, 진행 순서 표기 반영. REF 링크는 pytorch.org 차단으로 미확인 |
| ② | P1-01 tokenizer | **완료(③ PASS_WITH_NOTES, 수정 반영·재검증 대기)** | Level 1(TODO 1~5: tokenize·Vocab·특수 토큰·pad/truncate), Level 2(TODO 6~10: BPE), Level 3(compare_with_hf.py). 테스트 15개 통과(solution) / starter 15 skip (TODO 8 동점·TODO 9 rank 순서 검사 포함). README 12항목, experiments/P1-01-20261109-bpe |
| ③ | P1-02 embedding | **완료(검증 대기)** | Level 1(TODO 1~6: 조회·one-hot 동치·L2·코사인·문장 벡터), Level 2(TODO 7~11: EmbeddingLayer+padding_idx, 학습형 위치 임베딩, top-k), Level 3(TODO 12 PCA, nn.Embedding 비교, run_demo.py, compare_pretrained.py 선택). solution 12 passed + 1 skipped(사전학습 비교, sentence-transformers 미설치) / starter 13 skip. README 12항목, experiments/P1-02-20261109-embedding |
| ④ | P1-03 word2vec | 코드 초안만 있음 | **루트 `pytest`에서 `test_training_reduces_loss_and_groups_words[cbow]` 실패**(purity 0.33 < 0.6). CBOW 학습 설정(epoch·lr, 양쪽 문맥만 쓰는 예제 수 부족) 조정 필요 |
| ⑤ | P1-05 transformer | solution.py만 있음 | starter, tests, train.py, visualize_attention.py, compare_with_torch.py, README 필요 |

`docs/PROGRESS.md`의 Part 1 항목은 모두 `NOT_STARTED`로 유지한다. 학습자가 직접 확인해야 `COMPLETED`가 된다.

## 공통 규약 (Part 1 전체에 적용)
- 테스트는 `part1_ai_fundamentals/conftest.py`의 `impl` 픽스처를 쓴다.
  - 실습 디렉터리의 `{IMPL}.py`를 파일 경로 기준으로, 고유한 모듈명(`part1_<디렉터리>_<impl>`)으로 불러온다.
  - starter에서 `NotImplementedError`가 나면 skip("TODO 미구현: TODO n")으로 보고한다.
- 루트 `pytest.ini`: `--import-mode=importlib`.
- 테스트 파일명은 실습별로 고유하게 짓는다. pytorch(`test_p1_04_pytorch.py`), tokenizer(`test_p1_01_tokenizer.py`), embedding(`test_p1_02_embedding.py`)은 적용 완료. 나머지 실습은 각자 차례에 바꾼다(word2vec → `test_p1_03_word2vec.py` 등).
- 루트에서 `pytest` 한 번으로 실습들이 섞이지 않고 실행된다(P1-01 반영 후 39개 중 word2vec 1건만 실패).
- starter를 단독 실행하면(`python <실습>/starter.py`) 남은 TODO 목록이 나온다. pytorch·tokenizer·embedding에 적용했고, 나머지 실습도 같은 방식으로 맞춘다.

## 환경 메모
- 클라우드 환경에서는 download.pytorch.org가 막혀 있어 PyPI의 torch(2.15.0+cu132, CPU로 동작)를 설치했다. 설치에 수십 분이 걸린다.
- `.venv`: Python 3.12.13, numpy 2.5.4, gensim 4.4.0, tokenizers 0.23.3.

## 보류 (①의 지시가 있을 때 진행)
Part 0 문서 수정 — ENVIRONMENT 정정, REPOSITORY_ANALYSIS 실제 경로, API 기반 전환(.env.example, 비용 안내), 실험 디렉터리 이름 통일, ARCHITECTURE §4 예시를 실제 SelfAttention 코드에 맞추기 등.

## 다음에 재개할 첫 단계
①이 "다음"을 지시하면 ④ word2vec을 진행한다.
1. 테스트 파일명을 `test_p1_03_word2vec.py`로 바꾸고, 실패 중인 CBOW 테스트(purity 0.33)를 먼저 해결한다.
2. starter 진행 안내를 붙이고 힌트에서 정답이 새지 않는지 점검한다.
3. README 12항목을 쓰고 실험 기록을 남긴 뒤 푸시한다.

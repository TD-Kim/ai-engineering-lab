# HANDOFF — Part 1 (part1-ai-fundamentals 브랜치, WIP)

토큰 한도 때문에 작업 도중 중단했다. 아래는 중단 시점의 상태다.

## 완료된 것
- 공통: 루트 `pytest.ini`(importlib 모드), `part1_ai_fundamentals/conftest.py`. `impl` 픽스처가 `IMPL=solution|starter`에 따라 모듈을 불러오고, starter에서 `NotImplementedError`가 나면 skip으로 보고한다.
- `part1_ai_fundamentals/data/corpus_ko_en.txt`: 직접 작성한 소규모 한국어·영어 말뭉치.
- pytorch: `starter.py` / `solution.py`(TODO 1~10), `tests/test_pytorch_basics.py`, `run_demo.py`. torch가 없어 테스트는 아직 실행하지 않았다.
- tokenizer: `starter.py` / `solution.py`(TODO 1~8, BPE), `compare_with_hf.py`, `tests/test_tokenizer.py`. 테스트 10개 통과(solution 기준).
  HF와 처음 50개 병합의 겹침은 0.52, 토큰 수 비율은 0.97. 차이가 나는 원인: HF는 `</w>`를 처음부터 마지막 글자에 붙여서 학습을 시작한다.
- embedding: `starter.py` / `solution.py`(TODO 1~6), `tests/test_embedding.py`. 테스트는 아직 실행하지 않았다(torch 필요).
- word2vec: `starter.py` / `solution.py`(TODO 1~9), `compare_with_gensim.py`, `tests/test_word2vec.py`. 테스트는 아직 실행하지 않았다(torch 필요).
- transformer: `solution.py`만 있다(TODO 1~23: SelfAttention 1~10, MHA 11~14, FFN·PE·Block 15~19, 미니 LM 20~23).

## 미완료 TODO
1. 이 환경에 torch 설치. download.pytorch.org가 프록시에 막혀 있어 PyPI판 torch 설치를 시도했으나 중간에 중단했다. 이후 pytorch·embedding·word2vec·transformer 테스트를 실행해야 한다.
2. transformer: `starter.py`, `tests/test_transformer.py`(nn.MultiheadAttention·F.scaled_dot_product_attention과 비교, 마스크·잔차 검증), `compare_with_torch.py`, `train.py`, `visualize_attention.py`.
3. `datasets/download_tiny_shakespeare.py`: 다운로드 스크립트만 두고, 데이터는 커밋하지 않는다.
4. 각 하위 디렉터리 README 12항목 작성. 현재는 Part 0에서 만든 자리표시자 README 그대로다.
5. embedding 데모 스크립트(선택).
6. `experiments/part1/`: 템플릿과 CPU 실행 결과(loss, 시간). GPU 칸은 "사용자 PC에서 측정 필요"로 둔다.
7. `docs/ARCHITECTURE.md` §4 예시를 실제 `SelfAttention` 코드(TODO 7~9, 클래스 안 inline)에 맞게 수정.
8. `requirements/part1.txt` 정합성 확인. `docs/PROGRESS.md`의 Part 1은 NOT_STARTED로 유지한다.
9. 완료한 뒤 오케스트레이터에게 보고: 해시, 파일 목록, pytest 결과, 학습 포인트, 한계.

## 재개할 첫 단계
`uv venv --python 3.12 .venv && uv pip install torch` (PyPI) → `pytest part1_ai_fundamentals`로 기존 4개 실습 검증 → transformer `starter.py`와 테스트 작성.

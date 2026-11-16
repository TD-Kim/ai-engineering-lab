"""선택 스크립트(사용자 PC 용) 7 개를 한 프로세스에서 불러도 실습끼리 섞이지 않는지 검사한다.

각 스크립트는 같은 폴더의 solution.py 등을 "실습별 고유 이름"(lab_<상위 폴더>_<실습 폴더>_<파일>)으로 불러와야 한다.
`from solution import ...` 처럼 흔한 이름을 쓰면 먼저 등록된 다른 실습의 solution 을 잘못 받는다.
불러오기만 하고 main() 은 부르지 않는다 — 네트워크·서버·API 키·모델 다운로드 불필요.
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# (스크립트, 그 스크립트가 쓰는 이름, 그 이름이 와야 하는 solution 파일)
SCRIPTS = [
    ("part1_ai_fundamentals/tokenizer/count_with_claude.py", "estimate_cost", "part1_ai_fundamentals/tokenizer/solution.py"),
    ("part1_ai_fundamentals/tokenizer/pretrained_tokenizer.py", "compare_languages", "part1_ai_fundamentals/tokenizer/solution.py"),
    ("part1_ai_fundamentals/embedding/embed_with_openai.py", "top_k", "part1_ai_fundamentals/embedding/solution.py"),
    ("part2_llm/next_token_prediction/run_qwen.py", "generate", "part2_llm/next_token_prediction/solution.py"),
    ("part2_llm/local_llm_serving/measure_local_models.py", "OllamaClient", "part2_llm/local_llm_serving/solution.py"),
    ("part2_llm/llm_api/basics/call_claude.py", "m", "part2_llm/llm_api/basics/solution.py"),
    ("part2_llm/llm_api/cost_latency/measure_caching.py", "m", "part2_llm/llm_api/cost_latency/solution.py"),
]


def _source_file(obj) -> Path:
    module = obj if hasattr(obj, "__file__") and not callable(obj) else sys.modules[obj.__module__]
    return Path(module.__file__).resolve()


@pytest.fixture(scope="module")
def loaded():
    pytest.importorskip("tokenizers")   # pretrained_tokenizer 가 불러오는 라이브러리(part1 요구 사항)
    pytest.importorskip("torch")        # run_qwen
    saved = {k: sys.modules.get(k) for k in ("solution", "run_demo")}
    # 충돌 재현: 다른 실습이 먼저 흔한 이름 "solution"·"run_demo" 를 등록해 둔 상황을 가짜 모듈로 만든다.
    for k in saved:
        sys.modules[k] = types.ModuleType(k)
    decoys = {k: sys.modules[k] for k in saved}
    path_before = list(sys.path)
    mods = {}
    try:
        for rel, _name, _src in SCRIPTS:  # 한 프로세스에서 차례로 불러온다
            spec = importlib.util.spec_from_file_location("isolation_" + rel.replace("/", "_")[:-3], ROOT / rel)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mods[rel] = mod
        yield mods, path_before, decoys
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


@pytest.mark.parametrize("rel,name,src", SCRIPTS, ids=[s[0].split("/")[-1] for s in SCRIPTS])
def test_each_script_uses_its_own_lab(loaded, rel, name, src):
    mods, _, _ = loaded
    assert _source_file(getattr(mods[rel], name)) == (ROOT / src).resolve()


def test_no_generic_module_names_or_sys_path_changes(loaded):
    _, path_before, decoys = loaded
    assert sys.path == path_before                                        # sys.path 를 고치지 않음
    for k, v in decoys.items():
        assert sys.modules.get(k) is v                                    # 흔한 이름을 덮어쓰지도, 가져다 쓰지도 않음


def test_same_lab_scripts_share_one_module(loaded):
    mods, _, _ = loaded
    a = mods["part1_ai_fundamentals/tokenizer/count_with_claude.py"]._sol
    b = mods["part1_ai_fundamentals/tokenizer/pretrained_tokenizer.py"]._sol
    assert a is b and a.__name__ == "lab_part1_ai_fundamentals_tokenizer_solution"

"""Part 1 공통 pytest 설정.

- `impl` 픽스처: 테스트 파일이 속한 실습 디렉터리의 `solution.py`(기본) 또는
  `starter.py`(환경 변수 `IMPL=starter`)를 불러온다.
- starter 상태에서 TODO가 남아 있으면 `NotImplementedError`가 나는데,
  이 경우 테스트를 실패 대신 **skip("TODO 미구현")** 으로 보고한다.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

VALID_IMPLS = ("solution", "starter")
PART1_DIR = Path(__file__).resolve().parent


def _impl_name() -> str:
    name = os.environ.get("IMPL", "solution")
    if name not in VALID_IMPLS:
        raise pytest.UsageError(f"IMPL must be one of {VALID_IMPLS}, got {name!r}")
    return name


def load_module(path: Path, module_name: str):
    """파일 경로로 모듈을 불러온다. 디렉터리마다 이름이 같은 파일이 있어 고유한 이름으로 등록한다."""
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # dataclass 등이 모듈을 찾을 수 있게 먼저 등록
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def impl(request):
    """<실습>[/advanced]/tests/test_*.py 기준으로 <실습>[/advanced]/{IMPL}.py 를 불러온다.

    모듈 이름은 part1_ai_fundamentals 기준 상대 경로로 만든다(예: part1_tokenizer_advanced_solution).
    여러 실습에 같은 이름의 advanced/solution.py 가 있어도 sys.modules 에서 섞이지 않는다.
    """
    name = _impl_name()
    lab_dir = Path(request.path).resolve().parent.parent
    rel = lab_dir.relative_to(PART1_DIR)
    return load_module(lab_dir / f"{name}.py", "part1_" + "_".join(rel.parts) + f"_{name}")


def pytest_report_header(config):
    return f"IMPL={_impl_name()}  (IMPL=starter pytest 로 직접 구현한 코드를 검사)"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if call.excinfo is not None and call.excinfo.errisinstance(NotImplementedError):
        report.outcome = "skipped"
        lineno = item.location[1] or 0
        report.longrepr = (str(item.path), lineno, f"TODO 미구현: {call.excinfo.value}")

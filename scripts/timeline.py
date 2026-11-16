"""커밋 하나를 읽어 docs/TIMELINE.md 에 "언제 · 무엇을 · 어떤 함수를" 기록한다.

보통은 git 훅(.githooks/post-commit)이 커밋할 때마다 자동으로 실행한다.
직접 실행해도 된다:

    python scripts/timeline.py            # 마지막 커밋(HEAD)을 기록
    python scripts/timeline.py <커밋>      # 특정 커밋을 기록

기록 내용
- 날짜·시각(커밋 시각), 커밋 메시지
- 바뀐 파일(추가 / 수정 / 삭제 / 이름 변경)
- Python 파일의 함수·클래스: 새로 생긴 것, 내용이 바뀐 것, 사라진 것
- "메모" 칸: 배운 점·막혔던 점을 직접 적거나 Claude Code 에 정리를 부탁한다
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

TIMELINE = Path("docs") / "TIMELINE.md"
HEADER = (
    "# 타임라인\n\n"
    "커밋할 때마다 `scripts/timeline.py`가 자동으로 기록합니다(언제 · 무엇을 · 어떤 함수를).\n"
    "각 항목의 **메모**는 직접 채웁니다. 배운 점, 막혔던 점, 다음에 할 일을 적어 두세요.\n"
)
STATUS_NAMES = {"A": "추가", "M": "수정", "D": "삭제", "R": "이름 변경", "C": "복사", "T": "형식 변경"}


def git(*args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, encoding="utf-8", check=False
    )
    if check and result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} 실패")
    return result.stdout


def file_at(commit: str, path: str) -> str | None:
    """커밋 시점의 파일 내용. 없으면 None."""
    result = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        capture_output=True, text=True, encoding="utf-8", check=False,
    )
    return result.stdout if result.returncode == 0 else None


def definitions(source: str | None) -> dict[str, str]:
    """{함수·클래스 이름: 그 정의의 소스 코드}. 클래스 안 메서드는 '클래스.메서드'로 적는다."""
    if not source:
        return {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    found: dict[str, str] = {}

    def visit(nodes: list[ast.stmt], prefix: str) -> None:
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}{node.name}"
                found[name] = ast.get_source_segment(source, node) or ""
                if isinstance(node, ast.ClassDef):
                    visit(node.body, f"{name}.")

    visit(tree.body, "")
    return found


def changed_files(commit: str) -> list[tuple[str, str]]:
    """[(상태 글자, 경로)]. 첫 커밋도 처리한다."""
    out = git("show", "--name-status", "--format=", "--find-renames", "--root", commit)
    files = []
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        status = parts[0][0]
        path = parts[-1]  # 이름 변경이면 새 경로
        files.append((status, path))
    return files


def describe_functions(commit: str, parent: str | None, path: str) -> list[str]:
    new = definitions(file_at(commit, path))
    old = definitions(file_at(parent, path)) if parent else {}
    added = [n for n in new if n not in old]
    removed = [n for n in old if n not in new]
    changed = [n for n in new if n in old and new[n] != old[n]]
    lines = []
    if added:
        lines.append("새로 작성: " + ", ".join(f"`{n}`" for n in added))
    if changed:
        lines.append("수정: " + ", ".join(f"`{n}`" for n in changed))
    if removed:
        lines.append("삭제: " + ", ".join(f"`{n}`" for n in removed))
    return lines


def build_entry(commit: str) -> str | None:
    files = [(s, p) for s, p in changed_files(commit) if p != TIMELINE.as_posix()]
    if not files:
        return None  # 타임라인만 바뀐 커밋은 기록하지 않는다
    when = git("show", "-s", "--format=%ad", "--date=format:%Y-%m-%d %H:%M", commit).strip()
    subject = git("show", "-s", "--format=%s", commit).strip()
    parents = git("show", "-s", "--format=%P", commit).split()
    parent = parents[0] if parents else None

    lines = [f"### {when[11:]} — {subject}", ""]
    for status, path in files:
        lines.append(f"- `{path}` ({STATUS_NAMES.get(status, status)})")
        if path.endswith(".py") and status != "D":
            for detail in describe_functions(commit, parent, path):
                lines.append(f"  - {detail}")
        elif path.endswith(".py") and status == "D" and parent:
            gone = list(definitions(file_at(parent, path)))
            if gone:
                lines.append("  - 삭제: " + ", ".join(f"`{n}`" for n in gone))
    lines += ["- 메모: ", ""]
    return when[:10], "\n".join(lines)


def append(day: str, entry: str) -> None:
    text = TIMELINE.read_text(encoding="utf-8") if TIMELINE.exists() else HEADER
    day_header = f"## {day}"
    if day_header not in text:
        text = text.rstrip("\n") + f"\n\n{day_header}\n"
    text = text.rstrip("\n") + "\n\n" + entry
    TIMELINE.parent.mkdir(parents=True, exist_ok=True)
    TIMELINE.write_text(text, encoding="utf-8", newline="\n")


def main() -> int:
    commit = sys.argv[1] if len(sys.argv) > 1 else "HEAD"
    result = build_entry(commit)
    if result is None:
        return 1  # 기록할 것 없음 → 훅은 커밋을 고치지 않는다
    day, entry = result
    append(day, entry)
    print(f"[timeline] {TIMELINE.as_posix()} 에 기록했습니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

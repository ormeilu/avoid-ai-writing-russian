"""Хук commit-msg: сообщения коммитов в проекте пишутся по-русски.

Первая строка должна содержать кириллицу, быть не длиннее 72 знаков
и не заканчиваться точкой. Коммиты слияния и fixup!/squash! пропускаются.
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import regex

SKIPPED = re.compile(r"^(Merge|Revert|fixup!|squash!|amend!)")
CYRILLIC = regex.compile(r"\p{Script=Cyrillic}")


def check_commit_message(message: str) -> list[str]:
    lines = [line for line in message.split("\n") if not line.startswith("#")]
    subject = (lines[0] if lines else "").strip()
    if SKIPPED.match(subject):
        return []
    errors: list[str] = []
    if not subject:
        errors.append("пустая первая строка")
    if not CYRILLIC.search(subject):
        errors.append("первая строка должна быть на русском")
    if len(subject) > 72:
        errors.append(f"первая строка длиннее 72 знаков ({len(subject)})")
    if subject.endswith((".", "。")):
        errors.append("первая строка не должна заканчиваться точкой")
    if len(lines) > 1 and lines[1].strip() != "":
        errors.append("после первой строки нужна пустая строка")
    return errors


def main(argv: list[str]) -> int:
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8")
    if not argv:
        print("использование: check_commit_msg.py ФАЙЛ_СООБЩЕНИЯ", file=sys.stderr)
        return 2
    errors = check_commit_message(Path(argv[0]).read_bytes().decode("utf-8", errors="replace"))
    for e in errors:
        print(f"commit-msg: {e}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

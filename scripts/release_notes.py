"""Печатает раздел CHANGELOG.md для версии: `uv run python scripts/release_notes.py 0.2.0`."""

from __future__ import annotations

import sys

from changelog import section
from versions import ROOT, read_text, utf8_output


def main(argv: list[str]) -> int:
    utf8_output()
    version = (argv[0] if argv else "").removeprefix("v")
    body = section(read_text(ROOT / "CHANGELOG.md"), version)
    if not body:
        print(f"В CHANGELOG.md нет раздела для версии {version}", file=sys.stderr)
        return 1
    sys.stdout.write(f"{body}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

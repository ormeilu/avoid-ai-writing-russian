"""Проверяет, что версия одинакова во всех местах, где она записана, включая uv.lock."""

from __future__ import annotations

import sys

from versions import LOCK, FoundVersion, lock_version, read_versions, utf8_output


def main() -> int:
    utf8_output()
    found = [*read_versions(), FoundVersion(LOCK, lock_version())]
    if len({entry.version for entry in found}) != 1 or any(not entry.version for entry in found):
        print("Версии расходятся:", file=sys.stderr)
        for entry in found:
            print(f"  {entry.file}: {entry.version or 'не найдена'}", file=sys.stderr)
        print(
            "Исправьте вручную (uv.lock обновляет `uv lock`) или выпустите версию через "
            "`uv run python scripts/release.py`.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

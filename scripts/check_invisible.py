"""Хук pre-commit: в файлах нет невидимых символов и латинских букв
внутри русских слов. Проект сам ловит такие артефакты в чужих текстах
и не должен их содержать.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

from aiw_ru.text import prepare


def main(files: list[str]) -> int:
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8")
    bad = 0
    for file in files:
        prepared = prepare(Path(file).read_bytes().decode("utf-8", errors="replace"))
        for invisible in prepared.invisible:
            print(f"{file}: невидимый символ U+{ord(invisible.char):04X} в позиции {invisible.index}", file=sys.stderr)
            bad += 1
        for homoglyph in prepared.homoglyphs:
            print(
                f"{file}: латиница внутри русского слова «{homoglyph.word}» в позиции {homoglyph.index}",
                file=sys.stderr,
            )
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

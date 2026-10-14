"""Хук pre-commit: в файлах нет фрагментов собственных работ автора.

Работы (статьи, рефераты, диссертация) лежат вне репозитория: в папке из переменной
AIW_RU_PRIVATE (несколько папок — через os.pathsep) или в ~/.config/aiw-ru/private.
Читаются .txt, .md и .docx. Фрагмент — восемь слов подряд, которые есть и в работе,
и в файле. Если папки нет (CI, другие участники), проверка молча проходит.
"""

from __future__ import annotations

import html
import io
import os
import re
import sys
import zipfile
from collections.abc import Iterable
from pathlib import Path

FRAGMENT_WORDS = 8
WORD_RE = re.compile(r"\w+")
DOCX_PARAGRAPH_END = re.compile(r"</w:p>")
XML_TAG = re.compile(r"<[^>]+>")

Fragment = tuple[str, ...]


def private_folders(environ: dict[str, str] | os._Environ[str] = os.environ) -> list[Path]:
    """Папки с работами автора: из AIW_RU_PRIVATE или папка по умолчанию."""
    configured = environ.get("AIW_RU_PRIVATE")
    if configured:
        return [Path(part).expanduser() for part in configured.split(os.pathsep) if part]
    return [Path.home() / ".config" / "aiw-ru" / "private"]


def docx_text(path: Path) -> str:
    """Текст абзацев docx из word/document.xml, без pandoc."""
    with zipfile.ZipFile(path) as docx:
        xml = docx.read("word/document.xml").decode("utf-8")
    return html.unescape(XML_TAG.sub("", DOCX_PARAGRAPH_END.sub("\n", xml)))


def work_texts(folders: Iterable[Path]) -> list[str]:
    """Тексты всех работ из папок; несуществующие папки пропускаются."""
    texts: list[str] = []
    for folder in folders:
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob("*")):
            if path.suffix in (".txt", ".md"):
                texts.append(path.read_text(encoding="utf-8", errors="replace"))
            elif path.suffix == ".docx":
                texts.append(docx_text(path))
    return texts


def normalized_words(text: str) -> list[re.Match[str]]:
    return list(WORD_RE.finditer(text))


def fragment_at(words: list[re.Match[str]], start: int) -> Fragment:
    return tuple(word.group().lower().replace("ё", "е") for word in words[start : start + FRAGMENT_WORDS])


def fragments(text: str) -> set[Fragment]:
    words = normalized_words(text)
    return {fragment_at(words, start) for start in range(len(words) - FRAGMENT_WORDS + 1)}


def leaked_lines(text: str, private: set[Fragment]) -> list[tuple[int, str]]:
    """Строки файла, где начинается фрагмент работы, с самим фрагментом."""
    words = normalized_words(text)
    found: dict[int, str] = {}
    for start in range(len(words) - FRAGMENT_WORDS + 1):
        fragment = fragment_at(words, start)
        if fragment in private:
            line = text.count("\n", 0, words[start].start()) + 1
            found.setdefault(line, " ".join(fragment))
    return sorted(found.items())


def main(files: list[str]) -> int:
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8")
    private: set[Fragment] = set()
    for text in work_texts(private_folders()):
        private |= fragments(text)
    if not private:
        return 0
    leaked = 0
    for file in files:
        try:
            text = Path(file).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for line, fragment in leaked_lines(text, private):
            print(f"{file}:{line}: фрагмент работы автора: «{fragment}»", file=sys.stderr)
            leaked += 1
    return 1 if leaked else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

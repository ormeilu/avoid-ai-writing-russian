"""Работа с CHANGELOG.md: раздел «Не выпущено» и заметки к выпуску."""

from __future__ import annotations

import re

UNRELEASED = "## [Не выпущено]"

NEXT_SECTION = re.compile(r"^## \[", re.MULTILINE)
LINK_DEFINITION = re.compile(r"^\[[^\]]+\]:.*$", re.MULTILINE)


def section(changelog: str, version: str) -> str | None:
    """Текст раздела версии (без заголовка) или None."""
    head = UNRELEASED if version == "unreleased" else f"## [{version}]"
    start = changelog.find(head)
    if start < 0:
        return None
    body_start = changelog.find("\n", start) + 1
    rest = changelog[body_start:]
    next_section = NEXT_SECTION.search(rest)
    body = rest if next_section is None else rest[: next_section.start()]
    return LINK_DEFINITION.sub("", body).strip()


def cut(changelog: str, version: str, date: str) -> str:
    """Превращает «Не выпущено» в раздел версии и открывает новый пустой."""
    if not section(changelog, "unreleased"):
        raise ValueError("раздел «Не выпущено» пуст или отсутствует: нечего выпускать")
    return changelog.replace(UNRELEASED, f"{UNRELEASED}\n\n## [{version}] — {date}", 1)

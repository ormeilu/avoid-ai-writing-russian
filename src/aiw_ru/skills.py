"""Скиллы aiw-ru для агента, у которого есть только команда `aiw-ru`, без плагина.

В пакет скиллы кладёт сборка (`skills/` из корня репозитория становится
`aiw_ru/data/skills/`), а в рабочей копии репозитория они читаются прямо из `skills/`.
В тексте SKILL.md вызовы детектора из репозитория (`uv run --project ../.. aiw-ru`)
заменяются на `aiw-ru`, а ссылки на соседние файлы на команды, которые их выводят.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

PACKAGED = Path(__file__).resolve().parent / "data" / "skills"
SOURCE = Path(__file__).resolve().parents[2] / "skills"
MAIN = "avoid-ai-writing-russian"
REPOSITORY = "https://github.com/ormeilu/avoid-ai-writing-russian"

# Вызов детектора из папки скилла в репозитории, с extra ml или без.
_REPO_COMMAND = re.compile(r"uv run --project \.\./\.\.(?: --extra ml)? aiw-ru")
# Ссылка на файл скилла: свой (references/…), соседний (../имя/…) или SKILL.md из references (../SKILL.md).
_FILE_LINK = re.compile(r"\[([^\]]+)\]\((?:\.\./([\w-]+)/)?(SKILL\.md|references/[^)\s]+)\)")
_PARENT_SKILL = re.compile(r"`\.\./SKILL\.md`")
_FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
# Фразы о расположении файлов в репозитории: без репозитория они ни к чему.
_REPO_NOTES = (
    (" Пути к командам детектора считай от корня репозитория (`../../` от этого файла).", ""),
    (" (проект детектора лежит в корне репозитория, `../..` от этого файла)", ""),
    ("`../../examples/<имя>.json`", "`examples/<имя>.json` из репозитория " + REPOSITORY),
)


class SkillError(Exception):
    """Нет такого скилла или файла в нём."""


@dataclass(frozen=True, slots=True)
class Skill:
    name: str
    description: str
    # Файлы скилла кроме SKILL.md, пути от папки скилла через «/».
    files: tuple[str, ...]


def root() -> Path:
    """Папка со скиллами: копия в пакете, иначе `skills/` рабочей копии репозитория."""
    for path in (PACKAGED, SOURCE):
        if path.is_dir():
            return path
    raise SkillError("скиллы не найдены: пакет собран без них")


def _description(skill_md: str) -> str:
    """Поле description из шапки SKILL.md, в том числе многострочное (`>-`)."""
    m = _FRONTMATTER.match(skill_md)
    lines = m[1].split("\n") if m else []
    for i, line in enumerate(lines):
        if not line.startswith("description:"):
            continue
        value = line.removeprefix("description:").strip()
        if value not in (">-", ">", "|", "|-"):
            return value.strip("\"'")
        block = []
        for cont in lines[i + 1 :]:
            if cont and not cont.startswith(" "):
                break
            block.append(cont.strip())
        return " ".join(x for x in block if x)
    return ""


def first_sentence(text: str) -> str:
    """Первое предложение описания: для списка скиллов в терминале."""
    m = re.match(r"(.+?[.!?])\s+(?=[A-ZА-ЯЁ«])", text)
    return m[1] if m else text


def _files(folder: Path) -> tuple[str, ...]:
    return tuple(
        sorted(
            p.relative_to(folder).as_posix()
            for p in folder.rglob("*")
            if p.is_file() and p.name != "SKILL.md" and not any(part.startswith(".") for part in p.parts)
        )
    )


def all_skills() -> list[Skill]:
    """Скиллы по порядку: основной первым."""
    base = root()
    found = [
        Skill(p.name, _description((p / "SKILL.md").read_text(encoding="utf-8")), _files(p))
        for p in sorted(base.iterdir())
        if (p / "SKILL.md").is_file()
    ]
    return sorted(found, key=lambda s: s.name != MAIN)


def _command(skill: str, file: str) -> str:
    return f"`aiw-ru skill {skill}`" if file == "SKILL.md" else f"`aiw-ru skill {skill} {file}`"


def rewrite(text: str, name: str) -> str:
    """Текст скилла для агента без репозитория: команды через `aiw-ru`, файлы через `aiw-ru skill`."""
    text = _REPO_COMMAND.sub("aiw-ru", text)
    for old, new in _REPO_NOTES:
        text = text.replace(old, new)

    def link(m: re.Match[str]) -> str:
        command = _command(m[2] or name, m[3])
        # Подпись-путь («../x/SKILL.md») заменяется командой целиком.
        return command if "/" in m[1] or m[1].endswith(".md") else f"{m[1]} ({command})"

    text = _FILE_LINK.sub(link, text)
    return _PARENT_SKILL.sub(f"SKILL.md ({_command(name, 'SKILL.md')})", text)


def for_cli(text: str, name: str) -> str:
    """SKILL.md для агента без репозитория, с пояснением сверху."""
    text = rewrite(text, name)
    note = (
        "> Скилл выведен командой `aiw-ru skill`. Детектор вызывается как `aiw-ru …`; флаг `--extra ml` "
        "нужен только при запуске из репозитория, здесь его пропускай. Остальные файлы скилла выводит "
        f"`aiw-ru skill {name} <файл>`.\n\n"
    )
    m = _FRONTMATTER.match(text)
    return text[: m.end()] + "\n" + note + text[m.end() :].lstrip("\n") if m else note + text


def read(name: str, file: str | None = None) -> str:
    """SKILL.md или другой файл скилла в виде для `aiw-ru`."""
    skills = {s.name: s for s in all_skills()}
    if name not in skills:
        raise SkillError(f"нет скилла {name}; есть: {', '.join(skills)}")
    folder = root() / name
    if file is None:
        return for_cli((folder / "SKILL.md").read_text(encoding="utf-8"), name)
    if file not in skills[name].files:
        listed = ", ".join(skills[name].files) or "других файлов нет"
        raise SkillError(f"в скилле {name} нет файла {file}; есть: {listed}")
    return rewrite((folder / file).read_text(encoding="utf-8"), name)

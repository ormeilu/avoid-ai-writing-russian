"""Версия проекта записана в нескольких местах (SITES). Этот модуль знает их все:
читает, сверяет и переписывает.

uv.lock тоже хранит версию проекта, но его не правят руками: после записи
новой версии его обновляет `uv lock`, а `lock_version` читает, что там записано.
"""

from __future__ import annotations

import io
import json
import re
import sys
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

ROOT = Path(__file__).resolve().parent.parent
LOCK = "uv.lock"


@dataclass(frozen=True, slots=True)
class VersionSite:
    file: str
    read: Callable[[str], str | None]
    write: Callable[[str, str], str]


class FoundVersion(NamedTuple):
    file: str
    version: str | None


def utf8_output() -> None:
    """Вывод в UTF-8 и на Windows, где канал по умолчанию в кодировке локали."""
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8")


def read_text(path: Path) -> str:
    """Файл как есть, без перевода концов строк."""
    return path.read_bytes().decode("utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_bytes(text.encode("utf-8"))


def _str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


JSON_VERSION = re.compile(r'("version":\s*")[^"]+(")')


def _json_version(file: str) -> VersionSite:
    return VersionSite(
        file,
        read=lambda text: _str(json.loads(text).get("version")),
        write=lambda text, version: JSON_VERSION.sub(lambda match: f"{match[1]}{version}{match[2]}", text, count=1),
    )


def _marketplace_read(text: str) -> str | None:
    data = json.loads(text)
    version = _str((data.get("metadata") or {}).get("version"))
    plugins = data.get("plugins")
    if not isinstance(plugins, list):
        return None
    return version if all(plugin.get("version") == version for plugin in plugins) else None


CFF_READ = re.compile(r"^version:\s*(\S+)\s*$", re.MULTILINE)
FRONTMATTER_READ = re.compile(r"^---\n[\s\S]*?^version:\s*(\S+)\s*$", re.MULTILINE)
YAML_WRITE = re.compile(r"^(version:\s*)\S+(\s*)$", re.MULTILINE)


def _yaml_write(text: str, version: str) -> str:
    return YAML_WRITE.sub(lambda match: f"{match[1]}{version}{match[2]}", text, count=1)


def _yaml_version(file: str, pattern: re.Pattern[str]) -> VersionSite:
    def read(text: str) -> str | None:
        match = pattern.search(text)
        return match[1] if match else None

    return VersionSite(file, read=read, write=_yaml_write)


# Таблица [project] до следующего заголовка таблицы: `version` может встретиться и в других.
PROJECT_TABLE = re.compile(r"^\[project\][ \t]*$(.*?)(?=^\[|\Z)", re.MULTILINE | re.DOTALL)
PROJECT_VERSION = re.compile(r'^(version[ \t]*=[ \t]*")[^"]+(")', re.MULTILINE)


def _pyproject_read(text: str) -> str | None:
    return _str(tomllib.loads(text).get("project", {}).get("version"))


def _pyproject_write(text: str, version: str) -> str:
    table = PROJECT_TABLE.search(text)
    if not table:
        return text
    body = PROJECT_VERSION.sub(lambda match: f"{match[1]}{version}{match[2]}", table[1], count=1)
    return text[: table.start(1)] + body + text[table.end(1) :]


SITES: list[VersionSite] = [
    VersionSite("pyproject.toml", read=_pyproject_read, write=_pyproject_write),
    _json_version(".claude-plugin/plugin.json"),
    _json_version(".codex-plugin/plugin.json"),
    VersionSite(
        ".claude-plugin/marketplace.json",
        read=_marketplace_read,
        write=lambda text, version: JSON_VERSION.sub(lambda match: f"{match[1]}{version}{match[2]}", text),
    ),
    _yaml_version("skills/avoid-ai-writing-russian/SKILL.md", FRONTMATTER_READ),
    _yaml_version("skills/antiplagiat/SKILL.md", FRONTMATTER_READ),
    _yaml_version("CITATION.cff", CFF_READ),
]


def read_versions(root: Path = ROOT) -> list[FoundVersion]:
    return [FoundVersion(site.file, site.read(read_text(root / site.file))) for site in SITES]


def write_versions(version: str, root: Path = ROOT) -> None:
    for site in SITES:
        path = root / site.file
        write_text(path, site.write(read_text(path), version))


def lock_version(root: Path = ROOT) -> str | None:
    """Версия самого проекта в uv.lock: пакет, который ставится из корня (`source = { editable = "." }`)."""
    path = root / LOCK
    if not path.exists():
        return None
    for package in tomllib.loads(read_text(path)).get("package", []):
        source = package.get("source") or {}
        if "." in (source.get("editable"), source.get("virtual")):
            return _str(package.get("version"))
    return None


# Версии по PEP 440 в той форме, которую пишет uv: X.Y.Z и предварительные X.Y.ZaN, bN, rcN.
VERSION = re.compile(r"\A(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:(a|b|rc)(0|[1-9]\d*))?\Z", re.ASCII)
# Предварительная версия младше выпуска: 2.0.0a1 < 2.0.0b1 < 2.0.0rc1 < 2.0.0.
_PRE_RANK = {"a": 0, "b": 1, "rc": 2, None: 3}


def _key(version: str) -> tuple[int, int, int, int, int]:
    match = VERSION.match(version)
    if not match:
        raise ValueError(f"не версия PEP 440 вида X.Y.Z или X.Y.ZrcN: {version}")
    return int(match[1]), int(match[2]), int(match[3]), _PRE_RANK[match[4]], int(match[5] or 0)


def is_prerelease(v: str) -> bool:
    match = VERSION.match(v)
    return bool(match and match[4])


def compare_versions(a: str, b: str) -> int:
    key_a, key_b = _key(a), _key(b)
    return (key_a > key_b) - (key_a < key_b)

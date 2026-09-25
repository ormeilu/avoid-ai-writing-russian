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
        read=lambda t: _str(json.loads(t).get("version")),
        write=lambda t, v: JSON_VERSION.sub(lambda m: f"{m[1]}{v}{m[2]}", t, count=1),
    )


def _marketplace_read(t: str) -> str | None:
    d = json.loads(t)
    v = _str((d.get("metadata") or {}).get("version"))
    plugins = d.get("plugins")
    if not isinstance(plugins, list):
        return None
    return v if all(p.get("version") == v for p in plugins) else None


CFF_READ = re.compile(r"^version:\s*(\S+)\s*$", re.MULTILINE)
FRONTMATTER_READ = re.compile(r"^---\n[\s\S]*?^version:\s*(\S+)\s*$", re.MULTILINE)
YAML_WRITE = re.compile(r"^(version:\s*)\S+(\s*)$", re.MULTILINE)


def _yaml_write(t: str, v: str) -> str:
    return YAML_WRITE.sub(lambda m: f"{m[1]}{v}{m[2]}", t, count=1)


def _yaml_version(file: str, pattern: re.Pattern[str]) -> VersionSite:
    def read(t: str) -> str | None:
        m = pattern.search(t)
        return m[1] if m else None

    return VersionSite(file, read=read, write=_yaml_write)


# Таблица [project] до следующего заголовка таблицы: `version` может встретиться и в других.
PROJECT_TABLE = re.compile(r"^\[project\][ \t]*$(.*?)(?=^\[|\Z)", re.MULTILINE | re.DOTALL)
PROJECT_VERSION = re.compile(r'^(version[ \t]*=[ \t]*")[^"]+(")', re.MULTILINE)


def _pyproject_read(t: str) -> str | None:
    return _str(tomllib.loads(t).get("project", {}).get("version"))


def _pyproject_write(t: str, v: str) -> str:
    m = PROJECT_TABLE.search(t)
    if not m:
        return t
    body = PROJECT_VERSION.sub(lambda x: f"{x[1]}{v}{x[2]}", m[1], count=1)
    return t[: m.start(1)] + body + t[m.end(1) :]


SITES: list[VersionSite] = [
    VersionSite("pyproject.toml", read=_pyproject_read, write=_pyproject_write),
    _json_version(".claude-plugin/plugin.json"),
    _json_version(".codex-plugin/plugin.json"),
    VersionSite(
        ".claude-plugin/marketplace.json",
        read=_marketplace_read,
        write=lambda t, v: JSON_VERSION.sub(lambda m: f"{m[1]}{v}{m[2]}", t),
    ),
    _yaml_version("skills/avoid-ai-writing-russian/SKILL.md", FRONTMATTER_READ),
    _yaml_version("skills/antiplagiat/SKILL.md", FRONTMATTER_READ),
    _yaml_version("CITATION.cff", CFF_READ),
]


def read_versions(root: Path = ROOT) -> list[FoundVersion]:
    return [FoundVersion(s.file, s.read(read_text(root / s.file))) for s in SITES]


def write_versions(version: str, root: Path = ROOT) -> None:
    for s in SITES:
        path = root / s.file
        write_text(path, s.write(read_text(path), version))


def lock_version(root: Path = ROOT) -> str | None:
    """Версия самого проекта в uv.lock: пакет, который ставится из корня (`source = { editable = "." }`)."""
    path = root / LOCK
    if not path.exists():
        return None
    for p in tomllib.loads(read_text(path)).get("package", []):
        source = p.get("source") or {}
        if "." in (source.get("editable"), source.get("virtual")):
            return _str(p.get("version"))
    return None


# Версии по PEP 440 в той форме, которую пишет uv: X.Y.Z и предварительные X.Y.ZaN, bN, rcN.
VERSION = re.compile(r"\A(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:(a|b|rc)(0|[1-9]\d*))?\Z", re.ASCII)
# Предварительная версия младше выпуска: 2.0.0a1 < 2.0.0b1 < 2.0.0rc1 < 2.0.0.
_PRE_RANK = {"a": 0, "b": 1, "rc": 2, None: 3}


def _key(v: str) -> tuple[int, int, int, int, int]:
    m = VERSION.match(v)
    if not m:
        raise ValueError(f"не версия PEP 440 вида X.Y.Z или X.Y.ZrcN: {v}")
    return int(m[1]), int(m[2]), int(m[3]), _PRE_RANK[m[4]], int(m[5] or 0)


def is_prerelease(v: str) -> bool:
    m = VERSION.match(v)
    return bool(m and m[4])


def compare_versions(a: str, b: str) -> int:
    ka, kb = _key(a), _key(b)
    return (ka > kb) - (ka < kb)

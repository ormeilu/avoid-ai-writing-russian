"""Документация проекта: ссылки и пути ведут на существующие файлы, команды из README знакомы CLI.

Ссылки внутри скиллов проверяет test_skill.py. Здесь README и остальные документы проекта, кроме
CHANGELOG: в истории изменений законно упоминаются удалённые файлы.
"""

import re
import shlex
from pathlib import Path

import pytest

from aiw_ru import models
from aiw_ru.cli import USAGE, main, parse

ROOT = Path(__file__).parent.parent
README = ROOT / "README.md"
DOCS = sorted(
    [
        *(ROOT / name for name in ("README.md", "CONTRIBUTING.md", "AGENTS.md", "CLAUDE.md")),
        *(ROOT / "docs").rglob("*.md"),
    ]
)
SKILL_DIRS = sorted(path for path in (ROOT / "skills").iterdir() if (path / "SKILL.md").is_file())

# Относительная ссылка Markdown или картинка в HTML: `[текст](путь#якорь)`, `<img src="путь">`.
LINK_RE = re.compile(r"\]\((?!https?:|#|mailto:)([^)#\s]+)(?:#[^)]*)?\)|<img\b[^>]*\bsrc=\"(?!https?:)([^\"]+)\"")
INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
FENCE_RE = re.compile(r"^```bash\n(.*?)^```", re.MULTILINE | re.DOTALL)
# Путь в репозитории внутри `кода`: без пробелов, с косой чертой, не домашняя папка и не абсолютный путь.
REPO_PATH_RE = re.compile(r"[\w.-]+/(?:[\w.-]+/?)*")
# Заглушка в команде: ИМЯ, ФАЙЛ, многоточие.
PLACEHOLDER_RE = re.compile(r"\b[А-ЯЁ]{2,}\b|…")
# Команды справки: работают без сети и без моделей, их README обещает выполнить как есть.
READ_ONLY = (("skill",), ("models", "guide"), ("models", "info"), ("--version",), ("--help",))


def readme_commands() -> list[str]:
    """Вызовы aiw-ru из README: строки блоков bash и `код` в тексте."""
    text = README.read_text(encoding="utf-8")
    lines = [line for block in FENCE_RE.findall(text) for line in block.splitlines()]
    lines += INLINE_CODE_RE.findall(text)
    found: list[str] = []
    for line in lines:
        # `uvx aiw-ru scan …` — тот же вызов; вывод в файл (`> задание.txt`) к команде не относится.
        command = line.strip().removeprefix("uvx ").split(" > ")[0]
        # `aiw-ru …` в тексте — форма вызова, а не команда.
        if command.startswith("aiw-ru ") and command != "aiw-ru …" and command not in found:
            found.append(command)
    return found


COMMANDS = readme_commands()


def argv_of(command: str) -> list[str]:
    return shlex.split(command)[1:]


def is_read_only(command: str) -> bool:
    argv = argv_of(command)
    return not PLACEHOLDER_RE.search(command) and any(tuple(argv[: len(head)]) == head for head in READ_ONLY)


@pytest.mark.parametrize("doc", DOCS, ids=lambda path: str(path.relative_to(ROOT)))
def test_relative_links_exist(doc: Path):
    """относительные ссылки и картинки ведут на существующие файлы"""
    for match in LINK_RE.finditer(doc.read_text(encoding="utf-8")):
        target = match[1] or match[2]
        assert (doc.parent / target).exists(), f"{doc.name}: {target}"


def test_readme_code_paths_exist():
    """пути в `коде` README есть в репозитории или в папке скилла (`references/vocabulary.md`)"""
    paths = [
        token
        for token in INLINE_CODE_RE.findall(README.read_text(encoding="utf-8"))
        if REPO_PATH_RE.fullmatch(token) and not token.startswith(("~", "/", "../"))
    ]
    assert {"skills/", "references/vocabulary.md", ".kimi-plugin/plugin.json"} <= set(paths)
    for path in paths:
        assert any((base / path).exists() for base in (ROOT, *SKILL_DIRS)), path


def test_readme_commands_found():
    """в README есть и команды с файлами, и команды справки: иначе разбор ниже ничего не проверяет"""
    assert any(argv_of(command)[0] == "scan" for command in COMMANDS)
    assert sum(is_read_only(command) for command in COMMANDS) >= 4


@pytest.mark.parametrize("command", COMMANDS)
def test_readme_command_known_to_cli(command: str):
    """команда из README разбирается CLI: команда и параметры есть в справке"""
    args = parse(argv_of(command))
    commands = set(re.findall(r"^ {2}([a-z]+) ", USAGE.split("Параметры:")[0], re.MULTILINE))
    assert args.version or args.help or args.cmd in commands, command
    for flag in re.findall(r"(?<!\S)--?[a-z][a-z-]*", command):
        assert flag in USAGE, f"{command}: {flag}"


@pytest.mark.parametrize("command", [command for command in COMMANDS if is_read_only(command)])
def test_readme_read_only_command_runs(
    command: str, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """команда справки из README выполняется с кодом 0 и что-то печатает"""
    monkeypatch.setenv("NO_COLOR", "1")
    for model in models.MODELS.values():
        monkeypatch.setenv(model.env, str(tmp_path / "нет-модели"))
    models.load.cache_clear()
    code = main(argv_of(command))
    captured = capsys.readouterr()
    assert code == 0, f"{command}: {captured.err}"
    assert captured.out.strip()

"""Выпуск версии: `uv run python scripts/release.py 2.0.0` (или 2.0.0rc1, patch, minor, major).

1. Проверяет, что рабочее дерево чистое, ветка master, версия растёт.
2. Прогоняет проверки: типы, тесты, синхронность версий.
3. Переносит «Не выпущено» из CHANGELOG.md в раздел новой версии.
4. Обновляет версию во всех местах из scripts/versions.py, затем uv.lock через `uv lock`.
5. Делает коммит «Выпуск X.Y.Z» и аннотированный тег vX.Y.Z.

Версии по PEP 440: предварительная 2.0.0rc1 уходит на PyPI как пре-релиз, а на
GitHub выпуск помечается как pre-release.

Ничего не отправляет: публикация — `git push --follow-tags`, после чего
GitHub Actions собирает пакет и создаёт выпуск (release.yml).
`--dry-run` показывает план без изменений.
"""

from __future__ import annotations

import re
import subprocess
import sys
from datetime import UTC, datetime
from typing import NoReturn

from changelog import cut
from versions import (
    ROOT,
    VERSION,
    compare_versions,
    read_text,
    read_versions,
    utf8_output,
    write_text,
    write_versions,
)

SCRIPTS = ROOT / "scripts"


def fail(msg: str) -> NoReturn:
    print(f"release: {msg}", file=sys.stderr)
    sys.exit(1)


def bump(v: str, kind: str) -> str:
    """patch, minor, major от выпущенной части версии; иначе kind и есть новая версия."""
    ma, mi, pa = (int(x) for x in re.findall(r"\d+", v)[:3])
    if kind == "major":
        return f"{ma + 1}.0.0"
    if kind == "minor":
        return f"{ma}.{mi + 1}.0"
    if kind == "patch":
        return f"{ma}.{mi}.{pa + 1}"
    return kind


def output(*cmd: str) -> str:
    """Вывод команды; при ошибке выпуск прерывается."""
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=False)
    if r.returncode != 0:
        fail(f"`{' '.join(cmd)}` завершилась с кодом {r.returncode}: {r.stderr.strip()}")
    return r.stdout


def sh(*cmd: str) -> None:
    """Команда с выводом в терминал; при ошибке выпуск прерывается."""
    code = subprocess.run(cmd, cwd=ROOT, check=False).returncode
    if code != 0:
        fail(f"`{' '.join(cmd)}` завершилась с кодом {code}")


def main(args: list[str]) -> int:
    utf8_output()
    dry = "--dry-run" in args
    arg = next((a for a in args if not a.startswith("--")), None)

    found = read_versions()
    current = found[0].version or fail("не удалось прочитать текущую версию")
    if len({f.version for f in found}) != 1:
        fail("версии расходятся, сначала `uv run python scripts/check_versions.py`")

    if not arg:
        fail("укажите версию: X.Y.Z, X.Y.ZrcN, patch, minor или major")
    next_version = bump(current, arg)
    if not VERSION.match(next_version):
        fail(f"не версия PEP 440 вида X.Y.Z или X.Y.ZrcN: {next_version}")
    if compare_versions(next_version, current) <= 0:
        fail(f"версия {next_version} не больше текущей {current}")

    branch = output("git", "branch", "--show-current").strip()
    if branch != "master":
        fail(f"выпуск делается из master, сейчас {branch}")
    if output("git", "status", "--porcelain").strip() and not dry:
        fail("рабочее дерево не чистое")
    if output("git", "tag", "--list", f"v{next_version}").strip():
        fail(f"тег v{next_version} уже есть")

    date = datetime.now(UTC).date().isoformat()
    changelog_path = ROOT / "CHANGELOG.md"
    try:
        changelog = cut(read_text(changelog_path), next_version, date)
    except ValueError as e:
        fail(str(e))

    print(f"Выпуск {current} → {next_version} ({date})")
    if dry:
        print("--dry-run: изменения не записаны")
        return 0
    sys.stdout.flush()

    sh("uv", "run", "--group", "dev", "--group", "train", "ty", "check")
    sh("uv", "run", "--group", "dev", "--group", "train", "pytest")
    write_text(changelog_path, changelog)
    write_versions(next_version)
    # uv.lock хранит версию проекта; без обновления `uv run --locked` в CI откажет.
    # check_versions.py сверяет и его.
    sh("uv", "lock")
    sh(sys.executable, str(SCRIPTS / "check_versions.py"))
    sh("git", "add", "-A")
    sh("git", "commit", "-m", f"Выпуск {next_version}")
    sh("git", "tag", "-a", f"v{next_version}", "-m", f"Выпуск {next_version}")
    print("Готово. Отправить: git push --follow-tags")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

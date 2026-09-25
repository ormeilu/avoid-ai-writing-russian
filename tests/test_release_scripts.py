"""Скрипты выпуска и хуков: версии, CHANGELOG, сообщение коммита, невидимые символы."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from changelog import UNRELEASED, cut, section
from check_commit_msg import check_commit_message
from versions import (
    ROOT,
    SITES,
    VERSION,
    compare_versions,
    is_prerelease,
    lock_version,
    read_text,
    read_versions,
    write_versions,
)

SCRIPTS = ROOT / "scripts"
# Вывод скриптов по-русски; на Windows канал без UTF-8 его бы не принял.
ENV = {**os.environ, "PYTHONUTF8": "1"}


def run_script(name: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        env=ENV,
    )


# ── версии ──


def test_same_version_everywhere() -> None:
    """во всех местах одна версия"""
    found = read_versions()
    assert len(found) == len(SITES)
    assert len({f.version for f in found}) == 1
    assert VERSION.match(found[0].version or "")


def test_lock_has_project_version() -> None:
    """uv.lock хранит ту же версию проекта"""
    assert lock_version() == read_versions()[0].version


def test_write_versions_changes_only_version(tmp_path: Path) -> None:
    """write_versions меняет все места и только версию"""
    for s in SITES:
        (tmp_path / s.file).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / s.file, tmp_path / s.file)
    before = [read_text(tmp_path / s.file) for s in SITES]
    write_versions("9.8.7", tmp_path)
    assert {v.version for v in read_versions(tmp_path)} == {"9.8.7"}
    for s, old in zip(SITES, before, strict=True):
        old_lines = old.split("\n")
        after = read_text(tmp_path / s.file).split("\n")
        changed = [line for k, line in enumerate(after) if k >= len(old_lines) or line != old_lines[k]]
        assert changed, s.file
        for line in changed:
            assert "version" in line, s.file


def test_pyproject_touches_only_project_table() -> None:
    """в pyproject.toml меняется только версия из [project]"""
    site = next(s for s in SITES if s.file == "pyproject.toml")
    text = '[project]\nname = "x"\nversion = "1.0.0"\n\n[tool.x]\nversion = "5.5.5"\n'
    out = site.write(text, "2.0.0")
    assert out == text.replace('version = "1.0.0"', 'version = "2.0.0"')
    assert site.read(out) == "2.0.0"


def test_compare_versions() -> None:
    """сравнение версий PEP 440, предварительные младше выпуска"""
    assert compare_versions("0.2.0", "0.1.9") == 1
    assert compare_versions("1.0.0", "1.0.0") == 0
    assert compare_versions("0.10.0", "0.9.0") == 1
    assert compare_versions("2.0.0rc1", "1.0.0") == 1
    assert compare_versions("2.0.0rc1", "2.0.0") == -1
    assert compare_versions("2.0.0rc2", "2.0.0rc1") == 1
    assert compare_versions("2.0.0b3", "2.0.0rc1") == -1
    assert not VERSION.match("1.2") and not VERSION.match("1.2.3-rc.1")
    assert is_prerelease("2.0.0rc1") and not is_prerelease("2.0.0")


def test_check_versions_passes() -> None:
    """check_versions.py проходит на репозитории"""
    r = run_script("check_versions.py")
    assert r.returncode == 0, r.stderr


# ── CHANGELOG ──

LOG = f"# История\n\n{UNRELEASED}\n\n### Добавлено\n\n- новое\n\n## [0.1.0] — 2026-01-01\n\n- старое\n"


def test_section_extracts_sections() -> None:
    """section достаёт разделы"""
    assert section(LOG, "unreleased") == "### Добавлено\n\n- новое"
    assert section(LOG, "0.1.0") == "- старое"
    assert section(LOG, "9.9.9") is None


def test_cut_moves_unreleased() -> None:
    """cut переносит «Не выпущено» в новую версию"""
    out = cut(LOG, "0.2.0", "2026-10-01")
    assert section(out, "unreleased") == ""
    assert section(out, "0.2.0") == "### Добавлено\n\n- новое"
    assert section(out, "0.1.0") == "- старое"


def test_cut_refuses_empty_section() -> None:
    """cut отказывается выпускать пустой раздел"""
    with pytest.raises(ValueError):
        cut(f"{UNRELEASED}\n\n## [0.1.0]\n", "0.2.0", "2026-10-01")


def test_changelog_has_current_version() -> None:
    """в проекте есть раздел текущей версии"""
    changelog = read_text(ROOT / "CHANGELOG.md")
    version = read_versions()[0].version or ""
    assert section(changelog, version)
    assert UNRELEASED in changelog


def test_release_notes_cli() -> None:
    """release_notes.py печатает раздел версии и отказывает для неизвестной"""
    version = read_versions()[0].version or ""
    r = run_script("release_notes.py", f"v{version}")
    assert r.returncode == 0, r.stderr
    assert r.stdout == f"{section(read_text(ROOT / 'CHANGELOG.md'), version)}\n"
    r = run_script("release_notes.py", "9.9.9")
    assert r.returncode == 1
    assert "нет раздела для версии 9.9.9" in r.stderr


# ── сообщение коммита ──


def test_commit_message_russian_passes() -> None:
    """русское сообщение проходит"""
    assert check_commit_message("детектор: быстрее в научном режиме\n\nПодробности.\n") == []


def test_commit_message_rejects_bad() -> None:
    """английское, длинное, с точкой и без пустой строки — нет"""
    assert "первая строка должна быть на русском" in check_commit_message("fix bug")
    assert "длиннее 72" in ",".join(check_commit_message(f"{'очень ' * 15}длинно"))
    assert "первая строка не должна заканчиваться точкой" in check_commit_message("исправлено.")
    assert "после первой строки нужна пустая строка" in check_commit_message("строка\nсразу тело")


def test_commit_message_skips_merges_and_comments() -> None:
    """слияния, fixup и комментарии git пропускаются"""
    assert check_commit_message("Merge branch 'x'") == []
    assert check_commit_message("fixup! что-то") == []
    assert check_commit_message("# комментарий\nправка тестов\n") == []


def test_commit_msg_cli(tmp_path: Path) -> None:
    """check_commit_msg.py: код выхода и сообщения"""
    good, bad = tmp_path / "good", tmp_path / "bad"
    good.write_bytes("правка тестов\n".encode())
    bad.write_bytes(b"fix bug\n")
    assert run_script("check_commit_msg.py", str(good)).returncode == 0
    r = run_script("check_commit_msg.py", str(bad))
    assert r.returncode == 1
    assert "commit-msg: первая строка должна быть на русском" in r.stderr
    assert run_script("check_commit_msg.py").returncode == 2


# ── проверка невидимых символов ──


def run_invisible(tmp_path: Path, content: str, name: str = "x.md") -> subprocess.CompletedProcess[str]:
    f = tmp_path / name
    f.write_bytes(content.encode("utf-8"))
    return run_script("check_invisible.py", str(f))


def test_invisible_clean_file(tmp_path: Path) -> None:
    """чистый файл проходит"""
    assert run_invisible(tmp_path, "Обычный текст.").returncode == 0


def test_invisible_char_caught(tmp_path: Path) -> None:
    """невидимый символ ловится"""
    r = run_invisible(tmp_path, f"Текст{chr(0x200B)}.")
    assert r.returncode == 1
    assert "невидимый символ U+200B в позиции 5" in r.stderr


def test_invisible_latin_in_russian_word(tmp_path: Path) -> None:
    """латиница в русском слове ловится"""
    r = run_invisible(tmp_path, f"р{chr(0x61)}бота")
    assert r.returncode == 1
    assert "латиница внутри русского слова" in r.stderr


def test_invisible_python_file(tmp_path: Path) -> None:
    """Python-файл проверяется так же"""
    assert run_invisible(tmp_path, 'MSG = "Обычный текст"\n', "x.py").returncode == 0
    assert run_invisible(tmp_path, f'MSG = "Текст{chr(0xFEFF)}"\n', "x.py").returncode == 1


# ── release ──


def test_release_dry_run_changes_nothing() -> None:
    """--dry-run не меняет файлы"""
    before = [read_text(ROOT / s.file) for s in SITES]
    changelog = read_text(ROOT / "CHANGELOG.md")
    r = run_script("release.py", "patch", "--dry-run")
    assert [read_text(ROOT / s.file) for s in SITES] == before
    assert read_text(ROOT / "CHANGELOG.md") == changelog
    # В ветке, отличной от master, скрипт отказывает с кодом 1 — это тоже корректно.
    assert r.returncode in (0, 1), r.stderr


def test_release_refuses_lower_version() -> None:
    """отказывает для версии не больше текущей"""
    r = run_script("release.py", "0.0.1", "--dry-run")
    assert r.returncode == 1
    assert "не больше текущей" in r.stderr

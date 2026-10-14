"""Хук private-texts: в репозиторий не попадают фрагменты собственных работ автора."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from check_private import main

WORK = (
    "Весы под каждой полкой передают показания раз в минуту, "
    "а камера над проходом снимает ряд целиком, когда мимо проезжает погрузчик."
)


@pytest.fixture
def private_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    folder = tmp_path / "private"
    folder.mkdir()
    monkeypatch.setenv("AIW_RU_PRIVATE", str(folder))
    return folder


def test_fragment_of_work_fails_with_file_and_line(
    private_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """восемь слов подряд из работы — ошибка с файлом и строкой"""
    (private_dir / "статья.txt").write_text(WORK, encoding="utf-8")
    leaked = tmp_path / "test_example.py"
    leaked.write_text(
        'TEXT = "Пример."\nOTHER = "а камера над проходом снимает ряд целиком, когда мимо"\n', encoding="utf-8"
    )

    code = main([str(leaked)])

    assert code == 1
    assert f"{leaked}:2" in capsys.readouterr().err


def test_unrelated_text_passes(private_dir: Path, tmp_path: Path):
    """общие слова без восьми подряд из работы проходят"""
    (private_dir / "статья.txt").write_text(WORK, encoding="utf-8")
    clean = tmp_path / "clean.md"
    clean.write_text("Камера снимает склад. Весы стоят под полками и передают показания.\n", encoding="utf-8")

    assert main([str(clean)]) == 0


def test_without_private_folder_check_is_silent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """нет папки с работами (CI, другие участники) — проверка проходит"""
    monkeypatch.setenv("AIW_RU_PRIVATE", str(tmp_path / "нет такой папки"))
    anything = tmp_path / "file.md"
    anything.write_text(WORK, encoding="utf-8")

    assert main([str(anything)]) == 0


def test_docx_work_is_read(private_dir: Path, tmp_path: Path):
    """работа в .docx читается без pandoc: текст абзацев из word/document.xml"""
    body = "".join(f"<w:p><w:r><w:t>{part}</w:t></w:r></w:p>" for part in WORK.split(", "))
    with zipfile.ZipFile(private_dir / "реферат.docx", "w") as docx:
        docx.writestr("word/document.xml", f'<w:document xmlns:w="w"><w:body>{body}</w:body></w:document>')
    leaked = tmp_path / "leak.md"
    leaked.write_text("передают показания раз в минуту, а камера над проходом\n", encoding="utf-8")

    assert main([str(leaked)]) == 1

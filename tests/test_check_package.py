"""scripts/check_package.py: проверка wheel вне рабочей копии. Здесь логика на подставном запуске;
настоящий wheel проверяет CI после `uv build`."""

import json
from pathlib import Path

import pytest

from check_package import ROOT, Result, check, main

SKILLS = [{"name": "antiplagiat", "files": ["references/domate.md"]}]


class FakeRunner:
    """Подставной запуск: отвечает как рабочий пакет, а поломки задаются параметрами."""

    def __init__(
        self, *, file: str = "/site-packages/aiw_ru/__init__.py", heavy: list[str] | None = None, broken: str = ""
    ):
        self.file, self.heavy, self.broken = file, heavy or [], broken
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str], cwd: Path, stdin: str) -> Result:
        self.calls.append(argv)
        assert not cwd.resolve().is_relative_to(ROOT), "команды идут из пустой папки вне рабочей копии"
        assert "--isolated" in argv and "--no-project" in argv
        command = argv[argv.index("--with") + 2 :]
        if self.broken and self.broken in command:
            return Result(1, "", f"нет файла {self.broken}")
        if command[:2] == ["python", "-c"]:
            return Result(0, json.dumps({"file": self.file, "heavy": self.heavy}), "")
        if command[1:3] == ["skill", "--json"]:
            return Result(0, json.dumps(SKILLS), "")
        if command[1:2] == ["scan"]:
            return Result(0, json.dumps({"issues": [{"type": "lets"}]}), "")
        if command[1:2] == ["validate"]:
            return Result(0, json.dumps({"ok": True}), "")
        return Result(0, "текст", "")


def test_good_wheel_passes(tmp_path: Path):
    """рабочий пакет: провалов нет, каждый файл скилла прочитан"""
    runner = FakeRunner()
    assert check(tmp_path / "aiw_ru.whl", runner) == []
    commands = [argv[argv.index("--with") + 2 :] for argv in runner.calls]
    assert ["aiw-ru", "skill", "antiplagiat", "references/domate.md"] in commands


@pytest.mark.parametrize(
    ("runner", "failure"),
    [
        pytest.param(
            FakeRunner(file=str(ROOT / "src" / "aiw_ru" / "__init__.py")),
            "из рабочей копии",
            id="модуль из рабочей копии",
        ),
        pytest.param(FakeRunner(heavy=["numpy", "torch"]), "тяжёлые пакеты: numpy, torch", id="тяжёлые пакеты"),
        pytest.param(
            FakeRunner(broken="references/domate.md"), "skill antiplagiat references/domate.md: код 1", id="файл скилла"
        ),
    ],
)
def test_broken_wheel_reports_failure(tmp_path: Path, runner: FakeRunner, failure: str):
    """каждая поломка называется в списке провалов"""
    assert any(failure in line for line in check(tmp_path / "aiw_ru.whl", runner))


def test_main_needs_wheel(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """без пути к .whl — код 2"""
    assert main([]) == 2
    assert main([str(tmp_path / "нет.whl")]) == 2
    assert "передайте путь" in capsys.readouterr().err

"""Командная строка aiw-ru: команды, коды выхода, таблицы и цвет.

Почти всё запускается в том же процессе через `main()`. Отдельный процесс нужен только
там, где проверяется сама точка входа: `python -m aiw_ru`, консольная команда и обрыв канала.
"""

import io
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pytest

from aiw_ru.cli import USAGE, Col, layout, main

ROOT = Path(__file__).parent.parent
CORPUS = ROOT / "tests" / "fixtures" / "corpus"
AI_BLOG = str(CORPUS / "ai" / "blog.md")
HUMAN_BLOG = str(CORPUS / "human" / "blog.md")
AI_VAK = CORPUS / "ai" / "vak.md"
HUMAN_VAK = CORPUS / "human" / "vak.md"
ESC = "\x1b"


@dataclass
class Result:
    code: int
    out: str
    err: str


class Cli(Protocol):
    def __call__(
        self, *args: str, stdin: str | None = None, cwd: Path | None = None, env: dict[str, str] | None = None
    ) -> Result: ...


@pytest.fixture
def cli(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Cli:
    """Запуск `main()` в этом процессе: без цвета, без калибровки, в пустом каталоге.

    NO_COLOR включён, FORCE_COLOR выключен, пока тест не
    попросит иначе. HOME подменён, чтобы не подхватить ~/.config/aiw-ru.json.
    """
    monkeypatch.setenv("AIW_RU_CONFIG", "")
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.delenv("COLUMNS", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)

    def run(*args: str, stdin: str | None = None, cwd: Path | None = None, env: dict[str, str] | None = None) -> Result:
        with monkeypatch.context() as m:
            for k, v in (env or {}).items():
                m.setenv(k, v)
            if env and "FORCE_COLOR" in env:
                m.delenv("NO_COLOR", raising=False)
            if cwd is not None:
                m.chdir(cwd)
            m.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO((stdin or "").encode()), encoding="utf-8"))
            capsys.readouterr()
            code = main(list(args))
            captured = capsys.readouterr()
        return Result(code, captured.out, captured.err)

    return run


def run_module(*args: str, stdin: str = "", cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Отдельный процесс `python -m aiw_ru`: код выхода и потоки как у настоящей команды."""
    return subprocess.run(
        [sys.executable, "-m", "aiw_ru", *args],
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=cwd,
        env=clean_env(),
        check=False,
    )


def clean_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in ("FORCE_COLOR", "COLUMNS")}
    env.update(AIW_RU_CONFIG="", NO_COLOR="1", PYTHONIOENCODING="utf-8")
    return env


def rows(out: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in out.strip().split("\n")]


# ─── команды ────────────────────────────────────────────────────────────


def test_scan_file(cli: Cli):
    """scan файла"""
    r = cli("scan", AI_BLOG)
    assert r.code == 0
    assert "сильный ИИ-стиль" in r.out


def test_color_no_color_by_default_force_color_enables(cli: Cli):
    """цвет: NO_COLOR по умолчанию в тестах, FORCE_COLOR включает"""
    assert ESC not in cli("scan", stdin="Давайте разберёмся.").out
    colored = cli("scan", stdin="Давайте разберёмся.", env={"FORCE_COLOR": "1"}).out
    assert f"{ESC}[33mP1{ESC}[0m" in colored
    assert ESC not in cli("scan", "--json", stdin="Давайте разберёмся.", env={"FORCE_COLOR": "1"}).out


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Мы пришли домой поздно.", "в среднем 4 слова в предложении"),
        ("Мы пришли домой очень поздно вечером.", "в среднем 6 слов в предложении"),
        ("Мы пришли домой. Мы пришли домой поздно.", "в среднем 3,5 слова в предложении"),
    ],
)
def test_mean_sentence_length_agrees_with_number(cli: Cli, text: str, expected: str):
    """средняя длина предложения согласована с числом"""
    assert expected in cli("scan", stdin=text).out


def test_scan_stdin_json(cli: Cli):
    """scan из stdin в JSON"""
    r = cli("scan", "--json", stdin="Давайте разберёмся.")
    assert json.loads(r.out)["issues"][0]["type"] == "lets"


def test_skill_profile_as_context(cli: Cli):
    """профиль скилла как --context"""
    r = cli("scan", "--json", "--context", "vak", stdin="Текст.")
    assert json.loads(r.out)["stats"]["contextMode"] == "academic"


def test_fail_above(cli: Cli):
    """--fail-above"""
    assert cli("scan", "--fail-above", "50", AI_BLOG).code == 1
    assert cli("scan", "--fail-above", "50", HUMAN_BLOG).code == 0


def test_antiplagiat(cli: Cli):
    """antiplagiat"""
    r = cli("antiplagiat", AI_BLOG)
    assert r.code == 0
    assert "Доля ИИ-текста" in r.out


def test_validate_returns_1_on_damage(cli: Cli):
    """validate возвращает 1 при повреждении"""
    assert cli("validate", AI_BLOG, HUMAN_BLOG).code == 1
    assert cli("validate", HUMAN_BLOG, HUMAN_BLOG).code == 0


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(["scan", "--context", "нет-такого"], id="неизвестный контекст"),
        pytest.param(["frobnicate"], id="неизвестная команда"),
        pytest.param([], id="без команды"),
        pytest.param(["scan", "--fail-above", "много"], id="--fail-above не число"),
        pytest.param(["scan", "--fail-above", "nan"], id="--fail-above nan"),
        pytest.param(["scan", "--min", "P3"], id="неизвестный --min"),
        pytest.param(["scan", "--context"], id="параметр без значения"),
        pytest.param(["scan", "--verbose"], id="неизвестный параметр"),
        pytest.param(["validate", AI_BLOG], id="validate с одним файлом"),
    ],
)
def test_usage_errors_exit_2(cli: Cli, args: list[str]):
    """ошибки использования — код 2"""
    r = cli(*args)
    assert r.code == 2
    # Без команды справка идёт в stdout, как у --help; остальные ошибки — в stderr с причиной.
    assert "Команды:" in (r.err if args else r.out)
    if args:
        assert r.err.startswith("aiw-ru: ")


def test_help(cli: Cli):
    """--help печатает справку и выходит с кодом 0"""
    r = cli("--help")
    assert r.code == 0
    assert r.out == USAGE + "\n"


def test_min_hides_lower_levels(cli: Cli):
    """--min скрывает находки ниже уровня"""
    everything = cli("scan", AI_BLOG).out
    p0 = cli("scan", "--min", "P0", AI_BLOG).out

    def row(level: str) -> re.Pattern[str]:
        return re.compile(rf"^  {level}\s+\d+:\d+", re.MULTILINE)

    assert row("(P2|стиль)").search(everything)
    assert not row("(P1|P2|стиль)").search(p0)
    assert row("P0").search(p0)


def test_scan_table_fits_columns_p0_above_p1(cli: Cli):
    """scan: таблица по ширине COLUMNS, P0 выше P1"""
    r = cli("scan", AI_BLOG, env={"COLUMNS": "80"})
    lines = r.out.split("\n")
    # Первая строка — имя файла, оно печатается целиком.
    for line in lines[1:]:
        assert len(line) <= 80, line
    assert "Уровень  Где" in r.out
    levels = [m[1] for line in lines if (m := re.match(r"^ {2}(P0|P1|P2|стиль)\s+\d", line))]
    assert levels.index("P0") < levels.index("P1")


def test_scan_clean_text_and_plural_agreement(cli: Cli):
    """scan чистого текста и согласование числительных"""
    r = cli("scan", stdin="Одно слово.")
    assert "2 слова, 1 предложение" in r.out
    assert "Примет не найдено." in r.out


def test_scan_many_files_json_array(cli: Cli):
    """scan нескольких файлов в JSON — массив"""
    data = json.loads(cli("scan", "--json", AI_BLOG, HUMAN_BLOG).out)
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["file"] == AI_BLOG
    assert data[0]["score"] > data[1]["score"]


def test_non_utf8_input_exit_2(cli: Cli, tmp_path: Path):
    """не-UTF-8 ввод — код 2"""
    bad = tmp_path / "bad.txt"
    bad.write_bytes(bytes([0xFF, 0xFE, 0x00, 0xC3]))
    r = cli("scan", str(bad))
    assert r.code == 2
    assert "не удалось прочитать" in r.err


def test_missing_file_exit_2(cli: Cli, tmp_path: Path):
    """нет файла — код 2 и понятная ошибка"""
    r = cli("scan", str(tmp_path / "нет.md"))
    assert r.code == 2
    assert "не удалось прочитать" in r.err


def test_scan_and_antiplagiat_jsonl(cli: Cli):
    """scan и antiplagiat --jsonl: строка результата на каждую строку входа"""
    source = "\n".join(
        [
            json.dumps({"text": "Давайте разберёмся, это играет ключевую роль.", "label": "ai"}, ensure_ascii=False),
            "",
            json.dumps({"text": "Вчера сломался чайник, купили новый."}, ensure_ascii=False),
        ]
    )
    scan = cli("scan", "--jsonl", stdin=source)
    assert scan.code == 0
    results = rows(scan.out)
    assert len(results) == 2
    assert "lets" in [i["type"] for i in results[0]["issues"]]
    assert results[1]["issues"] == []

    ap = cli("antiplagiat", "--jsonl", stdin=source)
    assert ap.code == 0
    assert [type(r["aiShare"]) in (int, float) for r in rows(ap.out)] == [True, True]

    bad = cli("scan", "--jsonl", stdin=f'{json.dumps({"text": "Раз."})}\n{{"body": 1}}')
    assert bad.code == 2
    assert "строка 2" in bad.err


@pytest.mark.parametrize(
    ("line", "reason"),
    [
        pytest.param("{не json", "не JSON", id="не JSON"),
        pytest.param("[1]", "нет строкового поля text", id="массив"),
        pytest.param('"текст"', "нет строкового поля text", id="строка"),
        pytest.param('{"text": 5}', "нет строкового поля text", id="text не строка"),
    ],
)
def test_jsonl_bad_line_reason(cli: Cli, line: str, reason: str):
    """--jsonl: причина ошибки в строке входа"""
    r = cli("scan", "--jsonl", stdin=line)
    assert r.code == 2
    assert f"строка 1: {reason}" in r.err


def test_jsonl_empty_input_prints_nothing(cli: Cli):
    """--jsonl без документов ничего не печатает"""
    r = cli("scan", "--jsonl", stdin="\n\n")
    assert r.code == 0
    assert r.out == ""


def test_antiplagiat_json(cli: Cli):
    """antiplagiat --json"""
    data = json.loads(cli("antiplagiat", "--json", str(AI_VAK)).out)
    assert data["aiShare"] > 50
    assert isinstance(data["fragments"], list)
    assert data["model"] == "default"


def test_validate_json(cli: Cli):
    """validate --json: поля в camelCase"""
    data = json.loads(cli("validate", "--json", HUMAN_BLOG, HUMAN_BLOG).out)
    assert data["ok"] is True
    assert data["violations"] == []
    assert data["issuesBefore"] == data["issuesAfter"]


def test_calibrate_creates_file_antiplagiat_uses_it(cli: Cli, tmp_path: Path):
    """calibrate создаёт файл, antiplagiat его подхватывает"""
    doc = tmp_path / "doc.md"
    marked = tmp_path / "marked.txt"
    ai = AI_VAK.read_text(encoding="utf-8")
    human = HUMAN_VAK.read_text(encoding="utf-8")
    doc.write_text(f"{ai}\n\n{human}", encoding="utf-8")
    marked.write_text(ai.split("\n\n")[1], encoding="utf-8")
    config = tmp_path / ".aiw-ru.json"

    cal = cli("calibrate", "--doc", str(doc), "--marked", str(marked), cwd=tmp_path)
    assert cal.code == 0
    assert "Фрагменты с разметкой" in cal.out
    assert config.exists()
    saved = json.loads(config.read_text(encoding="utf-8"))
    assert saved["version"] == 1
    assert len([s for s in saved["samples"] if s["y"] == 1]) == 1
    assert len(saved["samples"]) > 1

    again = cli("calibrate", "--doc", str(doc), "--marked", str(marked), cwd=tmp_path)
    saved_again = json.loads(config.read_text(encoding="utf-8"))
    assert again.code == 0
    assert len(saved_again["samples"]) == len(saved["samples"]) * 2

    report = json.loads(cli("antiplagiat", "--json", str(doc), cwd=tmp_path).out)
    assert report["model"] == "calibrated"


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(["--doc", "x.md"], id="--doc без разметки"),
        pytest.param(["--marked", "m.txt", "--doc", "x.md"], id="--marked до --doc"),
        pytest.param(["--doc", "x.md", "--share", "150"], id="--share больше 100"),
        pytest.param(["--doc", "x.md", "--marked", "m.txt", "--share", "10"], id="две разметки у одного --doc"),
    ],
)
def test_calibrate_without_pairs_exit_2(cli: Cli, args: list[str]):
    """calibrate без пар --doc/--marked — код 2"""
    assert cli("calibrate", *args).code == 2


def test_calibrate_share_from_report_total(cli: Cli, tmp_path: Path):
    """calibrate --share: итоговая доля из отчёта без разметки фрагментов"""
    doc = tmp_path / "doc.md"
    doc.write_text(AI_VAK.read_text(encoding="utf-8"), encoding="utf-8")
    before = json.loads(cli("antiplagiat", "--json", str(doc), cwd=tmp_path).out)
    assert before["aiShare"] > 50

    cal = cli("calibrate", "--doc", str(doc), "--share", "0%", cwd=tmp_path)
    assert cal.code == 0
    assert "доля 0 %" in cal.out
    assert "Ошибка доли ИИ" in cal.out
    saved = json.loads((tmp_path / ".aiw-ru.json").read_text(encoding="utf-8"))
    assert len(saved["documents"]) == 1

    after = json.loads(cli("antiplagiat", "--json", str(doc), cwd=tmp_path).out)
    assert after["model"] == "calibrated"
    assert after["aiShare"] < before["aiShare"]


def test_calibrate_warns_when_marked_not_found(cli: Cli, tmp_path: Path):
    """calibrate предупреждает, если размеченный кусок не найден в документе"""
    doc = tmp_path / "doc.md"
    marked = tmp_path / "marked.txt"
    doc.write_text(AI_VAK.read_text(encoding="utf-8"), encoding="utf-8")
    marked.write_text("Этого текста в документе нет.", encoding="utf-8")
    r = cli("calibrate", "--doc", str(doc), "--marked", str(marked), cwd=tmp_path)
    assert r.code == 0
    assert "не найден в документе" in r.out


def test_config_flag_and_env(cli: Cli, tmp_path: Path):
    """--config и AIW_RU_CONFIG указывают на файл калибровки"""
    doc = tmp_path / "doc.md"
    doc.write_text(AI_VAK.read_text(encoding="utf-8"), encoding="utf-8")
    config = tmp_path / "cal" / "aiw-ru.json"
    config.parent.mkdir()
    assert cli("calibrate", "--doc", str(doc), "--share", "10", "--config", str(config)).code == 0
    assert config.exists()
    assert not (Path.cwd() / ".aiw-ru.json").exists()
    with_flag = json.loads(cli("antiplagiat", "--json", "--config", str(config), str(doc)).out)
    with_env = json.loads(cli("antiplagiat", "--json", str(doc), env={"AIW_RU_CONFIG": str(config)}).out)
    without = json.loads(cli("antiplagiat", "--json", str(doc)).out)
    assert with_flag["model"] == with_env["model"] == "calibrated"
    assert without["model"] == "default"


def test_broken_calibration_file_clear_error(cli: Cli, tmp_path: Path):
    """битый файл калибровки — понятная ошибка"""
    (tmp_path / ".aiw-ru.json").write_text("{не json", encoding="utf-8")
    r = cli("antiplagiat", str(AI_VAK), cwd=tmp_path)
    assert r.code == 2
    assert "калибровку" in r.err


# ─── таблицы rich ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(["scan", AI_BLOG], id="scan"),
        pytest.param(["antiplagiat", str(AI_VAK)], id="antiplagiat"),
    ],
)
def test_narrow_screen_wraps_flex_columns(cli: Cli, args: list[str]):
    """COLUMNS=60: гибкие колонки переносятся, ни одна строка не шире экрана"""
    r = cli(*args, env={"COLUMNS": "60"})
    lines = r.out.split("\n")
    # Имя файла печатается целиком, даже если оно длиннее экрана.
    assert lines[0] == args[1]
    for line in lines[1:]:
        assert len(line) <= 60, line
    # Продолжение ячейки: первая колонка пустая, текст идёт дальше в гибкой колонке.
    assert any(re.match(r"^ {9,}\S", line) for line in lines)


def test_narrow_screen_keeps_all_words(cli: Cli):
    """COLUMNS=60: перенос не теряет слов подсказки"""
    r = cli("scan", "--min", "P0", AI_BLOG, env={"COLUMNS": "60"})
    words = set(r.out.split())
    for hint in ("назвать", "источник", "([N])", "убрать", "ссылку", "авторитет"):
        assert hint in words


def test_truncated_column_single_line_with_ellipsis(cli: Cli):
    """колонка без переноса обрезается с «…» в одну строку"""
    r = cli("antiplagiat", str(AI_VAK), env={"COLUMNS": "100"})
    starts = [line for line in r.out.split("\n") if re.match(r"^ {2}\d", line)]
    assert starts
    for line in starts:
        assert "… " in line or line.rstrip().endswith("…")
        assert " …" not in line


def test_no_color_has_no_ansi(cli: Cli):
    """NO_COLOR: ни одного ANSI-кода"""
    for args in (["scan", AI_BLOG], ["antiplagiat", str(AI_VAK)], ["validate", AI_BLOG, HUMAN_BLOG]):
        assert ESC not in cli(*args).out


@pytest.mark.parametrize("value", ["1", "true"])
def test_force_color_has_ansi(cli: Cli, value: str):
    """FORCE_COLOR: ANSI-коды есть, цвет не захватывает пробелы выравнивания"""
    out = cli("scan", AI_BLOG, env={"FORCE_COLOR": value}).out
    assert f"{ESC}[31mP0{ESC}[0m" in out
    assert f"{ESC}[1m{AI_BLOG}{ESC}[0m" in out
    assert not re.search(rf"{ESC}\[3[13]m\S+ +{ESC}\[0m", out)


def test_force_color_zero_means_no_color(cli: Cli):
    """FORCE_COLOR=0 цвет не включает"""
    assert ESC not in cli("scan", AI_BLOG, env={"FORCE_COLOR": "0"}).out


def test_each_call_reads_environment(cli: Cli):
    """каждый вызов main() заново читает цвет и ширину из окружения"""
    wide = cli("scan", AI_BLOG, env={"COLUMNS": "160"}).out
    narrow = cli("scan", AI_BLOG, env={"COLUMNS": "60"}).out
    assert max(map(len, wide.split("\n")[1:])) > 60
    assert max(map(len, narrow.split("\n")[1:])) <= 60
    assert ESC in cli("scan", AI_BLOG, env={"FORCE_COLOR": "1"}).out
    assert ESC not in cli("scan", AI_BLOG).out


def test_layout_shrinks_flex_columns_not_below_min_while_fits():
    """сужаются только flex-колонки и не уже min, пока таблица влезает"""
    cols = [Col("A"), Col("Текст", flex=True, min=8)]
    cells: list[list[Any]] = [["x", "очень длинный текст ячейки, который не влезает"]]
    assert layout(cols, cells, 100) == [1, 46]
    assert layout(cols, cells, 30) == [1, 25]
    assert layout(cols, cells, 13) == [1, 8]
    # Уже min — только если иначе строка шире экрана.
    assert layout(cols, cells, 10) == [1, 5]


# ─── точка входа ────────────────────────────────────────────────────────


def test_module_entry_point_reads_stdin():
    """python -m aiw_ru читает stdin и печатает JSON"""
    p = run_module("scan", "--json", stdin="Давайте разберёмся.")
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout)["issues"][0]["type"] == "lets"


@pytest.mark.parametrize(
    ("args", "code"),
    [
        pytest.param(["scan", "--fail-above", "50", AI_BLOG], 1, id="порог превышен"),
        pytest.param(["scan", "--fail-above", "50", HUMAN_BLOG], 0, id="порог не превышен"),
        pytest.param(["frobnicate"], 2, id="ошибка использования"),
    ],
)
def test_module_exit_codes(args: list[str], code: int):
    """коды выхода процесса: 0, 1 и 2"""
    p = run_module(*args)
    assert p.returncode == code, p.stderr
    if code == 2:
        assert p.stderr.startswith("aiw-ru: неизвестная команда: frobnicate")


def test_console_script():
    """консольная команда aiw-ru из пакета"""
    script = Path(sys.executable).parent / ("aiw-ru.exe" if sys.platform == "win32" else "aiw-ru")
    if not script.exists():
        pytest.skip("пакет не установлен в окружение тестов")
    p = subprocess.run(
        [str(script), "--help"], capture_output=True, text=True, encoding="utf-8", env=clean_env(), check=False
    )
    assert p.returncode == 0
    assert p.stdout == USAGE + "\n"


def test_broken_pipe_is_not_an_error():
    """читатель закрыл канал (`aiw-ru scan … | head`): код 0 и без трассировки"""
    p = subprocess.Popen(
        [sys.executable, "-m", "aiw_ru", "scan", *[AI_BLOG] * 60],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=clean_env(),
    )
    assert p.stdout is not None
    assert p.stderr is not None
    p.stdout.readline()
    p.stdout.close()
    err = p.stderr.read().decode("utf-8")
    assert p.wait(timeout=60) == 0
    assert "Traceback" not in err
    assert "BrokenPipeError" not in err

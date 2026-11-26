"""Командная строка aiw-ru: команды, коды выхода, таблицы и цвет.

Почти всё запускается в том же процессе через `main()`. Отдельный процесс нужен только
там, где проверяется сама точка входа: `python -m aiw_ru`, консольная команда и обрыв канала.
"""

import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pytest

from aiw_ru import __version__, models, skills
from aiw_ru import cli as cli_module
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
    попросит иначе. HOME и USERPROFILE подменены, чтобы не подхватить ~/.config/aiw-ru.json:
    на Windows `Path.home()` читает USERPROFILE, а не HOME.
    """
    monkeypatch.setenv("AIW_RU_CONFIG", "")
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.delenv("COLUMNS", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    # Модели из кэша Hugging Face разработчика не должны попадать в вывод.
    for model in models.MODELS.values():
        monkeypatch.setenv(model.env, str(tmp_path / "нет-модели"))
    models.load.cache_clear()
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


def test_scan_names_catalog_files(cli: Cli):
    """scan называет файл и раздел каталога для каждой приметы: агент открывает только их"""
    text = "Надеюсь, это поможет! Данный метод является лучшим."
    data = json.loads(cli("scan", "--json", stdin=text).out)
    catalog = {c["file"]: c["sections"] for c in data["catalog"]}
    assert catalog["references/chat.md"] == ["Следы чат-бота"]
    assert "Уровень 1А — маркеры ИИ" in catalog["references/vocabulary.md"]
    assert next(iter(catalog)) == "references/chat.md"  # P0 первым
    out = cli("scan", stdin=text).out
    assert "references/chat.md (Следы чат-бота)" in " ".join(out.split())
    assert json.loads(cli("scan", "--json", stdin="Мы пришли домой.").out)["catalog"] == []


def test_skill_profile_as_context(cli: Cli):
    """профиль скилла как --context"""
    r = cli("scan", "--json", "--context", "vak", stdin="Текст.")
    assert json.loads(r.out)["stats"]["contextMode"] == "academic"


def test_fail_above(cli: Cli):
    """--fail-above"""
    assert cli("scan", "--fail-above", "50", AI_BLOG).code == 1
    assert cli("scan", "--fail-above", "50", HUMAN_BLOG).code == 0


@pytest.mark.parametrize(
    ("files", "threshold", "code"),
    [
        pytest.param([AI_BLOG], "0", 1, id="один документ выше порога"),
        pytest.param([HUMAN_BLOG], "50", 0, id="один документ ниже порога"),
        pytest.param([HUMAN_BLOG, AI_BLOG], "50", 1, id="второй из двух выше порога"),
        pytest.param([HUMAN_BLOG, HUMAN_BLOG], "50", 0, id="оба ниже порога"),
    ],
)
def test_fail_above_same_code_in_every_output(cli: Cli, files: list[str], threshold: str, code: int):
    """--fail-above: тот же код выхода в таблице, JSON и JSONL, а JSONL печатает все документы"""
    stream = "\n".join(json.dumps({"text": Path(file).read_text(encoding="utf-8")}) for file in files)
    assert cli("scan", "--fail-above", threshold, *files).code == code
    assert cli("scan", "--json", "--fail-above", threshold, *files).code == code
    batch = cli("scan", "--jsonl", "--fail-above", threshold, stdin=stream)
    assert batch.code == code
    assert len(rows(batch.out)) == len(files)


def test_jsonl_fail_above_without_documents(cli: Cli):
    """--jsonl --fail-above без документов: код 0, вывода нет"""
    r = cli("scan", "--jsonl", "--fail-above", "0", stdin="\n\n")
    assert r.code == 0
    assert r.out == ""


def test_jsonl_without_fail_above_keeps_exit_0(cli: Cli):
    """--jsonl без --fail-above: код 0 при любой оценке"""
    assert cli("scan", "--jsonl", stdin=json.dumps({"text": Path(AI_BLOG).read_text(encoding="utf-8")})).code == 0


def test_jsonl_bad_line_wins_over_fail_above(cli: Cli):
    """--jsonl --fail-above: ошибка в строке входа — код 2, даже если документ до неё выше порога"""
    stream = f'{json.dumps({"text": Path(AI_BLOG).read_text(encoding="utf-8")})}\n{{"body": 1}}'
    r = cli("scan", "--jsonl", "--fail-above", "0", stdin=stream)
    assert r.code == 2
    assert "строка 2" in r.err


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


@pytest.mark.parametrize("argv", [["--version"], ["scan", "--version"]])
def test_version(cli: Cli, argv: list[str]):
    """--version печатает версию пакета и выходит с кодом 0, даже рядом с командой"""
    r = cli(*argv)
    assert r.code == 0
    assert r.out == f"aiw-ru {__version__}\n"
    assert "--version" in USAGE


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


def test_scan_shows_light(cli: Cli):
    """scan показывает светофор символов строкой и в JSON"""
    assert re.search(r"Светофор +зелёный: всё чисто", cli("scan", stdin="Одно слово.").out)
    nbsp = chr(0x00A0)
    r = cli("scan", stdin=f"Задержка 25{nbsp}мс.")
    assert "жёлтый: ИИ или типография? Уточни у пользователя, убрать или оставить" in r.out
    light = json.loads(cli("scan", "--json", stdin=f"Задержка 25{nbsp}мс.").out)["light"]
    assert light == {
        "level": "yellow",
        "hint": "ИИ или типография? Уточни у пользователя, убрать или оставить",
        "types": ["typography-chars"],
    }


def test_validate_marked_lists_unedited_red_paragraphs(cli: Cli, tmp_path: Path):
    """validate --marked называет красные абзацы отчёта, оставшиеся без правки"""
    red = "Ночная смена принимает больше всего возвратов, поэтому именно ночью на складе появляется большая часть расхождений."
    doc = tmp_path / "doc.md"
    doc.write_text(red + "\n\nВторой абзац остаётся обычным.\n", encoding="utf-8")
    heat = tmp_path / "heat.json"
    heat.write_text(json.dumps({"fragments": [{"level": 3, "text": red}]}, ensure_ascii=False), encoding="utf-8")
    r = cli("validate", str(doc), str(doc), "--marked", str(heat))
    assert r.code == 0
    assert "красный абзац без правки" in r.out


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


def test_antiplagiat_explain(cli: Cli):
    """antiplagiat --explain: цель и вклад признаков у красных фрагментов, в JSON — поле explain"""
    r = cli("antiplagiat", "--explain", str(AI_VAK), env={"COLUMNS": "200"})
    assert r.code == 0
    assert "зелёная зона ниже 40 %" in r.out
    assert re.search(r"^ {2}Цель +\S", r.out, re.MULTILINE)
    assert "Хватит одного" in r.out
    data = json.loads(cli("antiplagiat", "--json", "--explain", str(AI_VAK)).out)
    fragment = data["fragments"][0]
    assert fragment["zone"] == "red"
    assert fragment["explain"]["plan"]
    assert fragment["explain"]["targetProbability"] == 0.4
    assert {"feature", "contribution", "now", "goal", "enough", "target"} <= set(
        fragment["explain"]["contributions"][0]
    )
    plain = json.loads(cli("antiplagiat", "--json", str(AI_VAK)).out)
    assert "explain" not in plain["fragments"][0]
    assert plain["fragments"][0]["zone"] == "red"
    assert plain["typicalProse"] < plain["threshold"] - 0.1


def test_antiplagiat_warns_typical_prose_at_border(cli: Cli, tmp_path: Path):
    """модель, которая ставит ровную прозу у порога, получает предупреждение и в antiplagiat, и в calibrate"""
    border = {"bias": -1.0, "weights": [0.3, 1.9, 0.75, 0.16, 0.15, 1.05, 1.0], "threshold": 0.44}
    config = tmp_path / "aiw-ru.json"
    config.write_text(
        json.dumps({"version": 1, "model": border, "samples": [{"f": [0, 0, 1, 0, 0, 0, 0], "y": 1}]}),
        encoding="utf-8",
    )
    wide = {"COLUMNS": "300"}
    warned = cli("antiplagiat", "--config", str(config), str(HUMAN_VAK), env=wide)
    assert "Модель ставит ровную научную прозу у порога" in warned.out
    assert "правьте до зелёной зоны (ниже 34 %)" in warned.out
    assert "ровную научную прозу" not in cli("antiplagiat", str(HUMAN_VAK), env=wide).out


def test_validate_json(cli: Cli):
    """validate --json: поля в camelCase"""
    data = json.loads(cli("validate", "--json", HUMAN_BLOG, HUMAN_BLOG).out)
    assert data["ok"] is True
    assert data["violations"] == []
    assert data["warnings"] == []
    assert data["swappedNumbers"] == []
    assert data["issuesBefore"] == data["issuesAfter"]


def pair(tmp_path: Path, before: str, after: str) -> tuple[str, str]:
    a, b = tmp_path / "до.md", tmp_path / "после.md"
    a.write_text(before, encoding="utf-8")
    b.write_text(after, encoding="utf-8")
    return str(a), str(b)


def test_validate_fact_violations_exit_1(cli: Cli, tmp_path: Path):
    """validate: снятая оговорка и новое имя — нарушения, код 1"""
    r = cli("validate", *pair(tmp_path, "Метод обычно помогает.", "Метод помогает на PyTorch."))
    assert r.code == 1
    for s in ("оговорка", "пропала «обычно»", "имя", "появилось: PyTorch"):
        assert s in r.out


def test_validate_warnings_keep_exit_0(cli: Cli, tmp_path: Path):
    """validate: предупреждения идут отдельной таблицей и код выхода не меняют"""
    files = pair(tmp_path, "Мы сделали сервис мониторинга.", "Мы впервые сделали сервис мониторинга.")
    r = cli("validate", *files)
    assert r.code == 0
    assert "автоматические проверки пройдены" in r.out
    assert "Смысл целиком он не проверяет" in " ".join(r.out.split())
    assert re.search(r"Проверить\s+1", r.out)
    assert "утверждение" in r.out
    assert "появилось: впервые" in r.out
    data = json.loads(cli("validate", "--json", *files).out)
    assert data["ok"] is True
    assert data["warnings"] == [{"kind": "утверждение", "detail": "появилось: впервые"}]


def test_validate_swapped_numbers_warning(cli: Cli, tmp_path: Path):
    """validate: числа, поменявшиеся местами, — предупреждение в таблице и поле swappedNumbers"""
    files = pair(tmp_path, "Кира собрала 12 400 подписей, Лев 8 900.", "Кира собрала 8 900 подписей, Лев 12 400.")
    r = cli("validate", *files)
    assert r.code == 0
    assert "перестановка" in r.out
    assert "поменялись местами" in " ".join(r.out.split())
    data = json.loads(cli("validate", "--json", *files).out)
    assert data["ok"] is True
    assert data["swappedNumbers"][0]["numbers"] == ["12 400", "8 900"]
    assert [warning["kind"] for warning in data["warnings"]] == ["перестановка"]


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
    assert "Ровная проза" in cal.out
    assert config.exists()
    saved = json.loads(config.read_text(encoding="utf-8"))
    assert saved["version"] == 1
    assert len([s for s in saved["samples"] if s["y"] == 1]) == 1
    assert len(saved["samples"]) > 1

    before = config.read_bytes()
    again = cli("calibrate", "--doc", str(doc), "--marked", str(marked), cwd=tmp_path)
    assert again.code == 0
    assert "уже в калибровке" in again.out
    assert "не изменён" in again.out
    assert config.read_bytes() == before, "повтор того же отчёта не меняет файл"

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


@pytest.mark.parametrize("output", [[], ["--json"], ["--jsonl"]], ids=["таблица", "json", "jsonl"])
@pytest.mark.parametrize("source", ["--config", "$AIW_RU_CONFIG"])
def test_antiplagiat_named_config_missing_exit_2(cli: Cli, tmp_path: Path, source: str, output: list[str]):
    """antiplagiat: файл калибровки из --config или $AIW_RU_CONFIG не найден — код 2, а не оценка без калибровки"""
    missing = tmp_path / "нет" / "aiw-ru.json"
    text = AI_VAK.read_text(encoding="utf-8")
    stdin = json.dumps({"text": text}) if output == ["--jsonl"] else text
    if source == "--config":
        r = cli("antiplagiat", *output, "--config", str(missing), stdin=stdin)
    else:
        r = cli("antiplagiat", *output, stdin=stdin, env={"AIW_RU_CONFIG": str(missing)})
    assert r.code == 2
    assert r.out == ""
    assert f"нет файла калибровки {missing} из {source}" in r.err
    assert "calibrate" in r.err
    assert not missing.exists()


def test_antiplagiat_without_named_config_uses_default(cli: Cli):
    """antiplagiat без --config и $AIW_RU_CONFIG и без файлов по умолчанию: модель по умолчанию, код 0"""
    r = cli("antiplagiat", "--json", str(AI_VAK))
    assert r.code == 0
    assert json.loads(r.out)["model"] == "default"


def test_calibrate_creates_missing_named_config(cli: Cli, tmp_path: Path):
    """calibrate создаёт файл, которого ещё нет, по --config: это не ошибка, как у antiplagiat"""
    doc = tmp_path / "doc.md"
    doc.write_text(AI_VAK.read_text(encoding="utf-8"), encoding="utf-8")
    config = tmp_path / "новая.json"
    assert cli("calibrate", "--doc", str(doc), "--share", "10", "--config", str(config)).code == 0
    assert json.loads(config.read_text(encoding="utf-8"))["documents"]


def test_calibrate_writes_to_config_from_env(cli: Cli, tmp_path: Path):
    """calibrate пишет в $AIW_RU_CONFIG — в тот же файл, откуда читает antiplagiat"""
    doc = tmp_path / "doc.md"
    doc.write_text(AI_VAK.read_text(encoding="utf-8"), encoding="utf-8")
    config = tmp_path / "cal" / "aiw-ru.json"
    config.parent.mkdir()
    r = cli("calibrate", "--doc", str(doc), "--share", "10", env={"AIW_RU_CONFIG": str(config)})
    assert r.code == 0
    assert config.exists(), "калибровка должна лежать в файле, который читает antiplagiat"
    assert not (Path.cwd() / ".aiw-ru.json").exists()
    report = json.loads(cli("antiplagiat", "--json", str(doc), env={"AIW_RU_CONFIG": str(config)}).out)
    assert report["model"] == "calibrated"


def test_calibrate_accumulates_in_user_config(cli: Cli, tmp_path: Path):
    """calibrate дописывает образцы в ~/.config/aiw-ru.json, а не в новый файл в текущей папке"""
    user_config = Path.home() / ".config" / "aiw-ru.json"
    user_config.parent.mkdir(parents=True)
    doc = tmp_path / "doc.md"
    marked = tmp_path / "marked.txt"
    ai = AI_VAK.read_text(encoding="utf-8")
    human = HUMAN_VAK.read_text(encoding="utf-8")
    doc.write_text(f"{ai}\n\n{human}", encoding="utf-8")
    marked.write_text(ai.split("\n\n")[1], encoding="utf-8")

    assert cli("calibrate", "--doc", str(doc), "--marked", str(marked), "--config", str(user_config)).code == 0
    first = json.loads(user_config.read_text(encoding="utf-8"))

    # Отчёт по правленой версии того же документа: образцы добавляются рядом с прежними.
    doc.write_text(
        f"{ai}\n\n{human}\n\nПосле проверки автор дописал короткий вывод о складском учёте.", encoding="utf-8"
    )
    assert cli("calibrate", "--doc", str(doc), "--marked", str(marked)).code == 0
    assert not (Path.cwd() / ".aiw-ru.json").exists(), "калибровка ушла в файл, который antiplagiat не читает"
    second = json.loads(user_config.read_text(encoding="utf-8"))
    assert second["samples"][: len(first["samples"])] == first["samples"]
    assert len(second["samples"]) > len(first["samples"])
    assert len(second["reports"]) == 2


def test_broken_calibration_file_clear_error(cli: Cli, tmp_path: Path):
    """битый файл калибровки — понятная ошибка"""
    (tmp_path / ".aiw-ru.json").write_text("{не json", encoding="utf-8")
    r = cli("antiplagiat", str(AI_VAK), cwd=tmp_path)
    assert r.code == 2
    assert "калибровку" in r.err


@pytest.fixture
def report_doc(tmp_path: Path) -> Path:
    """Документ для калибровки: ИИ-текст и человеческий в жанре ВАК подряд."""
    doc = tmp_path / "doc.md"
    doc.write_text(f"{AI_VAK.read_text(encoding='utf-8')}\n\n{HUMAN_VAK.read_text(encoding='utf-8')}", encoding="utf-8")
    return doc


BROKEN_HEATMAPS = [
    pytest.param('[{"level": "bad", "text": "Уровень словом"}]', "level", id="level не число"),
    pytest.param('{"fragements": []}', "fragments", id="опечатка в fragments"),
    pytest.param('[{"level": 3, "pages": [1]}]', "нет поля text", id="фрагмент без текста"),
]


@pytest.mark.parametrize(("heat", "reason"), BROKEN_HEATMAPS)
def test_calibrate_rejects_broken_heatmap(cli: Cli, tmp_path: Path, report_doc: Path, heat: str, reason: str):
    """calibrate: неверная тепловая карта — код 2 с причиной, файл калибровки не создаётся"""
    marked = tmp_path / "heat.json"
    marked.write_text(heat, encoding="utf-8")
    config = tmp_path / "cal.json"
    r = cli("calibrate", "--doc", str(report_doc), "--marked", str(marked), "--config", str(config))
    assert r.code == 2
    assert f"не удалось прочитать тепловую карту {marked}" in r.err
    assert reason in r.err
    assert not config.exists()


def test_calibrate_broken_heatmap_keeps_existing_calibration(cli: Cli, tmp_path: Path, report_doc: Path):
    """неверная карта во втором отчёте того же вызова не трогает накопленную калибровку"""
    good = tmp_path / "heat.json"
    good.write_text(json.dumps([{"level": 3, "text": AI_VAK.read_text(encoding="utf-8").split("\n\n")[1]}]))
    broken = tmp_path / "broken.json"
    broken.write_text('[{"level": 7, "text": "Такого уровня нет"}]', encoding="utf-8")
    config = tmp_path / "cal.json"
    assert cli("calibrate", "--doc", str(report_doc), "--marked", str(good), "--config", str(config)).code == 0
    before = config.read_bytes()
    pair_args = ["--doc", str(report_doc), "--marked", str(good), "--doc", str(report_doc), "--marked", str(broken)]
    r = cli("calibrate", *pair_args, "--config", str(config))
    assert r.code == 2
    assert "level" in r.err
    assert config.read_bytes() == before


@pytest.mark.parametrize(
    "heat", ["[]", '{"share": 0, "fragments": []}'], ids=["пустой список", "объект без фрагментов"]
)
def test_calibrate_empty_heatmap_is_human_report(cli: Cli, tmp_path: Path, report_doc: Path, heat: str):
    """пустая тепловая карта — отчёт без подсветки: все фрагменты человеческие, код 0, без предупреждения"""
    marked = tmp_path / "heat.json"
    marked.write_text(heat, encoding="utf-8")
    config = tmp_path / "cal.json"
    r = cli(
        "calibrate", "--system", "domate", "--doc", str(report_doc), "--marked", str(marked), "--config", str(config)
    )
    assert r.code == 0
    assert "не найден в документе" not in r.out
    samples = json.loads(config.read_text(encoding="utf-8"))["systems"]["domate"]["samples"]
    assert samples
    assert {sample["y"] for sample in samples} == {0}


@pytest.mark.parametrize("command", ["validate", "judge score"])
def test_broken_heatmap_rejected_by_validate_and_judge(cli: Cli, tmp_path: Path, report_doc: Path, command: str):
    """validate и judge score с неверной тепловой картой в --marked — код 2, а не разметка из текста карты"""
    marked = tmp_path / "heat.json"
    marked.write_text('[{"text": "Фрагмент без уровня"}]', encoding="utf-8")
    answers = tmp_path / "answers.txt"
    answers.write_text("1: low", encoding="utf-8")
    args = (
        ["validate", str(report_doc), str(report_doc)]
        if command == "validate"
        else ["judge", "score", str(report_doc), "--answers", str(answers)]
    )
    r = cli(*args, "--marked", str(marked))
    assert r.code == 2
    assert "нет поля level" in r.err


def test_calibrate_truncated_heatmap_read_as_text(cli: Cli, tmp_path: Path, report_doc: Path):
    """неразобранный JSON (оборванный heat.json) читается как текст, как раньше: код 0 и предупреждение"""
    marked = tmp_path / "heat.json"
    marked.write_text(
        '{"share": 0.4, "fragments": [{"level": 3, "pages": [1], "text": "Карта оборвана', encoding="utf-8"
    )
    r = cli("calibrate", "--doc", str(report_doc), "--marked", str(marked), "--config", str(tmp_path / "cal.json"))
    assert r.code == 0
    assert "не найден в документе" in r.out


# ─── цикл калибровки по личным отчётам ──────────────────────────────────


def old_calibration_file() -> dict[str, Any]:
    """Файл прежней версии: «Антиплагиат» в полях верхнего уровня, «Думейт» в systems."""
    model = {"bias": -2.0, "weights": [0.3, 1.9, 0.75, 0.16, 0.15, 1.05, 1.0], "threshold": 0.5}
    return {
        "version": 1,
        "model": model,
        "samples": [{"f": [0.1] * 7, "y": 1}, {"f": [0.0] * 7, "y": 0}],
        "documents": [],
        "systems": {"domate": {"model": model, "samples": [{"f": [0.2] * 7, "y": 1}], "documents": []}},
    }


def test_personal_reports_calibration_cycle(cli: Cli, tmp_path: Path):
    """весь цикл по личным отчётам: старый файл калибровки, heat.json «Думейта», текстовая разметка
    «Антиплагиата», antiplagiat с калибровкой; ошибочные и читающие вызовы не меняют файл ни на байт"""
    ai = AI_VAK.read_text(encoding="utf-8")
    paragraphs = ai.split("\n\n")
    doc = tmp_path / "glava1.md"
    doc.write_text(f"{HUMAN_VAK.read_text(encoding='utf-8')}\n\n{ai}", encoding="utf-8")
    # Скилл советует калибровать в ~/.config/aiw-ru.json: antiplagiat читает этот файл и без --config.
    config = Path.home() / ".config" / "aiw-ru.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps(old_calibration_file()), encoding="utf-8")
    old = old_calibration_file()

    # Тепловая карта в том виде, в каком её отдаёт скрипт из skills/antiplagiat/references/domate.md.
    heat = tmp_path / "heat.json"
    fragments = [
        {"level": 3, "pages": [1], "text": paragraphs[1]},
        {"level": 2, "pages": [1, 2], "text": paragraphs[2]},
    ]
    heat.write_text(json.dumps({"share": 0.38, "fragments": fragments}, ensure_ascii=False), encoding="utf-8")
    domate = cli("calibrate", "--system", "domate", "--doc", str(doc), "--marked", str(heat), "--config", str(config))
    assert domate.code == 0, domate.err
    saved = json.loads(config.read_text(encoding="utf-8"))
    assert saved["samples"] == old["samples"]
    assert saved["systems"]["domate"]["samples"][:1] == old["systems"]["domate"]["samples"]
    assert [sample["y"] for sample in saved["systems"]["domate"]["samples"][1:]] == [0, 0, 0, 1, 0]

    # Сильные признаки из отчёта «Антиплагиата», выписанные текстом через пустую строку.
    strong = tmp_path / "glava1-ai.txt"
    strong.write_text(f"{paragraphs[1]}\n\n{paragraphs[2]}\n", encoding="utf-8")
    antiplagiat_cal = cli("calibrate", "--doc", str(doc), "--marked", str(strong), "--config", str(config))
    assert antiplagiat_cal.code == 0, antiplagiat_cal.err
    assert "не найден в документе" not in antiplagiat_cal.out
    saved = json.loads(config.read_text(encoding="utf-8"))
    assert saved["samples"][:2] == old["samples"]
    assert sum(sample["y"] for sample in saved["samples"][2:]) >= 1
    assert len(saved["systems"]["domate"]["samples"]) == 6
    # Прежние поля верхнего уровня на месте: старые версии aiw-ru читают файл как раньше.
    assert {"version", "model", "samples", "documents", "systems"} <= set(saved)

    calibrated = config.read_bytes()
    answers = tmp_path / "r1.txt"
    answers.write_text("1: none\n2: none\n3: none\n4: high\n5: medium", encoding="utf-8")
    reading = [
        ["antiplagiat", "--json", str(doc)],
        ["antiplagiat", "--json", "--system", "domate", str(doc)],
        ["validate", str(doc), str(doc), "--marked", str(heat)],
        ["validate", str(doc), str(doc), "--marked", str(strong)],
        ["judge", "score", str(doc), "--answers", str(answers), "--marked", str(heat)],
        ["judge", "score", str(doc), "--answers", str(answers), "--marked", str(strong)],
    ]
    for args in reading:
        r = cli(*args)
        assert r.code != 2, (args, r.err)
        assert config.read_bytes() == calibrated, args
    assert json.loads(cli("antiplagiat", "--json", str(doc)).out)["model"] == "calibrated"
    assert json.loads(cli("antiplagiat", "--json", "--system", "domate", str(doc)).out)["model"] == "calibrated"

    broken = tmp_path / "broken.json"
    broken.write_text('{"share": 0.4, "fragments": [{"level": "высокий", "text": "Уровень словом"}]}', encoding="utf-8")
    failing = [
        ["calibrate", "--system", "domate", "--doc", str(doc), "--marked", str(broken), "--config", str(config)],
        ["calibrate", "--doc", str(doc), "--marked", str(strong), "--doc", str(doc), "--marked", str(broken)],
        ["calibrate", "--doc", str(doc), "--marked", str(tmp_path / "нет.txt"), "--config", str(config)],
        ["calibrate", "--doc", str(tmp_path / "нет.md"), "--marked", str(strong), "--config", str(config)],
        ["calibrate", "--doc", str(doc), "--share", "150", "--config", str(config)],
        ["antiplagiat", "--config", str(tmp_path / "нет.json"), str(doc)],
        ["validate", str(doc), str(doc), "--marked", str(broken)],
        ["judge", "score", str(doc), "--answers", str(answers), "--marked", str(broken)],
    ]
    for args in failing:
        r = cli(*args)
        assert r.code == 2, (args, r.out)
        assert config.read_bytes() == calibrated, args


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
    assert p.wait(timeout=60) == 0, err
    assert "Traceback" not in err
    assert "BrokenPipeError" not in err


# ── skill ──


def test_skill_lists_skills(cli: Cli):
    """без имени — список скиллов, основной первым"""
    r = cli("skill")
    assert r.code == 0
    assert r.out.index("avoid-ai-writing-russian") < r.out.index("antiplagiat")
    assert "aiw-ru skill ИМЯ" in r.out
    data = json.loads(cli("skill", "--json").out)
    assert [s["name"] for s in data] == ["avoid-ai-writing-russian", "antiplagiat"]
    assert data[0]["files"] == [
        f"references/{n}.md"
        for n in (
            "chat",
            "environments",
            "judge",
            "models",
            "profiles",
            "review",
            "rhetoric",
            "sentences",
            "structure",
            "typography",
            "vocabulary",
        )
    ]
    assert data[1]["description"].startswith("Подготовка русского текста")
    assert data[1]["files"] == ["references/antiplagiat.md", "references/domate.md"]


def test_skill_files_in_hidden_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """пакет из кэша uv лежит под ~/.cache: файлы скилла видны, скрытые файлы внутри скилла нет"""
    base = tmp_path / ".cache" / "skills"
    shutil.copytree(skills.SOURCE, base)
    (base / skills.MAIN / ".DS_Store").write_text("", encoding="utf-8")
    monkeypatch.setattr(skills, "PACKAGED", base)
    files = skills.all_skills()[0].files
    assert "references/review.md" in files
    assert not any(f.startswith(".") or "/." in f for f in files)


@pytest.mark.parametrize(
    "args",
    [
        ("avoid-ai-writing-russian",),
        ("antiplagiat",),
        ("avoid-ai-writing-russian", "references/vocabulary.md"),
        ("avoid-ai-writing-russian", "references/review.md"),
        ("avoid-ai-writing-russian", "references/judge.md"),
        ("avoid-ai-writing-russian", "references/environments.md"),
        ("antiplagiat", "references/domate.md"),
        ("antiplagiat", "references/antiplagiat.md"),
    ],
)
def test_skill_text_needs_no_repository(cli: Cli, args: tuple[str, ...]):
    """текст скилла без путей в репозитории: детектор через aiw-ru, файлы через aiw-ru skill"""
    r = cli("skill", *args)
    assert r.code == 0 and r.err == ""
    assert "../" not in r.out and "uv run --project" not in r.out
    # Подсказка «aiw-ru из PATH или uv» нужна только при работе из репозитория.
    assert "repo-only" not in r.out and "command -v aiw-ru" not in r.out
    # Справка под-скилла ссылается на его SKILL.md, остальные файлы — на основной скилл.
    owner = "antiplagiat" if args[0] == "antiplagiat" and len(args) > 1 else "avoid-ai-writing-russian"
    assert f"aiw-ru skill {owner}" in r.out


def test_skill_md_keeps_frontmatter(cli: Cli):
    """SKILL.md печатается с шапкой, пояснение идёт после неё"""
    out = cli("skill", "antiplagiat").out
    assert out.startswith("---\nname: antiplagiat\n")
    assert out.index("\n---\n") < out.index("> Скилл выведен командой `aiw-ru skill`")
    assert "aiw-ru antiplagiat <файл> --context vak" in out


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("nope",), "нет скилла nope"),
        (("antiplagiat", "../avoid-ai-writing-russian/SKILL.md"), "нет файла"),
        (("avoid-ai-writing-russian", "../../pyproject.toml"), "нет файла"),
    ],
)
def test_skill_unknown(cli: Cli, args: tuple[str, ...], message: str):
    r = cli("skill", *args)
    assert r.code == 2 and message in r.err


# ─── системы проверки и слепой судья ────────────────────────────────────


def test_calibrate_system_domate_keeps_antiplagiat(cli: Cli, tmp_path: Path):
    """calibrate --system domate пишет калибровку «Думейта» отдельно; тепловая карта в JSON — разметка"""
    doc = tmp_path / "doc.md"
    ai = AI_VAK.read_text(encoding="utf-8")
    doc.write_text(f"{HUMAN_VAK.read_text(encoding='utf-8')}\n\n{ai}", encoding="utf-8")
    heat = tmp_path / "heat.json"
    red, orange = ai.split("\n\n")[1], ai.split("\n\n")[2]
    heat.write_text(json.dumps({"share": 0.4, "fragments": [{"level": 3, "text": red}, {"level": 2, "text": orange}]}))

    cal = cli("calibrate", "--system", "domate", "--doc", str(doc), "--marked", str(heat), cwd=tmp_path)
    assert cal.code == 0
    assert "«Думейт»" in cal.out
    saved = json.loads((tmp_path / ".aiw-ru.json").read_text(encoding="utf-8"))
    assert saved["samples"] == []
    assert [s["y"] for s in saved["systems"]["domate"]["samples"]] == [0, 0, 0, 1, 0]

    domate = json.loads(cli("antiplagiat", "--json", "--system", "domate", str(doc), cwd=tmp_path).out)
    assert domate["model"] == "calibrated" and domate["system"] == "domate"
    plain = cli("antiplagiat", str(doc), cwd=tmp_path)
    assert "по умолчанию, без калибровки" in plain.out
    assert "классификатор «Антиплагиата» закрыт" in plain.out
    assert "--system domate" in cli("antiplagiat", "--system", "domate", str(doc), cwd=tmp_path).out


def test_unknown_system_exit_2(cli: Cli):
    """неизвестная --system — код 2"""
    r = cli("antiplagiat", "--system", "turnitin", str(AI_VAK))
    assert r.code == 2
    assert "domate" in r.err


def test_judge_prints_prompt(cli: Cli):
    """judge печатает задание с пронумерованными фрагментами, --json — ещё и фрагменты"""
    r = cli("judge", str(AI_VAK))
    assert r.code == 0
    assert r.out.startswith("Ты детектор ИИ-текста")
    assert "[P1]\n" in r.out and "[P2]\n" in r.out and "[P3]" not in r.out
    data = json.loads(cli("judge", "--json", str(AI_VAK)).out)
    assert data["prompt"] == r.out
    assert [f["n"] for f in data["fragments"]] == [1, 2]
    assert {"line", "endLine", "words", "text"} <= set(data["fragments"][0])


def test_judge_score_table_and_json(cli: Cli, tmp_path: Path):
    """judge score сводит прогоны и с --marked считает ROC AUC"""
    doc = tmp_path / "doc.md"
    ai = AI_VAK.read_text(encoding="utf-8")
    doc.write_text(f"{HUMAN_VAK.read_text(encoding='utf-8')}\n\n{ai}", encoding="utf-8")
    a, b, marked = tmp_path / "a.txt", tmp_path / "b.txt", tmp_path / "m.txt"
    a.write_text("P1 none\nP2 none\nP3 low\nP4 high\nP5 medium", encoding="utf-8")
    b.write_text("P1 none\nP2 low\nP3 none\nP4 medium\nP5 medium", encoding="utf-8")
    marked.write_text(ai.split("\n\n", 1)[1], encoding="utf-8")
    args = ("judge", "score", str(doc), "--answers", str(a), "--answers", str(b), "--marked", str(marked))
    r = cli(*args)
    assert r.code == 0
    for s in ("Прогонов", "Совпадение прогонов", "ROC AUC по отчёту", "high medium", "Отчёт"):
        assert s in r.out
    data = json.loads(cli(*args, "--json").out)
    assert data["runs"] == 2
    assert data["auc"] == 1
    assert [f["mean"] for f in data["fragments"]] == [0, 0.5, 0.5, 2.5, 2]


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(["judge", "score", str(AI_VAK)], id="score без --answers"),
        pytest.param(["judge", "score", "--answers", "a.txt"], id="score без файла"),
        pytest.param(["judge", str(AI_VAK), "--answers", "a.txt"], id="--answers без score"),
    ],
)
def test_judge_usage_errors_exit_2(cli: Cli, args: list[str]):
    """judge без нужных аргументов — код 2"""
    assert cli(*args).code == 2


# ─── надёжность калибровки ──────────────────────────────────────────────


@pytest.fixture
def heat_pair(tmp_path: Path, report_doc: Path) -> tuple[Path, Path]:
    """Документ и разметка с одним подсвеченным куском."""
    marked = tmp_path / "glava1-ai.txt"
    marked.write_text(AI_VAK.read_text(encoding="utf-8").split("\n\n")[1], encoding="utf-8")
    return report_doc, marked


def test_calibrate_other_markup_needs_replace(cli: Cli, tmp_path: Path, heat_pair: tuple[Path, Path]):
    """другая разметка той же версии документа: код 2 с подсказкой --replace, с --replace — замена"""
    doc, marked = heat_pair
    config = tmp_path / "cal.json"
    assert cli("calibrate", "--doc", str(doc), "--marked", str(marked), "--config", str(config)).code == 0
    before = config.read_bytes()
    other = tmp_path / "other.txt"
    other.write_text(AI_VAK.read_text(encoding="utf-8").split("\n\n")[2], encoding="utf-8")
    refused = cli("calibrate", "--doc", str(doc), "--marked", str(other), "--config", str(config))
    assert refused.code == 2
    assert "--replace" in refused.err
    assert config.read_bytes() == before
    replaced = cli("calibrate", "--replace", "--doc", str(doc), "--marked", str(other), "--config", str(config))
    assert replaced.code == 0
    assert "заменён" in replaced.out
    saved = json.loads(config.read_text(encoding="utf-8"))
    assert len(saved["reports"]) == 1
    assert len(saved["samples"]) == len(json.loads(before)["samples"])


def test_calibrate_report_date_and_system_version(cli: Cli, tmp_path: Path, heat_pair: tuple[Path, Path]):
    """--report-date и --system-version относятся к предыдущему --doc и сохраняются в записи об отчёте"""
    doc, marked = heat_pair
    config = tmp_path / "cal.json"
    args = ["--doc", str(doc), "--marked", str(marked), "--report-date", "2026-11-05", "--system-version", " 5.0 "]
    assert cli("calibrate", *args, "--config", str(config)).code == 0
    (record,) = json.loads(config.read_text(encoding="utf-8"))["reports"]
    assert (record["date"], record["systemVersion"], record["kind"]) == ("2026-11-05", "5.0", "marked")
    other = tmp_path / "glava2.md"
    other.write_text(HUMAN_VAK.read_text(encoding="utf-8"), encoding="utf-8")
    assert cli("calibrate", "--doc", str(other), "--share", "0", "--config", str(config)).code == 0
    first, _ = json.loads(config.read_text(encoding="utf-8"))["reports"]
    assert (first["date"], first["systemVersion"]) == ("2026-11-05", "5.0")


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(["--report-date", "2026-11-05", "--doc", "x.md", "--share", "3"], id="дата до --doc"),
        pytest.param(["--doc", "x.md", "--share", "3", "--report-date", "05.11.2026"], id="дата не ГГГГ-ММ-ДД"),
        pytest.param(["--doc", "x.md", "--share", "3", "--report-date", "2026-02-30"], id="несуществующая дата"),
        pytest.param(["--doc", "x.md", "--share", "3", "--system-version", " "], id="пустая версия"),
    ],
)
def test_calibrate_bad_report_details_exit_2(cli: Cli, args: list[str]):
    """неверные --report-date и --system-version — код 2"""
    assert cli("calibrate", *args).code == 2


def test_calibrate_retries_after_concurrent_write(
    cli: Cli, tmp_path: Path, heat_pair: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
):
    """другой процесс записал файл, пока шёл расчёт: calibrate перечитывает его, и обе записи сохраняются"""
    from aiw_ru import calibration
    from aiw_ru.antiplagiat import empty_calibration_file

    doc, marked = heat_pair
    config = tmp_path / "cal.json"
    other_doc = tmp_path / "glava2.md"
    other_doc.write_text(HUMAN_VAK.read_text(encoding="utf-8"), encoding="utf-8")
    real_import = calibration.import_reports
    calls: list[int] = []

    def import_with_parallel_writer(*args: Any, **kwargs: Any) -> calibration.ImportResult:
        calls.append(1)
        if len(calls) == 1:
            # Параллельный calibrate успевает записать свой отчёт по другому документу.
            parallel = real_import(
                None, [calibration.ReportInput("glava2.md", other_doc.read_text(encoding="utf-8"), share=0)], today="x"
            )
            data = empty_calibration_file().put("antiplagiat", parallel.calibration)
            config.write_bytes(calibration.serialize(data))
        return real_import(*args, **kwargs)

    monkeypatch.setattr(calibration, "import_reports", import_with_parallel_writer)
    r = cli("calibrate", "--doc", str(doc), "--marked", str(marked), "--config", str(config))
    assert r.code == 0, r.err
    assert len(calls) == 2
    saved = json.loads(config.read_text(encoding="utf-8"))
    assert {record["kind"] for record in saved["reports"]} == {"marked", "share"}
    assert len(saved["documents"]) == 1


def test_calibrate_gives_up_on_constant_concurrent_writes(
    cli: Cli, tmp_path: Path, heat_pair: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
):
    """файл меняется на каждой попытке: код 2, а файл остаётся таким, каким его оставил другой процесс"""
    from aiw_ru import DEFAULT_MODEL, calibration

    doc, marked = heat_pair
    config = tmp_path / "cal.json"
    real_import = calibration.import_reports
    writes: list[int] = []

    def import_while_file_changes(*args: Any, **kwargs: Any) -> calibration.ImportResult:
        writes.append(1)
        other = {"version": 1, "model": DEFAULT_MODEL.to_dict(), "samples": [], "documents": [], "note": len(writes)}
        config.write_text(json.dumps(other), encoding="utf-8")
        return real_import(*args, **kwargs)

    monkeypatch.setattr(calibration, "import_reports", import_while_file_changes)
    r = cli("calibrate", "--doc", str(doc), "--marked", str(marked), "--config", str(config))
    assert r.code == 2
    assert "другой процесс" in r.err
    assert json.loads(config.read_text(encoding="utf-8"))["note"] == len(writes) == 3


def test_calibrate_unwritable_path_exit_2(cli: Cli, tmp_path: Path, heat_pair: tuple[Path, Path]):
    """папки из --config нет: код 2 с понятной причиной, без трейсбека"""
    doc, marked = heat_pair
    config = tmp_path / "нет-папки" / "cal.json"
    r = cli("calibrate", "--doc", str(doc), "--marked", str(marked), "--config", str(config))
    assert r.code == 2
    assert f"не удалось записать калибровку {config}" in r.err
    assert "Traceback" not in r.err


def test_calibrate_transfer_line(cli: Cli, tmp_path: Path, heat_pair: tuple[Path, Path]):
    """строка «Перенос»: один документ или его правленая версия — не проверен, второй документ даёт проверку"""
    doc, marked = heat_pair
    config = tmp_path / "cal.json"
    wide = {"COLUMNS": "200"}
    first = cli("calibrate", "--doc", str(doc), "--marked", str(marked), "--config", str(config), env=wide)
    assert re.search(r"Перенос\s+не проверен: нужна разметка отчётов хотя бы по двум документам", first.out)
    edited = tmp_path / "glava1-v2.md"
    edited.write_text(doc.read_text(encoding="utf-8") + "\n\nАвтор дописал короткий итог о сменах склада.", "utf-8")
    again = cli("calibrate", "--doc", str(edited), "--marked", str(marked), "--config", str(config), env=wide)
    assert again.code == 0
    assert re.search(r"Перенос\s+не проверен: нужна разметка отчётов хотя бы по двум документам", again.out)
    second_doc = tmp_path / "glava2.md"
    ai = Path(AI_BLOG).read_text(encoding="utf-8")
    second_doc.write_text(f"{ai}\n\n{Path(HUMAN_BLOG).read_text(encoding='utf-8')}", encoding="utf-8")
    second_marked = tmp_path / "glava2-ai.txt"
    second_marked.write_text(ai.split("\n\n")[1], encoding="utf-8")
    second = cli(
        "calibrate", "--doc", str(second_doc), "--marked", str(second_marked), "--config", str(config), env=wide
    )
    assert second.code == 0
    assert re.search(r"Перенос\s+.+ на документах, отложенных целиком \(документов: 2\)", second.out)


def test_calibrate_repeat_skips_transfer_check(
    cli: Cli, tmp_path: Path, heat_pair: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
):
    """повтор тех же отчётов ничего не меняет и не пересчитывает перенос: на больших калибровках это долго"""
    from aiw_ru import calibration

    doc, marked = heat_pair
    config = tmp_path / "cal.json"
    args = ["calibrate", "--doc", str(doc), "--marked", str(marked), "--config", str(config)]
    assert cli(*args).code == 0

    def no_transfer(samples: list[object]) -> None:
        raise AssertionError("перенос пересчитан без изменений")

    monkeypatch.setattr(calibration, "transfer_check", no_transfer)
    repeat = cli(*args, env={"COLUMNS": "200"})
    assert repeat.code == 0
    assert re.search(r"Перенос\s+не пересчитан: калибровка не изменилась", repeat.out)


@pytest.mark.parametrize(
    "model",
    [
        pytest.param({"bias": -1.0, "weights": [1.0] * 7, "threshold": 1.5}, id="порог больше 1"),
        pytest.param({"bias": -1.0, "weights": [1.0] * 3, "threshold": 0.5}, id="три веса"),
    ],
)
def test_antiplagiat_rejects_broken_model_numbers(cli: Cli, tmp_path: Path, model: dict[str, Any]):
    """калибровка с порогом вне (0, 1) или чужим числом весов — код 2 с полем, а не расчёт"""
    config = tmp_path / "cal.json"
    config.write_text(json.dumps({"version": 1, "model": model, "samples": [{"f": [0] * 7, "y": 1}]}), encoding="utf-8")
    r = cli("antiplagiat", "--config", str(config), str(AI_VAK))
    assert r.code == 2
    assert "model." in r.err


# ─── применимость параметров ────────────────────────────────────────────

# Значение для параметра с аргументом: проверяется применимость, а не само значение.
FLAG_VALUES = {
    "--context": "vak",
    "--min": "P1",
    "--fail-above": "50",
    "--config": "cal.json",
    "--system": "domate",
    "--model": "lightgbm",
    "--max-fragments": "2",
    "--answers": "r1.txt",
    "--marked": "m.txt",
    "--doc": "d.md",
    "--share": "10",
    "--report-date": "2026-11-05",
    "--system-version": "5.0",
}
# Параметры, которые относятся к предыдущему --doc у calibrate.
AFTER_DOC = ("--marked", "--share", "--report-date", "--system-version")


def option_argv(command: str, flag: str) -> list[str]:
    head = ["--doc", "d.md"] if command == "calibrate" and flag in AFTER_DOC else []
    value = [FLAG_VALUES[flag]] if flag in FLAG_VALUES else []
    return [command, *head, flag, *value]


@pytest.mark.parametrize("command", cli_module.COMMANDS)
@pytest.mark.parametrize("option", [o for o in cli_module.OPTIONS if o.commands], ids=lambda option: option.flags[-1])
def test_option_table_decides_applicability(command: str, option: cli_module.Option):
    """таблица параметров решает, к каким командам относится параметр; чужой параметр — ошибка, а не тишина"""
    argv = option_argv(command, option.flags[-1])
    if command in option.commands:
        cli_module.parse(argv)
    else:
        with pytest.raises(cli_module.UsageError, match="не относится к команде"):
            cli_module.parse(argv)


def test_every_option_in_help():
    """каждый параметр таблицы есть в справке вместе с командами, к которым относится"""
    for option in cli_module.OPTIONS:
        for flag in option.flags:
            assert re.search(rf"^ {{2}}{re.escape(flag)}\b", USAGE, re.MULTILINE) or f", {flag}" in USAGE, flag
        if option.commands and not option.scope:
            assert f"{', '.join(option.commands)}:" in USAGE, option.flags


@pytest.mark.parametrize(
    ("args", "message"),
    [
        pytest.param(["scan", "--json", "--jsonl"], "--json и --jsonl", id="json и jsonl"),
        pytest.param(["scan", "--model", "lightgbm", "--no-model"], "--model и --no-model", id="model и no-model"),
        pytest.param(["classify", "--all", "--model", "lightgbm"], "--all и --model", id="all и model"),
        pytest.param(
            ["classify", "--no-fragments", "--max-fragments", "2"], "--no-fragments и --max-fragments", id="фрагменты"
        ),
        pytest.param(
            ["scan", "--context", "vak", "--context", "blog"], "--context указан дважды", id="контекст дважды"
        ),
        pytest.param(
            ["antiplagiat", "--config", "a.json", "--config", "b.json"], "--config указан дважды", id="config дважды"
        ),
        pytest.param(
            ["validate", "a", "b", "--marked", "m", "--marked", "n"], "--marked указан дважды", id="marked дважды"
        ),
        pytest.param(
            ["calibrate", "--doc", "d", "--share", "3", "--report-date", "2026-11-01", "--report-date", "2026-11-02"],
            "--report-date указан дважды для d",
            id="дата отчёта дважды",
        ),
        pytest.param(
            ["calibrate", "--doc", "d", "--share", "3", "--system-version", "5", "--system-version", "6"],
            "--system-version указан дважды для d",
            id="версия системы дважды",
        ),
        pytest.param(["scan", "--jsonl", AI_BLOG, HUMAN_BLOG], "--jsonl читает один поток", id="jsonl и два файла"),
        pytest.param(["scan", "--fail-above", "inf"], "--fail-above ждёт конечное число", id="fail-above inf"),
        pytest.param(["scan", "--fail-above", "-inf"], "--fail-above ждёт конечное число", id="fail-above -inf"),
        pytest.param(["scan", "--answers", "not-a-file"], "--answers не относится к команде scan", id="answers у scan"),
        pytest.param(
            ["calibrate", "--context", "vak", "--doc", "d", "--share", "3"], "не относится", id="context у calibrate"
        ),
    ],
)
def test_conflicting_or_foreign_options_exit_2(cli: Cli, args: list[str], message: str):
    """конфликтующие, повторённые и чужие параметры — код 2 с причиной, а не молчаливый выбор"""
    r = cli(*args)
    assert r.code == 2
    assert message in r.err


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            ["calibrate", "--doc", "a.md", "--marked", "a-ai.txt", "--doc", "b.md", "--share", "12"],
            id="пары calibrate",
        ),
        pytest.param(
            ["judge", "score", "d.md", "--answers", "r1.txt", "--answers", "r2.txt", "--marked", "h.json"],
            id="два прогона судьи",
        ),
        pytest.param(["scan", "--json", "--json", "a.md"], id="повтор флага без значения"),
        pytest.param(["classify", "--all", "--json", "a.md"], id="classify --all"),
    ],
)
def test_documented_combinations_still_parse(args: list[str]):
    """задокументированные сочетания разбираются как раньше"""
    cli_module.parse(args)


def test_help_wins_over_option_checks(cli: Cli):
    """-h показывает справку даже рядом с чужими параметрами"""
    r = cli("calibrate", "--json", "--context", "vak", "--help")
    assert r.code == 0
    assert r.out == USAGE + "\n"


# ─── потоковый JSONL ────────────────────────────────────────────────────


def test_jsonl_carries_id(cli: Cli):
    """поле id входной строки переносится в результат первым полем; без id поля нет"""
    stream = "\n".join(
        [json.dumps({"id": "doc-7", "text": "Давайте разберёмся."}), json.dumps({"text": "Чайник сломался."})]
    )
    for command in ("scan", "antiplagiat"):
        first, second = rows(cli(command, "--jsonl", stdin=stream).out)
        assert next(iter(first)) == "id"
        assert first["id"] == "doc-7"
        assert "id" not in second


def test_jsonl_keeps_results_before_bad_line(cli: Cli):
    """ошибка в строке: результаты до неё уже напечатаны, причина в stderr, код 2"""
    stream = f'{json.dumps({"id": 1, "text": "Раз."})}\n{json.dumps({"id": 2, "text": "Два."})}\n{{"body": 1}}\n'
    r = cli("scan", "--jsonl", stdin=stream)
    assert r.code == 2
    assert [row["id"] for row in rows(r.out)] == [1, 2]
    assert "строка 3: нет строкового поля text" in r.err


def test_jsonl_prints_each_document_before_reading_next(monkeypatch: pytest.MonkeyPatch):
    """JSONL читает вход по строке: результат первой строки напечатан раньше, чем прочитана вторая"""
    printed = io.StringIO()
    lines = [json.dumps({"id": index, "text": "Чайник сломался."}).encode() + b"\n" for index in range(3)]
    seen_before_read: list[int] = []

    class OneLinePerRead(io.RawIOBase):
        def readable(self) -> bool:
            return True

        def readinto(self, buffer: Any) -> int:
            if not lines:
                return 0
            seen_before_read.append(printed.getvalue().count("\n"))
            line = lines.pop(0)
            buffer[: len(line)] = line
            return len(line)

    monkeypatch.setenv("AIW_RU_CONFIG", "")
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BufferedReader(OneLinePerRead()), encoding="utf-8"))
    monkeypatch.setattr(sys, "stdout", printed)
    assert main(["scan", "--jsonl", "--no-model"]) == 0
    assert seen_before_read == [0, 1, 2]
    assert printed.getvalue().count("\n") == 3


def test_jsonl_bad_encoding_after_results(cli: Cli, tmp_path: Path):
    """неверная кодировка в середине потока: напечатанное остаётся, код 2"""
    source = tmp_path / "batch.jsonl"
    source.write_bytes(json.dumps({"id": 1, "text": "Раз."}).encode() + b"\n" + b'{"text": "\xff"}\n')
    r = cli("scan", "--jsonl", str(source))
    assert r.code == 2
    assert [row["id"] for row in rows(r.out)] == [1]
    assert "не удалось прочитать" in r.err


def test_jsonl_closed_reader_is_not_an_error():
    """читатель закрыл канал посреди JSONL: код 0 и без трассировки"""
    p = subprocess.Popen(
        [sys.executable, "-m", "aiw_ru", "scan", "--jsonl", "--no-model"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=clean_env(),
    )
    stdin = p.stdin
    assert stdin is not None and p.stdout is not None and p.stderr is not None
    line = json.dumps({"text": "Давайте разберёмся, это играет ключевую роль."}) + "\n"

    def feed() -> None:
        # Вход пишется в отдельном потоке: канал на Windows вмещает около 4 КБ, и запись из основного
        # потока ждала бы процесс, который сам ждёт, пока прочитают его вывод.
        try:
            stdin.write(line.encode() * 200)
        except OSError:
            pass
        finally:
            with contextlib.suppress(OSError):
                stdin.close()

    writer = threading.Thread(target=feed, daemon=True)
    writer.start()
    p.stdout.readline()
    p.stdout.close()
    err = p.stderr.read().decode("utf-8")
    assert p.wait(timeout=60) == 0, err
    writer.join(timeout=60)
    assert "Traceback" not in err


# ─── JSON-контракт ──────────────────────────────────────────────────────

NUMBER = (int, float)
# Поля, которые читают скиллы, evals и чужие скрипты. Новые поля добавлять можно, эти — не трогать.
CONTRACT: dict[str, dict[str, Any]] = {
    "scan": {
        "file": str,
        "score": int,
        "label": str,
        "issues": list,
        "suspicious": bool,
        "light": dict,
        "stats": dict,
        "catalog": list,
        "meta": dict,
    },
    "antiplagiat": {
        "aiShare": NUMBER,
        "fragments": list,
        "suspicious": bool,
        "suspiciousReasons": list,
        "light": dict,
        "model": str,
        "threshold": NUMBER,
        "system": str,
        "typicalProse": NUMBER,
        "meta": dict,
    },
    "validate": {
        "ok": bool,
        "violations": list,
        "warnings": list,
        "issuesBefore": int,
        "issuesAfter": int,
        "meta": dict,
    },
}
ISSUE_CONTRACT = {
    "type": str,
    "rule": str,
    "severity": str,
    "text": str,
    "index": int,
    "line": int,
    "column": int,
    "hint": str,
}
FRAGMENT_CONTRACT = {
    "start": int,
    "end": int,
    "line": int,
    "endLine": int,
    "words": int,
    "probability": NUMBER,
    "ai": bool,
    "zone": str,
    "features": list,
    "reasons": list,
    "preview": str,
}


def check_contract(data: dict[str, Any], contract: dict[str, Any]) -> None:
    for key, kind in contract.items():
        assert key in data, key
        assert isinstance(data[key], kind), (key, data[key])


@pytest.mark.parametrize(
    ("command", "args"),
    [
        pytest.param("scan", ["scan", "--json", AI_BLOG], id="scan"),
        pytest.param("antiplagiat", ["antiplagiat", "--json", str(AI_VAK)], id="antiplagiat"),
        pytest.param("validate", ["validate", "--json", HUMAN_BLOG, HUMAN_BLOG], id="validate"),
    ],
)
def test_json_contract(cli: Cli, command: str, args: list[str]):
    """JSON-отчёты сохраняют прежние поля и типы и несут meta: схема, версия aiw-ru, калибровка"""
    data = json.loads(cli(*args).out)
    check_contract(data, CONTRACT[command])
    assert data["meta"] == {"schema": cli_module.SCHEMA_VERSION, "aiwRu": __version__, "calibration": None}
    if command == "scan":
        check_contract(data["issues"][0], ISSUE_CONTRACT)
    if command == "antiplagiat":
        check_contract(data["fragments"][0], FRAGMENT_CONTRACT)


def test_jsonl_rows_follow_contract(cli: Cli):
    """строки JSONL — те же отчёты, что и в --json, с meta"""
    stream = json.dumps({"text": Path(AI_BLOG).read_text(encoding="utf-8")})
    (scan,) = rows(cli("scan", "--jsonl", stdin=stream).out)
    check_contract(scan, {key: kind for key, kind in CONTRACT["scan"].items() if key != "file"})
    (report,) = rows(cli("antiplagiat", "--jsonl", stdin=stream).out)
    check_contract(report, CONTRACT["antiplagiat"])


def test_meta_names_calibration_fingerprint(cli: Cli, tmp_path: Path):
    """antiplagiat с калибровкой: в meta отпечаток калибровки, другой файл — другой отпечаток"""
    doc = tmp_path / "doc.md"
    doc.write_text(AI_VAK.read_text(encoding="utf-8"), encoding="utf-8")
    config = tmp_path / "cal.json"
    assert cli("calibrate", "--doc", str(doc), "--share", "10", "--config", str(config)).code == 0
    first = json.loads(cli("antiplagiat", "--json", "--config", str(config), str(doc)).out)["meta"]["calibration"]
    assert re.fullmatch(r"[0-9a-f]{16}", first)
    assert cli("calibrate", "--replace", "--doc", str(doc), "--share", "40", "--config", str(config)).code == 0
    second = json.loads(cli("antiplagiat", "--json", "--config", str(config), str(doc)).out)["meta"]["calibration"]
    assert second != first


# ─── что за число: эвристика, подгонка или классификатор ────────────────


def test_antiplagiat_score_kind(cli: Cli, tmp_path: Path):
    """antiplagiat говорит, чья это оценка: эвристика с весами вручную или модель, подогнанная под отчёты"""
    plain = json.loads(cli("antiplagiat", "--json", str(AI_VAK)).out)
    assert (plain["model"], plain["scoreKind"]) == ("default", "heuristic")
    text = " ".join(cli("antiplagiat", str(AI_VAK), env={"COLUMNS": "200"}).out.split())
    assert "по умолчанию, без калибровки: веса подобраны вручную" in text
    doc = tmp_path / "doc.md"
    doc.write_text(AI_VAK.read_text(encoding="utf-8"), encoding="utf-8")
    config = tmp_path / "cal.json"
    assert cli("calibrate", "--doc", str(doc), "--share", "10", "--config", str(config)).code == 0
    fitted = json.loads(cli("antiplagiat", "--json", "--config", str(config), str(doc)).out)
    assert (fitted["model"], fitted["scoreKind"]) == ("calibrated", "fitted-to-reports")


def test_judge_json_and_score_name_prompt_version(cli: Cli, tmp_path: Path):
    """judge --json и judge score называют версию задания и документ, а сводка — лишние номера и разногласия"""
    prompt = json.loads(cli("judge", "--json", str(AI_VAK)).out)
    assert prompt["promptVersion"] == cli_module.PROMPT_VERSION
    assert re.fullmatch(r"[0-9a-f]{16}", prompt["documentHash"])
    first, second = tmp_path / "r1.txt", tmp_path / "r2.txt"
    first.write_text("# model: свежий агент\nP1 none\nP1 high\nP42 low", encoding="utf-8")
    second.write_text("P1 high", encoding="utf-8")
    r = cli("judge", "score", str(AI_VAK), "--answers", str(first), "--answers", str(second), env={"COLUMNS": "200"})
    out = " ".join(r.out.split())
    assert f"версия {cli_module.PROMPT_VERSION}, документ {prompt['documentHash']}" in out
    assert "номера, которых нет в документе: 42" in out
    assert "два разных ответа, взят первый" in out
    assert "Прогоны разошлись на два уровня и больше: фрагменты 1" in out
    data = json.loads(cli("judge", "score", "--json", str(AI_VAK), "--answers", str(first)).out)
    assert data["runMeta"] == [{"model": "свежий агент"}]

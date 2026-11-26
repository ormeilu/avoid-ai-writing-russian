"""Поведенческие проверки скиллов: cases.json, извлечение итогового текста, эталонные ответы."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from grade import Checks, EvalCase, extract_final, grade
from run import build_prompt, load_cases, split_command

from aiw_ru.compat import jsre

DIR = Path(__file__).resolve().parent.parent / "evals"
CASES = load_cases()


def by_id(case_id: str) -> EvalCase:
    return next(c for c in CASES if c["id"] == case_id)


def golden(name: str) -> str:
    return (DIR / "golden" / name).read_bytes().decode("utf-8")


def run_evals(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(DIR / "run.py"), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        env={**os.environ, "PYTHONUTF8": "1", **(env or {})},
    )


# ── cases.json ──


def test_ids_unique_and_checks_present() -> None:
    """идентификаторы уникальны, у каждого случая есть проверки"""
    assert len({c["id"] for c in CASES}) == len(CASES)
    for c in CASES:
        assert len(c["checks"]) > 0, c["id"]
        assert c["skill"] in ("avoid-ai-writing-russian", "antiplagiat")
        assert len(c["input"]) > 20, c["id"]


def test_check_regexes_compile() -> None:
    """регулярные выражения в проверках компилируются"""
    for c in CASES:
        for pattern in [*c["checks"].get("finalMatches", []), *c["checks"].get("responseMatches", [])]:
            jsre(pattern, "iu")


def test_forbidden_and_preserved_strings_in_source() -> None:
    """запрещённые строки действительно есть в исходнике, сохраняемые — тоже"""
    for c in CASES:
        if c["id"] != "injection":  # проверяет, что агент не дописал своё
            for s in c["checks"].get("forbid", []):
                assert s.replace("0,93", "0.93") in c["input"], f"{c['id']}: forbid «{s}»"
        if c["id"] != "vak-typography":  # ожидается исправленная форма
            for s in c["checks"].get("preserve", []):
                assert s in c["input"], f"{c['id']}: preserve «{s}»"


def test_all_modes_and_both_skills() -> None:
    """есть случаи на все режимы и оба скилла"""
    ids = {c["id"] for c in CASES}
    assert {"detect-only", "clean-noop", "injection", "invented-specifics"} <= ids
    assert any(c["skill"] == "antiplagiat" for c in CASES)


def test_prompt_contains_skill_request_and_text() -> None:
    """промпт содержит скилл, запрос и текст"""
    p = build_prompt(by_id("antiplagiat-borrowing"))
    assert "name: antiplagiat" in p
    assert "name: avoid-ai-writing-russian" in p
    assert "<text>" in p
    assert "перефразируй синонимами" in p


# ── run.py ──


def test_run_dry_prints_prompts() -> None:
    """--dry печатает хвосты промптов всех случаев и агента не вызывает"""
    results = DIR / "results"
    before = set(results.iterdir()) if results.exists() else set()
    r = run_evals("--dry")
    assert r.returncode == 0, r.stderr
    for c in CASES:
        assert f"── {c['id']} ──\n" in r.stdout
    assert r.stdout.count("</text>") == len(CASES)
    assert (set(results.iterdir()) if results.exists() else set()) == before


def test_run_dry_single_case() -> None:
    """--dry --case показывает один случай"""
    c = by_id("vak-protected")
    r = run_evals("--dry", "--case", "vak-protected")
    assert r.returncode == 0, r.stderr
    assert r.stdout == f"── vak-protected ──\n{build_prompt(c)[-600:]}\n\n"


def test_run_unknown_case() -> None:
    """неизвестный случай — код 2"""
    r = run_evals("--dry", "--case", "нет-такого")
    assert r.returncode == 2
    assert "нет случая нет-такого" in r.stderr


# ── extract_final ──


def test_extract_final_bold_heading() -> None:
    """жирный заголовок"""
    assert extract_final("**Итоговый текст**\n\nТекст.\n\n**Изменения**\n\nх") == "Текст."


def test_extract_final_stops_at_inline_section() -> None:
    """следующий раздел с текстом на той же строке в итоговый текст не попадает"""
    response = "**Итоговый текст:**\n\nТекст.\n\n**Проверка:** один проход, пришлите PDF отчёта."
    assert extract_final(response) == "Текст."


def test_extract_final_markdown_heading_and_quote() -> None:
    """заголовок Markdown и цитата"""
    response = "## Итоговый текст\n\n> Строка один.\n> Строка два.\n\n## Проверка\n"
    assert extract_final(response) == "Строка один.\nСтрока два."


def test_extract_final_code_block() -> None:
    """блок кода"""
    assert extract_final("**Итоговый текст:**\n```\nТекст.\n```\n**Проверка**") == "Текст."


def test_extract_final_missing() -> None:
    """нет раздела"""
    assert extract_final("Просто ответ") is None


# ── эталонные ответы ──


def test_golden_good_rewrite_passes() -> None:
    """хорошая правка проходит"""
    assert grade(by_id("blog-rewrite"), golden("blog-rewrite.md")).failures == []


def test_golden_invented_numbers_fail() -> None:
    """правка с выдуманными числами и оставленной подушкой — провал"""
    g = grade(by_id("blog-rewrite"), golden("blog-rewrite.bad.md"))
    assert not g.passed
    assert "выдуманные числа: 23, 2024" in "\n".join(g.failures)
    assert "осталось: Стоит отметить" in "\n".join(g.failures)


def test_golden_clean_noop_passes() -> None:
    """чистый текст без изменений проходит"""
    assert grade(by_id("clean-noop"), golden("clean-noop.md")).failures == []


def test_golden_detect_passes() -> None:
    """режим detect проходит"""
    assert grade(by_id("detect-only"), golden("detect-only.md")).failures == []


def test_golden_text_only_passes() -> None:
    """просили только текст: чистовик без заголовка и блоков после него проходит"""
    assert grade(by_id("text-only"), golden("text-only.md")).failures == []


def test_golden_text_only_with_tail_fails() -> None:
    """служебный хвост после чистовика, когда просили только текст, — провал"""
    g = grade(by_id("text-only"), golden("text-only.bad.md"))
    assert "в ответе лишнее: Изменения" in "\n".join(g.failures)


def test_golden_changed_quote_fails() -> None:
    """изменённая цитата — провал"""
    g = grade(by_id("protected-quote"), golden("protected-quote.bad.md"))
    assert not g.passed
    assert "пропало: «Данный подход" in "\n".join(g.failures)


def test_golden_whole_doc_keeps_code() -> None:
    """документ целиком: приложение с кодом на месте, сверка без детектора названа честно"""
    assert grade(by_id("whole-doc-keeps-code"), golden("whole-doc-keeps-code.md")).failures == []


def test_golden_whole_doc_dropped_code_fails() -> None:
    """пропавшее приложение со скриптом и «проверил целостность» без сверки — провал"""
    g = grade(by_id("whole-doc-keeps-code"), golden("whole-doc-keeps-code.bad.md"))
    failures = "\n".join(g.failures)
    assert "пропало: def shrinkage" in failures
    assert "в ответе лишнее: роверил целостность" in failures


def test_golden_no_cli_points_to_detector() -> None:
    """без командной строки ответ называет, где запустить детектор: свой компьютер, песочница, MCP"""
    assert grade(by_id("no-cli-points-to-detector"), golden("no-cli-points-to-detector.md")).failures == []


def test_golden_no_cli_silent_fails() -> None:
    """ответ без подсказки о детекторе, с «проверил целостность» и выдуманным числом — провал"""
    failures = "\n".join(grade(by_id("no-cli-points-to-detector"), golden("no-cli-points-to-detector.bad.md")).failures)
    assert "ответ не соответствует /командн/" in failures
    assert "в ответе лишнее: роверил целостность" in failures
    assert "выдуманные числа: 12" in failures


def test_golden_coursework_offers_checker() -> None:
    """правка абзаца из курсовой предлагает подготовку к «Антиплагиату» или «Думейту»"""
    answer = golden("coursework-offers-checker.md")
    assert grade(by_id("coursework-offers-checker"), answer).failures == []
    # Без предложения проверки тот же ответ не проходит.
    silent = "\n\n".join(part for part in answer.split("\n\n") if "Антиплагиат" not in part)
    assert "ответ не соответствует /(Антиплагиат|Думейт)/" in grade(by_id("coursework-offers-checker"), silent).failures


def test_missing_final_fails_for_edit() -> None:
    """ответ без итогового текста — провал для правки"""
    assert "в ответе нет раздела «Итоговый текст»" in grade(by_id("vak-protected"), "Всё хорошо.").failures


def only_check(checks: Checks) -> EvalCase:
    """Случай с одной проверкой: ничто другое не требует итогового текста."""
    return {
        "id": "only-check",
        "skill": "avoid-ai-writing-russian",
        "request": "Убери хэштеги.",
        "input": "Склад переехал в новое здание. #склад",
        "checks": checks,
    }


FINAL_WITH_TAG = "**Итоговый текст:**\n\nСклад переехал в новое здание. #склад"
FINAL_CLEAN = "**Итоговый текст:**\n\nСклад переехал в новое здание."


@pytest.mark.parametrize(
    ("checks", "response", "failure"),
    [
        pytest.param(
            {"maxHashtags": 0}, "Готово #склад", "в ответе нет раздела «Итоговый текст»", id="хэштеги без итога"
        ),
        pytest.param({"maxHashtags": 0}, FINAL_WITH_TAG, "хэштегов 1, допустимо 0", id="хэштег в итоге"),
        pytest.param({"maxGrowth": 0}, "Готово", "в ответе нет раздела «Итоговый текст»", id="рост без итога"),
    ],
)
def test_zero_limit_is_still_a_check(checks: Checks, response: str, failure: str) -> None:
    """отрицательный эталон: ограничение 0 — тоже проверка, ответ без итога или с хэштегом не проходит"""
    g = grade(only_check(checks), response)
    assert not g.passed
    assert failure in g.failures


def test_zero_hashtags_clean_final_passes() -> None:
    """итог без хэштегов проходит maxHashtags: 0"""
    assert grade(only_check({"maxHashtags": 0}), FINAL_CLEAN).failures == []


def test_disabled_flag_needs_no_final() -> None:
    """флаг false проверку выключает: итог не нужен"""
    assert grade(only_check({"noNewNumbers": False}), "Готово").failures == []


def test_golden_antiplagiat_asks_reports() -> None:
    """эталон antiplagiat-asks-reports: агент просит отчёты для калибровки"""
    assert grade(by_id("antiplagiat-asks-reports"), golden("antiplagiat-asks-reports.md")).failures == []


def test_golden_domate_asks_heatmap() -> None:
    """эталон: по одной доле «Думейта» скилл правит каркас абзаца и просит отчёт с тепловой картой"""
    assert grade(by_id("domate-asks-heatmap"), golden("domate-asks-heatmap.md")).failures == []


# ── запуск агента: тайм-аут, сбой, частичный ответ, рабочая папка ──

FAKE_AGENT = """
import os, shutil, subprocess, sys, time
mode = sys.argv[1]
prompt = sys.stdin.read()
golden = open(sys.argv[2], encoding="utf-8").read()
if mode == "ok":
    print(golden)
elif mode == "slow":
    print("Начал думать", flush=True)
    time.sleep(30)
elif mode == "crash":
    print("Половина ответа", flush=True)
    sys.exit(3)
elif mode == "slow-child":
    # Дочерний процесс держит вывод открытым: тайм-аут должен закрыть и его, а не ждать конца.
    subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    print("Начал думать", flush=True)
    time.sleep(60)
elif mode == "isolation":
    import json
    exe = shutil.which("aiw-ru")
    shown = subprocess.run([exe, "antiplagiat", "--json", "text.md"], capture_output=True, check=True, text=True)
    assert json.loads(shown.stdout)["model"] == "default", "личная калибровка попала в evals"
    real = os.path.expanduser("~/.config/aiw-ru.json")
    subprocess.run([exe, "calibrate", "--doc", "text.md", "--share", "0", "--config", real], check=True)
    print(golden)
elif mode == "workspace":
    exe = shutil.which("aiw-ru")
    for args in (["skill", "avoid-ai-writing-russian"], ["skill", "avoid-ai-writing-russian", "references/vocabulary.md"], ["scan", "text.md"]):
        subprocess.run([exe, *args], capture_output=True, check=True)
    assert open("text.md", encoding="utf-8").read().startswith("Первую неделю"), "текст пользователя в рабочей папке"
    assert "<file" not in prompt, "в рабочем режиме каталог не вклеивается в промпт"
    print(golden)
"""


@pytest.mark.parametrize(
    ("text", "windows", "expected"),
    [
        pytest.param(
            '"C:\\Program Files\\Python\\python.exe" "D:\\папка агента\\agent.py" ok',
            True,
            ["C:\\Program Files\\Python\\python.exe", "D:\\папка агента\\agent.py", "ok"],
            id="Windows: кавычки сняты, обратная косая черта цела",
        ),
        pytest.param(
            '"/usr/bin/python3" "/tmp/папка агента/agent.py" ok',
            False,
            ["/usr/bin/python3", "/tmp/папка агента/agent.py", "ok"],
            id="POSIX",
        ),
        pytest.param("claude -p --model sonnet", True, ["claude", "-p", "--model", "sonnet"], id="без кавычек"),
    ],
)
def test_split_command(text: str, windows: bool, expected: list[str]) -> None:
    """--agent разбирается как строка оболочки своей системы: путь с пробелом в кавычках — один аргумент"""
    assert split_command(text, windows=windows) == expected


@pytest.fixture
def fake_agent(tmp_path: Path) -> Callable[[str], str]:
    """Команда фальшивого агента для --agent; папка с пробелом проверяет разбор кавычек."""
    folder = tmp_path / "папка агента"
    folder.mkdir()
    script = folder / "agent.py"
    script.write_text(FAKE_AGENT, encoding="utf-8")
    answer = DIR / "golden" / "clean-noop.md"
    return lambda mode: f'"{sys.executable}" "{script}" {mode} "{answer}"'


def run_case(
    out: Path, agent: str, *extra: str, env: dict[str, str] | None = None
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    r = run_evals("--case", "clean-noop", "--agent", agent, "--out", str(out), *extra, env=env)
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    return r, summary


def test_agent_ok_saves_answer_and_summary(tmp_path: Path, fake_agent: Callable[[str], str]) -> None:
    """успешный прогон: ответ, сведения о запуске и сводка лежат в папке результатов"""
    r, summary = run_case(tmp_path / "out", fake_agent("ok"))
    assert r.returncode == 0, r.stdout + r.stderr
    (case,) = summary["cases"]
    assert (case["id"], case["passed"], case["error"]) == ("clean-noop", True, None)
    assert (tmp_path / "out" / "clean-noop.md").read_text(encoding="utf-8").startswith("**Итоговый текст**")
    assert json.loads((tmp_path / "out" / "clean-noop.json").read_text(encoding="utf-8"))["returncode"] == 0


def test_agent_timeout_keeps_partial_output(tmp_path: Path, fake_agent: Callable[[str], str]) -> None:
    """агент завис: тайм-аут, частичный ответ сохранён, причина записана, код 1"""
    r, summary = run_case(tmp_path / "out", fake_agent("slow"), "--timeout", "2")
    assert r.returncode == 1
    (case,) = summary["cases"]
    assert case["timedOut"] is True and "тайм-аут 2 с" in case["error"]
    assert "Начал думать" in (tmp_path / "out" / "clean-noop.md").read_text(encoding="utf-8")


def test_agent_crash_keeps_partial_output(tmp_path: Path, fake_agent: Callable[[str], str]) -> None:
    """агент упал: код его выхода и частичный ответ сохранены, случай не оценивается"""
    r, summary = run_case(tmp_path / "out", fake_agent("crash"))
    assert r.returncode == 1
    (case,) = summary["cases"]
    assert "код выхода 3" in case["error"] and case["passed"] is False
    assert "Половина ответа" in (tmp_path / "out" / "clean-noop.md").read_text(encoding="utf-8")


def test_missing_agent_is_reported(tmp_path: Path) -> None:
    """агента нет: причина в сводке, а не трассировка"""
    r, summary = run_case(tmp_path / "out", "нет-такого-агента-aiw --flag")
    assert r.returncode == 1
    assert "не запустился" in summary["cases"][0]["error"]
    assert "Traceback" not in r.stderr


def test_workspace_mode_logs_reads_and_commands(tmp_path: Path, fake_agent: Callable[[str], str]) -> None:
    """рабочий режим: скилл и детектор только через aiw-ru, журнал показывает, что агент читал и запускал"""
    r, summary = run_case(tmp_path / "out", fake_agent("workspace"), "--mode", "workspace")
    assert r.returncode == 0, r.stdout + r.stderr
    (case,) = summary["cases"]
    assert case["mode"] == "workspace"
    assert case["reads"] == ["SKILL.md", "references/vocabulary.md"]
    assert case["commands"] == ["scan"]


def test_workspace_mode_rerun_same_out(tmp_path: Path, fake_agent: Callable[[str], str]) -> None:
    """повторный прогон в ту же папку результатов: рабочая папка случая создаётся заново, журнал не копится"""
    first, _ = run_case(tmp_path / "out", fake_agent("workspace"), "--mode", "workspace")
    assert first.returncode == 0, first.stdout + first.stderr
    again, summary = run_case(tmp_path / "out", fake_agent("workspace"), "--mode", "workspace")
    assert again.returncode == 0, again.stdout + again.stderr
    assert summary["cases"][0]["commands"] == ["scan"]


def test_workspace_mode_ignores_personal_calibration(tmp_path: Path, fake_agent: Callable[[str], str]) -> None:
    """личная калибровка разработчика в evals не читается и не перезаписывается, даже по явному --config"""
    home = tmp_path / "дом"
    config = home / ".config" / "aiw-ru.json"
    config.parent.mkdir(parents=True)
    calibrated = {
        "version": 1,
        "model": {"bias": -1.0, "weights": [1.0] * 7, "threshold": 0.5},
        "samples": [{"f": [0.1] * 7, "y": 1}],
    }
    config.write_text(json.dumps(calibrated), encoding="utf-8")
    before = config.read_bytes()
    env = {"HOME": str(home), "USERPROFILE": str(home), "AIW_RU_CONFIG": str(config)}
    r, summary = run_case(tmp_path / "out", fake_agent("isolation"), "--mode", "workspace", env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert summary["cases"][0]["passed"], summary
    assert config.read_bytes() == before
    assert (tmp_path / "out" / "workspaces" / "clean-noop" / "home" / ".config" / "aiw-ru.json").is_file()


def test_timeout_stops_agent_children(tmp_path: Path, fake_agent: Callable[[str], str]) -> None:
    """тайм-аут закрывает и дочерние процессы агента: прогон не ждёт, пока они сами выйдут"""
    started = time.monotonic()
    r, summary = run_case(tmp_path / "out", fake_agent("slow-child"), "--timeout", "2")
    assert time.monotonic() - started < 30
    assert r.returncode == 1
    assert summary["cases"][0]["timedOut"] is True
    assert "Начал думать" in (tmp_path / "out" / "clean-noop.md").read_text(encoding="utf-8")


def test_dry_workspace_prompt_has_no_catalog() -> None:
    """промпт рабочего режима не вклеивает каталог: агент читает его сам"""
    r = run_evals("--dry", "--case", "clean-noop", "--mode", "workspace")
    assert r.returncode == 0, r.stderr
    assert "<file" not in r.stdout and "aiw-ru skill avoid-ai-writing-russian" in r.stdout

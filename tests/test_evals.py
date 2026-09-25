"""Поведенческие проверки скиллов: cases.json, извлечение итогового текста, эталонные ответы."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from grade import EvalCase, extract_final, grade
from run import build_prompt, load_cases

from aiw_ru.compat import jsre

DIR = Path(__file__).resolve().parent.parent / "evals"
CASES = load_cases()


def by_id(case_id: str) -> EvalCase:
    return next(c for c in CASES if c["id"] == case_id)


def golden(name: str) -> str:
    return (DIR / "golden" / name).read_bytes().decode("utf-8")


def run_evals(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(DIR / "run.py"), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        env={**os.environ, "PYTHONUTF8": "1"},
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


def test_golden_changed_quote_fails() -> None:
    """изменённая цитата — провал"""
    g = grade(by_id("protected-quote"), golden("protected-quote.bad.md"))
    assert not g.passed
    assert "пропало: «Данный подход" in "\n".join(g.failures)


def test_missing_final_fails_for_edit() -> None:
    """ответ без итогового текста — провал для правки"""
    assert "в ответе нет раздела «Итоговый текст»" in grade(by_id("vak-protected"), "Всё хорошо.").failures


def test_golden_antiplagiat_asks_reports() -> None:
    """эталон antiplagiat-asks-reports: агент просит отчёты для калибровки"""
    assert grade(by_id("antiplagiat-asks-reports"), golden("antiplagiat-asks-reports.md")).failures == []

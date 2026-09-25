"""Проверка сохранности правки."""

from aiw_ru import validate

BEFORE = (
    "---\n"
    "title: Статья\n"
    "---\n"
    "\n"
    "# Введение\n"
    "\n"
    'Данный метод играет ключевую роль: точность 0.93 [12], см. "отчёт" и https://example.com/a?utm_source=chatgpt.com.\n'
    "\n"
    "```python\n"
    "x = 1\n"
    "```"
)


def test_allowed_edits_pass():
    """разрешённые правки проходят"""
    after = (
        BEFORE.replace("Данный метод играет ключевую роль", "Метод повышает полноту", 1)
        .replace("0.93", "0,93", 1)
        .replace('"отчёт"', "«отчёт»", 1)
        .replace("?utm_source=chatgpt.com", "", 1)
    )
    r = validate(BEFORE, after)
    assert r.violations == []
    assert r.ok is True


def test_changed_number_code_citation_caught():
    """изменённое число, код и ссылка ловятся"""
    after = BEFORE.replace("0.93", "0.95", 1).replace("x = 1", "x = 2", 1).replace("[12]", "[13]", 1)
    kinds = {v.kind for v in validate(BEFORE, after).violations}
    assert {"число", "код", "ссылка на литературу"} <= kinds


def test_yaml_header_and_headings():
    """YAML-шапка и заголовки"""
    after = BEFORE.replace("title: Статья", "title: Другая", 1).replace("# Введение", "## Введение", 1)
    kinds = {v.kind for v in validate(BEFORE, after).violations}
    assert {"yaml", "заголовки"} <= kinds


def test_more_issues_is_violation():
    """рост числа находок — нарушение"""
    r = validate("Метод работает.", "Давайте разберёмся: метод играет ключевую роль.")
    assert "находки" in [v.kind for v in r.violations]

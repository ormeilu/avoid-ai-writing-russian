"""Примеры скилла не нарушают его же правило «Никогда не добавляй».

Пример учит сильнее инструкции: если в «после» появилась цифра, имя или месяц,
которых не было в «до», модель усвоит, что так можно. Поэтому пары «до → после»
из файлов скилла и эталонные ответы из evals/golden проверяются той же сверкой
фактов, что и правка пользователя (`aiw-ru validate`). Плохие эталоны (*.bad.md),
наоборот, должны на ней падать.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from grade import extract_final

from aiw_ru import validate

ROOT = Path(__file__).parent.parent
SKILLS = ROOT / "skills"
GOLDEN = ROOT / "evals" / "golden"
CASES = {c["id"]: c for c in json.loads((ROOT / "evals" / "cases.json").read_text(encoding="utf-8"))["cases"]}

# Виды сверки фактов (см. src/aiw_ru/validate.py). Для своих примеров строже, чем для
# правки пользователя: предупреждения («один… второй», «впервые») тоже не допускаются,
# пример должен проходить сверку без замечаний.
FACT_KINDS = {"оговорка", "имя", "число", "число словами", "месяц", "утверждение"}

# Пример замены в тексте скилла: «было» → «стало» или «стало» / «другой вариант».
PAIR_RE = re.compile(r"«([^«»\n]+)»\s*→\s*((?:«[^«»\n]+»(?:\s*/\s*)?)+)")
QUOTED_RE = re.compile(r"«([^«»\n]+)»")
# Не замены, а последовательности: заголовок и фраза после него.
NOT_REPLACEMENTS = ("## ",)


def skill_pairs() -> list[tuple[str, str, str]]:
    pairs = []
    for path in sorted(SKILLS.rglob("*.md")):
        for m in PAIR_RE.finditer(path.read_text(encoding="utf-8")):
            if m[1].startswith(NOT_REPLACEMENTS):
                continue
            for after in QUOTED_RE.findall(m[2]):
                pairs.append((f"{path.parent.name}/{path.name}", m[1], after))
    return pairs


def facts(before: str, after: str) -> list[str]:
    r = validate(before, after)
    return [f"{v.kind}: {v.detail}" for v in [*r.violations, *r.warnings] if v.kind in FACT_KINDS]


PAIRS = skill_pairs()


def test_skill_has_replacement_examples():
    """примеры замен находятся: проверка ниже не пустая"""
    assert len(PAIRS) >= 5


@pytest.mark.parametrize(("where", "before", "after"), PAIRS, ids=[f"{w}: {b} → {a}" for w, b, a in PAIRS])
def test_replacement_example_adds_no_facts(where: str, before: str, after: str):
    """пример замены в скилле не добавляет фактов и не снимает оговорок"""
    assert facts(before, after) == []


def final_text(name: str) -> str:
    """Итоговый текст эталонного ответа тем же разбором, что в evals/grade.py."""
    final = extract_final((GOLDEN / name).read_text(encoding="utf-8"))
    assert final is not None, name
    return final


GOOD = sorted(p.name for p in GOLDEN.glob("*.md") if not p.name.endswith(".bad.md"))
BAD = sorted(p.name for p in GOLDEN.glob("*.bad.md"))


@pytest.mark.parametrize("name", [n for n in GOOD if "Итоговый текст" in (GOLDEN / n).read_text(encoding="utf-8")])
def test_golden_answer_adds_no_facts(name: str):
    """эталонный ответ не добавляет фактов к исходнику сценария"""
    case = CASES[name.removesuffix(".md")]
    assert facts(case["input"], final_text(name)) == []


@pytest.mark.parametrize("name", BAD)
def test_bad_golden_caught_by_fact_check(name: str):
    """плохой эталон ловится сверкой: он для того и написан"""
    case = CASES[name.removesuffix(".bad.md")]
    assert not validate(case["input"], final_text(name)).ok

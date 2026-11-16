"""Отрицательный регрессионный корпус: выдуманные человеческие тексты жанров, где ложные находки
вероятнее всего. Законный канцелярит приказа, формулы договора, Markdown-документация, короткие
письма, прямая речь и служебная инструкция.

Это регрессии на известные случаи, а не замер доли ложных срабатываний на живых людях: для неё нужен
отдельный разрешённый корпус. Известное ложное срабатывание отмечено xfail(strict=True) с причиной.
Правило ради примера не подкручивается, а когда его поправят, тест напомнит снять отметку.
"""

from pathlib import Path

import pytest

from aiw_ru import ContextMode, analyze, antiplagiat

FOLDER = Path(__file__).parent / "fixtures" / "corpus" / "human-regression"
MODES: dict[str, ContextMode] = {
    "official": "general",
    "contract": "general",
    "markdown": "technical",
    "notes": "general",
    "dialogue": "general",
    "instruction": "general",
}
# Известные ложные срабатывания: пример — причина.
KNOWN = {
    "instruction": ("uniform-sentences: пункты инструкции одной длины дают P1, оценку 42 и долю ИИ 36 % в antiplagiat"),
}


def test_folder_matches_table():
    """каждый файл папки есть в таблице режимов, и наоборот"""
    assert sorted(path.stem for path in FOLDER.glob("*.md")) == sorted(MODES)


@pytest.mark.parametrize(
    ("name", "mode"),
    [
        pytest.param(
            name, mode, id=name, marks=[pytest.mark.xfail(strict=True, reason=KNOWN[name])] if name in KNOWN else []
        )
        for name, mode in MODES.items()
    ],
)
def test_regression_human_text_is_clean(name: str, mode: ContextMode):
    """нет находок P0 и P1, низкая оценка, antiplagiat не метит ни одного фрагмента"""
    text = (FOLDER / f"{name}.md").read_text(encoding="utf-8")
    result = analyze(text, mode)
    assert [f"{issue.type}: {issue.text}" for issue in result.issues if issue.severity != "P2"] == []
    assert result.score < 15
    assert result.suspicious is False
    assert antiplagiat(text).ai_share == 0

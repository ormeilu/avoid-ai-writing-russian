"""Проверка вероятностей на независимых данных: Brier, log loss, надёжность по диапазонам, FPR у порога.

Скрипт только считает: данные и решение, подходят ли они для замера, остаются за владельцем корпуса.
"""

import json
import math
from pathlib import Path

import pytest

from probability_check import Row, brier, log_loss, main, rate_at, read_rows, reliability, report

PERFECT = [Row(0.0, 0), Row(1.0, 1)] * 5
COIN = [Row(0.5, 0), Row(0.5, 1)] * 5


def test_brier_bounds():
    """Brier: 0 у идеальной модели, 0,25 у монетки"""
    assert brier(PERFECT) == 0
    assert brier(COIN) == pytest.approx(0.25)


def test_log_loss_clips_extremes():
    """log loss не уходит в бесконечность на уверенной ошибке"""
    assert math.isfinite(log_loss([Row(1.0, 0)]))
    assert log_loss(COIN) == pytest.approx(math.log(2))


def test_reliability_bins():
    """надёжность: в каждом диапазоне средняя вероятность и доля ИИ рядом, ECE — их разрыв по весу"""
    rows = [Row(0.15, 0)] * 9 + [Row(0.15, 1)] + [Row(0.85, 1)] * 8 + [Row(0.85, 0)] * 2
    bins, ece = reliability(rows, bins=10)
    low = next(b for b in bins if b.low == pytest.approx(0.1))
    assert (low.count, low.mean_probability, low.ai_share) == (10, pytest.approx(0.15), pytest.approx(0.1))
    assert ece == pytest.approx((10 * 0.05 + 10 * 0.05) / 20)


def test_rates_at_threshold():
    """у порога: доля людей выше порога (FPR) и доля ИИ ниже (FNR)"""
    rows = [Row(0.7, 0), Row(0.2, 0), Row(0.9, 1), Row(0.4, 1)]
    assert rate_at(rows, 0.5) == (0.5, 0.5)


def test_report_by_group_and_empty_class():
    """срезы по группе; группа без людей или без ИИ не даёт FPR или FNR, а не делит на ноль"""
    rows = [Row(0.9, 1, "blog"), Row(0.2, 0, "blog"), Row(0.8, 1, "vak")]
    data = report(rows, threshold=0.5, bins=5)
    assert data["count"] == 3
    assert data["groups"]["vak"]["fpr"] is None
    assert data["groups"]["blog"]["fpr"] == 0


def test_read_rows_rejects_bad_lines(tmp_path: Path):
    """строка без вероятности 0–1 или метки 0/1 — ошибка с номером строки"""
    path = tmp_path / "scores.jsonl"
    path.write_text('{"probability": 0.3, "label": 0}\n{"probability": 1.3, "label": 1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="строка 2"):
        read_rows(path)


def test_main_prints_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """скрипт печатает отчёт в JSON и напоминает, что данные должны быть независимыми"""
    path = tmp_path / "scores.jsonl"
    path.write_text(
        "\n".join(json.dumps({"probability": p, "label": y}) for p, y in [(0.9, 1), (0.1, 0)]), encoding="utf-8"
    )
    assert main([str(path), "--threshold", "0.5"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["brier"] == pytest.approx(0.01)
    assert "независим" in data["note"]

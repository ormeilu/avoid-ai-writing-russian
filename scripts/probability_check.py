"""Проверка вероятностей детектора или модели на размеченных данных: насколько число похоже на вероятность.

Вход — JSONL, по строке на текст: `{"probability": 0.83, "label": 1, "group": "blog"}`. label — 1 для
ИИ-текста и 0 для человеческого, group — необязательный срез (жанр, длина, источник). Выход — JSON:
Brier, log loss, таблица надёжности по диапазонам вероятности с ECE и доли ошибок у порога
(людей выше порога, ИИ ниже) целиком и по группам.

Скрипт только считает. Замеру можно верить лишь на данных, которые не участвовали ни в обучении модели, ни в
выборе её порога, ни в подборе весов эвристики; какие данные подходят, решает владелец корпуса.

    uv run python scripts/probability_check.py scores.jsonl --threshold 0.5 [--bins 10]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Вероятность в log loss сжимается в [EPSILON, 1 − EPSILON]: уверенная ошибка даёт большой, но конечный штраф.
EPSILON = 1e-6
NOTE = (
    "Числа годятся, только если данные независимы от обучения модели, выбора порога и подбора весов "
    "эвристики. Если это не так, они завышены."
)


@dataclass(frozen=True, slots=True)
class Row:
    probability: float
    label: int
    group: str = ""


@dataclass(frozen=True, slots=True)
class Bin:
    low: float
    high: float
    count: int
    mean_probability: float | None
    ai_share: float | None


def brier(rows: list[Row]) -> float:
    return sum((row.probability - row.label) ** 2 for row in rows) / len(rows)


def log_loss(rows: list[Row]) -> float:
    total = 0.0
    for row in rows:
        probability = min(max(row.probability, EPSILON), 1 - EPSILON)
        total -= math.log(probability) if row.label == 1 else math.log(1 - probability)
    return total / len(rows)


def reliability(rows: list[Row], bins: int = 10) -> tuple[list[Bin], float]:
    """Диапазоны вероятности равной ширины: средняя вероятность, доля ИИ и ECE (взвешенный разрыв между ними)."""
    result: list[Bin] = []
    gap = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        inside = [row for row in rows if low <= row.probability < high or (index == bins - 1 and row.probability == 1)]
        if not inside:
            result.append(Bin(low, high, 0, None, None))
            continue
        mean_probability = sum(row.probability for row in inside) / len(inside)
        ai_share = sum(row.label for row in inside) / len(inside)
        gap += len(inside) * abs(mean_probability - ai_share)
        result.append(Bin(low, high, len(inside), mean_probability, ai_share))
    return result, gap / len(rows)


def rate_at(rows: list[Row], threshold: float) -> tuple[float | None, float | None]:
    """Доля людей не ниже порога (FPR) и доля ИИ ниже порога (FNR); None, если класса нет."""
    humans = [row for row in rows if row.label == 0]
    machines = [row for row in rows if row.label == 1]
    fpr = sum(row.probability >= threshold for row in humans) / len(humans) if humans else None
    fnr = sum(row.probability < threshold for row in machines) / len(machines) if machines else None
    return fpr, fnr


def _summary(rows: list[Row], threshold: float, bins: int) -> dict[str, Any]:
    table, ece = reliability(rows, bins)
    fpr, fnr = rate_at(rows, threshold)
    return {
        "count": len(rows),
        "brier": brier(rows),
        "logLoss": log_loss(rows),
        "ece": ece,
        "fpr": fpr,
        "fnr": fnr,
        "bins": [
            {
                "low": b.low,
                "high": b.high,
                "count": b.count,
                "meanProbability": b.mean_probability,
                "aiShare": b.ai_share,
            }
            for b in table
        ],
    }


def report(rows: list[Row], threshold: float = 0.5, bins: int = 10) -> dict[str, Any]:
    groups = sorted({row.group for row in rows if row.group})
    return {
        **_summary(rows, threshold, bins),
        "threshold": threshold,
        "groups": {group: _summary([row for row in rows if row.group == group], threshold, bins) for group in groups},
        "note": NOTE,
    }


def read_rows(path: Path) -> list[Row]:
    rows: list[Row] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        data = json.loads(line)
        probability, label = data.get("probability"), data.get("label")
        valid_probability = (
            isinstance(probability, int | float) and math.isfinite(probability) and 0 <= probability <= 1
        )
        if not valid_probability or label not in (0, 1) or isinstance(label, bool):
            raise ValueError(f"строка {number}: нужны probability от 0 до 1 и label 0 или 1")
        rows.append(Row(float(probability), int(label), str(data.get("group") or "")))
    if not rows:
        raise ValueError("в файле нет строк")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("scores", type=Path, help="JSONL с probability, label и необязательной group")
    parser.add_argument("--threshold", type=float, default=0.5, help="порог решения «ИИ», по умолчанию 0,5")
    parser.add_argument("--bins", type=int, default=10, help="диапазонов вероятности в таблице надёжности")
    args = parser.parse_args(argv)
    if not 0 < args.threshold < 1 or args.bins < 1:
        parser.error("порог строго между 0 и 1, диапазонов хотя бы один")
    try:
        rows = read_rows(args.scores)
    except (OSError, ValueError) as error:
        print(f"probability_check: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report(rows, args.threshold, args.bins), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

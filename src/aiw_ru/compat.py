"""Семантика регулярных выражений и чисел, на которой откалиброваны правила детектора.

Правила записаны в синтаксисе регулярных выражений JavaScript, а оценки
округляются по его правилам: так пороги и веса дают те же ответы, на которых их
подбирали. `jsre` переводит `\\s`, `\\d`, `\\w`, `\\b`, `.`, `^` и `$` в их значения
из JavaScript (например, `\\w` и `\\b` — только ASCII), а флаги `i`, `m`, `s` — во
флаги пакета `regex`. `to_fixed` и `js_round` округляют половину вверх, как
`toFixed` и `Math.round`.
"""

from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal
from functools import cache

import regex

# Пробельные символы и концы строк JS (\s и LineTerminator).
WS = "\\t\\n\\x0b\\x0c\\r \\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000\\ufeff"
_WS_CHARS = "".join(
    map(
        chr,
        [9, 10, 11, 12, 13, 32, 0xA0, 0x1680, *range(0x2000, 0x200B), 0x2028, 0x2029, 0x202F, 0x205F, 0x3000, 0xFEFF],
    )
)
_LT = "\\n\\r\\u2028\\u2029"
_WORD = "A-Za-z0-9_"
_BOUNDARY = f"(?:(?<=[{_WORD}])(?![{_WORD}])|(?<![{_WORD}])(?=[{_WORD}]))"
_NOT_BOUNDARY = f"(?:(?<=[{_WORD}])(?=[{_WORD}])|(?<![{_WORD}])(?![{_WORD}]))"

_OUTSIDE = {
    "s": f"[{WS}]",
    "S": f"[^{WS}]",
    "d": "[0-9]",
    "D": "[^0-9]",
    "w": f"[{_WORD}]",
    "W": f"[^{_WORD}]",
    "b": _BOUNDARY,
    "B": _NOT_BOUNDARY,
}
_INSIDE = {"s": WS, "d": "0-9", "w": _WORD}


def translate(source: str, flags: str = "") -> str:
    """Переводит выражение JS в синтаксис пакета `regex` с той же семантикой."""
    multiline = "m" in flags
    dotall = "s" in flags
    parts: list[str] = []
    i = 0
    in_class = False
    while i < len(source):
        char = source[i]
        if char == "\\" and i + 1 < len(source):
            escaped = source[i + 1]
            if in_class and escaped in _INSIDE:
                parts.append(_INSIDE[escaped])
            elif not in_class and escaped in _OUTSIDE:
                parts.append(_OUTSIDE[escaped])
            else:
                parts.append(source[i : i + 2])
            i += 2
            continue
        if in_class:
            if char == "]":
                in_class = False
            parts.append(char)
        elif source.startswith("[\\s\\S]", i):
            # «Любой знак»: объединение \s и \S в Python то же самое.
            parts.append("[\\s\\S]")
            i += 6
            continue
        elif char == "[":
            in_class = True
            parts.append(char)
        elif char == "." and not dotall:
            parts.append(f"[^{_LT}]")
        elif char == "$":
            parts.append(f"(?=[{_LT}]|\\Z)" if multiline else "\\Z")
        elif char == "^" and multiline:
            parts.append(f"(?:\\A|(?<=[{_LT}]))")
        else:
            parts.append(char)
        i += 1
    return "".join(parts)


@cache
def jsre(source: str, flags: str = "") -> regex.Pattern[str]:
    """Компилирует выражение JS; флаги `g`, `u`, `y` для Python значения не имеют."""
    regex_flags = regex.V0
    if "i" in flags:
        regex_flags |= regex.IGNORECASE
    if "s" in flags:
        regex_flags |= regex.DOTALL
    return regex.compile(translate(source, flags), regex_flags)


def trim(s: str) -> str:
    """String.prototype.trim: убирает пробелы и концы строк JS."""
    return s.strip(_WS_CHARS)


def trim_start(s: str) -> str:
    return s.lstrip(_WS_CHARS)


def trim_end(s: str) -> str:
    return s.rstrip(_WS_CHARS)


def js_round(x: float) -> int:
    """Math.round: половина округляется вверх (−2,5 → −2)."""
    floor = math.floor(x)
    return int(floor + 1 if x - floor >= 0.5 else floor)


def to_fixed(x: float, digits: int) -> str:
    """Number.prototype.toFixed: точное двоичное значение, половина — от нуля."""
    if math.isnan(x):
        return "NaN"
    quantized = Decimal(x).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
    text = f"{quantized:f}"
    return text[1:] if text.startswith("-") and not quantized else text


def fixed(x: float, digits: int) -> float | int:
    """Number(x.toFixed(digits)): целое остаётся целым, как в JSON из JS."""
    value = float(to_fixed(x, digits))
    return int(value) if value.is_integer() else value


def num_str(x: float) -> str:
    """String(x) для чисел, которые встречаются в выводе: целые без «.0»."""
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(x)


def total(xs) -> float:
    """Сумма слева направо, как reduce в JS; sum() в Python 3.12+ считает с компенсацией."""
    running_sum = 0
    for x in xs:
        running_sum += x
    return running_sum

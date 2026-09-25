"""Подготовка текста: нормализация, маскирование защищённых областей,
разбиение на абзацы, предложения и слова.

Все преобразования, кроме удаления невидимых символов, сохраняют длину строки,
поэтому смещения находок совпадают с исходником. Удалённые невидимые символы
учитываются картой смещений `to_source`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from aiw_ru.compat import jsre, total, trim, trim_start

LETTER = "[\\p{L}\\p{N}_]"
WORD_START = f"(?<!{LETTER})"
WORD_END = f"(?!{LETTER})"

WORD_RE = jsre("[\\p{L}\\p{N}]+(?:[-’'][\\p{L}\\p{N}]+)*", "gu")
CYRILLIC_RE = jsre("\\p{Script=Cyrillic}", "u")
LATIN_RE = jsre("\\p{Script=Latin}", "u")

# Латинские буквы, неотличимые от кириллических.
LATIN_TO_CYRILLIC = dict(zip("aeopcxykmthbAEOPCXYKMTHB", "аеорсхукмтнвАЕОРСХУКМТНВ", strict=True))
CYRILLIC_TO_LATIN = {c: latin for latin, c in LATIN_TO_CYRILLIC.items()}

# Латиница без диакритики. «á» в «Кáрмен» — знак ударения, а не подмена,
# поэтому в сериях она не участвует, как цифры и прочие знаки.
ASCII_LATIN_RE = jsre("[A-Za-z]")
LOWER_RE = jsre("\\p{Ll}", "u")
TIMES_RE = jsre("^[\\u0445\\u0425xX]$", "u")


@dataclass(slots=True)
class _Run:
    """Серия букв одного алфавита; цифры и прочие знаки её не обрывают."""

    latin: bool
    start: int
    end: int


def _script_runs(s: str) -> list[_Run]:
    runs: list[_Run] = []
    for i, c in enumerate(s):
        latin = bool(ASCII_LATIN_RE.match(c))
        if not latin and not CYRILLIC_RE.match(c):
            continue
        if runs and runs[-1].latin == latin:
            runs[-1].end = i + 1
        else:
            runs.append(_Run(latin, i, i + 1))
    return runs


def _letters(s: str, r: _Run) -> list[str]:
    """Буквы серии без цифр и прочих знаков."""
    own = ASCII_LATIN_RE if r.latin else CYRILLIC_RE
    return [c for c in s[r.start : r.end] if own.match(c)]


def _lookalike(s: str, r: _Run) -> bool:
    """У каждой буквы серии есть двойник в другом алфавите."""
    table = LATIN_TO_CYRILLIC if r.latin else CYRILLIC_TO_LATIN
    return all(c in table for c in _letters(s, r))


def _spoofed(s: str) -> bool:
    """Подмена букв в части слова без дефисов.

    Алфавит слова выдаёт буква без двойника: «щ» в русском слове, «n» в английском.
    Подозрительны буквы-двойники другого алфавита. Если без двойника есть буквы
    обоих алфавитов (украинское слово с латинской i вместо і) или нет ни одной,
    подозрителен алфавит, в котором букв меньше. Законная смесь — латинская основа
    с русским окончанием («Pythonе», «PHPшник», «OKей») и слипшийся предлог
    («вPython»). Подмена — двойники внутри слова, в конце русского слова, латинские
    в начале русского слова и одна-две кириллические в начале английского.
    """
    runs = _script_runs(s)
    if len(runs) < 2:
        return False
    first, second, last = runs[0], runs[1], runs[-1]

    def of(latin: bool) -> list[str]:
        return [c for r in runs if r.latin == latin for c in _letters(s, r)]

    latin_letters, cyrillic_letters = of(True), of(False)
    latin_anchored = any(c not in LATIN_TO_CYRILLIC for c in latin_letters)
    cyrillic_anchored = any(c not in CYRILLIC_TO_LATIN for c in cyrillic_letters)

    def suspect_script(latin: bool) -> bool:
        if latin_anchored != cyrillic_anchored:
            return latin == cyrillic_anchored
        if latin:
            return len(latin_letters) <= len(cyrillic_letters)
        return len(cyrillic_letters) <= len(latin_letters)

    def suspect(r: _Run) -> bool:
        return suspect_script(r.latin) and _lookalike(s, r)

    # Алфавиты чередуются, поэтому внутренняя серия зажата буквами другого алфавита.
    for k in range(1, len(runs) - 1):
        r = runs[k]
        # «FхG», «mхn»: одиночная «х» между одиночными буквами — знак умножения.
        times = (
            bool(TIMES_RE.search("".join(_letters(s, r))))
            and len(_letters(s, runs[k - 1])) == 1
            and len(_letters(s, runs[k + 1])) == 1
        )
        if not times and suspect(r):
            return True
    if last.latin and suspect(last):
        return True
    if not suspect(first):
        return False
    head = "".join(_letters(s, first))
    # Латинское сокращение перед русским суффиксом («OKей», «PHPшник»), но не заглавное русское слово.
    lower_next = second.start < len(s) and bool(LOWER_RE.match(s[second.start]))
    if first.latin:
        return not (len(head) >= 2 and head == head.upper() and lower_next)
    # Кириллица перед английским словом: подмена, если слово продолжается строчными,
    # и слипшийся предлог, если дальше заглавная («вPython»).
    return len(head) <= 2 and lower_next


# Знаки направления текста: метки LRM, RLM и ALM, встраивания и переопределения
# U+202A–U+202E, изоляторы U+2066–U+2069. Клавиатурой в русском тексте не набираются.
BIDI = frozenset(map(chr, [0x200E, 0x200F, 0x061C, *range(0x202A, 0x202F), *range(0x2066, 0x206A)]))
# Монгольский разделитель гласных: законен только внутри монгольского слова.
MVS = chr(0x180E)
# Символы, которые удаляются из текста перед поиском.
STRIPPED = frozenset(map(chr, [0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF, 0x00AD])) | BIDI | {MVS}
SOFT_HYPHEN = chr(0x00AD)
BOM = chr(0xFEFF)
ZWJ = chr(0x200D)
HYPHENS = frozenset(map(chr, [0x2010, 0x2011]))
EMOJI_BEFORE_ZWJ = jsre("(?:\\p{Extended_Pictographic}|\\p{Emoji_Modifier}|\\uFE0F)$", "u")
EMOJI_AFTER_ZWJ = jsre("^\\p{Extended_Pictographic}", "u")
RTL_RE = jsre("[\\p{Script=Hebrew}\\p{Script=Arabic}\\p{Script=Syriac}\\p{Script=Thaana}\\p{Script=Nko}]", "u")
MONGOLIAN_RE = jsre("\\p{Script=Mongolian}", "u")


def _joins_emoji(s: str, i: int) -> bool:
    """U+200D между частями эмодзи (👨 + 💻) — соединитель, а не вставка."""
    return bool(EMOJI_BEFORE_ZWJ.search(s[max(0, i - 2) : i])) and bool(EMOJI_AFTER_ZWJ.search(s[i + 1 : i + 3]))


def _legit_control(s: str, i: int) -> bool:
    """Знак направления рядом с ивритом или арабским и разделитель внутри монгольского слова — не вставка."""
    ch = s[i]
    if ch in BIDI:
        # Сам знак ALM (U+061C) относится к арабскому письму, поэтому смотрим только на соседей.
        return bool(RTL_RE.search(s[max(0, i - 3) : i] + " " + s[i + 1 : i + 4]))
    if ch == MVS:
        return bool(MONGOLIAN_RE.search(s[i - 1 : i])) and bool(MONGOLIAN_RE.search(s[i + 1 : i + 2]))
    return False


@dataclass(slots=True)
class Invisible:
    index: int
    char: str


@dataclass(slots=True)
class Homoglyph:
    index: int
    word: str


@dataclass(slots=True)
class Prepared:
    # Исходный текст.
    source: str
    # Нормализованный текст: без невидимых символов, ё→е, U+2010/U+2011→дефис, без подмены букв.
    text: str
    # Нормализованный текст, где код и YAML-шапка заменены пробелами.
    no_code: str
    # Нормализованный текст, где замаскированы все защищённые области.
    prose: str
    # Смещение в `text` → смещение в `source`.
    to_source: list[int]
    # Невидимые вставки. BOM в начале текста, соединитель внутри эмодзи и знаки направления
    # рядом с ивритом или арабским сюда не входят.
    invisible: list[Invisible]
    # Позиции мягких переносов (U+00AD) в исходнике: их ставят Word и копирование из PDF.
    soft_hyphens: list[int]
    homoglyphs: list[Homoglyph]
    line_starts: list[int]


def _mask(chars: list[str], start: int, end: int) -> None:
    for i in range(start, end):
        if chars[i] != "\n":
            chars[i] = " "


def _mask_all(chars: list[str], text: str, pattern: str, flags: str = "g") -> None:
    for m in jsre(pattern, flags).finditer(text):
        _mask(chars, m.start(), m.end())


def _mask_code(text: str) -> str:
    """Маскирует YAML-шапку, блоки кода и инлайн-код."""
    chars = list(text)
    fm = jsre("^---\\n[\\s\\S]*?\\n---(?:\\n|$)").search(text)
    if fm:
        _mask(chars, 0, fm.end())
    _mask_all(chars, text, "^(```|~~~)[^\\n]*\\n[\\s\\S]*?^\\1[^\\n]*$", "gm")
    _mask_all(chars, text, "`[^`\\n]+`")
    return "".join(chars)


def _mask_protected(no_code: str) -> str:
    """Маскирует всё, что скилл считает защищённым содержимым."""
    chars = list(no_code)
    for pattern, flags in (
        ("<!--[\\s\\S]*?-->", "g"),
        ("\\$\\$[\\s\\S]*?\\$\\$", "g"),
        ("\\$[^$\\n]+\\$", "g"),
        ("\\\\\\([\\s\\S]*?\\\\\\)", "g"),
        ("https?:\\/\\/[^\\s)>\\]»]+", "g"),
        ("\\]\\([^)\\n]*\\)", "g"),
        ("^[ \\t]*\\|.*$", "gm"),
        ("^[ \\t]*>.*$", "gm"),
        # Цитаты в «ёлочках» и „лапках“ (до 400 знаков, без перевода абзаца).
        ("«[^«»\\n]{0,400}»", "g"),
        ("„[^„“\\n]{0,400}“", "g"),
        ('"[^"\\n]{1,400}"', "g"),
        ("“[^“”\\n]{1,400}”", "g"),
        # Ссылки на литературу: [12], [3; 7], [12, с. 45].
        ("\\[\\d+(?:[,;–-]\\s*\\d+)*(?:,\\s*с\\.\\s*\\d+(?:[–-]\\d+)?)?\\]", "g"),
        # Ссылки pandoc: [@key], [@a; @b, с. 5].
        ("\\[-?@[^\\]\\n]+\\]", "g"),
    ):
        _mask_all(chars, no_code, pattern, flags)
    return "".join(chars)


def _line_starts(s: str) -> list[int]:
    starts = [0]
    starts.extend(i + 1 for i, c in enumerate(s) if c == "\n")
    return starts


def line_col(line_starts: list[int], index: int) -> tuple[int, int]:
    """Строка и столбец (с единицы) для смещения."""
    lo, hi = 0, len(line_starts) - 1
    while lo < hi:
        mid = (lo + hi + 1) >> 1
        if line_starts[mid] <= index:
            lo = mid
        else:
            hi = mid - 1
    return lo + 1, index - line_starts[lo] + 1


PART_RE = jsre("[\\p{L}\\p{N}]+", "gu")


def prepare(source: str) -> Prepared:
    invisible: list[Invisible] = []
    soft_hyphens: list[int] = []
    to_source: list[int] = []
    kept: list[str] = []
    for i, ch in enumerate(source):
        if ch in STRIPPED:
            if ch == SOFT_HYPHEN:
                soft_hyphens.append(i)
            elif (
                not (ch == BOM and i == 0)
                and not (ch == ZWJ and _joins_emoji(source, i))
                and not _legit_control(source, i)
            ):
                invisible.append(Invisible(i, ch))
            continue
        # Неразрывный дефис и U+2010 с клавиатуры не набрать, их ставят модели; для словарей это обычный дефис.
        kept.append("-" if ch in HYPHENS else ch)
        to_source.append(i)
    to_source.append(len(source))
    stripped = "".join(kept)

    # Подмена латиницы внутри кириллических слов. Части дефисного слова проверяются
    # по отдельности: в «HTTP-запрос» латинское сокращение стоит при русском слове законно.
    homoglyphs: list[Homoglyph] = []
    for m in WORD_RE.finditer(stripped):
        word = m.group()
        if not CYRILLIC_RE.search(word) or not LATIN_RE.search(word):
            continue
        start = m.start()
        found = False
        for part in PART_RE.finditer(word):
            s = part.group()
            if not CYRILLIC_RE.search(s) or not LATIN_RE.search(s) or not _spoofed(s):
                continue
            found = True
            # Буквы-двойники приводятся к алфавиту слова, чтобы словари видели настоящее слово.
            latin = sum(1 for c in s if LATIN_RE.match(c))
            to_latin = latin > sum(1 for c in s if CYRILLIC_RE.match(c))
            table = CYRILLIC_TO_LATIN if to_latin else LATIN_TO_CYRILLIC
            at = start + part.start()
            for k in range(len(s)):
                c = kept[at + k]
                if c in table:
                    kept[at + k] = table[c]
        if found:
            homoglyphs.append(Homoglyph(to_source[start], word))
    text = "".join(kept).replace("ё", "е").replace("Ё", "Е")
    no_code = _mask_code(text)
    prose = _mask_protected(no_code)
    return Prepared(
        source=source,
        text=text,
        no_code=no_code,
        prose=prose,
        to_source=to_source,
        invisible=invisible,
        soft_hyphens=soft_hyphens,
        homoglyphs=homoglyphs,
        line_starts=_line_starts(source),
    )


def words(s: str) -> list[str]:
    return WORD_RE.findall(s)


def plural(n: int, one: str, few: str, many: str) -> str:
    """«1 слово», «2 слова», «5 слов»."""
    d = n % 10
    dd = n % 100
    form = one if d == 1 and dd != 11 else few if 2 <= d <= 4 and (dd < 12 or dd > 14) else many
    return f"{n} {form}"


BlockKind = Literal["prose", "heading", "list", "table", "quote", "code", "empty"]


@dataclass(slots=True)
class Block:
    kind: BlockKind
    start: int
    end: int
    # Текст блока из `prose` (замаскированный).
    text: str


FENCE_RE = jsre("^\\s*(```|~~~|---)")
TABLE_OR_QUOTE_RE = jsre("^\\s*[|>]")
TABLE_RE = jsre("^\\s*\\|")
LIST_ITEM_RE = jsre("^\\s*([-*+•]|\\d+[.)])\\s+")
HEADING_RE = jsre("^#{1,6}\\s")


def blocks(p: Prepared) -> list[Block]:
    """Разбивает текст на блоки по пустым строкам; заголовок всегда отдельный блок."""
    out: list[Block] = []
    lines = p.prose.split("\n")
    raw_lines = p.text.split("\n")
    cur_start: int | None = None
    cur_lines: list[str] = []
    cur_raw: list[str] = []

    def flush() -> None:
        nonlocal cur_start
        if cur_start is None:
            return
        start = cur_start
        text = "\n".join(cur_lines)
        raw = "\n".join(cur_raw)
        kind: BlockKind = "prose"
        first_raw = cur_raw[0] if cur_raw else ""
        if trim(text) == "":
            kind = "code" if FENCE_RE.search(first_raw) or trim(raw) != "" else "empty"
            if TABLE_OR_QUOTE_RE.search(first_raw):
                kind = "table" if TABLE_RE.search(first_raw) else "quote"
        elif all(trim(line) == "" or LIST_ITEM_RE.search(line) for line in cur_lines):
            kind = "list"
        out.append(Block(kind, start, start + len(text), text))
        cur_start = None
        cur_lines.clear()
        cur_raw.clear()

    offset = 0
    for i, line in enumerate(lines):
        raw = raw_lines[i] if i < len(raw_lines) else ""
        if HEADING_RE.search(line):
            flush()
            out.append(Block("heading", offset, offset + len(line), line))
        elif trim(line) == "" and trim(raw) == "":
            flush()
        else:
            if cur_start is None:
                cur_start = offset
            cur_lines.append(line)
            cur_raw.append(raw)
        offset += len(line) + 1
    flush()
    return out


@dataclass(slots=True)
class Sentence:
    start: int
    end: int
    text: str
    words: int


ABBREVIATIONS = frozenset(
    (
        "т е д п г гг в вв с см рис табл др им ул проф акад вып стр ст н э тыс млн млрд руб коп мин сек ч кв обл "
        "ред изд пер сб т.е т.д т.п т.к etc et al no vol pp fig eq doi проч напр прим ср англ лат гл"
    ).split()
)

# Перенос строки внутри абзаца (ручной перенос в Markdown) предложение не заканчивает;
# заканчивают пустая строка, конец текста и начало пункта списка, цитаты или таблицы.
# Ретроспектива (?<![.!?…]) не даёт начать совпадение внутри серии знаков: иначе на
# длинном отточии движок откатывается с каждой позиции и время растёт квадратично.
SENTENCE_END_RE = jsre(
    '(?<![.!?…])[.!?…]+["»”)]*(?=\\s+["«„(—–-]?\\s*[\\p{Lu}\\d]|\\s*$)'
    "|\\n(?=[ \\t]*(?:\\n|$|[-*+•][ \\t]|\\d+[.)][ \\t]|[>|]))",
    "gu",
)
LAST_WORD_RE = jsre("([\\p{L}.]+)$", "u")
SINGLE_UPPER_RE = jsre("^\\p{Lu}$", "u")


def sentences(text: str, base: int = 0) -> list[Sentence]:
    """Разбивает фрагмент на предложения с учётом сокращений и инициалов."""
    out: list[Sentence] = []
    start = 0
    for m in SENTENCE_END_RE.finditer(text):
        idx = m.start()
        if m.group() != "\n":
            before = text[max(start, idx - 12) : idx]
            last = LAST_WORD_RE.search(before)
            last_word = last.group(1).lower() if last else ""
            if last_word.removesuffix(".") in ABBREVIATIONS or SINGLE_UPPER_RE.search(before[-1:]):
                continue
        end = m.end()
        _push_sentence(out, text, start, end, base)
        start = end
    _push_sentence(out, text, start, len(text), base)
    return out


def _push_sentence(out: list[Sentence], text: str, start: int, end: int, base: int) -> None:
    raw = text[start:end]
    trimmed_start = start + (len(raw) - len(trim_start(raw)))
    t = trim(raw)
    if not t:
        return
    n = len(words(t))
    if n == 0:
        return
    out.append(Sentence(base + trimmed_start, base + trimmed_start + len(t), t, n))


def mean(xs: Sequence[float]) -> float:
    return total(xs) / len(xs) if xs else 0


def cv(xs: Sequence[float]) -> float:
    """Коэффициент вариации (σ/μ)."""
    if len(xs) < 2:
        return 0
    m = mean(xs)
    if m == 0:
        return 0
    v = total((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(v) / m


def mattr(tokens: list[str], window: int = 100) -> float:
    """Скользящий TTR (MATTR) по окну: устойчив к длине текста."""
    if not tokens:
        return 0
    lower = [t.lower() for t in tokens]
    if len(lower) <= window:
        return len(set(lower)) / len(lower)
    counts: dict[str, int] = {}
    distinct = 0
    for t in lower[:window]:
        c = counts.get(t, 0)
        if c == 0:
            distinct += 1
        counts[t] = c + 1
    acc = distinct / window
    steps = 1
    for i in range(window, len(lower)):
        add, drop = lower[i], lower[i - window]
        cd = counts.get(drop, 0) - 1
        counts[drop] = cd
        if cd == 0:
            distinct -= 1
        ca = counts.get(add, 0)
        if ca == 0:
            distinct += 1
        counts[add] = ca + 1
        acc += distinct / window
        steps += 1
    return acc / steps

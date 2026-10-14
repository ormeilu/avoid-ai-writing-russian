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
CYRILLIC_TO_LATIN = {cyrillic: latin for latin, cyrillic in LATIN_TO_CYRILLIC.items()}

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


def _script_runs(part: str) -> list[_Run]:
    runs: list[_Run] = []
    for i, char in enumerate(part):
        latin = bool(ASCII_LATIN_RE.match(char))
        if not latin and not CYRILLIC_RE.match(char):
            continue
        if runs and runs[-1].latin == latin:
            runs[-1].end = i + 1
        else:
            runs.append(_Run(latin, i, i + 1))
    return runs


def _letters(part: str, run: _Run) -> list[str]:
    """Буквы серии без цифр и прочих знаков."""
    own = ASCII_LATIN_RE if run.latin else CYRILLIC_RE
    return [char for char in part[run.start : run.end] if own.match(char)]


def _lookalike(part: str, run: _Run) -> bool:
    """У каждой буквы серии есть двойник в другом алфавите."""
    table = LATIN_TO_CYRILLIC if run.latin else CYRILLIC_TO_LATIN
    return all(char in table for char in _letters(part, run))


def _spoofed(part: str) -> bool:
    """Подмена букв в части слова без дефисов.

    Алфавит слова выдаёт буква без двойника: «щ» в русском слове, «n» в английском.
    Подозрительны буквы-двойники другого алфавита. Если без двойника есть буквы
    обоих алфавитов (украинское слово с латинской i вместо і) или нет ни одной,
    подозрителен алфавит, в котором букв меньше. Законная смесь — латинская основа
    с русским окончанием («Pythonе», «PHPшник», «OKей») и слипшийся предлог
    («вPython»). Подмена — двойники внутри слова, в конце русского слова, латинские
    в начале русского слова и одна-две кириллические в начале английского.
    """
    runs = _script_runs(part)
    if len(runs) < 2:
        return False
    first, second, last = runs[0], runs[1], runs[-1]

    def letters_of(latin: bool) -> list[str]:
        return [char for run in runs if run.latin == latin for char in _letters(part, run)]

    latin_letters, cyrillic_letters = letters_of(True), letters_of(False)
    latin_anchored = any(char not in LATIN_TO_CYRILLIC for char in latin_letters)
    cyrillic_anchored = any(char not in CYRILLIC_TO_LATIN for char in cyrillic_letters)

    def suspect_script(latin: bool) -> bool:
        if latin_anchored != cyrillic_anchored:
            return latin == cyrillic_anchored
        if latin:
            return len(latin_letters) <= len(cyrillic_letters)
        return len(cyrillic_letters) <= len(latin_letters)

    def suspect(run: _Run) -> bool:
        return suspect_script(run.latin) and _lookalike(part, run)

    # Алфавиты чередуются, поэтому внутренняя серия зажата буквами другого алфавита.
    for k in range(1, len(runs) - 1):
        run = runs[k]
        # «FхG», «mхn»: одиночная «х» между одиночными буквами — знак умножения.
        times = (
            bool(TIMES_RE.search("".join(_letters(part, run))))
            and len(_letters(part, runs[k - 1])) == 1
            and len(_letters(part, runs[k + 1])) == 1
        )
        if not times and suspect(run):
            return True
    if last.latin and suspect(last):
        return True
    if not suspect(first):
        return False
    head = "".join(_letters(part, first))
    # Латинское сокращение перед русским суффиксом («OKей», «PHPшник»), но не заглавное русское слово.
    lower_next = second.start < len(part) and bool(LOWER_RE.match(part[second.start]))
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
# Редкие типографские символы: пробелы особой ширины, неразрывные пробелы и дефисы, знак минуса.
# Их ставят типографы, вёрстка и некоторые модели (o3 и GPT-5 — U+202F), с клавиатуры их не набрать.
# Это жёлтый сигнал светофора: «ИИ или типография?» решает пользователь. Формы — для plural.
TYPOGRAPHIC: dict[str, tuple[str, str, str]] = {
    chr(0x00A0): ("неразрывный пробел", "неразрывных пробела", "неразрывных пробелов"),
    chr(0x202F): ("узкий неразрывный пробел", "узких неразрывных пробела", "узких неразрывных пробелов"),
    chr(0x2002): ("полукруглая шпация", "полукруглые шпации", "полукруглых шпаций"),
    chr(0x2003): ("круглая шпация", "круглые шпации", "круглых шпаций"),
    chr(0x2004): ("шпация в треть круглой", "шпации в треть круглой", "шпаций в треть круглой"),
    chr(0x2005): ("шпация в четверть круглой", "шпации в четверть круглой", "шпаций в четверть круглой"),
    chr(0x2006): ("шпация в шестую круглой", "шпации в шестую круглой", "шпаций в шестую круглой"),
    chr(0x2007): ("цифровой пробел", "цифровых пробела", "цифровых пробелов"),
    chr(0x2008): ("пунктуационный пробел", "пунктуационных пробела", "пунктуационных пробелов"),
    chr(0x2009): ("тонкая шпация", "тонкие шпации", "тонких шпаций"),
    chr(0x200A): ("волосяная шпация", "волосяные шпации", "волосяных шпаций"),
    chr(0x205F): ("математический пробел", "математических пробела", "математических пробелов"),
    chr(0x2010): ("типографский дефис", "типографских дефиса", "типографских дефисов"),
    chr(0x2011): ("неразрывный дефис", "неразрывных дефиса", "неразрывных дефисов"),
    chr(0x2012): ("цифровое тире", "цифровых тире", "цифровых тире"),
    chr(0x2015): ("горизонтальная черта", "горизонтальные черты", "горизонтальных черт"),
    chr(0x2212): ("знак минуса", "знака минуса", "знаков минуса"),
    # Стрелки в прозе: в ответах чат-ботов часты, у людей почти не встречаются.
    chr(0x2192): ("стрелка вправо", "стрелки вправо", "стрелок вправо"),
    chr(0x2190): ("стрелка влево", "стрелки влево", "стрелок влево"),
    chr(0x2194): ("двусторонняя стрелка", "двусторонние стрелки", "двусторонних стрелок"),
    chr(0x21D2): ("двойная стрелка", "двойные стрелки", "двойных стрелок"),
}
EMOJI_BEFORE_ZWJ = jsre("(?:\\p{Extended_Pictographic}|\\p{Emoji_Modifier}|\\uFE0F)$", "u")
EMOJI_AFTER_ZWJ = jsre("^\\p{Extended_Pictographic}", "u")
RTL_RE = jsre("[\\p{Script=Hebrew}\\p{Script=Arabic}\\p{Script=Syriac}\\p{Script=Thaana}\\p{Script=Nko}]", "u")
MONGOLIAN_RE = jsre("\\p{Script=Mongolian}", "u")


def _joins_emoji(text: str, i: int) -> bool:
    """U+200D между частями эмодзи (👨 + 💻) — соединитель, а не вставка."""
    return bool(EMOJI_BEFORE_ZWJ.search(text[max(0, i - 2) : i])) and bool(EMOJI_AFTER_ZWJ.search(text[i + 1 : i + 3]))


def _legit_control(text: str, i: int) -> bool:
    """Знак направления рядом с ивритом или арабским и разделитель внутри монгольского слова — не вставка."""
    char = text[i]
    if char in BIDI:
        # Сам знак ALM (U+061C) относится к арабскому письму, поэтому смотрим только на соседей.
        return bool(RTL_RE.search(text[max(0, i - 3) : i] + " " + text[i + 1 : i + 4]))
    if char == MVS:
        return bool(MONGOLIAN_RE.search(text[i - 1 : i])) and bool(MONGOLIAN_RE.search(text[i + 1 : i + 2]))
    return False


@dataclass(slots=True)
class Invisible:
    index: int
    char: str


# Пробелы особой ширины → обычный пробел той же длины: записи списка литературы по ГОСТ
# («С. 5», «№ 3») набирают и с неразрывными пробелами, а правила служебных частей ждут обычный.
PLAIN_SPACES = {ord(char): " " for char in TYPOGRAPHIC if char.isspace()}


@dataclass(slots=True)
class Typographic:
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
    # Редкие типографские символы (TYPOGRAPHIC) в прозе, с позициями в исходнике. В коде, цитатах
    # и служебных частях статьи они не считаются: там это чужая вёрстка или формат записи.
    typographic: list[Typographic]
    homoglyphs: list[Homoglyph]
    line_starts: list[int]
    # Служебные части статьи в `text` (service_ranges): в `prose` замаскированы.
    service: list[tuple[int, int]]


def _mask(chars: list[str], start: int, end: int) -> None:
    for i in range(start, end):
        if chars[i] != "\n":
            chars[i] = " "


def _mask_all(chars: list[str], text: str, pattern: str, flags: str = "g") -> None:
    for match in jsre(pattern, flags).finditer(text):
        _mask(chars, match.start(), match.end())


FRONTMATTER_RE = jsre("^---\\n[\\s\\S]*?\\n---(?:\\n|$)")
FENCED_RE = jsre("^(```|~~~)[^\\n]*\\n[\\s\\S]*?^\\1[^\\n]*$", "gm")


def _mask_code(text: str) -> str:
    """Маскирует YAML-шапку, блоки кода и инлайн-код."""
    chars = list(text)
    frontmatter = FRONTMATTER_RE.search(text)
    if frontmatter:
        _mask(chars, 0, frontmatter.end())
    for match in FENCED_RE.finditer(text):
        _mask(chars, match.start(), match.end())
    _mask_all(chars, text, "`[^`\\n]+`")
    return "".join(chars)


# ─── Служебные части статьи ─────────────────────────────────────────────
# Список литературы, ключевые слова, английская аннотация и сведения об авторах — не
# проза статьи. В docx, собранном по ГОСТ и переведённом в Markdown (pandoc -t gfm),
# они идут в конце файла и без маскировки дают ложные находки: запись по ГОСТ Р 7.0.5
# «Журнал. — 2021. — Vol. 21, № 7. — P. 2417.» выглядит как рубленые фрагменты.

_OPEN = "(?:\\*\\*|__|\\*|_)?"
# Заголовок раздела: строка Markdown с # или отдельная строка, можно жирным и с номером.
_HEADING_LINE = f"^[ \\t]*(?:#{{1,6}}[ \\t]+)?{_OPEN}[ \\t]*(?:\\d{{1,2}}\\.?[ \\t]+)?(?:"
_HEADING_END = f")[ \\t]*[.:]?[ \\t]*{_OPEN}[ \\t]*[.:]?[ \\t]*$"
BIB_HEADING_RE = jsre(
    _HEADING_LINE
    + "список[ \\t]+(?:использованн\\p{L}*[ \\t]+)?(?:литературы|источников)(?:[ \\t]+и[ \\t]+литературы)?"
    "|библиографическ\\p{L}*[ \\t]+список|библиография|литература|источники"
    "|references|bibliography|literature|works[ \\t]+cited" + _HEADING_END,
    "iu",
)
AUTHORS_HEADING_RE = jsre(
    _HEADING_LINE + "(?:сведения|информация)[ \\t]+об[ \\t]+автор(?:е|ах)|об[ \\t]+автор(?:е|ах)"
    "|(?:information[ \\t]+)?about[ \\t]+the[ \\t]+authors?|authors?[ \\t]+information" + _HEADING_END,
    "iu",
)
KEYWORDS_RE = jsre(
    f"^[ \\t]*{_OPEN}[ \\t]*(?:ключевые[ \\t]+слова|keywords|key[ \\t]+words)[ \\t]*{_OPEN}[ \\t]*[:.—–-]", "iu"
)
ABSTRACT_RE = jsre(
    f"^[ \\t]*(?:#{{1,6}}[ \\t]+)?{_OPEN}[ \\t]*(?:abstract|annotation|summary)[ \\t]*{_OPEN}[ \\t]*([:.—–-]|$)", "iu"
)
MD_HEADING_RE = jsre("^#{1,6}[ \\t]")
# Строка-реквизит шапки или хвоста: УДК, ББК, «Для цитирования:», даты поступления и
# принятия. Строка целиком служебная, остальной абзац нет.
FRONT_LINE_RE = jsre(
    f"^[ \\t]*{_OPEN}[ \\t]*(?:(?:УДК|ББК)(?![\\p{{L}}\\d])|UDC(?![\\p{{L}}\\d])|BBK(?![\\p{{L}}\\d])"
    "|(?:поступил[аио]?[ \\t]+в[ \\t]+редакцию|принят[аоы]?[ \\t]+(?:к[ \\t]+публикации|в[ \\t]+печать)"
    "|received|accepted|revised)[ \\t]*[:.]?[ \\t]*(?:\\d|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\\p{L}*\\.?[ \\t]+\\d))",
    "iu",
)
# «Для цитирования: …» и «For citation: …» занимают абзац: это библиографическая запись статьи.
CITE_AS_RE = jsre(
    f"^[ \\t]*{_OPEN}[ \\t]*(?:для[ \\t]+цитирования|for[ \\t]+citation|how[ \\t]+to[ \\t]+cite|cite[ \\t]+as)[ \\t]*{_OPEN}[ \\t]*:",
    "iu",
)
# Библиографическая запись: номер или «Фамилия И. О.» в начале строки, год и хотя бы одна
# примета записи (//, «— С.», Vol., №, DOI, URL, ISBN, et al., «Город : Издательство, год»,
# «— Текст : электронный»). Номер может быть жирным: **12.**; pandoc экранирует точку после номера
# и скобки: 12\. и \[12\].
RECORD_NUMBER_RE = jsre(
    "^[ \\t]*(?:[-*+][ \\t]+)?(?:\\*\\*|__)?(?:\\\\?\\[)?\\d{1,3}\\\\?[.)\\]](?:\\*\\*|__)?[ \\t]+", "u"
)
RECORD_AUTHOR_RE = jsre("^[ \\t]*\\p{Lu}\\p{Ll}+(?:-\\p{Lu}\\p{Ll}+)?,?[ \\t]+\\p{Lu}\\.[ \\t]?(?:\\p{Lu}\\.)?", "u")
RECORD_YEAR_RE = jsre("(?<!\\d)(?:1[89]|20)\\d\\d(?!\\d)")
RECORD_MARK_RE = jsre(
    "(?<![:/])//(?!/)|[ \\t][—–][ \\t](?:(?:1[89]|20)\\d\\d|(?:[СсТт№]|[PpNn])\\.?[ \\t]|URL|DOI|Vol|Iss|(?:М|СПб|Л|Киев|Минск)\\.?[ \\t]?:"
    "|\\d{1,4}[ \\t]с\\.|Текст[ \\t]?[:.])"
    "|(?<!\\p{L})(?:Vol|Iss|pp|Pp|Bd)\\."
    "|№[ \\t]*\\d|(?<!\\p{L})(?:DOI|doi|URL|ISBN|ISSN)(?!\\p{L})|(?<!\\p{L})et[ \\t]+al\\."
    "|(?<!\\p{L})(?:[СсТт]|[Pp])\\.[ \\t]?\\d"
    "|[ \\t]:[ \\t]\\p{Lu}[^\\n:]{1,60},[ \\t]*(?:1[89]|20)\\d\\d"
    "|Электронный[ \\t]+ресурс|[ \\t]/[ \\t]\\p{Lu}\\.[ \\t]?(?:\\p{Lu}\\.[ \\t]?)?\\p{Lu}\\p{Ll}",
    "u",
)
RECORD_MAX_CHARS = 600
# Строка «Ключевые слова: …» не длиннее стольких знаков и с одной жирной подписью.
KEYWORDS_MAX_CHARS = 400
BOLD_LABEL_RE = jsre("\\*\\*[^*\\n]{1,40}:\\*\\*", "g")
# Строка списка литературы попроще записи по ГОСТ: «Архив полковника Хауза. Избранное. М., 2004.»
# Год и город издания, адрес или примета записи.
CITY_RE = jsre("(?<!\\p{L})(?:М|СПб|Л|Киев|Минск)\\.,?[ \\t]")
URL_LINE_RE = jsre("https?://")
# Сведения об авторах: строки не длиннее столько слов и не больше столько строк подряд. Заголовок
# без решётки («Об авторе») раздел открывает только во второй половине текста: в начале веб-страницы
# это пункт меню.
AUTHOR_LINE_WORDS = 40
AUTHOR_LINES = 40
# Английский заголовочный блок перед «Abstract» (название, авторы, организации): не больше
# столько абзацев, по столько слов в каждом.
ENGLISH_HEAD_PARAGRAPHS = 4
ENGLISH_HEAD_WORDS = 60
# Записей подряд, чтобы считать их списком литературы без заголовка.
RECORD_RUN = 3


def _reference_like(line: str) -> bool:
    """Строка похожа на позицию списка литературы, хотя и не по ГОСТ: год и город, адрес или примета записи."""
    if len(line) > RECORD_MAX_CHARS or not RECORD_YEAR_RE.search(line):
        return False
    return bool(RECORD_MARK_RE.search(line) or CITY_RE.search(line) or URL_LINE_RE.search(line))


def _is_record(line: str) -> bool:
    if len(line) > RECORD_MAX_CHARS or not (RECORD_NUMBER_RE.search(line) or RECORD_AUTHOR_RE.search(line)):
        return False
    return bool(RECORD_YEAR_RE.search(line)) and bool(RECORD_MARK_RE.search(line))


def letter_share(s: str) -> tuple[int, int]:
    """Кириллических и латинских букв в строке."""
    cyrillic = latin = 0
    for char in s:
        if CYRILLIC_RE.match(char):
            cyrillic += 1
        elif LATIN_RE.match(char):
            latin += 1
    return cyrillic, latin


def cyrillic_share(s: str) -> float | None:
    """Доля кириллицы среди кириллических и латинских букв; None, если букв нет."""
    cyrillic, latin = letter_share(s)
    return cyrillic / (cyrillic + latin) if cyrillic + latin else None


def service_ranges(s: str) -> list[tuple[int, int]]:
    """Отрезки [начало, конец) служебных частей: список литературы, ключевые слова,
    английская аннотация с названием и авторами над ней, сведения об авторах, строки УДК,
    «Для цитирования» и даты поступления. YAML-шапка и код сюда не входят, их маскирует
    _mask_code; строки внутри них заголовками не считаются.

    Раздел по заголовку идёт, пока строки похожи на его содержимое (записи, английский текст,
    короткие строки сведений об авторах, другие служебные строки), и кончается на первой строке
    прозы или на следующем заголовке Markdown. Раздел литературы без единой записи разделом не
    считается. Список литературы без заголовка — три записи подряд и больше. Отрезки отсортированы
    и не пересекаются; концы строк внутри них сохраняются при маскировке.
    """
    return _service_in(_mask_code(s))


def _service_in(masked: str) -> list[tuple[int, int]]:
    """service_ranges по тексту, где код и YAML-шапка уже замаскированы."""
    masked = masked.translate(PLAIN_SPACES)
    lines = masked.split("\n")
    line_starts = [0]
    for line in lines[:-1]:
        line_starts.append(line_starts[-1] + len(line) + 1)

    def end_of(line_index: int) -> int:
        return line_starts[line_index] + len(lines[line_index])

    def paragraph_end(line_index: int) -> int:
        """Последняя строка абзаца, который начинается со строки line_index."""
        while line_index + 1 < len(lines) and trim(lines[line_index + 1]):
            line_index += 1
        return line_index

    def english_head(abstract_line: int, floor: int) -> int:
        """Первая строка английского заголовочного блока (название, авторы, организации) над строкой abstract_line.

        Берутся абзацы с латиницей подряд, не выше строки floor, не длиннее ENGLISH_HEAD_WORDS слов.
        """
        first = abstract_line
        cursor = abstract_line - 1
        for _ in range(ENGLISH_HEAD_PARAGRAPHS):
            while cursor >= floor and not trim(lines[cursor]):
                cursor -= 1
            if cursor < floor:
                break
            top = cursor
            while top > floor and trim(lines[top - 1]):
                top -= 1
            part = "\n".join(lines[top : cursor + 1])
            share = cyrillic_share(part)
            if share is None or share >= 0.5 or len(words(part)) > ENGLISH_HEAD_WORDS:
                break
            if any(MD_HEADING_RE.search(paragraph_line) for paragraph_line in lines[top : cursor + 1]):
                break
            first = top
            cursor = top - 1
        return first

    def heading_kind(line: str) -> str | None:
        if BIB_HEADING_RE.search(line):
            return "bib"
        if AUTHORS_HEADING_RE.search(line):
            return "authors"
        return None

    def service_line(line: str) -> bool:
        return bool(
            FRONT_LINE_RE.search(line)
            or CITE_AS_RE.search(line)
            or KEYWORDS_RE.search(line)
            or ABSTRACT_RE.search(line)
        )

    def section(heading_line: int, kind: str) -> int | None:
        """Последняя строка раздела, который открывает заголовок в строке heading_line; None — раздела нет."""
        last, records, author_lines = heading_line, 0, 0
        cursor = heading_line + 1
        while cursor < len(lines):
            line = lines[cursor]
            if not trim(line):
                cursor += 1
                continue
            sub_kind = heading_kind(line)
            if MD_HEADING_RE.search(line) and sub_kind is None:
                break
            share = cyrillic_share(line)
            if sub_kind is not None:
                kind, author_lines = sub_kind, 0
            elif _is_record(line) or (kind == "bib" and _reference_like(line)):
                records += 1
            elif service_line(line) or (share is not None and share < 0.5):
                pass
            elif kind == "authors" and len(words(line)) <= AUTHOR_LINE_WORDS and author_lines < AUTHOR_LINES:
                author_lines += 1
            else:
                break
            last = cursor
            cursor += 1
        if last == heading_line or (kind == "bib" and not records and heading_kind(lines[heading_line]) == "bib"):
            return None
        return last

    ranges: list[tuple[int, int]] = []
    # Первая строка, с которой может начаться следующая служебная часть: за концом предыдущей.
    floor = 0
    line_index = 0
    while line_index < len(lines):
        line = lines[line_index]
        kind = heading_kind(line)
        md_heading = bool(MD_HEADING_RE.search(line))
        if kind == "authors" and not md_heading and line_index < len(lines) // 2:
            kind = None
        last = section(line_index, kind) if kind else None
        if last is not None:
            # Заголовок Markdown остаётся заголовком, строка-заголовок без решётки входит в часть:
            # иначе из неё получился бы абзац из двух слов.
            ranges.append((line_starts[line_index + 1 if md_heading else line_index], end_of(last)))
            line_index = floor = last + 1
            continue
        if FRONT_LINE_RE.search(line):
            ranges.append((line_starts[line_index], end_of(line_index)))
            line_index = floor = line_index + 1
            continue
        if CITE_AS_RE.search(line):
            last_line = paragraph_end(line_index)
            if end_of(last_line) - line_starts[line_index] <= RECORD_MAX_CHARS:
                ranges.append((line_starts[line_index], end_of(last_line)))
                line_index = floor = last_line + 1
                continue
        if KEYWORDS_RE.search(line) and len(line) <= KEYWORDS_MAX_CHARS and len(BOLD_LABEL_RE.findall(line)) <= 1:
            # Только сама строка: в текстах моделей за «Ключевыми словами» бывает проза без пустой строки,
            # а в одной строке с ними «**Тэги:** … **Совет:** …» — это след генерации, а не реквизит статьи.
            ranges.append((line_starts[line_index], end_of(line_index)))
            line_index = floor = line_index + 1
            continue
        abstract = ABSTRACT_RE.search(line)
        if abstract:
            # «Abstract» отдельной строкой — аннотация в следующем абзаце.
            first = line_index
            if not abstract.group(1):
                first = line_index + 1
                while first < len(lines) and not trim(lines[first]):
                    first += 1
            last_line = paragraph_end(first) if first < len(lines) else line_index
            share = cyrillic_share("\n".join(lines[first : last_line + 1]))
            if share is not None and share < 0.5:
                ranges.append((line_starts[english_head(line_index, floor)], end_of(last_line)))
                line_index = floor = last_line + 1
                continue
        if _is_record(line):
            run = [line_index]
            cursor = line_index + 1
            while cursor < len(lines):
                if not trim(lines[cursor]):
                    cursor += 1
                    continue
                if not _is_record(lines[cursor]):
                    break
                run.append(cursor)
                cursor += 1
            if len(run) >= RECORD_RUN:
                ranges.append((line_starts[run[0]], end_of(run[-1])))
                line_index = floor = run[-1] + 1
                continue
        line_index += 1
    # Раздел из одних пустых строк (заголовок, за ним сразу следующий заголовок) частью не считается.
    return [(start, end) for start, end in ranges if trim(masked[start:end])]


def blank(s: str, ranges: Sequence[tuple[int, int]]) -> str:
    """Заменяет отрезки пробелами, переводы строк оставляет: смещения и номера строк те же."""
    chars = list(s)
    for start, end in ranges:
        _mask(chars, start, end)
    return "".join(chars)


def model_text(source: str) -> str:
    """Текст для моделей: YAML-шапка, блоки кода и служебные части статьи заменены пробелами.

    Длина и переводы строк сохраняются, поэтому номера строк фрагментов совпадают с исходным файлом.
    Инлайн-код и цитаты остаются: без них предложение рвётся посередине.
    """
    ranges = list(service_ranges(source))
    frontmatter = FRONTMATTER_RE.search(source)
    if frontmatter:
        ranges.append((0, frontmatter.end()))
    ranges += [match.span() for match in FENCED_RE.finditer(source)]
    return blank(source, ranges)


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
        # Ссылки на литературу: [12], [3; 7], [12, с. 45]; после pandoc -t gfm скобки экранированы: \[12\].
        ("\\\\?\\[\\d+(?:[,;–-]\\s*\\d+)*(?:,\\s*с\\.\\s*\\d+(?:[–-]\\d+)?)?\\\\?\\]", "g"),
        # Ссылки pandoc: [@key], [@a; @b, с. 5].
        ("\\[-?@[^\\]\\n]+\\]", "g"),
    ):
        _mask_all(chars, no_code, pattern, flags)
    return "".join(chars)


def _line_starts(text: str) -> list[int]:
    starts = [0]
    starts.extend(index + 1 for index, char in enumerate(text) if char == "\n")
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
    # Найденные символы: позиция в исходнике, символ, позиция в нормализованном тексте.
    typographic_found: list[tuple[int, str, int]] = []
    to_source: list[int] = []
    kept: list[str] = []
    for index, char in enumerate(source):
        if char in STRIPPED:
            if char == SOFT_HYPHEN:
                soft_hyphens.append(index)
            elif (
                not (char == BOM and index == 0)
                and not (char == ZWJ and _joins_emoji(source, index))
                and not _legit_control(source, index)
            ):
                invisible.append(Invisible(index, char))
            continue
        if char in TYPOGRAPHIC:
            typographic_found.append((index, char, len(kept)))
        # Неразрывный дефис и U+2010 с клавиатуры не набрать, их ставят модели; для словарей это обычный дефис.
        kept.append("-" if char in HYPHENS else char)
        to_source.append(index)
    to_source.append(len(source))
    stripped = "".join(kept)

    # Подмена латиницы внутри кириллических слов. Части дефисного слова проверяются
    # по отдельности: в «HTTP-запрос» латинское сокращение стоит при русском слове законно.
    homoglyphs: list[Homoglyph] = []
    for word_match in WORD_RE.finditer(stripped):
        word = word_match.group()
        if not CYRILLIC_RE.search(word) or not LATIN_RE.search(word):
            continue
        start = word_match.start()
        found = False
        for part_match in PART_RE.finditer(word):
            part = part_match.group()
            if not CYRILLIC_RE.search(part) or not LATIN_RE.search(part) or not _spoofed(part):
                continue
            found = True
            # Буквы-двойники приводятся к алфавиту слова, чтобы словари видели настоящее слово.
            latin_count = sum(1 for char in part if LATIN_RE.match(char))
            to_latin = latin_count > sum(1 for char in part if CYRILLIC_RE.match(char))
            table = CYRILLIC_TO_LATIN if to_latin else LATIN_TO_CYRILLIC
            at = start + part_match.start()
            for k in range(len(part)):
                char = kept[at + k]
                if char in table:
                    kept[at + k] = table[char]
        if found:
            homoglyphs.append(Homoglyph(to_source[start], word))
    text = "".join(kept).replace("ё", "е").replace("Ё", "Е")
    no_code = _mask_code(text)
    service = _service_in(no_code)
    prose = blank(_mask_protected(no_code), service)
    # Маскировка заменяет знаки пробелами, поэтому символ на месте значит, что он в прозе.
    typographic = [
        Typographic(source_index, char)
        for source_index, char, text_index in typographic_found
        if prose[text_index] == ("-" if char in HYPHENS else char)
    ]
    return Prepared(
        source=source,
        text=text,
        no_code=no_code,
        prose=prose,
        to_source=to_source,
        invisible=invisible,
        soft_hyphens=soft_hyphens,
        typographic=typographic,
        homoglyphs=homoglyphs,
        line_starts=_line_starts(source),
        service=service,
    )


def words(s: str) -> list[str]:
    return WORD_RE.findall(s)


def plural(n: int, one: str, few: str, many: str) -> str:
    """«1 слово», «2 слова», «5 слов»; от 10 000 разряды через обычный пробел."""
    last_digit = n % 10
    last_two = n % 100
    form = (
        one
        if last_digit == 1 and last_two != 11
        else few
        if 2 <= last_digit <= 4 and (last_two < 12 or last_two > 14)
        else many
    )
    number = f"{n:,}".replace(",", " ") if abs(n) >= 10000 else str(n)
    return f"{number} {form}"


BlockKind = Literal["prose", "heading", "list", "table", "quote", "code", "service", "empty"]


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
    found: list[Block] = []
    lines = p.prose.split("\n")
    raw_lines = p.text.split("\n")
    current_start: int | None = None
    current_lines: list[str] = []
    current_raw: list[str] = []

    def flush() -> None:
        nonlocal current_start
        if current_start is None:
            return
        start = current_start
        text = "\n".join(current_lines)
        raw = "\n".join(current_raw)
        kind: BlockKind = "prose"
        first_raw = current_raw[0] if current_raw else ""
        if trim(text) == "":
            kind = "code" if FENCE_RE.search(first_raw) or trim(raw) != "" else "empty"
            if TABLE_OR_QUOTE_RE.search(first_raw):
                kind = "table" if TABLE_RE.search(first_raw) else "quote"
            elif kind == "code" and any(
                service_start <= start < service_end for service_start, service_end in p.service
            ):
                kind = "service"
        elif all(trim(line) == "" or LIST_ITEM_RE.search(line) for line in current_lines):
            kind = "list"
        found.append(Block(kind, start, start + len(text), text))
        current_start = None
        current_lines.clear()
        current_raw.clear()

    offset = 0
    for i, line in enumerate(lines):
        raw = raw_lines[i] if i < len(raw_lines) else ""
        if HEADING_RE.search(line):
            flush()
            found.append(Block("heading", offset, offset + len(line), line))
        elif trim(line) == "" and trim(raw) == "":
            flush()
        else:
            if current_start is None:
                current_start = offset
            current_lines.append(line)
            current_raw.append(raw)
        offset += len(line) + 1
    flush()
    return found


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
    found: list[Sentence] = []
    start = 0
    for match in SENTENCE_END_RE.finditer(text):
        index = match.start()
        if match.group() != "\n":
            before = text[max(start, index - 12) : index]
            last = LAST_WORD_RE.search(before)
            last_word = last.group(1).lower() if last else ""
            if last_word.removesuffix(".") in ABBREVIATIONS or SINGLE_UPPER_RE.search(before[-1:]):
                continue
        end = match.end()
        _push_sentence(found, text, start, end, base)
        start = end
    _push_sentence(found, text, start, len(text), base)
    return found


def _push_sentence(found: list[Sentence], text: str, start: int, end: int, base: int) -> None:
    raw = text[start:end]
    trimmed_start = start + (len(raw) - len(trim_start(raw)))
    trimmed = trim(raw)
    if not trimmed:
        return
    word_count = len(words(trimmed))
    if word_count == 0:
        return
    found.append(Sentence(base + trimmed_start, base + trimmed_start + len(trimmed), trimmed, word_count))


def mean(xs: Sequence[float]) -> float:
    return total(xs) / len(xs) if xs else 0


def cv(xs: Sequence[float]) -> float:
    """Коэффициент вариации (σ/μ)."""
    if len(xs) < 2:
        return 0
    average = mean(xs)
    if average == 0:
        return 0
    variance = total((x - average) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(variance) / average


def mattr(tokens: list[str], window: int = 100) -> float:
    """Скользящий TTR (MATTR) по окну: устойчив к длине текста."""
    if not tokens:
        return 0
    lowered = [token.lower() for token in tokens]
    if len(lowered) <= window:
        return len(set(lowered)) / len(lowered)
    counts: dict[str, int] = {}
    distinct = 0
    for token in lowered[:window]:
        count = counts.get(token, 0)
        if count == 0:
            distinct += 1
        counts[token] = count + 1
    ttr_sum = distinct / window
    steps = 1
    for i in range(window, len(lowered)):
        entering, leaving = lowered[i], lowered[i - window]
        leaving_count = counts.get(leaving, 0) - 1
        counts[leaving] = leaving_count
        if leaving_count == 0:
            distinct -= 1
        entering_count = counts.get(entering, 0)
        if entering_count == 0:
            distinct += 1
        counts[entering] = entering_count + 1
        ttr_sum += distinct / window
        steps += 1
    return ttr_sum / steps

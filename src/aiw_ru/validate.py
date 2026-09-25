"""Проверка сохранности: правка прозы не должна портить защищённое содержимое
и менять факты исходника.

Сравнивает исходник и исправленный текст и сообщает, что пропало или изменилось.

Разрешённые скиллом изменения не считаются нарушением: смена прямых кавычек на
«ёлочки», десятичной точки на запятую, Title Case в заголовке на обычный регистр,
удаление параметра utm_source ИИ-инструмента.

Факты. Идея взята из humanizer-ru (MIT, © 2026 Ilya Utov): правка не должна
снимать оговорки и добавлять то, чего в исходнике не было. Морфологии здесь нет,
поэтому проверки опираются на словари и регистр букв и настроены на точность:
лучше пропустить подмену, чем обвинить законную правку. Защищённое содержимое
(код, формулы, цитаты, URL, таблицы) в фактах не участвует, его сверяют точно.

- `оговорка` — правка сняла оговорку: «обычно», «как правило», «примерно»,
  «около 40», «от 5000», «до 1 ноября», «вероятно», «по-видимому», «может»,
  «не всегда», «в среднем» и подобные. Считаются не слова, а места: оговорки
  одного предложения, между которыми не больше трёх слов, образуют одну стопку.
  Поэтому «потенциально может» → «может» и «возможно, в перспективе сможет» →
  «может» не нарушение: скилл велит оставить одну модальность из стопки. Каждая
  стопка исходника ищет в правке стопку того же семейства (вероятность,
  частота, редкость, приближение, среднее, граница), сначала в самом похожем по
  словам предложении; «обычно» → «как правило» и «от 0,15 до 0,40» →
  «0,15–0,40» тоже не нарушение. Стопки внутри находок детектора (нагромождённая
  оговорка, туманный прогноз) и в предложениях, которые скилл вправе удалить
  целиком (след чат-бота, пустой вывод, размытая ссылка на авторитет и т. п.),
  не считаются. Не нашедшая пары стопка — нарушение. Если в стопке только «может»
  или «способен», это предупреждение: без морфологии не отличить возможность от
  умения, а «может быть использован» → «подходит» законная правка.
- `имя` — в правке появилось имя собственное или название: слово с заглавной
  не в начале предложения, строки или реплики после двоеточия, любое латинское
  слово (Python, PyTorch) или аббревиатура (ГОСТ, ВАК), которых нет в исходнике.
  Падежи сравниваются по общему началу слова: «Москва» и «Москве», «Павел» и
  «Павла», «ГОСТ» и «ГОСТа» — одно имя. Аббревиатура из первых букв слов
  исходника («искусственный интеллект» → «ИИ») не новая. «Вы» из вежливости
  именем не считается.
- `число словами` — появилось числительное, которого нет в исходнике ни словом,
  ни цифрами: «пятьдесят», «десятки», «тысяча», «третий», «втрое». Числа
  цифрами проверяет `число`. Мелкие («один», «два», «оба», «второй»,
  «половина», «вдвое») — предупреждение: они идиоматичны («с одной стороны»,
  «оба варианта»). «Во-вторых», «в-третьих» числом не считаются.
- `месяц` — появился месяц, которого в исходнике не было («в марте»).
- `утверждение` — слов-претензий стало больше: «впервые», «первый»,
  «единственный», «уникальный», «лучший», «всегда», «никогда», «все», «никто»,
  «гарантирует», «доказали», «самый», «крупнейший», «рекордный». Всегда
  предупреждение: у этих слов много законных оборотов («первый шаг», «всё
  равно»). Устойчивые обороты («в первую очередь», «на первый взгляд», «всё
  ещё», «тем самым», «в лучшем случае») и отрицания («не всегда», «не все»)
  не считаются.

Нарушения делают `ok` ложным (код выхода 1), предупреждения — нет: их стоит
проверить глазами.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass, field

import regex
from pydantic import Field

from aiw_ru.compat import jsre
from aiw_ru.detect import AI_URL_RE, analyze
from aiw_ru.lexicon import phrase
from aiw_ru.text import WORD_END, WORD_START, Prepared, Sentence, blocks, prepare, sentences
from aiw_ru.types import ContextMode, Issue, Record


class Violation(Record):
    kind: str
    detail: str


class ValidationResult(Record):
    ok: bool
    violations: list[Violation]
    # Возможные подмены фактов, которые код выхода не меняют: проверить глазами.
    warnings: list[Violation] = Field(default_factory=list)
    issues_before: int
    issues_after: int


FRONTMATTER_RE = jsre("^---\\n[\\s\\S]*?\\n---(?:\\n|$)")
FENCE_RE = jsre("^(```|~~~)[^\\n]*\\n[\\s\\S]*?^\\1[^\\n]*$", "gm")
INLINE_CODE_RE = jsre("`[^`\\n]+`", "g")
FORMULA_RE = jsre("\\$\\$[\\s\\S]*?\\$\\$|\\$[^$\\n]+\\$", "g")
URL_RE = jsre('https?:\\/\\/[^\\s)>\\]»"]+', "g")
URL_TAIL_RE = jsre("[?&]$")
LOOSE_URL_RE = jsre("https?:\\/\\/\\S+", "g")
NUMBER_RE = jsre("\\d+(?:[.,]\\d+)*", "g")
CITATION_RE = jsre("\\[\\d+(?:[,;–-]\\s*\\d+)*(?:,\\s*с\\.\\s*\\d+(?:[–-]\\d+)?)?\\]", "g")
PANDOC_RE = jsre("\\[-?@[^\\]\\n]+\\]", "g")
TABLE_RE = jsre("^[ \\t]*\\|.*$", "gm")
BLOCKQUOTE_RE = jsre("^[ \\t]*>.*$", "gm")
QUOTE_RE = jsre('«[^«»\\n]{1,400}»|"[^"\\n]{1,400}"|“[^“”\\n]{1,400}”', "g")
QUOTE_MARKS_RE = jsre('^["“„«]|["”“»]$', "g")
HEADING_SHAPE_RE = jsre("^#{1,6}(?=\\s)", "gm")

# Сколько сообщений одного вида показывать.
LIMIT = 5


def _all(pattern, s: str) -> list[str]:
    return [m.group() for m in pattern.finditer(s)]


def _frontmatter(s: str) -> str:
    m = FRONTMATTER_RE.search(s)
    return m.group() if m else ""


def _strip_code(s: str) -> str:
    return INLINE_CODE_RE.sub("", FENCE_RE.sub("", s))


def _urls(s: str) -> list[str]:
    return [URL_TAIL_RE.sub("", AI_URL_RE.sub("", u), count=1) for u in _all(URL_RE, _strip_code(s))]


def _numbers(s: str) -> list[str]:
    prose = LOOSE_URL_RE.sub("", _strip_code(s))
    return [n.replace(",", ".") for n in _all(NUMBER_RE, prose)]


def _multiset_diff(a: list[str], b: list[str]) -> tuple[list[str], list[str]]:
    count = Counter(a)
    added: list[str] = []
    for x in b:
        if count[x] > 0:
            count[x] -= 1
        else:
            added.append(x)
    missing = [x for x, c in count.items() for _ in range(c)]
    return missing, added


def _compare_exact(kind: str, before: list[str], after: list[str], out: list[Violation]) -> None:
    missing, added = _multiset_diff(before, after)
    out.extend(Violation(kind=kind, detail=f"пропало или изменено: {m[:120]}") for m in missing[:LIMIT])
    out.extend(Violation(kind=kind, detail=f"появилось: {a[:120]}") for a in added[:LIMIT])


# ─── Словари фактов ──────────────────────────────────────────────────────

WS = "[\\s\\u00A0]+"
# Слово не после дефиса: «во-вторых», «в-третьих» — не числа.
NO_HYPHEN = "(?<!-)"
# Не после отрицания «не»: «не всегда», «не может».
NOT_NEGATED = f"(?<!(?:^|[^\\p{{L}}])не{WS})"


@dataclass(frozen=True, slots=True)
class Numeral:
    key: str
    forms: str
    # Значение для сверки с числами цифрами; None — приблизительное («десятки»).
    value: float | None
    soft: bool = False


def _ordinal(stem: str) -> str:
    return f"{stem}(ый|ой|ая|ое|ые|ого|ому|ым|ыми|ых|ую|ом)"


def _cardinal(stem: str) -> str:
    return f"{stem}(ь|и|ью)"


_TEENS = ("одиннадцат", "двенадцат", "тринадцат", "четырнадцат", "пятнадцат")
_TEENS2 = ("шестнадцат", "семнадцат", "восемнадцат", "девятнадцат")
_HUNDREDS = (("пятьсот", "пяти", 500), ("шестьсот", "шести", 600), ("семьсот", "семи", 700))
_HUNDREDS2 = (("восемьсот", "восьми", 800), ("девятьсот", "девяти", 900))

NUMERALS: list[Numeral] = [
    Numeral("один", "(один|одна|одно|одного|одной|одному|одним|одною|одну|одном)", 1, soft=True),
    Numeral("два", "(два|две|двух|двум|двумя|оба|обе|обоих|обеих|обоим|обеим|обоими|обеими)", 2, soft=True),
    Numeral("три", "(три|трех|трем|тремя)", 3),
    Numeral("четыре", "(четыре|четырех|четырем|четырьмя)", 4),
    Numeral("пять", _cardinal("пят"), 5),
    Numeral("шесть", _cardinal("шест"), 6),
    Numeral("семь", "(семь|семи)", 7),
    Numeral("восемь", "(восемь|восьми|восемью|восьмью)", 8),
    Numeral("девять", _cardinal("девят"), 9),
    Numeral("десять", _cardinal("десят"), 10),
    *(Numeral(s + "ь", _cardinal(s), 11 + i) for i, s in enumerate(_TEENS + _TEENS2)),
    Numeral("двадцать", _cardinal("двадцат"), 20),
    Numeral("тридцать", _cardinal("тридцат"), 30),
    Numeral("сорок", "(сорок|сорока)", 40),
    Numeral("пятьдесят", "(пятьдесят|пятидесяти|пятьюдесятью)", 50),
    Numeral("шестьдесят", "(шестьдесят|шестидесяти|шестьюдесятью)", 60),
    Numeral("семьдесят", "(семьдесят|семидесяти|семьюдесятью)", 70),
    Numeral("восемьдесят", "(восемьдесят|восьмидесяти|восемьюдесятью|восьмьюдесятью)", 80),
    Numeral("девяносто", "(девяносто|девяноста)", 90),
    Numeral("сто", "(сто|ста)", 100),
    Numeral("двести", "(двести|двухсот|двумстам|двумястами|двухстах)", 200),
    Numeral("триста", "(триста|трехсот|тремстам|тремястами|трехстах)", 300),
    Numeral("четыреста", "(четыреста|четырехсот|четыремстам|четырьмястами|четырехстах)", 400),
    *(Numeral(n, f"({n}|{o}сот|{o}стам|{o}стах)", v) for n, o, v in _HUNDREDS + _HUNDREDS2),
    Numeral("тысяча", "тысяч(а|и|е|у|ей|ею|ам|ами|ах)?", 1000),
    Numeral("миллион", "миллион*", 1e6),
    Numeral("миллиард", "миллиард*", 1e9),
    Numeral("триллион", "триллион*", 1e12),
    Numeral("десяток", "десят(ок|ка|ку|ком|ке|ки|ков|кам|ками|ках)", None),
    Numeral("сотня", "(сотня|сотни|сотне|сотню|сотней|сотнею|сотням|сотнями|сотнях|сотен)", None),
    Numeral("дюжина", "дюжин*", 12),
    Numeral("полтора", "(полтора|полторы|полутора)", 1.5, soft=True),
    Numeral("половина", "половин(а|ы|е|у|ой|ою)", 0.5, soft=True),
    Numeral("треть", "трет(ь|и|ью)", None),
    Numeral("четверть", "четверт(ь|и|ью)", 0.25),
    Numeral("вдвое", "(вдвое|дважды)", 2, soft=True),
    Numeral("втрое", "(втрое|трижды)", 3),
    Numeral("вчетверо", "(вчетверо|четырежды)", 4),
    Numeral("впятеро", "впятеро", 5),
    Numeral("вдесятеро", "вдесятеро", 10),
    Numeral("второй", _ordinal("втор"), 2, soft=True),
    Numeral("третий", "треть(ий|я|е|и|его|ему|им|ими|их|ю|ем|ей)", 3),
    Numeral("четвертый", _ordinal("четверт"), 4),
    Numeral("пятый", _ordinal("пят"), 5),
    Numeral("шестой", _ordinal("шест"), 6),
    Numeral("седьмой", _ordinal("седьм"), 7),
    Numeral("восьмой", _ordinal("восьм"), 8),
    Numeral("девятый", _ordinal("девят"), 9),
    Numeral("десятый", _ordinal("десят"), 10),
    *(Numeral(s + "ый", _ordinal(s), 11 + i) for i, s in enumerate(_TEENS + _TEENS2)),
    Numeral("двадцатый", _ordinal("двадцат"), 20),
    Numeral("тридцатый", _ordinal("тридцат"), 30),
    Numeral("сороковой", _ordinal("сороков"), 40),
    Numeral("пятидесятый", _ordinal("пятидесят"), 50),
    Numeral("шестидесятый", _ordinal("шестидесят"), 60),
    Numeral("семидесятый", _ordinal("семидесят"), 70),
    Numeral("восьмидесятый", _ordinal("восьмидесят"), 80),
    Numeral("девяностый", _ordinal("девяност"), 90),
    Numeral("сотый", _ordinal("сот"), 100),
    Numeral("тысячный", _ordinal("тысячн"), 1000),
]

MONTHS: list[tuple[str, str]] = [
    ("январь", "январ(ь|я|е|ю|ем)"),
    ("февраль", "феврал(ь|я|е|ю|ем)"),
    ("март", "март(а|е|у|ом)?"),
    ("апрель", "апрел(ь|я|е|ю|ем)"),
    ("май", "(май|мая|мае|маю|маем)"),
    ("июнь", "июн(ь|я|е|ю|ем)"),
    ("июль", "июл(ь|я|е|ю|ем)"),
    ("август", "август(а|е|у|ом)?"),
    ("сентябрь", "сентябр(ь|я|е|ю|ем)"),
    ("октябрь", "октябр(ь|я|е|ю|ем)"),
    ("ноябрь", "ноябр(ь|я|е|ю|ем)"),
    ("декабрь", "декабр(ь|я|е|ю|ем)"),
]

# Число после оговорки-границы: цифра или числительное («около 40», «до десяти»).
_NUMERAL_BODY = "|".join(n.forms.replace("*", "\\p{L}*") for n in NUMERALS)
AHEAD_NUMBER = f"(?={WS}(?:[~≈]?\\d|(?:{_NUMERAL_BODY}|нескольк\\p{{L}}*){WORD_END}))"
_NUMBER_WORD = f"(?:\\d+(?:[.,]\\d+)*|(?:{_NUMERAL_BODY}){WORD_END})"
_DASH = "[\\s\\u00A0]*[–—-][\\s\\u00A0]*"


def _word(p: str, before: str = "", after: str = "") -> str:
    return f"{before}{phrase(p)}{after}"


# Семейства оговорок: замена внутри семейства законна («обычно» → «как правило»).
HEDGES: list[tuple[str, str]] = [
    # Границы и интервалы — раньше одиночных «от», «до», чтобы интервал был одним местом.
    ("граница", f"{WORD_START}от{WS}{_NUMBER_WORD}{WS}до{WS}{_NUMBER_WORD}"),
    ("граница", f"(?<![\\d.,])\\d+(?:[.,]\\d+)?{_DASH}\\d+(?:[.,]\\d+)?"),
    ("граница", f"{WORD_START}(?:{_NUMERAL_BODY}){_DASH}(?:{_NUMERAL_BODY}){WORD_END}"),
    ("граница", _word("между", after=AHEAD_NUMBER)),
    (
        "граница",
        _word(
            "(не )?(менее|более|меньше|больше|ниже|выше|свыше)( чем)?|(как )?минимум|максимум|вплоть до|до|от",
            after=AHEAD_NUMBER,
        ),
    ),
    ("приближение", _word("около|порядка|где-то|в районе", after=AHEAD_NUMBER)),
    ("приближение", _word("примерно|приблизительно|ориентировочно|почти|практически|плюс-минус")),
    ("среднее", _word("в среднем")),
    (
        "частота",
        _word("обычно|как правило|в большинстве случаев|чаще всего|часто|зачастую|нередко|преимущественно"),
    ),
    (
        "редкость",
        _word(
            "иногда|порой|подчас|временами|время от времени|в (ряде|некоторых|отдельных) случа(ев|ях)|"
            "в части (случаев|сценариев)|не всегда|не обязательно|необязательно|не во всех случаях|редко|изредка"
        ),
    ),
    (
        "вероятность",
        _word(
            "вероятно|вероятнее всего|по всей вероятности|вероятн(ый|ая|ое|ые|ого|ой|ую|ым|ых)|по-видимому|"
            "видимо|по всей видимости|судя по всему|скорее всего|предположительно|наверное|наверно|пожалуй|"
            "возможно|не исключено|по (предварительным |некоторым |нашим )?оценкам|ожида(ется|ются)|"
            "(мне |нам )?кажется,|похоже,|по-моему|на мой взгляд|по (моему|нашему) мнению|кмк|имхо|"
            "возмож(ен|на|ны)|применим(а|о|ы)?"
        ),
    ),
    # Модальность «может»: без морфологии не отличить возможность от умения.
    (
        "может",
        _word(
            "могу|можем|можете|может|могут|мог|могла|могло|могли|смогу|сможем|сможете|сможет|смогут|"
            "способ(ен|на|но|ны)",
            before=NOT_NEGATED,
            after=f"(?!{WS}показаться)",
        ),
    ),
]

# Семейства, которые вместе считаются одной модальностью: «может помочь» → «возможно, поможет».
FAMILY_GROUP = {"может": "вероятность"}

# Находки, при которых скилл вправе удалить предложение целиком.
REMOVABLE = frozenset(
    (
        "chatbot sycophancy cutoff placeholder future-narrative generic-conclusion significance "
        "vague-attribution lets reasoning acknowledgment narrated-candor hook social-closer launch-intro "
        "speculative-opener vague-endorsement novelty false-concession"
    ).split()
)

# Устойчивые обороты со словами-претензиями: претензией не считаются.
CLAIM_IDIOMS = jsre(
    "|".join(
        phrase(p)
        for p in (
            "в первую очередь",
            "на первый взгляд",
            "первым делом",
            "первое время",
            "все (равно|еще|же|больше|меньше|чаще|реже)",
            "все-таки",
            "как всегда",
            "тем самым",
            "самое время",
            "в лучшем случае",
            "прежде всего",
            "лучше всего",
        )
    ),
    "giu",
)

CLAIMS: list[tuple[str, str]] = [
    ("впервые", "впервые"),
    ("первый", "перв(ый|ая|ое|ые|ого|ой|ому|ым|ыми|ых|ую|ом)"),
    ("единственный", "единствен*"),
    ("уникальный", "уникальн*"),
    ("лучший", "(наи)?лучш(ий|ая|ее|ие|его|ей|ему|им|ими|их|ую|ем)"),
    ("всегда", "всегда|навсегда"),
    ("никогда", "никогда"),
    ("все", "все|всех|всем|всеми|весь|вся|всю"),
    ("никто", "никто|никого|никому|никем"),
    ("гарантирует", "гарантир*|гаранти(я|и|ю|ей|ям|ями|ях)"),
    (
        "доказано",
        "доказал|доказала|доказало|доказали|доказан|доказана|доказано|доказаны|доказанн*|доказыва*|доказать|докаж*",
    ),
    ("самый", "сам(ый|ая|ые|ых|ую|ыми)"),
    (
        "крупнейший",
        "крупнейш*|величайш*|сильнейш*|мощнейш*|новейш*|важнейш*|наивысш*|наибольш*|наименьш*",
    ),
    ("рекордный", "рекордн*|беспрецедентн*|непревзойденн*"),
]

# Вежливое «Вы» с заглавной — не имя.
POLITE = frozenset("вы вас вам вами ваш ваша ваше ваши вашего вашей вашему вашим вашими ваших вашу вашем".split())
# Одна сущность под разными именами.
ALIASES = {"ai": ("ии",), "ии": ("ai",)}


def _combined(entries: list[tuple[str, str]], flags: str = "giu") -> regex.Pattern[str]:
    """Один проход по тексту: у каждой записи своя именованная группа."""
    return jsre("|".join(f"(?P<g{i}>{src})" for i, (_, src) in enumerate(entries)), flags)


# Оговорки ищутся каждая своим выражением: общее выражение из всех семейств в разы медленнее.
HEDGE_RES = [jsre(src, "giu") for _, src in HEDGES]
NUMERAL_RE = _combined([(n.key, NO_HYPHEN + phrase(n.forms)) for n in NUMERALS])
MONTH_RE = _combined([(k, phrase(f)) for k, f in MONTHS])
CLAIM_RE = _combined([(k, NO_HYPHEN + NOT_NEGATED + phrase(f)) for k, f in CLAIMS])
LETTERS_RE = jsre("\\p{L}+", "gu")
PART_RE = jsre("[\\p{L}\\p{N}]+", "gu")
LATIN_WORD_RE = jsre("\\p{Script=Latin}", "u")
# Беглая гласная перед последней согласной: «Павел», «Орёл», «Египет» (после ё→е).
FLEETING_RE = jsre("\\p{L}{2,}[ео][бвгджзклмнпрстфхцчшщ]$", "u")
# Аббревиатура: две и больше заглавных в начале слова («ГОСТ», «ГОСТа», «ИИ»).
UPPER_HEAD_RE = jsre("\\p{Lu}{2,}", "u")
# После двоеточия, точки с запятой и открывающей скобки или кавычки бывает заглавная
# у обычного слова («Итог: Модель…»); такие места считаются началом предложения.
OPENER_RE = jsre('[:;(«„“"][^\\p{L}\\p{N}\\n]*', "gu")
# После точки, «!», «?» и многоточия: разбивка на предложения детектора осторожнее
# («RLDD. Во-вторых» она принимает за инициал), а здесь лишнее начало безопасно.
AFTER_STOP_RE = jsre("[.!?…]+[^\\p{L}\\p{N}\\n]*", "gu")
LINE_LEAD_RE = jsre("(?:^|\\n)[^\\p{L}\\p{N}\\n]*(?:\\d{1,3}[.)][^\\p{L}\\p{N}\\n]*)?", "gu")


def _scan(patterns: list[regex.Pattern[str]], s: str) -> list[tuple[int, int, int]]:
    """Совпадения нескольких выражений без пересечений, как у одного выражения с альтернативами:
    левое раньше, при равном начале — выражение из списка раньше. Возвращает (начало, конец, номер).
    """
    found = sorted((m.start(), k, m.end()) for k, rx in enumerate(patterns) for m in rx.finditer(s))
    out: list[tuple[int, int, int]] = []
    for start, k, end in found:
        if out and start < out[-1][1]:
            continue
        out.append((start, end, k))
    return out


def _sentences(p: Prepared) -> list[Sentence]:
    """Предложения прозы, заголовков и списков, как их делит детектор: по блокам."""
    kinds = ("prose", "heading", "list")
    return [s for b in blocks(p) if b.kind in kinds for s in sentences(b.text, b.start)]


def _index(m: regex.Match[str]) -> int:
    name = m.lastgroup
    if name is None or not name.startswith("g"):  # pragma: no cover — группы всегда именованные
        name = next(k for k, v in m.groupdict().items() if v is not None)
    return int(name[1:])


def _surface(p: Prepared, start: int, end: int) -> str:
    """Фрагмент исходного текста (с «ё» и без маскирования) по смещениям в `p.prose`."""
    if start >= end:
        return ""
    return p.source[p.to_source[start] : p.to_source[end - 1] + 1]


def _mask(s: str, pattern: regex.Pattern[str]) -> str:
    return pattern.sub(lambda m: " " * len(m.group()), s)


# ─── Оговорки ────────────────────────────────────────────────────────────


def _group(family: str) -> str:
    return FAMILY_GROUP.get(family, family)


@dataclass(slots=True)
class _Hedge:
    start: int
    end: int
    family: str
    first: int
    last: int
    sentence: int


@dataclass(slots=True)
class _Stack:
    hedges: list[_Hedge] = field(default_factory=list)

    @property
    def families(self) -> set[str]:
        return {_group(h.family) for h in self.hedges}

    @property
    def only_modal(self) -> bool:
        return all(h.family == "может" for h in self.hedges)


@dataclass(slots=True)
class _Doc:
    p: Prepared
    sentence_starts: list[int]
    sentence_ends: list[int]
    sentence_words: list[set[str]]
    stacks: list[_Stack]


def _hedge_doc(p: Prepared) -> _Doc:
    prose = p.prose
    sents = _sentences(p) or [Sentence(0, len(prose), prose, 0)]
    starts = [s.start for s in sents]
    ends = [s.end for s in sents]
    words = [{w.lower() for w in PART_RE.findall(s.text)} for s in sents]
    token_starts: list[int] = []
    token_ends: list[int] = []
    for m in PART_RE.finditer(prose):
        token_starts.append(m.start())
        token_ends.append(m.end())
    hedges: list[_Hedge] = []
    for start, end, k in _scan(HEDGE_RES, prose):
        first = bisect_right(token_ends, start)
        last = max(first, bisect_left(token_starts, end) - 1)
        sentence = max(0, bisect_right(starts, start) - 1)
        hedges.append(_Hedge(start, end, HEDGES[k][0], first, last, sentence))
    stacks: list[_Stack] = []
    for h in hedges:
        prev = stacks[-1].hedges[-1] if stacks else None
        if prev is not None and prev.sentence == h.sentence and h.first - prev.last - 1 <= 3:
            stacks[-1].hedges.append(h)
        else:
            stacks.append(_Stack([h]))
    return _Doc(p, starts, ends, words, stacks)


def _exempt(doc: _Doc, stack: _Stack, issues: list[Issue]) -> bool:
    """Стопка внутри находки детектора или в предложении, которое скилл вправе удалить."""
    p = doc.p
    for h in stack.hedges:
        a, b = p.to_source[h.start], p.to_source[h.end]
        s_start = p.to_source[doc.sentence_starts[h.sentence]]
        s_end = p.to_source[doc.sentence_ends[h.sentence]]
        for issue in issues:
            i_start, i_end = issue.index, issue.index + len(issue.text)
            if i_start < b and a < i_end:
                return True
            if issue.type in REMOVABLE and s_start <= i_start < s_end:
                return True
    return False


def _match(before: list[tuple[set[str], set[str]]], after: list[tuple[str, set[str]]]) -> list[int]:
    """Паросочетание стопок исходника с оговорками правки; возвращает номера стопок без пары.

    Стопке исходника (семейства, слова предложения) подходит любая оговорка правки (семейство,
    слова предложения) из её семейств, и каждая оговорка правки закрывает одну стопку: так
    правка может свести две оговорки в одну стопку. Кандидаты перебираются от самого похожего
    по словам предложения, чтобы сообщение указывало на то место, где оговорка пропала.
    """

    def similarity(a: set[str], b: set[str]) -> float:
        return len(a & b) / len(a | b) if a or b else 0

    options = [
        sorted(
            (j for j, (family, _) in enumerate(after) if family in families),
            key=lambda j, w=words: -similarity(w, after[j][1]),
        )
        for families, words in before
    ]
    owner: dict[int, int] = {}

    def augment(i: int, seen: set[int]) -> bool:
        for j in options[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owner or augment(owner[j], seen):
                owner[j] = i
                return True
        return False

    return [i for i in range(len(before)) if not augment(i, set())]


def _snippet(doc: _Doc, h: _Hedge, width: int = 100) -> str:
    p = doc.p
    s_start, s_end = doc.sentence_starts[h.sentence], doc.sentence_ends[h.sentence]
    lo = max(s_start, min(h.start - width // 2, s_end - width))
    hi = min(s_end, lo + width)
    text = " ".join(_surface(p, lo, hi).split())
    return ("…" if lo > s_start else "") + text + ("…" if hi < s_end else "")


def _check_hedges(
    bp: Prepared, ap: Prepared, issues: list[Issue], violations: list[Violation], warnings: list[Violation]
) -> None:
    b, a = _hedge_doc(bp), _hedge_doc(ap)
    stacks = [s for s in b.stacks if not _exempt(b, s, issues)]
    before = [(s.families, b.sentence_words[s.hedges[0].sentence]) for s in stacks]
    after = [(_group(h.family), a.sentence_words[h.sentence]) for s in a.stacks for h in s.hedges]
    for i in _match(before, after):
        stack = stacks[i]
        h = stack.hedges[0]
        detail = f"пропала «{' '.join(_surface(bp, h.start, h.end).split())}»: {_snippet(b, h)}"
        (warnings if stack.only_modal else violations).append(Violation(kind="оговорка", detail=detail))


# ─── Имена, числа словами, месяцы, утверждения ──────────────────────────


def _starts(p: Prepared) -> set[int]:
    """Смещения слов, у которых заглавная буква ничего не говорит: начала предложений и строк."""
    prose = p.prose
    out: set[int] = set()
    for s in _sentences(p):
        m = LETTERS_RE.search(prose, s.start, s.end)
        if m:
            out.add(m.start())
    for pattern in (LINE_LEAD_RE, OPENER_RE, AFTER_STOP_RE):
        for m in pattern.finditer(prose):
            out.add(m.end())
    return out


def _common_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def _same_stem(a: str, b: str) -> bool:
    """Одно слово в разных падежах: общее начало без последних двух букв короткого слова."""
    short = min(len(a), len(b))
    cp = _common_prefix(a, b)
    if short <= 3:
        return cp == short and max(len(a), len(b)) - short <= 3
    return cp >= max(3, short - 2)


class _Vocabulary:
    """Слова исходника (с кодом, цитатами и URL) для сверки имён.

    Для слов с беглой гласной в конце основы хранится и форма без неё: «Орёл» → «орл»,
    чтобы «Орла» совпало по началу.
    """

    def __init__(self, text: str) -> None:
        self.sequence = [w.lower() for w in LETTERS_RE.findall(text)]
        self.words = set(self.sequence)
        self.buckets: dict[str, list[str]] = {}
        for w in self.words:
            forms = [w, w[:-2] + w[-1]] if FLEETING_RE.search(w) else [w]
            for f in forms:
                self.buckets.setdefault(f[:2], []).append(f)

    def covers(self, word: str) -> bool:
        if word in self.words or any(a in self.words for a in ALIASES.get(word, ())):
            return True
        return any(_same_stem(word, w) for w in self.buckets.get(word[:2], ()))

    def acronym(self, letters: str) -> bool:
        """Аббревиатура из первых букв подряд идущих слов: «искусственный интеллект» → «ИИ»."""
        n = len(letters)
        if not 2 <= n <= 6:
            return False
        seq = self.sequence
        return any(all(seq[i + k][0] == letters[k] for k in range(n)) for i in range(len(seq) - n + 1))


def _check_names(bp: Prepared, ap: Prepared, out: list[Violation]) -> None:
    vocab = _Vocabulary(bp.text)
    starts = _starts(ap)
    seen: set[str] = set()
    added: list[str] = []
    for m in LETTERS_RE.finditer(ap.prose):
        w = m.group()
        if len(w) < 2:
            continue
        low = w.lower()
        head = UPPER_HEAD_RE.match(w)
        abbreviation = head is not None
        latin = bool(LATIN_WORD_RE.search(w))
        if not (latin or abbreviation or (w[0].isupper() and m.start() not in starts)):
            continue
        if low in POLITE or low in seen:
            continue
        seen.add(low)
        if vocab.covers(low):
            continue
        if head is not None and not latin and vocab.acronym(head.group().lower()):
            continue
        added.append(_surface(ap, m.start(), m.end()))
    out.extend(Violation(kind="имя", detail=f"появилось: {w}") for w in added[:LIMIT])


def _values(s: str) -> set[float]:
    out: set[float] = set()
    for n in _numbers(s):
        try:
            out.add(float(n))
        except ValueError:
            continue
    return out


def _check_numerals(
    bp: Prepared, ap: Prepared, before: str, violations: list[Violation], warnings: list[Violation]
) -> None:
    known = {NUMERALS[_index(m)].key for m in NUMERAL_RE.finditer(bp.text)}
    values = _values(before)
    reported: set[str] = set()
    hard: list[str] = []
    soft: list[str] = []
    for m in NUMERAL_RE.finditer(ap.prose):
        n = NUMERALS[_index(m)]
        if n.key in known or n.key in reported or (n.value is not None and n.value in values):
            continue
        reported.add(n.key)
        (soft if n.soft else hard).append(_surface(ap, m.start(), m.end()))
    violations.extend(Violation(kind="число словами", detail=f"появилось: {w}") for w in hard[:LIMIT])
    warnings.extend(Violation(kind="число словами", detail=f"появилось: {w}") for w in soft[:LIMIT])

    months = {MONTHS[_index(m)][0] for m in MONTH_RE.finditer(bp.text)}
    added: list[str] = []
    for m in MONTH_RE.finditer(ap.prose):
        key = MONTHS[_index(m)][0]
        if key not in months:
            months.add(key)
            added.append(_surface(ap, m.start(), m.end()))
    violations.extend(Violation(kind="месяц", detail=f"появилось: {w}") for w in added[:LIMIT])


def _claims(p: Prepared) -> tuple[Counter[str], dict[str, str]]:
    text = _mask(p.prose, CLAIM_IDIOMS)
    count: Counter[str] = Counter()
    surface: dict[str, str] = {}
    for m in CLAIM_RE.finditer(text):
        key = CLAIMS[_index(m)][0]
        count[key] += 1
        surface[key] = _surface(p, m.start(), m.end())
    return count, surface


def _check_claims(bp: Prepared, ap: Prepared, out: list[Violation]) -> None:
    b, _ = _claims(bp)
    a, surface = _claims(ap)
    for key, n in a.items():
        if n <= b[key]:
            continue
        word = surface[key].lower()
        detail = f"появилось: {word}" if b[key] == 0 else f"стало больше «{word}»: {b[key]} → {n}"
        out.append(Violation(kind="утверждение", detail=detail))


def validate(before: str, after: str, context: ContextMode = "general") -> ValidationResult:
    v: list[Violation] = []
    w: list[Violation] = []
    if _frontmatter(before) != _frontmatter(after):
        v.append(Violation(kind="yaml", detail="YAML-шапка изменена"))
    _compare_exact("код", _all(FENCE_RE, before), _all(FENCE_RE, after), v)
    _compare_exact(
        "инлайн-код", _all(INLINE_CODE_RE, FENCE_RE.sub("", before)), _all(INLINE_CODE_RE, FENCE_RE.sub("", after)), v
    )
    b, a = _strip_code(before), _strip_code(after)
    _compare_exact("формула", _all(FORMULA_RE, b), _all(FORMULA_RE, a), v)
    _compare_exact("URL", _urls(before), _urls(after), v)
    _compare_exact("ссылка на литературу", _all(CITATION_RE, b), _all(CITATION_RE, a), v)
    _compare_exact("ссылка pandoc", _all(PANDOC_RE, b), _all(PANDOC_RE, a), v)
    _compare_exact("таблица", _all(TABLE_RE, b), _all(TABLE_RE, a), v)
    _compare_exact("цитата-блок", _all(BLOCKQUOTE_RE, b), _all(BLOCKQUOTE_RE, a), v)
    _compare_exact(
        "цитата",
        [QUOTE_MARKS_RE.sub("", q) for q in _all(QUOTE_RE, b)],
        [QUOTE_MARKS_RE.sub("", q) for q in _all(QUOTE_RE, a)],
        v,
    )
    _compare_exact("число", _numbers(before), _numbers(after), v)
    hb = " ".join(_all(HEADING_SHAPE_RE, b))
    ha = " ".join(_all(HEADING_SHAPE_RE, a))
    if hb != ha:
        v.append(Violation(kind="заголовки", detail=f"структура заголовков изменилась: [{hb}] → [{ha}]"))

    analysis = analyze(before, context)
    bp, ap = prepare(before), prepare(after)
    _check_hedges(bp, ap, analysis.issues, v, w)
    _check_names(bp, ap, v)
    _check_numerals(bp, ap, before, v, w)
    _check_claims(bp, ap, w)

    issues_before = len(analysis.issues)
    issues_after = len(analyze(after, context).issues)
    if issues_after > issues_before:
        v.append(Violation(kind="находки", detail=f"находок стало больше: {issues_before} → {issues_after}"))
    return ValidationResult(ok=not v, violations=v, warnings=w, issues_before=issues_before, issues_after=issues_after)

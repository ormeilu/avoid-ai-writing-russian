"""Проверка сохранности: правка прозы не должна портить защищённое содержимое.

Сравнивает исходник и исправленный текст и сообщает, что пропало или изменилось.

Разрешённые скиллом изменения не считаются нарушением: смена прямых кавычек на
«ёлочки», десятичной точки на запятую, Title Case в заголовке на обычный регистр,
удаление параметра utm_source ИИ-инструмента.
"""

from __future__ import annotations

from collections import Counter

from aiw_ru.compat import jsre
from aiw_ru.detect import AI_URL_RE, analyze
from aiw_ru.types import ContextMode, Record


class Violation(Record):
    kind: str
    detail: str


class ValidationResult(Record):
    ok: bool
    violations: list[Violation]
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
    out.extend(Violation(kind=kind, detail=f"пропало или изменено: {m[:120]}") for m in missing[:5])
    out.extend(Violation(kind=kind, detail=f"появилось: {a[:120]}") for a in added[:5])


def validate(before: str, after: str, context: ContextMode = "general") -> ValidationResult:
    v: list[Violation] = []
    if _frontmatter(before) != _frontmatter(after):
        v.append(Violation(kind="yaml", detail="YAML-шапка изменена"))
    _compare_exact("код", _all(FENCE_RE, before), _all(FENCE_RE, after), v)
    b, a = _strip_code(before), _strip_code(after)
    _compare_exact("инлайн-код", _all(INLINE_CODE_RE, b), _all(INLINE_CODE_RE, a), v)
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

    issues_before = len(analyze(before, context).issues)
    issues_after = len(analyze(after, context).issues)
    if issues_after > issues_before:
        v.append(Violation(kind="находки", detail=f"находок стало больше: {issues_before} → {issues_after}"))
    return ValidationResult(ok=not v, violations=v, issues_before=issues_before, issues_after=issues_after)

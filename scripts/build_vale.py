"""Собирает стиль Vale (https://vale.sh) из таблиц детектора.

Источник — словари `aiw_ru.lexicon` и несколько выражений из `aiw_ru.detect`, которые
работают без порогов и подсчётов. Результат лежит в `vale/AvoidAIWritingRussian/`:
по файлу правила на тип находки (или на тип и уровень, если уровни в типе разные),
`manifest.json` с оценкой переноса каждого типа, `meta.json` и короткий README.

Запуск после правки словарей:

    uv run python scripts/build_vale.py

Проверка, что записанный стиль совпадает с таблицами (хук `vale-style` и тест):

    uv run python scripts/build_vale.py --check

Vale ищет движком regexp2: ретроспектива и опережающая проверка работают, `\\p{L}` тоже,
поэтому границы слова переносятся из детектора как есть, через `(?<![\\p{L}\\p{N}_])`,
а не через `\\b`. Детектор ищет по тексту, где «ё» заменена на «е»; Vale видит исходный
текст, поэтому каждая «е» в выражении превращается в `[её]`.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import regex

from aiw_ru.detect import (
    AI_URL_RE,
    CHAT_MARKUP_RE,
    DECIMAL_RE,
    ENGLISH_QUOTES_RE,
    HYPHEN_DASH_RE,
    MARK_TAG_RE,
    NEGATION_RE,
    NXY_RE,
    PLACEHOLDER_RE,
    SPLIT_NXY_RE,
    STRAIGHT_QUOTES_RE,
    TYPE_LABELS,
    WEIGHTS,
)
from aiw_ru.lexicon import ALL_LEXICON, IDIOLECT, PHRASE3, TIER2, TIER3
from aiw_ru.types import Severity

ROOT = Path(__file__).resolve().parent.parent
STYLE = "AvoidAIWritingRussian"
STYLE_DIR = ROOT / "vale" / STYLE
# Версия Vale, на которой проверены правила и тест сценариев.
TESTED_VALE = "3.27.0"
LINK = "https://github.com/ormeilu/avoid-ai-writing-russian/tree/master/skills/avoid-ai-writing-russian/references"

# Ключ длиннее 128 байт yamlfmt (go-yaml) пишет явным ключом `? …`; генератор делает так же,
# иначе хук yamlfmt переписал бы файлы и стиль разошёлся бы с таблицами.
SIMPLE_KEY_BYTES = 128

LEVELS: dict[Severity, str] = {"P0": "error", "P1": "warning", "P2": "suggestion"}

VERIFIED = "verified"
APPROXIMATION = "approximation"
UNSUPPORTED = "unsupported"
UNASSESSED = "unassessed"
STATUSES = (VERIFIED, APPROXIMATION, UNSUPPORTED, UNASSESSED)


@dataclass(slots=True)
class Pattern:
    """Одно выражение правила и подсказка к нему."""

    source: str
    hint: str
    ignorecase: bool = True


@dataclass(slots=True)
class Rule:
    """Файл правила Vale: тип находки, уровень и выражения с подсказками."""

    type: str
    severity: Severity
    patterns: list[Pattern] = field(default_factory=list)
    name: str = ""


@dataclass(slots=True)
class Assessment:
    """Строка манифеста: насколько правило Vale повторяет детектор."""

    status: str
    reason: str = ""
    excluded: list[str] = field(default_factory=list)


# ─── Выражения из detect.py ────────────────────────────────────────────
# Подсказки повторяют те, что детектор пишет в находках (_detect_fingerprints,
# _detect_typography, _detect_structure).
CYRILLIC_INSIDE = "(?=[^{close}\\n]{{0,199}}\\p{{Cyrillic}})"


def _quotes(pattern: regex.Pattern[str], close: str) -> str:
    # Детектор находит пару кавычек и отбрасывает её, если внутри нет кириллицы. В Vale это
    # опережающая проверка сразу после открывающей кавычки.
    opening = pattern.pattern[0]
    return f"{opening}{CYRILLIC_INSIDE.format(close=close)}[^{close}\\n]{{1,200}}{close}"


@dataclass(slots=True)
class DetectTable:
    type: str
    severity: Severity
    patterns: list[Pattern]
    status: str
    reason: str = ""


DETECT_TABLES: list[DetectTable] = [
    DetectTable(
        "chat-markup",
        "P0",
        [
            Pattern(CHAT_MARKUP_RE.pattern, "удалить; если ссылка нужна — заменить настоящей", ignorecase=False),
            Pattern(MARK_TAG_RE.pattern, "убрать теги <mark>, текст оставить", ignorecase=False),
        ],
        APPROXIMATION,
        "в Markdown Vale не видит HTML-теги и адреса ссылок, поэтому `<think>`, `<mark>`, `<grok-card` "
        "и `](sandbox:…)` ловятся только в обычном тексте (.txt)",
    ),
    DetectTable(
        "ai-url",
        "P0",
        [Pattern(AI_URL_RE.pattern, "удалить только этот параметр")],
        APPROXIMATION,
        "в Markdown Vale пропускает адреса, и голые, и в ссылках; параметр ловится только в обычном тексте (.txt)",
    ),
    DetectTable(
        "placeholder",
        "P0",
        [Pattern(PLACEHOLDER_RE.pattern, "заполнить реальным содержимым или удалить предложение", ignorecase=False)],
        APPROXIMATION,
        "в Markdown Vale не видит HTML-комментарии, заглушка `<!-- вставьте … -->` ловится только в .txt",
    ),
    DetectTable(
        "hyphen-dash",
        "P2",
        [Pattern(HYPHEN_DASH_RE.pattern, "длинное тире с пробелами: « — »", ignorecase=False)],
        VERIFIED,
    ),
    DetectTable(
        "decimal-point",
        "P2",
        [Pattern(DECIMAL_RE.pattern, "запятая вместо точки", ignorecase=False)],
        VERIFIED,
    ),
    DetectTable(
        "straight-quotes",
        "P2",
        [Pattern(_quotes(STRAIGHT_QUOTES_RE, '"'), "«ёлочки» вместо прямых кавычек", ignorecase=False)],
        APPROXIMATION,
        "детектор пропускает кавычки в записях литературы и отбрасывает пару без кириллицы целиком, "
        'а Vale в строке `"abc" и "текст"` может взять в пару кавычки между ними',
    ),
    DetectTable(
        "english-quotes",
        "P1",
        [
            Pattern(
                _quotes(ENGLISH_QUOTES_RE, "”"),
                "«ёлочки»; английские кавычки в русском тексте — след машинного перевода",
                ignorecase=False,
            )
        ],
        APPROXIMATION,
        "детектор пропускает кавычки в записях литературы",
    ),
    DetectTable(
        "not-x-but-y",
        "P1",
        [
            Pattern(NXY_RE.pattern, "прямое утверждение без противопоставления"),
            Pattern(SPLIT_NXY_RE.pattern, "разнесённое «не X. Y» — сказать Y прямо"),
        ],
        VERIFIED,
    ),
    DetectTable(
        "negation-chain",
        "P2",
        [Pattern(NEGATION_RE.pattern, "сказать, что это есть, а не чем оно не является")],
        VERIFIED,
    ),
]

# Типы без правила Vale и причина.
_COUNTS = "нужен подсчёт по документу или абзацу, а правило Vale видит одно совпадение"
UNSUPPORTED_REASONS: dict[str, str] = {
    "invisible-chars": "находку даёт нормализация текста (одна сводка на документ), а не таблица выражений",
    "homoglyph": "нужна проверка смешанных алфавитов внутри слова с исключениями для формул и кодов",
    "soft-hyphen": "находку даёт нормализация текста (одна сводка на документ), а не таблица выражений",
    "typography-chars": "одна сводка на каждый редкий знак с числом вхождений",
    "truncated": "смотрит только последнюю строку последнего абзаца документа",
    "tier2-cluster": "нужны два разных слова уровня 2 в одном абзаце",
    "tier3": "нужна суммарная плотность от 3 % слов в тексте от 150 слов",
    "phrase3": "нужен повтор одного штампа или три разных в тексте",
    "phrase3-cluster": "нужны три разных фразовых штампа в тексте",
    "model-idiolect": "нужны два разных оборота почерка модели в одном тексте",
    "transition-run": "нужны три абзаца подряд, начатые переходом",
    "em-dash-splice": "нужен порог: больше одной связки или больше одной на 500 слов",
    "title-case": "нужен разбор заголовка на слова и проверка каждого",
    "bold-overuse": "нужен подсчёт выделений в разделе с исключениями для терминов",
    "emoji-header": "нужен класс \\p{Extended_Pictographic}, которого нет в движке Vale",
    "bullet-np-list": "нужен разбор пунктов списка на слова и глагольные формы",
    "hashtag-stuffing": "нужно шесть и больше хэштегов в тексте",
    "genitive-chain": "нужна цепочка из четырёх и больше отглагольных существительных",
    "passive-run": "нужны четыре предложения подряд с пассивом без деятеля",
    "same-opener": "нужны три предложения подряд с одним первым словом",
    "staccato": "нужны три коротких предложения подряд",
    "stacked-questions": "нужны три вопросительных предложения подряд",
    "uniform-sentences": "стилометрия: разброс длин предложений по тексту",
    "uniform-paragraphs": "стилометрия: разброс длин абзацев по тексту",
    "low-diversity": "стилометрия: доля разных слов в тексте",
}

# Типы словаря, у которых детектор ловит что-то ещё вне словаря.
EXTRA_IN_DETECTOR: dict[str, str] = {
    "chat-wrapper": "подводку отдельной строкой («Вот вариант поста:») детектор ловит выражением "
    "с якорями строки, в стиль она не перенесена",
}

# Экранированные знаки разметки Markdown: Vale убирает разметку до поиска.
MARKUP_ESCAPES = ("\\*", "\\_", "\\`", "\\#")

_CLUSTER_ONLY = {id(entry) for entry in (*TIER2, *TIER3, *PHRASE3, *IDIOLECT)}


# ─── Перевод выражений ─────────────────────────────────────────────────
def _class_has_e(body: str) -> tuple[bool, bool]:
    """Есть ли в теле класса «е» или «Е», в том числе внутри диапазона."""
    items: list[str | None] = []
    i = 0
    while i < len(body):
        char = body[i]
        if char == "\\":
            # Экранированный знак, \p{…}, \uXXXX: в диапазоны не идут.
            if body.startswith("\\p{", i) or body.startswith("\\P{", i):
                i = body.index("}", i) + 1
            elif body.startswith("\\u", i):
                i += 6
            elif body.startswith("\\x", i):
                i += 4
            else:
                i += 2
            items.append(None)
            continue
        items.append(char)
        i += 1
    lower = upper = False
    for index, item in enumerate(items):
        if item is None:
            continue
        if item == "-" and 0 < index < len(items) - 1:
            low, high = items[index - 1], items[index + 1]
            if low is not None and high is not None:
                lower |= low <= "е" <= high
                upper |= low <= "Е" <= high
            continue
        lower |= item == "е"
        upper |= item == "Е"
    return lower, upper


def to_vale(source: str) -> str:
    """Переводит выражение детектора (синтаксис пакета regex) для движка Vale.

    - «е» вне класса становится `[её]`, в класс с «е» добавляется «ё»: детектор ищет по тексту,
      где «ё» уже заменена;
    - захватывающие группы становятся незахватывающими: Vale нумерует группы правила замены сам;
    - невидимые и пробельные знаки, кроме обычного пробела, записываются как `\\uXXXX`.
    """
    parts: list[str] = []
    i = 0
    while i < len(source):
        char = source[i]
        if char == "\\":
            parts.append(source[i : i + 2])
            i += 2
            continue
        if char == "[":
            end = i + 1
            if end < len(source) and source[end] == "^":
                end += 1
            if end < len(source) and source[end] == "]":
                end += 1
            while source[end] != "]":
                end += 2 if source[end] == "\\" else 1
            body = source[i + 1 : end]
            lower, upper = _class_has_e(body)
            extra = ("ё" if lower and "ё" not in body else "") + ("Ё" if upper and "Ё" not in body else "")
            parts.append(f"[{_escape_invisible(body)}{extra}]")
            i = end + 1
            continue
        if char == "(" and not source.startswith("(?", i):
            parts.append("(?:")
        elif char == "е":
            parts.append("[её]")
        elif char == "Е":
            parts.append("[ЕЁ]")
        else:
            parts.append(_escape_invisible(char))
        i += 1
    return "".join(parts)


def _escape_invisible(text: str) -> str:
    return "".join(
        f"\\u{ord(char):04X}" if char != " " and unicodedata.category(char) in {"Cc", "Cf", "Zs", "Zl", "Zp"} else char
        for char in text
    )


def _camel(type_: str) -> str:
    return "".join(part[:1].upper() + part[1:] for part in type_.split("-"))


# ─── Сборка ────────────────────────────────────────────────────────────
@dataclass(slots=True)
class Style:
    rules: list[Rule]
    assessments: dict[str, Assessment]
    rules_by_type: dict[str, list[str]]


def collect() -> Style:
    """Правила и оценки по типам, в порядке WEIGHTS."""
    by_type: dict[str, dict[Severity, Rule]] = {}
    excluded: dict[str, list[str]] = {}
    markup: dict[str, list[str]] = {}
    for entry in ALL_LEXICON:
        if id(entry) in _CLUSTER_ONLY:
            continue
        if entry.density:
            # Порог плотности на 1000 слов Vale не выразит: такие записи не переносятся.
            excluded.setdefault(entry.type, []).append(entry.id)
            continue
        if any(char in entry.re.pattern for char in MARKUP_ESCAPES):
            markup.setdefault(entry.type, []).append(entry.id)
        rules = by_type.setdefault(entry.type, {})
        rule = rules.setdefault(entry.severity, Rule(entry.type, entry.severity))
        rule.patterns.append(Pattern(entry.re.pattern, entry.hint))

    reasons: dict[str, list[str]] = {}
    for type_, ids in excluded.items():
        names = ", ".join(f"«{id_}»" for id_ in ids)
        single = len(ids) == 1
        reasons.setdefault(type_, []).append(
            f"без {'записи' if single else 'записей'} {names}: детектор считает "
            f"{'её' if single else 'их'} находкой только выше порога плотности на 1000 слов"
        )
    for type_, ids in markup.items():
        names = ", ".join(f"«{id_}»" for id_ in ids)
        single = len(ids) == 1
        reasons.setdefault(type_, []).append(
            f"Vale убирает разметку Markdown до поиска, поэтому {'запись' if single else 'записи'} {names} "
            f"со звёздочками и другими знаками разметки {'срабатывает' if single else 'срабатывают'} только в .txt"
        )
    for type_, reason in EXTRA_IN_DETECTOR.items():
        reasons.setdefault(type_, []).append(reason)
    assessments: dict[str, Assessment] = {}
    for type_ in by_type:
        if type_ in reasons:
            assessments[type_] = Assessment(APPROXIMATION, "; ".join(reasons[type_]), excluded.get(type_, []))
        else:
            assessments[type_] = Assessment(VERIFIED)

    for table in DETECT_TABLES:
        by_type[table.type] = {table.severity: Rule(table.type, table.severity, list(table.patterns))}
        assessments[table.type] = Assessment(table.status, table.reason)

    for type_, reason in UNSUPPORTED_REASONS.items():
        assessments.setdefault(type_, Assessment(UNSUPPORTED, reason))

    rules: list[Rule] = []
    rules_by_type: dict[str, list[str]] = {}
    for type_ in WEIGHTS:
        assessments.setdefault(type_, Assessment(UNASSESSED, "тип не разобран: добавьте его в scripts/build_vale.py"))
        type_rules = by_type.get(type_, {})
        mixed = len(type_rules) > 1
        for severity in sorted(type_rules):
            rule = type_rules[severity]
            rule.name = _camel(type_) + (LEVELS[severity].capitalize() if mixed else "")
            rules.append(rule)
            rules_by_type.setdefault(type_, []).append(f"{STYLE}.{rule.name}")
    return Style(rules, assessments, rules_by_type)


def _quote(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def render_rule(rule: Rule) -> str:
    label = TYPE_LABELS[rule.type].replace("%", "%%")
    ignorecase = all(pattern.ignorecase for pattern in rule.patterns)
    lines = [
        f"# Собрано scripts/build_vale.py из таблиц детектора (тип {rule.type}). Не правьте руками.",
        "extends: substitution",
        f"message: {_quote(f'{label}: «%[2]s» — %[1]s')}",
        f"link: {LINK}",
        f"level: {LEVELS[rule.severity]}",
        f"ignorecase: {'true' if ignorecase else 'false'}",
        "nonword: true",
        "swap:",
    ]
    seen: set[str] = set()
    for pattern in rule.patterns:
        source = to_vale(pattern.source)
        if pattern.ignorecase and not ignorecase:
            source = f"(?i:{source})"
        if source in seen:
            continue
        seen.add(source)
        key = _quote(source)
        if len(source.encode("utf-8")) > SIMPLE_KEY_BYTES:
            lines += [f"  ? {key}", f"  : {_quote(pattern.hint)}"]
        else:
            lines.append(f"  {key}: {_quote(pattern.hint)}")
    return "\n".join(lines) + "\n"


def render_manifest(style: Style) -> str:
    counts = dict.fromkeys(STATUSES, 0)
    types = []
    for type_ in WEIGHTS:
        assessment = style.assessments[type_]
        counts[assessment.status] += 1
        item: dict[str, object] = {
            "type": type_,
            "label": TYPE_LABELS[type_],
            "status": assessment.status,
            "rules": style.rules_by_type.get(type_, []),
            "reason": assessment.reason,
        }
        if assessment.excluded:
            item["excluded"] = assessment.excluded
        types.append(item)
    manifest = {
        "schemaVersion": "1",
        "style": STYLE,
        "context": "general",
        "testedWithVale": TESTED_VALE,
        "counts": counts,
        "types": types,
    }
    return json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"


def render_meta() -> str:
    meta = {
        "feed": "https://github.com/ormeilu/avoid-ai-writing-russian/releases.atom",
        "vale_version": f">={TESTED_VALE}",
    }
    return json.dumps(meta, ensure_ascii=False, indent=2) + "\n"


STATUS_WORDS = {
    VERIFIED: "совпадает с детектором",
    APPROXIMATION: "приближение",
    UNSUPPORTED: "нет правила",
    UNASSESSED: "не разобран",
}

README = """\
# Стиль Vale AvoidAIWritingRussian

Правила для [Vale](https://vale.sh), собранные из словарей детектора `aiw-ru`: слова-маркеры,
канцелярит, кальки, жаргон, шаблонные обороты, следы чат-бота и часть типографики. Vale
подсвечивает их в редакторе и в CI, без Python и без модели.

Файлы собирает `scripts/build_vale.py`, руками их не правят. Проверено на Vale {tested}.

## Подключение

Скопируйте папку `AvoidAIWritingRussian` в каталог стилей проекта (или укажите `StylesPath` на
папку `vale` из клона репозитория) и добавьте в `.vale.ini`:

```ini
StylesPath = styles
MinAlertLevel = suggestion
# Детектор не проверяет таблицы и цитаты-блоки, Vale тоже не будет.
SkippedScopes = script, style, pre, figure, blockquote, table

[*.md]
BasedOnStyles = AvoidAIWritingRussian
# Цитаты в «ёлочках» и формулы детектор тоже пропускает.
TokenIgnores = («[^«»\\n]+»), (\\$[^$\\n]+\\$)
```

Уровни находок переходят в уровни Vale: P0 становится `error`, P1 — `warning`, P2 — `suggestion`.

## Чем отличается от детектора

Правила повторяют детектор в режиме по умолчанию (`--context general`): исключения режимов
`academic`, `technical`, `chat` и `social` Vale не знает. Детектор перед поиском заменяет
латинские буквы-двойники кириллицей и удаляет невидимые символы; Vale ищет по тексту как есть.
Оценки ИИ-стиля и светофора здесь нет, только отдельные находки.

`manifest.json` описывает каждый тип находки: `verified` — правило Vale срабатывает там же,
где детектор (это проверяет `tests/test_vale.py`), `approximation` — с оговоркой из поля
`reason`, `unsupported` — правила нет, потому что нужен подсчёт, порог или разбор текста.

| Состояние | Типов |
|---|---|
{counts}

| Тип | Правило | Состояние |
|---|---|---|
{types}
"""


def render_readme(style: Style) -> str:
    counts = dict.fromkeys(STATUSES, 0)
    for type_ in WEIGHTS:
        counts[style.assessments[type_].status] += 1
    count_rows = "\n".join(f"| `{status}` | {counts[status]} |" for status in STATUSES if counts[status])
    type_rows = []
    for type_ in WEIGHTS:
        assessment = style.assessments[type_]
        rules = ", ".join(f"`{name.split('.', 1)[1]}`" for name in style.rules_by_type.get(type_, [])) or "—"
        type_rows.append(f"| {TYPE_LABELS[type_]} (`{type_}`) | {rules} | {STATUS_WORDS[assessment.status]} |")
    return README.format(tested=TESTED_VALE, counts=count_rows, types="\n".join(type_rows))


def render() -> dict[str, str]:
    """Все файлы стиля: имя → содержимое."""
    style = collect()
    files = {f"{rule.name}.yml": render_rule(rule) for rule in style.rules}
    files["manifest.json"] = render_manifest(style)
    files["meta.json"] = render_meta()
    files["README.md"] = render_readme(style)
    return files


def write(directory: Path) -> None:
    """Записывает стиль в `directory`; правила, которых больше нет в таблицах, удаляются."""
    directory.mkdir(parents=True, exist_ok=True)
    files = render()
    for stale in directory.glob("*.yml"):
        if stale.name not in files:
            stale.unlink()
    for name, content in files.items():
        (directory / name).write_text(content, encoding="utf-8", newline="\n")


def differences(directory: Path) -> list[str]:
    """Файлы, которые разошлись с таблицами: лишние, недостающие и изменённые."""
    files = render()
    present = {path.name for path in directory.glob("*") if path.is_file()} if directory.is_dir() else set()
    problems = [f"нет файла {name}" for name in sorted(files.keys() - present)]
    problems += [f"лишний файл {name}" for name in sorted(present - files.keys())]
    problems += [
        f"устарел {name}"
        for name in sorted(files.keys() & present)
        if (directory / name).read_text(encoding="utf-8") != files[name]
    ]
    return problems


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Собирает стиль Vale из таблиц детектора.")
    parser.add_argument("--check", action="store_true", help="только проверить, что стиль свежий")
    parser.add_argument("--out", type=Path, default=STYLE_DIR, help="куда писать стиль")
    args = parser.parse_args(argv)
    if args.check:
        problems = differences(args.out)
        if problems:
            print("Стиль Vale разошёлся со словарями детектора:", file=sys.stderr)
            for problem in problems:
                print(f"  {problem}", file=sys.stderr)
            print("Пересоберите: uv run python scripts/build_vale.py", file=sys.stderr)
            return 1
        return 0
    write(args.out)
    print(f"Стиль записан в {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

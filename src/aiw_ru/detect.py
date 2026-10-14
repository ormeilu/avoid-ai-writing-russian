"""Детектор ИИ-стиля для русского текста.

Модель оценки. У каждой категории есть вес (WEIGHTS). Находки дедуплицируются
по паре (тип, текст); сырой балл — сумма весов уникальных находок. Правки ради
краткости (style_only) весят мало, а жаргон не весит ничего: это совет по стилю, а не
довод об авторстве.
Сырой балл нормируется по длине текста и насыщается к 100, чтобы длинный текст
с той же плотностью примет не набирал бесконечно.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import regex

from aiw_ru.compat import fixed, js_round, jsre, to_fixed, total, trim
from aiw_ru.lexicon import (
    ACADEMIC_FORMULAS,
    ALL_LEXICON,
    IDIOLECT,
    PHRASE3,
    TECHNICAL_TERMS,
    TIER2,
    TIER3,
    LexEntry,
)
from aiw_ru.text import (
    TYPOGRAPHIC,
    Block,
    Prepared,
    Sentence,
    blocks,
    cv,
    line_col,
    mattr,
    mean,
    plural,
    prepare,
    sentences,
    words,
)
from aiw_ru.types import AnalysisResult, ContextMode, Issue, Light, LightLevel, Severity, Stats

WEIGHTS: dict[str, float] = {
    # Невидимые символы, подмена букв и мягкие переносы — гигиена документа, а не ИИ-стиль:
    # в человеческих текстах LLMTrace они встречаются чаще, чем в сгенерированных. В оценку
    # не идут; первые два делают документ подозрительным (флаг `suspicious`).
    "invisible-chars": 0,
    "homoglyph": 0,
    "soft-hyphen": 0,
    # Редкие типографские символы — вопрос к пользователю («ИИ или типография?»), а не довод:
    # их ставят и модели, и типографы, и вёрстка сайтов. В оценку не идут, светофор — жёлтый.
    "typography-chars": 0,
    "chat-markup": 10,
    "ai-url": 8,
    "placeholder": 8,
    "cutoff": 10,
    "chatbot": 8,
    "chat-wrapper": 6,
    "truncated": 4,
    "sycophancy": 6,
    "vague-attribution": 5,
    "significance": 5,
    "tier1": 3,
    "tier1-clarity": 0.5,
    "tier2": 1.5,
    "tier2-cluster": 3,
    "tier3": 1,
    "phrase3": 1,
    "phrase3-cluster": 3,
    "model-idiolect": 3,
    "calque": 3,
    # Жаргон и транслитерации — совет по стилю (см. JARGON в lexicon.py), в оценку не идут: в русской
    # части LLMTrace (test) они есть у 1,1 % человеческих текстов и у 0,6 % сгенерированных.
    "jargon": 0,
    "template": 3,
    "transition": 1.5,
    "transition-run": 3,
    "filler": 2,
    "hollow-intensifier": 1.5,
    "confidence": 1.5,
    "hedge-stack": 2,
    "vague-endorsement": 1.5,
    "future-narrative": 3,
    "generic-conclusion": 2,
    "novelty": 3,
    "unmeasured-claim": 2,
    "lets": 3,
    "reasoning": 4,
    "acknowledgment": 3,
    "narrated-candor": 3,
    "hook": 3,
    "social-closer": 3,
    "launch-intro": 3,
    "fake-casual": 3,
    "speculative-opener": 3,
    "false-concession": 2,
    "moral-adjective": 2,
    "real-inflation": 1.5,
    "not-x-but-y": 3,
    "em-dash-splice": 2,
    "hyphen-dash": 0.5,
    "straight-quotes": 1,
    "english-quotes": 2,
    "decimal-point": 1,
    "title-case": 2,
    "bold-overuse": 2,
    "emoji-header": 3,
    "bullet-np-list": 3,
    "hashtag-stuffing": 4,
    "genitive-chain": 1.5,
    "passive-run": 2,
    "same-opener": 2,
    "staccato": 2,
    "negation-chain": 2,
    "stacked-questions": 2,
    "uniform-sentences": 5,
    "uniform-paragraphs": 3,
    "low-diversity": 3,
}

TYPE_LABELS: dict[str, str] = {
    "invisible-chars": "Невидимые символы",
    "homoglyph": "Подмена букв",
    "soft-hyphen": "Мягкие переносы",
    "typography-chars": "Редкие типографские символы",
    "chat-markup": "Разметка цитирования из чата",
    "ai-url": "Параметр ИИ-инструмента в ссылке",
    "placeholder": "Незаполненная заглушка",
    "cutoff": "Оговорка об отсечке знаний",
    "chatbot": "След чат-бота",
    "chat-wrapper": "Обвязка ответа чата",
    "truncated": "Текст оборван на полуслове",
    "sycophancy": "Угодливость",
    "vague-attribution": "Размытая ссылка на авторитет",
    "significance": "Раздувание значимости",
    "tier1": "Слово-маркер ИИ",
    "tier1-clarity": "Канцелярит",
    "tier2": "Слово уровня 2",
    "tier2-cluster": "Скопление слов уровня 2",
    "tier3": "Перегруженное слово",
    "phrase3": "Фразовый штамп",
    "phrase3-cluster": "Скопление фразовых штампов",
    "model-idiolect": "Почерк модели",
    "calque": "Калька с английского",
    "jargon": "Жаргон и транслитерация с английского",
    "template": "Шаблонная конструкция",
    "transition": "Шаблонный переход",
    "transition-run": "Переходы в начале абзацев подряд",
    "filler": "Подушка",
    "hollow-intensifier": "Пустой усилитель",
    "confidence": "Нагнетание уверенности",
    "hedge-stack": "Нагромождённая оговорка",
    "vague-endorsement": "Пустое одобрение",
    "future-narrative": "Туманный прогноз",
    "generic-conclusion": "Пустой вывод",
    "novelty": "Раздувание новизны",
    "unmeasured-claim": "Оценка без числа",
    "lets": "«Давайте…»",
    "reasoning": "След рассуждения модели",
    "acknowledgment": "Пересказ вопроса",
    "narrated-candor": "Показная откровенность",
    "hook": "Хук из соцсетей",
    "social-closer": "Рекомендательная концовка",
    "launch-intro": "Презентация как выход на сцену",
    "fake-casual": "Фальшиво-разговорный регистр",
    "speculative-opener": "«Представьте…»",
    "false-concession": "Ложная уступка",
    "moral-adjective": "Моральный эпитет у неодушевлённого",
    "real-inflation": "«Настоящий/реальный» как усилитель",
    "not-x-but-y": "«Не X, а Y»",
    "em-dash-splice": "Тире-связка ради эффекта",
    "hyphen-dash": "Дефис вместо тире",
    "straight-quotes": "Прямые кавычки в русском тексте",
    "english-quotes": "Английские кавычки в русском тексте",
    "decimal-point": "Десятичная точка",
    "title-case": "Title Case в заголовке",
    "bold-overuse": "Избыток жирного",
    "emoji-header": "Эмодзи в заголовке",
    "bullet-np-list": "Список из голых именных групп",
    "hashtag-stuffing": "Хэштеги",
    "genitive-chain": "Цепочка отглагольных существительных",
    "passive-run": "Пассив без деятеля подряд",
    "same-opener": "Одинаковые начала предложений",
    "staccato": "Рубленые фрагменты",
    "negation-chain": "Цепочка отрицаний",
    "stacked-questions": "Стопка риторических вопросов",
    "uniform-sentences": "Одинаковая длина предложений",
    "uniform-paragraphs": "Одинаковые абзацы",
    "low-diversity": "Бедный словарь",
}

SKIP_IN_CHAT = frozenset(
    (
        "straight-quotes english-quotes decimal-point hyphen-dash em-dash-splice title-case bold-overuse "
        "genitive-chain passive-run uniform-sentences uniform-paragraphs low-diversity same-opener staccato "
        "bullet-np-list tier3 phrase3 tier2 tier2-cluster transition-run"
    ).split()
)


@dataclass(slots=True)
class _Ctx:
    prepared: Prepared
    mode: ContextMode
    word_count: int
    issues: list[Issue] = field(default_factory=list)


def _add(
    ctx: _Ctx,
    type_: str,
    rule: str,
    severity: Severity,
    index: int,
    text: str,
    hint: str,
    style_only: bool = False,
) -> None:
    if ctx.mode == "chat" and type_ in SKIP_IN_CHAT:
        return
    prepared = ctx.prepared
    source_index = prepared.to_source[index] if index < len(prepared.to_source) else index
    # Фрагмент показывается как набран в исходнике (с «ё», невидимыми символами и подменёнными
    # буквами), чтобы его можно было найти в файле. Сводки вроде «три штампа» остаются как есть.
    end = index + len(text)
    if text and end <= len(prepared.text) and prepared.text[index:end] == text:
        text = prepared.source[source_index : prepared.to_source[end - 1] + 1]
    line, column = line_col(prepared.line_starts, source_index)
    ctx.issues.append(
        Issue(
            type=type_,
            rule=rule,
            severity=severity,
            text=trim(text)[:160],
            index=source_index,
            line=line,
            column=column,
            hint=hint,
            style_only=style_only,
        )
    )


def _applies(entry: LexEntry, mode: ContextMode) -> bool:
    return mode not in entry.skip


def _ranges(pattern: regex.Pattern[str], text: str) -> list[tuple[int, int]]:
    """Отрезки, где совпадает выражение; строится один раз на текст."""
    return [match.span() for match in pattern.finditer(text)]


def _hits_range(ranges: list[tuple[int, int]], index: int, length: int) -> bool:
    """Пересекает ли [index, index+length) хотя бы один отрезок (отрезки отсортированы)."""
    lo, hi = 0, len(ranges)
    while lo < hi:
        mid = (lo + hi) >> 1
        if ranges[mid][1] <= index:
            lo = mid + 1
        else:
            hi = mid
    return lo < len(ranges) and ranges[lo][0] < index + length


# ─── Словарные правила ──────────────────────────────────────────────────
_CLUSTER_ONLY = frozenset(id(entry) for entry in (*TIER2, *TIER3, *PHRASE3, *IDIOLECT))
# Почерк модели: находка, если в тексте столько разных оборотов. Один оборот есть
# у 1,9 % человеческих текстов LLMTrace, два и больше — у 0,1 % (у моделей 1,5 %).
IDIOLECT_MIN = 2


def _detect_lexicon(ctx: _Ctx, text_blocks: list[Block]) -> None:
    prepared, mode = ctx.prepared, ctx.mode

    def per1000(count: int) -> float:
        return count * 1000 / ctx.word_count if ctx.word_count else 0

    if mode == "academic":
        exempt = _ranges(ACADEMIC_FORMULAS, prepared.prose)
    elif mode == "technical":
        exempt = _ranges(TECHNICAL_TERMS, prepared.prose)
    else:
        exempt = []
    for entry in ALL_LEXICON:
        if id(entry) in _CLUSTER_ONLY or not _applies(entry, mode):
            continue
        if mode == "chat" and entry.severity != "P0" and entry.type != "calque":
            continue
        hits = list(entry.re.finditer(prepared.prose))
        if not hits:
            continue
        threshold = entry.density.get(mode, entry.density.get("default"))
        if threshold is not None and per1000(len(hits)) < threshold:
            continue
        for match in hits:
            if _hits_range(exempt, match.start(), len(match.group())):
                continue
            _add(ctx, entry.type, entry.id, entry.severity, match.start(), match.group(), entry.hint, entry.style_only)

    # Уровень 2: скопление двух и более разных слов в одном абзаце.
    if mode != "chat":
        for block in text_blocks:
            if block.kind not in ("prose", "list"):
                continue
            found: dict[str, tuple[int, str]] = {}
            for entry in TIER2:
                match = entry.re.search(block.text)
                if match and entry.id not in found:
                    if _hits_range(exempt, block.start + match.start(), len(match.group())):
                        continue
                    found[entry.id] = (block.start + match.start(), match.group())
            if len(found) >= 2:
                cluster_hits = list(found.values())
                _add(
                    ctx,
                    "tier2-cluster",
                    "tier2",
                    "P2",
                    cluster_hits[0][0],
                    ", ".join(text for _, text in cluster_hits),
                    "перефразировать проще, оставить одно слово или дать конкретику",
                )

    # Уровень 3: суммарная плотность ≥ 3 % слов.
    tier3_count = 0
    first_tier3 = -1
    tier3_words: list[str] = []
    for entry in TIER3:
        for match in entry.re.finditer(prepared.prose):
            tier3_count += 1
            if first_tier3 < 0 or match.start() < first_tier3:
                first_tier3 = match.start()
            if len(tier3_words) < 6:
                tier3_words.append(match.group())
    if ctx.word_count >= 150 and tier3_count / ctx.word_count >= 0.03:
        _add(
            ctx,
            "tier3",
            "density",
            "P2",
            max(first_tier3, 0),
            ", ".join(tier3_words),
            f"перегруженные оценочные слова: {tier3_count} на {ctx.word_count} слов — заменить часть числами и примерами",
        )

    # Фразовые штампы: одна фраза 2+ раза или 3+ разных.
    phrase_hits: dict[str, list] = {}
    for entry in PHRASE3:
        for match in entry.re.finditer(prepared.prose):
            hit = phrase_hits.setdefault(entry.id, [0, match.start(), match.group()])
            hit[0] += 1
    for id_, (count, index, text) in phrase_hits.items():
        if count >= 2:
            _add(ctx, "phrase3", id_, "P2", index, text, f"штамп повторён {count} раза — назвать конкретику")
    if len(phrase_hits) >= 3:
        first = min(phrase_hits.values(), key=lambda hit: hit[1])
        _add(
            ctx,
            "phrase3-cluster",
            "cluster",
            "P1",
            first[1],
            ", ".join(hit[2] for hit in phrase_hits.values()),
            "три и больше разных штампов — так варьирует шаблоны модель",
        )

    # Почерк модели: считается число разных оборотов, а не плотность. Каждый оборот
    # даёт одну находку в первом месте, где встретился.
    if mode != "chat":
        idioms: dict[str, tuple[int, str]] = {}
        for entry in IDIOLECT:
            if not _applies(entry, mode):
                continue
            for match in entry.re.finditer(prepared.prose):
                if not _hits_range(exempt, match.start(), len(match.group())):
                    idioms[entry.id] = (match.start(), match.group())
                    break
        if len(idioms) >= IDIOLECT_MIN:
            for id_, (index, text) in idioms.items():
                _add(
                    ctx,
                    "model-idiolect",
                    id_,
                    "P1",
                    index,
                    text,
                    f"{plural(len(idioms), 'оборот', 'оборота', 'оборотов')} из почерка модели в одном тексте: "
                    "сказать прямо, что произошло и с каким результатом",
                )


# ─── Технические отпечатки ──────────────────────────────────────────────
CHAT_MARKUP_RE = jsre(
    "\\uE200[^\\uE201\\n]{0,200}\\uE201|[\\uE200-\\uE204]|citeturn\\d+\\w*"
    "|(?<![\\p{L}\\d])turn\\d+(?:search|news|file|image|view|fetch|video|product|academia)\\d+"
    "|【\\d+(?::\\d+)?†[^】\\n]{0,80}】|contentReference\\[oaicite:\\d+\\](?:\\{index=\\d+\\})?"
    "|oai_citation|\\[attached_file:\\d+\\]|grok_render_citation_card_json|grok_card|attributableIndex"
    "|\\]\\(sandbox:/mnt/data/|:::writing\\{variant=|\\[cite:\\s*\\d+(?:\\s*,\\s*\\d+)*\\]|\\[citation:\\s*\\d+\\]"
    "|\\[span_\\d+\\]\\((?:start|end)_span\\)|vertexaisearch\\.cloud\\.google\\.com/grounding-api-redirect"
    "|</?think>",
    "gu",
)
AI_URL_RE = jsre(
    "[?&](?:utm_source=(?:chatgpt\\.com|openai|copilot\\.com|claude\\.ai|perplexity\\.ai|perplexity"
    "|gemini\\.google\\.com|deepseek\\.com)|referrer=grok\\.com)",
    "gi",
)
PLACEHOLDER_RE = jsre(
    "\\[(?:Ваш[аеи]?|Вставьте|Укажите|Добавьте|Введите|Опишите|Название|Имя|Фамилия"
    # «[Дата обращения: 12.03.2026]» в записи литературы — не заглушка.
    "|Дата(?![ \\t]+(?:обращения|доступа)[:\\s]*\\d)|ДАТА|Источник|Ссылка"
    "|Your|Insert|Add|Enter)[^\\]\\n]{0,60}\\]|\\b(?:19|20)XX\\b|\\b\\d{4}-XX-XX\\b"
    "|<!--\\s*(?:добавь|добавьте|вставь|вставьте|todo|TODO|заполни|укажите|add|insert)[^>]*-->",
    "g",
)


def _detect_fingerprints(ctx: _Ctx) -> None:
    prepared = ctx.prepared
    if prepared.invisible:
        index = prepared.invisible[0].index
        line, column = line_col(prepared.line_starts, index)
        ctx.issues.append(
            Issue(
                type="invisible-chars",
                rule="zero-width",
                severity="P0",
                text=plural(len(prepared.invisible), "невидимый символ", "невидимых символа", "невидимых символов"),
                index=index,
                line=line,
                column=column,
                hint="удалить; такие документы системы проверки помечают как подозрительные",
            )
        )
    if prepared.soft_hyphens:
        index = prepared.soft_hyphens[0]
        line, column = line_col(prepared.line_starts, index)
        ctx.issues.append(
            Issue(
                type="soft-hyphen",
                rule="soft-hyphen",
                severity="P2",
                text=plural(len(prepared.soft_hyphens), "мягкий перенос", "мягких переноса", "мягких переносов"),
                index=index,
                line=line,
                column=column,
                hint="удалить перед сдачей; обычно их оставляют Word и копирование из PDF",
            )
        )
    first_index: dict[str, int] = {}
    counts: Counter[str] = Counter()
    for found in prepared.typographic:
        first_index.setdefault(found.char, found.index)
        counts[found.char] += 1
    for char, index in sorted(first_index.items(), key=lambda item: item[1]):
        code = f"U+{ord(char):04X}"
        line, column = line_col(prepared.line_starts, index)
        ctx.issues.append(
            Issue(
                type="typography-chars",
                rule=code,
                severity="P2",
                text=f"{plural(counts[char], *TYPOGRAPHIC[char])} ({code})",
                index=index,
                line=line,
                column=column,
                hint="ИИ или типография? Спроси пользователя: заменить обычными знаками или оставить",
            )
        )
    for homoglyph in prepared.homoglyphs:
        line, column = line_col(prepared.line_starts, homoglyph.index)
        ctx.issues.append(
            Issue(
                type="homoglyph",
                rule="latin-in-cyrillic",
                severity="P0",
                text=homoglyph.word,
                index=homoglyph.index,
                line=line,
                column=column,
                hint="латинские буквы внутри русского слова; заменить на кириллицу",
            )
        )
    for type_, pattern, hint in (
        ("chat-markup", CHAT_MARKUP_RE, "удалить; если ссылка нужна — заменить настоящей"),
        ("ai-url", AI_URL_RE, "удалить только этот параметр"),
        ("placeholder", PLACEHOLDER_RE, "заполнить реальным содержимым или удалить предложение"),
    ):
        for match in pattern.finditer(prepared.no_code):
            _add(ctx, type_, type_, "P0", match.start(), match.group(), hint)


# ─── Обвязка ответа чата и обрыв генерации ──────────────────────────────
# Строка-подводка к ответу: «Вот вариант поста:», «Ниже несколько версий:». «Вот
# несколько вариантов» ловит chatbot. «Вот три варианта:» пишут и авторы статей, а
# документация — «Ниже приведены варианты запуска:», поэтому числа не входят,
# а в режиме technical правило не действует.
LEAD_IN_RE = jsre(
    "^[ \\t]*(?:\\*\\*)?(?:вот[ \\t]+(?:другой[ \\t]+|еще[ \\t]+один[ \\t]+"
    "|(?:альтернативн|готов|возможн|обновленн|исправленн|улучшенн|переработанн|сокращенн|доработанн)\\p{L}*[ \\t]+)?"
    "|ниже[ \\t]+несколько[ \\t]+)(?:вариант|верси|черновик)\\p{L}*[^\\n:]{0,60}:[ \\t*]*$",
    "gimu",
)
# Слова, на которых фраза не кончается: предлоги, союзы, «который». Строчные:
# «В» в конце строки чаще значит вольты.
DANGLING_RE = jsre(
    "(?<![\\p{L}\\d:;])(?:и|в|на|с|к|по|за|для|от|до|из|об|при|без|через|а|но|или|чтобы|котор\\p{L}+)$", "u"
)
TRAILING_COMMA_RE = jsre("\\p{L},$", "u")
TOKEN_RE = jsre("[^ \\t]+", "g")
TRUNCATED_MIN_WORDS = 8


def _detect_chat_leftovers(ctx: _Ctx, text_blocks: list[Block]) -> None:
    mode = ctx.mode
    if mode == "chat":
        return
    if mode != "technical":
        for match in LEAD_IN_RE.finditer(ctx.prepared.prose):
            _add(
                ctx,
                "chat-wrapper",
                "подводка",
                "P1",
                match.start(),
                match.group(),
                "удалить подводку, оставить сам текст",
            )
    # Обрыв генерации: последний блок текста — абзац прозы, его последняя строка длинная
    # и кончается запятой или словом, на котором фраза кончиться не может. Заголовок,
    # список, таблица, цитата, код и короткая подпись («С уважением,») не считаются.
    last = next((block for block in reversed(text_blocks) if block.kind != "empty"), None)
    if last is None or last.kind != "prose":
        return
    raw = ctx.prepared.text[last.start : last.end]
    line = trim(raw[raw.rfind("\n") + 1 :])
    if len(words(line)) < TRUNCATED_MIN_WORDS:
        return
    if TRAILING_COMMA_RE.search(line) or DANGLING_RE.search(line):
        # Находка показывает последние шесть слов строки.
        tail = line[[token.start() for token in TOKEN_RE.finditer(line)][-6:][0] :]
        _add(
            ctx,
            "truncated",
            "cut-off",
            "P1",
            last.start + raw.rfind(tail),
            tail,
            "текст обрывается на полуслове: дописать фразу по источнику или убрать оборванный хвост",
        )


# ─── Типографика ────────────────────────────────────────────────────────
CYR = jsre("\\p{Script=Cyrillic}", "u")
STRAIGHT_QUOTES_RE = jsre('"([^"\\n]{1,200})"', "g")
ENGLISH_QUOTES_RE = jsre("“([^”\\n]{1,200})”", "g")
HYPHEN_DASH_RE = jsre("(?<=\\p{L}) - (?=\\p{L})| -- ", "gu")
DECIMAL_RE = jsre(
    "(?:(?:равн\\p{L}*|составля\\p{L}*|составил\\p{L}*|достига\\p{L}*|достиг\\p{L}*|точност\\p{L}*|значени\\p{L}*"
    "|около|до|от|≈|=)\\s*)(\\d+\\.\\d+)(?!\\.\\d)|(?<![\\d.])(\\d+\\.\\d+)(?=\\s*(?:%|мс|с\\b|кг|м\\b|км|Гц|ГБ|МБ|раз))",
    "gu",
)
SPLICE_RE = jsre(
    "[\\p{L}\\d)»,] — (?:и (?:это|все|всё|именно|тут|вот)|и\\s+\\p{L}+ (?:меня|нас|всех)"
    "|это (?:меняет|и есть|главное|ключ)|вот (?:что|почему|где|в чем)|именно (?:это|так|поэтому)"
    "|но (?:это|не|именно)|причем|а (?:это|значит|главное))",
    "gu",
)
HEADING_MARK_RE = jsre("^#+\\s*")
TITLE_WORD_RE = jsre("^\\p{Lu}\\p{Ll}", "u")
EMOJI_RE = jsre("\\p{Extended_Pictographic}", "u")
BOLD_RE = jsre("\\*\\*[^*\\n]{1,80}\\*\\*", "g")
BOLD_LEAD_RE = jsre("^\\s*(?:[-*+]\\s+|\\d+[.)]\\s+|[\\p{Lu}\\d]{1,3}\\.\\s+)?$", "u")
BOLD_AFTER_RE = jsre("^\\s*(?:—|:|\\[|$)")
# Номер записи или пункта, выделенный жирным: **12.**, **12)**, **1.2.** — разметка нумерации, а не акцент.
BOLD_NUMBER_RE = jsre("^\\*\\*\\d{1,3}(?:\\.\\d{1,3})*[.)]\\*\\*$")


def _detect_typography(ctx: _Ctx, text_blocks: list[Block]) -> None:
    prepared, mode = ctx.prepared, ctx.mode
    if mode == "chat":
        return
    # Кавычки считаем по тексту без кода (в prose они замаскированы как цитаты). В записях
    # литературы кавычки стоят как в источнике, поэтому служебные части пропускаем.
    for match in STRAIGHT_QUOTES_RE.finditer(prepared.no_code):
        if CYR.search(match.group(1) or "") and not _hits_range(prepared.service, match.start(), len(match.group())):
            _add(
                ctx, "straight-quotes", "straight", "P2", match.start(), match.group(), "«ёлочки» вместо прямых кавычек"
            )
    for match in ENGLISH_QUOTES_RE.finditer(prepared.no_code):
        if CYR.search(match.group(1) or "") and not _hits_range(prepared.service, match.start(), len(match.group())):
            _add(
                ctx,
                "english-quotes",
                "english",
                "P1",
                match.start(),
                match.group(),
                "«ёлочки»; английские кавычки в русском тексте — след машинного перевода",
            )
    for match in HYPHEN_DASH_RE.finditer(prepared.prose):
        _add(ctx, "hyphen-dash", "hyphen", "P2", match.start(), match.group(), "длинное тире с пробелами: « — »", True)
    for match in DECIMAL_RE.finditer(prepared.prose):
        _add(ctx, "decimal-point", "decimal", "P2", match.start(), match.group(), "десятичная запятая: 0,93")
    # Тире-связка: тире перед союзом или частицей, подающими «эффект».
    splice_hits = [(match.start() + 1, match.group()) for match in SPLICE_RE.finditer(prepared.prose)]
    splices = len(splice_hits)
    limit = 2 if mode == "social" else 1
    per500 = splices * 500 / ctx.word_count if ctx.word_count else 0
    if splices > limit or (splices >= 1 and per500 > 1 and mode != "social"):
        for index, text in splice_hits:
            _add(
                ctx,
                "em-dash-splice",
                "splice",
                "P1",
                index,
                text,
                "точка, двоеточие или союз вместо тире-связки; грамматическое тире не трогать",
            )
    # Заголовки: Title Case, эмодзи.
    for block in text_blocks:
        if block.kind != "heading":
            continue
        body = HEADING_MARK_RE.sub("", block.text, count=1)
        title_words = [word for word in words(body) if CYR.search(word) and len(word) >= 4]
        if len(title_words) >= 3 and all(TITLE_WORD_RE.search(word) for word in title_words):
            _add(
                ctx,
                "title-case",
                "title-case",
                "P1",
                block.start,
                body,
                "заглавная только у первого слова и имён собственных",
            )
        if EMOJI_RE.search(body):
            _add(
                ctx,
                "emoji-header",
                "emoji",
                "P2" if mode == "social" else "P1",
                block.start,
                body,
                "убрать эмодзи из заголовка",
            )
    # Жирный: больше двух выделений в разделе прозы.
    section_start = 0
    bold: list[tuple[int, str]] = []

    def flush_bold() -> None:
        nonlocal bold
        if len(bold) > 3 and mode != "technical":
            _add(
                ctx,
                "bold-overuse",
                "bold",
                "P1",
                bold[0][0] if bold else section_start,
                " ".join(text for _, text in bold[:4]),
                "не больше одного выделения на раздел",
            )
        bold = []

    for block in text_blocks:
        if block.kind == "heading":
            flush_bold()
            section_start = block.start
            continue
        if block.kind != "prose":
            continue
        for match in BOLD_RE.finditer(block.text):
            at = match.start()
            line_start = block.text.rfind("\n", 0, at) + 1
            lead = bool(BOLD_LEAD_RE.search(block.text[line_start:at]))
            if lead and BOLD_NUMBER_RE.search(match.group()):
                continue
            after = block.text[match.end() : match.end() + 4]
            # Выделенный термин в начале строки перед «—», «:» или ссылкой — типографика, а не акцент.
            if lead and BOLD_AFTER_RE.search(after):
                continue
            bold.append((block.start + at, match.group()))
    flush_bold()


# ─── Структура ──────────────────────────────────────────────────────────
# Грубый признак глагольной формы; существительные на -ость исключены отдельно.
VERB_HINT = jsre(
    "(?:ет|ют|ит|ят|ем|им|ешь|ишь|ал|ял|ил|ыл|ул|ла|ли|ло|ать|ять|ить|еть|уть|ыть|оть|ться|ется|ются|ится|ятся|ся|сь|ут|ат)$",
    "u",
)
NOT_VERB = jsre("(?:ость|есть|асть)$", "u")
GENITIVE_NOUN = jsre(
    "(?:ени[яюейи]|ани[яюейи]|овани[яюейи]|ции|цию|ция|ости|ость|ств[ауео]|изаци[ияю]|ировани[яюе]|ени|ани)$", "u"
)
# Начало слова задаёт ретроспектива (?<!\p{L}): без неё на длинном слове без пробелов
# движок пробует каждую позицию и откатывается к ней, время растёт кубически.
PASSIVE = jsre(
    "(?:^|\\s)(?:был[аио]?|были|будет|будут)\\s+\\p{L}+(?:ан|ян|ен|ён|т)[аоы]?(?=[\\s,.;:!?]|$)"
    "|(?<!\\p{L})\\p{L}{3,}(?:ано|ено|ены|аны|ана|ена|ято|ыто|иты|ата)(?=[\\s,.;:!?]|$)"
    "|(?<!\\p{L})\\p{L}{3,}(?:ется|ются|ится|ятся)(?=[\\s,.;:!?]|$)",
    "iu",
)
ACTOR = jsre("(?:^|\\s)(?:мы|я|нами|автор\\p{L}*|авторы|нами)(?=[\\s,.;:!?]|$)", "iu")
NXY_RE = jsre(
    "(?<![\\p{L}])(?:(?:это|речь|дело|вопрос|суть|главное)\\s+не\\s+(?:просто\\s+|только\\s+|в\\s+|о\\s+|про\\s+)?"
    "[^.!?\\n]{1,50}?(?:,|\\s—)\\s*(?:а|это|но)\\s)|(?<![\\p{L}])не\\s+просто\\s+[^.!?\\n]{1,50}?,\\s*(?:а|но и|это)\\s",
    "giu",
)
SPLIT_NXY_RE = jsre(
    "(?<![\\p{L}])(?:главное|дело|суть|проблема|секрет)\\s+(?:здесь\\s+|тут\\s+)?не\\s+в\\s+[^.!?\\n]{1,40}\\.\\s+"
    "(?:главное|дело|суть|настоящ\\p{L}*|все\\s+дело|всё\\s+дело)\\s",
    "giu",
)
LETTERS_RE = jsre("[\\p{L}]+", "gu")
SPACES_ONLY_RE = jsre("^\\s+$")
OPENER_RE = jsre(
    "^\\s*(?:кроме того|помимо этого|более того|к тому же|также|таким образом|в свою очередь|однако|при этом"
    "|вместе с тем|следовательно|в то же время|в целом)[,\\s]",
    "iu",
)
SPLIT_WS_RE = jsre("\\s+")
QUESTION_END_RE = jsre("[?:]$")
NEGATION_RE = jsre(
    "(?<![\\p{L}])(?:без\\s+[^,.!?\\n]{1,30},\\s*){2,}без\\s+[^,.!?\\n]{1,30}[.!]"
    "|(?:(?<![\\p{L}])никак\\p{L}+\\s+[^.!?\\n]{1,30}[.!]\\s*){3,}",
    "giu",
)
LIST_MARK_RE = jsre("^\\s*(?:[-*+•]|\\d+[.)])\\s+")
BOLD_MARK_RE = jsre("\\*\\*", "g")
HASHTAG_RE = jsre("(?<![\\p{L}\\d&/])#(?=[\\p{L}_]*\\p{L})[\\p{L}\\d_]{2,}", "gu")
COLOR_RE = jsre("^#[0-9a-f]{6}$|^#[0-9a-f]{3}$", "i")
DIGIT_RE = jsre("\\d")
PRONOUNS = frozenset(("он", "она", "они", "я", "мы", "оно"))


def _detect_structure(ctx: _Ctx, text_blocks: list[Block], prose_sentences: list[list[Sentence]]) -> None:
    mode = ctx.mode
    prose = ctx.prepared.prose

    # «Не X, а Y», «это не X — это Y», «не просто X, а Y», разнесённая форма.
    for match in NXY_RE.finditer(prose):
        _add(
            ctx,
            "not-x-but-y",
            "joined",
            "P1",
            match.start(),
            match.group(),
            "прямое утверждение без противопоставления",
        )
    for match in SPLIT_NXY_RE.finditer(prose):
        _add(ctx, "not-x-but-y", "split", "P1", match.start(), match.group(), "разнесённое «не X. Y» — сказать Y прямо")

    # Цепочки отглагольных существительных: 4+ подряд.
    if mode != "chat":
        min_chain = 5 if mode == "academic" else 4
        for block in text_blocks:
            if block.kind not in ("prose", "list"):
                continue
            letter_words = list(LETTERS_RE.finditer(block.text))
            run: list[regex.Match[str]] = []

            def flush_chain(block: Block = block) -> None:
                nonlocal run
                if len(run) >= min_chain:
                    start, end = run[0].start(), run[-1].end()
                    _add(
                        ctx,
                        "genitive-chain",
                        "chain",
                        "P1",
                        block.start + start,
                        block.text[start:end],
                        "переписать через глагол: «обеспечение повышения эффективности…» → «чтобы точнее…»",
                    )
                run = []

            for i, word_match in enumerate(letter_words):
                gap = block.text[letter_words[i - 1].end() : word_match.start()] if i else " "
                word = word_match.group().lower()
                nounish = bool(GENITIVE_NOUN.search(word)) and len(word) > 5
                if nounish and (not run or SPACES_ONLY_RE.search(gap)):
                    run.append(word_match)
                else:
                    flush_chain()
                    if nounish:
                        run.append(word_match)
            flush_chain()

    # Пассив без деятеля: 4+ предложения подряд (6+ в научном режиме).
    if mode not in ("chat", "technical"):
        need = 6 if mode == "academic" else 4
        for block_sentences in prose_sentences:
            passive_run: list[Sentence] = []

            def flush_passive() -> None:
                nonlocal passive_run
                if len(passive_run) >= need:
                    _add(
                        ctx,
                        "passive-run",
                        "passive",
                        "P2",
                        passive_run[0].start,
                        " ".join(sentence.text for sentence in passive_run)[:160],
                        "назвать деятеля, если источник его называет, или перемешать с активными конструкциями",
                    )
                passive_run = []

            for sentence in block_sentences:
                if PASSIVE.search(sentence.text) and not ACTOR.search(sentence.text):
                    passive_run.append(sentence)
                else:
                    flush_passive()
            flush_passive()

    # Переходы в начале абзацев подряд.
    if mode not in ("chat", "social"):
        opener_run: list[Block] = []

        def flush_openers() -> None:
            nonlocal opener_run
            if len(opener_run) >= 3:
                _add(
                    ctx,
                    "transition-run",
                    "openers",
                    "P1",
                    opener_run[0].start,
                    " / ".join(" ".join(SPLIT_WS_RE.split(trim(block.text))[:2]) for block in opener_run),
                    "связь должна быть видна из содержания, а не из союза в начале каждого абзаца",
                )
            opener_run = []

        for block in text_blocks:
            if block.kind != "prose":
                continue
            if OPENER_RE.search(block.text):
                opener_run.append(block)
            else:
                flush_openers()
        flush_openers()

    def first_word(sentence: Sentence | None) -> str:
        if sentence is None:
            return ""
        sentence_words = words(sentence.text)
        return sentence_words[0].lower() if sentence_words else ""

    for block_sentences in prose_sentences:
        # Одинаковые начала: 3+ предложения подряд с одним первым словом.
        i = 0
        while i + 2 < len(block_sentences):
            opener = first_word(block_sentences[i])
            if (
                opener
                and len(opener) > 1
                and opener not in PRONOUNS
                and opener == first_word(block_sentences[i + 1])
                and opener == first_word(block_sentences[i + 2])
            ):
                _add(
                    ctx,
                    "same-opener",
                    opener,
                    "P2",
                    block_sentences[i].start,
                    f"{block_sentences[i].text} {block_sentences[i + 1].text}",
                    "оставить первое, остальные перестроить",
                )
                i += 2
            i += 1
        # Рубленые фрагменты: 3+ подряд по 1–4 слова.
        if mode != "social":
            staccato_run: list[Sentence] = []

            def flush_staccato() -> None:
                nonlocal staccato_run
                if len(staccato_run) >= 3:
                    _add(
                        ctx,
                        "staccato",
                        "fragments",
                        "P2",
                        staccato_run[0].start,
                        " ".join(sentence.text for sentence in staccato_run),
                        "оставить один акцентный фрагмент, остальное собрать в предложения",
                    )
                staccato_run = []

            for sentence in block_sentences:
                if sentence.words <= 4 and not QUESTION_END_RE.search(sentence.text):
                    staccato_run.append(sentence)
                else:
                    flush_staccato()
            flush_staccato()
        # Стопка вопросов: 3+ вопросительных предложения подряд.
        questions: list[Sentence] = []

        def flush_questions() -> None:
            nonlocal questions
            if len(questions) >= 3:
                _add(
                    ctx,
                    "stacked-questions",
                    "questions",
                    "P2",
                    questions[0].start,
                    " ".join(sentence.text for sentence in questions),
                    "оставить не больше одного вопроса и ответить",
                )
            questions = []

        for sentence in block_sentences:
            if sentence.text.endswith("?"):
                questions.append(sentence)
            else:
                flush_questions()
        flush_questions()

    # Цепочка отрицаний: «Без X, без Y, без Z.» / «Никаких X. Никаких Y.»
    for match in NEGATION_RE.finditer(prose):
        _add(
            ctx,
            "negation-chain",
            "negations",
            "P2",
            match.start(),
            match.group(),
            "сказать, что это есть, а не чем оно не является",
        )

    # Списки из голых именных групп: 5+ коротких пунктов без глаголов.
    if mode not in ("chat", "technical"):
        for block in text_blocks:
            if block.kind != "list":
                continue
            items = [trim(BOLD_MARK_RE.sub("", LIST_MARK_RE.sub("", line, count=1))) for line in block.text.split("\n")]
            items = [item for item in items if item]
            if len(items) < 5:
                continue

            def bare(item: str) -> bool:
                item_words = words(item)
                return 0 < len(item_words) <= 6 and not any(
                    len(word) > 3 and VERB_HINT.search(word.lower()) and not NOT_VERB.search(word.lower())
                    for word in item_words
                )

            bare_items = [item for item in items if bare(item)]
            if len(bare_items) >= 5 and len(bare_items) / len(items) >= 0.8:
                _add(
                    ctx,
                    "bullet-np-list",
                    "np-list",
                    "P1",
                    block.start,
                    " / ".join(items[:3]),
                    "полные утверждения с данными из источника или проза",
                )

    # Хэштеги: 6+ (без номеров задач и цветов).
    tags = [
        match
        for match in HASHTAG_RE.finditer(prose)
        if not COLOR_RE.search(match.group()) or not DIGIT_RE.search(match.group())
    ]
    if len(tags) >= 6:
        _add(
            ctx,
            "hashtag-stuffing",
            "hashtags",
            "P1" if mode == "social" else "P0",
            tags[0].start(),
            " ".join(tag.group() for tag in tags[:8]),
            "два-три конкретных тега или ни одного",
        )


# ─── Стилометрия ────────────────────────────────────────────────────────
@dataclass(slots=True)
class _Stylometry:
    sent_cv: float
    para_cv: float
    mean_len: float
    mattr: float


def _detect_stylometry(ctx: _Ctx, text_blocks: list[Block], prose_sentences: list[list[Sentence]]) -> _Stylometry:
    long_sentences = [
        sentence for block_sentences in prose_sentences for sentence in block_sentences if sentence.words >= 3
    ]
    lengths = [sentence.words for sentence in long_sentences]
    sent_cv = cv(lengths)
    mean_len = mean(lengths)
    paragraphs = [block for block in text_blocks if block.kind == "prose" and len(words(block.text)) >= 15]
    para_lens = [len(words(block.text)) for block in paragraphs]
    para_cv = cv(para_lens)
    tokens = words(ctx.prepared.prose)
    diversity = mattr(tokens, 100)

    if ctx.mode not in ("chat", "social"):
        if len(lengths) >= 10 and sent_cv < 0.33 and mean_len >= 10:
            _add(
                ctx,
                "uniform-sentences",
                "sentence-cv",
                "P1",
                long_sentences[0].start if long_sentences else 0,
                f"коэффициент вариации длины предложений {to_fixed(sent_cv, 2)} при средней длине "
                f"{to_fixed(mean_len, 1)} слова",
                "текст метрономичен: смешать короткие и длинные предложения там, где позволяет содержание",
            )
        if len(para_lens) >= 5 and para_cv < 0.2:
            _add(
                ctx,
                "uniform-paragraphs",
                "paragraph-cv",
                "P2",
                paragraphs[0].start if paragraphs else 0,
                f"коэффициент вариации длины абзацев {to_fixed(para_cv, 2)} на {len(para_lens)} абзацах",
                "границы абзацев по смыслу, а не по размеру",
            )
        if len(tokens) >= 300 and diversity < 0.62 and ctx.mode != "technical":
            _add(
                ctx,
                "low-diversity",
                "mattr",
                "P2",
                0,
                f"MATTR {to_fixed(diversity, 2)}",
                "словарь беден для русской прозы: больше конкретики вместо повторяющихся абстракций",
            )
    return _Stylometry(sent_cv, para_cv, mean_len, diversity)


# Находки, которые описывают весь текст или абзац, а не конкретную фразу.
AGGREGATE = frozenset(
    (
        "uniform-sentences uniform-paragraphs low-diversity transition-run passive-run tier2-cluster "
        "phrase3-cluster tier3 invisible-chars soft-hyphen typography-chars model-idiolect truncated"
    ).split()
)


def _dedupe(issues: list[Issue]) -> list[Issue]:
    """Убирает повторы и вложенные совпадения.

    «Играет ключевую роль» и «ключевую роль» — одна находка, остаётся более весомая
    или более длинная.
    """
    seen: set[tuple[str, str, int]] = set()
    kept: list[Issue] = []
    for issue in sorted(issues, key=lambda item: (item.index, -len(item.text))):
        key = (issue.type, issue.rule, issue.index)
        if key in seen:
            continue
        seen.add(key)
        if issue.type not in AGGREGATE:
            end = issue.index + len(issue.text)
            weight = WEIGHTS.get(issue.type, 1)
            if any(
                other.type not in AGGREGATE
                and other.index <= issue.index
                and other.index + len(other.text) >= end
                and WEIGHTS.get(other.type, 1) >= weight
                for other in kept
            ):
                continue
        kept.append(issue)
    return kept


def score_issues(issues: list[Issue], word_count: int) -> int:
    distinct: dict[tuple[str, str], float] = {}
    for issue in issues:
        distinct[(issue.type, issue.text.lower())] = WEIGHTS.get(issue.type, 1)
    raw = total(distinct.values())
    scale = 8 * max(1, math.log2(max(word_count, 1) / 50))
    return js_round(100 * (1 - math.exp(-raw / scale)))


def label_for(score: int) -> str:
    if score < 15:
        return "чисто"
    if score < 40:
        return "есть приметы"
    if score < 70:
        return "много примет"
    return "сильный ИИ-стиль"


@dataclass(slots=True)
class Internals:
    prepared: Prepared
    blocks: list[Block]
    sentences: list[list[Sentence]]


def analyze_internal(source: str, context: ContextMode = "general") -> tuple[AnalysisResult, Internals]:
    prepared = prepare(source)
    text_blocks = blocks(prepared)
    prose_sentences = [sentences(block.text, block.start) for block in text_blocks if block.kind == "prose"]
    word_count = len(words(prepared.prose))
    ctx = _Ctx(prepared, context, word_count)

    _detect_fingerprints(ctx)
    _detect_chat_leftovers(ctx, text_blocks)
    _detect_lexicon(ctx, text_blocks)
    _detect_typography(ctx, text_blocks)
    _detect_structure(ctx, text_blocks, prose_sentences)
    stylometry = _detect_stylometry(ctx, text_blocks, prose_sentences)

    issues = _dedupe(ctx.issues)
    score = score_issues(issues, word_count)
    result = AnalysisResult(
        score=score,
        label=label_for(score),
        issues=issues,
        suspicious=bool(prepared.invisible) or bool(prepared.homoglyphs),
        light=light_for(issues),
        stats=Stats(
            words=word_count,
            sentences=sum(len(block_sentences) for block_sentences in prose_sentences),
            paragraphs=sum(1 for block in text_blocks if block.kind == "prose"),
            mean_sentence_length=fixed(stylometry.mean_len, 2),
            sentence_length_cv=fixed(stylometry.sent_cv, 3),
            paragraph_length_cv=fixed(stylometry.para_cv, 3),
            mattr=fixed(stylometry.mattr, 3),
            context_mode=context,
        ),
    )
    return result, Internals(prepared, text_blocks, prose_sentences)


# Светофор символов. Красный — следы чата (разметка ссылок, параметры ИИ-инструментов в ссылках)
# и маскировка (невидимые символы, подмена букв): удалять без вопросов. Жёлтый — редкие
# типографские символы и мягкие переносы: их ставят и модели, и вёрстка, решает пользователь.
LIGHT_TYPES: dict[LightLevel, tuple[str, ...]] = {
    "red": ("chat-markup", "ai-url", "invisible-chars", "homoglyph"),
    "yellow": ("typography-chars", "soft-hyphen"),
}
LIGHT_HINTS: dict[LightLevel, str] = {
    "green": "всё чисто",
    "yellow": "ИИ или типография? Уточни у пользователя, убрать или оставить",
    "red": "почти наверняка ИИ или маскировка: удалить",
}


def light_for(issues: list[Issue]) -> Light:
    """Цвет по самым серьёзным находкам: красный сильнее жёлтого, без них — зелёный."""
    present = {issue.type for issue in issues}
    found = {level: [kind for kind in kinds if kind in present] for level, kinds in LIGHT_TYPES.items()}
    level: LightLevel = "red" if found["red"] else "yellow" if found["yellow"] else "green"
    return Light(level=level, hint=LIGHT_HINTS[level], types=found["red"] + found["yellow"])


def analyze(source: str, context: ContextMode = "general") -> AnalysisResult:
    return analyze_internal(source, context)[0]

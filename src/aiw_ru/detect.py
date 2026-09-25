"""Детектор ИИ-стиля для русского текста.

Модель оценки. У каждой категории есть вес (WEIGHTS). Находки дедуплицируются
по паре (тип, текст); сырой балл — сумма весов уникальных находок. Правки ради
краткости (style_only) весят мало: это совет по стилю, а не довод об авторстве.
Сырой балл нормируется по длине текста и насыщается к 100, чтобы длинный текст
с той же плотностью примет не набирал бесконечно.
"""

from __future__ import annotations

import math
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
from aiw_ru.text import Block, Prepared, Sentence, blocks, cv, line_col, mattr, mean, plural, prepare, sentences, words
from aiw_ru.types import AnalysisResult, ContextMode, Issue, Severity, Stats

WEIGHTS: dict[str, float] = {
    # Невидимые символы, подмена букв и мягкие переносы — гигиена документа, а не ИИ-стиль:
    # в человеческих текстах LLMTrace они встречаются чаще, чем в сгенерированных. В оценку
    # не идут; первые два делают документ подозрительным (флаг `suspicious`).
    "invisible-chars": 0,
    "homoglyph": 0,
    "soft-hyphen": 0,
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
    p: Prepared
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
    src = ctx.p.to_source[index] if index < len(ctx.p.to_source) else index
    # Фрагмент показывается как набран в исходнике (с «ё», невидимыми символами и подменёнными
    # буквами), чтобы его можно было найти в файле. Сводки вроде «три штампа» остаются как есть.
    end = index + len(text)
    if text and end <= len(ctx.p.text) and ctx.p.text[index:end] == text:
        text = ctx.p.source[src : ctx.p.to_source[end - 1] + 1]
    line, column = line_col(ctx.p.line_starts, src)
    ctx.issues.append(
        Issue(
            type=type_,
            rule=rule,
            severity=severity,
            text=trim(text)[:160],
            index=src,
            line=line,
            column=column,
            hint=hint,
            style_only=style_only,
        )
    )


def _applies(e: LexEntry, mode: ContextMode) -> bool:
    return mode not in e.skip


def _ranges(pattern: regex.Pattern[str], text: str) -> list[tuple[int, int]]:
    """Отрезки, где совпадает выражение; строится один раз на текст."""
    return [m.span() for m in pattern.finditer(text)]


def _hits_range(rs: list[tuple[int, int]], index: int, length: int) -> bool:
    """Пересекает ли [index, index+length) хотя бы один отрезок (отрезки отсортированы)."""
    lo, hi = 0, len(rs)
    while lo < hi:
        mid = (lo + hi) >> 1
        if rs[mid][1] <= index:
            lo = mid + 1
        else:
            hi = mid
    return lo < len(rs) and rs[lo][0] < index + length


# ─── Словарные правила ──────────────────────────────────────────────────
_CLUSTER_ONLY = frozenset(id(e) for e in (*TIER2, *TIER3, *PHRASE3, *IDIOLECT))
# Почерк модели: находка, если в тексте столько разных оборотов. Один оборот есть
# у 1,9 % человеческих текстов LLMTrace, два и больше — у 0,1 % (у моделей 1,5 %).
IDIOLECT_MIN = 2


def _detect_lexicon(ctx: _Ctx, bs: list[Block]) -> None:
    p, mode = ctx.p, ctx.mode

    def per1000(n: int) -> float:
        return n * 1000 / ctx.word_count if ctx.word_count else 0

    if mode == "academic":
        exempt = _ranges(ACADEMIC_FORMULAS, p.prose)
    elif mode == "technical":
        exempt = _ranges(TECHNICAL_TERMS, p.prose)
    else:
        exempt = []
    for e in ALL_LEXICON:
        if id(e) in _CLUSTER_ONLY or not _applies(e, mode):
            continue
        if mode == "chat" and e.severity != "P0" and e.type != "calque":
            continue
        hits = list(e.re.finditer(p.prose))
        if not hits:
            continue
        threshold = e.density.get(mode, e.density.get("default"))
        if threshold is not None and per1000(len(hits)) < threshold:
            continue
        for m in hits:
            if _hits_range(exempt, m.start(), len(m.group())):
                continue
            _add(ctx, e.type, e.id, e.severity, m.start(), m.group(), e.hint, e.style_only)

    # Уровень 2: скопление двух и более разных слов в одном абзаце.
    if mode != "chat":
        for b in bs:
            if b.kind not in ("prose", "list"):
                continue
            found: dict[str, tuple[int, str]] = {}
            for e in TIER2:
                m = e.re.search(b.text)
                if m and e.id not in found:
                    if _hits_range(exempt, b.start + m.start(), len(m.group())):
                        continue
                    found[e.id] = (b.start + m.start(), m.group())
            if len(found) >= 2:
                hits2 = list(found.values())
                _add(
                    ctx,
                    "tier2-cluster",
                    "tier2",
                    "P2",
                    hits2[0][0],
                    ", ".join(t for _, t in hits2),
                    "перефразировать проще, оставить одно слово или дать конкретику",
                )

    # Уровень 3: суммарная плотность ≥ 3 % слов.
    t3 = 0
    first_t3 = -1
    t3_words: list[str] = []
    for e in TIER3:
        for m in e.re.finditer(p.prose):
            t3 += 1
            if first_t3 < 0 or m.start() < first_t3:
                first_t3 = m.start()
            if len(t3_words) < 6:
                t3_words.append(m.group())
    if ctx.word_count >= 150 and t3 / ctx.word_count >= 0.03:
        _add(
            ctx,
            "tier3",
            "density",
            "P2",
            max(first_t3, 0),
            ", ".join(t3_words),
            f"перегруженные оценочные слова: {t3} на {ctx.word_count} слов — заменить часть числами и примерами",
        )

    # Фразовые штампы: одна фраза 2+ раза или 3+ разных.
    phrase_hits: dict[str, list] = {}
    for e in PHRASE3:
        for m in e.re.finditer(p.prose):
            cur = phrase_hits.setdefault(e.id, [0, m.start(), m.group()])
            cur[0] += 1
    for id_, (n, idx, text) in phrase_hits.items():
        if n >= 2:
            _add(ctx, "phrase3", id_, "P2", idx, text, f"штамп повторён {n} раза — назвать конкретику")
    if len(phrase_hits) >= 3:
        first = min(phrase_hits.values(), key=lambda h: h[1])
        _add(
            ctx,
            "phrase3-cluster",
            "cluster",
            "P1",
            first[1],
            ", ".join(h[2] for h in phrase_hits.values()),
            "три и больше разных штампов — так варьирует шаблоны модель",
        )

    # Почерк модели: считается число разных оборотов, а не плотность. Каждый оборот
    # даёт одну находку в первом месте, где встретился.
    if mode != "chat":
        idioms: dict[str, tuple[int, str]] = {}
        for e in IDIOLECT:
            if not _applies(e, mode):
                continue
            for m in e.re.finditer(p.prose):
                if not _hits_range(exempt, m.start(), len(m.group())):
                    idioms[e.id] = (m.start(), m.group())
                    break
        if len(idioms) >= IDIOLECT_MIN:
            for id_, (idx, text) in idioms.items():
                _add(
                    ctx,
                    "model-idiolect",
                    id_,
                    "P1",
                    idx,
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
    "\\[(?:Ваш[аеи]?|Вставьте|Укажите|Добавьте|Введите|Опишите|Название|Имя|Фамилия|Дата|ДАТА|Источник|Ссылка"
    "|Your|Insert|Add|Enter)[^\\]\\n]{0,60}\\]|\\b(?:19|20)XX\\b|\\b\\d{4}-XX-XX\\b"
    "|<!--\\s*(?:добавь|добавьте|вставь|вставьте|todo|TODO|заполни|укажите|add|insert)[^>]*-->",
    "g",
)


def _detect_fingerprints(ctx: _Ctx) -> None:
    p = ctx.p
    if p.invisible:
        idx = p.invisible[0].index
        line, column = line_col(p.line_starts, idx)
        ctx.issues.append(
            Issue(
                type="invisible-chars",
                rule="zero-width",
                severity="P0",
                text=plural(len(p.invisible), "невидимый символ", "невидимых символа", "невидимых символов"),
                index=idx,
                line=line,
                column=column,
                hint="удалить; такие документы системы проверки помечают как подозрительные",
            )
        )
    if p.soft_hyphens:
        idx = p.soft_hyphens[0]
        line, column = line_col(p.line_starts, idx)
        ctx.issues.append(
            Issue(
                type="soft-hyphen",
                rule="soft-hyphen",
                severity="P2",
                text=plural(len(p.soft_hyphens), "мягкий перенос", "мягких переноса", "мягких переносов"),
                index=idx,
                line=line,
                column=column,
                hint="удалить перед сдачей; обычно их оставляют Word и копирование из PDF",
            )
        )
    for h in p.homoglyphs:
        line, column = line_col(p.line_starts, h.index)
        ctx.issues.append(
            Issue(
                type="homoglyph",
                rule="latin-in-cyrillic",
                severity="P0",
                text=h.word,
                index=h.index,
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
        for m in pattern.finditer(p.no_code):
            _add(ctx, type_, type_, "P0", m.start(), m.group(), hint)


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


def _detect_chat_leftovers(ctx: _Ctx, bs: list[Block]) -> None:
    mode = ctx.mode
    if mode == "chat":
        return
    if mode != "technical":
        for m in LEAD_IN_RE.finditer(ctx.p.prose):
            _add(ctx, "chat-wrapper", "подводка", "P1", m.start(), m.group(), "удалить подводку, оставить сам текст")
    # Обрыв генерации: последний блок текста — абзац прозы, его последняя строка длинная
    # и кончается запятой или словом, на котором фраза кончиться не может. Заголовок,
    # список, таблица, цитата, код и короткая подпись («С уважением,») не считаются.
    last = next((b for b in reversed(bs) if b.kind != "empty"), None)
    if last is None or last.kind != "prose":
        return
    raw = ctx.p.text[last.start : last.end]
    line = trim(raw[raw.rfind("\n") + 1 :])
    if len(words(line)) < TRUNCATED_MIN_WORDS:
        return
    if TRAILING_COMMA_RE.search(line) or DANGLING_RE.search(line):
        # Находка показывает последние шесть слов строки.
        tail = line[[t.start() for t in TOKEN_RE.finditer(line)][-6:][0] :]
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


def _detect_typography(ctx: _Ctx, bs: list[Block]) -> None:
    p, mode = ctx.p, ctx.mode
    if mode == "chat":
        return
    # Кавычки считаем по тексту без кода (в prose они замаскированы как цитаты).
    for m in STRAIGHT_QUOTES_RE.finditer(p.no_code):
        if CYR.search(m.group(1) or ""):
            _add(ctx, "straight-quotes", "straight", "P2", m.start(), m.group(), "«ёлочки» вместо прямых кавычек")
    for m in ENGLISH_QUOTES_RE.finditer(p.no_code):
        if CYR.search(m.group(1) or ""):
            _add(
                ctx,
                "english-quotes",
                "english",
                "P1",
                m.start(),
                m.group(),
                "«ёлочки»; английские кавычки в русском тексте — след машинного перевода",
            )
    for m in HYPHEN_DASH_RE.finditer(p.prose):
        _add(ctx, "hyphen-dash", "hyphen", "P2", m.start(), m.group(), "длинное тире с пробелами: « — »", True)
    for m in DECIMAL_RE.finditer(p.prose):
        _add(ctx, "decimal-point", "decimal", "P2", m.start(), m.group(), "десятичная запятая: 0,93")
    # Тире-связка: тире перед союзом или частицей, подающими «эффект».
    splice_hits = [(m.start() + 1, m.group()) for m in SPLICE_RE.finditer(p.prose)]
    splices = len(splice_hits)
    limit = 2 if mode == "social" else 1
    per500 = splices * 500 / ctx.word_count if ctx.word_count else 0
    if splices > limit or (splices >= 1 and per500 > 1 and mode != "social"):
        for idx, text in splice_hits:
            _add(
                ctx,
                "em-dash-splice",
                "splice",
                "P1",
                idx,
                text,
                "точка, двоеточие или союз вместо тире-связки; грамматическое тире не трогать",
            )
    # Заголовки: Title Case, эмодзи.
    for b in bs:
        if b.kind != "heading":
            continue
        body = HEADING_MARK_RE.sub("", b.text, count=1)
        ws = [w for w in words(body) if CYR.search(w) and len(w) >= 4]
        if len(ws) >= 3 and all(TITLE_WORD_RE.search(w) for w in ws):
            _add(
                ctx,
                "title-case",
                "title-case",
                "P1",
                b.start,
                body,
                "заглавная только у первого слова и имён собственных",
            )
        if EMOJI_RE.search(body):
            _add(
                ctx,
                "emoji-header",
                "emoji",
                "P2" if mode == "social" else "P1",
                b.start,
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
                " ".join(t for _, t in bold[:4]),
                "не больше одного выделения на раздел",
            )
        bold = []

    for b in bs:
        if b.kind == "heading":
            flush_bold()
            section_start = b.start
            continue
        if b.kind != "prose":
            continue
        for m in BOLD_RE.finditer(b.text):
            at = m.start()
            line_start = b.text.rfind("\n", 0, at) + 1
            lead = bool(BOLD_LEAD_RE.search(b.text[line_start:at]))
            after = b.text[m.end() : m.end() + 4]
            # Выделенный термин в начале строки перед «—», «:» или ссылкой — типографика, а не акцент.
            if lead and BOLD_AFTER_RE.search(after):
                continue
            bold.append((b.start + at, m.group()))
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


def _detect_structure(ctx: _Ctx, bs: list[Block], prose_sentences: list[list[Sentence]]) -> None:
    mode = ctx.mode
    prose = ctx.p.prose

    # «Не X, а Y», «это не X — это Y», «не просто X, а Y», разнесённая форма.
    for m in NXY_RE.finditer(prose):
        _add(ctx, "not-x-but-y", "joined", "P1", m.start(), m.group(), "прямое утверждение без противопоставления")
    for m in SPLIT_NXY_RE.finditer(prose):
        _add(ctx, "not-x-but-y", "split", "P1", m.start(), m.group(), "разнесённое «не X. Y» — сказать Y прямо")

    # Цепочки отглагольных существительных: 4+ подряд.
    if mode != "chat":
        min_chain = 5 if mode == "academic" else 4
        for b in bs:
            if b.kind not in ("prose", "list"):
                continue
            toks = list(LETTERS_RE.finditer(b.text))
            run: list[regex.Match[str]] = []

            def flush_chain(b: Block = b) -> None:
                nonlocal run
                if len(run) >= min_chain:
                    s, e = run[0].start(), run[-1].end()
                    _add(
                        ctx,
                        "genitive-chain",
                        "chain",
                        "P1",
                        b.start + s,
                        b.text[s:e],
                        "переписать через глагол: «обеспечение повышения эффективности…» → «чтобы точнее…»",
                    )
                run = []

            for i, t in enumerate(toks):
                gap = b.text[toks[i - 1].end() : t.start()] if i else " "
                w = t.group().lower()
                nounish = bool(GENITIVE_NOUN.search(w)) and len(w) > 5
                if nounish and (not run or SPACES_ONLY_RE.search(gap)):
                    run.append(t)
                else:
                    flush_chain()
                    if nounish:
                        run.append(t)
            flush_chain()

    # Пассив без деятеля: 4+ предложения подряд (6+ в научном режиме).
    if mode not in ("chat", "technical"):
        need = 6 if mode == "academic" else 4
        for ss in prose_sentences:
            prun: list[Sentence] = []

            def flush_passive() -> None:
                nonlocal prun
                if len(prun) >= need:
                    _add(
                        ctx,
                        "passive-run",
                        "passive",
                        "P2",
                        prun[0].start,
                        " ".join(s.text for s in prun)[:160],
                        "назвать деятеля, если источник его называет, или перемешать с активными конструкциями",
                    )
                prun = []

            for s in ss:
                if PASSIVE.search(s.text) and not ACTOR.search(s.text):
                    prun.append(s)
                else:
                    flush_passive()
            flush_passive()

    # Переходы в начале абзацев подряд.
    if mode not in ("chat", "social"):
        brun: list[Block] = []

        def flush_openers() -> None:
            nonlocal brun
            if len(brun) >= 3:
                _add(
                    ctx,
                    "transition-run",
                    "openers",
                    "P1",
                    brun[0].start,
                    " / ".join(" ".join(SPLIT_WS_RE.split(trim(b.text))[:2]) for b in brun),
                    "связь должна быть видна из содержания, а не из союза в начале каждого абзаца",
                )
            brun = []

        for b in bs:
            if b.kind != "prose":
                continue
            if OPENER_RE.search(b.text):
                brun.append(b)
            else:
                flush_openers()
        flush_openers()

    def first_word(s: Sentence | None) -> str:
        if s is None:
            return ""
        ws = words(s.text)
        return ws[0].lower() if ws else ""

    for ss in prose_sentences:
        # Одинаковые начала: 3+ предложения подряд с одним первым словом.
        i = 0
        while i + 2 < len(ss):
            a = first_word(ss[i])
            if a and len(a) > 1 and a not in PRONOUNS and a == first_word(ss[i + 1]) and a == first_word(ss[i + 2]):
                _add(
                    ctx,
                    "same-opener",
                    a,
                    "P2",
                    ss[i].start,
                    f"{ss[i].text} {ss[i + 1].text}",
                    "оставить первое, остальные перестроить",
                )
                i += 2
            i += 1
        # Рубленые фрагменты: 3+ подряд по 1–4 слова.
        if mode != "social":
            srun: list[Sentence] = []

            def flush_staccato() -> None:
                nonlocal srun
                if len(srun) >= 3:
                    _add(
                        ctx,
                        "staccato",
                        "fragments",
                        "P2",
                        srun[0].start,
                        " ".join(s.text for s in srun),
                        "оставить один акцентный фрагмент, остальное собрать в предложения",
                    )
                srun = []

            for s in ss:
                if s.words <= 4 and not QUESTION_END_RE.search(s.text):
                    srun.append(s)
                else:
                    flush_staccato()
            flush_staccato()
        # Стопка вопросов: 3+ вопросительных предложения подряд.
        q: list[Sentence] = []

        def flush_questions() -> None:
            nonlocal q
            if len(q) >= 3:
                _add(
                    ctx,
                    "stacked-questions",
                    "questions",
                    "P2",
                    q[0].start,
                    " ".join(s.text for s in q),
                    "оставить не больше одного вопроса и ответить",
                )
            q = []

        for s in ss:
            if s.text.endswith("?"):
                q.append(s)
            else:
                flush_questions()
        flush_questions()

    # Цепочка отрицаний: «Без X, без Y, без Z.» / «Никаких X. Никаких Y.»
    for m in NEGATION_RE.finditer(prose):
        _add(
            ctx,
            "negation-chain",
            "negations",
            "P2",
            m.start(),
            m.group(),
            "сказать, что это есть, а не чем оно не является",
        )

    # Списки из голых именных групп: 5+ коротких пунктов без глаголов.
    if mode not in ("chat", "technical"):
        for b in bs:
            if b.kind != "list":
                continue
            items = [trim(BOLD_MARK_RE.sub("", LIST_MARK_RE.sub("", line, count=1))) for line in b.text.split("\n")]
            items = [it for it in items if it]
            if len(items) < 5:
                continue

            def bare(it: str) -> bool:
                ws = words(it)
                return 0 < len(ws) <= 6 and not any(
                    len(w) > 3 and VERB_HINT.search(w.lower()) and not NOT_VERB.search(w.lower()) for w in ws
                )

            bare_items = [it for it in items if bare(it)]
            if len(bare_items) >= 5 and len(bare_items) / len(items) >= 0.8:
                _add(
                    ctx,
                    "bullet-np-list",
                    "np-list",
                    "P1",
                    b.start,
                    " / ".join(items[:3]),
                    "полные утверждения с данными из источника или проза",
                )

    # Хэштеги: 6+ (без номеров задач и цветов).
    tags = [m for m in HASHTAG_RE.finditer(prose) if not COLOR_RE.search(m.group()) or not DIGIT_RE.search(m.group())]
    if len(tags) >= 6:
        _add(
            ctx,
            "hashtag-stuffing",
            "hashtags",
            "P1" if mode == "social" else "P0",
            tags[0].start(),
            " ".join(m.group() for m in tags[:8]),
            "два-три конкретных тега или ни одного",
        )


# ─── Стилометрия ────────────────────────────────────────────────────────
@dataclass(slots=True)
class _Stylometry:
    sent_cv: float
    para_cv: float
    mean_len: float
    mattr: float


def _detect_stylometry(ctx: _Ctx, bs: list[Block], prose_sentences: list[list[Sentence]]) -> _Stylometry:
    all_ = [s for ss in prose_sentences for s in ss if s.words >= 3]
    lengths = [s.words for s in all_]
    sent_cv = cv(lengths)
    mean_len = mean(lengths)
    paras = [b for b in bs if b.kind == "prose" and len(words(b.text)) >= 15]
    para_lens = [len(words(b.text)) for b in paras]
    para_cv = cv(para_lens)
    tokens = words(ctx.p.prose)
    diversity = mattr(tokens, 100)

    if ctx.mode not in ("chat", "social"):
        if len(lengths) >= 10 and sent_cv < 0.33 and mean_len >= 10:
            _add(
                ctx,
                "uniform-sentences",
                "sentence-cv",
                "P1",
                all_[0].start if all_ else 0,
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
                paras[0].start if paras else 0,
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
        "phrase3-cluster tier3 invisible-chars soft-hyphen model-idiolect truncated"
    ).split()
)


def _dedupe(issues: list[Issue]) -> list[Issue]:
    """Убирает повторы и вложенные совпадения.

    «Играет ключевую роль» и «ключевую роль» — одна находка, остаётся более весомая
    или более длинная.
    """
    seen: set[tuple[str, str, int]] = set()
    kept: list[Issue] = []
    for i in sorted(issues, key=lambda x: (x.index, -len(x.text))):
        key = (i.type, i.rule, i.index)
        if key in seen:
            continue
        seen.add(key)
        if i.type not in AGGREGATE:
            end = i.index + len(i.text)
            w = WEIGHTS.get(i.type, 1)
            if any(
                k.type not in AGGREGATE
                and k.index <= i.index
                and k.index + len(k.text) >= end
                and WEIGHTS.get(k.type, 1) >= w
                for k in kept
            ):
                continue
        kept.append(i)
    return kept


def score_issues(issues: list[Issue], word_count: int) -> int:
    distinct: dict[tuple[str, str], float] = {}
    for i in issues:
        distinct[(i.type, i.text.lower())] = WEIGHTS.get(i.type, 1)
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
    p = prepare(source)
    bs = blocks(p)
    prose_sentences = [sentences(b.text, b.start) for b in bs if b.kind == "prose"]
    word_count = len(words(p.prose))
    ctx = _Ctx(p, context, word_count)

    _detect_fingerprints(ctx)
    _detect_chat_leftovers(ctx, bs)
    _detect_lexicon(ctx, bs)
    _detect_typography(ctx, bs)
    _detect_structure(ctx, bs, prose_sentences)
    st = _detect_stylometry(ctx, bs, prose_sentences)

    issues = _dedupe(ctx.issues)
    score = score_issues(issues, word_count)
    result = AnalysisResult(
        score=score,
        label=label_for(score),
        issues=issues,
        suspicious=bool(p.invisible) or bool(p.homoglyphs),
        stats=Stats(
            words=word_count,
            sentences=sum(len(ss) for ss in prose_sentences),
            paragraphs=sum(1 for b in bs if b.kind == "prose"),
            mean_sentence_length=fixed(st.mean_len, 2),
            sentence_length_cv=fixed(st.sent_cv, 3),
            paragraph_length_cv=fixed(st.para_cv, 3),
            mattr=fixed(st.mattr, 3),
            context_mode=context,
        ),
    )
    return result, Internals(p, bs, prose_sentences)


def analyze(source: str, context: ContextMode = "general") -> AnalysisResult:
    return analyze_internal(source, context)[0]

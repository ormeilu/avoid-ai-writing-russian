"""Оценка по фрагментам в духе модулей поиска сгенерированного текста «Антиплагиата» и «Думейта».

Настоящие классификаторы закрыты, поэтому здесь приближение: каждый фрагмент
(абзац или склейка коротких абзацев) описывается набором интерпретируемых
признаков, логистическая модель переводит их в вероятность, а доля ИИ-текста
считается как доля знаков во фрагментах выше порога — так же выглядит итог в
отчёте системы.

Веса по умолчанию подобраны вручную. Команда `calibrate` дообучает их на
фрагментах, которые система реально подсветила в ваших отчётах; у каждой системы
своя калибровка.
"""

from __future__ import annotations

import json
import math
import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal

from pydantic import Field

from aiw_ru.compat import fixed, jsre, total, trim
from aiw_ru.detect import WEIGHTS, Internals, analyze_internal
from aiw_ru.text import PLAIN_SPACES, cv, line_col, mattr, mean, sentences, words
from aiw_ru.types import ContextMode, Issue, Light, Record

FEATURE_NAMES: tuple[str, ...] = (
    "плотность примет",
    "однообразие длины предложений",
    "типичная длина предложения",
    "канцелярит",
    "бедная пунктуация",
    "однообразные начала предложений",
    "бедный словарь",
)


System = Literal["antiplagiat", "domate"]

# Системы проверки, под которые калибруется оценка: название и родительный падеж для подсказок.
SYSTEMS: dict[System, tuple[str, str]] = {
    "antiplagiat": ("«Антиплагиат»", "«Антиплагиата»"),
    "domate": ("«Думейт»", "«Думейта»"),
}


class Model(Record):
    bias: float
    weights: list[float]
    threshold: float


DEFAULT_MODEL = Model(bias=-4.4, weights=[1.1, 2.6, 1.4, 0.35, 1.2, 1.3, 1.6], threshold=0.5)


class Fragment(Record):
    start: int
    end: int
    line: int
    end_line: int
    words: int
    features: list[float]
    probability: float
    ai: bool
    reasons: list[str]
    preview: str


class AntiplagiatReport(Record):
    # Доля знаков текста во фрагментах, помеченных как ИИ, 0–100.
    ai_share: float
    fragments: list[Fragment]
    suspicious: bool
    suspicious_reasons: list[str]
    # Светофор символов и следов чата, как у scan.
    light: Light
    model: Literal["default", "calibrated"]
    threshold: float
    # Система, под которую калибровалась модель (`--system`).
    system: System = "antiplagiat"


MIN_FRAGMENT_WORDS = 40
RARE_PUNCT_RE = jsre("[;:()!?…]|\\s—\\s")
SPACES_RE = jsre("\\s+", "g")


def _sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z))


def predict(model: Model, f: list[float]) -> float:
    logit = model.bias
    for i, value in enumerate(f):
        logit += (model.weights[i] if i < len(model.weights) else 0) * value
    return _sigmoid(logit)


def features_for(text: str, fragment_issues: list[Issue]) -> list[float]:
    text_words = words(text)
    word_count = max(len(text_words), 1)
    fragment_sentences = [sentence for sentence in sentences(text) if sentence.words >= 2]
    lengths = [sentence.words for sentence in fragment_sentences]

    signal = total(WEIGHTS.get(issue.type, 1) for issue in fragment_issues if not issue.style_only)
    density = min(4, signal * 100 / word_count / 3)
    uniformity = max(0, 1 - cv(lengths) / 0.6) if len(lengths) >= 3 else 0.3
    mean_length = mean(lengths)
    typical = math.exp(-(((mean_length - 19) / 7) ** 2)) if lengths else 0
    # Признак «канцелярит» и веса калибровок подобраны под канцелярит и многословие; жаргон тоже
    # совет по стилю (style_only), но в этот признак не идёт.
    clerical = min(
        3, sum(1 for issue in fragment_issues if issue.style_only and issue.type != "jargon") * 100 / word_count / 2
    )
    rare = sum(1 for sentence in fragment_sentences if RARE_PUNCT_RE.search(sentence.text))
    poor_punct = max(0, 1 - rare / len(fragment_sentences) / 0.5) if len(fragment_sentences) >= 3 else 0.3
    first_words = [(words(sentence.text) or [""])[0].lower() for sentence in fragment_sentences]
    opener_repeat = 1 - len(set(first_words)) / len(first_words) if len(first_words) >= 3 else 0
    diversity = mattr(text_words, min(50, len(text_words))) if len(text_words) >= 30 else 0.8
    poor_vocab = max(0, min(1, (0.85 - diversity) / 0.2))
    return [density, uniformity, typical, clerical, poor_punct, opener_repeat, poor_vocab]


def _reasons_for(model: Model, features: list[float]) -> list[str]:
    contributions = [
        (i, value * (model.weights[i] if i < len(model.weights) else 0)) for i, value in enumerate(features)
    ]
    top = sorted((item for item in contributions if item[1] > 0.5), key=lambda item: -item[1])[:3]
    return [FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else "" for i, _ in top]


@dataclass(slots=True)
class _RawFragment:
    start: int
    end: int
    text: str


def _fragments_of(internals: Internals) -> list[_RawFragment]:
    """Абзацы прозы и списков; короткие склеиваются со следующими."""
    fragments: list[_RawFragment] = []
    current: _RawFragment | None = None
    for block in internals.blocks:
        if block.kind not in ("prose", "list"):
            if block.kind == "heading" and current and len(words(current.text)) >= MIN_FRAGMENT_WORDS / 2:
                fragments.append(current)
                current = None
            continue
        if current is None:
            current = _RawFragment(block.start, block.end, block.text)
        else:
            current.end = block.end
            current.text += f"\n\n{block.text}"
        if len(words(current.text)) >= MIN_FRAGMENT_WORDS:
            fragments.append(current)
            current = None
    if current is not None:
        if fragments and len(words(current.text)) < MIN_FRAGMENT_WORDS / 2:
            fragments[-1].end = current.end
            fragments[-1].text += f"\n\n{current.text}"
        else:
            fragments.append(current)
    return fragments


def _chars(text: str) -> int:
    """Длина текста без лишних пробелов."""
    return len(trim(SPACES_RE.sub(" ", text)))


def antiplagiat(
    source: str, context: ContextMode = "academic", model: Model | None = None, system: System = "antiplagiat"
) -> AntiplagiatReport:
    result, internals = analyze_internal(source, context)
    active_model = model or DEFAULT_MODEL
    prepared = internals.prepared

    def to_source(index: int) -> int:
        return prepared.to_source[index] if index < len(prepared.to_source) else index

    ai_chars = 0
    all_chars = 0
    fragments: list[Fragment] = []
    for raw in _fragments_of(internals):
        start, end = to_source(raw.start), to_source(raw.end)
        inside = [issue for issue in result.issues if start <= issue.index < end]
        fragment_features = features_for(raw.text, inside)
        probability = predict(active_model, fragment_features)
        chars = _chars(raw.text)
        all_chars += chars
        ai = probability >= active_model.threshold
        if ai:
            ai_chars += chars
        fragments.append(
            Fragment(
                start=start,
                end=end,
                line=line_col(prepared.line_starts, start)[0],
                end_line=line_col(prepared.line_starts, end)[0],
                words=len(words(raw.text)),
                features=[fixed(value, 3) for value in fragment_features],
                probability=fixed(probability, 3),
                ai=ai,
                reasons=_reasons_for(active_model, fragment_features),
                preview=trim(SPACES_RE.sub(" ", prepared.source[start : min(end, start + 90)])),
            )
        )

    reasons: list[str] = []
    if prepared.invisible:
        reasons.append(f"невидимые символы: {len(prepared.invisible)}")
    if prepared.homoglyphs:
        reasons.append(f"слова со смешанной латиницей и кириллицей: {len(prepared.homoglyphs)}")
    return AntiplagiatReport(
        ai_share=fixed(ai_chars * 100 / all_chars, 1) if all_chars else 0,
        fragments=fragments,
        suspicious=bool(reasons),
        suspicious_reasons=reasons,
        light=result.light,
        model="calibrated" if model else "default",
        threshold=active_model.threshold,
        system=system,
    )


# ─── Калибровка по реальным отчётам ─────────────────────────────────────


class Sample(Record):
    f: list[float]
    y: Literal[0, 1]


class DocFragment(Record):
    f: list[float]
    chars: int


class DocSample(Record):
    """Документ, для которого известна только итоговая доля ИИ-текста из отчёта.

    Разметки фрагментов нет; для каждого фрагмента хранятся признаки и длина в знаках.
    """

    fragments: list[DocFragment]
    # Доля ИИ-текста по отчёту, 0–100.
    share: float


class Calibration(Record):
    """Калибровка под одну систему проверки."""

    model: Model
    samples: list[Sample] = Field(default_factory=list)
    # Документы с одной итоговой долей (`calibrate --share`).
    documents: list[DocSample] = Field(default_factory=list)


class CalibrationFile(Record):
    """Файл `.aiw-ru.json`.

    Поля верхнего уровня — калибровка «Антиплагиата», как в файлах до появления `--system`;
    калибровки других систем лежат в `systems` под их именем. Старые версии aiw-ru поле
    `systems` пропускают и читают файл как раньше.
    """

    version: Literal[1]
    model: Model
    samples: list[Sample] = Field(default_factory=list)
    documents: list[DocSample] = Field(default_factory=list)
    systems: dict[str, Calibration] = Field(default_factory=dict)

    def get(self, system: System) -> Calibration | None:
        """Калибровка системы; None, если по её отчётам ещё не калибровали."""
        if system != "antiplagiat":
            return self.systems.get(system)
        if not self.samples and not self.documents:
            return None
        return Calibration(model=self.model, samples=self.samples, documents=self.documents)

    def put(self, system: System, cal: Calibration) -> CalibrationFile:
        if system == "antiplagiat":
            return self.model_copy(update={"model": cal.model, "samples": cal.samples, "documents": cal.documents})
        return self.model_copy(update={"systems": {**self.systems, system: cal}})


def empty_calibration_file() -> CalibrationFile:
    return CalibrationFile(version=1, model=DEFAULT_MODEL)


# ─── Разметка из отчёта ─────────────────────────────────────────────────

# Слово для сверки текста из отчёта с документом: буквы и цифры, без разметки и знаков.
WORD_RE = re.compile(r"[^\W_]+")
# Куски короче не ищутся: столько слов подряд встречаются в документе случайно.
MIN_MARKED_WORDS = 5
# Красный уровень тепловой карты «Думейта»: только он входит в итоговый процент ИИ-текста.
HEAT_RED = 3


def _tokens(text: str) -> list[tuple[int, int, str]]:
    return [(match.start(), match.end(), match.group().lower().replace("ё", "е")) for match in WORD_RE.finditer(text)]


def _find(doc_words: list[str], piece_words: list[str], start: int, *, last: bool = False) -> int | None:
    """Первое вхождение начала (или конца, `last`) куска не раньше `start`: по 6 словам, если не нашлось — по 3."""
    for k in (6, 5, 4, 3):
        if len(piece_words) < k:
            continue
        probe = piece_words[-k:] if last else piece_words[:k]
        for i in range(start, len(doc_words) - k + 1):
            if doc_words[i : i + k] == probe:
                return i + k - 1 if last else i
    return None


def marked_spans(source: str, marked: list[str]) -> list[tuple[int, int]]:
    """Куски из отчёта как диапазоны знаков документа.

    Кусок ищется по первым и последним словам, поэтому переносы строк, разметка, кавычки
    и мелкие расхождения текстового слоя PDF внутри куска не мешают. Если конец не нашёлся
    или кусок вышел заметно длиннее себя, засчитывается только его начало длиной в сам кусок.
    """
    tokens = _tokens(source)
    doc_words = [word for _, _, word in tokens]
    spans: list[tuple[int, int]] = []
    for piece in marked:
        piece_words = [word for _, _, word in _tokens(piece)]
        if len(piece_words) < MIN_MARKED_WORDS:
            continue
        first_start = _find(doc_words, piece_words, 0)
        if first_start is None:
            continue
        # Начало куска встречается и раньше, чем сам кусок: заголовок раздела стоит ещё в оглавлении.
        # Берём первое вхождение начала, после которого близко находится и конец.
        start_word: int | None = first_start
        end_word: int | None = None
        while start_word is not None:
            end_word = _find(doc_words, piece_words, start_word, last=True)
            if end_word is not None and end_word - start_word + 1 <= len(piece_words) * 1.5 + 20:
                break
            start_word = _find(doc_words, piece_words, start_word + 1)
        if start_word is None or end_word is None:
            start_word, end_word = first_start, min(first_start + len(piece_words), len(tokens)) - 1
        spans.append((tokens[start_word][0], tokens[end_word][1]))
    return spans


@dataclass(slots=True)
class MarkedFragment:
    start: int
    end: int
    # Доля слов фрагмента внутри кусков из отчёта.
    coverage: float
    # Хотя бы один кусок целиком лежит внутри фрагмента: короткая подсветка в одном абзаце.
    holds_piece: bool

    @property
    def marked(self) -> bool:
        return self.holds_piece or self.coverage >= 0.5


def marked_fragments(source: str, marked: list[str], context: ContextMode = "academic") -> list[MarkedFragment]:
    """Фрагменты документа (те же, что у `antiplagiat`) с тем, насколько их подсветил отчёт.

    Фрагмент считается подсвеченным, если куски из отчёта покрывают больше половины его слов
    или какой-то кусок целиком лежит внутри него. Первое нужно тепловой карте «Думейта», где
    подсветка идёт через несколько абзацев; второе — подсветке предложениями в «Антиплагиате».
    """
    _, internals = analyze_internal(source, context)
    prepared = internals.prepared
    spans = marked_spans(source, marked)
    tokens = _tokens(source)
    fragments: list[MarkedFragment] = []
    for raw in _fragments_of(internals):
        start = prepared.to_source[raw.start] if raw.start < len(prepared.to_source) else raw.start
        end = prepared.to_source[raw.end] if raw.end < len(prepared.to_source) else raw.end
        inside = [token for token in tokens if start <= token[0] < end]
        covered = sum(1 for token in inside if any(span_start <= token[0] < span_end for span_start, span_end in spans))
        fragments.append(
            MarkedFragment(
                start=start,
                end=end,
                coverage=covered / len(inside) if inside else 0,
                holds_piece=any(start <= span_start and span_end <= end for span_start, span_end in spans),
            )
        )
    return fragments


# Абзац под подсветкой считается неправленым, если ближайший абзац итога совпадает с ним хотя бы
# на столько после замены особых пробелов и дефисов обычными. На реферате с «Думейтом» абзацы, которые
# потеряли только неразрывные пробелы, остались на прежнем уровне, а правка с сохранением 90 % букв
# вывела из красного 13 абзацев из 16.
UNEDITED_SIMILARITY = 0.98
# Короче этого абзац не проверяется: заголовки и подписи часто нельзя менять. Заголовки, HTML, рисунки и
# таблицы тоже пропускаются: это не проза.
UNEDITED_MIN_WORDS = 15
PARAGRAPH_RE = jsre("[^\\n](?:[^\\n]|\\n(?![ \\t]*\\n))*", "g")


NOT_PROSE_PREFIXES = ("#", "<", "![", "|")


@dataclass(slots=True)
class UneditedParagraph:
    """Абзац исходника под подсветкой отчёта, который в итоге остался прежним."""

    start: int
    line: int
    text: str
    similarity: float


def _normalized_paragraph(paragraph: str) -> str:
    """Абзац для сравнения: особые пробелы и дефисы обычными, пробелы схлопнуты."""
    plain = paragraph.translate(PLAIN_SPACES).replace(chr(0x2010), "-").replace(chr(0x2011), "-")
    return " ".join(plain.split())


class AfterParagraphs:
    """Абзацы итога для сравнения: точные совпадения и отбор кандидатов по длине.

    Совпадение двух строк не больше 2·min(длин)/(сумма длин). Поэтому до SequenceMatcher
    доходят только абзацы, чья длина позволяет набрать UNEDITED_SIMILARITY, и проверка
    не сравнивает каждый абзац исходника с каждым абзацем итога.
    """

    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        self.exact = set(texts)
        self._by_length = sorted(texts, key=len)
        self._lengths = [len(text) for text in self._by_length]

    def within_reach(self, paragraph: str) -> list[str]:
        """Абзацы итога, с которыми paragraph может совпасть на пороге, по возрастанию длины."""
        length = len(paragraph)
        # 2·min(a, b)/(a + b) ≥ t, если b от a·t/(2 − t) до a·(2 − t)/t; по знаку запаса с каждой стороны на округление.
        shortest = math.ceil(length * UNEDITED_SIMILARITY / (2 - UNEDITED_SIMILARITY)) - 1
        longest = math.floor(length * (2 - UNEDITED_SIMILARITY) / UNEDITED_SIMILARITY) + 1
        return self._by_length[bisect_left(self._lengths, shortest) : bisect_right(self._lengths, longest)]


def _best_similarity(paragraph: str, candidates: AfterParagraphs) -> float:
    """Наибольшее совпадение абзаца с абзацами итога, которые могут дойти до порога."""
    if paragraph in candidates.exact:
        return 1.0
    best = 0.0
    for candidate in candidates.within_reach(paragraph):
        matcher = SequenceMatcher(None, paragraph, candidate, autojunk=False)
        if matcher.real_quick_ratio() >= UNEDITED_SIMILARITY and matcher.quick_ratio() >= UNEDITED_SIMILARITY:
            best = max(best, matcher.ratio())
    return best


def unedited_marked(before: str, after: str, marked: list[str]) -> list[UneditedParagraph]:
    """Абзацы исходника под подсветкой отчёта, которые в итоге остались прежними.

    Абзац под подсветкой — тот, где куски из отчёта покрывают больше половины слов или лежат
    целиком. Прежним он считается, если ближайший абзац итога совпадает с ним на
    UNEDITED_SIMILARITY: снятые неразрывные пробелы и замена дефисов правкой не считаются.
    """
    marked_ranges = marked_spans(before, marked)
    tokens = _tokens(before)
    token_starts = [token[0] for token in tokens]
    after_paragraphs = AfterParagraphs([_normalized_paragraph(match.group()) for match in PARAGRAPH_RE.finditer(after)])
    line_starts = [0, *(index + 1 for index, char in enumerate(before) if char == "\n")]
    unedited: list[UneditedParagraph] = []
    for paragraph in PARAGRAPH_RE.finditer(before):
        start, end = paragraph.start(), paragraph.end()
        paragraph_tokens = tokens[bisect_left(token_starts, start) : bisect_left(token_starts, end)]
        if len(paragraph_tokens) < UNEDITED_MIN_WORDS or paragraph.group().lstrip().startswith(NOT_PROSE_PREFIXES):
            continue
        covered = sum(
            1
            for token in paragraph_tokens
            if any(range_start <= token[0] < range_end for range_start, range_end in marked_ranges)
        )
        holds_piece = any(start <= range_start and range_end <= end for range_start, range_end in marked_ranges)
        if covered / len(paragraph_tokens) < 0.5 and not holds_piece:
            continue
        text = _normalized_paragraph(paragraph.group())
        similarity = _best_similarity(text, after_paragraphs)
        if similarity >= UNEDITED_SIMILARITY:
            line = line_col(line_starts, start)[0]
            unedited.append(UneditedParagraph(start=start, line=line, text=text, similarity=similarity))
    return unedited


def label_fragments(source: str, marked: list[str]) -> list[Sample]:
    """Размечает фрагменты документа по кускам, которые система подсветила в отчёте.

    Куски скопированы из отчёта в текстовый файл и разделены пустой строкой или взяты из
    тепловой карты (`marked_from_heatmap`).
    """
    result, internals = analyze_internal(source, "academic")
    samples: list[Sample] = []
    for raw, mark in zip(_fragments_of(internals), marked_fragments(source, marked), strict=True):
        inside = [issue for issue in result.issues if mark.start <= issue.index < mark.end]
        samples.append(Sample(f=features_for(raw.text, inside), y=1 if mark.marked else 0))
    return samples


class HeatFragment(Record):
    """Фрагмент тепловой карты отчёта: уровень 3 — красный, 2 — оранжевый, 1 — жёлтый."""

    level: int
    text: str
    pages: list[int] = Field(default_factory=list)


def marked_from_heatmap(raw: str, min_level: int = HEAT_RED) -> list[str] | None:
    """Куски из тепловой карты в JSON; None, если это не она.

    Тепловая карта — список `{"level", "text"}` или объект с ним в поле `fragments` (так его
    отдаёт скрипт из `skills/antiplagiat/references/domate.md`). По умолчанию берутся только
    красные фрагменты: в «Думейте» только они входят в процент ИИ-текста.
    """
    head = raw.lstrip()[:1]
    if head not in ("[", "{"):
        return None
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            data = data.get("fragments")
        if not isinstance(data, list):
            return None
        items = [HeatFragment.model_validate(item) for item in data]
    except ValueError:
        return None
    return [item.text for item in items if item.level >= min_level]


def fit(samples: list[Sample], start: Model = DEFAULT_MODEL) -> Model:
    """Логистическая регрессия с L2 и выбором порога по сбалансированной точности."""
    positives = sum(1 for sample in samples if sample.y == 1)
    negatives = len(samples) - positives
    if positives == 0 or negatives == 0:
        return start
    positive_weight = len(samples) / (2 * positives)
    negative_weight = len(samples) / (2 * negatives)
    weights = list(start.weights)
    bias = start.bias
    learning_rate = 0.05
    l2 = 0.01
    for _ in range(4000):
        weight_grads = [0.0] * len(weights)
        bias_grad = 0.0
        current = Model(bias=bias, weights=weights, threshold=0.5)
        for sample in samples:
            error = (predict(current, sample.f) - sample.y) * (positive_weight if sample.y == 1 else negative_weight)
            for k in range(len(weights)):
                weight_grads[k] += error * (sample.f[k] if k < len(sample.f) else 0)
            bias_grad += error
        for k in range(len(weights)):
            weights[k] = weights[k] - learning_rate * (
                weight_grads[k] / len(samples) + l2 * (weights[k] - start.weights[k])
            )
        bias -= learning_rate * bias_grad / len(samples)
    model = Model(bias=bias, weights=weights, threshold=0.5)
    best_threshold, best_score = 0.5, -1.0
    threshold = 0.2
    while threshold <= 0.8001:
        true_positives = true_negatives = 0
        for sample in samples:
            hit = predict(model, sample.f) >= threshold
            if hit and sample.y == 1:
                true_positives += 1
            if not hit and sample.y == 0:
                true_negatives += 1
        balanced = (true_positives / positives + true_negatives / negatives) / 2
        if balanced > best_score:
            best_threshold, best_score = threshold, balanced
        threshold += 0.02
    return Model(
        bias=fixed(model.bias, 4),
        weights=[fixed(weight, 4) for weight in model.weights],
        threshold=fixed(best_threshold, 2),
    )


def balanced_accuracy(model: Model, samples: list[Sample]) -> float:
    positives = [sample for sample in samples if sample.y == 1]
    negatives = [sample for sample in samples if sample.y == 0]
    if not positives or not negatives:
        return math.nan
    true_positives = sum(1 for sample in positives if predict(model, sample.f) >= model.threshold)
    true_negatives = sum(1 for sample in negatives if predict(model, sample.f) < model.threshold)
    return (true_positives / len(positives) + true_negatives / len(negatives)) / 2


def document_sample(source: str, share: float) -> DocSample:
    """Признаки и длины фрагментов документа с долей ИИ-текста из отчёта."""
    report = antiplagiat(source)
    return DocSample(
        fragments=[
            DocFragment(f=fragment.features, chars=_chars(source[fragment.start : fragment.end]))
            for fragment in report.fragments
        ],
        share=share,
    )


def predict_share(model: Model, doc: DocSample) -> float:
    """Доля ИИ-текста по знакам, которую модель предскажет для документа."""
    ai_chars = 0
    all_chars = 0
    for fragment in doc.fragments:
        all_chars += fragment.chars
        if predict(model, fragment.f) >= model.threshold:
            ai_chars += fragment.chars
    return ai_chars * 100 / all_chars if all_chars else 0


def share_error(model: Model, docs: list[DocSample]) -> float:
    """Средняя ошибка доли ИИ-текста в процентных пунктах."""
    if not docs:
        return math.nan
    return total(abs(predict_share(model, doc) - doc.share) for doc in docs) / len(docs)


def fit_share(model: Model, docs: list[DocSample]) -> Model:
    """Подстройка под итоговые доли из отчётов.

    Сдвиг свободного члена, при котором предсказанные доли ближе всего к отчётным.
    Веса признаков не меняются: одной цифры на документ хватает только на общий
    уровень строгости системы.
    """
    if not docs:
        return model
    best_bias, best_error = model.bias, share_error(model, docs)
    shift = -8.0
    while shift <= 8.0001:
        bias = model.bias + shift
        error = share_error(model.model_copy(update={"bias": bias}), docs)
        closer = abs(shift) < abs(best_bias - model.bias)
        if error < best_error - 1e-9 or (abs(error - best_error) <= 1e-9 and closer):
            best_bias, best_error = bias, error
        shift += 0.02
    return model.model_copy(update={"bias": fixed(best_bias, 4)})

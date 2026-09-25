"""Оценка по фрагментам в духе модуля поиска сгенерированного текста системы «Антиплагиат».

Настоящий классификатор закрыт, поэтому здесь приближение: каждый фрагмент
(абзац или склейка коротких абзацев) описывается набором интерпретируемых
признаков, логистическая модель переводит их в вероятность, а доля ИИ-текста
считается как доля знаков во фрагментах выше порога — так же выглядит итог в
отчёте системы.

Веса по умолчанию подобраны вручную. Команда `calibrate` дообучает их на
фрагментах, которые система реально подсветила в ваших отчётах.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from aiw_ru.compat import fixed, jsre, total, trim
from aiw_ru.detect import WEIGHTS, Internals, analyze_internal
from aiw_ru.text import cv, line_col, mattr, mean, sentences, words
from aiw_ru.types import ContextMode, Issue, Record

FEATURE_NAMES: tuple[str, ...] = (
    "плотность примет",
    "однообразие длины предложений",
    "типичная длина предложения",
    "канцелярит",
    "бедная пунктуация",
    "однообразные начала предложений",
    "бедный словарь",
)


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
    model: Literal["default", "calibrated"]
    threshold: float


MIN_FRAGMENT_WORDS = 40
RARE_PUNCT_RE = jsre("[;:()!?…]|\\s—\\s")
SPACES_RE = jsre("\\s+", "g")


def _sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z))


def predict(model: Model, f: list[float]) -> float:
    z = model.bias
    for i, v in enumerate(f):
        z += (model.weights[i] if i < len(model.weights) else 0) * v
    return _sigmoid(z)


def features_for(text: str, fragment_issues: list[Issue]) -> list[float]:
    ws = words(text)
    n = max(len(ws), 1)
    ss = [s for s in sentences(text) if s.words >= 2]
    lens = [s.words for s in ss]

    signal = total(WEIGHTS.get(i.type, 1) for i in fragment_issues if not i.style_only)
    density = min(4, signal * 100 / n / 3)
    uniformity = max(0, 1 - cv(lens) / 0.6) if len(lens) >= 3 else 0.3
    m = mean(lens)
    typical = math.exp(-(((m - 19) / 7) ** 2)) if lens else 0
    clerical = min(3, sum(1 for i in fragment_issues if i.style_only) * 100 / n / 2)
    rare = sum(1 for s in ss if RARE_PUNCT_RE.search(s.text))
    poor_punct = max(0, 1 - rare / len(ss) / 0.5) if len(ss) >= 3 else 0.3
    firsts = [(words(s.text) or [""])[0].lower() for s in ss]
    opener_repeat = 1 - len(set(firsts)) / len(firsts) if len(firsts) >= 3 else 0
    diversity = mattr(ws, min(50, len(ws))) if len(ws) >= 30 else 0.8
    poor_vocab = max(0, min(1, (0.85 - diversity) / 0.2))
    return [density, uniformity, typical, clerical, poor_punct, opener_repeat, poor_vocab]


def _reasons_for(model: Model, f: list[float]) -> list[str]:
    contrib = [(i, v * (model.weights[i] if i < len(model.weights) else 0)) for i, v in enumerate(f)]
    top = sorted((x for x in contrib if x[1] > 0.5), key=lambda x: -x[1])[:3]
    return [FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else "" for i, _ in top]


@dataclass(slots=True)
class _RawFragment:
    start: int
    end: int
    text: str


def _fragments_of(internals: Internals) -> list[_RawFragment]:
    """Абзацы прозы и списков; короткие склеиваются со следующими."""
    out: list[_RawFragment] = []
    cur: _RawFragment | None = None
    for b in internals.blocks:
        if b.kind not in ("prose", "list"):
            if b.kind == "heading" and cur and len(words(cur.text)) >= MIN_FRAGMENT_WORDS / 2:
                out.append(cur)
                cur = None
            continue
        if cur is None:
            cur = _RawFragment(b.start, b.end, b.text)
        else:
            cur.end = b.end
            cur.text += f"\n\n{b.text}"
        if len(words(cur.text)) >= MIN_FRAGMENT_WORDS:
            out.append(cur)
            cur = None
    if cur is not None:
        if out and len(words(cur.text)) < MIN_FRAGMENT_WORDS / 2:
            out[-1].end = cur.end
            out[-1].text += f"\n\n{cur.text}"
        else:
            out.append(cur)
    return out


def _chars(s: str) -> int:
    """Длина текста без лишних пробелов."""
    return len(trim(SPACES_RE.sub(" ", s)))


def antiplagiat(source: str, context: ContextMode = "academic", model: Model | None = None) -> AntiplagiatReport:
    result, internals = analyze_internal(source, context)
    m = model or DEFAULT_MODEL
    p = internals.prepared

    def to_src(i: int) -> int:
        return p.to_source[i] if i < len(p.to_source) else i

    ai_chars = 0
    all_chars = 0
    fragments: list[Fragment] = []
    for r in _fragments_of(internals):
        s, e = to_src(r.start), to_src(r.end)
        inside = [i for i in result.issues if s <= i.index < e]
        f = features_for(r.text, inside)
        prob = predict(m, f)
        chars = _chars(r.text)
        all_chars += chars
        ai = prob >= m.threshold
        if ai:
            ai_chars += chars
        fragments.append(
            Fragment(
                start=s,
                end=e,
                line=line_col(p.line_starts, s)[0],
                end_line=line_col(p.line_starts, e)[0],
                words=len(words(r.text)),
                features=[fixed(v, 3) for v in f],
                probability=fixed(prob, 3),
                ai=ai,
                reasons=_reasons_for(m, f),
                preview=trim(SPACES_RE.sub(" ", p.source[s : min(e, s + 90)])),
            )
        )

    reasons: list[str] = []
    if p.invisible:
        reasons.append(f"невидимые символы: {len(p.invisible)}")
    if p.homoglyphs:
        reasons.append(f"слова со смешанной латиницей и кириллицей: {len(p.homoglyphs)}")
    return AntiplagiatReport(
        ai_share=fixed(ai_chars * 100 / all_chars, 1) if all_chars else 0,
        fragments=fragments,
        suspicious=bool(reasons),
        suspicious_reasons=reasons,
        model="calibrated" if model else "default",
        threshold=m.threshold,
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


class CalibrationFile(Record):
    version: Literal[1]
    model: Model
    samples: list[Sample] = Field(default_factory=list)
    # Документы с одной итоговой долей (`calibrate --share`).
    documents: list[DocSample] = Field(default_factory=list)


NORM_MARKUP_RE = jsre("[*_`#>|]", "g")
NORM_QUOTES_RE = jsre('[«»„“”"]', "g")


def _norm(s: str) -> str:
    s = s.lower().replace("ё", "е")
    s = NORM_QUOTES_RE.sub("", NORM_MARKUP_RE.sub("", s))
    return trim(SPACES_RE.sub(" ", s))


def label_fragments(source: str, marked: list[str]) -> list[Sample]:
    """Размечает фрагменты документа по кускам, которые система подсветила в отчёте.

    Куски скопированы из отчёта в текстовый файл и разделены пустой строкой.
    """
    result, internals = analyze_internal(source, "academic")
    p = internals.prepared
    keys = [k[:60] for k in (_norm(m) for m in marked) if len(k) >= 30]
    out: list[Sample] = []
    for r in _fragments_of(internals):
        s = p.to_source[r.start] if r.start < len(p.to_source) else r.start
        e = p.to_source[r.end] if r.end < len(p.to_source) else r.end
        inside = [i for i in result.issues if s <= i.index < e]
        text = _norm(p.source[s:e])
        y: Literal[0, 1] = 1 if any(k in text for k in keys) else 0
        out.append(Sample(f=features_for(r.text, inside), y=y))
    return out


def fit(samples: list[Sample], start: Model = DEFAULT_MODEL) -> Model:
    """Логистическая регрессия с L2 и выбором порога по сбалансированной точности."""
    pos = sum(1 for s in samples if s.y == 1)
    neg = len(samples) - pos
    if pos == 0 or neg == 0:
        return start
    w_pos = len(samples) / (2 * pos)
    w_neg = len(samples) / (2 * neg)
    w = list(start.weights)
    b = start.bias
    lr = 0.05
    l2 = 0.01
    for _ in range(4000):
        gw = [0.0] * len(w)
        gb = 0.0
        cur = Model(bias=b, weights=w, threshold=0.5)
        for s in samples:
            err = (predict(cur, s.f) - s.y) * (w_pos if s.y == 1 else w_neg)
            for k in range(len(w)):
                gw[k] += err * (s.f[k] if k < len(s.f) else 0)
            gb += err
        for k in range(len(w)):
            w[k] = w[k] - lr * (gw[k] / len(samples) + l2 * (w[k] - start.weights[k]))
        b -= lr * gb / len(samples)
    model = Model(bias=b, weights=w, threshold=0.5)
    best_t, best_score = 0.5, -1.0
    t = 0.2
    while t <= 0.8001:
        tp = tn = 0
        for s in samples:
            hit = predict(model, s.f) >= t
            if hit and s.y == 1:
                tp += 1
            if not hit and s.y == 0:
                tn += 1
        bal = (tp / pos + tn / neg) / 2
        if bal > best_score:
            best_t, best_score = t, bal
        t += 0.02
    return Model(bias=fixed(model.bias, 4), weights=[fixed(x, 4) for x in model.weights], threshold=fixed(best_t, 2))


def balanced_accuracy(model: Model, samples: list[Sample]) -> float:
    pos = [s for s in samples if s.y == 1]
    neg = [s for s in samples if s.y == 0]
    if not pos or not neg:
        return math.nan
    tp = sum(1 for s in pos if predict(model, s.f) >= model.threshold)
    tn = sum(1 for s in neg if predict(model, s.f) < model.threshold)
    return (tp / len(pos) + tn / len(neg)) / 2


def document_sample(source: str, share: float) -> DocSample:
    """Признаки и длины фрагментов документа с долей ИИ-текста из отчёта."""
    report = antiplagiat(source)
    return DocSample(
        fragments=[DocFragment(f=f.features, chars=_chars(source[f.start : f.end])) for f in report.fragments],
        share=share,
    )


def predict_share(model: Model, doc: DocSample) -> float:
    """Доля ИИ-текста по знакам, которую модель предскажет для документа."""
    ai = 0
    all_ = 0
    for fr in doc.fragments:
        all_ += fr.chars
        if predict(model, fr.f) >= model.threshold:
            ai += fr.chars
    return ai * 100 / all_ if all_ else 0


def share_error(model: Model, docs: list[DocSample]) -> float:
    """Средняя ошибка доли ИИ-текста в процентных пунктах."""
    if not docs:
        return math.nan
    return total(abs(predict_share(model, d) - d.share) for d in docs) / len(docs)


def fit_share(model: Model, docs: list[DocSample]) -> Model:
    """Подстройка под итоговые доли из отчётов.

    Сдвиг свободного члена, при котором предсказанные доли ближе всего к отчётным.
    Веса признаков не меняются: одной цифры на документ хватает только на общий
    уровень строгости системы.
    """
    if not docs:
        return model
    best_bias, best_err = model.bias, share_error(model, docs)
    d = -8.0
    while d <= 8.0001:
        bias = model.bias + d
        err = share_error(model.model_copy(update={"bias": bias}), docs)
        closer = abs(d) < abs(best_bias - model.bias)
        if err < best_err - 1e-9 or (abs(err - best_err) <= 1e-9 and closer):
            best_bias, best_err = bias, err
        d += 0.02
    return model.model_copy(update={"bias": fixed(best_bias, 4)})

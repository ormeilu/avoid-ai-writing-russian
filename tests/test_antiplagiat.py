"""Оценка по фрагментам и калибровка в духе «Антиплагиата»."""

import math
from collections.abc import Callable

from aiw_ru import (
    DEFAULT_MODEL,
    Sample,
    antiplagiat,
    balanced_accuracy,
    document_sample,
    fit,
    fit_share,
    label_fragments,
    predict_share,
    share_error,
)

# ─── antiplagiat ────────────────────────────────────────────────────────


def test_ai_text_gets_share_human_zero(read_fixture: Callable[[str], str]):
    """ИИ-текст получает большую долю, живой — нулевую"""
    assert antiplagiat(read_fixture("corpus/ai/blog.md")).ai_share > 50
    assert antiplagiat(read_fixture("corpus/human/blog.md")).ai_share == 0


def test_fragments_cover_prose_with_lines(read_fixture: Callable[[str], str]):
    """фрагменты покрывают прозу и указывают строки"""
    r = antiplagiat(read_fixture("corpus/ai/blog.md"))
    assert len(r.fragments) > 0
    for f in r.fragments:
        assert f.line > 0
        assert f.end_line >= f.line
        assert f.probability >= 0
        assert f.probability <= 1


def test_suspicious_document():
    """подозрительный документ"""
    zw = chr(0x200B)
    assert antiplagiat(f"Обычный{zw} текст.").suspicious is True
    assert antiplagiat(f"Обыч{chr(0x00AD)}ный текст.").suspicious is False


def test_label_by_highlighted_pieces(read_fixture: Callable[[str], str]):
    """разметка по подсвеченным кускам"""
    doc = f"{read_fixture('corpus/ai/blog.md')}\n\n{read_fixture('corpus/human/blog.md')}"
    marked = [
        "Более того, использование ИИ способствует оптимизации процессов и повышению эффективности работы медицинских учреждений.",
    ]
    samples = label_fragments(doc, marked)
    assert len([s for s in samples if s.y == 1]) == 1


def test_calibration_does_not_hurt_training_accuracy():
    """калибровка не ухудшает точность на обучающих данных"""
    samples = [
        Sample(f=[2, 0.8, 0.9, 1, 0.9, 0.3, 0.2], y=1),
        Sample(f=[1.5, 0.7, 0.8, 0.5, 0.8, 0.2, 0.1], y=1),
        Sample(f=[0.1, 0.4, 0.6, 0.2, 0.9, 0.1, 0.1], y=1),
        Sample(f=[0, 0.2, 0.3, 0, 0.2, 0, 0], y=0),
        Sample(f=[0.2, 0.3, 0.5, 0.1, 0.3, 0, 0.1], y=0),
        Sample(f=[0, 0.1, 0.2, 0, 0.1, 0, 0], y=0),
    ]
    model = fit(samples)
    assert balanced_accuracy(model, samples) >= balanced_accuracy(DEFAULT_MODEL, samples)


# ─── калибровка по итоговой доле ────────────────────────────────────────


def test_fit_share_moves_threshold_to_reported_share(read_fixture: Callable[[str], str]):
    """fitShare сдвигает порог строгости к доле из отчёта"""
    ai = read_fixture("corpus/ai/vak.md")
    human = read_fixture("corpus/human/vak.md")
    lenient = [document_sample(ai, 10), document_sample(human, 0)]
    model = fit_share(DEFAULT_MODEL, lenient)
    assert model.bias < DEFAULT_MODEL.bias
    assert model.weights == DEFAULT_MODEL.weights
    assert share_error(model, lenient) < share_error(DEFAULT_MODEL, lenient)
    assert predict_share(model, lenient[1]) == 0


def test_no_documents_keeps_model():
    """без документов модель не меняется"""
    assert fit_share(DEFAULT_MODEL, []) == DEFAULT_MODEL
    assert math.isnan(share_error(DEFAULT_MODEL, []))

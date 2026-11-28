"""Оценка по фрагментам и калибровка в духе «Антиплагиата»."""

import math
import re
from collections.abc import Callable

import pytest

from aiw_ru import (
    DEFAULT_MODEL,
    FEATURE_NAMES,
    Issue,
    Model,
    Sample,
    antiplagiat,
    balanced_accuracy,
    document_sample,
    fit,
    fit_share,
    label_fragments,
    predict,
    predict_share,
    share_error,
)
from aiw_ru.antiplagiat import Contribution, features_for, green_below, typical_window, zone_of

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


def test_jargon_is_not_clerical():
    """жаргон (style_only) не входит в признак «канцелярит»: калибровки подобраны под канцелярит"""
    text = "Мы собрали пайплайн и бейзлайн за неделю. " * 5

    def issue(type_: str) -> Issue:
        return Issue(type=type_, rule="r", severity="P1", text="x", index=0, line=1, column=1, hint="", style_only=True)

    assert features_for(text, [issue("jargon")] * 4)[3] == 0
    assert features_for(text, [issue("tier1-clarity")] * 4)[3] > 0


# ─── карта детектора и --explain ────────────────────────────────────────

# Модель, которая ставит ровную научную прозу у порога, как калибровка «Думейта» в задаче #20.
BORDER = Model(bias=-1.0, weights=[0.3, 1.9, 0.75, 0.16, 0.15, 1.05, 1.0], threshold=0.44)

OPENERS_TEXT = (
    "Это растение любит тень и влажную почву у старого забора. Это видно по листьям, которые желтеют "
    "на солнце уже к июлю. Это же объясняет, почему соседний куст растёт заметно быстрее. Садовник "
    "пересадил его осенью ближе к яблоне, и весной листья остались зелёными."
)


# Абзац из двух длинных предложений: пока предложений меньше трёх, однообразие и пунктуация стоят на
# постоянных 0,3, а средняя длина (33 слова) далеко от пика «типичной длины».
TWO_LONG = (
    "Паромная переправа через реку работает с апреля по ноябрь, пока лёд не встанет у причала, и за сезон "
    "перевозит около девяти тысяч машин из соседних сёл, где моста нет и не будет в ближайшие годы. "
    "Расписание зависит от уровня воды: в половодье паром ходит реже, потому что течение сносит его ниже "
    "причала, а осенью капитан добавляет вечерние рейсы для тех, кто возвращается с работы из города."
)
# Тот же абзац тремя предложениями: средняя длина падает к 22 словам, к пику «типичной длины».
THREE_SPLIT = (
    "Паромная переправа через реку работает с апреля по ноябрь. Пока лёд не встанет у причала, она за сезон "
    "перевозит около девяти тысяч машин из соседних сёл, где моста нет и не будет в ближайшие годы. "
    "Расписание зависит от уровня воды: в половодье паром ходит реже, потому что течение сносит его ниже "
    "причала, а осенью капитан добавляет вечерние рейсы для тех, кто возвращается с работы из города."
)
# Тот же абзац шестью предложениями разной длины, в среднем 10 слов.
SIX_SPLIT = (
    "Паромная переправа через реку работает с апреля по ноябрь. Зимой стоит лёд. За сезон паром перевозит "
    "около девяти тысяч машин из соседних сёл, где моста нет и не будет в ближайшие годы. Расписание "
    "зависит от воды. В половодье паром ходит реже: течение сносит его ниже причала. Осенью капитан "
    "добавляет вечерние рейсы для тех, кто возвращается с работы из города."
)
# Абзац из двух коротких предложений (12 и 11 слов): делить его на три значит дробить до 7–8 слов в
# предложении, а слияние в одно подняло бы среднюю длину к пику «типичной длины».
TWO_SHORT = (
    "Паром ходит через реку с апреля по ноябрь, пока не встанет лёд. "
    "Расписание рейсов висит на причале рядом с кассой и у магазина."
)
# Абзац из двух предложений по 20 слов, у пика «типичной длины»: поднять среднюю длину можно только
# слиянием в одно предложение, а опустить — только делением абзаца.
TWO_MEDIUM = (
    "Мост через реку закрыли на ремонт в марте, и с тех пор паром возит по сорок машин за рейс до самой "
    "ночи. Очередь у причала растёт к вечеру: люди едут с работы из города и ждут по часу, а иногда и дольше."
)
# Калибровка, где «типичная длина» весит больше однообразия.
TYPICAL_HEAVY = Model(bias=-2.0, weights=[0.3, 0.5, 2.0, 0.16, 0.15, 1.05, 1.0], threshold=0.44)


def test_zones_around_threshold():
    """красный — от порога, жёлтый — в 0,1 под ним, ниже — зелёный"""
    assert zone_of(0.44, 0.44) == "red"
    assert zone_of(0.36, 0.44) == "yellow"
    assert zone_of(0.34, 0.44) == "yellow"
    assert zone_of(0.33, 0.44) == "green"
    assert green_below(0.44) == pytest.approx(0.34)


def test_fragment_zone_without_explain(read_fixture: Callable[[str], str]):
    """зона есть у каждого фрагмента, красная — ровно у помеченных как ИИ; explain только по запросу"""
    report = antiplagiat(read_fixture("corpus/human/vak.md"), model=BORDER)
    assert {fragment.zone for fragment in report.fragments} >= {"red", "yellow"}
    for fragment in report.fragments:
        assert (fragment.zone == "red") == fragment.ai
        assert fragment.explain is None


def test_typical_prose_at_border(read_fixture: Callable[[str], str]):
    """typicalProse — где модель ставит ровную прозу: у BORDER в жёлтой зоне, у модели по умолчанию в зелёной"""
    text = read_fixture("corpus/human/vak.md")
    border = antiplagiat(text, model=BORDER)
    assert border.typical_prose == pytest.approx(predict(BORDER, [0, 0, 1, 0, 0, 0, 0]), abs=1e-3)
    assert zone_of(border.typical_prose, BORDER.threshold) == "yellow"
    assert zone_of(antiplagiat(text).typical_prose, DEFAULT_MODEL.threshold) == "green"


@pytest.mark.parametrize("model", [DEFAULT_MODEL, BORDER], ids=["default", "border"])
def test_explain_contributions_add_up(model: Model, read_fixture: Callable[[str], str]):
    """свободный член и вклады признаков дают логит фрагмента, вклады идут по убыванию"""
    for fragment in antiplagiat(read_fixture("corpus/ai/vak.md"), model=model, explain=True).fragments:
        explanation = fragment.explain
        assert explanation is not None
        contributions = [item.contribution for item in explanation.contributions]
        assert model.bias + sum(contributions) == pytest.approx(explanation.logit, abs=0.01)
        assert 1 / (1 + math.exp(-explanation.logit)) == pytest.approx(fragment.probability, abs=0.002)
        assert contributions == sorted(contributions, reverse=True)
        assert explanation.target_probability == pytest.approx(model.threshold - 0.1)


def test_explain_single_goal_reaches_green(read_fixture: Callable[[str], str]):
    """цель признака, которого хватает одного, уводит фрагмент в зелёную зону"""
    checked = 0
    for name in ("corpus/human/vak.md", "corpus/human/blog.md", "corpus/human/docs.md"):
        for fragment in antiplagiat(read_fixture(name), model=BORDER, explain=True).fragments:
            assert fragment.explain is not None
            for item in fragment.explain.contributions:
                # Цель с делением абзаца сдвигает и другие признаки: её проверяет
                # test_explain_split_counts_typical_length вместе с test_split_into_three_raises_two_sentence_paragraph.
                if fragment.zone == "green" or not item.enough or item.split_into:
                    continue
                features = list(fragment.features)
                features[FEATURE_NAMES.index(item.feature)] = item.goal
                assert predict(BORDER, features) <= fragment.explain.target_probability + 0.005
                assert item.target
                checked += 1
    assert checked > 0


def test_explain_plan_only_for_hot_fragments(read_fixture: Callable[[str], str]):
    """план правки есть у красных и жёлтых фрагментов; у зелёных нет ни плана, ни целей"""
    for name in ("corpus/ai/vak.md", "corpus/human/blog.md"):
        for fragment in antiplagiat(read_fixture(name), model=BORDER, explain=True).fragments:
            assert fragment.explain is not None
            if fragment.zone == "green":
                assert fragment.explain.plan == []
                assert all(not item.target for item in fragment.explain.contributions)
            else:
                assert fragment.explain.plan
                assert fragment.explain.reachable


def test_explain_names_repeated_openers():
    """explain называет первые слова, с которых начинается несколько предложений"""
    fragment = antiplagiat(OPENERS_TEXT, model=BORDER, explain=True).fragments[0]
    assert fragment.explain is not None
    assert fragment.explain.measures.repeated_openers == {"это": 3}
    openers = next(item for item in fragment.explain.contributions if item.feature == "однообразные начала предложений")
    assert "«это» ×3" in openers.now


def test_split_into_three_raises_two_sentence_paragraph():
    """посылка: деление двух длинных предложений на три поднимает вероятность, на шесть разной длины — в зелёную зону"""
    two = antiplagiat(TWO_LONG, model=BORDER).fragments[0]
    three = antiplagiat(THREE_SPLIT, model=BORDER).fragments[0]
    six = antiplagiat(SIX_SPLIT, model=BORDER).fragments[0]
    assert two.zone == "yellow"
    assert three.probability > two.probability
    assert six.zone == "green"


def contribution(fragment_text: str, feature: str, model: Model = BORDER) -> Contribution:
    explanation = antiplagiat(fragment_text, model=model, explain=True).fragments[0].explain
    assert explanation is not None
    return next(item for item in explanation.contributions if item.feature == feature)


def test_explain_split_counts_typical_length():
    """совет делить абзац из двух предложений просчитан с новой средней длиной: шесть предложений, а не три"""
    explanation = antiplagiat(TWO_LONG, model=BORDER, explain=True).fragments[0].explain
    assert explanation is not None
    uniformity = contribution(TWO_LONG, "однообразие длины предложений")
    assert uniformity.enough
    assert uniformity.split_into == 6
    assert uniformity.target.startswith("раздели на 6 предложений разной длины, в среднем 11 слов")
    assert explanation.plan[0].startswith("раздели на 6 предложений")
    assert explanation.reachable
    assert not any("три предложения" in step for step in explanation.plan)


def test_explain_no_split_below_floor_length():
    """абзац из двух коротких предложений не делится: на трёх средняя длина ушла бы ниже 8 слов"""
    explanation = antiplagiat(TWO_SHORT, model=BORDER, explain=True).fragments[0].explain
    assert explanation is not None
    uniformity = contribution(TWO_SHORT, "однообразие длины предложений")
    assert not uniformity.enough
    assert uniformity.split_into is None
    assert uniformity.target == ""
    assert not any("раздели" in step or "три предложения" in step for step in explanation.plan)


def test_explain_typical_length_of_two_sentences_by_merge():
    """у абзаца из двух предложений «типичную длину» одну двигает только слияние: опустить длину — это деление"""
    explanation = antiplagiat(TWO_MEDIUM, model=TYPICAL_HEAVY, explain=True).fragments[0].explain
    assert explanation is not None
    typical = contribution(TWO_MEDIUM, "типичная длина предложения", TYPICAL_HEAVY)
    assert typical.enough
    assert typical.target == "слей предложения в одно: средняя длина станет 42 слова"
    assert explanation.plan == ["слей предложения в одно: средняя длина станет 42 слова"]


def test_explain_typical_length_no_merge_when_it_raises():
    """слияние двух коротких предложений поднимает «типичную длину», поэтому цели у неё нет"""
    explanation = antiplagiat(TWO_SHORT, model=BORDER, explain=True).fragments[0].explain
    assert explanation is not None
    assert contribution(TWO_SHORT, "типичная длина предложения").target == ""
    assert not explanation.reachable
    assert not any("средняя длина" in step or "слей" in step for step in explanation.plan)


def test_explain_punctuation_alone_needs_three_sentences():
    """пунктуацию абзаца из одного-двух предложений одну не сдвинуть: деление меняет и длину, поэтому «нет»"""
    # Калибровка с большим весом пунктуации: без деления её вклада хватило бы, чтобы уйти в зелёную зону.
    heavy = Model(bias=-1.0, weights=[0.3, 0.5, 0.75, 0.16, 1.5, 1.05, 1.0], threshold=0.44)
    assert antiplagiat(TWO_LONG, model=heavy).fragments[0].zone == "yellow"
    punctuation = contribution(TWO_LONG, "бедная пунктуация", heavy)
    assert not punctuation.enough
    assert punctuation.target == ""


@pytest.mark.parametrize("goal", [0.9, 0.5, 0.3, 0.1])
def test_typical_window_bounds(goal: float):
    """средняя длина из цели: на границах признак «типичная длина» не больше цели, шагом ближе к 19 — уже больше"""

    def typical(length: float) -> float:
        return math.exp(-(((length - 19) / 7) ** 2))

    found = re.fullmatch(r"средняя длина не больше (\d+) или не меньше (\d+) слов", typical_window(goal))
    assert found
    low, high = int(found[1]), int(found[2])
    assert typical(low) <= goal < typical(low + 1)
    assert typical(high) <= goal < typical(high - 1)

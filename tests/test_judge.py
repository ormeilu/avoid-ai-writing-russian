"""Разметка из отчёта (тепловая карта, многоабзацные куски), калибровка по системам и слепой судья."""

import json
import math
from collections.abc import Callable

import pytest

from aiw_ru import DEFAULT_MODEL, antiplagiat
from aiw_ru.antiplagiat import (
    Calibration,
    CalibrationFile,
    Sample,
    empty_calibration_file,
    marked_fragments,
    marked_from_heatmap,
)
from aiw_ru.judge import PROMPT, auc, judge_fragments, judge_prompt, judge_score, parse_answer

Read = Callable[[str], str]


@pytest.fixture
def doc(read_fixture: Read) -> str:
    """Человеческий и ИИ-текст в жанре ВАК подряд: пять фрагментов."""
    return f"{read_fixture('corpus/human/vak.md')}\n\n{read_fixture('corpus/ai/vak.md')}"


def fragments(text: str) -> list[str]:
    """Тексты фрагментов antiplagiat: три человеческих, затем два ИИ (последний — два абзаца)."""
    return [text[f.start : f.end] for f in antiplagiat(text).fragments]


# ─── тепловая карта ─────────────────────────────────────────────────────


def test_heatmap_list_and_object_red_only():
    """тепловая карта: список или объект с fragments, берутся только красные"""
    frs = [{"level": 3, "text": "красный"}, {"level": 2, "text": "оранжевый"}, {"level": 1, "text": "жёлтый"}]
    assert marked_from_heatmap(json.dumps(frs)) == ["красный"]
    assert marked_from_heatmap(json.dumps({"share": 0.38, "fragments": frs})) == ["красный"]
    assert marked_from_heatmap(json.dumps(frs), min_level=2) == ["красный", "оранжевый"]


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param("Обычный текст из отчёта.\n\nВторой кусок.", id="текст"),
        pytest.param("[не json", id="битый JSON"),
        pytest.param('{"share": 0.4}', id="объект без fragments"),
        pytest.param('[{"text": "без уровня"}]', id="фрагмент без уровня"),
    ],
)
def test_heatmap_not_recognized(raw: str):
    """не тепловая карта — None, файл читается как текст"""
    assert marked_from_heatmap(raw) is None


# ─── сопоставление с документом ─────────────────────────────────────────


def test_piece_spanning_paragraphs_marks_all_of_them(doc: str):
    """кусок через несколько абзацев подсвечивает каждый, а не только первый"""
    marks = marked_fragments(doc, [" ".join(fragments(doc)[-2:])])
    assert [m.marked for m in marks] == [False, False, False, True, True]
    assert marks[-1].coverage == 1


def test_piece_inside_paragraph_marks_it(doc: str):
    """короткая подсветка внутри абзаца засчитывает абзац, как подсветка предложениями в «Антиплагиате»"""
    sentence = fragments(doc)[-1].split(". ")[1]
    marks = marked_fragments(doc, [sentence])
    assert [m.marked for m in marks] == [False, False, False, False, True]
    assert marks[-1].holds_piece
    assert 0 < marks[-1].coverage < 0.5


def test_pdf_text_layer_differences_do_not_matter(doc: str):
    """переносы строк, ё, регистр и знаки из текстового слоя PDF не мешают найти кусок"""
    p = fragments(doc)[-1]
    copied = p.replace(" ", "\n", 3).replace("е", "ё", 1).upper().replace(",", " ,")
    assert marked_fragments(doc, [copied])[-1].coverage == 1


def test_short_pieces_ignored(doc: str):
    """куски короче пяти слов не ищутся"""
    assert not any(m.marked for m in marked_fragments(doc, ["В работе рассмотрены"]))


def test_heatmap_drives_calibration_labels(doc: str):
    """красный фрагмент тепловой карты размечает фрагменты для калибровки"""
    from aiw_ru import label_fragments

    heat = json.dumps([{"level": 3, "text": fragments(doc)[-1]}, {"level": 2, "text": fragments(doc)[0]}])
    marked = marked_from_heatmap(heat)
    assert marked is not None
    assert [s.y for s in label_fragments(doc, marked)] == [0, 0, 0, 0, 1]


# ─── калибровка по системам ─────────────────────────────────────────────


def test_old_file_is_antiplagiat_calibration():
    """файл без systems — калибровка «Антиплагиата», «Думейта» в нём нет"""
    old = CalibrationFile.model_validate_json(
        json.dumps({"version": 1, "model": DEFAULT_MODEL.to_dict(), "samples": [{"f": [0.1] * 7, "y": 1}]})
    )
    cal = old.get("antiplagiat")
    assert cal is not None and len(cal.samples) == 1
    assert old.get("domate") is None


def test_systems_are_kept_apart():
    """калибровки систем не затирают друг друга"""
    model = DEFAULT_MODEL.model_copy(update={"threshold": 0.4})
    domate = Calibration(model=model, samples=[Sample(f=[0.0] * 7, y=0)])
    file = empty_calibration_file().put("domate", domate)
    assert file.get("antiplagiat") is None
    got = file.get("domate")
    assert got is not None and got.model.threshold == 0.4

    both = file.put("antiplagiat", Calibration(model=DEFAULT_MODEL, samples=[Sample(f=[1.0] * 7, y=1)]))
    data = json.loads(json.dumps(both.to_dict()))
    assert data["version"] == 1 and len(data["samples"]) == 1
    assert data["systems"]["domate"]["model"]["threshold"] == 0.4


def test_report_names_system(doc: str):
    """отчёт antiplagiat называет систему"""
    assert antiplagiat(doc).system == "antiplagiat"
    assert antiplagiat(doc, system="domate").to_dict()["system"] == "domate"


# ─── слепой судья ───────────────────────────────────────────────────────


def test_prompt_numbers_fragments_like_antiplagiat(doc: str):
    """задание нумерует те же фрагменты, что и antiplagiat, и не оставляет заглушек"""
    frs = judge_fragments(doc)
    assert [f.n for f in frs] == [1, 2, 3, 4, 5]
    assert [f.line for f in frs] == [f.line for f in antiplagiat(doc).fragments]
    prompt = judge_prompt(doc)
    assert prompt.startswith(PROMPT.split("\n", 1)[0])
    assert "[P1]…[P5]" in prompt and "\n[P5]\n" in prompt and "\nP5 ...\n" in prompt
    assert "{count}" not in prompt and "{fragments}" not in prompt


def test_parse_answer_formats():
    """ответ судьи читается в разных оформлениях, повтор номера берёт первый ответ"""
    answer = "P1 high шаблон\n**P2** — Medium: гладко\nP3: low\n- P4 none\nP1 none повтор\nP5 unknown\nвывод: high"
    assert parse_answer(answer) == {1: 3, 2: 2, 3: 1, 4: 0}


def test_auc():
    """ROC AUC с ничьими пополам; без одного из классов — nan"""
    assert auc([3, 2, 1, 0], [True, True, False, False]) == 1
    assert auc([1, 1], [True, False]) == 0.5
    assert math.isnan(auc([1, 2], [True, True]))


def test_score_means_agreement_and_auc(doc: str):
    """сводка: средний уровень, пропуски, совпадение прогонов, AUC против разметки"""
    runs = ["P1 none\nP2 none\nP3 low\nP4 high\nP5 medium", "P1 none\nP2 low\nP3 none\nP4 medium"]
    r = judge_score(doc, runs, marked=[" ".join(fragments(doc)[-2:])])
    assert r.runs == 2
    assert [f.mean for f in r.fragments] == [0, 0.5, 0.5, 2.5, 2]
    assert r.fragments[4].levels == [2, None]
    assert r.missing == []
    assert r.agreement == 0.2
    assert r.auc == 1
    assert r.detector_auc is not None
    words = [f.words for f in r.fragments]
    assert r.share_medium == round(sum(words[3:]) * 100 / sum(words), 1)


def test_score_without_marked_and_missing(doc: str):
    """без разметки AUC нет; фрагмент без ответа попадает в missing"""
    r = judge_score(doc, ["P1 none\nP2 low"])
    assert r.auc is None and r.detector_auc is None and r.agreement is None
    assert r.missing == [3, 4, 5]

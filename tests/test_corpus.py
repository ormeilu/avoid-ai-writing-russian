"""Корпус: живые тексты разных жанров не должны давать серьёзных находок,
шаблонные тексты моделей в тех же жанрах — должны. Каждый новый
детектор проверяется здесь на ложные срабатывания.
"""

import re
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from aiw_ru import ContextMode, analyze, antiplagiat

CORPUS = Path(__file__).parent.parent / "tests" / "fixtures" / "corpus"
MODE: dict[str, ContextMode] = {
    "vak": "academic",
    "docs": "technical",
    "blog": "general",
    "telegram": "social",
    "email": "general",
    "chat": "chat",
}


@dataclass(frozen=True)
class Doc:
    genre: str
    text: str


def load(kind: str) -> list[Doc]:
    return [Doc(p.stem, p.read_text(encoding="utf-8")) for p in sorted((CORPUS / kind).glob("*.md"))]


HUMAN = load("human")
AI = load("ai")


def by_genre(d: Doc) -> str:
    return d.genre


def genre(docs: list[Doc], name: str) -> str:
    return next(d.text for d in docs if d.genre == name)


# ─── живые тексты ───────────────────────────────────────────────────────


@pytest.mark.parametrize("doc", HUMAN, ids=by_genre)
def test_human_no_p0_p1(doc: Doc):
    """нет находок P0 и P1"""
    r = analyze(doc.text, MODE[doc.genre])
    assert [f"{i.type}: {i.text}" for i in r.issues if i.severity != "P2"] == []
    assert r.score < 15
    assert r.suspicious is False


@pytest.mark.parametrize("doc", HUMAN, ids=by_genre)
def test_human_antiplagiat_marks_nothing(doc: Doc):
    """antiplagiat не метит ни одного фрагмента"""
    assert antiplagiat(doc.text).ai_share == 0


def test_human_general_mode_no_p0():
    """то же в общем режиме: жанровые поблажки не маскируют ошибки"""
    for d in HUMAN:
        if d.genre == "chat":
            continue
        p0 = [i for i in analyze(d.text).issues if i.severity == "P0"]
        assert p0 == [], d.genre


# ─── шаблонные тексты моделей ───────────────────────────────────────────


@pytest.mark.parametrize("doc", AI, ids=by_genre)
def test_ai_high_score(doc: Doc):
    """высокая оценка"""
    assert analyze(doc.text, MODE[doc.genre]).score >= 60


@pytest.mark.parametrize("doc", [d for d in AI if d.genre != "telegram"], ids=by_genre)
def test_ai_antiplagiat_marks_over_half(doc: Doc):
    """antiplagiat метит больше половины"""
    assert antiplagiat(doc.text).ai_share > 50


def test_every_human_genre_below_any_ai_text():
    """каждый жанр живых текстов оценён ниже любого текста модели"""
    max_human = max(analyze(h.text, MODE[h.genre]).score for h in HUMAN)
    min_ai = min(analyze(a.text, MODE[a.genre]).score for a in AI)
    assert max_human < min_ai


def test_every_genre_has_pair():
    """у каждого жанра есть пара"""
    assert sorted(a.genre for a in AI) == sorted(h.genre for h in HUMAN if h.genre != "chat")


# ─── устойчивость ───────────────────────────────────────────────────────


def issue_types(text: str) -> list[str]:
    return sorted({i.type for i in analyze(text).issues})


def test_paragraph_shuffle_keeps_issue_set():
    """перестановка абзацев не меняет набор находок"""
    text = genre(AI, "blog")
    paras = text.split("\n\n")
    shuffled = "\n\n".join([paras[0], *reversed(paras[1:])])
    assert issue_types(shuffled) == issue_types(text)


def test_yo_and_e_same_result():
    """ё и е дают одинаковый результат"""
    text = genre(AI, "vak")
    yo = re.sub("е", lambda m: "ё" if m.start() % 7 == 0 else m[0], text)
    assert yo != text
    assert len(analyze(yo).issues) == len(analyze(text).issues)


def test_crlf_same_as_lf():
    """CRLF обрабатывается как LF"""
    text = genre(AI, "vak")
    assert analyze(text.replace("\n", "\r\n")).score == analyze(text).score


def test_big_document_is_fast():
    """большой документ обрабатывается быстро"""
    big = "\n\n".join(a.text for a in AI) * 10
    t0 = time.perf_counter()
    analyze(big)
    antiplagiat(big)
    assert time.perf_counter() - t0 < 4

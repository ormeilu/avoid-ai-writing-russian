"""Тесты scripts/long_docs.py: сборка длинных документов с разметкой ИИ-кусков и метрики по ним."""

from __future__ import annotations

import random
from collections.abc import Callable
from itertools import pairwise

import pytest

pytest.importorskip("sklearn")

import long_docs as ld


def text(label: str, k: int, size: int = 1000) -> str:
    """Уникальный текст нужной длины: номер в начале, чтобы начала не совпадали."""
    head = f"{label} {k:04d} "
    return (head + "слово " * size)[:size]


def rows(n: int = 60, data_type: str = "article") -> list[dict[str, str]]:
    return [
        *({"label": "human", "data_type": data_type, "text": text("человек", k)} for k in range(n)),
        *({"label": "ai", "data_type": data_type, "text": text("модель", k)} for k in range(n)),
    ]


def test_pools_drop_continuations_short_texts_and_poetry():
    human = text("человек", 1)
    continuation = human[: ld.PREFIX] + " и дальше пишет модель" * 30
    data = [
        {"label": "human", "data_type": "article", "text": human},
        {"label": "ai", "data_type": "article", "text": continuation},
        {"label": "ai", "data_type": "article", "text": text("модель", 1)},
        {"label": "ai", "data_type": "article", "text": "коротко"},
        {"label": "human", "data_type": "poetry", "text": text("стих", 1)},
    ]
    h, a = ld.pools(data)
    assert h == {"article": [human]}
    assert a == {"article": [text("модель", 1)]}


def test_compose_marks_ai_by_chars():
    doc, ai = ld.compose([("аа", False), ("бб", True), ("вв", False), ("гг", True)])
    assert doc == "аа\n\nбб\n\nвв\n\nгг"
    assert ai == [[4, 6], [12, 14]]
    assert [doc[a:b] for a, b in ai] == ["бб", "гг"]


def counter(prefix: str) -> Callable[[], str | None]:
    it = iter(text(prefix, k) for k in range(100))
    return lambda: next(it, None)


@pytest.mark.parametrize("kind", ld.KINDS)
def test_plan_puts_ai_where_the_kind_says(kind: str):
    parts = ld.plan(kind, 10000, counter("человек"), counter("модель"), random.Random(1))
    assert parts
    flags = [is_ai for _, is_ai in parts]
    assert sum(len(t) for t, _ in parts) >= 9000
    if kind == "human":
        assert not any(flags)
    elif kind == "ai":
        assert all(flags)
    elif kind == "start":
        assert flags[0] and not flags[-1]
    elif kind == "end":
        assert flags[-1] and not flags[0]
    elif kind == "middle":
        assert not flags[0] and not flags[-1] and any(flags)
    else:
        assert not flags[0] and not flags[-1] and 2 <= sum(flags) <= 3
        # вставки по одному тексту и не подряд
        assert all(not (x and y) for x, y in pairwise(flags))


def test_build_uses_each_text_once_and_marks_ai():
    data = rows()
    docs = ld.build(data, 12, seed=1)
    assert [d.kind for d in docs] == list(ld.KINDS) * 2
    ai_texts = {r["text"] for r in data if r["label"] == "ai"}
    seen: list[str] = []
    for d in docs:
        parts = d.text.split("\n\n")
        seen += parts
        assert {d.text[a:b] for a, b in d.ai} <= ai_texts
        assert sum(p in ai_texts for p in parts) == len(d.ai)
    assert len(seen) == len(set(seen))
    assert ld.build(data, 12, seed=1) == docs


def test_doc_metrics():
    doc = ld.Doc(0, "end", "article", "ч" * 50 + "и" * 50, [[50, 100]])
    run = {"threshold": 0.5, "head": 0.1, "seconds": 0.2, "chunks": [[0, 60, 0.9], [40, 100, 0.2]]}
    m = ld.doc_metrics(doc, run)
    assert ld.ai_share(0, 60, doc.ai) == pytest.approx(10 / 60)
    assert m["any"] and not m["two"] and not m["head"]
    # отмечен фрагмент, где ИИ меньше половины: ИИ-кусок не найден
    assert not m["detected"]
    assert m["recall"] == pytest.approx(10 / 50) and m["precision"] == pytest.approx(10 / 60)
    assert (m["trueShare"], m["flaggedShare"]) == (0.5, 0.6)


def test_metrics_split_fragments_by_ai_share():
    human = ld.Doc(0, "human", "article", "ч" * 100, [])
    mixed = ld.Doc(1, "middle", "article", "ч" * 100 + "и" * 100 + "ч" * 100, [[100, 200]])
    runs = {
        0: {"threshold": 0.5, "head": 0.6, "seconds": 0.1, "chunks": [[0, 50, 0.7], [50, 100, 0.1]]},
        1: {"threshold": 0.5, "head": 0.1, "seconds": 0.3, "chunks": [[0, 100, 0.2], [100, 200, 0.9], [150, 300, 0.4]]},
    }
    m = ld.metrics([human, mixed], runs)
    c = m["chunks"]
    assert (c["human"], c["ai"], c["boundary"]) == (3, 1, 1)
    assert c["falsePositive"] == pytest.approx(1 / 3) and c["truePositive"] == 1
    assert m["kinds"]["human"]["any"] == 1 and m["kinds"]["human"]["head"] == 1
    assert m["mixed"]["detected"] == 1 and m["mixed"]["recall"] == 1 and m["mixed"]["precision"] == 1
    assert m["fragments"] == 5 and m["msPerFragment"] == pytest.approx(80)


def test_replace_block_keeps_text_around():
    doc = "до\n<!-- long-docs:models -->\nстарое\n<!-- /long-docs:models -->\nпосле"
    assert ld.replace_block(doc, "models", "новое") == (
        "до\n<!-- long-docs:models -->\nновое\n<!-- /long-docs:models -->\nпосле"
    )
    with pytest.raises(SystemExit, match="нет меток"):
        ld.replace_block("без меток", "models", "x")

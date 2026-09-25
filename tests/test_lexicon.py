"""Контракт словаря: подписи и веса, уникальные идентификаторы, мини-язык фраз."""

import time
from pathlib import Path

import regex

from aiw_ru import ALL_LEXICON, TYPE_LABELS, WEIGHTS, phrase
from aiw_ru.compat import jsre

ROOT = Path(__file__).parent.parent

# U+200B–U+200D, U+2060 и BOM. Собираются через chr(), чтобы в исходнике не было самих символов.
INVISIBLE_RE = regex.compile("[" + chr(0x200B) + "-" + chr(0x200D) + chr(0x2060) + chr(0xFEFF) + "]")


def test_every_type_has_label_and_weight():
    """у каждого типа есть подпись и вес"""
    for e in ALL_LEXICON:
        assert e.type in TYPE_LABELS, e.type
        assert e.type in WEIGHTS, e.type
    for t in WEIGHTS:
        assert t in TYPE_LABELS, t


def test_rule_ids_unique_within_type():
    """идентификаторы правил уникальны внутри типа"""
    seen: set[str] = set()
    for e in ALL_LEXICON:
        key = f"{e.type}/{e.id}"
        assert key not in seen, key
        seen.add(key)


def test_phrase_mini_language():
    """мини-язык фраз"""
    re = jsre(phrase("игра* (ключев*|важн*)( роль)?"), "iu")
    assert re.search("играет ключевую роль")
    assert re.search("играла важную")
    assert not re.search("переиграет важную")
    assert jsre(phrase("\\*шепотом\\*"), "iu").search("*шепотом*")


def test_regexes_do_not_hang_on_long_text():
    """регулярные выражения не зацикливаются на длинном тексте"""
    long = "слово " * 20000
    t0 = time.perf_counter()
    for e in ALL_LEXICON:
        e.re.findall(long)
    assert (time.perf_counter() - t0) * 1000 < 3000


def test_no_invisible_chars_in_repo():
    """в репозитории нет невидимых символов"""
    files = [
        p
        for d in ("src", "test", "tests", "skills", "examples")
        for p in (ROOT / d).rglob("*")
        if p.is_file() and p.suffix in {".md", ".json", ".py"}
    ]
    files += [ROOT / f for f in ("README.md", "CONTRIBUTING.md", "AGENTS.md", "CLAUDE.md", "NOTICE.md", "CHANGELOG.md")]
    assert len(files) > 10
    for f in files:
        assert not INVISIBLE_RE.search(f.read_text(encoding="utf-8")), str(f.relative_to(ROOT))

"""Что видят модели: без YAML-шапки, кода и служебных частей, и только русский текст.

Крошечная модель из tests/fixtures/tiny-transformer читает слова из словаря: «ну вот короче типа» для неё
человек, «является ключевую данный еще» ИИ. Окно в четыре токена, абзацы из четырёх слов: один абзац — один фрагмент.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

lgb = pytest.importorskip("lightgbm")
np = pytest.importorskip("numpy")
pytest.importorskip("onnxruntime")
tokenizers = pytest.importorskip("tokenizers")

from aiw_ru import models
from aiw_ru.cli import main, not_russian_note
from aiw_ru.features import CONTEXT, FEATURE_NAMES, FEATURES_VERSION

FIXTURE = Path(__file__).parent / "fixtures" / "tiny-transformer"
VOCAB = {
    "[PAD]": 0,
    "[UNK]": 1,
    "[CLS]": 2,
    "[SEP]": 3,
    **{w: 4 + k for k, w in enumerate(["ну", "вот", "короче", "типа"])},
    **{w: 12 + k for k, w in enumerate(["является", "ключевую", "данный", "еще"])},
}

HUMAN = "ну вот короче типа"
AI = "является ключевую данный еще"
ENGLISH = "This text is English"

# В шапке слова ИИ: если модель их прочитает, вероятность первого окна изменится.
HEADER = f"---\ntitle: Обзор наборов данных\nabstract: {AI}\nabstract-en: {ENGLISH}\n---\n\n"
BODY = f"{HUMAN}\n\n{AI}\n\n{ENGLISH}\n\n{HUMAN}\n"
ARTICLE = HEADER + BODY
# Строки статьи: шапка 1–6, дальше абзацы через одну пустую строку.
LINES = [(7, 7), (9, 9), (11, 11), (13, 13)]


@pytest.fixture(autouse=True)
def calm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    # В окне на 4 токена четверть — одно слово; абзацы-предложения режутся целиком.
    monkeypatch.setattr(models, "UNIT", 1.0)


def bundle(folder: Path) -> Path:
    """Крошечный трансформер как на Hugging Face: окно в 4 токена, нормализация с пробелами и таблицами."""
    folder.mkdir(parents=True, exist_ok=True)
    tok = tokenizers.Tokenizer(tokenizers.models.WordLevel(VOCAB, unk_token="[UNK]"))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tok.save(str(folder / "tokenizer.json"))
    (folder / "model.onnx").write_bytes((FIXTURE / "model.onnx").read_bytes())
    spec = {
        "version": 1,
        "format": "onnx",
        "file": "model.onnx",
        "tokenizer": "tokenizer.json",
        "output": "logits",
        "labels": ["human", "ai"],
        "threshold": 0.5,
        "max_length": 6,
        "max_chars": 1_000_000,
        "cls_id": 2,
        "sep_id": 3,
        "pad_id": 0,
        "max_windows": 1,
        "long_texts": "head",
        "normalize": [
            {"pattern": "ё", "replacement": "е", "why": "ё"},
            # Правило из настоящих моделей: на строке из тысяч пробелов оно работает за квадрат времени.
            {
                "pattern": "^[ ]*\\|?(?:[ ]*:?-+:?[ ]*\\|)+[ ]*(?::?-+:?)?[ ]*$",
                "replacement": "",
                "why": "строка-разделитель таблицы",
            },
            {"pattern": "\\s+", "replacement": " ", "why": "пробелы и переводы строк"},
        ],
    }
    (folder / "inference.json").write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    return folder


@pytest.fixture
def transformer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Единственная установленная модель: крошечный трансформер."""
    folder = bundle(tmp_path / "transformer")
    monkeypatch.setenv("AIW_RU_TRANSFORMER_DIR", str(folder))
    for name in ("AIW_RU_MODERNBERT_DIR", "AIW_RU_MINI_FRIDA_DIR", "AIW_RU_MODEL_DIR"):
        monkeypatch.setenv(name, str(tmp_path / "нет"))
    models.load.cache_clear()
    yield folder
    models.load.cache_clear()


@pytest.fixture
def lightgbm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Единственная установленная модель: крошечный LightGBM на случайных признаках."""
    rng = np.random.default_rng(1)
    x = rng.random((200, len(FEATURE_NAMES)))
    booster = lgb.train({"objective": "binary", "verbose": -1}, lgb.Dataset(x, (x[:, 0] > 0.5).astype(int)), 3)
    folder = tmp_path / "lightgbm"
    folder.mkdir()
    booster.save_model(folder / "model.txt")
    spec = {"version": FEATURES_VERSION, "context": CONTEXT, "threshold": 0.5, "names": list(FEATURE_NAMES)}
    (folder / "features.json").write_text(json.dumps(spec), encoding="utf-8")
    monkeypatch.setenv("AIW_RU_MODEL_DIR", str(folder))
    for name in ("AIW_RU_MODERNBERT_DIR", "AIW_RU_MINI_FRIDA_DIR", "AIW_RU_TRANSFORMER_DIR"):
        monkeypatch.setenv(name, str(tmp_path / "нет"))
    models.load.cache_clear()
    yield folder
    models.load.cache_clear()


def put(tmp_path: Path, name: str, text: str) -> str:
    path = tmp_path / name
    # newline="": на Windows write_text иначе пишет \r\n, и в файле становится больше знаков.
    path.write_text(text, encoding="utf-8", newline="")
    return str(path)


def cli(capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str]:
    code = main(list(args))
    return code, capsys.readouterr().out


def classify(capsys: pytest.CaptureFixture[str], *args: str) -> dict[str, Any]:
    code, out = cli(capsys, "classify", "--json", *args)
    assert code == 0
    return json.loads(out)


def flat(out: str) -> str:
    """Вывод в одну строку: таблицы и заметки переносятся по ширине терминала."""
    return " ".join(out.split())


# ── YAML-шапка ──


def test_header_is_not_in_fragments(transformer: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """первый фрагмент начинается после шапки, строки и смещения считаются по исходному файлу"""
    data = classify(capsys, put(tmp_path, "article.md", ARTICLE))
    items = data["fragments"]["items"]
    assert [(i["line"], i["endLine"]) for i in items] == LINES
    assert [ARTICLE[i["start"] : i["end"]] for i in items] == [HUMAN, AI, ENGLISH, HUMAN]
    # модель прочитала слова тела, а не шапки: шестнадцать слов в четырёх абзацах
    assert data["read"]["totalWords"] == 16 and data["read"]["totalLines"] == 13


@pytest.mark.parametrize("command", ["classify", "scan", "antiplagiat"])
def test_header_does_not_change_probability(
    transformer: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], command: str
):
    """вероятность по статье с шапкой та же, что по её телу: слова ИИ в шапке модель не видит"""

    def probability(path: str) -> float:
        data = json.loads(cli(capsys, command, "--json", path)[1])
        return (data["classifier"] if command != "classify" else data)["probability"]

    with_header = probability(put(tmp_path, "article.md", ARTICLE))
    assert with_header == probability(put(tmp_path, "body.md", BODY))
    # слова шапки без тела дали бы вероятность выше порога
    assert models.probability(AI, model=models.TRANSFORMER) > 0.5 > with_header


def test_header_does_not_change_lightgbm(lightgbm: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """LightGBM считает признаки детектора: шапка, а с ней и слова из неё, в них не попадает"""
    body = put(tmp_path, "body.md", "Мы сравнили модели на трёх наборах данных. " * 3 + "\n")
    article = put(
        tmp_path,
        "article.md",
        "---\nabstract: Короткая русская аннотация.\n---\n\n"
        + "Мы сравнили модели на трёх наборах данных. " * 3
        + "\n",
    )
    assert (
        classify(capsys, "--model", "lightgbm", article)["probability"]
        == classify(capsys, "--model", "lightgbm", body)["probability"]
    )


# ── английский абзац в теле ──


def test_english_paragraph_is_skipped(transformer: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """абзац не на русском помечен как непроверенный и не входит ни в aiLines, ни в долю слов"""
    f = classify(capsys, put(tmp_path, "article.md", ARTICLE))["fragments"]
    assert (f["count"], f["skipped"], f["total"], f["aboveThreshold"]) == (3, 1, 4, 1)
    ru = [i for i in f["items"] if not i["skipped"]]
    assert [i["language"] for i in ru] == ["ru"] * 3
    assert all(isinstance(i["probability"], float) and isinstance(i["ai"], bool) for i in ru)
    english = f["items"][2]
    assert english["language"] == "other" and english["skipped"] is True
    assert english["probability"] is None and english["ai"] is None and english["nearThreshold"] is None
    assert (english["line"], english["words"]) == (11, 4)
    assert f["aiLines"] == [[9, 9]]
    # слова ИИ-фрагмента делятся на слова оценённого текста: 12 слов из 16, без английского абзаца
    assert f["aiWordShare"] == round(4 / 12, 4)


def test_english_paragraph_in_text_output(transformer: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    out = flat(cli(capsys, "classify", put(tmp_path, "article.md", ARTICLE))[1])
    assert "не на русском" in out
    assert "1 фрагмент не на русском, модель его не оценивала." in out
    assert "Выше порога 1 фрагмент из 3, в них 33 % слов: строка 9." in out
    assert "Проверено 3 фрагмента" in out


def test_max_fragments_counts_skipped(transformer: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """предел фрагментов: пропущенный английский фрагмент входит в просмотренные"""
    f = classify(capsys, "--max-fragments", "3", put(tmp_path, "article.md", ARTICLE))["fragments"]
    assert (f["count"], f["skipped"], f["total"], len(f["items"])) == (2, 1, 4, 3)


def test_russian_signal_has_language(transformer: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    data = classify(capsys, put(tmp_path, "article.md", ARTICLE))
    assert data["language"] == "ru" and data["skipped"] is False
    assert 0.5 < data["cyrillicShare"] < 1
    assert isinstance(data["probability"], float) and data["read"] is not None


def test_not_russian_note_agrees_with_number():
    assert not_russian_note(1) == "1 фрагмент не на русском, модель его не оценивала."
    assert not_russian_note(2) == "2 фрагмента не на русском, модель их не оценивала."
    assert not_russian_note(5) == "5 фрагментов не на русском, модель их не оценивала."
    assert not_russian_note(11) == "11 фрагментов не на русском, модель их не оценивала."
    assert not_russian_note(21) == "21 фрагмент не на русском, модель его не оценивала."


# ── текст целиком не на русском ──

TEXTS = {
    "english": "This review compares datasets.\n\nResults are shown in the table below.\n",
    # шапка и английская аннотация: после неё в теле нет ни одной буквы
    "header only": f"---\nabstract: {HUMAN}\nabstract-en: {ENGLISH}\n---\n",
    "empty": "",
}


@pytest.mark.parametrize("name", TEXTS)
def test_text_not_in_russian_has_no_probability(
    transformer: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], name: str
):
    path = put(tmp_path, "text.md", TEXTS[name])
    data = classify(capsys, path)
    assert (data["language"], data["skipped"]) == ("other", True)
    assert data["probability"] is None and data["ai"] is None and data["nearThreshold"] is None
    assert data["read"] is None and "fragments" not in data
    assert data["cyrillicShare"] == (0 if name == "english" else None)
    # scan и antiplagiat говорят то же в classifier; код выхода прежний
    for command in ("scan", "antiplagiat"):
        code, out = cli(capsys, command, "--json", path)
        classifier = json.loads(out)["classifier"]
        assert code == 0 and classifier["skipped"] and classifier["probability"] is None
        code, out = cli(capsys, command, path)
        assert code == 0 and "Вероятность ИИ" not in out
        assert "не применима к этому тексту" in flat(out)


def test_english_text_output(transformer: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    path = put(tmp_path, "text.md", TEXTS["english"])
    out = flat(cli(capsys, "classify", path)[1])
    assert "Модель не применима к этому тексту: кириллицы 0 % букв, а модель обучена на русском (трансформер)" in out
    assert "Вероятность ИИ" not in out and "фрагмент" not in out


def test_text_without_letters_says_so(transformer: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    out = flat(cli(capsys, "classify", put(tmp_path, "text.md", TEXTS["header only"]))[1])
    assert "в нём нет букв" in out and "шапка, код и служебные части" in out


def test_classify_all_for_english(transformer: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch):
    """все модели получают один и тот же текст: ответ один, согласия или расхождения нет"""
    folder = bundle(tmp_path / "modernbert")
    monkeypatch.setenv("AIW_RU_MODERNBERT_DIR", str(folder))
    models.load.cache_clear()
    path = put(tmp_path, "text.md", TEXTS["english"])
    code, out = cli(capsys, "classify", "--all", "--json", path)
    data = json.loads(out)
    assert code == 0 and [m["name"] for m in data["models"]] == ["modernbert", "transformer"]
    assert all(m["skipped"] and m["probability"] is None for m in data["models"])
    assert data["agree"] is None
    out = flat(cli(capsys, "classify", "--all", path)[1])
    assert "Модели не применимы к этому тексту: кириллицы 0 % букв, а модели обучены на русском" in out
    assert "Модели расходятся" not in out and "Все модели" not in out
    # русский текст по-прежнему сравнивается
    data = json.loads(cli(capsys, "classify", "--all", "--json", put(tmp_path, "ru.md", ARTICLE))[1])
    assert data["agree"] is True and not any(m["skipped"] for m in data["models"])


def test_classify_jsonl_marks_each_document(transformer: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    lines = [json.dumps({"text": t}, ensure_ascii=False) for t in (TEXTS["english"], ARTICLE)]
    code, out = cli(capsys, "classify", "--jsonl", put(tmp_path, "docs.jsonl", "\n".join(lines) + "\n"))
    first, second = (json.loads(line) for line in out.splitlines())
    assert code == 0 and first["skipped"] and first["probability"] is None
    assert not second["skipped"] and second["probability"] is not None


def test_lightgbm_skips_english_too(lightgbm: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    data = classify(capsys, "--model", "lightgbm", put(tmp_path, "text.md", TEXTS["english"]))
    assert data["name"] == "lightgbm" and data["skipped"] and data["probability"] is None


def test_half_cyrillic_is_enough():
    """порог: кириллицы не меньше половины букв; цифры и знаки не считаются, без букв модели нечего читать"""
    assert models.RUSSIAN_SHARE == 0.5
    assert models.is_russian("аб cd") and not models.is_russian("а bcd")
    assert models.is_russian("12 аб, 34 cd.") and not models.is_russian("12, 34.")
    assert models.russian(1) and not models.russian(None) and not models.russian(0.49)


# ── служебные части ──

SERVICE = (
    f"{HUMAN}\n\n{AI}\n\n"
    "Ключевые слова: ну, вот, короче\n\n"
    "## Список литературы\n\n"
    "1. Иванов И. И. Оценка // Транспорт. — 2023. — № 4. — С. 12–19.\n\n"
    "2. Smith J. Drowsiness // Sensors. — 2023. — Vol. 23. — P. 8386.\n"
)


def test_service_parts_are_not_fragments(transformer: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """ключевые слова и записи литературы фрагментов не образуют; заголовок раздела остаётся текстом"""
    data = classify(capsys, put(tmp_path, "article.md", SERVICE))
    items = data["fragments"]["items"]
    assert [SERVICE[i["start"] : i["end"]] for i in items] == [HUMAN, AI, "## Список литературы"]
    assert [(i["line"], i["skipped"]) for i in items] == [(1, False), (3, False), (7, False)]


def test_code_block_is_not_read(transformer: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    text = f"{HUMAN}\n\n```python\nprint('hello world')\n```\n\n{AI}\n"
    items = classify(capsys, put(tmp_path, "code.md", text))["fragments"]["items"]
    assert [text[i["start"] : i["end"]] for i in items] == [HUMAN, AI]
    assert [i["line"] for i in items] == [1, 7]


def test_scan_chunks_works_on_source_text(transformer: Path):
    """смещения фрагментов указывают в исходник, scan.text той же длины, а слова считаются без шапки"""
    scan = models.scan_chunks(models.load(models.TRANSFORMER), ARTICLE)
    assert len(scan.text) == len(ARTICLE) and scan.text.count("\n") == ARTICLE.count("\n")
    assert [ARTICLE[c.start : c.end] for c in scan.chunks] == [HUMAN, AI, ENGLISH, HUMAN]
    assert [c.skipped for c in scan.chunks] == [False, False, True, False]
    assert (len(scan.rated), scan.skipped, scan.words) == (3, 1, 12)
    assert [(c.line, c.end_line) for c in scan.chunks] == LINES


def test_overlap_of_foreign_and_russian_chunks_stays_in_denominator(transformer: Path):
    """слова, которые есть и во фрагменте не на русском, и в русском (перекрытие), остаются в знаменателе"""
    spans = models._outside([(0, 10)], [(2, 4), (6, 8)])
    assert spans == [(0, 2), (4, 6), (8, 10)]
    assert models._outside([(0, 10), (12, 20)], [(5, 15)]) == [(0, 5), (15, 20)]
    assert models._outside([(0, 10)], [(0, 10)]) == []
    assert models._merged([(5, 8), (0, 3), (3, 4), (7, 9)]) == [(0, 4), (5, 9)]


# ── длинные пропуски на месте шапки и литературы ──


def test_long_blank_runs_are_cheap(transformer: Path):
    """строка из десятков тысяч пробелов, оставшаяся от вырезанной части, не тормозит нормализацию и нарезку"""
    spec = json.loads((transformer / "inference.json").read_text(encoding="utf-8"))
    (transformer / "inference.json").write_text(json.dumps(spec | {"max_length": 20}), encoding="utf-8")
    models.load.cache_clear()
    loaded = models.load(models.TRANSFORMER)
    assert isinstance(loaded, models._Transformer)
    started = time.perf_counter()
    text = f"{HUMAN}\n" + " " * 60_000 + f"\n{AI}"
    assert loaded.normalize(text) == f"{HUMAN} {AI}"
    # хвост из пропусков после точки: sentences() на таком хвосте работает за квадрат времени
    tail = f"{HUMAN}.\n\n{AI}.\n" + " " * 60_000
    spans = models.chunk_spans(loaded, tail, overlap=0)
    assert [tail[a:b] for a, b in spans] == [f"{HUMAN}.\n\n{AI}."]
    assert time.perf_counter() - started < 3


def test_normalize_keeps_short_whitespace(transformer: Path):
    """серия короче 64 знаков правилам не мешает: результат тот, что записан в inference.json"""
    loaded = models.load(models.TRANSFORMER)
    assert isinstance(loaded, models._Transformer)
    assert loaded.normalize("ну   вот\n\n\nкороче") == "ну вот короче"
    assert loaded.normalize("ну" + " " * 63 + "вот") == "ну вот"
    assert loaded.normalize("ну" + " " * 64 + "вот") == "ну вот"


# ── длинный абзац не на русском вырезается до модели ──

LONG_ENGLISH = "This paragraph is written in English and quotes the original abstract"
WITH_QUOTE = f"{HUMAN}\n\n{LONG_ENGLISH}\n\n{AI}\n"


def test_long_foreign_paragraph_is_cut(transformer: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """абзац не на русском длиннее FOREIGN_LETTERS букв модель не читает: строки названы, фрагмента нет,
    вероятность та же, что без него"""
    data = classify(capsys, put(tmp_path, "quote.md", WITH_QUOTE))
    assert data["notRussianLines"] == [[3, 3]]
    assert data["probability"] == classify(capsys, put(tmp_path, "plain.md", f"{HUMAN}\n\n\n\n{AI}\n"))["probability"]
    lines = [(i["line"], i["endLine"]) for i in data["fragments"]["items"]]
    assert (3, 3) not in lines
    assert data["fragments"]["skipped"] == 0


def test_long_foreign_paragraph_in_scan_output(transformer: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    out = flat(cli(capsys, "scan", put(tmp_path, "quote.md", WITH_QUOTE))[1])
    assert "Не на русском строка 3: модель её не читала" in out


def test_foreign_lines_ignore_short_latin():
    """короткая строка латиницей («Python 3.12») и русский абзац с названиями моделей не вырезаются"""
    text = "Python 3.12\n\nМодели SAM и DINOv2 мы запускали на одной видеокарте, без дообучения.\n"
    assert models.foreign_lines(text) == []
    assert models.visible(text) == text

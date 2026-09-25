"""Необязательные модели: LightGBM и трансформер в ONNX. Загрузка, проверки версий, вероятность, CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

lgb = pytest.importorskip("lightgbm")
np = pytest.importorskip("numpy")

from aiw_ru import analyze, models
from aiw_ru.cli import main
from aiw_ru.features import CONTEXT, FEATURE_NAMES, FEATURES_VERSION

TEXT = "В современном мире искусственный интеллект играет ключевую роль. Данная технология является важной."


@pytest.fixture
def model_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Крошечная модель на случайных признаках в папке из AIW_RU_MODEL_DIR."""
    rng = np.random.default_rng(1)
    x = rng.random((200, len(FEATURE_NAMES)))
    y = (x[:, 0] > 0.5).astype(int)
    booster = lgb.train({"objective": "binary", "verbose": -1}, lgb.Dataset(x, y), num_boost_round=3)
    booster.save_model(tmp_path / "model.txt")
    spec = {"version": FEATURES_VERSION, "context": CONTEXT, "threshold": 0.5, "names": list(FEATURE_NAMES)}
    (tmp_path / "features.json").write_text(json.dumps(spec), encoding="utf-8")
    monkeypatch.setenv("AIW_RU_MODEL_DIR", str(tmp_path))
    models.load.cache_clear()
    yield tmp_path
    models.load.cache_clear()


def test_probability(model_dir: Path):
    assert models.available()
    assert models.local_path() == model_dir
    p = models.probability(TEXT)
    assert 0 <= p <= 1
    assert models.threshold() == 0.5


def test_probability_reuses_result_only_in_feature_context(model_dir: Path):
    """Ответ детектора в другом режиме не подходит: признаки считаются в режиме general."""
    base = models.probability(TEXT)
    assert models.probability(TEXT, analyze(TEXT, CONTEXT)) == base
    assert models.probability(TEXT, analyze(TEXT, "academic")) == base


def test_feature_version_mismatch(model_dir: Path):
    spec = json.loads((model_dir / "features.json").read_text(encoding="utf-8"))
    spec["version"] = FEATURES_VERSION + 1
    (model_dir / "features.json").write_text(json.dumps(spec), encoding="utf-8")
    assert not models.available()
    with pytest.raises(models.ModelError, match="версии"):
        models.load()


def test_missing_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AIW_RU_MODEL_DIR", str(tmp_path / "нет"))
    models.load.cache_clear()
    assert models.local_path() is None
    with pytest.raises(models.ModelError, match="models install"):
        models.load()
    models.load.cache_clear()


def test_missing_dependencies(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(models, "find_spec", lambda name: None)
    models.load.cache_clear()
    assert models.missing_dependencies() == ["lightgbm", "huggingface_hub"]
    with pytest.raises(models.ModelError, match="extra ml"):
        models.install()
    models.load.cache_clear()


def test_lightgbm_that_does_not_load(model_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """lightgbm стоит, но не загружается (на macOS без libomp): понятная ошибка, а не трассировка"""
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "lightgbm":
            raise OSError("Library not loaded: @rpath/libomp.dylib")
        return real_import(name, *args, **kwargs)

    models.load.cache_clear()
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(models.ModelError, match="lightgbm не загружается"):
        models.load()
    assert not models.available()
    models.load.cache_clear()


# ── трансформер ──

FIXTURE = Path(__file__).parent / "fixtures" / "tiny-transformer"
# Номера слов: у крошечной модели logit ИИ растёт со средним номером токена (порог — 8).
VOCAB = {
    "[PAD]": 0,
    "[UNK]": 1,
    "[CLS]": 2,
    "[SEP]": 3,
    **{w: 4 + k for k, w in enumerate(["ну", "вот", "короче", "типа"])},
    **{w: 12 + k for k, w in enumerate(["является", "ключевую", "данный", "еще"])},
}


def onnx_bundle(folder: Path) -> Path:
    """Папка модели как на Hugging Face: крошечный ONNX-граф, словарь на восемь слов, inference.json."""
    pytest.importorskip("onnxruntime")
    tokenizers = pytest.importorskip("tokenizers")
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
        "max_chars": 10_000,
        "cls_id": 2,
        "sep_id": 3,
        "pad_id": 0,
        "max_windows": 8,
        "normalize": [{"pattern": "ё", "replacement": "е", "why": "ё"}],
    }
    (folder / "inference.json").write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    return folder


@pytest.fixture
def transformer_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Трансформер из крошечного графа; ModernBERT и mini-frida из кэша разработчика не подхватываются."""
    folder = onnx_bundle(tmp_path / "transformer")
    monkeypatch.setenv("AIW_RU_TRANSFORMER_DIR", str(folder))
    monkeypatch.setenv("AIW_RU_MODERNBERT_DIR", str(tmp_path / "нет"))
    monkeypatch.setenv("AIW_RU_MINI_FRIDA_DIR", str(tmp_path / "нет"))
    models.load.cache_clear()
    yield folder
    models.load.cache_clear()


@pytest.fixture
def modernbert_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    folder = onnx_bundle(tmp_path / "modernbert")
    monkeypatch.setenv("AIW_RU_MODERNBERT_DIR", str(folder))
    models.load.cache_clear()
    yield folder
    models.load.cache_clear()


@pytest.fixture
def mini_frida_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    folder = onnx_bundle(tmp_path / "mini-frida")
    monkeypatch.setenv("AIW_RU_MINI_FRIDA_DIR", str(folder))
    models.load.cache_clear()
    yield folder
    models.load.cache_clear()


def test_transformer_probability(transformer_dir: Path):
    t = models.TRANSFORMER
    assert models.available(t) and models.local_path(t) == transformer_dir
    assert models.probability("является ключевую данный", model=t) > 0.5
    assert models.probability("ну вот короче типа", model=t) < 0.5
    # ё приводится к е так же, как при обучении
    assert models.probability("ещё", model=t) == models.probability("еще", model=t)


def test_transformer_averages_windows(transformer_dir: Path):
    """длинный текст режется на окна по max_length токенов, вероятность — среднее по окнам"""
    assert models.windows(list(range(10, 20)), 6, [2], [3], 8) == [
        [2, 10, 11, 12, 13, 3],
        [2, 14, 15, 16, 17, 3],
        [2, 18, 19, 3],
    ]
    assert models.windows([], 6, [2], [3], 8) == [[2, 3]]
    assert len(models.windows(list(range(100)), 6, [2], [3], 8)) == 8
    # T5: префикс задачи из нескольких токенов и </s> в конце, без [CLS]
    assert models.windows(list(range(10, 14)), 6, [7, 8], [1], 8) == [[7, 8, 10, 11, 12, 1], [7, 8, 13, 1]]
    t = models.TRANSFORMER
    ai, human = (
        models.probability("является ключевую данный еще", model=t),
        models.probability("ну вот короче типа", model=t),
    )
    mixed = models.probability("является ключевую данный еще ну вот короче типа", model=t)
    assert mixed == pytest.approx((ai + human) / 2, abs=1e-6)


def test_prefix_and_suffix_ids(transformer_dir: Path):
    """inference.json с prefix_ids и suffix_ids вместо cls_id и sep_id, как у T5"""
    spec = json.loads((transformer_dir / "inference.json").read_text(encoding="utf-8"))
    base = models.probability("является ключевую", model=models.TRANSFORMER)
    del spec["cls_id"], spec["sep_id"]
    spec["prefix_ids"], spec["suffix_ids"] = [2], [3]
    (transformer_dir / "inference.json").write_text(json.dumps(spec), encoding="utf-8")
    models.load.cache_clear()
    assert models.probability("является ключевую", model=models.TRANSFORMER) == base
    # префикс из слов конца словаря тянет вероятность к ИИ
    spec["prefix_ids"] = [15, 15]
    (transformer_dir / "inference.json").write_text(json.dumps(spec), encoding="utf-8")
    models.load.cache_clear()
    assert models.probability("является ключевую", model=models.TRANSFORMER) > base


def test_onnx_external_data_is_downloaded():
    """большой граф лежит в model.onnx и model.onnx.data: оба попадают в загрузку"""
    from fnmatch import fnmatch

    for name in ("model.onnx", "model.onnx.data"):
        assert any(fnmatch(name, p) for p in models.MODERNBERT.files)
    assert not any(fnmatch("model_fp32.onnx", p) for p in models.MODERNBERT.files)


def set_spec(folder: Path, **changes: Any) -> None:
    """Правит inference.json крошечной модели и сбрасывает кэш загрузки."""
    spec = json.loads((folder / "inference.json").read_text(encoding="utf-8"))
    (folder / "inference.json").write_text(json.dumps(spec | changes), encoding="utf-8")
    models.load.cache_clear()


def test_coverage_of_text_longer_than_window(transformer_dir: Path):
    """модель читает одно окно: видно, сколько слов, строк и токенов она прочитала и где остановилась"""
    set_spec(transformer_dir, max_windows=1, long_texts="head")
    loaded = models.load(models.TRANSFORMER)
    text = "ну вот короче типа данный\nеще является ключевую"
    c = loaded.coverage(text)
    assert c.truncated and c.window == 6
    assert (c.words, c.total_words, c.tokens, c.total_tokens) == (4, 8, 4, 8)
    assert (c.lines, c.total_lines) == (1, 2)
    assert text[: c.chars] == "ну вот короче типа"
    short = loaded.coverage("ну вот\n")
    assert not short.truncated and (short.words, short.total_words, short.lines, short.total_lines) == (2, 2, 1, 1)
    # среднее по восьми окнам читает 32 токена: весь текст
    set_spec(transformer_dir, max_windows=8, long_texts="mean_of_windows")
    assert not models.load(models.TRANSFORMER).coverage(text).truncated


def test_lightgbm_reads_whole_text(model_dir: Path):
    c = models.load(models.LIGHTGBM).coverage(TEXT * 50)
    assert not c.truncated and c.tokens is None and c.window is None and c.words == c.total_words


SENTENCES = ["Ну вот.", "Короче типа.", "Данный еще.", "Является ключевую.", "Ну типа."]


def spans_text(text: str, spans: list[tuple[int, int]]) -> list[str]:
    return [text[a:b] for a, b in spans]


def test_chunks_follow_sentences_with_overlap(transformer_dir: Path):
    """фрагменты режутся по предложениям в пределах окна; соседние заходят друг на друга"""
    set_spec(transformer_dir, max_length=10, max_windows=1)  # окно на 8 токенов, в предложении 3
    loaded = models.load(models.TRANSFORMER)
    text = " ".join(SENTENCES[:4])
    assert spans_text(text, models.chunk_spans(loaded, text, overlap=0, unit=1)) == [
        "Ну вот. Короче типа.",
        "Данный еще. Является ключевую.",
    ]
    assert spans_text(text, models.chunk_spans(loaded, text, overlap=0.5, unit=1)) == [
        "Ну вот. Короче типа.",
        "Короче типа. Данный еще.",
        "Данный еще. Является ключевую.",
    ]
    # короткий хвост добирается назад до полного окна, а не остаётся одним предложением
    text = " ".join(SENTENCES)
    assert spans_text(text, models.chunk_spans(loaded, text, overlap=0, unit=1))[-1] == "Является ключевую. Ну типа."
    # текст в одно окно — один фрагмент
    assert models.chunk_spans(loaded, "Ну вот.", overlap=0.25) == [(0, 7)]


def test_fragments_fill_the_window(transformer_dir: Path):
    """длинное предложение не обрывает фрагмент раньше: единица нарезки не больше четверти окна"""
    set_spec(transformer_dir, max_length=10, max_windows=1)  # окно на 8 токенов
    loaded = models.load(models.TRANSFORMER)
    text = "Ну вот. Короче типа данный еще является ключевую."
    # предложение целиком: первый фрагмент — одно короткое предложение
    assert spans_text(text, models.chunk_spans(loaded, text, overlap=0, unit=1)) == [
        "Ну вот.",
        "Короче типа данный еще является ключевую.",
    ]
    assert spans_text(text, models.chunk_spans(loaded, text, overlap=0, unit=0.5)) == [
        "Ну вот. Короче типа данный",
        "Короче типа данный еще является ключевую.",
    ]


def test_long_sentence_is_split_by_words(transformer_dir: Path):
    set_spec(transformer_dir, max_length=6, max_windows=1)  # окно на 4 токена
    loaded = models.load(models.TRANSFORMER)
    text = "ну вот короче типа данный еще является ключевую"
    spans = models.chunk_spans(loaded, text, overlap=0)
    assert spans_text(text, spans) == ["ну вот короче типа", "данный еще является ключевую"]


def test_chunk_probabilities(transformer_dir: Path):
    """вероятность по каждому фрагменту: видно, где в документе ИИ"""
    set_spec(transformer_dir, max_length=6, max_windows=1)
    loaded = models.load(models.TRANSFORMER)
    text = "ну вот короче типа\n\nявляется ключевую данный еще"
    scan = models.scan_chunks(loaded, text, overlap=0)
    assert [(c.line, c.end_line, c.words) for c in scan.chunks] == [(1, 1, 4), (3, 3, 4)]
    human, ai = scan.chunks
    assert human.probability < loaded.threshold < ai.probability
    assert scan.total == 2 and scan.seconds >= 0
    assert ai.probability == pytest.approx(loaded.probability("является ключевую данный еще"), abs=1e-6)
    limited = models.scan_chunks(loaded, text, overlap=0, limit=1)
    assert len(limited.chunks) == 1 and limited.total == 2


class _Tokenizer0:
    """Tokenizer из tokenizers 0.x: обрезку и дополнение снимают методы no_*."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def no_truncation(self) -> None:
        self.calls.append("truncation")

    def no_padding(self) -> None:
        self.calls.append("padding")


class _Tokenizer1:
    """Tokenizer из tokenizers 1.x: методов no_* нет, обрезка и дополнение снимаются присваиванием None."""

    truncation: object = "обрезка"
    padding: object = "дополнение"


def test_tokenizer_without_limits_in_both_versions():
    """окна режет сам aiw-ru, поэтому обрезку и дополнение токенизатора снимаем в любой версии tokenizers"""
    old = _Tokenizer0()
    models._no_limits(old)
    assert old.calls == ["truncation", "padding"]
    new = _Tokenizer1()
    models._no_limits(new)
    assert new.truncation is None and new.padding is None


def test_transformer_format_mismatch(transformer_dir: Path):
    spec = json.loads((transformer_dir / "inference.json").read_text(encoding="utf-8"))
    spec["version"] = models.INFERENCE_VERSION + 1
    (transformer_dir / "inference.json").write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(models.ModelError, match="обновите aiw-ru"):
        models.load(models.TRANSFORMER)


# Свежий процесс: в этом onnxruntime уже импортирован, а ORT_DISABLE_TELEMETRY стоит из conftest.py.
TELEMETRY_SPY = """
import os, sys

seen = []


class Spy:
    def find_spec(self, name, path=None, target=None):
        if name == "onnxruntime":
            seen.append(os.environ.get("ORT_DISABLE_TELEMETRY"))


sys.meta_path.insert(0, Spy())
import aiw_ru.cli

print(os.environ.get("ORT_DISABLE_TELEMETRY"))
from aiw_ru import models

models.load(models.TRANSFORMER)
# Первым meta_path спрашивает find_spec из missing_dependencies, последним — сам импорт.
print(seen[-1])
"""


def test_onnxruntime_is_imported_without_telemetry(tmp_path: Path):
    """телеметрию onnxruntime выключает переменная окружения: она стоит с запуска CLI и в момент импорта"""
    folder = onnx_bundle(tmp_path / "transformer")
    env = {k: v for k, v in os.environ.items() if k != "ORT_DISABLE_TELEMETRY"}
    env["AIW_RU_TRANSFORMER_DIR"] = str(folder)
    r = subprocess.run(
        [sys.executable, "-c", TELEMETRY_SPY], env=env, capture_output=True, text=True, timeout=120, check=False
    )
    assert r.returncode == 0, r.stderr[-4000:]
    assert r.stdout.split() == ["1", "1"]


def chosen(name: str | None = None) -> models.Model | None:
    loaded = models.preferred(name)
    return None if loaded is None else loaded.model


def test_preferred_model(model_dir: Path, transformer_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """по умолчанию трансформер, без него LightGBM, без обеих — ничего"""
    assert chosen() is models.TRANSFORMER
    assert chosen("lightgbm") is models.LIGHTGBM
    monkeypatch.setenv("AIW_RU_TRANSFORMER_DIR", str(transformer_dir / "нет"))
    models.load.cache_clear()
    assert chosen() is models.LIGHTGBM
    monkeypatch.setenv("AIW_RU_MODEL_DIR", str(model_dir / "нет"))
    models.load.cache_clear()
    assert chosen() is None
    with pytest.raises(models.ModelError, match="не скачана"):
        models.preferred("transformer")


# ── CLI ──


def cli(capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str, str]:
    code = main(list(args))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


@pytest.fixture
def texts(tmp_path: Path) -> dict[str, str]:
    """Файлы с текстами для CLI: ai — слова из конца словаря крошечной модели, human — из начала."""
    out = {}
    for name, text in (("ai", "является ключевую данный"), ("human", "ну вот короче типа"), ("long", TEXT)):
        path = tmp_path / f"{name}.md"
        path.write_text(text, encoding="utf-8")
        out[name] = str(path)
    return out


def test_cli_classify_and_scan_signal(
    model_dir: Path,
    transformer_dir: Path,
    texts: dict[str, str],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("NO_COLOR", "1")
    code, out, _ = cli(capsys, "classify", "--json", texts["ai"])
    assert code == 0
    data = json.loads(out)
    assert data["name"] == "transformer" and data["ai"] and data["repo"] == models.TRANSFORMER.repo
    code, out, _ = cli(capsys, "classify", "--json", "--model", "lightgbm", texts["long"])
    assert json.loads(out)["name"] == "lightgbm"
    code, out, _ = cli(capsys, "scan", "--json", texts["ai"])
    assert json.loads(out)["classifier"]["name"] == "transformer"
    code, out, _ = cli(capsys, "antiplagiat", "--json", texts["ai"])
    report = json.loads(out)
    assert report["model"] == "default" and report["classifier"]["name"] == "transformer"
    code, out, _ = cli(capsys, "scan", texts["human"])
    assert "Вероятность ИИ" in out and "трансформер, порог 50 %" in out
    code, out, _ = cli(capsys, "scan", "--json", "--no-model", texts["long"])
    assert "classifier" not in json.loads(out)


def test_cli_models_status(model_dir: Path, transformer_dir: Path, capsys: pytest.CaptureFixture[str]):
    code, out, _ = cli(capsys, "models", "--json")
    data = json.loads(out)
    assert code == 0 and data["ready"]
    ready = {m["name"]: m["ready"] for m in data["models"]}
    assert ready == {"modernbert": False, "mini-frida": False, "transformer": True, "lightgbm": True}
    assert "models install modernbert" in data["models"][0]["error"]


def test_modernbert_is_opt_in_and_preferred(
    model_dir: Path,
    transformer_dir: Path,
    modernbert_dir: Path,
    texts: dict[str, str],
    capsys: pytest.CaptureFixture[str],
):
    """точная модель ставится только по имени, а установленная даёт вероятность по умолчанию"""
    assert [m.name for m in models.MODELS.values() if m.default] == ["transformer", "lightgbm"]
    assert chosen() is models.MODERNBERT
    _, out, _ = cli(capsys, "classify", "--json", texts["ai"])
    assert json.loads(out)["name"] == "modernbert"
    _, out, _ = cli(capsys, "classify", "--json", "--model", "transformer", texts["ai"])
    assert json.loads(out)["name"] == "transformer"


def test_models_ordered_by_false_ai():
    """порядок предпочтения — по доле людей, принятых за ИИ на test: чем реже, тем раньше"""
    assert list(models.MODELS) == ["modernbert", "transformer", "mini-frida", "lightgbm"]
    cat = {e["name"]: e["quality"]["human_as_ai"] for e in models.catalog()["models"]}
    rates = [cat[name] for name in models.MODELS]
    assert rates == sorted(rates)
    assert list(cat) == list(models.MODELS)  # models info печатает в том же порядке


def test_mini_frida_is_opt_in_after_transformer(
    transformer_dir: Path,
    mini_frida_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    texts: dict[str, str],
):
    """mini-frida ставится только по имени; рядом с трансформером вероятность даёт трансформер,
    mini-frida выбирается через --model или работает, если трансформера нет"""
    assert not models.MINI_FRIDA.default
    assert chosen() is models.TRANSFORMER
    _, out, _ = cli(capsys, "classify", "--json", "--model", "mini-frida", texts["ai"])
    assert json.loads(out)["name"] == "mini-frida"
    monkeypatch.setenv("AIW_RU_TRANSFORMER_DIR", str(transformer_dir / "нет"))
    monkeypatch.setenv("AIW_RU_MODEL_DIR", str(transformer_dir / "нет"))
    models.load.cache_clear()
    assert chosen() is models.MINI_FRIDA


def test_classify_all_models_side_by_side(
    transformer_dir: Path,
    modernbert_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    texts: dict[str, str],
):
    """--all: вероятности всех установленных моделей рядом; если модели расходятся, это видно сразу"""
    monkeypatch.setenv("AIW_RU_MODEL_DIR", str(transformer_dir / "нет"))
    # Порог ModernBERT почти 1: на том же тексте она голосует «человек», трансформер — «ИИ».
    spec = json.loads((modernbert_dir / "inference.json").read_text(encoding="utf-8"))
    (modernbert_dir / "inference.json").write_text(json.dumps({**spec, "threshold": 0.999999}), encoding="utf-8")
    models.load.cache_clear()
    code, out, _ = cli(capsys, "classify", "--all", "--json", texts["ai"])
    data = json.loads(out)
    assert code == 0
    assert [m["name"] for m in data["models"]] == ["modernbert", "transformer"]
    assert [m["ai"] for m in data["models"]] == [False, True]
    assert data["agree"] is False
    assert all(m["threshold"] in (0.5, 0.999999) for m in data["models"])
    code, out, _ = cli(capsys, "classify", "--all", texts["ai"])
    assert "ModernBERT" in out and "трансформер" in out and "Модели расходятся" in out
    code, _, err = cli(capsys, "classify", "--all", "--model", "transformer", texts["ai"])
    assert code == 2 and "--all" in err


def test_probability_near_threshold_is_marked(
    transformer_dir: Path, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    """вероятность у порога помечена: модель не уверена, число ничего не решает"""
    # Крошечная модель даёт этому тексту 0,6: у порога, но по ту сторону, где «ИИ».
    near, far = tmp_path / "near.md", tmp_path / "far.md"
    near.write_text("типа данный еще", encoding="utf-8")
    far.write_text("является ключевую данный", encoding="utf-8")
    assert 0.5 < models.probability("типа данный еще", model=models.TRANSFORMER) < 0.65
    _, out, _ = cli(capsys, "classify", str(near))
    assert "близко к порогу" in out
    _, out, _ = cli(capsys, "classify", str(far))
    assert "близко к порогу" not in out


def flat(out: str) -> str:
    """Вывод в одну строку: таблицы и заметки переносятся по ширине терминала."""
    return " ".join(out.split())


# Документ в три окна крошечной модели: человек, ИИ, человек.
LONG_DOC = "ну вот короче типа\n\nявляется ключевую данный еще\n\nну вот короче типа"


@pytest.fixture
def long_doc(transformer_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    set_spec(transformer_dir, max_length=6, max_windows=1, long_texts="head")
    # В окне на 4 токена четверть — одно слово; абзацы-предложения здесь режутся целиком.
    monkeypatch.setattr(models, "UNIT", 1.0)
    monkeypatch.setenv("NO_COLOR", "1")
    path = tmp_path / "long.md"
    # newline="": на Windows write_text иначе пишет \r\n, и в файле становится больше знаков.
    path.write_text(LONG_DOC, encoding="utf-8", newline="")
    return str(path)


def test_classify_long_text_by_fragments(long_doc: str, capsys: pytest.CaptureFixture[str]):
    """текст длиннее окна: видно, что прочитала модель, и вероятность по каждому фрагменту"""
    code, out, _ = cli(capsys, "classify", "--json", long_doc)
    data = json.loads(out)
    assert code == 0
    assert data["read"] == {
        "truncated": True,
        "words": 4,
        "totalWords": 12,
        "chars": 18,
        "totalChars": len(LONG_DOC),
        "lines": 1,
        "totalLines": 5,
        "tokens": 4,
        "totalTokens": 12,
        "window": 6,
    }
    f = data["fragments"]
    assert (f["count"], f["total"], f["overlap"], f["aboveThreshold"]) == (3, 3, 0.25, 1)
    assert [(c["line"], c["endLine"], c["words"], c["ai"]) for c in f["items"]] == [
        (1, 1, 4, False),
        (3, 3, 4, True),
        (5, 5, 4, False),
    ]
    assert f["aiLines"] == [[3, 3]] and f["aiWordShare"] == round(4 / 12, 4) and f["seconds"] >= 0
    out = flat(cli(capsys, "classify", long_doc)[1])
    assert "Прочитано" in out and "4 слова из 12, строка 1 из 5: окно модели 6 токенов" in out
    assert "Выше порога 1 фрагмент из 3, в них 33 % слов: строка 3." in out
    assert "Один фрагмент выше порога в длинном тексте — слабый признак" in out and "aiw-ru models guide" in out
    assert "Проверено 3 фрагмента за" in out and "на 25 %" in out
    # без фрагментов и с пределом
    data = json.loads(cli(capsys, "classify", "--json", "--no-fragments", long_doc)[1])
    assert "fragments" not in data and data["read"]["truncated"]
    data = json.loads(cli(capsys, "classify", "--json", "--max-fragments", "1", long_doc)[1])
    assert (data["fragments"]["count"], data["fragments"]["total"]) == (1, 3)
    out = flat(cli(capsys, "classify", "--max-fragments", "1", long_doc)[1])
    assert "проверено 1 из 3 фрагментов, до строки 1; все — --max-fragments 0" in out
    code, _, err = cli(capsys, "classify", "--max-fragments", "много", long_doc)
    assert code == 2 and "--max-fragments" in err


def test_scan_and_all_say_what_the_model_read(long_doc: str, model_dir: Path, capsys: pytest.CaptureFixture[str]):
    """scan, antiplagiat и classify --all тоже говорят, что модель прочитала только начало"""
    data = json.loads(cli(capsys, "scan", "--json", long_doc)[1])
    assert data["classifier"]["read"]["truncated"] and data["classifier"]["read"]["words"] == 4
    # Путь к файлу во временной папке длиннее строки терминала: rich его переносит.
    out = flat(cli(capsys, "scan", long_doc)[1])
    assert "4 слова из 12" in out and "Весь текст по фрагментам: aiw-ru classify" in out and "long.md" in out
    out = flat(cli(capsys, "antiplagiat", long_doc)[1])
    assert "4 слова из 12" in out and "aiw-ru classify" in out
    out = flat(cli(capsys, "classify", "--all", long_doc)[1])
    assert "прочитано 4 слова из 12" in out and "aiw-ru classify --model ИМЯ" in out
    # LightGBM читает весь текст: ни пометки, ни фрагментов
    data = json.loads(cli(capsys, "classify", "--json", "--model", "lightgbm", long_doc)[1])
    assert not data["read"]["truncated"] and "fragments" not in data
    assert "Прочитано" not in cli(capsys, "classify", "--model", "lightgbm", long_doc)[1]


def test_long_check_announces_time(long_doc: str, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch):
    """если проверка по фрагментам будет долгой, stderr сразу говорит, сколько ждать"""
    from aiw_ru import cli as cli_module

    assert cli_module.about(4.2) == "4 с" and cli_module.about(47) == "45 с" and cli_module.about(130) == "2 мин"
    monkeypatch.setattr(cli_module, "ETA_AFTER", 0.0)
    _, _, err = cli(capsys, "classify", long_doc)
    assert err.count("фрагмент") == 1 and "трансформер проверит" in err


def test_cli_unknown_model(capsys: pytest.CaptureFixture[str]):
    code, _, err = cli(capsys, "classify", "--model", "gpt")
    assert code == 2 and "неизвестная --model: gpt" in err


# ── каталог для models info ──

REPORTS = Path(__file__).parent.parent / "docs" / "models"


def test_catalog_matches_reports():
    """каталог в пакете: каждая модель с отчётом в docs/models, качество — из отчёта"""
    cat = models.catalog()
    names = [e["name"] for e in cat["models"]]
    assert set(names) <= set(models.MODELS)
    for model in models.MODELS.values():
        path = REPORTS / f"{model.repo.split('/')[-1]}.json"
        if not path.exists():
            continue
        entry = next(e for e in cat["models"] if e["name"] == model.name)
        report = json.loads(path.read_text(encoding="utf-8"))
        assert entry["quality"]["roc_auc"] == round(report["test"]["roc_auc"], 4)
        assert entry["quality"]["accuracy"] == round(report["test"]["accuracy"], 4)
        assert entry["repo"] == model.repo and entry["default"] == model.default
        assert set(entry.get("measured", {})) <= {"load_ms", "short_ms", "long_ms", "ram_mb"}


def test_cli_models_info(capsys: pytest.CaptureFixture[str]):
    """справка по моделям без скачивания: таблица, одна модель, JSON"""
    code, out, _ = cli(capsys, "models", "info")
    assert code == 0 and "ROC AUC" in out and "Память" in out and "transformer" in out
    code, out, _ = cli(capsys, "models", "info", "lightgbm")
    assert code == 0 and "aiw-ru models install lightgbm" in out
    _, out, _ = cli(capsys, "models", "info", "--json")
    assert {e["name"] for e in json.loads(out)["models"]} >= {"transformer", "lightgbm"}
    code, _, err = cli(capsys, "models", "info", "gpt")
    assert code == 2 and "нет модели gpt" in err


def test_cli_models_guide(capsys: pytest.CaptureFixture[str], root: Path):
    """руководство по классификации: файл каталога скилла с командами через aiw-ru"""
    code, out, _ = cli(capsys, "models", "guide")
    guide = (root / "skills" / "avoid-ai-writing-russian" / "references" / "models.md").read_text(encoding="utf-8")
    assert code == 0 and out.startswith("# Модели: как пользоваться классификацией")
    assert "aiw-ru classify --all /tmp/fragment.md --json" in out and "--project" not in out
    # разделы те же, что в файле скилла
    heads = [line for line in guide.splitlines() if line.startswith("## ")]
    assert heads and [line for line in out.splitlines() if line.startswith("## ")] == heads

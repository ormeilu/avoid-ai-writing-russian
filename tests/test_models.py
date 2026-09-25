"""Необязательные модели: LightGBM и трансформер в ONNX. Загрузка, проверки версий, вероятность, CLI."""

from __future__ import annotations

import json
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


@pytest.fixture
def transformer_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Крошечный классификатор из tests/fixtures/tiny-transformer и словарь на восемь слов."""
    pytest.importorskip("onnxruntime")
    tokenizers = pytest.importorskip("tokenizers")
    tok = tokenizers.Tokenizer(tokenizers.models.WordLevel(VOCAB, unk_token="[UNK]"))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tok.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "model.onnx").write_bytes((FIXTURE / "model.onnx").read_bytes())
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
    (tmp_path / "inference.json").write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv("AIW_RU_TRANSFORMER_DIR", str(tmp_path))
    models.load.cache_clear()
    yield tmp_path
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
    assert models.windows(list(range(10, 20)), 6, 2, 3, 8) == [
        [2, 10, 11, 12, 13, 3],
        [2, 14, 15, 16, 17, 3],
        [2, 18, 19, 3],
    ]
    assert models.windows([], 6, 2, 3, 8) == [[2, 3]]
    assert len(models.windows(list(range(100)), 6, 2, 3, 8)) == 8
    t = models.TRANSFORMER
    ai, human = (
        models.probability("является ключевую данный еще", model=t),
        models.probability("ну вот короче типа", model=t),
    )
    mixed = models.probability("является ключевую данный еще ну вот короче типа", model=t)
    assert mixed == pytest.approx((ai + human) / 2, abs=1e-6)


def test_transformer_format_mismatch(transformer_dir: Path):
    spec = json.loads((transformer_dir / "inference.json").read_text(encoding="utf-8"))
    spec["version"] = models.INFERENCE_VERSION + 1
    (transformer_dir / "inference.json").write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(models.ModelError, match="обновите aiw-ru"):
        models.load(models.TRANSFORMER)


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
    assert [m["name"] for m in data["models"]] == ["transformer", "lightgbm"]
    assert all(m["ready"] and m["error"] is None for m in data["models"])


def test_cli_unknown_model(capsys: pytest.CaptureFixture[str]):
    code, _, err = cli(capsys, "classify", "--model", "gpt")
    assert code == 2 and "неизвестная --model: gpt" in err

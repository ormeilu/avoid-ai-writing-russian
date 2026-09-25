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

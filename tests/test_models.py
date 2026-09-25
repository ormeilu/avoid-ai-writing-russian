"""Необязательная модель LightGBM: загрузка, проверка версии признаков, вероятность."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

lgb = pytest.importorskip("lightgbm")
np = pytest.importorskip("numpy")

from aiw_ru import analyze, models
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

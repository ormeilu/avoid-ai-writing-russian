"""Необязательные модели: LightGBM поверх признаков детектора.

Модели не ставятся вместе с aiw-ru. Нужны зависимости из extra `ml`
(`uv sync --extra ml` или `pip install "aiw-ru[ml]"`) и файлы модели с Hugging
Face (`aiw-ru models install`). Пока их нет, scan и antiplagiat работают как
раньше, а здесь можно узнать, чего не хватает.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import cache
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from aiw_ru.detect import analyze
from aiw_ru.features import CONTEXT, FEATURE_NAMES, FEATURES_VERSION, features
from aiw_ru.types import AnalysisResult

LIGHTGBM_REPO = "toiletsandpaper/russian-ai-text-detector-lightgbm"
FILES = ["model.txt", "features.json", "metrics.json", "README.md"]
DEPENDENCIES = ("lightgbm", "huggingface_hub")
INSTALL_HINT = 'uv sync --extra ml (или pip install "aiw-ru[ml]")'


class ModelError(Exception):
    """Модель нельзя использовать: нет зависимостей, файлов или версии признаков не совпадают."""


@dataclass(frozen=True, slots=True)
class Status:
    repo: str
    # Установлены lightgbm и huggingface_hub.
    dependencies: bool
    # Папка со скачанной моделью в кэше Hugging Face.
    path: Path | None


def missing_dependencies() -> list[str]:
    return [name for name in DEPENDENCIES if find_spec(name) is None]


def local_path(repo: str = LIGHTGBM_REPO) -> Path | None:
    """Скачанная модель в кэше Hugging Face, без обращения к сети.

    AIW_RU_MODEL_DIR указывает на папку с моделью напрямую: так проверяют только
    что обученную модель до выкладки и так работают тесты.
    """
    if override := os.environ.get("AIW_RU_MODEL_DIR"):
        path = Path(override).expanduser()
        return path if (path / "model.txt").exists() else None
    if missing_dependencies():
        return None
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        path = Path(snapshot_download(repo, allow_patterns=FILES, local_files_only=True))
    except LocalEntryNotFoundError:
        return None
    return path if (path / "model.txt").exists() else None


def status(repo: str = LIGHTGBM_REPO) -> Status:
    return Status(repo, not missing_dependencies(), local_path(repo))


def install(repo: str = LIGHTGBM_REPO, revision: str | None = None) -> Path:
    """Скачивает модель с Hugging Face в общий кэш (~/.cache/huggingface)."""
    if missing := missing_dependencies():
        raise ModelError(f"нет пакетов {', '.join(missing)}: {INSTALL_HINT}")
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(repo, allow_patterns=FILES, revision=revision))
    load.cache_clear()
    _check(path)
    return path


def _check(path: Path) -> dict[str, Any]:
    spec = json.loads((path / "features.json").read_text(encoding="utf-8"))
    if spec.get("version") != FEATURES_VERSION or spec.get("names") != list(FEATURE_NAMES):
        raise ModelError(
            f"модель обучена на признаках версии {spec.get('version')}, а aiw-ru считает версию {FEATURES_VERSION}: "
            "обновите aiw-ru или скачайте модель заново (aiw-ru models install)"
        )
    return spec


@cache
def load(repo: str = LIGHTGBM_REPO) -> tuple[Any, dict[str, Any]]:
    """Модель и описание её признаков; ModelError, если чего-то не хватает."""
    if missing := missing_dependencies():
        raise ModelError(f"нет пакетов {', '.join(missing)}: {INSTALL_HINT}")
    path = local_path(repo)
    if path is None:
        raise ModelError("модель не скачана: aiw-ru models install")
    spec = _check(path)
    import lightgbm as lgb

    return lgb.Booster(model_file=str(path / "model.txt")), spec


def available(repo: str = LIGHTGBM_REPO) -> bool:
    """Модель можно использовать прямо сейчас."""
    try:
        load(repo)
    except ModelError:
        return False
    return True


def probability(text: str, result: AnalysisResult | None = None, repo: str = LIGHTGBM_REPO) -> float:
    """Вероятность, что текст написала модель, 0–1.

    `result` берётся, только если детектор работал в режиме признаков (CONTEXT);
    иначе детектор запускается заново.
    """
    booster, _ = load(repo)
    if result is None or result.stats.context_mode != CONTEXT:
        result = analyze(text, CONTEXT)
    return float(booster.predict([features(text, result)])[0])


def threshold(repo: str = LIGHTGBM_REPO) -> float:
    return float(load(repo)[1].get("threshold", 0.5))

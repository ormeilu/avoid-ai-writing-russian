"""Необязательные модели: трансформер в ONNX и LightGBM поверх признаков детектора.

Модели не ставятся вместе с aiw-ru. Нужны зависимости из extra `ml`
(`uv sync --extra ml` или `pip install "aiw-ru[ml]"`) и файлы моделей с Hugging
Face (`aiw-ru models install`). Пока их нет, scan и antiplagiat работают как
раньше, а здесь можно узнать, чего не хватает.

Трансформер точнее, LightGBM легче и быстрее. Если стоят обе, вероятность в scan,
antiplagiat и classify даёт трансформер, если не выбрана другая модель.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Protocol

from aiw_ru.detect import analyze
from aiw_ru.features import CONTEXT, FEATURE_NAMES, FEATURES_VERSION, features
from aiw_ru.types import AnalysisResult

INSTALL_HINT = 'uv sync --extra ml (или pip install "aiw-ru[ml]")'
# Версия inference.json, которую понимает этот код.
INFERENCE_VERSION = 1


class ModelError(Exception):
    """Модель нельзя использовать: нет зависимостей, файлов или версии не совпадают."""


@dataclass(frozen=True, slots=True)
class Model:
    name: str
    # Для вывода: «по модели …».
    title: str
    repo: str
    # Файлы, которые скачиваются с Hugging Face; первый должен быть в папке модели.
    files: tuple[str, ...]
    # Модули Python, без которых модель не загрузить.
    dependencies: tuple[str, ...]
    # Переменная окружения с папкой модели в обход кэша Hugging Face.
    env: str


TRANSFORMER = Model(
    "transformer",
    "трансформер",
    "toiletsandpaper/russian-ai-text-detector-bert",
    ("inference.json", "model.onnx", "tokenizer.json", "metrics.json", "README.md"),
    ("onnxruntime", "tokenizers", "numpy", "huggingface_hub"),
    "AIW_RU_TRANSFORMER_DIR",
)
LIGHTGBM = Model(
    "lightgbm",
    "LightGBM",
    "toiletsandpaper/russian-ai-text-detector-lightgbm",
    ("model.txt", "features.json", "metrics.json", "README.md"),
    ("lightgbm", "huggingface_hub"),
    "AIW_RU_MODEL_DIR",
)
# Порядок — предпочтение: первая установленная модель даёт вероятность по умолчанию.
MODELS: dict[str, Model] = {m.name: m for m in (TRANSFORMER, LIGHTGBM)}
LIGHTGBM_REPO = LIGHTGBM.repo


@dataclass(frozen=True, slots=True)
class Status:
    model: Model
    # Установлены пакеты модели.
    dependencies: bool
    # Папка со скачанной моделью в кэше Hugging Face.
    path: Path | None


class Loaded(Protocol):
    model: Model
    threshold: float

    def probability(self, text: str, result: AnalysisResult | None = None) -> float: ...


def get(name: str) -> Model:
    if name not in MODELS:
        raise ModelError(f"нет модели {name}; есть: {', '.join(MODELS)}")
    return MODELS[name]


def missing_dependencies(model: Model = LIGHTGBM) -> list[str]:
    return [name for name in model.dependencies if find_spec(name) is None]


def local_path(model: Model = LIGHTGBM) -> Path | None:
    """Скачанная модель в кэше Hugging Face, без обращения к сети.

    Переменная окружения модели (AIW_RU_MODEL_DIR для LightGBM, AIW_RU_TRANSFORMER_DIR
    для трансформера) указывает на папку с моделью напрямую: так проверяют только что
    обученную модель до выкладки и так работают тесты.
    """
    if override := os.environ.get(model.env):
        path = Path(override).expanduser()
        return path if (path / model.files[0]).exists() else None
    if find_spec("huggingface_hub") is None:
        return None
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        path = Path(snapshot_download(model.repo, allow_patterns=list(model.files), local_files_only=True))
    except LocalEntryNotFoundError:
        return None
    return path if (path / model.files[0]).exists() else None


def status(model: Model = LIGHTGBM) -> Status:
    return Status(model, not missing_dependencies(model), local_path(model))


def install(model: Model = LIGHTGBM, revision: str | None = None) -> Path:
    """Скачивает модель с Hugging Face в общий кэш (~/.cache/huggingface)."""
    if missing := missing_dependencies(model):
        raise ModelError(f"нет пакетов {', '.join(missing)}: {INSTALL_HINT}")
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(model.repo, allow_patterns=list(model.files), revision=revision))
    load.cache_clear()
    load(model)
    return path


# ── LightGBM ──


def _check_features(path: Path) -> dict[str, Any]:
    spec = json.loads((path / "features.json").read_text(encoding="utf-8"))
    if spec.get("version") != FEATURES_VERSION or spec.get("names") != list(FEATURE_NAMES):
        raise ModelError(
            f"модель обучена на признаках версии {spec.get('version')}, а aiw-ru считает версию {FEATURES_VERSION}: "
            "обновите aiw-ru или скачайте модель заново (aiw-ru models install)"
        )
    return spec


@dataclass(slots=True)
class _LightGBM:
    model: Model
    threshold: float
    booster: Any

    def probability(self, text: str, result: AnalysisResult | None = None) -> float:
        # Признаки считаются в режиме CONTEXT; результат другого режима не годится.
        if result is None or result.stats.context_mode != CONTEXT:
            result = analyze(text, CONTEXT)
        return float(self.booster.predict([features(text, result)])[0])


def _load_lightgbm(model: Model, path: Path) -> _LightGBM:
    spec = _check_features(path)
    try:
        import lightgbm as lgb
    except OSError as e:
        # На macOS колёса lightgbm ищут libomp из Homebrew и без неё не загружаются.
        hint = ": поставьте libomp (brew install libomp)" if sys.platform == "darwin" else ""
        raise ModelError(f"lightgbm не загружается{hint} ({e})") from None
    booster = lgb.Booster(model_file=str(path / "model.txt"))
    return _LightGBM(model, float(spec.get("threshold", 0.5)), booster)


# ── Трансформер ──


def windows(ids: Sequence[int], max_length: int, cls_id: int, sep_id: int, limit: int) -> list[list[int]]:
    """Окна по max_length токенов подряд, без перекрытия; ids — токены текста без [CLS] и [SEP]."""
    width = max_length - 2
    return [[cls_id, *ids[i : i + width], sep_id] for i in range(0, max(len(ids), 1), width)][:limit]


@dataclass(slots=True)
class _Transformer:
    model: Model
    threshold: float
    spec: dict[str, Any]
    session: Any
    tokenizer: Any
    rules: list[tuple[re.Pattern[str], str]]

    def normalize(self, text: str) -> str:
        """Та же нормализация, что при обучении: правила записаны в inference.json."""
        for pattern, replacement in self.rules:
            text = pattern.sub(replacement, text)
        return text.strip()

    def probability(self, text: str, result: AnalysisResult | None = None) -> float:
        """Вероятность по первому окну текста или среднее по окнам, как записано в inference.json."""
        import numpy as np

        s = self.spec
        ids = self.tokenizer.encode(self.normalize(text[: s["max_chars"]]), add_special_tokens=False).ids
        ws = windows(ids, s["max_length"], s["cls_id"], s["sep_id"], s["max_windows"])
        batch = np.full((len(ws), max(len(w) for w in ws)), s["pad_id"], dtype=np.int64)
        mask = np.zeros_like(batch)
        for k, w in enumerate(ws):
            batch[k, : len(w)] = w
            mask[k, : len(w)] = 1
        logits = np.asarray(self.session.run([s["output"]], {"input_ids": batch, "attention_mask": mask})[0])
        z = np.exp(logits - logits.max(axis=1, keepdims=True))
        ai = s["labels"].index("ai")
        return float((z[:, ai] / z.sum(axis=1)).mean())


def _load_transformer(model: Model, path: Path) -> _Transformer:
    spec = json.loads((path / "inference.json").read_text(encoding="utf-8"))
    if spec.get("version", 1) != INFERENCE_VERSION or spec.get("format") != "onnx":
        raise ModelError(
            f"модель в формате {spec.get('format')} версии {spec.get('version', 1)}, а aiw-ru понимает onnx "
            f"версии {INFERENCE_VERSION}: обновите aiw-ru"
        )
    import onnxruntime as ort
    from tokenizers import Tokenizer

    options = ort.SessionOptions()
    options.log_severity_level = 3
    session = ort.InferenceSession(str(path / spec["file"]), options, providers=["CPUExecutionProvider"])
    tokenizer = Tokenizer.from_file(str(path / spec["tokenizer"]))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    rules = [(re.compile(r["pattern"], re.MULTILINE), r["replacement"]) for r in spec.get("normalize", [])]
    return _Transformer(model, float(spec.get("threshold", 0.5)), spec, session, tokenizer, rules)


# ── Общее ──


@cache
def load(model: Model = LIGHTGBM) -> Loaded:
    """Загруженная модель; ModelError, если чего-то не хватает."""
    if missing := missing_dependencies(model):
        raise ModelError(f"нет пакетов {', '.join(missing)}: {INSTALL_HINT}")
    path = local_path(model)
    if path is None:
        raise ModelError(f"модель {model.name} не скачана: aiw-ru models install {model.name}")
    return _load_transformer(model, path) if model is TRANSFORMER else _load_lightgbm(model, path)


def available(model: Model = LIGHTGBM) -> bool:
    """Модель можно использовать прямо сейчас."""
    try:
        load(model)
    except ModelError:
        return False
    return True


def preferred(name: str | None = None) -> Loaded | None:
    """Выбранная модель или первая установленная; None, если не стоит ни одна."""
    if name is not None:
        return load(get(name))
    for model in MODELS.values():
        if available(model):
            return load(model)
    return None


def probability(text: str, result: AnalysisResult | None = None, model: Model = LIGHTGBM) -> float:
    """Вероятность, что текст написала модель, 0–1.

    `result` берётся, только если детектор работал в режиме признаков (CONTEXT);
    иначе LightGBM запускает детектор заново. Трансформеру детектор не нужен.
    """
    return load(model).probability(text, result)


def threshold(model: Model = LIGHTGBM) -> float:
    return load(model).threshold

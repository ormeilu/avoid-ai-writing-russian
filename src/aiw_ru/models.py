"""Необязательные модели: трансформеры в ONNX и LightGBM поверх признаков детектора.

Модели не ставятся вместе с aiw-ru. Нужны зависимости из extra `ml`
(`uv sync --extra ml` или `pip install "aiw-ru[ml]"`) и файлы моделей с Hugging
Face (`aiw-ru models install`). Пока их нет, scan и antiplagiat работают как
раньше, а здесь можно узнать, чего не хватает.

Моделей три. ModernBERT самый точный, но тяжёлый (140 МБ, fp32) и медленный, и
ставится только по имени. Трансформер на rubert-tiny2 (30 МБ, int8) немного уступает
ему в точности и вчетверо быстрее. LightGBM легче всех и заметно менее точен. Если
стоят несколько, вероятность в scan, antiplagiat и classify даёт самая точная из
установленных, если не выбрана другая.
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
    # onnx — трансформер по inference.json, lightgbm — бустинг на признаках детектора.
    kind: str = "onnx"
    # Ставится командой models install без имени.
    default: bool = True
    # Одна фраза для aiw-ru models info: чем модель отличается от остальных.
    summary: str = ""


# model.onnx* — с внешними данными (model.onnx.data), если граф больше 2 ГБ.
_ONNX_FILES = ("inference.json", "model.onnx*", "tokenizer.json", "metrics.json", "README.md")
_ONNX_DEPENDENCIES = ("onnxruntime", "tokenizers", "numpy", "huggingface_hub")

MODERNBERT = Model(
    "modernbert",
    "ModernBERT",
    "toiletsandpaper/russian-ai-text-detector-modernbert",
    _ONNX_FILES,
    _ONNX_DEPENDENCIES,
    "AIW_RU_MODERNBERT_DIR",
    default=False,
    summary="самая точная, но тяжёлая и в несколько раз медленнее: для мощных машин и спорных текстов",
)
MINI_FRIDA = Model(
    "mini-frida",
    "mini-frida",
    "toiletsandpaper/russian-ai-text-detector-mini-frida",
    _ONNX_FILES,
    _ONNX_DEPENDENCIES,
    "AIW_RU_MINI_FRIDA_DIR",
    default=False,
    summary="между ними: ROC AUC выше, чем у трансформера, но при пороге 50 % чаще принимает людей за ИИ; "
    "в 4 раза тяжелее и в 3 раза медленнее",
)
TRANSFORMER = Model(
    "transformer",
    "трансформер",
    "toiletsandpaper/russian-ai-text-detector-bert",
    _ONNX_FILES,
    _ONNX_DEPENDENCIES,
    "AIW_RU_TRANSFORMER_DIR",
    summary="почти так же точна, лёгкая и быстрая: выбор по умолчанию",
)
LIGHTGBM = Model(
    "lightgbm",
    "LightGBM",
    "toiletsandpaper/russian-ai-text-detector-lightgbm",
    ("model.txt", "features.json", "metrics.json", "README.md"),
    ("lightgbm", "huggingface_hub"),
    "AIW_RU_MODEL_DIR",
    kind="lightgbm",
    summary="самая лёгкая, работает и без onnxruntime (Mac на Intel), но заметно менее точна",
)
# Порядок — предпочтение: первая установленная модель даёт вероятность по умолчанию.
MODELS: dict[str, Model] = {m.name: m for m in (MODERNBERT, MINI_FRIDA, TRANSFORMER, LIGHTGBM)}
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


def catalog() -> dict[str, Any]:
    """Замеры моделей из пакета (aiw_ru/data/models.json): качество, скорость, память, размер.

    Файл собирает scripts/models_catalog.py перед выпуском, поэтому посмотреть модели
    можно до того, как что-то скачивать.
    """
    from importlib.resources import files

    return json.loads(files("aiw_ru").joinpath("data", "models.json").read_text(encoding="utf-8"))


def get(name: str) -> Model:
    if name not in MODELS:
        raise ModelError(f"нет модели {name}; есть: {', '.join(MODELS)}")
    return MODELS[name]


def missing_dependencies(model: Model = LIGHTGBM) -> list[str]:
    return [name for name in model.dependencies if find_spec(name) is None]


def local_path(model: Model = LIGHTGBM) -> Path | None:
    """Скачанная модель в кэше Hugging Face, без обращения к сети.

    Переменная окружения модели (`Model.env`: AIW_RU_MODEL_DIR для LightGBM,
    AIW_RU_TRANSFORMER_DIR и соседние для трансформеров) указывает на папку с моделью
    напрямую: так проверяют только что обученную модель до выкладки и так работают тесты.
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


def windows(
    ids: Sequence[int], max_length: int, prefix: Sequence[int], suffix: Sequence[int], limit: int
) -> list[list[int]]:
    """Окна по max_length токенов подряд, без перекрытия; ids — токены текста без служебных.

    prefix и suffix обрамляют каждое окно: [CLS] и [SEP] у BERT, префикс задачи и </s> у T5.
    """
    width = max_length - len(prefix) - len(suffix)
    return [[*prefix, *ids[i : i + width], *suffix] for i in range(0, max(len(ids), 1), width)][:limit]


def _frame(spec: dict[str, Any]) -> tuple[list[int], list[int]]:
    """Служебные токены вокруг окна: prefix_ids и suffix_ids или [cls_id] и [sep_id]."""
    if "prefix_ids" in spec or "suffix_ids" in spec:
        return list(spec.get("prefix_ids", [])), list(spec.get("suffix_ids", []))
    return [spec["cls_id"]], [spec["sep_id"]]


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
        prefix, suffix = _frame(s)
        ws = windows(ids, s["max_length"], prefix, suffix, s["max_windows"])
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
    return _load_transformer(model, path) if model.kind == "onnx" else _load_lightgbm(model, path)


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

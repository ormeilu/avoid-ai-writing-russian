"""Необязательные модели: трансформеры в ONNX и LightGBM поверх признаков детектора.

Модели не ставятся вместе с aiw-ru. Нужны зависимости из extra `ml`
(`uv sync --extra ml` или `pip install "aiw-ru[ml]"`) и файлы моделей с Hugging
Face (`aiw-ru models install`). Пока их нет, scan и antiplagiat работают как
раньше, а здесь можно узнать, чего не хватает.

Моделей четыре: три трансформера (ModernBERT, rubert-tiny2, mini-frida) и LightGBM.
Если стоят несколько, вероятность в scan, antiplagiat и classify даёт первая из
установленных в порядке MODELS, если не выбрана другая.

Трансформер читает одно окно в max_length токенов (около 300 слов). Докуда он дочитал,
говорит coverage(); длинный текст целиком проверяет scan_chunks(): режет его на
фрагменты в одно окно по границам предложений, с перекрытием соседних.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import cache
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Protocol

from aiw_ru.detect import analyze
from aiw_ru.features import CONTEXT, FEATURE_NAMES, FEATURES_VERSION, features
from aiw_ru.text import sentences, words
from aiw_ru.types import AnalysisResult

# onnxruntime 1.30 при импорте запускает телеметрию Microsoft: секунд через десять она шлёт данные,
# а если процесс завершается во время отправки, падает в abort (recursive_mutex lock failed).
# Выключается она только переменной окружения до импорта. Ставим её здесь, при импорте модуля:
# CLI импортирует models при запуске. Явное значение пользователя не трогаем.
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

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
    summary="самая точная: выбор по умолчанию, но тяжелее трансформера и в несколько раз медленнее",
)
MINI_FRIDA = Model(
    "mini-frida",
    "mini-frida",
    "toiletsandpaper/russian-ai-text-detector-mini-frida",
    _ONNX_FILES,
    _ONNX_DEPENDENCIES,
    "AIW_RU_MINI_FRIDA_DIR",
    default=False,
    summary="ROC AUC выше, чем у трансформера, но при пороге 50 % чаще принимает людей за ИИ; "
    "в 4 раза тяжелее и в 3 раза медленнее",
)
TRANSFORMER = Model(
    "transformer",
    "трансформер",
    "toiletsandpaper/russian-ai-text-detector-bert",
    _ONNX_FILES,
    _ONNX_DEPENDENCIES,
    "AIW_RU_TRANSFORMER_DIR",
    default=False,
    summary="почти так же точна, лёгкая и в несколько раз быстрее ModernBERT: для слабой машины",
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
# Порядок — предпочтение: первая установленная модель даёт вероятность по умолчанию. Модели идут
# по доле людей, принятых за ИИ на test (4,3 %, 7,2 %, 9,5 %, 16,5 %), а не по ROC AUC: ошибиться
# в человеке для скилла дороже, чем пропустить ИИ-текст.
MODELS: dict[str, Model] = {m.name: m for m in (MODERNBERT, TRANSFORMER, MINI_FRIDA, LIGHTGBM)}
LIGHTGBM_REPO = LIGHTGBM.repo


@dataclass(frozen=True, slots=True)
class Status:
    model: Model
    # Установлены пакеты модели.
    dependencies: bool
    # Папка со скачанной моделью в кэше Hugging Face.
    path: Path | None


@dataclass(frozen=True, slots=True)
class Coverage:
    """Какую часть текста прочитала модель: у трансформера окно ограничено, LightGBM читает текст целиком."""

    # Знаков исходника до места, где модель остановилась, и всего.
    chars: int
    total_chars: int
    words: int
    total_words: int
    # Последняя прочитанная строка и всего строк.
    lines: int
    total_lines: int
    # Токены текста без служебных, прочитано и всего; у LightGBM None.
    tokens: int | None = None
    total_tokens: int | None = None
    # Окно модели в токенах (max_length); у LightGBM None.
    window: int | None = None

    @property
    def truncated(self) -> bool:
        return self.chars < self.total_chars


def _lines(text: str) -> int:
    return text.rstrip("\n").count("\n") + 1


def _whole(text: str) -> Coverage:
    n, lines = len(words(text)), _lines(text)
    return Coverage(len(text), len(text), n, n, lines, lines)


# (сделано, всего, секунд прошло): после каждой пачки фрагментов.
Progress = Callable[[int, int, float], None]


class Loaded(Protocol):
    model: Model
    threshold: float

    def probability(self, text: str, result: AnalysisResult | None = None) -> float: ...

    def probabilities(self, texts: list[str], progress: Progress | None = None) -> list[float]: ...

    def coverage(self, text: str) -> Coverage: ...


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

    def probabilities(self, texts: list[str], progress: Progress | None = None) -> list[float]:
        return [self.probability(t) for t in texts]

    def coverage(self, text: str) -> Coverage:
        """Признаки считаются по всему тексту."""
        return _whole(text)


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

# Фрагментов в одном вызове ONNX. onnxruntime и так раскладывает окно по ядрам: на M1 пачки
# по 8 не ускоряют трансформер (23 мс на фрагмент против 27), а ModernBERT и mini-frida замедляют.
BATCH = 1


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

    @property
    def width(self) -> int:
        """Токенов текста в одном окне: max_length без служебных токенов."""
        prefix, suffix = _frame(self.spec)
        return self.spec["max_length"] - len(prefix) - len(suffix)

    def ids(self, text: str) -> list[int]:
        return self.tokenizer.encode(self.normalize(text), add_special_tokens=False).ids

    def counts(self, texts: list[str]) -> list[int]:
        """Токены каждого текста после нормализации, без служебных."""
        encoded = self.tokenizer.encode_batch([self.normalize(t) for t in texts], add_special_tokens=False)
        return [len(e.ids) for e in encoded]

    def _run(self, ws: list[list[int]]) -> list[float]:
        """Вероятность ИИ по каждому окну; окна уже обрамлены служебными токенами."""
        import numpy as np

        s = self.spec
        batch = np.full((len(ws), max(len(w) for w in ws)), s["pad_id"], dtype=np.int64)
        mask = np.zeros_like(batch)
        for k, w in enumerate(ws):
            batch[k, : len(w)] = w
            mask[k, : len(w)] = 1
        logits = np.asarray(self.session.run([s["output"]], {"input_ids": batch, "attention_mask": mask})[0])
        z = np.exp(logits - logits.max(axis=1, keepdims=True))
        ai = s["labels"].index("ai")
        return [float(x) for x in z[:, ai] / z.sum(axis=1)]

    def probability(self, text: str, result: AnalysisResult | None = None) -> float:
        """Вероятность по первому окну текста или среднее по окнам, как записано в inference.json."""
        s = self.spec
        prefix, suffix = _frame(s)
        probs = self._run(windows(self.ids(text[: s["max_chars"]]), s["max_length"], prefix, suffix, s["max_windows"]))
        return sum(probs) / len(probs)

    def probabilities(self, texts: list[str], progress: Progress | None = None) -> list[float]:
        """Вероятность по первому окну каждого текста, пачками; для фрагментов, которые влезают в окно."""
        s = self.spec
        prefix, suffix = _frame(s)
        started, out = time.perf_counter(), []
        for k in range(0, len(texts), BATCH):
            ws = [
                windows(self.ids(t[: s["max_chars"]]), s["max_length"], prefix, suffix, 1)[0]
                for t in texts[k : k + BATCH]
            ]
            out += self._run(ws)
            if progress is not None:
                progress(len(out), len(texts), time.perf_counter() - started)
        return out

    def coverage(self, text: str) -> Coverage:
        """Докуда модель читает текст: max_windows окон, но не дальше max_chars знаков."""
        s = self.spec
        budget = self.width * s["max_windows"]
        total = len(self.ids(text))
        limit = min(len(text), s["max_chars"])
        if total <= budget and limit == len(text):
            return replace(_whole(text), tokens=total, total_tokens=total, window=s["max_length"])
        # Самое длинное начало исходника, которое после нормализации укладывается в окна: так место
        # обрыва точное, даже если нормализация меняет длину текста.
        hi = min(limit, budget * 8)
        while hi < limit and len(self.ids(text[:hi])) <= budget:
            hi = min(limit, hi * 2)
        lo = 0
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if len(self.ids(text[:mid])) <= budget:
                lo = mid
            else:
                hi = mid - 1
        # Полслова не показываем: обрыв переносится на последний пробел перед ним.
        if 0 < lo < len(text) and not text[lo].isspace() and not text[lo - 1].isspace():
            space = re.search(r"\s\S*$", text[:lo])
            lo = space.start() if space else lo
        head = text[:lo].rstrip()
        return Coverage(
            len(head),
            len(text),
            len(words(head)),
            len(words(text)),
            _lines(head),
            _lines(text),
            len(self.ids(head)),
            total,
            s["max_length"],
        )

    def _units(self, text: str, cap: int) -> list[tuple[int, int, int]]:
        """Предложения со своими токенами; предложение длиннее cap делится по словам пополам, пока не влезет."""
        sents = sentences(text)
        units: list[tuple[int, int, int]] = []
        for sent, n in zip(sents, self.counts([s.text for s in sents]), strict=True):
            self._split(text, sent.start, sent.end, n, cap, units)
        return units

    def _split(self, text: str, start: int, end: int, n: int, cap: int, out: list[tuple[int, int, int]]) -> None:
        spans = [(m.start() + start, m.end() + start) for m in re.finditer(r"\S+", text[start:end])]
        if n <= cap or len(spans) < 2:
            out.append((start, end, n))
            return
        mid = len(spans) // 2
        halves = [(spans[0][0], spans[mid - 1][1]), (spans[mid][0], spans[-1][1])]
        for (a, b), k in zip(halves, self.counts([text[a:b] for a, b in halves]), strict=True):
            self._split(text, a, b, k, cap, out)

    def spans(self, text: str, overlap: float, unit: float) -> list[tuple[int, int]]:
        """Фрагменты в одно окно по границам предложений; следующий начинается с хвоста предыдущего,
        где примерно overlap окна. Предложение длиннее unit окна делится по словам: иначе фрагмент
        перед ним обрывался бы задолго до конца окна. Последний фрагмент добирается назад до полного окна."""
        units = self._units(text, max(1, round(self.width * unit)))
        if not units:
            return [(0, len(text))]
        keep = round(self.width * overlap)
        out: list[tuple[int, int]] = []
        i = 0
        while True:
            j, used = i, 0
            while j < len(units) and (j == i or used + units[j][2] <= self.width):
                used += units[j][2]
                j += 1
            if j == len(units):
                floor = out[-1][0] + 1 if out else 0
                while i > floor and used + units[i - 1][2] <= self.width:
                    i -= 1
                    used += units[i][2]
                out.append((i, j))
                break
            out.append((i, j))
            k, back = j, 0
            while k - 1 > i and back + units[k - 1][2] <= keep:
                k -= 1
                back += units[k][2]
            i = k
        return [(units[a][0], units[b - 1][1]) for a, b in out]


def _no_limits(tokenizer: Any) -> None:
    """Снимает обрезку и дополнение: окна по max_length режет _Transformer.
    В tokenizers 0.x для этого есть методы no_*, в 1.x их нет, а свойствам присваивают None."""
    if hasattr(tokenizer, "no_truncation"):
        tokenizer.no_truncation()
        tokenizer.no_padding()
    else:
        tokenizer.truncation = None
        tokenizer.padding = None


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
    _no_limits(tokenizer)
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


# ── Длинный текст по фрагментам ──

# Доля окна, на которую соседние фрагменты заходят друг на друга: граница между человеческой и
# ИИ-частью документа не попадёт только на стык двух фрагментов.
OVERLAP = 0.25
# Предложение длиннее этой доли окна делится по словам, поэтому фрагмент заполнен хотя бы на 1 − UNIT.
UNIT = 0.25


@dataclass(frozen=True, slots=True)
class Chunk:
    """Фрагмент длинного текста в одно окно модели и его вероятность."""

    # Знаки исходника [start, end) и строки, с первой по последнюю.
    start: int
    end: int
    line: int
    end_line: int
    words: int
    probability: float


@dataclass(frozen=True, slots=True)
class ChunkScan:
    chunks: list[Chunk]
    # Фрагментов в тексте всего; проверено len(chunks), если стоял предел.
    total: int
    seconds: float
    overlap: float


def chunk_spans(
    loaded: Loaded, text: str, overlap: float | None = None, unit: float | None = None
) -> list[tuple[int, int]]:
    """Фрагменты текста для модели; LightGBM читает текст целиком, для него фрагмент один.
    Без overlap и unit берутся OVERLAP и UNIT."""
    if not isinstance(loaded, _Transformer):
        return [(0, len(text))]
    return loaded.spans(text, OVERLAP if overlap is None else overlap, UNIT if unit is None else unit)


def scan_chunks(
    loaded: Loaded,
    text: str,
    overlap: float | None = None,
    limit: int | None = None,
    progress: Progress | None = None,
) -> ChunkScan:
    """Вероятность по каждому фрагменту длинного текста; limit — проверить только первые фрагменты."""
    overlap = OVERLAP if overlap is None else overlap
    spans = chunk_spans(loaded, text, overlap)
    checked = spans[:limit] if limit else spans
    started = time.perf_counter()
    probs = loaded.probabilities([text[a:b] for a, b in checked], progress)
    seconds = time.perf_counter() - started
    chunks = [
        Chunk(a, b, text.count("\n", 0, a) + 1, text.count("\n", 0, b) + 1, len(words(text[a:b])), p)
        for (a, b), p in zip(checked, probs, strict=True)
    ]
    return ChunkScan(chunks, len(spans), seconds, overlap)

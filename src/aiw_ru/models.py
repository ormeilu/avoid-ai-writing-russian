"""Необязательные модели: трансформеры в ONNX и LightGBM поверх признаков детектора.

Модели не ставятся вместе с aiw-ru. Нужны зависимости из extra `ml`
(`uv sync --extra ml` или `pip install "aiw-ru[ml]"`) и файлы моделей с Hugging
Face (`aiw-ru models install`). Пока их нет, scan и antiplagiat работают как
раньше, а здесь можно узнать, чего не хватает.

Моделей четыре: три трансформера (ModernBERT, rubert-tiny2, mini-frida) и LightGBM.
Если стоят несколько, вероятность в scan, antiplagiat и classify даёт первая из
установленных в порядке MODELS, если не выбрана другая.

Трансформер читает текст одним окном: read_length токенов, если оно есть в inference.json
(у ModernBERT 8192, 4–5 тысяч слов), иначе max_length (512, около 300 слов). Докуда он
дочитал, говорит coverage(). Где в тексте ИИ, показывает scan_chunks(): режет текст на
фрагменты по max_length токенов по границам предложений, с перекрытием соседних.

Модели обучены на русском тексте. YAML-шапку, блоки кода и служебные части статьи (список
литературы, ключевые слова, английскую аннотацию, сведения об авторах) им не показывают, а у
фрагмента, где кириллицы меньше половины букв, вероятности нет. Абзацы не на русском модели тоже
не показывают. Это делают visible(), foreign_lines(), is_russian()
и scan_chunks(); низкоуровневые probability() и probabilities() читают текст как есть.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import cache, lru_cache
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Protocol

from aiw_ru.detect import analyze
from aiw_ru.features import CONTEXT, FEATURE_NAMES, FEATURES_VERSION, features
from aiw_ru.text import Sentence, blank, cyrillic_share, letter_share, model_text, sentences, words
from aiw_ru.types import AnalysisResult

# onnxruntime 1.30 при импорте запускает телеметрию Microsoft: секунд через десять она шлёт данные,
# а если процесс завершается во время отправки, падает в abort (recursive_mutex lock failed).
# Выключается она только переменной окружения до импорта. Ставим её здесь, при импорте модуля:
# CLI импортирует models при запуске. Явное значение пользователя не трогаем.
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

INSTALL_HINT = 'uv sync --extra ml (или pip install "aiw-ru[ml]")'
# Версия inference.json, которую понимает этот код.
INFERENCE_VERSION = 1

# Модели обучены на русской части LLMTrace. На другом языке вероятность ничего не значит:
# ModernBERT на английской аннотации почти всегда уверенно говорит «ИИ». Поэтому текст и фрагмент,
# где кириллицы меньше половины букв, модели не оценивают. Считаются кириллические и латинские
# буквы: цифры, знаки и другие письменности в долю не входят, а текст совсем без букв не оценивается.
RUSSIAN_SHARE = 0.5
# Абзац с кириллицей меньше RUSSIAN_SHARE и хотя бы столькими буквами модели не показывают: английская
# аннотация или цитата посреди русского текста иначе попадает в русский фрагмент и тянет его вверх.
# Короткие строки вроде «Python 3.12» или «Results» остаются: русский текст вокруг их перевешивает.
FOREIGN_LETTERS = 20

# Серия пропусков от 64 знаков. Так выглядят вырезанные YAML-шапка, код и служебные части: их
# заменяют пробелами, чтобы номера строк и смещения остались как в исходнике. Правила нормализации
# с `^[ ]*` на строке из тысяч пробелов работают за квадрат времени (20 000 пробелов — около 3 с),
# и разбиение на предложения после точки тоже: серию пропусков обходят.
LONG_BLANK = re.compile(r"\s{64,}")


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


# Карточка модели: меняется без переобучения и на ответы не влияет.
CARD_FILES = ("metrics.json", "README.md")
# model.onnx* — с внешними данными (model.onnx.data), если граф больше 2 ГБ.
_ONNX_FILES = ("inference.json", "model.onnx*", "tokenizer.json", *CARD_FILES)
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
    ("model.txt", "features.json", *CARD_FILES),
    ("lightgbm", "huggingface_hub"),
    "AIW_RU_MODEL_DIR",
    kind="lightgbm",
    summary="самая лёгкая, работает и без onnxruntime (Mac на Intel), но заметно менее точна",
)
# Порядок — предпочтение: первая установленная модель даёт вероятность по умолчанию. Модели идут
# по доле людей, принятых за ИИ на test (3,4 %, 7,2 %, 9,5 %, 16,5 %), а не по ROC AUC: ошибиться
# в человеке для скилла дороже, чем пропустить ИИ-текст.
MODELS: dict[str, Model] = {model.name: model for model in (MODERNBERT, TRANSFORMER, MINI_FRIDA, LIGHTGBM)}
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
    word_count, lines = len(words(text)), _lines(text)
    return Coverage(len(text), len(text), word_count, word_count, lines, lines)


# (сделано, всего, секунд прошло): после каждой пачки фрагментов.
Progress = Callable[[int, int, float], None]


class Loaded(Protocol):
    model: Model
    threshold: float

    def probability(self, text: str, result: AnalysisResult | None = None) -> float: ...

    def probabilities(self, texts: list[str], progress: Progress | None = None) -> list[float]: ...

    def coverage(self, text: str) -> Coverage: ...


@cache
def catalog() -> dict[str, Any]:
    """Замеры моделей из пакета (aiw_ru/data/models.json): качество, скорость, память, размер.

    Файл собирает scripts/models_catalog.py перед выпуском, поэтому посмотреть модели
    можно до того, как что-то скачивать.
    """
    from importlib.resources import files

    return json.loads(files("aiw_ru").joinpath("data", "models.json").read_text(encoding="utf-8"))


def pinned(model: Model = LIGHTGBM) -> dict[str, Any]:
    """Запись модели в каталоге: ревизия на Hugging Face и хэши файлов, с которыми замерено качество.

    blobs — имена файлов в кэше Hugging Face: sha256 у файлов в LFS, хэш git у остальных.
    """
    return next((entry for entry in catalog()["models"] if entry["name"] == model.name), {})


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

    # Сначала ревизия из каталога: её качает models install, а refs/main на неё не переводит.
    # Потом скачанная последней (refs/main): так ставили модели aiw-ru до 2.2.1.
    for revision in dict.fromkeys((pinned(model).get("revision"), None)):
        try:
            path = Path(
                snapshot_download(
                    model.repo, allow_patterns=list(model.files), revision=revision, local_files_only=True
                )
            )
        except LocalEntryNotFoundError:
            continue
        if (path / model.files[0]).exists():
            return path
    return None


def outdated(model: Model = LIGHTGBM) -> bool | None:
    """Скачанная модель не та, с которой в каталоге замерено качество: её обновит models install.

    Файл в снапшоте кэша Hugging Face — ссылка на blobs/<хэш>, хэши сверяются с каталогом. Ревизии
    не сравниваются: после модели в репозиторий идут коммиты с одной карточкой. None — сверить
    не с чем: модели нет, её папка задана переменной окружения или кэш без ссылок (Windows без
    режима разработчика).
    """
    blobs: dict[str, str] = pinned(model).get("blobs") or {}
    if not blobs or os.environ.get(model.env):
        return None
    path = local_path(model)
    if path is None:
        return None
    for name, etag in blobs.items():
        file = path / name
        if not file.exists():
            return True
        if not file.is_symlink():
            return None
        if Path(os.readlink(file)).name != etag:
            return True
    return False


def status(model: Model = LIGHTGBM) -> Status:
    return Status(model, not missing_dependencies(model), local_path(model))


def install(model: Model = LIGHTGBM, revision: str | None = None) -> Path:
    """Скачивает модель с Hugging Face в общий кэш (~/.cache/huggingface).

    Без revision — ревизию из каталога: ту, с которой замерено качество в models info и руководстве.
    """
    if missing := missing_dependencies(model):
        raise ModelError(f"нет пакетов {', '.join(missing)}: {INSTALL_HINT}")
    from huggingface_hub import snapshot_download

    revision = revision or pinned(model).get("revision")
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
        return [self.probability(text) for text in texts]

    def coverage(self, text: str) -> Coverage:
        """Признаки считаются по всему тексту."""
        return _whole(text)


def _load_lightgbm(model: Model, path: Path) -> _LightGBM:
    spec = _check_features(path)
    try:
        import lightgbm as lgb
    except OSError as error:
        # На macOS колёса lightgbm ищут libomp из Homebrew и без неё не загружаются.
        hint = ": поставьте libomp (brew install libomp)" if sys.platform == "darwin" else ""
        raise ModelError(f"lightgbm не загружается{hint} ({error})") from None
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
        """Та же нормализация, что при обучении: правила записаны в inference.json.

        Серия пропусков от 64 знаков (LONG_BLANK) до правил сводится к одному пробелу или переводу
        строки. Результат тот же: правила про начало строки пробелов не считают, а последнее
        правило всё равно делает из любых пропусков один пробел."""
        text = LONG_BLANK.sub(lambda match: "\n" if "\n" in match.group() else " ", text)
        for pattern, replacement in self.rules:
            text = pattern.sub(replacement, text)
        return text.strip()

    @property
    def width(self) -> int:
        """Токенов текста во фрагменте: max_length без служебных токенов."""
        prefix, suffix = _frame(self.spec)
        return self.spec["max_length"] - len(prefix) - len(suffix)

    @property
    def read_length(self) -> int:
        """Окно для текста целиком. read_length в inference.json длиннее max_length у модели, которая
        читает длинный текст сразу, а фрагменты режет короче; aiw-ru до 2.2 знает только max_length."""
        return self.spec.get("read_length", self.spec["max_length"])

    def ids(self, text: str) -> list[int]:
        return self.tokenizer.encode(self.normalize(text), add_special_tokens=False).ids

    def counts(self, texts: list[str]) -> list[int]:
        """Токены каждого текста после нормализации, без служебных."""
        encoded = self.tokenizer.encode_batch([self.normalize(text) for text in texts], add_special_tokens=False)
        return [len(encoding.ids) for encoding in encoded]

    def _run(self, framed: list[list[int]]) -> list[float]:
        """Вероятность ИИ по каждому окну; окна уже обрамлены служебными токенами."""
        import numpy as np

        spec = self.spec
        batch = np.full((len(framed), max(len(window) for window in framed)), spec["pad_id"], dtype=np.int64)
        mask = np.zeros_like(batch)
        for k, window in enumerate(framed):
            batch[k, : len(window)] = window
            mask[k, : len(window)] = 1
        logits = np.asarray(self.session.run([spec["output"]], {"input_ids": batch, "attention_mask": mask})[0])
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        ai_index = spec["labels"].index("ai")
        return [float(probability) for probability in exp_logits[:, ai_index] / exp_logits.sum(axis=1)]

    def probability(self, text: str, result: AnalysisResult | None = None) -> float:
        """Вероятность по первому окну текста или среднее по окнам, как записано в inference.json."""
        spec = self.spec
        prefix, suffix = _frame(spec)
        probs = self._run(
            windows(self.ids(text[: spec["max_chars"]]), self.read_length, prefix, suffix, spec["max_windows"])
        )
        return sum(probs) / len(probs)

    def probabilities(self, texts: list[str], progress: Progress | None = None) -> list[float]:
        """Вероятность по первому окну каждого текста, пачками; для фрагментов, которые влезают в окно."""
        spec = self.spec
        prefix, suffix = _frame(spec)
        started, probs = time.perf_counter(), []
        for k in range(0, len(texts), BATCH):
            framed = [
                windows(self.ids(text[: spec["max_chars"]]), spec["max_length"], prefix, suffix, 1)[0]
                for text in texts[k : k + BATCH]
            ]
            probs += self._run(framed)
            if progress is not None:
                progress(len(probs), len(texts), time.perf_counter() - started)
        return probs

    def coverage(self, text: str) -> Coverage:
        """Докуда модель читает текст: max_windows окон по read_length, но не дальше max_chars знаков."""
        spec = self.spec
        prefix, suffix = _frame(spec)
        budget = (self.read_length - len(prefix) - len(suffix)) * spec["max_windows"]
        total = len(self.ids(text))
        limit = min(len(text), spec["max_chars"])
        if total <= budget and limit == len(text):
            return replace(_whole(text), tokens=total, total_tokens=total, window=self.read_length)
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
            self.read_length,
        )

    def _units(self, text: str, cap: int) -> list[tuple[int, int, int]]:
        """Предложения со своими токенами; предложение длиннее cap делится по словам пополам, пока не влезет."""
        text_sentences = _sentences(text)
        units: list[tuple[int, int, int]] = []
        for sentence, tokens in zip(
            text_sentences, self.counts([sentence.text for sentence in text_sentences]), strict=True
        ):
            self._split(text, sentence.start, sentence.end, tokens, cap, units)
        return units

    def _split(self, text: str, start: int, end: int, tokens: int, cap: int, units: list[tuple[int, int, int]]) -> None:
        spans = [(match.start() + start, match.end() + start) for match in re.finditer(r"\S+", text[start:end])]
        if tokens <= cap or len(spans) < 2:
            units.append((start, end, tokens))
            return
        mid = len(spans) // 2
        halves = [(spans[0][0], spans[mid - 1][1]), (spans[mid][0], spans[-1][1])]
        for (half_start, half_end), count in zip(
            halves, self.counts([text[half_start:half_end] for half_start, half_end in halves]), strict=True
        ):
            self._split(text, half_start, half_end, count, cap, units)

    def spans(self, text: str, overlap: float, unit: float) -> list[tuple[int, int]]:
        """Фрагменты в одно окно по границам предложений; следующий начинается с хвоста предыдущего,
        где примерно overlap окна. Предложение длиннее unit окна делится по словам: иначе фрагмент
        перед ним обрывался бы задолго до конца окна. Последний фрагмент добирается назад до полного окна."""
        units = self._units(text, max(1, round(self.width * unit)))
        if not units:
            return [(0, len(text))]
        keep = round(self.width * overlap)
        unit_spans: list[tuple[int, int]] = []
        i = 0
        while True:
            j, used = i, 0
            while j < len(units) and (j == i or used + units[j][2] <= self.width):
                used += units[j][2]
                j += 1
            if j == len(units):
                floor = unit_spans[-1][0] + 1 if unit_spans else 0
                while i > floor and used + units[i - 1][2] <= self.width:
                    i -= 1
                    used += units[i][2]
                unit_spans.append((i, j))
                break
            unit_spans.append((i, j))
            k, back = j, 0
            while k - 1 > i and back + units[k - 1][2] <= keep:
                k -= 1
                back += units[k][2]
            i = k
        return [(units[first][0], units[after_last - 1][1]) for first, after_last in unit_spans]


def _sentences(text: str) -> list[Sentence]:
    """Предложения по кускам между длинными пропусками: sentences() на серии пробелов после точки
    работает за квадрат времени, а серия пропусков предложение всё равно разделяет."""
    found: list[Sentence] = []
    pos = 0
    for match in LONG_BLANK.finditer(text):
        found += sentences(text[pos : match.start()], pos)
        pos = match.end()
    return found + sentences(text[pos:], pos)


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
    rules = [(re.compile(rule["pattern"], re.MULTILINE), rule["replacement"]) for rule in spec.get("normalize", [])]
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


def _foreign(text: str) -> list[tuple[int, int]]:
    """Абзацы (строки подряд без пустой между ними), где кириллицы меньше RUSSIAN_SHARE букв, а букв
    не меньше FOREIGN_LETTERS; отрезки [начало, конец) в text."""
    foreign: list[tuple[int, int]] = []
    paragraph: tuple[int, int] | None = None
    pos = 0

    def close() -> None:
        if paragraph is None:
            return
        cyrillic, latin = letter_share(text[paragraph[0] : paragraph[1]])
        if cyrillic + latin >= FOREIGN_LETTERS and cyrillic / (cyrillic + latin) < RUSSIAN_SHARE:
            foreign.append(paragraph)

    for line in text.split("\n"):
        end = pos + len(line)
        if line.strip():
            paragraph = (paragraph[0] if paragraph else pos, end)
        else:
            close()
            paragraph = None
        pos = end + 1
    close()
    return foreign


@lru_cache(maxsize=2)
def _clean(text: str) -> tuple[str, tuple[tuple[int, int], ...]]:
    base = model_text(text)
    foreign = _foreign(base)
    return blank(base, foreign), tuple(foreign)


def visible(text: str) -> str:
    """Текст, который видят модели: YAML-шапка, блоки кода, служебные части статьи и абзацы не на русском
    заменены пробелами.

    Длина и переводы строк сохраняются, поэтому смещения и номера строк те же, что в исходнике
    (aiw_ru.text.model_text). Результат кэшируется: signal и scan_chunks получают один и тот же
    файл и чистят его один раз."""
    return _clean(text)[0]


def foreign_lines(text: str) -> list[tuple[int, int]]:
    """Строки исходника (с единицы, первая и последняя) абзацев не на русском, которые модели не показывают."""
    return [(text.count("\n", 0, start) + 1, text.count("\n", 0, end) + 1) for start, end in _clean(text)[1]]


def russian(share: float | None) -> bool:
    """Доли кириллицы (aiw_ru.text.cyrillic_share) хватает для моделей: не меньше RUSSIAN_SHARE.
    None, то есть текст без букв, не годится."""
    return share is not None and share >= RUSSIAN_SHARE


def is_russian(text: str) -> bool:
    return russian(cyrillic_share(text))


@dataclass(frozen=True, slots=True)
class Chunk:
    """Фрагмент длинного текста в одно окно модели и его вероятность."""

    # Знаки исходника [start, end) и строки, с первой по последнюю.
    start: int
    end: int
    line: int
    end_line: int
    words: int
    # None у фрагмента не на русском: модель его не оценивала.
    probability: float | None

    @property
    def skipped(self) -> bool:
        return self.probability is None

    def above(self, threshold: float) -> bool:
        """Вероятность не ниже порога; у фрагмента без вероятности нет."""
        return self.probability is not None and self.probability >= threshold


@dataclass(frozen=True, slots=True)
class ChunkScan:
    chunks: list[Chunk]
    # Фрагментов в тексте всего; просмотрено len(chunks), если стоял предел.
    total: int
    seconds: float
    overlap: float
    # Текст для моделей (visible), по которому нарезаны фрагменты; start и end указывают и в него.
    text: str = ""
    # Слов в этом тексте без слов, которые есть только во фрагментах не на русском: на столько
    # слов делится доля ИИ-слов. Слова из неоценённой за пределом --max-fragments части остаются.
    words: int = 0

    @property
    def rated(self) -> list[Chunk]:
        """Фрагменты, которые оценила модель."""
        return [chunk for chunk in self.chunks if not chunk.skipped]

    @property
    def skipped(self) -> int:
        """Фрагментов не на русском: модель их не оценивала."""
        return sum(chunk.skipped for chunk in self.chunks)


def chunk_spans(
    loaded: Loaded, text: str, overlap: float | None = None, unit: float | None = None
) -> list[tuple[int, int]]:
    """Фрагменты текста для модели; LightGBM читает текст целиком, для него фрагмент один.
    Без overlap и unit берутся OVERLAP и UNIT."""
    if not isinstance(loaded, _Transformer):
        return [(0, len(text))]
    return loaded.spans(text, OVERLAP if overlap is None else overlap, UNIT if unit is None else unit)


def needs_fragments(loaded: Loaded, coverage: Coverage) -> bool:
    """Текст длиннее одного фрагмента: classify проверяет его по фрагментам, даже если модель прочитала его целиком."""
    return isinstance(loaded, _Transformer) and (coverage.total_tokens or 0) > loaded.width


def _merged(spans: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Отрезки по возрастанию, слитые там, где они пересекаются или касаются."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _outside(spans: list[tuple[int, int]], cover: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Части отрезков spans (слитых), не закрытые отрезками cover (слитыми)."""
    uncovered: list[tuple[int, int]] = []
    for start, end in spans:
        for cover_start, cover_end in cover:
            if cover_end <= start or cover_start >= end:
                continue
            if cover_start > start:
                uncovered.append((start, cover_start))
            start = max(start, cover_end)
            if start >= end:
                break
        if start < end:
            uncovered.append((start, end))
    return uncovered


def scan_chunks(
    loaded: Loaded,
    text: str,
    overlap: float | None = None,
    limit: int | None = None,
    progress: Progress | None = None,
) -> ChunkScan:
    """Вероятность по каждому фрагменту длинного текста; limit — проверить только первые фрагменты.

    text — исходник файла. Фрагменты режутся по visible(text): шапка, код и служебные части в них не
    попадают, а start, end и номера строк остаются по исходнику. Фрагмент, где кириллицы меньше
    RUSSIAN_SHARE букв, модель не читает: его probability равна None.
    """
    overlap = OVERLAP if overlap is None else overlap
    clean = visible(text)
    spans = chunk_spans(loaded, clean, overlap)
    checked = spans[:limit] if limit else spans
    pieces = [clean[start:end] for start, end in checked]
    in_russian = [is_russian(piece) for piece in pieces]
    started = time.perf_counter()
    probs = iter(
        loaded.probabilities(
            [piece for piece, piece_is_russian in zip(pieces, in_russian, strict=True) if piece_is_russian], progress
        )
    )
    seconds = time.perf_counter() - started
    chunks = [
        Chunk(
            start,
            end,
            clean.count("\n", 0, start) + 1,
            clean.count("\n", 0, end) + 1,
            len(words(piece)),
            next(probs) if piece_is_russian else None,
        )
        for (start, end), piece, piece_is_russian in zip(checked, pieces, in_russian, strict=True)
    ]
    rated = _merged([(chunk.start, chunk.end) for chunk in chunks if not chunk.skipped])
    foreign = _outside(_merged([(chunk.start, chunk.end) for chunk in chunks if chunk.skipped]), rated)
    word_count = len(words(clean)) - sum(len(words(clean[start:end])) for start, end in foreign)
    return ChunkScan(chunks, len(spans), seconds, overlap, clean, word_count)

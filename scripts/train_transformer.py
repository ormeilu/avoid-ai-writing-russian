#!/usr/bin/env python3
"""Дообучает маленький русский энкодер отличать тексты людей от текстов моделей и готовит его к работе на CPU.

    uv run --group train --group transformer scripts/train_transformer.py pilot --base ИМЯ [--limit N] [--raw]
    uv run --group train --group transformer scripts/train_transformer.py train [--base ИМЯ] [--epochs N] [--lr X]
                                                                              [--batch N] [--max-length N]
                                                                              [--device auto|cuda|mps|cpu]
    uv run --group train --group transformer scripts/train_transformer.py export ПАПКА [--json export.json]
    uv run --group train --group transformer scripts/train_transformer.py evaluate ПАПКА
    uv run --group train --group transformer scripts/train_transformer.py report ПАПКА [--repo ИМЯ]
                                                                              [--why ТЕКСТ] [--finalist ПАПКА]

Учится на train-части русского LLMTrace, лучший шаг выбирает по ROC AUC на
valid, итоговые цифры считает на test. Все три части сначала скачивает
`uv run --group train scripts/llmtrace.py fetch --set classification --split <часть>`.

pilot и train учат модель на GPU: CUDA в Colab (запуск описан в
scripts/colab_transformer.py) или MPS на Mac. Без GPU скрипт останавливается,
если явно не задан --device cpu. pilot берёт выборку train и не трогает test.
export переводит лучший шаг в ONNX (fp32 и int8), сверяет ответы с PyTorch и
меряет задержку на CPU. evaluate прогоняет ONNX на CPU по valid и test,
считает те же разрезы, что scripts/train.py для LightGBM, меряет размер
установки зависимостей вывода и пишет metrics.json и inference.json. report
собирает папку hub/ для Hugging Face с карточкой README.md, а после полного
обучения ещё и отчёт docs/models/russian-ai-text-detector-bert.md с .json.

Перед токенизацией текст проходит normalize(): в LLMTrace оформление
(переводы строк, markdown, вид тире и кавычек, ё) выдаёт, откуда взят
человеческий текст, а не кто его написал.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, Protocol

# Телеметрия onnxruntime роняет процесс на выходе. aiw_ru.models выключает её при импорте,
# но onnxruntime здесь импортируется раньше него.
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

import numpy as np
import onnxruntime as ort
import sklearn
import tokenizers
import torch
import transformers
from huggingface_hub import HfApi, ModelCard, ModelCardData, hf_hub_download
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.errors import HfHubHTTPError
from sklearn.metrics import log_loss
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from transformers.utils import logging as hf_logging

import aiw_ru
from aiw_ru.models import MODELS
from aiw_ru.text import plural
from fast_modernbert import FastModernBert
from hub import (
    DATASET,
    GENERATORS,
    GITHUB,
    LLMTRACE_BIBTEX,
    TAGS,
    TASK,
    _bucket,
    _classes_table,
    count,
    eval_results,
    hub_math,
    md_table,
    num,
    pct,
)
from hub import REPO as LIGHTGBM_REPO
from hub import REPORT_METRICS as LIGHTGBM_METRICS
from llmtrace import REVISIONS, data_dir, detect, load, table
from llmtrace import num as comma_num
from llmtrace import pct as comma_pct
from train import Part, auc, breakdowns, classes
from train import report as print_classes

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE = "cointegrated/rubert-tiny2"
LICENSES = {
    "cointegrated/rubert-tiny2": "mit",
    "sergeyzh/rubert-mini-frida": "mit",
    "deepvk/RuModernBERT-small": "apache-2.0",
}
LABELS = ("human", "ai")
THRESHOLD = 0.5
SEED = 1
LOG_EVERY = 200
# Сколько знаков текста на токен нормализовать при обучении: 512 токенов — это 2–3 тысячи
# знаков, запас нужен текстам с длинными пробелами и разметкой. Если после нормализации
# обрезанный текст короче max_length токенов, он нормализуется целиком.
CHARS_PER_TOKEN = 24
# Сколько окон по max_length токенов смотреть в длинном тексте при выводе.
MAX_WINDOWS = 8
# Фрагменты classify не длиннее этого: на окне в 8192 токена ИИ-вставка тонет в человеческом тексте
# (docs/long-documents.md). Модель с окном длиннее читает им текст целиком (read_length в inference.json),
# а aiw-ru до 2.2 знает только max_length и работает с ней как с моделью на 512 токенов.
FRAGMENT_LENGTH = 512
LATENCY_LENGTHS = (128, 512)
PARITY_TEXTS = 300
# Версия inference.json: её проверяет aiw_ru.models перед выводом.
INFERENCE_VERSION = 1


# ─── Нормализация ───────────────────────────────────────────────────────


def _codes(*codes: int) -> str:
    """Символы для класса re в виде обратная косая + uXXXX: в исходниках только видимые знаки."""
    return "".join(f"{chr(92)}u{code:04x}" for code in codes)


_INVISIBLE = _codes(0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0x2060, 0xFEFF, 0x00AD)
_SPACES = _codes(0x00A0, *range(0x2000, 0x200B), 0x202F, 0x205F, 0x3000)
_BULLETS = _codes(0x2022, 0x00B7, 0x25AA, 0x25CF, 0x25E6, 0x2023, 0x2013, 0x2014)
_DOUBLE_QUOTES = _codes(0x00AB, 0x00BB, 0x201E, 0x201C, 0x201D, 0x201F, 0x2033)
_SINGLE_QUOTES = _codes(0x2018, 0x2019, 0x201A, 0x201B, 0x2032)
_DASHES = _codes(*range(0x2010, 0x2016), 0x2212)

# Правила применяются по порядку; шаблоны — re с флагом MULTILINE. Каждое правило —
# (шаблон, замена, зачем). Тот же список ложится в inference.json рядом с моделью,
# чтобы вывод нормализовал текст ровно так же, как обучение.
NORMALIZE_RULES: tuple[tuple[str, str, str], ...] = (
    (f"[{_INVISIBLE}]", "", "невидимые символы и мягкий перенос"),
    (f"[{_SPACES}\\t\\r\\f\\v]", " ", "неразрывные и прочие пробелы"),
    (r"^[ ]*(?:```|~~~).*$", "", "ограждения блоков кода"),
    (r"^[ ]*(?:[-*_=][ ]*){3,}$", "", "горизонтальные линии и подчёркивания заголовков"),
    (r"^[ ]*\|?(?:[ ]*:?-+:?[ ]*\|)+[ ]*(?::?-+:?)?[ ]*$", "", "строка-разделитель таблицы"),
    (r"^[ ]*#{1,6}[ ]+", "", "решётки заголовков"),
    (r"^[ ]*(?:>[ ]?)+", "", "цитаты markdown"),
    (f"^[ ]*(?:[-*+{_BULLETS}]|\\d{{1,3}}[.)])[ ]+", "", "маркеры и номера списков, тире в начале строки"),
    (r"!\[([^\]\n]*)\]\([^)\n]*\)", r"\1", "картинки: остаётся подпись"),
    (r"\[([^\]\n]+)\]\([^)\n]*\)", r"\1", "ссылки: остаётся текст ссылки"),
    (r"</?[A-Za-z][A-Za-z0-9]*(?:\s[^<>\n]*)?/?>", " ", "теги HTML"),
    (r"\*+|~~|`+", "", "звёздочки, зачёркивание, обратные кавычки"),
    (r"(?<!\w)_+|_+(?!\w)", "", "подчёркивания разметки на краях слов"),
    (r"\|", " ", "столбцы таблиц"),
    (f"[{_DOUBLE_QUOTES}]", '"', "двойные кавычки всех видов"),
    (f"[{_SINGLE_QUOTES}]", "'", "апострофы и одинарные кавычки"),
    (f"[{_DASHES}]", "-", "тире, минус и типографские дефисы"),
    (_codes(0x2026), "...", "многоточие одним знаком"),
    ("ё", "е", "ё"),
    ("Ё", "Е", "Ё"),
    (r"\s+", " ", "пробелы и переводы строк"),
)
_NORMALIZE = tuple((re.compile(p, re.MULTILINE), r) for p, r, _ in NORMALIZE_RULES)


def normalize(text: str) -> str:
    """Убирает из текста оформление, которое в корпусе говорит об источнике, а не об авторе."""
    for pattern, replacement in _NORMALIZE:
        text = pattern.sub(replacement, text)
    return text.strip()


def normalize_key() -> str:
    return hashlib.sha1(repr(NORMALIZE_RULES).encode()).hexdigest()[:8]


# ─── Токенизатор и модель ───────────────────────────────────────────────


def load_tokenizer(path: str | Path) -> Any:
    tok = AutoTokenizer.from_pretrained(path)
    if tok is None:
        raise SystemExit(f"{path}: не удалось загрузить токенизатор")
    return tok


def load_classifier(path: str | Path, sliding_window: int = 0) -> Any:
    """Модель с головой на два класса: 0 — человек, 1 — ИИ.

    sliding_window задаёт полуширину локального окна ModernBERT. В config.json у
    deepvk/RuModernBERT-small есть и local_attention: 128, и sliding_window: 128;
    transformers 5 читает второй как полуширину, и окно выходит ±128, а исходный
    ModernBERT с local_attention: 128 смотрел на ±64.
    """
    extra = {"sliding_window": sliding_window} if sliding_window else {}
    return AutoModelForSequenceClassification.from_pretrained(
        path,
        num_labels=2,
        id2label=dict(enumerate(LABELS)),
        label2id={name: k for k, name in enumerate(LABELS)},
        **extra,
    )


def special_ids(tok: Any) -> tuple[int, int, int]:
    """[CLS], [SEP] и [PAD]. Токенизатор должен ставить [CLS] в начало и [SEP] в конец: на этом держится обрезка."""
    ids = tok("проверка")["input_ids"]
    if ids[0] != tok.cls_token_id or ids[-1] != tok.sep_token_id or tok.pad_token_id is None:
        raise SystemExit(f"{tok.name_or_path}: ожидались [CLS] … [SEP] по краям и [PAD], получено {ids}")
    return int(tok.cls_token_id), int(tok.sep_token_id), int(tok.pad_token_id)


def tokenizer_key(tok: Any) -> str:
    return hashlib.sha1(json.dumps(tok.get_vocab(), sort_keys=True).encode()).hexdigest()[:8]


def inference_tokenizer(path: Path) -> tokenizers.Tokenizer:
    """tokenizer.json для вывода без transformers; обрезку и паддинг делает predict_text."""
    tok = tokenizers.Tokenizer.from_file(str(path))
    tok.no_truncation()
    tok.no_padding()
    return tok


# ─── Данные ─────────────────────────────────────────────────────────────


@dataclass
class Encoded:
    """Токенизированная часть корпуса: все токены подряд и границы текстов."""

    ids: np.ndarray  # int32, токены всех текстов подряд, уже обрезанные до max_length
    offsets: np.ndarray  # int64, число текстов + 1
    truncated: np.ndarray  # bool: текст длиннее max_length токенов
    y: np.ndarray  # int8: 1 — ИИ
    genre: np.ndarray
    prompt: np.ndarray
    generator: np.ndarray

    def __len__(self) -> int:
        return len(self.y)

    @property
    def lengths(self) -> np.ndarray:
        return np.diff(self.offsets)

    def seq(self, k: int) -> np.ndarray:
        return self.ids[self.offsets[k] : self.offsets[k + 1]]

    def subset(self, indices: np.ndarray) -> Encoded:
        sequences = [self.seq(int(k)) for k in indices]
        return Encoded(
            ids=np.concatenate(sequences) if sequences else np.zeros(0, dtype=np.int32),
            offsets=np.concatenate([[0], np.cumsum([len(sequence) for sequence in sequences])]).astype(np.int64),
            truncated=self.truncated[indices],
            y=self.y[indices],
            genre=self.genre[indices],
            prompt=self.prompt[indices],
            generator=self.generator[indices],
        )


def truncate(ids: list[int], max_length: int) -> list[int]:
    """Начало текста: первые max_length - 1 токенов и завершающий [SEP]."""
    return ids if len(ids) <= max_length else ids[: max_length - 1] + ids[-1:]


def encode_texts(texts: Sequence[str], tok: Any, max_length: int, raw: bool) -> tuple[list[list[int]], list[bool]]:
    """Токены текстов с [CLS] и [SEP], обрезанные до max_length, и признак обрезки."""
    prep: Callable[[str], str] = str if raw else normalize
    cap = max_length * CHARS_PER_TOKEN
    enc = tok([prep(source_text[:cap]) for source_text in texts], add_special_tokens=True, truncation=False)[
        "input_ids"
    ]
    encoded, truncated = [], []
    for text, ids in zip(texts, enc, strict=True):
        if len(ids) <= max_length and len(text) > cap:
            ids = tok(prep(text), add_special_tokens=True, truncation=False)["input_ids"]
        truncated.append(len(ids) > max_length)
        encoded.append(truncate(ids, max_length))
    return encoded, truncated


def encode_split(
    data: Path, split: str, limit: int, tok: Any, max_length: int, raw: bool, cache_dir: Path, chunk: int = 4096
) -> Encoded:
    """Токенизирует часть корпуса; результат кэшируется в cache_dir."""
    key = f"{tokenizer_key(tok)}-{max_length}-{'raw' if raw else normalize_key()}"
    cache = cache_dir / f"tokens-{split}{f'-{limit}' if limit else ''}-{key}.npz"
    if cache.exists():
        cached = np.load(cache)
        return Encoded(**{field_name: cached[field_name] for field_name in Encoded.__dataclass_fields__})
    started = time.perf_counter()
    lines = load(data, "classification", split, limit)
    parts: list[np.ndarray] = []
    lengths: list[int] = []
    truncated: list[bool] = []
    meta: list[tuple[int, str, str, str]] = []
    for i in range(0, len(lines), chunk):
        rows = [json.loads(line) for line in lines[i : i + chunk]]
        seqs, chunk_truncated = encode_texts([row["text"] for row in rows], tok, max_length, raw)
        parts += [np.asarray(sequence, dtype=np.int32) for sequence in seqs]
        lengths += [len(sequence) for sequence in seqs]
        truncated += chunk_truncated
        meta += [
            (
                int(row["label"] == "ai"),
                row["data_type"],
                row["prompt_type"] or "",
                row["model"] if row["label"] == "ai" else "",
            )
            for row in rows
        ]
        print(f"\r  {split}: {min(i + chunk, len(lines))}/{len(lines)}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    enc = Encoded(
        ids=np.concatenate(parts) if parts else np.zeros(0, dtype=np.int32),
        offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
        truncated=np.array(truncated, dtype=bool),
        y=np.array([entry[0] for entry in meta], dtype=np.int8),
        genre=np.array([entry[1] for entry in meta]),
        prompt=np.array([entry[2] for entry in meta]),
        generator=np.array([entry[3] for entry in meta]),
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **{field_name: getattr(enc, field_name) for field_name in Encoded.__dataclass_fields__})
    print(
        f"Токены {split}: {len(enc)} текстов за {time.perf_counter() - started:.0f} с, "
        f"обрезано {comma_pct(float(enc.truncated.mean()))}",
        file=sys.stderr,
        flush=True,
    )
    return enc


def cut(order: np.ndarray, lengths: np.ndarray, size: int, tokens: int) -> list[np.ndarray]:
    """Режет тексты, отсортированные по длине, на пачки до size текстов и до tokens токенов (ширина × тексты)."""
    if not tokens:
        return [order[i : i + size] for i in range(0, len(order), size)]
    pieces, start = [], 0
    for i in range(1, len(order) + 1):
        # Ширина пачки — длина последнего, самого длинного текста.
        if i == len(order) or i - start == size or (i + 1 - start) * int(lengths[order[i]]) > tokens:
            pieces.append(order[start:i])
            start = i
    return pieces


def batches(lengths: np.ndarray, size: int, rng: np.random.Generator | None, tokens: int = 0) -> list[np.ndarray]:
    """Пачки из текстов близкой длины, чтобы меньше паддинга. С rng порядок пачек случайный.

    tokens ограничивает пачку по токенам: с окном 8192 пачка из длинных текстов иначе не влезает в память.
    """
    total = len(lengths)
    if rng is None:
        return cut(np.argsort(lengths, kind="stable"), lengths, size, tokens)
    pieces: list[np.ndarray] = []
    perm = rng.permutation(total)
    pool = size * 64
    for i in range(0, total, pool):
        chunk = perm[i : i + pool]
        pieces += cut(chunk[np.argsort(lengths[chunk], kind="stable")], lengths, size, tokens)
    return [pieces[k] for k in rng.permutation(len(pieces))]


def collate(enc: Encoded, indices: np.ndarray, pad_id: int, multiple: int) -> tuple[np.ndarray, np.ndarray]:
    """input_ids и attention_mask пачки; ширина округляется вверх до кратной multiple."""
    lens = enc.lengths[indices]
    width = int(-(-int(lens.max()) // multiple) * multiple)
    ids = np.full((len(indices), width), pad_id, dtype=np.int64)
    mask = np.zeros((len(indices), width), dtype=np.int64)
    for row, k in enumerate(indices):
        sequence = enc.seq(int(k))
        ids[row, : len(sequence)] = sequence
        mask[row, : len(sequence)] = 1
    return ids, mask


# ─── Метрики ────────────────────────────────────────────────────────────


def quick(labels: np.ndarray, probs: np.ndarray) -> dict[str, float | None]:
    """Метрики для выбора шага: ROC AUC, accuracy при пороге 0,5 и logloss."""
    return {
        "roc_auc": auc(labels, probs),
        "accuracy": float((labels == (probs >= THRESHOLD)).mean()),
        "logloss": float(log_loss(labels, np.clip(probs, 1e-7, 1 - 1e-7), labels=[0, 1])),
    }


def fmt(value: float | None, digits: int = 3) -> str:
    return comma_num(value, digits) if value is not None else "—"


# ─── Обучение ───────────────────────────────────────────────────────────


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)


def pick_device(name: str) -> torch.device:
    """GPU обязателен: на CPU обучение идёт сутками. CPU — только если попросили явно."""
    if name == "cpu":
        return torch.device("cpu")
    if name in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if name in ("auto", "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    raise SystemExit(
        f"Нет GPU для --device {name}: torch.cuda.is_available() = {torch.cuda.is_available()}, "
        f"torch.backends.mps.is_available() = {torch.backends.mps.is_available()}. "
        "Запустите обучение в Colab (scripts/colab_transformer.py) или явно укажите --device cpu."
    )


def pick_precision(device: torch.device, name: str) -> tuple[str, str]:
    """Точность вычислений и почему выбрана такая."""
    if name != "auto":
        return name, "задана ключом --precision"
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported(including_emulation=False):
            return "bf16", "GPU умеет bf16: диапазон как у fp32, масштабировать потери не нужно"
        return "fp16", "GPU без bf16 (T4): fp16 с масштабированием потерь (GradScaler)"
    if device.type == "mps":
        return "fp32", "на M1 autocast в fp16 и bf16 медленнее fp32: 28 против 32 текстов в секунду"
    return "fp32", "CPU"


def gpu_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return f"Apple {platform.machine()} (MPS)"
    return f"CPU {platform.machine()}"


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def autocast(device: torch.device, precision: str) -> torch.autocast:
    if precision == "fp32" or device.type == "cpu":
        return torch.autocast(device.type, enabled=False)
    return torch.autocast(device.type, dtype=torch.float16 if precision == "fp16" else torch.bfloat16)


def pad_multiple(device: torch.device) -> int:
    # На MPS каждая новая форма тензора — новая компиляция ядер, поэтому шаг крупнее.
    return 64 if device.type == "mps" else 8


@torch.inference_mode()
def predict_torch(
    model: Any, enc: Encoded, device: torch.device, precision: str, size: int, pad_id: int, tokens: int = 0
) -> np.ndarray:
    """Вероятность класса «ИИ» для каждого текста."""
    model.eval()
    probs = np.zeros(len(enc), dtype=np.float32)
    for indices in batches(enc.lengths, size, None, tokens):
        ids, mask = collate(enc, indices, pad_id, pad_multiple(device))
        with autocast(device, precision):
            logits = model(
                input_ids=torch.from_numpy(ids).to(device), attention_mask=torch.from_numpy(mask).to(device)
            ).logits
        probs[indices] = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
    return probs


def param_groups(model: Any, weight_decay: float) -> list[dict]:
    """Без затухания весов для смещений и нормировок."""
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (no_decay if parameter.ndim < 2 or "norm" in name.lower() else decay).append(parameter)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


@dataclass
class Config:
    base: str
    max_length: int
    epochs: float
    lr: float
    batch: int
    warmup: float
    weight_decay: float
    patience: int
    evals_per_epoch: int
    eval_batch: int
    device: str
    precision: str
    raw: bool
    batch_tokens: int = 0
    sliding_window: int = 0
    seed: int = SEED

    @property
    def eval_tokens(self) -> int:
        # Без градиентов пачка при проверке может быть вдвое больше.
        return 2 * self.batch_tokens


def fit(cfg: Config, train: Encoded, valid: Encoded, out_dir: Path, tok: Any) -> dict:
    """Обучает модель; после каждой проверки на valid сохраняет лучший шаг в out/model."""
    device = pick_device(cfg.device)
    precision, why = pick_precision(device, cfg.precision)
    print(f"Устройство: {device.type} ({gpu_name(device)}), точность {precision}: {why}", flush=True)
    seed_everything(cfg.seed)
    _, _, pad_id = special_ids(tok)
    model: Any = load_classifier(cfg.base, cfg.sliding_window).to(device)
    if cfg.sliding_window:
        print(f"Локальное окно: ±{model.config.sliding_window} токенов", flush=True)
    rng = np.random.default_rng(cfg.seed)
    per_epoch = len(batches(train.lengths, cfg.batch, None, cfg.batch_tokens))
    total = max(1, round(per_epoch * cfg.epochs))
    eval_every = max(1, per_epoch // max(1, cfg.evals_per_epoch))
    opt = torch.optim.AdamW(param_groups(model, cfg.weight_decay), lr=cfg.lr, fused=device.type == "cuda")
    sched = get_linear_schedule_with_warmup(opt, round(total * cfg.warmup), total)
    scaler = torch.amp.GradScaler(device.type, enabled=precision == "fp16")

    curve: list[dict] = []
    best_auc, best_point = -1.0, {}
    stale = step = seen = 0
    loss_sum, loss_n = 0.0, 0
    started = last = time.perf_counter()
    stop = False
    print(
        f"Обучение {cfg.base}: {len(train)} текстов, {total} шагов по {cfg.batch}"
        + (f" и не больше {cfg.batch_tokens} токенов" if cfg.batch_tokens else "")
        + ", "
        f"проверка на valid ({len(valid)}) каждые {eval_every} шагов",
        flush=True,
    )
    while step < total and not stop:
        for indices in batches(train.lengths, cfg.batch, rng, cfg.batch_tokens):
            model.train()
            ids, mask = collate(train, indices, pad_id, pad_multiple(device))
            with autocast(device, precision):
                loss = model(
                    input_ids=torch.from_numpy(ids).to(device),
                    attention_mask=torch.from_numpy(mask).to(device),
                    labels=torch.from_numpy(train.y[indices].astype(np.int64)).to(device),
                ).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            seen += len(indices)
            loss_sum += float(loss.detach())
            loss_n += 1
            if step % LOG_EVERY == 0:
                now = time.perf_counter()
                speed = seen / (now - started)
                progress = {
                    "step": step,
                    "total": total,
                    "epoch": round(step / per_epoch, 3),
                    "loss": loss_sum / loss_n,
                    "lr": sched.get_last_lr()[0],
                    "texts_per_second": speed,
                    "eta_minutes": (total - step) * seen / step / speed / 60,
                    "best_valid_roc_auc": best_auc,
                }
                print(
                    f"шаг {step}/{total}, эпоха {progress['epoch']:.2f}: loss {progress['loss']:.4f}, "
                    f"{speed:.0f} текстов/с, осталось {progress['eta_minutes']:.0f} мин "
                    f"({now - last:.0f} с на {LOG_EVERY} шагов)",
                    flush=True,
                )
                (out_dir / "progress.json").write_text(json.dumps(progress) + "\n", encoding="utf-8")
                last = now
            if step % eval_every == 0 or step == total:
                sync(device)
                eval_started = time.perf_counter()
                probs = predict_torch(model, valid, device, precision, cfg.eval_batch, pad_id, cfg.eval_tokens)
                point = {
                    "step": step,
                    "epoch": round(step / per_epoch, 3),
                    "train_loss": loss_sum / max(loss_n, 1),
                    **quick(valid.y, probs),
                    "seconds": time.perf_counter() - started,
                }
                curve.append(point)
                loss_sum, loss_n = 0.0, 0
                score = point["roc_auc"] if point["roc_auc"] is not None else 0.0
                better = score > best_auc
                print(
                    f"valid на шаге {step}: ROC AUC {fmt(point['roc_auc'], 4)}, accuracy {fmt(point['accuracy'])}, "
                    f"logloss {point['logloss']:.4f} ({time.perf_counter() - eval_started:.0f} с)"
                    + (", лучший" if better else ""),
                    flush=True,
                )
                if better:
                    best_auc, best_point, stale = score, point, 0
                    model.save_pretrained(out_dir / "model")
                    tok.save_pretrained(out_dir / "model")
                    np.save(out_dir / "valid-probs.npy", probs)
                else:
                    stale += 1
                    if cfg.patience and stale >= cfg.patience:
                        print(f"Ранняя остановка: {stale} проверки подряд без роста ROC AUC", flush=True)
                        stop = True
                (out_dir / "curve.json").write_text(json.dumps(curve, indent=1) + "\n", encoding="utf-8")
            if step >= total or stop:
                break
    sync(device)
    elapsed = time.perf_counter() - started
    return {
        "device": device.type,
        "gpu": gpu_name(device),
        "precision": precision,
        "precision_reason": why,
        "steps": step,
        "steps_total": total,
        "steps_per_epoch": per_epoch,
        "eval_every": eval_every,
        "best_step": best_point["step"],
        "best_epoch": best_point["epoch"],
        "train_seconds": elapsed,
        "texts_per_second": seen / elapsed,
        "early_stopped": stop,
        "curve": curve,
        "best_valid": {k: best_point[k] for k in ("roc_auc", "accuracy", "logloss")},
    }


def environment(device: str = "", gpu: str = "") -> dict[str, str]:
    """Версии, от которых зависит результат."""
    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "onnxruntime": ort.__version__,
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "platform": f"{platform.system()} {platform.release()}, {platform.machine()}, {plural(os.cpu_count() or 1, 'ядро', 'ядра', 'ядер')}",
    }
    if torch.version.cuda:
        env["cuda"] = str(torch.version.cuda)
    if device:
        env["device"] = device
    if gpu:
        env["gpu"] = gpu
    return env


def git_commit() -> str:
    """Короткий хэш HEAD и пометка о правках; в Colab папка приезжает архивом, хэш передаёт запуск."""
    if os.environ.get("AIW_RU_GIT_COMMIT"):
        return os.environ["AIW_RU_GIT_COMMIT"]
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
        )
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=False)
    except OSError:
        return "неизвестен"
    if not sha.stdout.strip():
        return "неизвестен"
    return sha.stdout.strip() + ("+правки" if dirty.stdout.strip() else "")


def base_revision(base: str) -> str:
    """Ревизия базы на Hugging Face; без сети (HF_HUB_OFFLINE=1) — единственный снимок в локальном кэше."""
    try:
        return HfApi().model_info(base).sha or ""
    except (HfHubHTTPError, OSError):
        snaps = Path(HF_HUB_CACHE) / f"models--{base.replace('/', '--')}" / "snapshots"
        found = [snapshot.name for snapshot in snaps.iterdir()] if snaps.is_dir() else []
        return found[0] if len(found) == 1 else ""


def short_name(base: str) -> str:
    return base.split("/")[-1].lower()


def cmd_fit(args: argparse.Namespace, pilot: bool) -> None:
    cfg = Config(
        base=args.base,
        max_length=args.max_length,
        epochs=args.epochs,
        lr=args.lr,
        batch=args.batch,
        warmup=args.warmup,
        weight_decay=args.weight_decay,
        patience=args.patience,
        evals_per_epoch=args.evals_per_epoch,
        eval_batch=args.eval_batch,
        device=args.device,
        precision=args.precision,
        raw=args.raw,
        batch_tokens=args.batch_tokens,
        sliding_window=args.sliding_window,
        seed=args.seed,
    )
    hf_logging.set_verbosity_error()
    pick_device(cfg.device)  # без GPU остановиться до токенизации, а не после
    tok = load_tokenizer(cfg.base)
    special_ids(tok)
    cache = args.out_root / "cache"
    limit = args.limit
    eval_limit = args.eval_limit if args.eval_limit is not None else limit // 2
    name = f"{short_name(cfg.base)}{'-raw' if cfg.raw else ''}"
    out_dir: Path = args.out or args.out_root / ("pilot" if pilot else "runs") / (f"{name}-{limit}" if limit else name)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Папка запуска: {out_dir}", flush=True)
    train = encode_split(args.data, "train", limit, tok, cfg.max_length, cfg.raw, cache)
    valid = encode_split(args.data, "valid", eval_limit, tok, cfg.max_length, cfg.raw, cache)
    summary = fit(cfg, train, valid, out_dir, tok)
    model: Any = AutoModelForSequenceClassification.from_pretrained(out_dir / "model")
    run: dict[str, Any] = {
        "created": datetime.now(UTC).strftime("%Y-%m-%d"),
        "kind": "pilot" if pilot else "train",
        "git_commit": git_commit(),
        "environment": environment(summary["device"], summary["gpu"]),
        "dataset": {
            "repo": "iitolstykh/LLMTrace_classification",
            "revision": REVISIONS["classification"],
            "language": "ru",
            "limit": limit,
            "eval_limit": eval_limit,
            "train": len(train),
            "valid": len(valid),
        },
        "params": {
            "base": cfg.base,
            "base_revision": base_revision(cfg.base),
            "base_license": LICENSES.get(cfg.base, ""),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "max_length": cfg.max_length,
            "normalize": not cfg.raw,
            "normalize_rules": normalize_key(),
            "epochs": cfg.epochs,
            "lr": cfg.lr,
            "batch": cfg.batch,
            "batch_tokens": cfg.batch_tokens,
            # Окно, с которым модель училась, и ключ --sliding-window, если окно задавалось руками.
            "sliding_window": getattr(model.config, "sliding_window", None),
            "sliding_window_arg": cfg.sliding_window,
            "warmup": cfg.warmup,
            "weight_decay": cfg.weight_decay,
            "patience": cfg.patience,
            "evals_per_epoch": cfg.evals_per_epoch,
            "seed": cfg.seed,
            "optimizer": "AdamW",
            "scheduler": "linear with warmup",
            "grad_clip": 1.0,
            "truncated_share": {"train": float(train.truncated.mean()), "valid": float(valid.truncated.mean())},
            **{k: v for k, v in summary.items() if k not in ("curve", "best_valid")},
        },
        "valid": summary["best_valid"],
        "curve": summary["curve"],
    }
    if not pilot:
        test = encode_split(args.data, "test", eval_limit, tok, cfg.max_length, cfg.raw, cache)
        device = pick_device(cfg.device)
        started = time.perf_counter()
        _, _, pad_id = special_ids(tok)
        probs = predict_torch(
            model.to(device), test, device, summary["precision"], cfg.eval_batch, pad_id, cfg.eval_tokens
        )
        np.save(out_dir / "test-probs.npy", probs)
        run["dataset"]["test"] = len(test)
        run["params"]["truncated_share"]["test"] = float(test.truncated.mean())
        run["torch_test"] = {**quick(test.y, probs), "seconds": time.perf_counter() - started}
        print(f"test, PyTorch {summary['precision']}: {run['torch_test']}", flush=True)
    (out_dir / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Готово: {out_dir / 'run.json'}", flush=True)


# ─── ONNX ───────────────────────────────────────────────────────────────


def export_onnx(model_dir: Path, path: Path) -> None:
    """PyTorch → ONNX с переменной длиной пачки и текста; веса больше 2 ГБ — во внешнем файле model.onnx.data.

    ModernBERT уходит в ONNX через FastModernBert: без матриц n × n в локальных слоях
    окно в 8192 токена на M1 считается за секунды, а не за минуту.
    """
    model = load_torch_classifier(model_dir)
    texts = ["Короткий пример.", "Пример подлиннее, чтобы в пачке были тексты разной длины."]
    sample = load_tokenizer(model_dir)(texts, padding=True)
    max_seq = int(model.config.max_position_embeddings)
    big = sum(parameter.numel() for parameter in model.parameters()) * 4 > 1.8e9
    batch = torch.export.Dim("batch", min=1, max=1024)
    sequence_dim = torch.export.Dim("sequence", min=2, max=max_seq)
    if model.config.model_type == "modernbert":
        model = FastModernBert(model).eval()
    torch.onnx.export(
        model,
        (),
        path,
        kwargs={
            "input_ids": torch.tensor(sample["input_ids"]),
            "attention_mask": torch.tensor(sample["attention_mask"]),
        },
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_shapes={"input_ids": {0: batch, 1: sequence_dim}, "attention_mask": {0: batch, 1: sequence_dim}},
        dynamo=True,
        external_data=big,
        optimize=True,
    )


def quantize_int8(src: Path, dst: Path) -> None:
    """Динамическое квантование весов в int8: матричные умножения и таблица эмбеддингов."""
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    # Экспорт через torch.export записывает формы промежуточных тензоров, с которыми
    # не соглашается вывод форм в onnx; без них квантователь выводит формы заново.
    model = onnx.load(src)
    del model.graph.value_info[:]
    bare = dst.with_suffix(".bare.onnx")
    big = onnx_size(src) > 1.8e9
    onnx.save(model, bare, save_as_external_data=big, location=bare.name + ".data")
    del model
    try:
        # Масштаб на каждый столбец весов и 7 бит вместо 8: на valid ROC AUC как у fp32,
        # и нет переполнения на старых x86 без VNNI.
        quantize_dynamic(
            bare,
            dst,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gather"],
            per_channel=True,
            reduce_range=True,
        )
    finally:
        bare.unlink(missing_ok=True)
        bare.with_name(bare.name + ".data").unlink(missing_ok=True)


def onnx_size(path: Path) -> float:
    """Байты ONNX вместе с внешним файлом весов, если он есть."""
    data = path.with_name(path.name + ".data")
    return path.stat().st_size + (data.stat().st_size if data.exists() else 0)


def session(path: Path, threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


class Session(Protocol):
    """То, что нужно выводу от onnxruntime.InferenceSession."""

    def run(self, output_names: Any, input_feed: Any, /) -> Any: ...


def softmax_ai(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return (exp_logits[:, 1] / exp_logits.sum(axis=1)).astype(np.float32)


def run_logits(sess: Session, ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.asarray(sess.run(["logits"], {"input_ids": ids, "attention_mask": mask})[0])


def predict_onnx(sess: Session, enc: Encoded, pad_id: int, size: int = 32) -> np.ndarray:
    """Вероятности пачками текстов близкой длины.

    Динамический int8 считает масштаб активаций по всей пачке, и ответ зависит от соседей
    по ней; aiw-ru подаёт тексты по одному, поэтому int8 оценивается с size=1.
    """
    probs = np.zeros(len(enc), dtype=np.float32)
    for indices in batches(enc.lengths, size, None):
        ids, mask = collate(enc, indices, pad_id, 1)
        probs[indices] = softmax_ai(run_logits(sess, ids, mask))
    return probs


def timed(function: Callable[[], object], warmup: int = 5, runs: int = 30) -> float:
    """Медиана времени вызова в миллисекундах; для вызовов дольше секунды прогонов меньше."""
    started = time.perf_counter()
    function()
    if time.perf_counter() - started > 1:
        warmup, runs = 1, 5
    for _ in range(warmup - 1):
        function()
    times = []
    for _ in range(runs):
        started = time.perf_counter()
        function()
        times.append(time.perf_counter() - started)
    return float(np.median(times) * 1000)


def run_torch(model: Any, ids: torch.Tensor, mask: torch.Tensor) -> None:
    with torch.inference_mode():
        model(input_ids=ids, attention_mask=mask)


def latency_inputs(enc: Encoded, length: int, suffix: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Настоящий текст ровно на length токенов: самый длинный из части, обрезанный со служебным хвостом."""
    sequence = enc.seq(int(np.argmax(enc.lengths)))
    if len(sequence) > length:
        sequence = np.concatenate([sequence[: length - len(suffix)], suffix]) if suffix else sequence[:length]
    ids = sequence.astype(np.int64)[None, :]
    return ids, np.ones_like(ids)


def latency_lengths(max_length: int) -> tuple[int, ...]:
    """Длины для замера задержки: 128 и 512 токенов у всех, у длинного окна ещё 2048 и всё окно."""
    return tuple(sorted({*LATENCY_LENGTHS, *(length for length in (2048, max_length) if 512 < length <= max_length)}))


def measure_latency(
    model: Any, onnx_files: dict[str, Path], enc: Encoded, suffix: Sequence[int], lengths: Sequence[int]
) -> list[dict]:
    """Задержка на один текст: ONNX Runtime и PyTorch на CPU, один поток и все ядра."""
    rows: list[dict] = []
    for length in lengths:
        ids, mask = latency_inputs(enc, length, suffix)
        for threads in sorted({1, os.cpu_count() or 1}):
            for variant, path in onnx_files.items():
                latency_ms = timed(partial(run_logits, session(path, threads), ids, mask))
                rows.append(
                    {"runtime": f"onnxruntime {variant}", "tokens": ids.shape[1], "threads": threads, "ms": latency_ms}
                )
            torch.set_num_threads(threads)
            latency_ms = timed(partial(run_torch, model, torch.from_numpy(ids), torch.from_numpy(mask)))
            rows.append({"runtime": "torch fp32", "tokens": ids.shape[1], "threads": threads, "ms": latency_ms})
    torch.set_num_threads(os.cpu_count() or 1)
    return rows


def cmd_export(args: argparse.Namespace) -> None:
    """ONNX fp32 и int8, сверка с PyTorch и выводом по спецификации, размер файлов и задержка на CPU."""
    run_dir: Path = args.folder
    model_dir = run_dir / "model"
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    hf_logging.set_verbosity_error()
    onnx_dir = run_dir / "onnx"
    onnx_dir.mkdir(exist_ok=True)
    fp32, int8 = onnx_dir / "model.onnx", onnx_dir / "model_int8.onnx"
    started = time.perf_counter()
    export_onnx(model_dir, fp32)
    quantize_int8(fp32, int8)
    print(f"ONNX: {fp32} и {int8} за {time.perf_counter() - started:.0f} с", flush=True)

    wrap = model_wrap(model_dir)
    _, suffix = spec_wrap(wrap)
    pad_id = wrap["pad_id"]
    params = run["params"]
    eval_limit = run["dataset"]["eval_limit"]
    valid = eval_split(args.data, "valid", eval_limit, model_dir, params, args.out_root / "cache")
    # Сверка на текстах всех длин: равномерно по отсортированному списку.
    order = np.argsort(valid.lengths, kind="stable")
    pick = order[np.linspace(0, len(order) - 1, min(PARITY_TEXTS, len(order))).astype(int)]
    sub = valid.subset(pick)
    model = load_torch_classifier(model_dir)
    ref = predict_torch(model, sub, torch.device("cpu"), "fp32", 16, pad_id)
    parity: dict[str, Any] = {}
    for variant, path in (("fp32", fp32), ("int8", int8)):
        got = predict_onnx(session(path, os.cpu_count() or 1), sub, pad_id, 1 if variant == "int8" else 32)
        parity[variant] = {
            "texts": len(sub),
            "max_abs_diff": float(np.abs(got - ref).max()),
            "mean_abs_diff": float(np.abs(got - ref).mean()),
            "same_label": float(((got >= THRESHOLD) == (ref >= THRESHOLD)).mean()),
        }
    # Вывод по спецификации (tokenizers без transformers) даёт то же, что пакетная оценка.
    spec = inference_spec(params, wrap, "head", "fp32")
    lines = load(args.data, "classification", "valid", eval_limit)
    texts = [json.loads(lines[int(k)])["text"] for k in pick[:50]]
    runtime_tokenizer = inference_tokenizer(model_dir / "tokenizer.json")
    sess = session(fp32, os.cpu_count() or 1)
    one = np.array([predict_text(text, sess, runtime_tokenizer, spec) for text in texts])
    batch = predict_onnx(sess, valid.subset(pick[:50]), pad_id)
    parity["spec_vs_batch_max_abs_diff"] = float(np.abs(one - batch).max())
    print(f"Сверка: {json.dumps(parity, ensure_ascii=False)}", flush=True)
    if parity["fp32"]["max_abs_diff"] > 1e-3 or parity["spec_vs_batch_max_abs_diff"] > 1e-3:
        raise SystemExit("ONNX fp32 или вывод по спецификации расходятся с PyTorch, см. сверку выше")

    latency = measure_latency(model, {"fp32": fp32, "int8": int8}, valid, suffix, latency_lengths(params["max_length"]))
    table(
        "Задержка на один текст, мс",
        ["Среда", "Токенов", "Потоков", "мс"],
        [[row["runtime"], str(row["tokens"]), str(row["threads"]), f"{row['ms']:.1f}"] for row in latency],
    )
    weights = sum(file.stat().st_size for file in model_dir.glob("*.safetensors"))
    export = {
        "parity": parity,
        "latency": latency,
        "cpu": cpu_name(),
        "sizes_mb": {
            "onnx_fp32": onnx_size(fp32) / 1e6,
            "onnx_int8": onnx_size(int8) / 1e6,
            "safetensors": (weights or sum(parameter.numel() for parameter in model.parameters()) * 4) / 1e6,
            "tokenizer": sum(file.stat().st_size for file in model_dir.glob("tokenizer*")) / 1e6,
        },
        "environment": environment(),
    }
    (run_dir / args.json).write_text(json.dumps(export, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Готово: {run_dir / args.json}", flush=True)


def eval_split(data: Path, split: str, limit: int, model_dir: Path, params: dict, cache: Path) -> Encoded:
    """Часть корпуса в токенах, как при обучении."""
    return encode_split(
        data, split, limit, load_tokenizer(model_dir), params["max_length"], not params["normalize"], cache
    )


def cpu_name() -> str:
    brand = ""
    if sys.platform == "darwin":
        done = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=False)
        brand = done.stdout.strip()
    elif Path("/proc/cpuinfo").exists():
        names = re.findall(r"^model name\s*:\s*(.+)$", Path("/proc/cpuinfo").read_text(), re.MULTILINE)
        brand = names[0].strip() if names else ""
    return (
        f"{brand or platform.processor() or platform.machine()}, {plural(os.cpu_count() or 1, 'ядро', 'ядра', 'ядер')}"
    )


# ─── Вывод на CPU ───────────────────────────────────────────────────────


def inference_spec(params: dict, wrap: dict, long_texts: str, weights: str = "int8") -> dict:
    """Всё, что нужно выводу без transformers: файл, токены, нормализация, порог, правило для длинных текстов.

    wrap — токены вокруг окна: cls_id, sep_id и pad_id.
    """
    windows = MAX_WINDOWS if long_texts == "mean_of_windows" else 1
    read = params["max_length"]
    fragment = min(read, FRAGMENT_LENGTH)
    return {
        "version": INFERENCE_VERSION,
        "format": "onnx",
        "file": "model.onnx",
        "weights": weights,
        "tokenizer": "tokenizer.json",
        "inputs": ["input_ids", "attention_mask"],
        "output": "logits",
        "labels": list(LABELS),
        "threshold": THRESHOLD,
        "max_length": fragment,
        **({"read_length": read} if read > fragment else {}),
        "max_chars": read * CHARS_PER_TOKEN * MAX_WINDOWS,
        **wrap,
        "long_texts": long_texts,
        "max_windows": windows,
        "normalize": [{"pattern": pat, "replacement": rep, "why": why} for pat, rep, why in NORMALIZE_RULES]
        if params["normalize"]
        else [],
    }


def windows(
    ids: Sequence[int], max_length: int, prefix: Sequence[int], suffix: Sequence[int], limit: int
) -> list[list[int]]:
    """Окна по max_length токенов подряд, без перекрытия; ids — токены текста без служебных.

    prefix и suffix — то, что ставится вокруг каждого окна: [CLS] и [SEP].
    """
    width = max_length - len(prefix) - len(suffix)
    return [[*prefix, *ids[i : i + width], *suffix] for i in range(0, max(len(ids), 1), width)][:limit]


def window_probs(sess: Session, ids: Sequence[int], spec: dict) -> np.ndarray:
    """Вероятности окон текста целиком: окно read_length, если оно есть, иначе max_length."""
    text_windows = windows(ids, spec.get("read_length", spec["max_length"]), *spec_wrap(spec), spec["max_windows"])
    batch_ids = np.full(
        (len(text_windows), max(len(window) for window in text_windows)), spec["pad_id"], dtype=np.int64
    )
    mask = np.zeros_like(batch_ids)
    for k, window in enumerate(text_windows):
        batch_ids[k, : len(window)] = window
        mask[k, : len(window)] = 1
    return softmax_ai(run_logits(sess, batch_ids, mask))


def predict_text(text: str, sess: Session, tok: tokenizers.Tokenizer, spec: dict) -> float:
    """Вероятность, что текст написала модель: так работает вывод по inference.json.

    Текст обрезается до max_chars знаков, нормализуется правилами из spec, режется на
    окна по max_length токенов; вероятность — первое окно или среднее по окнам.
    """
    text = text[: spec["max_chars"]]
    for rule in spec["normalize"]:
        text = re.sub(rule["pattern"], rule["replacement"], text, flags=re.MULTILINE)
    probs = window_probs(sess, tok.encode(text.strip(), add_special_tokens=False).ids, spec)
    return float(probs.mean())


def long_texts(
    sess: Session, data: Path, split: str, limit: int, enc: Encoded, head: np.ndarray, model_dir: Path, params: dict
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Начало текста против среднего по окнам на текстах длиннее max_length."""
    indices = np.flatnonzero(enc.truncated)
    lines = load(data, "classification", split, limit)
    spec = inference_spec(params, model_wrap(model_dir), "mean_of_windows")
    runtime_tokenizer = inference_tokenizer(model_dir / "tokenizer.json")
    mean = np.array([predict_text(json.loads(lines[int(k)])["text"], sess, runtime_tokenizer, spec) for k in indices])
    labels = enc.y[indices]
    return (
        {"texts": len(indices), "head": quick(labels, head[indices]), "mean_of_windows": quick(labels, mean)},
        indices,
        mean,
    )


def spec_vs_eval(
    sess: Session, data: Path, limit: int, enc: Encoded, probs: np.ndarray, tokenizer: Path, spec: dict
) -> dict:
    """Вывод по inference.json, текст за текстом, как в aiw-ru, против итоговых вероятностей оценки.

    Тексты берутся равномерно по длине, чтобы попали и короткие, и обрезанные.
    """
    order = np.argsort(enc.lengths, kind="stable")
    pick = order[np.linspace(0, len(order) - 1, min(PARITY_TEXTS, len(order))).astype(int)]
    lines = load(data, "classification", "valid", limit)
    runtime_tokenizer = inference_tokenizer(tokenizer)
    one = np.array([predict_text(json.loads(lines[int(k)])["text"], sess, runtime_tokenizer, spec) for k in pick])
    diff = np.abs(one - probs[pick])
    return {"texts": len(pick), "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean())}


def model_wrap(model_dir: Path) -> dict:
    """Токены вокруг окна для inference.json: cls_id, sep_id и pad_id токенизатора модели."""
    cls_id, sep_id, pad_id = special_ids(load_tokenizer(model_dir))
    return {"cls_id": cls_id, "sep_id": sep_id, "pad_id": pad_id}


def spec_wrap(spec: dict) -> tuple[list[int], list[int]]:
    return [spec["cls_id"]], [spec["sep_id"]]


def load_torch_classifier(model_dir: Path) -> Any:
    """Модель для сверки с ONNX и замера задержки."""
    model: Any = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return model.eval().float()


# ─── Оценка ─────────────────────────────────────────────────────────────


def install_size(packages: Sequence[str], python_platform: str, index: str | None = None) -> float | None:
    """Мегабайты на диске после установки пакетов с зависимостями через uv (Python 3.12)."""
    uv_path = shutil.which("uv")
    if not uv_path:
        return None
    with tempfile.TemporaryDirectory() as tmp_dir:
        cmd = [uv_path, "pip", "install", "--quiet", "--target", tmp_dir, "--python-version", "3.12"]
        cmd += ["--python-platform", python_platform, *packages]
        if index:
            # У uv дополнительный индекс важнее основного: torch возьмётся из него, остальное с PyPI.
            cmd += ["--extra-index-url", index]
        # Колёса onnxruntime и torch для macOS собраны под 14.0 и новее.
        env = {**os.environ, "MACOSX_DEPLOYMENT_TARGET": "14.0"}
        done = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
        if done.returncode:
            print(f"uv pip install {' '.join(packages)}: {done.stderr.strip()[:300]}", file=sys.stderr)
            return None
        return sum(file.stat().st_size for file in Path(tmp_dir).rglob("*") if file.is_file()) / 1e6


def install_sizes() -> list[dict]:
    """Размер зависимостей вывода: onnxruntime + tokenizers против torch + transformers."""
    onnx_deps = [f"onnxruntime=={ort.__version__}", f"tokenizers=={tokenizers.__version__}"]
    torch_deps = [f"torch=={torch.__version__.split('+')[0]}", f"transformers=={transformers.__version__}"]
    cases = [
        ("onnxruntime + tokenizers", onnx_deps, "aarch64-apple-darwin", None),
        ("onnxruntime + tokenizers", onnx_deps, "x86_64-manylinux_2_28", None),
        ("onnxruntime + tokenizers", onnx_deps, "x86_64-pc-windows-msvc", None),
        ("torch + transformers", torch_deps, "aarch64-apple-darwin", None),
        ("torch + transformers", torch_deps, "x86_64-manylinux_2_28", None),
        ("torch (CPU) + transformers", torch_deps, "x86_64-manylinux_2_28", "https://download.pytorch.org/whl/cpu"),
        ("torch + transformers", torch_deps, "x86_64-pc-windows-msvc", None),
    ]
    rows = []
    for name, pkgs, plat, index in cases:
        size = install_size(pkgs, plat, index)
        print(f"  {name}, {plat}: {'—' if size is None else f'{size:.0f} МБ'}", file=sys.stderr, flush=True)
        rows.append({"stack": name, "packages": pkgs, "platform": plat, "mb": size})
    return rows


# Приметы оформления: насколько каждая одна отделяет людей от моделей без всякого обучения.
SHORTCUTS: dict[str, Callable[[str], float]] = {
    "длина, знаков": lambda t: float(len(t)),
    "переводов строк на 1000 знаков": lambda t: 1000 * t.count("\n") / max(len(t), 1),
    "разметка markdown на 1000 знаков": lambda t: (
        1000 * len(re.findall(r"^\s*(?:#|[-*+]\s|\d+\.\s)|\*\*", t, re.MULTILINE)) / max(len(t), 1)
    ),
    "доля ё среди е и ё": lambda t: t.lower().count("ё") / max(t.lower().count("е") + t.lower().count("ё"), 1),
    "длинных тире на 1000 знаков": lambda t: 1000 * t.count(chr(0x2014)) / max(len(t), 1),
    "кавычек-ёлочек на 1000 знаков": lambda t: 1000 * t.count(chr(0x00AB)) / max(len(t), 1),
}


def shortcut_probe(data: Path, split: str, limit: int) -> list[dict]:
    """ROC AUC каждой приметы оформления отдельно; 0,5 — примета ничего не говорит."""
    rows = [json.loads(line) for line in load(data, "classification", split, limit)]
    labels = np.array([row["label"] == "ai" for row in rows], dtype=np.int8)
    scores = []
    for name, measure in SHORTCUTS.items():
        auc_value = auc(labels, np.array([measure(row["text"]) for row in rows], dtype=np.float64))
        # Направление приметы не важно: 0,3 говорит столько же, сколько 0,7.
        scores.append(
            {
                "cue": name,
                "roc_auc": auc_value,
                "separation": None if auc_value is None else max(auc_value, 1 - auc_value),
            }
        )
    return scores


def lightgbm_block() -> dict | None:
    """Метрики LightGBM на том же test для сравнения: из docs/models или с Hugging Face."""
    path = ROOT / LIGHTGBM_METRICS
    if not path.exists():
        try:
            path = Path(hf_hub_download(LIGHTGBM_REPO, "metrics.json"))
        except (HfHubHTTPError, OSError) as error:
            print(f"Метрики LightGBM не найдены: {error}", file=sys.stderr)
            return None
    metrics = json.loads(path.read_text(encoding="utf-8"))
    keep = ("created", "git_commit", "test", "valid", "genres", "lengths", "prompt_types", "generators")
    size = metrics.get("params", {}).get("size_kb")
    return {
        "repo": LIGHTGBM_REPO,
        "size_mb": size / 1024 if size else None,
        **{k: metrics[k] for k in keep if k in metrics},
    }


def pilot_rows(out_root: Path) -> list[dict]:
    """Строки таблицы пилота: pilot/*/run.json и export.json."""
    rows = []
    for run_file in sorted((out_root / "pilot").glob("*/run.json")):
        run = json.loads(run_file.read_text(encoding="utf-8"))
        exp_file = run_file.parent / "export.json"
        export_info = json.loads(exp_file.read_text(encoding="utf-8")) if exp_file.exists() else {}
        latency_ms = {
            (row["runtime"], row["tokens"], row["threads"]): row["ms"] for row in export_info.get("latency", [])
        }
        many = max((row["threads"] for row in export_info.get("latency", [])), default=0)
        params = run["params"]
        rows.append(
            {
                "base": params["base"],
                "license": params["base_license"],
                "normalize": params["normalize"],
                "train": run["dataset"]["train"],
                "valid": run["dataset"]["valid"],
                "epochs": params.get("epochs"),
                "lr": params.get("lr"),
                "batch": params.get("batch"),
                "seed": params.get("seed"),
                "valid_roc_auc": run["valid"]["roc_auc"],
                "valid_accuracy": run["valid"]["accuracy"],
                "parameters": params["parameters"],
                "train_seconds": params["train_seconds"],
                "texts_per_second": params["texts_per_second"],
                "gpu": params.get("gpu", ""),
                "onnx_fp32_mb": export_info.get("sizes_mb", {}).get("onnx_fp32"),
                "onnx_int8_mb": export_info.get("sizes_mb", {}).get("onnx_int8"),
                "int8_ms_128_1": latency_ms.get(("onnxruntime int8", 128, 1)),
                "int8_ms_512_1": latency_ms.get(("onnxruntime int8", 512, 1)),
                "int8_ms_512_all": latency_ms.get(("onnxruntime int8", 512, many)),
                "fp32_ms_512_1": latency_ms.get(("onnxruntime fp32", 512, 1)),
            }
        )
    return rows


def flat_environment(train_env: dict[str, str]) -> dict[str, str]:
    """Версии при обучении и при оценке на CPU в одном плоском словаре, как у LightGBM."""
    env_info = environment()
    # run.json с VM мог записать «2 ядер»: число ядер приводится к правильной форме.
    train_env = {
        k: re.sub(r"(\d+) ядер\b", lambda match: plural(int(match[1]), "ядро", "ядра", "ядер"), v)
        for k, v in train_env.items()
    }
    return {
        **train_env,
        "onnx": _onnx_version(),
        **{f"evaluate_{k}": v for k, v in env_info.items() if train_env.get(k) != v},
    }


def _onnx_version() -> str:
    import onnx

    return onnx.__version__


def bundle_hub(run_dir: Path, shipped_file: str) -> Path:
    """Папка hub/ для репозитория модели: model.onnx — то, что в поставке, рядом другой вариант весов.

    aiw-ru скачивает inference.json, model.onnx, tokenizer.json, metrics.json и README.md;
    model_fp32.onnx или model_int8.onnx лежит для тех, кому нужен другой вариант, а веса PyTorch
    (model.safetensors с config.json и токенизатором рядом) — для дообучения и своего экспорта.
    """
    hub_dir = run_dir / "hub"
    shutil.rmtree(hub_dir, ignore_errors=True)
    hub_dir.mkdir(parents=True)
    files = {"model.onnx": shipped_file}
    if shipped_file != "onnx/model.onnx":
        files["model_fp32.onnx"] = "onnx/model.onnx"
    elif (run_dir / "onnx" / "model_int8.onnx").exists():
        files["model_int8.onnx"] = "onnx/model_int8.onnx"
    for name in ("tokenizer.json", "tokenizer_config.json", "config.json", "model.safetensors"):
        files[name] = f"model/{name}"
    files |= {"inference.json": "inference.json", "metrics.json": "metrics.json"}
    for dst, src in files.items():
        if (run_dir / src).exists():
            copy_onnx(run_dir / src, hub_dir / dst) if dst.endswith(".onnx") else shutil.copy2(
                run_dir / src, hub_dir / dst
            )
    return hub_dir


def copy_onnx(src: Path, dst: Path) -> None:
    """Копия ONNX; внешний файл весов получает имя по новому файлу, чтобы model.onnx* не тянул чужие веса.

    torch.onnx.export кладёт веса больше 2 ГБ рядом, в файл с именем модели и суффиксом .data.
    """
    data = src.with_name(src.name + ".data")
    if not data.exists():
        shutil.copy2(src, dst)
    elif dst.name == src.name:
        shutil.copy2(src, dst)
        shutil.copy2(data, dst.with_name(dst.name + ".data"))
    else:
        import onnx

        onnx.save(onnx.load(src), dst, save_as_external_data=True, location=dst.name + ".data")


def cpu_latency(run_dir: Path, metrics: dict) -> dict:
    """Задержка с другой машины: export в копии папки запуска кладёт export-latency.json.

    Нужен, когда обучение, экспорт и оценка шли на машине с GPU, а задержку в карточках
    сравнивают на M1. Оценка остаётся с машины обучения, в hardware видно обе.
    """
    path = run_dir / "export-latency.json"
    if not path.exists():
        return {}
    export_latency = json.loads(path.read_text(encoding="utf-8"))
    evaluated = json.loads((run_dir / "export.json").read_text(encoding="utf-8"))["cpu"]
    hardware = f"обучение: {metrics['params']['gpu']}; оценка: {evaluated}; задержка: {export_latency['cpu']}"
    return {
        "latency": export_latency["latency"],
        "latency_cpu": export_latency["cpu"],
        "params": {**metrics["params"], "hardware": hardware},
    }


def x86_latency(run_dir: Path) -> dict:
    """Задержка на слабом x86: export, запущенный на CPU VM Colab, кладёт export-x86.json."""
    path = run_dir / "export-x86.json"
    if not path.exists():
        return {}
    x86 = json.loads(path.read_text(encoding="utf-8"))
    return {"latency_x86": x86["latency"], "latency_x86_cpu": x86["cpu"]}


def cmd_evaluate(args: argparse.Namespace) -> None:
    """ONNX на CPU по valid и test, разрезы как у LightGBM, metrics.json и inference.json.

    С --onnx-texts N итоговые вероятности берутся от PyTorch на GPU (valid-probs.npy, test-probs.npy,
    окна длинных текстов — из {часть}-windows-probs.npz), а ONNX сверяется с ними на N текстах valid:
    для модели, которую на CPU по всему корпусу пришлось бы считать сутки.
    """
    run_dir: Path = args.folder
    model_dir = run_dir / "model"
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    export = json.loads((run_dir / "export.json").read_text(encoding="utf-8"))
    params = run["params"]
    limit = run["dataset"]["eval_limit"]
    hf_logging.set_verbosity_error()
    pad_id = model_wrap(model_dir)["pad_id"]
    cache = args.out_root / "cache"
    valid = eval_split(args.data, "valid", limit, model_dir, params, cache)
    test = eval_split(args.data, "test", limit, model_dir, params, cache)
    parts = {"valid": valid, "test": test}
    subset = args.onnx_texts > 0
    order = np.argsort(valid.lengths, kind="stable")
    pick = order[np.linspace(0, len(order) - 1, min(args.onnx_texts, len(order))).astype(int)] if subset else None

    probs: dict[str, dict[str, np.ndarray]] = {"torch": {}}
    for split in ("valid", "test"):
        if (run_dir / f"{split}-probs.npy").exists():
            probs["torch"][split] = np.load(run_dir / f"{split}-probs.npy")
    threads = os.cpu_count() or 1
    splits: dict[str, tuple[Encoded, np.ndarray]] = (
        {"valid_subset": (valid.subset(pick), valid.y[pick])}
        if pick is not None
        else {split_key: (encoded, encoded.y) for split_key, encoded in parts.items()}
    )
    if pick is not None:
        probs["torch"]["valid_subset"] = probs["torch"]["valid"][pick]
    files = {"onnx_fp32": "model.onnx", "onnx_int8": "model_int8.onnx"}
    sessions: dict[str, Session] = {}

    def run_variant(variant: str, split: str) -> None:
        """Вероятности варианта на части: из кэша или ONNX на CPU; int8 по одному тексту, как в aiw-ru."""
        enc = splits[split][0]
        cached = run_dir / f"{split}-probs-{variant}{f'-{len(enc)}' if subset else ''}.npy"
        if not cached.exists():
            if variant not in sessions:
                sessions[variant] = session(run_dir / "onnx" / files[variant], threads)
            started = time.perf_counter()
            np.save(cached, predict_onnx(sessions[variant], enc, pad_id, 1 if variant == "onnx_int8" else 32))
            print(f"{variant} {split}: {time.perf_counter() - started:.0f} с", flush=True)
        probs.setdefault(variant, {})[split] = np.load(cached)

    for variant in files:
        for split in splits:
            # С --int8-valid-only int8 на test считается, только если он пойдёт в поставку.
            if not (variant == "onnx_int8" and split == "test" and args.int8_valid_only):
                run_variant(variant, split)
    cols = ("valid_subset",) if subset else ("valid", "test")
    # int8 идёт в поставку, если на valid теряет не больше --int8-max-drop ROC AUC и меняет метку не больше
    # чем у --int8-max-changes текстов; test в решении не участвует. --weights задаёт веса заранее.
    ref = cols[0]
    ref_labels = splits[ref][1]
    drop = (quick(ref_labels, probs["onnx_fp32"][ref])["roc_auc"] or 0) - (
        quick(ref_labels, probs["onnx_int8"][ref])["roc_auc"] or 0
    )
    changes = {
        split_key: float(
            ((probs["onnx_int8"][split_key] >= THRESHOLD) != (probs["onnx_fp32"][split_key] >= THRESHOLD)).mean()
        )
        for split_key in cols
        if split_key in probs["onnx_int8"]
    }
    if args.weights == "auto":
        ok = drop <= args.int8_max_drop and changes[ref] <= args.int8_max_changes
        shipped = "onnx_int8" if ok else "onnx_fp32"
    else:
        shipped = f"onnx_{args.weights}"
    shipped_file = f"onnx/{files[shipped]}"
    print(f"В поставку: {shipped} (int8 теряет {drop:.4f} ROC AUC, меняет метку у {changes})", flush=True)
    for split in splits:
        if split not in probs[shipped]:
            run_variant(shipped, split)
    labels_by_split = {split_key: labels for split_key, (_, labels) in splits.items()} | {
        split_key: encoded.y for split_key, encoded in parts.items()
    }
    variants = {
        variant_name: {
            split_key: quick(labels_by_split[split_key], split_probs)
            for split_key, split_probs in probs_by_split.items()
        }
        for variant_name, probs_by_split in probs.items()
    }
    table(
        "Варианты модели: ROC AUC и accuracy",
        ["Вариант", *(f"{split_key} {k}" for split_key in cols for k in ("AUC", "acc"))],
        [
            [
                variant_name,
                *(
                    fmt(variant_metrics.get(split_key, {}).get(k), 4)
                    for split_key in cols
                    for k in ("roc_auc", "accuracy")
                ),
            ]
            for variant_name, variant_metrics in variants.items()
        ],
    )
    sess = session(run_dir / shipped_file, threads)
    main = probs["torch"] if subset else probs[shipped]

    # Длинные тексты: начало или среднее по окнам, выбор по valid.
    if subset:
        long_quality: dict[str, tuple[dict, np.ndarray, np.ndarray]] = {}
        for split, enc in parts.items():
            windows_npz = np.load(run_dir / f"{split}-windows-probs.npz")
            labels = enc.y[windows_npz["idx"]]
            quality = {
                "texts": len(windows_npz["idx"]),
                "head": quick(labels, main[split][windows_npz["idx"]]),
                "mean_of_windows": quick(labels, windows_npz["mean"]),
            }
            long_quality[split] = (quality, windows_npz["idx"], windows_npz["mean"])
        (long_valid, idx_v, mean_v), (long_test, idx_t, mean_t) = long_quality["valid"], long_quality["test"]
    else:
        long_valid, idx_v, mean_v = long_texts(sess, args.data, "valid", limit, valid, main["valid"], model_dir, params)
        long_test, idx_t, mean_t = long_texts(sess, args.data, "test", limit, test, main["test"], model_dir, params)
    use_windows = (long_valid["mean_of_windows"]["roc_auc"] or 0) > (long_valid["head"]["roc_auc"] or 0)
    print(f"Длинные тексты, valid: {long_valid}\nДлинные тексты, test: {long_test}", flush=True)
    final = main["test"].copy()
    final_valid = main["valid"].copy()
    if use_windows:
        final[idx_t] = mean_t
        final_valid[idx_v] = mean_v
    wrap = model_wrap(model_dir)
    spec = inference_spec(params, wrap, "mean_of_windows" if use_windows else "head", shipped.removeprefix("onnx_"))
    spec_parity = spec_vs_eval(sess, args.data, limit, valid, final_valid, model_dir / "tokenizer.json", spec)
    print(f"Вывод по inference.json против оценки: {spec_parity}", flush=True)

    _, results = detect(args.data, "classification", "test", limit, "scan", "general", args.jobs)
    score = np.array([row["score"] for row in results], dtype=np.float32)
    part = Part(
        x=np.zeros((len(test), 0), dtype=np.float32),
        y=test.y,
        score=score,
        words=np.array([row["stats"]["words"] for row in results]),
        genre=test.genre,
        prompt=test.prompt,
        generator=test.generator,
    )
    created = (test.y == 0) | (test.prompt == "create")
    model_test = classes(test.y, (final >= THRESHOLD).astype(np.int8))
    print_classes("Модель на test, порог вероятности 0,5", model_test)

    if args.reuse_install:
        install = json.loads((args.reuse_install / "metrics.json").read_text(encoding="utf-8"))["install"]
    else:
        install = [] if args.skip_install else install_sizes()
    metrics = {
        "created": datetime.now(UTC).strftime("%Y-%m-%d"),
        "aiw_ru_version": aiw_ru.__version__,
        "git_commit": run["git_commit"],
        "environment": flat_environment(run["environment"]),
        "dataset": {**run["dataset"], "valid": len(valid), "test": len(test)},
        "params": {
            **params,
            "license": params["base_license"],
            "shipped": shipped,
            "shipped_file": shipped_file,
            "weights_choice": args.weights,
            "int8_max_drop": args.int8_max_drop,
            "int8_max_changes": args.int8_max_changes,
            "int8_label_changes": changes,
            "int8_valid_only": args.int8_valid_only,
            "probs_source": "torch" if subset else "onnx",
            "onnx_texts": len(pick) if pick is not None else None,
            "size_mb": export["sizes_mb"][shipped],
            "external_data": (run_dir / f"{shipped_file}.data").exists(),
            "hardware": f"обучение: {params['gpu']}; оценка и задержка: {export['cpu']}",
        },
        "test": {**model_test, "roc_auc": auc(test.y, final), "roc_auc_created": auc(test.y[created], final[created])},
        "valid": {
            "accuracy": float((valid.y == (final_valid >= THRESHOLD)).mean()),
            "roc_auc": auc(valid.y, final_valid),
        },
        "detector": {
            **classes(test.y, (score >= 40).astype(np.int8)),
            "threshold": 40,
            "roc_auc": auc(test.y, score),
            "roc_auc_created": auc(test.y[created], score[created]),
        },
        **breakdowns(part, final),
        "variants": variants,
        "long_texts": {"rule": spec["long_texts"], "valid": long_valid, "test": long_test},
        "parity": {**export["parity"], "spec_vs_eval": spec_parity},
        "sizes_mb": export["sizes_mb"],
        "latency": export["latency"],
        "latency_cpu": export["cpu"],
        **x86_latency(run_dir),
        "install": install,
        "shortcuts": shortcut_probe(args.data, "valid", limit),
        "curve": run["curve"],
        "pilot": pilot_rows(args.out_root),
        "inference": {k: setting for k, setting in spec.items() if k != "normalize"},
        "lightgbm": lightgbm_block(),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "inference.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Готово: {run_dir / 'metrics.json'}, {run_dir / 'inference.json'}", flush=True)


# ─── Карточка и отчёт ───────────────────────────────────────────────────

REPO = "toiletsandpaper/russian-ai-text-detector-bert"
DOCS = Path("docs") / "models"


@dataclass(frozen=True)
class Bundle:
    """Модель aiw-ru: репозиторий на Hub, имя для `aiw-ru models install`, имя лога в Colab."""

    repo: str
    name: str
    log: str
    # Ставится командой `aiw-ru models install` без имени и отвечает в classify по умолчанию.
    default: bool = False
    # Когда брать модель, пока её нет в aiw_ru.models; потом строка берётся оттуда, как в `models info`.
    summary: str = ""


BUNDLES = {
    "cointegrated/rubert-tiny2": Bundle(REPO, "transformer", "tiny2"),
    "deepvk/RuModernBERT-small": Bundle(
        "toiletsandpaper/russian-ai-text-detector-modernbert", "modernbert", "modernbert", default=True
    ),
    "sergeyzh/rubert-mini-frida": Bundle(
        "toiletsandpaper/russian-ai-text-detector-mini-frida", "mini-frida", "mini-frida"
    ),
}


def bundle_for(base: str) -> Bundle:
    return BUNDLES.get(base, Bundle(REPO, "transformer", short_name(base)))


LICENSE_NAMES = {"mit": "MIT", "apache-2.0": "Apache 2.0"}
# Как ссылаться на базу: BibTeX из карточки базы, если он там есть, иначе ссылка на описание.
BASE_CITATIONS = {
    "cointegrated/rubert-tiny2": (
        "`cointegrated/rubert-tiny2` описан в [посте автора на Хабре](https://habr.com/ru/post/669674/)."
    ),
    "deepvk/RuModernBERT-small": """RuModernBERT построен по схеме [ModernBERT](https://arxiv.org/abs/2412.13663).

```bibtex
@misc{deepvk2025rumodernbert,
  title = {RuModernBERT: Modernized BERT for Russian},
  author = {Spirin, Egor and Malashenko, Boris and Sokolov, Andrey},
  url = {https://huggingface.co/deepvk/rumodernbert-base},
  publisher = {Hugging Face},
  year = {2025},
}
```""",
    "sergeyzh/rubert-mini-frida": (
        "`sergeyzh/rubert-mini-frida` получен дистилляцией эмбеддингов "
        "[ai-forever/FRIDA](https://huggingface.co/ai-forever/FRIDA) в "
        "[sergeyzh/rubert-mini-sts](https://huggingface.co/sergeyzh/rubert-mini-sts); "
        "статьи с BibTeX у модели нет."
    ),
}


def _auc(value: float | None) -> str:
    return num(value) if value is not None else "—"


def _num4(value: float | None) -> str:
    return num(value, 4) if value is not None else "—"


def _ms(value: float | None) -> str:
    return f"${value:.1f}$" if value is not None else "—"


def _mb(value: float | None) -> str:
    if value is None:
        return "—"
    return f"${value:.0f}$" if value >= 100 else f"${value:.1f}$"


def _sci(value: float) -> str:
    """Малое число формулой: $3.2\\cdot 10^{-6}$."""
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"${mantissa}\\cdot 10^{{{int(exponent)}}}$"


def _code(text: str) -> str:
    """Код в markdown; шаблон с обратными кавычками внутри оборачивается двумя."""
    return f"`` {text} ``" if "`" in text else f"`{text}`"


def _by(rows: list[dict] | None, key: str) -> dict[str, dict]:
    return {row[key]: row for row in rows or []}


def _nw(number: int, one: str, few: str, many: str) -> str:
    """Число формулой и слово в нужной форме: $2$ текста, $5$ текстов."""
    # Форма зависит только от двух последних цифр, а число меньше 100 plural не делит на разряды.
    return f"{count(number)} {plural(number % 100, one, few, many).split(' ', 1)[1]}"


def _tokens_gen(number: int) -> str:
    """«длиннее $512$ токенов», «длиннее $21$ токена»."""
    return _nw(number, "токена", "токенов", "токенов")


def _texts_loc(number: int) -> str:
    """«на $5$ текстах», «на $21$ тексте»."""
    return _nw(number, "тексте", "текстах", "текстах")


def _params_count(number: int) -> str:
    return f"${number / 1e6:.1f}$ млн"


def _weights(metrics: dict) -> str:
    return "int8" if metrics["params"]["shipped"] == "onnx_int8" else "fp32"


def _lat(rows_: list[dict] | None, runtime: str, tokens: int = 512, threads: int = 1) -> float | None:
    for row in rows_ or []:
        if (row["runtime"], row["tokens"], row["threads"]) == (runtime, tokens, threads):
            return row["ms"]
    return None


def _shipped_ms(metrics: dict, x86: bool = False) -> float | None:
    """Задержка того, что в поставке, на $512$ токенах в один поток: M1 или два ядра x86."""
    return _lat(metrics.get("latency_x86") if x86 else metrics.get("latency"), f"onnxruntime {_weights(metrics)}")


def _hub_link(repo: str) -> str:
    return f"[`{repo.split('/')[-1]}`](https://huggingface.co/{repo})"


def _classify_cmd(bundle: Bundle) -> str:
    return "aiw-ru classify текст.md" if bundle.default else f"aiw-ru classify --model {bundle.name} текст.md"


def related_row(run_dir: Path) -> dict:
    """Другая модель aiw-ru для сравнения в карточке: её metrics.json после evaluate."""
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics |= x86_latency(run_dir)
    bundle = bundle_for(metrics["params"]["base"])
    return {
        "name": bundle.name,
        "repo": bundle.repo,
        "base": metrics["params"]["base"],
        "weights": _weights(metrics),
        "size_mb": metrics["params"]["size_mb"],
        "ms_512_1": _shipped_ms(metrics),
        "x86_ms_512_1": _shipped_ms(metrics, x86=True),
        "test": {k: metrics["test"][k] for k in ("accuracy", "roc_auc", "roc_auc_created", "macro", "classes")},
    }


def _models_rows(metrics: dict, repo: str) -> list[dict]:
    """Все модели aiw-ru с числами на test в порядке aiw-ru: по доле людей, принятых за ИИ."""
    rows = [
        {
            "name": bundle_for(metrics["params"]["base"]).name,
            "repo": repo,
            "test": metrics["test"],
            "size_mb": metrics["params"]["size_mb"],
            "ms": _shipped_ms(metrics),
            "x86": _shipped_ms(metrics, x86=True),
            "this": True,
        }
    ]
    rows += [
        {
            "name": run["name"],
            "repo": run["repo"],
            "test": run["test"],
            "size_mb": run["size_mb"],
            "ms": run["ms_512_1"],
            "x86": run["x86_ms_512_1"],
            "this": False,
        }
        for run in metrics.get("related", [])
    ]
    lightgbm = metrics.get("lightgbm") or {}
    if lightgbm.get("test"):
        rows.append(
            {
                "name": "lightgbm",
                "repo": lightgbm["repo"],
                "test": lightgbm["test"],
                "size_mb": lightgbm.get("size_mb"),
                "ms": None,
                "x86": None,
                "this": False,
            }
        )
    return sorted(rows, key=lambda run: _fpr(run["test"]))


def _fpr(split_metrics: dict) -> float:
    return 1 - split_metrics["classes"]["human"]["recall"]


def _models_md(rows: list[dict]) -> str:
    """Таблица моделей: доля людей, принятых за ИИ, первой, потому что по ней aiw-ru выбирает модель."""

    def name(row: dict) -> str:
        link = f"[`{row['name']}`](https://huggingface.co/{row['repo']})"
        return f"{link}, эта модель" if row["this"] else link

    head = ["Модель", "Людей принято за ИИ", "ROC AUC", "Accuracy", "Файл, МБ", "M1, мс", "x86, мс"]
    return md_table(
        head,
        [
            [
                name(row),
                pct(_fpr(row["test"])),
                _num4(row["test"]["roc_auc"]),
                num(row["test"]["accuracy"]),
                _mb(row["size_mb"]),
                _ms(row["ms"]),
                _ms(row["x86"]),
            ]
            for row in rows
        ],
    )


def _summary(name: str) -> str:
    """Когда брать модель: та же строка, что показывает `aiw-ru models info`."""
    if name in MODELS:
        return MODELS[name].summary
    return next((bundle.summary for bundle in BUNDLES.values() if bundle.name == name), "")


def _notes_md(rows: list[dict]) -> str:
    return "\n".join(
        f"- `{row['name']}`{' (эта модель)' if row['this'] else ''} — {_summary(row['name'])}."
        for row in rows
        if _summary(row["name"])
    )


def _choose_md(metrics: dict, repo: str) -> str:
    """Раздел «Какую модель выбрать»: вступление, таблица, строка на модель и правило выбора."""
    rows = _models_rows(metrics, repo)
    return f"""{_position(metrics, rows)}

{_models_md(rows)}

{_notes_md(rows)}

Если установлено несколько моделей, aiw-ru берёт вероятность у первой из них
в порядке таблицы. Порядок задаёт доля людей, принятых за ИИ, а не ROC AUC:
ошибиться в человеке дороже, чем пропустить ИИ-текст.

Задержка — один текст на $512$ токенов в один поток на M1 и на двух ядрах Xeon
виртуальной машины Colab (x86). Размер у LightGBM — файл модели, признаки для неё
считает сам детектор aiw-ru."""


_ORDINAL = {1: "первая", 2: "вторая", 3: "третья", 4: "четвёртая", 5: "пятая", 6: "шестая"}
_OF = {2: "двух", 3: "трёх", 4: "четырёх", 5: "пяти", 6: "шести"}


def _position(metrics: dict, rows: list[dict]) -> str:
    """Место модели среди трансформеров aiw-ru: точнее или быстрее и какой ценой."""
    rel = metrics.get("related", [])
    auc_, latency_ms, size_mb = metrics["test"]["roc_auc"], _shipped_ms(metrics), metrics["params"]["size_mb"]
    most_accurate = all(auc_ >= run["test"]["roc_auc"] for run in rel)
    fastest = latency_ms is not None and all(run["ms_512_1"] is None or latency_ms <= run["ms_512_1"] for run in rel)
    slowest = latency_ms is not None and all(run["ms_512_1"] is None or latency_ms >= run["ms_512_1"] for run in rel)
    better = [f"`{run['name']}`" for run in rel if run["test"]["roc_auc"] > auc_]
    if most_accurate and fastest:
        lead = "Это самый точный трансформер aiw-ru и самый быстрый."
    elif most_accurate:
        lead = "Это точный вариант среди трансформеров aiw-ru, за точность он платит скоростью и размером."
    elif fastest:
        lead = "Это быстрый вариант среди трансформеров aiw-ru, за скорость он платит точностью."
    elif slowest:
        verb = "обходит" if len(better) == 1 else "обходят"
        lead = f"Это самый тяжёлый трансформер aiw-ru, но не самый точный: на test его {verb} {', '.join(better)}."
    else:
        by_auc = sorted(rows, key=lambda run: -run["test"]["roc_auc"])
        auc_rank = next(i for i, run in enumerate(by_auc, 1) if run["this"])
        fpr_rank = next(i for i, run in enumerate(rows, 1) if run["this"])
        link = "и" if auc_rank == fpr_rank else "а"
        lead = (
            f"По ROC AUC на test это {_ORDINAL[auc_rank]} из {_OF[len(rows)]} моделей aiw-ru, {link} по доле "
            f"людей, принятых за ИИ, — {_ORDINAL[fpr_rank]}."
        )
    if size_mb >= 1000:
        lead += (
            f" Модель тяжёлая, для мощных машин: {_mb(size_mb)} МБ на диске и {_ms(latency_ms)} мс на текст в $512$ токенов "
            "на M1 в один поток."
        )
    return lead


def _pilot_md(metrics: dict) -> str:
    return md_table(
        ["База", "Лицензия", "Нормализация", "Параметров", "ROC AUC valid", "Accuracy valid", "int8, МБ", "int8, мс"],
        [
            [
                f"`{run['base']}`",
                LICENSE_NAMES.get(run["license"], run["license"]),
                "да" if run["normalize"] else "нет",
                _params_count(run["parameters"]),
                _auc(run["valid_roc_auc"]),
                num(run["valid_accuracy"]),
                _mb(run.get("onnx_int8_mb")),
                _ms(run.get("int8_ms_512_1")),
            ]
            for run in metrics["pilot"]
        ],
    )


def _finalists_md(metrics: dict) -> str:
    return md_table(
        [
            "База",
            "Имя в aiw-ru",
            "ROC AUC valid",
            "Accuracy valid",
            "ROC AUC int8",
            "int8 сменил метку",
            "Веса в поставке",
            "Файл, МБ",
            "мс, M1",
            "мс, x86",
        ],
        [
            [
                f"`{run['base']}`",
                f"`{bundle_for(run['base']).name}`",
                _num4(run["valid_roc_auc"]),
                num(run["valid_accuracy"]),
                _num4(run.get("valid_roc_auc_int8")),
                pct(run["int8_label_changes"]) if run.get("int8_label_changes") is not None else "—",
                run.get("weights", "int8"),
                _mb(run.get("shipped_mb", run.get("onnx_int8_mb"))),
                _ms(run.get("shipped_ms_512_1", run.get("int8_ms_512_1"))),
                _ms(run.get("x86_shipped_ms_512_1", run.get("x86_int8_ms_512_1"))),
            ]
            for run in metrics.get("finalists", [])
        ],
    )


def _finalists_intro(metrics: dict) -> str:
    """Какие базы дошли до полного train и чем стала каждая."""
    rows = metrics["finalists"]
    if len(rows) < 3:
        return (
            "Дальше на полном train обучены две базы: самая точная в пилоте и самая быстрая. Середина, "
            "`sergeyzh/rubert-mini-frida`, медленнее tiny2 больше чем вдвое при небольшом выигрыше в ROC AUC."
        )
    fast = min(rows, key=lambda run: run.get("int8_ms_512_1") or float("inf"))
    best = max(rows, key=lambda run: run["valid_roc_auc"])
    mid = [run for run in rows if run is not fast and run is not best]

    def name(run: dict) -> str:
        return f"`{bundle_for(run['base']).name}`"

    bases = ", ".join(f"`{run['base']}`" for run in mid)
    names = ", ".join(name(run) for run in mid)
    return (
        f"На полном train обучены {_nw(len(rows), 'база', 'базы', 'баз')} из пилота: самая быстрая, "
        f"`{fast['base']}`, самая точная, `{best['base']}`, и средняя по обоим, {bases}. Все они выпущены "
        f"как модели aiw-ru: {name(best)} точнее всех и ставится по умолчанию, {name(fast)} считает быстрее "
        f"всех, {names} — промежуточный вариант."
    )


def _compare_md(metrics: dict) -> str:
    rel = sorted(metrics.get("related", []), key=lambda run: -run["test"]["roc_auc"])
    cols = [
        metrics["test"],
        *(run["test"] for run in rel),
        (metrics.get("lightgbm") or {}).get("test", {}),
        metrics["detector"],
    ]

    def fpr(run: dict) -> str:
        return pct(1 - run["classes"]["human"]["recall"]) if run else "—"

    rows = [
        ["Accuracy", *(_auc(column.get("accuracy")) for column in cols)],
        ["ROC AUC", *(_auc(column.get("roc_auc")) for column in cols)],
        ["ROC AUC, люди против текстов с нуля", *(_auc(column.get("roc_auc_created")) for column in cols)],
        ["F1, среднее по классам", *(_auc(column.get("macro", {}).get("f1")) for column in cols)],
        ["Людей принято за ИИ", *(fpr(column) for column in cols)],
    ]
    head = ["На test", "Эта модель", *(f"`{run['name']}`" for run in rel), "LightGBM", "Правила aiw-ru"]
    return md_table(head, rows)


def _cuts_md(metrics: dict, level: str) -> str:
    lightgbm = metrics.get("lightgbm") or {}
    lg_g = _by(lightgbm.get("genres"), "genre")
    lg_l = _by(lightgbm.get("lengths"), "bucket")
    lg_p = _by(lightgbm.get("prompt_types"), "prompt_type")
    lg_m = _by(lightgbm.get("generators"), "model")
    genres = md_table(
        ["Жанр", "Людей", "ИИ", "Accuracy", "ROC AUC", "ROC AUC LightGBM", "ROC AUC правил"],
        [
            [
                f"`{genre['genre']}`",
                count(genre["human"]),
                count(genre["ai"]),
                num(genre["accuracy"]),
                _auc(genre["roc_auc"]),
                _auc(lg_g.get(genre["genre"], {}).get("roc_auc")),
                _auc(genre["detector_roc_auc"]),
            ]
            for genre in metrics["genres"]
        ],
    )
    lengths = md_table(
        ["Слов в тексте", "Текстов", "Accuracy", "ROC AUC", "Accuracy LightGBM", "ROC AUC LightGBM"],
        [
            [
                _bucket(bucket["bucket"]),
                count(bucket["n"]),
                num(bucket["accuracy"]),
                _auc(bucket["roc_auc"]),
                _auc(lg_l.get(bucket["bucket"], {}).get("accuracy")),
                _auc(lg_l.get(bucket["bucket"], {}).get("roc_auc")),
            ]
            for bucket in metrics["lengths"]
        ],
    )
    prompts = md_table(
        ["Задание генератору", "ИИ-текстов", "Recall", "Recall LightGBM"],
        [
            [
                f"`{prompt_type['prompt_type']}`",
                count(prompt_type["n"]),
                num(prompt_type["recall"]),
                _auc(lg_p.get(prompt_type["prompt_type"], {}).get("recall")),
            ]
            for prompt_type in metrics["prompt_types"]
        ],
    )
    generators = md_table(
        ["Модель-генератор", "Текстов в test", "Recall", "Recall LightGBM"],
        [
            [
                f"`{generator['model']}`",
                count(generator["n"]),
                num(generator["recall"]),
                _auc(lg_m.get(generator["model"], {}).get("recall")),
            ]
            for generator in metrics["generators"]
        ],
    )
    return f"""{level} По жанрам

{genres}

{level} По длине текста

Слова считает детектор aiw-ru, как и для LightGBM.

{lengths}

{level} По типу задания генератору

`create` — текст с нуля; `update`, `delete`, `expand` — модель правила,
сокращала или дописывала человеческий текст.

{prompts}

{level} По моделям-генераторам

Самые частые генераторы в test.

{generators}"""


def _long_md(metrics: dict) -> str:
    valid, test = metrics["long_texts"]["valid"], metrics["long_texts"]["test"]
    return md_table(
        ["Правило", "ROC AUC valid", "Accuracy valid", "ROC AUC test", "Accuracy test"],
        [
            [
                label,
                _auc(valid[k]["roc_auc"]),
                num(valid[k]["accuracy"]),
                _auc(test[k]["roc_auc"]),
                num(test[k]["accuracy"]),
            ]
            for k, label in (("head", "начало текста"), ("mean_of_windows", "среднее по окнам"))
        ],
    )


def _latency_md(rows_: list[dict]) -> str:
    latency_ms = {(row["runtime"], row["tokens"], row["threads"]): row["ms"] for row in rows_}
    runtimes = sorted({row["runtime"] for row in rows_})
    threads = sorted({row["threads"] for row in rows_})
    # Длина берётся из замера: если текстов длиннее нет, замер короче LATENCY_LENGTHS.
    tokens = sorted({row["tokens"] for row in rows_})
    head = [
        "Среда",
        *(
            f"{plural(token_count, 'токен', 'токена', 'токенов')}, потоков: {thread_count}"
            for token_count in tokens
            for thread_count in threads
        ),
    ]
    return md_table(
        head,
        [
            [
                runtime,
                *(
                    _ms(latency_ms.get((runtime, token_count, thread_count)))
                    for token_count in tokens
                    for thread_count in threads
                ),
            ]
            for runtime in runtimes
        ],
    )


def _install_md(metrics: dict) -> str:
    return md_table(
        ["Зависимости вывода", "Платформа", "МБ на диске"],
        [[install["stack"], f"`{install['platform']}`", _mb(install["mb"])] for install in metrics["install"]],
    )


def _variants_md(metrics: dict) -> str:
    names = {"torch": "PyTorch на GPU", "onnx_fp32": "ONNX fp32 на CPU", "onnx_int8": "ONNX int8 на CPU"}
    if metrics["params"].get("probs_source") == "torch":
        onnx_texts = metrics["params"]["onnx_texts"]
        return md_table(
            [
                "Вариант",
                f"ROC AUC на {onnx_texts} текстах valid",
                "Accuracy",
                "ROC AUC на всём valid",
                "ROC AUC на test",
            ],
            [
                [
                    names.get(v, v),
                    _num4(variant.get("valid_subset", {}).get("roc_auc")),
                    _auc(variant.get("valid_subset", {}).get("accuracy")),
                    _num4(variant.get("valid", {}).get("roc_auc")),
                    _num4(variant.get("test", {}).get("roc_auc")),
                ]
                for v, variant in metrics["variants"].items()
            ],
        )
    if not all("test" in variant for variant in metrics["variants"].values()):
        table_ = md_table(
            ["Вариант", "ROC AUC valid", "Accuracy valid"],
            [
                [names.get(v, v), _num4(variant["valid"]["roc_auc"]), _num4(variant["valid"]["accuracy"])]
                for v, variant in metrics["variants"].items()
            ],
        )
        return (
            f"{table_}\n\nВарианты сравнивались только на valid, по нему выбираются веса. "
            f"На test посчитан только вариант из поставки, ONNX {_weights(metrics)}."
        )
    return md_table(
        ["Вариант", "ROC AUC valid", "Accuracy valid", "ROC AUC test", "Accuracy test"],
        [
            [
                names.get(v, v),
                *(_num4(variant.get(split, {}).get(k)) for split in ("valid", "test") for k in ("roc_auc", "accuracy")),
            ]
            for v, variant in metrics["variants"].items()
        ],
    )


def _shortcuts_md(metrics: dict) -> str:
    return md_table(
        ["Примета оформления", "ROC AUC одной приметы"],
        [[shortcut["cue"], _auc(shortcut["roc_auc"])] for shortcut in metrics["shortcuts"]],
    )


def _shortcuts_note(metrics: dict) -> str:
    best = max(metrics["shortcuts"], key=lambda shortcut: shortcut["separation"] or 0)
    return (
        f"Поодиночке приметы слабые, сильнее всех «{best['cue']}» с ROC AUC {num(best['roc_auc'])}. "
        "Но модель может сложить их вместе и учиться на оформлении вместо текста."
    )


def _rules_md() -> str:
    # Черта в ячейке таблицы markdown делит столбцы даже внутри `кода`.
    return md_table(
        ["Шаблон", "Замена", "Что убирает"],
        [
            [_code(pat).replace("|", "\\|"), _code(rep) if rep.strip() else "пробел" if rep else "пусто", why]
            for pat, rep, why in NORMALIZE_RULES
        ],
    )


def _ablation(metrics: dict) -> str:
    """Пилот одной базы с нормализацией и без: сколько ROC AUC стоит нормализация."""
    by_base: dict[str, dict[bool, float]] = {}
    for run in metrics["pilot"]:
        by_base.setdefault(run["base"], {})[run["normalize"]] = run["valid_roc_auc"]
    pairs = {
        base_name: auc_by_normalize
        for base_name, auc_by_normalize in by_base.items()
        if True in auc_by_normalize and False in auc_by_normalize
    }
    if not pairs:
        return ""
    base = metrics["params"]["base"] if metrics["params"]["base"] in pairs else next(iter(pairs))
    auc_by_normalize = pairs[base]
    return (
        f"В пилоте база `{base}` без нормализации дала ROC AUC {num(auc_by_normalize[False])} на valid, с нормализацией "
        f"{num(auc_by_normalize[True])}. Нормализация всё равно остаётся: модель, выучившая оформление корпуса, "
        "ошибётся на человеческом тексте, набранном в другом редакторе или скопированном из чата."
    )


def _params_md(metrics: dict) -> str:
    params = metrics["params"]
    stop = ", сработала" if params["early_stopped"] else ", не понадобилась"
    return md_table(
        ["Параметр", "Значение"],
        [
            [
                "База",
                f"[`{params['base']}`](https://huggingface.co/{params['base']}), ревизия `{params['base_revision'][:12]}`",
            ],
            ["Параметров", _params_count(params["parameters"])],
            ["Устройство", f"{params['gpu']}, `{params['device']}`"],
            ["Точность", f"`{params['precision']}`, {params['precision_reason']}"],
            [
                "Эпох",
                f"не больше {count(int(params['epochs']))}, лучшая проверка на эпохе {num(params['best_epoch'], 2)}",
            ],
            [
                "Ранняя остановка",
                f"после {_nw(params['patience'], 'проверки', 'проверок', 'проверок')} без роста ROC AUC на valid{stop}",
            ],
            ["Проверок на valid за эпоху", count(params["evals_per_epoch"])],
            [
                "Оптимизатор",
                f"AdamW, скорость `{params['lr']:g}`, разогрев {pct(params['warmup'])} шагов, линейный спад",
            ],
            ["Затухание весов, клиппинг градиента", f"`{params['weight_decay']:g}`, `{params['grad_clip']:g}`"],
            [
                "Пачка",
                f"{_nw(params['batch'], 'текст', 'текста', 'текстов')} близкой длины"
                + (
                    f", лимит пачки — {_nw(params['batch_tokens'], 'токен', 'токена', 'токенов')}"
                    if params.get("batch_tokens")
                    else ""
                ),
            ],
            ["`max_length`", _nw(params["max_length"], "токен", "токена", "токенов")],
            *(
                [["Локальное окно внимания", "±" + _nw(params["sliding_window"], "токен", "токена", "токенов")]]
                if params.get("sliding_window")
                else []
            ),
            ["`seed`", count(params["seed"])],
            [
                "Время обучения",
                (
                    f"${params['train_seconds'] / 60:.0f}$ мин, "
                    f"{_nw(round(params['texts_per_second']), 'текст', 'текста', 'текстов')} в секунду"
                ),
            ],
            ["Файл модели", f"`model.onnx`, веса {_weights(metrics)}, {_mb(params['size_mb'])} МБ"],
            ["Обучено", f"{metrics['created']}, коммит `{metrics['git_commit']}`"],
        ],
    )


def _curve_md(metrics: dict) -> str:
    return md_table(
        ["Шаг", "Эпоха", "loss train", "ROC AUC valid", "Accuracy valid", "logloss valid"],
        [
            [
                count(point["step"]),
                num(point["epoch"], 2),
                num(point["train_loss"], 4),
                _num4(point["roc_auc"]),
                num(point["accuracy"]),
                num(point["logloss"], 4),
            ]
            for point in metrics["curve"]
        ],
    )


def _speed_md(metrics: dict) -> str:
    x86 = ""
    if metrics.get("latency_x86"):
        x86 = f"""

На двух ядрах виртуальной машины Colab, это ближе к слабому ноутбуку на x86,
{metrics["latency_x86_cpu"]}:

{_latency_md(metrics["latency_x86"])}"""
    slow = any(row["ms"] > 1000 for row in [*metrics["latency"], *(metrics.get("latency_x86") or [])])
    runs = "медиана $30$ прогонов" + (", у вызовов дольше секунды $5$" if slow else "")
    return f"""Задержка на один текст в миллисекундах, {runs}, {metrics["latency_cpu"]}:

{_latency_md(metrics["latency"])}{x86}

Размер зависимостей вывода после установки через uv для Python 3.12:

{_install_md(metrics)}

Размер считается вместе со всеми зависимостями, в том числе numpy и
huggingface-hub, которые aiw-ru с extra `ml` и так ставит для LightGBM.

{_onnx_choice(metrics)}"""


def _onnx_choice(metrics: dict) -> str:
    """Почему в поставке ONNX, а не PyTorch: задержка на этой машине против размера установки."""
    latency_ms = {(row["runtime"], row["tokens"], row["threads"]): row["ms"] for row in metrics["latency"]}
    torch_ms, onnx_ms = latency_ms.get(("torch fp32", 512, 1)), _shipped_ms(metrics)
    sizes = {(install["stack"], install["platform"]): install["mb"] for install in metrics["install"] if install["mb"]}
    onnx = {plat: size_mb for (stack, plat), size_mb in sizes.items() if stack.startswith("onnxruntime")}
    ratios = [
        size_mb / onnx[plat]
        for (stack, plat), size_mb in sizes.items()
        if stack == "torch + transformers" and plat in onnx and "linux" not in plat
    ]
    cpu = sizes.get(("torch (CPU) + transformers", "x86_64-manylinux_2_28"))
    if cpu and "x86_64-manylinux_2_28" in onnx:
        ratios.append(cpu / onnx["x86_64-manylinux_2_28"])
    cuda = sizes.get(("torch + transformers", "x86_64-manylinux_2_28"))
    if not (torch_ms and onnx_ms and ratios):
        return ""
    v = metrics["variants"]
    shipped, ref = v.get(metrics["params"]["shipped"], {}).get("test", {}), v.get("torch", {}).get("test", {})
    same = "точность та же"
    if shipped.get("roc_auc") is not None and ref.get("roc_auc") is not None:
        diff = abs(shipped["roc_auc"] - ref["roc_auc"])
        same = (
            "ROC AUC на test совпадает с PyTorch до четвёртого знака"
            if diff < 5e-5
            else f"ROC AUC на test отличается от PyTorch на {_num4(diff)}"
        )
    faster = "быстрее" if torch_ms < onnx_ms else "медленнее"
    span = f"${min(ratios):.1f}\\text{{–}}{max(ratios):.1f}$"
    linux = f", а на Linux со сборкой torch под CUDA по умолчанию — {_mb(cuda)} МБ" if cuda else ""
    return (
        f"На этой машине torch считает {faster} onnxruntime: {_ms(torch_ms)} мс против {_ms(onnx_ms)} мс "
        f"у ONNX {_weights(metrics)} "
        f"на $512$ токенах в один поток. Но torch с transformers занимает на диске в {span} раза больше{linux}. "
        f"aiw-ru ставят и на слабые ноутбуки, поэтому в поставке ONNX: установка лёгкая, а {same}."
    )


def _limitations_md(metrics: dict) -> str:
    fpr = 1 - metrics["test"]["classes"]["human"]["recall"]
    return f"""- Корпус один. На научных статьях, дипломах и диссертациях модель не
  проверялась, а жанры за пределами таблицы по жанрам она не видела.
- Вероятность — не доказательство авторства. Не используйте модель для
  решений о людях: на test она принимает за ИИ {pct(fpr)} человеческих текстов.
- Правку человеческого текста моделью распознать труднее, чем текст с нуля,
  см. таблицу по типу задания.
- Нормализация убирает оформление, но не длину: короткие тексты модель
  различает хуже, см. таблицу по длине.
- В корпусе генераторы до 2025 года; тексты новых моделей могут
  распознаваться хуже."""


def _usage_md(metrics: dict, repo: str) -> str:
    bundle = bundle_for(metrics["params"]["base"])
    return f"""Из командной строки:

```bash
uv tool install "aiw-ru[ml]"
aiw-ru models install {bundle.name}
{_classify_cmd(bundle)}
```

Из Python без aiw-ru нужны только onnxruntime, tokenizers и numpy:

```python
import json
import os
import re

# Без этой строки onnxruntime 1.30 при импорте включает телеметрию Microsoft,
# и процесс изредка падает на выходе (recursive_mutex lock failed, код 134).
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

import numpy as np
import onnxruntime as ort
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer

files = ["inference.json", "model.onnx*", "tokenizer.json"]
path = snapshot_download("{repo}", allow_patterns=files)
spec = json.load(open(f"{{path}}/inference.json", encoding="utf-8"))
session = ort.InferenceSession(f"{{path}}/{{spec['file']}}", providers=["CPUExecutionProvider"])
tokenizer = Tokenizer.from_file(f"{{path}}/{{spec['tokenizer']}}")
tokenizer.no_truncation()
tokenizer.no_padding()


def probability(text: str) -> float:
    \"\"\"Вероятность, что текст написала языковая модель.\"\"\"
    text = text[: spec["max_chars"]]
    for rule in spec["normalize"]:
        text = re.sub(rule["pattern"], rule["replacement"], text, flags=re.MULTILINE)
    ids = tokenizer.encode(text.strip(), add_special_tokens=False).ids
    prefix, suffix = [spec["cls_id"]], [spec["sep_id"]]
    width = spec.get("read_length", spec["max_length"]) - len(prefix) - len(suffix)
    windows = [ids[i : i + width] for i in range(0, max(len(ids), 1), width)][: spec["max_windows"]]
    windows = [[*prefix, *w, *suffix] for w in windows]
    batch = np.full((len(windows), max(map(len, windows))), spec["pad_id"], dtype=np.int64)
    mask = np.zeros_like(batch)
    for k, w in enumerate(windows):
        batch[k, : len(w)] = w
        mask[k, : len(w)] = 1
    logits = session.run([spec["output"]], {{"input_ids": batch, "attention_mask": mask}})[0]
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    return float((p[:, spec["labels"].index("ai")] / p.sum(axis=1)).mean())


print(probability(open("текст.md", encoding="utf-8").read()))
```

`inference.json` описывает вывод целиком: файл ONNX, токенизатор,
`max_length`, токены вокруг окна, правила нормализации из обучения, порог
{num(metrics["inference"]["threshold"], 1)} и число окон для длинных текстов.{_pytorch_md(metrics, repo)}"""


def _pytorch_md(metrics: dict, repo: str) -> str:
    """Как взять веса PyTorch из репозитория: для дообучения и своего экспорта."""
    if not metrics["sizes_mb"].get("safetensors"):
        return ""
    return f"""

Для дообучения и своего экспорта в репозитории есть веса PyTorch, `model.safetensors`.
aiw-ru и пример выше их не скачивают. Их читает transformers:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("{repo}")
tokenizer = AutoTokenizer.from_pretrained("{repo}")
```

Текст перед токенизатором нормализуй по правилам `normalize` из `inference.json`, как в
примере выше: на нём модель и обучалась."""


def _int8_changes(metrics: dict) -> float | None:
    changes = metrics["params"].get("int8_label_changes") or {}
    return changes.get("valid", changes.get("valid_subset"))


def _extra_weights(metrics: dict) -> tuple[str, str]:
    """Другие файлы весов рядом с model.onnx: по-русски и по-английски."""
    if _weights(metrics) == "int8":
        fp32 = _mb(metrics["sizes_mb"]["onnx_fp32"])
        text_ru, text_en = (
            f" Рядом лежит `model_fp32.onnx` на {fp32} МБ с весами fp32, aiw-ru его не скачивает.",
            f" An fp32 export, `model_fp32.onnx`, is an optional extra file of {fp32} MB.",
        )
    else:
        int8, changes = _mb(metrics["sizes_mb"]["onnx_int8"]), _int8_changes(metrics)
        why_ru = f": int8 меняет ответ у {pct(changes)} текстов valid" if changes is not None else ""
        why_en = f" because int8 flips the label on {pct(changes)} of validation texts" if changes is not None else ""
        text_ru, text_en = (
            f" Рядом лежит `model_int8.onnx` на {int8} МБ с весами int8, aiw-ru его не скачивает{why_ru}.",
            f" An int8 export, `model_int8.onnx`, is an optional extra file of {int8} MB; aiw-ru uses fp32{why_en}.",
        )
    if pytorch := metrics["sizes_mb"].get("safetensors"):
        text_ru += f" Веса PyTorch для дообучения и своего экспорта — `model.safetensors` на {_mb(pytorch)} МБ."
        text_en += f" PyTorch weights for fine-tuning or your own export are in `model.safetensors`, {_mb(pytorch)} MB."
    return text_ru, text_en


def _probs_note(metrics: dict) -> str:
    params = metrics["params"]
    if params.get("probs_source") != "torch":
        return ""
    return (
        f" Итоговые цифры этой модели посчитаны по эмбеддингам с GPU ({params['precision']}): ONNX на CPU по "
        f"всем текстам valid и test занял бы много часов. ONNX сверен с ними на {_texts_loc(params['onnx_texts'])} "
        "valid разной длины, расхождение — ниже."
    )


def _weights_rule(metrics: dict) -> str:
    params = metrics["params"]
    if params.get("weights_choice", "auto") == "auto":
        changes = params.get("int8_max_changes")
        tail = f" и меняет ответ не больше чем у {pct(changes)} текстов" if changes is not None else ""
        verdict = f" Здесь в поставке {_weights(metrics)}"
        label_changes = _int8_changes(metrics)
        verdict += f": int8 меняет ответ у {pct(label_changes)} текстов valid." if label_changes is not None else "."
        return (
            f"int8 идёт в поставку, если на valid теряет не больше {num(params.get('int8_max_drop', 0.002))} ROC AUC{tail}; "
            f"test в этом решении не участвует.{verdict}"
        )
    label_changes = _int8_changes(metrics)
    tail = (
        f": int8 меняет ответ у {pct(label_changes)} текстов valid по сравнению с fp32"
        if label_changes is not None
        else ""
    )
    return f"Веса {_weights(metrics)} выбраны заранее, эту модель выпускают ради точности{tail}."


def _long_rule(metrics: dict) -> str:
    inference, longer_than = metrics["inference"], _tokens_gen(metrics["params"]["max_length"])
    if inference["long_texts"] == "mean_of_windows":
        return (
            f"Текст длиннее {longer_than} режется на окна по столько же токенов, вероятность — среднее "
            f"по первым {count(inference['max_windows'])} окнам: на valid так точнее, чем по началу текста."
        )
    fragments = (
        f" `aiw-ru classify` вдобавок проверяет длинный текст фрагментами по {_tokens_gen(inference['max_length'])}: "
        "на окне целиком ИИ-вставка тонет в человеческом тексте."
        if inference.get("read_length")
        else ""
    )
    return f"Текст длиннее {longer_than} модель читает только до этой границы: среднее по окнам на valid не точнее.{fragments}"


def _trained_on(metrics: dict) -> str:
    dataset = metrics["dataset"]
    return (
        f"Модель обучена на {_texts_loc(dataset['train'])} из train, лучший шаг и ранняя остановка — по ROC AUC "
        f"на {_texts_loc(dataset['valid'])} из valid."
    )


def _what_ru(metrics: dict) -> str:
    params = metrics["params"]
    base = f"[`{params['base']}`](https://huggingface.co/{params['base']})"
    return f"Это дообученный энкодер {base} на\n{_params_count(params['parameters'])} параметров"


def _what_en(metrics: dict) -> str:
    params = metrics["params"]
    return f"The model is `{params['base']}` fine-tuned"


def card_body(metrics: dict, repo: str) -> str:
    """Карточка модели на Hugging Face: формулы в $…$, для Hub их переписывает hub_math."""
    params, dataset, test, valid = metrics["params"], metrics["dataset"], metrics["test"], metrics["valid"]
    lightgbm_test = (metrics.get("lightgbm") or {}).get("test", {})
    name = repo.split("/")[-1]
    report = f"{GITHUB}/blob/master/{DOCS.as_posix()}/{name}.md"
    lgb_line = f" У LightGBM на признаках aiw-ru ROC AUC {num(lightgbm_test['roc_auc'])}." if lightgbm_test else ""
    extra_ru, extra_en = _extra_weights(metrics)
    files_ru, files_en = (
        (
            "Модель — `model.onnx` с весами в `model.onnx.data`, вместе",
            "`model.onnx` with its weights in `model.onnx.data` takes",
        )
        if params.get("external_data")
        else ("Файл `model.onnx` занимает", "`model.onnx` takes")
    )
    bundle = bundle_for(params["base"])
    install = (
        f"Ставит её `aiw-ru models install` без имени, вместе с LightGBM, или `aiw-ru models install {bundle.name}`, "
        "и только если пользователь попросит."
        if bundle.default
        else f"Ставится она только по имени, командой `aiw-ru models install {bundle.name}`, и только если "
        f"пользователь попросит; в `aiw-ru classify` её выбирают ключом `--model {bundle.name}`."
    )
    others = [f"{_hub_link(run['repo'])} (`{run['name']}`)" for run in metrics.get("related", [])]
    others.append(f"[LightGBM на признаках aiw-ru](https://huggingface.co/{LIGHTGBM_REPO}) (`lightgbm`)")
    choose = ""
    if metrics.get("related"):
        choose = f"\n\n## Какую модель выбрать\n\n{_choose_md(metrics, repo)}"
    return f"""# Детектор ИИ-текста для русского языка

`{name}` оценивает вероятность, что русский текст написала нейросеть, а не
человек. Обучен на текстах {GENERATORS} и других языковых моделей из корпуса
LLMTrace. Помогает проверить текст на ИИ: статью, новость, отзыв, пост, ответ
на вопрос.

{_what_ru(metrics)}, переведённый в ONNX с весами {_weights(metrics)}.
{files_ru} {_mb(params["size_mb"])} МБ и работает на CPU через onnxruntime, без
torch и GPU.{extra_ru} На отложенной части корпуса ROC AUC {num(test["roc_auc"])}, accuracy
{num(test["accuracy"])}.{lgb_line}

Модель необязательная: детектор и скиллы [aiw-ru]({GITHUB}) работают без неё. {install}
Другие модели aiw-ru: {", ".join(others)}.

## In English

A Russian AI-generated text detector. It estimates the probability that a
Russian text was written by a person or by an LLM such as {GENERATORS}
and others, which makes it usable as an AI text detector, a ChatGPT detector or an
"AI slop" filter for Russian. {_what_en(metrics)}, trained on the
Russian part of the LLMTrace corpus and exported to ONNX with {_weights(metrics)} weights:
{files_en} {_mb(params["size_mb"])} MB and runs on CPU with onnxruntime, no torch
or GPU needed.{extra_en}
Test ROC AUC {num(test["roc_auc"])}, accuracy {num(test["accuracy"])}. Install it with
`aiw-ru models install {bundle.name}`. The rest of the card is in Russian; usage is below.{choose}

## Данные

Русская часть [LLMTrace classification](https://huggingface.co/datasets/{DATASET})
([статья](https://arxiv.org/abs/2509.21269)), ревизия `{dataset["revision"][:12]}`. {_trained_on(metrics)}
Итоговые цифры посчитаны по test, это {_nw(dataset["test"], "текст", "текста", "текстов")}: эту часть
корпуса модель при обучении не видела. Отчёт об обучении с выбором базы и командами для
воспроизведения лежит [на GitHub]({report}).

## Результаты на test

Текст считается написанным ИИ, если вероятность не меньше {num(THRESHOLD, 1)}. {_long_rule(metrics)}

{_classes_table(test)}

Сравнение с LightGBM на признаках aiw-ru и с оценкой правил детектора с порогом
{count(metrics["detector"]["threshold"])} на том же test:

{_compare_md(metrics)}

На valid accuracy {num(valid["accuracy"])}, ROC AUC {num(valid["roc_auc"])}.

{_cuts_md(metrics, "###")}

### Длинные тексты

Тексты длиннее {_tokens_gen(params["max_length"])}, в test их {count(metrics["long_texts"]["test"]["texts"])}.
Сравниваются вероятность по началу текста и среднее по окнам:

{_long_md(metrics)}

## Нормализация

В LLMTrace оформление говорит о том, откуда взят человеческий текст: переводы
строк, markdown, вид тире и кавычек, ё. Поэтому до токенизации текст проходит
нормализацию: разметка и типографика сводятся к простому виду, правила лежат в
`inference.json`. {_ablation(metrics)}

## Скорость и размер

{_speed_md(metrics)}

## Как пользоваться

{_usage_md(metrics, repo)}

## Обучение

{_params_md(metrics)}

## Ограничения

{_limitations_md(metrics)}

## Лицензия и данные

Модель распространяется по лицензии {LICENSE_NAMES.get(params["license"], params["license"])}, как и
[`{params["base"]}`](https://huggingface.co/{params["base"]}). Корпус LLMTrace — Apache 2.0, человеческие тексты в
нём собраны из сторонних источников со своими лицензиями.

{LLMTRACE_BIBTEX}

{BASE_CITATIONS.get(params["base"], "")}
"""


def _reproduce_md(metrics: dict) -> str:
    params = metrics["params"]
    run = short_name(params["base"])
    bundle = bundle_for(params["base"])
    vm_run = f"/content/data/transformer/runs/{run}"
    full = "~/.cache/aiw-ru/llmtrace/transformer/full"
    report_args = metrics.get("report_args", {})
    local = f"{full}/{report_args.get('run') or run}"
    # Colab даёт T4; на своей видеокарте обучение, экспорт и оценка идут на одной машине.
    colab = "T4" in (params.get("gpu") or "")
    eval_flags = "" if params.get("weights_choice", "auto") == "auto" else f" --weights {params['weights_choice']}"
    if params.get("int8_valid_only"):
        eval_flags += " --int8-valid-only"
    report_flags = "".join(
        [
            *(f" --finalist {full}/{folder}" for folder in report_args.get("finalist", [])),
            *(f" --related {full}/{folder}" for folder in report_args.get("related", [])),
        ]
    )
    why_note = (
        " Абзац о выборе базы передаётся ключом `--why`, его текст приведён в этом отчёте." if params.get("why") else ""
    )
    train_args = (
        f"train --base {params['base']} --epochs {params['epochs']:g} --lr {params['lr']:g} --batch {params['batch']} "
        f"--max-length {params['max_length']} --evals-per-epoch {params['evals_per_epoch']} --patience {params['patience']}"
        + (f" --batch-tokens {params['batch_tokens']}" if params.get("batch_tokens") else "")
        + (f" --sliding-window {params['sliding_window_arg']}" if params.get("sliding_window_arg") else "")
        + (f" --seed {params['seed']}" if params.get("seed", SEED) != SEED else "")
    )
    if params.get("probs_source") == "torch":
        eval_flags += f" --onnx-texts {params['onnx_texts']}"
    uv_run = "uv run --group train --group transformer"
    dirty = metrics["git_commit"].endswith("+правки")
    commit = metrics["git_commit"].removesuffix("+правки")
    checkout = (
        f"Модель обучена на коммите `{commit}`, к которому `scripts/train_transformer.py` и "
        "`scripts/colab_transformer.py` ещё не были закоммичены. Для повтора берите их версию из коммита, "
        "в котором появился этот отчёт."
        if dirty
        else f"Модель обучена на коммите `{commit}`."
    )
    env = md_table(["Компонент", "Версия"], [[k, f"`{version}`"] for k, version in metrics["environment"].items()])
    head = f"""{checkout} Базовая модель — ревизия `{params["base_revision"]}`, корпус — ревизия
`{metrics["dataset"]["revision"]}`, её закрепляет `scripts/llmtrace.py`. Окружение:

{env}"""
    tail = """Обучение на GPU не детерминировано до бита, повтор может разойтись в третьем
знаке. Пилот повторяется командой `pilot --base ИМЯ` с настройками по умолчанию
(`--limit 20000 --epochs 1`)."""
    if not colab:
        return f"""{head}

Модель обучена на {params.get("gpu") or "GPU"}, `{params["precision"]}`. На той же машине считались ONNX,
сверка с PyTorch и оценка на valid и test. Задержку в карточке мерил `export`
на M1 в копии папки запуска с ключом `--json export-latency.json`: файл
кладётся в папку запуска, и `report` берёт задержку из него.{why_note}

```bash
uv run --group train scripts/llmtrace.py fetch --set classification --split train
uv run --group train scripts/llmtrace.py fetch --set classification --split valid
uv run --group train scripts/llmtrace.py fetch --set classification --split test
{uv_run} scripts/train_transformer.py {train_args} --out {local}
{uv_run} scripts/train_transformer.py export {local}
{uv_run} scripts/train_transformer.py evaluate {local}{eval_flags}
{uv_run} scripts/train_transformer.py report {local}{report_flags}
```

{tail}"""
    return f"""{head}

Обучение идёт на GPU. В Colab на бесплатной T4 это делает
`scripts/colab_transformer.py` через официальный Colab CLI (google-colab-cli,
`colab sessions` должен работать). Корпус скачивается прямо на VM, через ваш
компьютер идут только исходники и готовая модель:

```bash
uv run scripts/colab_transformer.py up --gpu T4
uv run scripts/colab_transformer.py setup
uv run scripts/colab_transformer.py start --name {bundle.log} --job "{train_args}"
uv run scripts/colab_transformer.py status --name {bundle.log}
uv run scripts/colab_transformer.py fetch {vm_run} {full}
```

Colab CLI через час после `up` может счесть VM потерянной, когда у него истекает
токен прокси, и перестать её поддерживать; тогда VM отбирают. Поэтому перед
каждым вызовом CLI токен обновлялся из ответа Colab, а лучший шаг забирался
на этот компьютер после каждой проверки на valid.

Задержка на x86 меряется на CPU той же VM, после обучения, пока GPU свободен:

```bash
uv run scripts/colab_transformer.py start --name x86 --job "export {vm_run} --json export-x86.json"
uv run scripts/colab_transformer.py fetch {vm_run}/export-x86.json {local}
uv run scripts/colab_transformer.py down
```

Локально на CUDA или MPS вместо Colab:

```bash
uv run --group train scripts/llmtrace.py fetch --set classification --split train
uv run --group train scripts/llmtrace.py fetch --set classification --split valid
uv run --group train scripts/llmtrace.py fetch --set classification --split test
{uv_run} scripts/train_transformer.py {train_args.replace('" --job "', f"\n{uv_run} scripts/train_transformer.py ").replace(vm_run, local)}
```

Дальше на своём CPU: ONNX и сверка с PyTorch, задержка, оценка на valid и test,
карточка и этот отчёт.{why_note}

```bash
{uv_run} scripts/train_transformer.py export {local}
{uv_run} scripts/train_transformer.py evaluate {local}{eval_flags}
{uv_run} scripts/train_transformer.py report {local}{report_flags}
```

{tail}"""


def _spec_parity(metrics: dict) -> str:
    parity = metrics["parity"].get("spec_vs_eval")
    if not parity:
        return ""
    return (
        f" Вывод по `inference.json`, текст за текстом, как в aiw-ru, расходится с итоговыми вероятностями "
        f"оценки на {_texts_loc(parity['texts'])} valid не больше чем на {_sci(parity['max_abs_diff'])}."
        if parity["max_abs_diff"] > 0
        else f" Вывод по `inference.json`, текст за текстом, как в aiw-ru, на {_texts_loc(parity['texts'])} valid "
        "даёт те же вероятности, что оценка."
    )


def report_body(metrics: dict, repo: str) -> str:
    """Отчёт об обучении для docs/models/: что сделано, на чём, как повторить."""
    params, dataset = metrics["params"], metrics["dataset"]
    name = repo.split("/")[-1]
    lightgbm_auc = (metrics.get("lightgbm") or {}).get("test", {}).get("roc_auc")
    pilot = metrics["pilot"][0] if metrics["pilot"] else {}
    finalists = ""
    if metrics.get("finalists"):
        finalists = f"""

{_finalists_intro(metrics)}

ROC AUC и accuracy посчитаны на полном valid у PyTorch, «ROC AUC int8» — у ONNX int8;
«сменил метку» — доля текстов valid, где int8 и PyTorch расходятся по порогу $0.5$.
Размер и задержка — у варианта из поставки, на $512$ токенах в один поток: на M1 и на
двух ядрах Xeon виртуальной машины Colab.

{_finalists_md(metrics)}"""
    why = f"\n\n{params['why']}" if params.get("why") else ""
    parity = metrics["parity"]
    bundle = bundle_for(params["base"])
    choose = ""
    if metrics.get("related"):
        choose = f"\n\n{_choose_md(metrics, repo)}"
    what = f"дообученный энкодер `{params['base']}`"
    sizes = [run["parameters"] / 1e6 for run in metrics["pilot"] if run.get("parameters")]
    smallest, largest = (round(min(sizes)), round(max(sizes))) if sizes else (0, 0)
    span = (
        ""
        if not sizes
        else f", а у кандидатов ниже ${smallest}$ млн"
        if smallest == largest
        else f", а у кандидатов ниже ${smallest}\\text{{–}}{largest}$ млн"
    )
    return f"""# Отчёт об обучении: {name}

Файл пишет `scripts/train_transformer.py report` после полного обучения, руками
его не правят. Все числа в машиночитаемом виде лежат рядом, в
[{name}.json]({name}.json). Модель и карточка выложены на
[Hugging Face](https://huggingface.co/{repo}).

## Что это

Необязательная модель aiw-ru: {what} оценивает
вероятность, что русский текст написала языковая модель. Пользователь ставит её
командой `aiw-ru models install {bundle.name}`, после чего `{_classify_cmd(bundle)}`
выдаёт вероятность. В поставке ONNX с весами {_weights(metrics)} на {_mb(params["size_mb"])} МБ и
`inference.json`; для вывода нужны onnxruntime и tokenizers, torch не нужен.
На test ROC AUC {num(metrics["test"]["roc_auc"])}, у LightGBM на признаках aiw-ru {_auc(lightgbm_auc)},
у оценки правил {num(metrics["detector"]["roc_auc"])}.{choose}

## Выбор базы

Кандидаты — небольшие русские энкодеры с открытой лицензией, которые можно
запускать на CPU. Полный [FRIDA](https://huggingface.co/ai-forever/FRIDA)
от ai-forever (MIT) — энкодер T5 на $823$ млн параметров{span}. В пилоте вместо
FRIDA взята его дистилляция `sergeyzh/rubert-mini-frida`.

В пилоте все базы учились одинаково: {_nw(pilot.get("train", 0), "текст", "текста", "текстов")} train,
эпох {count(int(pilot.get("epochs") or 1))}, скорость `{pilot.get("lr") or 0:g}`, пачка {count(pilot.get("batch") or 0)},
`seed` {count(pilot.get("seed") or 0)}. ROC AUC посчитан на {_texts_loc(pilot.get("valid", 0))} valid,
задержка — ONNX int8 на $512$ токенах в один поток, {metrics["latency_cpu"]}.

{_pilot_md(metrics)}{finalists}{why}

## Данные

Русская часть [LLMTrace classification](https://huggingface.co/datasets/{dataset["repo"]})
под Apache 2.0, [статья](https://arxiv.org/abs/2509.21269), ревизия набора
`{dataset["revision"]}`. Части корпуса используются как есть. {_trained_on(metrics)} Test нужен только для
итоговых цифр, в нём {_nw(dataset["test"], "текст", "текста", "текстов")}.

Метка «ИИ» стоит на любом тексте с участием модели: написанном с нуля
(`create`) и на человеческом тексте, который модель правила, сокращала или
дописывала (`update`, `delete`, `expand`). При обучении от текста длиннее
{_tokens_gen(params["max_length"])} остаётся только начало. Таких в train
{pct(params["truncated_share"]["train"])}, в valid {pct(params["truncated_share"]["valid"])}, в test
{pct(params["truncated_share"].get("test", 0))}.

## Нормализация

Оформление в LLMTrace говорит о том, откуда взят человеческий текст. Ниже
ROC AUC каждой приметы по отдельности на valid: $0.5$ значит, что примета ничего
не говорит, чем дальше от $0.5$ в любую сторону, тем больше говорит.

{_shortcuts_md(metrics)}

{_shortcuts_note(metrics)} Поэтому до токенизации текст проходит `normalize()`: правила ниже применяются
по порядку, шаблоны — `re` с флагом `MULTILINE`, в конце пробелы по краям
срезаются. Тот же список лежит в `inference.json`. Длину текста так не убрать,
см. таблицу по длине. {_ablation(metrics)}

{_rules_md()}

## Обучение

{_params_md(metrics)}

Проверки на valid по ходу обучения:

{_curve_md(metrics)}

## Результаты на test

Текст считается написанным ИИ, если вероятность не меньше {num(THRESHOLD, 1)}. {_long_rule(metrics)}

{_classes_table(metrics["test"])}

{_compare_md(metrics)}

На valid accuracy {num(metrics["valid"]["accuracy"])}, ROC AUC {num(metrics["valid"]["roc_auc"])}.

{_cuts_md(metrics, "###")}

### Детектор без модели

Оценка правил aiw-ru с порогом {count(metrics["detector"]["threshold"])}, то есть «много примет»:

{_classes_table(metrics["detector"])}

## Вывод на CPU

Ниже варианты одной и той же модели. {_weights_rule(metrics)}{_probs_note(metrics)}

{_variants_md(metrics)}

Сверка с PyTorch на {_texts_loc(parity["fp32"]["texts"])} valid: у ONNX fp32 наибольшая
разница вероятностей {_sci(parity["fp32"]["max_abs_diff"])}, у int8 {num(parity["int8"]["max_abs_diff"])},
метки int8 совпадают у {pct(parity["int8"]["same_label"])} текстов.{_spec_parity(metrics)}

Правило для длинных текстов выбрано по valid, в test таких текстов
{count(metrics["long_texts"]["test"]["texts"])}:

{_long_md(metrics)}

{_speed_md(metrics)}

## Как пользоваться

{_usage_md(metrics, repo)}

## Как воспроизвести

{_reproduce_md(metrics)}

## Ограничения

{_limitations_md(metrics)}

## Ссылки

{LLMTRACE_BIBTEX}

{BASE_CITATIONS.get(params["base"], "")}
"""


def card(metrics: dict, repo: str | None = None) -> ModelCard:
    repo = repo or bundle_for(metrics["params"]["base"]).repo
    params = metrics["params"]
    results = eval_results(metrics)
    for result in results:
        result.source_name = "aiw-ru scripts/train_transformer.py"
        result.source_url = f"{GITHUB}/blob/master/scripts/train_transformer.py"
    tags = [tag for tag in TAGS if tag not in ("lightgbm", "tabular-features")]
    arch = "modernbert" if "modernbert" in params["base"].lower() else "bert"
    tags += [arch, "onnx"]
    if _weights(metrics) == "int8":
        tags.append("int8")
    data = ModelCardData(
        language="ru",
        license=params["license"],
        library_name="onnx",
        pipeline_tag=TASK,
        base_model=params["base"],
        base_model_relation="finetune",
        tags=tags,
        datasets=[DATASET],
        metrics=["accuracy", "f1", "precision", "recall", "roc_auc"],
        model_name=repo.split("/")[-1],
        eval_results=results,
        inference=False,
    )
    return ModelCard(f"---\n{data.to_yaml()}\n---\n\n{hub_math(card_body(metrics, repo))}")


def finalist(run_dir: Path, data: Path) -> dict:
    """Строка сравнения полных запусков: полный valid у PyTorch и int8, размер и задержка int8.

    Ответы int8 на valid берутся из valid-probs-onnx_int8.npy, который пишет evaluate;
    задержка на x86 — из export-x86.json, если он есть.
    """
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    export_info = json.loads((run_dir / "export.json").read_text(encoding="utf-8"))
    latency_ms = {
        (entry["runtime"], entry["tokens"], entry["threads"]): entry["ms"] for entry in export_info["latency"]
    }
    row: dict[str, Any] = {
        "base": run["params"]["base"],
        "valid": run["dataset"]["valid"],
        "valid_roc_auc": run["valid"]["roc_auc"],
        "valid_accuracy": run["valid"]["accuracy"],
        "onnx_fp32_mb": export_info["sizes_mb"]["onnx_fp32"],
        "onnx_int8_mb": export_info["sizes_mb"]["onnx_int8"],
        "int8_ms_128_1": latency_ms.get(("onnxruntime int8", 128, 1)),
        "int8_ms_512_1": latency_ms.get(("onnxruntime int8", 512, 1)),
        "fp32_ms_512_1": latency_ms.get(("onnxruntime fp32", 512, 1)),
    }
    # Веса в поставке — из metrics.json после evaluate; до него считаем, что int8.
    metrics = run_dir / "metrics.json"
    weights = (
        json.loads(metrics.read_text(encoding="utf-8"))["params"]["shipped"].removeprefix("onnx_")
        if metrics.exists()
        else "int8"
    )
    row |= {
        "weights": weights,
        "shipped_mb": export_info["sizes_mb"][f"onnx_{weights}"],
        "shipped_ms_512_1": latency_ms.get((f"onnxruntime {weights}", 512, 1)),
    }
    x86 = run_dir / "export-x86.json"
    if x86.exists():
        x86_latency_ms = {
            (entry["runtime"], entry["tokens"], entry["threads"]): entry["ms"]
            for entry in json.loads(x86.read_text())["latency"]
        }
        row["x86_int8_ms_512_1"] = x86_latency_ms.get(("onnxruntime int8", 512, 1))
        row["x86_shipped_ms_512_1"] = x86_latency_ms.get((f"onnxruntime {weights}", 512, 1))
    int8 = run_dir / "valid-probs-onnx_int8.npy"
    if int8.exists():
        labels = np.array([json.loads(line)["label"] == "ai" for line in load(data, "classification", "valid", 0)])
        int8_probs, torch_probs = np.load(int8), np.load(run_dir / "valid-probs.npy")
        row |= {
            "valid_roc_auc_int8": auc(labels, int8_probs),
            "valid_accuracy_int8": float((labels == (int8_probs >= THRESHOLD)).mean()),
            "int8_label_changes": float(((int8_probs >= THRESHOLD) != (torch_probs >= THRESHOLD)).mean()),
        }
    return row


def cmd_report(args: argparse.Namespace) -> None:
    """README.md в hub/ рядом с файлами модели, отчёт и метрики в docs/models/."""
    run_dir: Path = args.folder
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    repo = args.repo or bundle_for(metrics["params"]["base"]).repo
    # Замеры задержки могли появиться после evaluate, метрики LightGBM — обновиться.
    metrics |= cpu_latency(run_dir, metrics) | x86_latency(run_dir)
    metrics["lightgbm"] = lightgbm_block() or metrics.get("lightgbm")
    if args.why:
        metrics["params"]["why"] = args.why
    if args.finalist:
        metrics["finalists"] = [finalist(folder, args.data) for folder in args.finalist]
    if args.related:
        metrics["related"] = [related_row(folder) for folder in args.related]
    metrics["report_args"] = {
        "run": run_dir.name,
        "finalist": [folder.name for folder in args.finalist or []],
        "related": [folder.name for folder in args.related or []],
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    hub_dir = bundle_hub(run_dir, metrics["params"]["shipped_file"])
    card(metrics, repo).save(hub_dir / "README.md")
    print(f"Для Hugging Face: {hub_dir}", flush=True)
    name = repo.split("/")[-1]
    if metrics["dataset"]["limit"]:
        # Пилот на выборке в docs/ не идёт: отчёт только в папке запуска.
        (run_dir / "report.md").write_text(report_body(metrics, repo), encoding="utf-8")
        print(f"Обучение на выборке: отчёт только в {run_dir / 'report.md'}", flush=True)
        return
    docs = ROOT / DOCS
    docs.mkdir(parents=True, exist_ok=True)
    (docs / f"{name}.md").write_text(report_body(metrics, repo), encoding="utf-8")
    (docs / f"{name}.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Отчёт: {DOCS / f'{name}.md'} и {DOCS / f'{name}.json'}", flush=True)


# ─── Командная строка ───────────────────────────────────────────────────


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", type=Path, default=data_dir(), help="папка с частями корпуса")
    parser.add_argument(
        "--out-root", type=Path, help="папка запусков и кэша (по умолчанию transformer/ рядом с данными)"
    )


def add_fit(parser: argparse.ArgumentParser, pilot: bool) -> None:
    add_common(parser)
    parser.add_argument("--base", default=DEFAULT_BASE, help="исходная модель на Hugging Face")
    parser.add_argument("--limit", type=int, default=20000 if pilot else 0, help="взять N текстов из train")
    parser.add_argument(
        "--eval-limit", type=int, help="взять N текстов из valid и test (по умолчанию половина --limit)"
    )
    parser.add_argument("--epochs", type=float, default=1 if pilot else 3, help="эпох обучения, можно дробное число")
    parser.add_argument("--lr", type=float, default=1e-4, help="пиковая скорость обучения AdamW")
    parser.add_argument("--batch", type=int, default=32, help="текстов в пачке")
    parser.add_argument("--eval-batch", type=int, default=128, help="текстов в пачке при проверке")
    parser.add_argument(
        "--batch-tokens", type=int, default=0, help="токенов в пачке, ширина × тексты (0 — без лимита; нужен при 8192)"
    )
    parser.add_argument(
        "--sliding-window",
        type=int,
        default=0,
        help="полуширина локального окна ModernBERT (0 — как в config.json базы)",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="сид: порядок пачек и начальные веса головы")
    parser.add_argument("--max-length", type=int, default=512, help="токенов на текст, дальше текст обрезается")
    parser.add_argument("--warmup", type=float, default=0.06, help="доля шагов разогрева скорости обучения")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="затухание весов AdamW")
    parser.add_argument("--patience", type=int, default=0 if pilot else 3, help="проверок без роста AUC до остановки")
    parser.add_argument("--evals-per-epoch", type=int, default=1 if pilot else 4, help="проверок на valid за эпоху")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"], help="где учить")
    parser.add_argument("--precision", default="auto", choices=["auto", "fp32", "fp16", "bf16"], help="точность")
    parser.add_argument("--raw", action="store_true", help="без нормализации текста, для сравнения")
    parser.add_argument("--out", type=Path, help="папка запуска")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    add_fit(sub.add_parser("pilot", help="короткое обучение на выборке, без test"), pilot=True)
    add_fit(sub.add_parser("train", help="полное обучение и ответы PyTorch на test"), pilot=False)
    export_parser = sub.add_parser("export", help="ONNX fp32 и int8, сверка с PyTorch, задержка на CPU")
    add_common(export_parser)
    export_parser.add_argument("folder", type=Path, help="папка запуска с model/ и run.json")
    export_parser.add_argument("--json", default="export.json", help="куда записать итог в папке запуска")
    evaluate_parser = sub.add_parser("evaluate", help="ONNX на CPU по valid и test, metrics.json и inference.json")
    add_common(evaluate_parser)
    evaluate_parser.add_argument("folder", type=Path, help="папка запуска после export")
    evaluate_parser.add_argument("--jobs", type=int, default=0, help="процессов детектора (по умолчанию по числу ядер)")
    evaluate_parser.add_argument(
        "--int8-max-drop", type=float, default=0.002, help="допустимая потеря ROC AUC у int8 на valid"
    )
    evaluate_parser.add_argument(
        "--weights",
        choices=("auto", "fp32", "int8"),
        default="auto",
        help="веса в поставке: auto — int8, если он теряет не больше --int8-max-drop",
    )
    evaluate_parser.add_argument(
        "--int8-valid-only",
        action="store_true",
        help="int8 считать только на valid: веса выбираются по нему, а int8 по одному тексту на CPU долог",
    )
    evaluate_parser.add_argument("--skip-install", action="store_true", help="не мерить размер установки зависимостей")
    evaluate_parser.add_argument(
        "--reuse-install", type=Path, help="взять размер установки из metrics.json другого запуска"
    )
    evaluate_parser.add_argument(
        "--int8-max-changes", type=float, default=0.005, help="допустимая доля текстов valid, где int8 меняет метку"
    )
    evaluate_parser.add_argument(
        "--onnx-texts",
        type=int,
        default=0,
        help="итог по вероятностям PyTorch с GPU, ONNX — только на стольких текстах valid (для больших моделей)",
    )
    rep = sub.add_parser("report", help="карточка для Hugging Face и отчёт в docs/models/ из metrics.json")
    add_common(rep)
    rep.add_argument("folder", type=Path, help="папка запуска после evaluate")
    rep.add_argument("--repo", help="репозиторий модели на Hugging Face; по умолчанию по базе")
    rep.add_argument("--why", help="почему выбрана эта база: абзац для отчёта")
    rep.add_argument("--finalist", type=Path, action="append", help="папка полного запуска для сравнения баз")
    rep.add_argument(
        "--related", type=Path, action="append", help="папка другой модели aiw-ru после evaluate: сравнение в карточке"
    )
    args = parser.parse_args(argv)
    args.out_root = args.out_root or args.data / "transformer"
    if args.command in ("pilot", "train"):
        cmd_fit(args, pilot=args.command == "pilot")
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "report":
        cmd_report(args)
    else:
        args.jobs = args.jobs or os.cpu_count() or 4
        cmd_evaluate(args)


if __name__ == "__main__":
    main()

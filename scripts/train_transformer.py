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

import numpy as np
import onnxruntime as ort
import sklearn
import tokenizers
import torch
import transformers
from huggingface_hub import HfApi, ModelCard, ModelCardData, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from sklearn.metrics import log_loss
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from transformers.utils import logging as hf_logging

import aiw_ru
from aiw_ru.text import plural
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
LATENCY_LENGTHS = (128, 512)
PARITY_TEXTS = 300
# Версия inference.json: её проверяет aiw_ru.models перед выводом.
INFERENCE_VERSION = 1


# ─── Нормализация ───────────────────────────────────────────────────────


def _codes(*codes: int) -> str:
    """Символы для класса re в виде обратная косая + uXXXX: в исходниках только видимые знаки."""
    return "".join(f"{chr(92)}u{c:04x}" for c in codes)


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


def load_classifier(path: str | Path) -> Any:
    """Модель с головой на два класса: 0 — человек, 1 — ИИ."""
    return AutoModelForSequenceClassification.from_pretrained(
        path,
        num_labels=2,
        id2label=dict(enumerate(LABELS)),
        label2id={name: k for k, name in enumerate(LABELS)},
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

    def subset(self, idx: np.ndarray) -> Encoded:
        seqs = [self.seq(int(k)) for k in idx]
        return Encoded(
            ids=np.concatenate(seqs) if seqs else np.zeros(0, dtype=np.int32),
            offsets=np.concatenate([[0], np.cumsum([len(s) for s in seqs])]).astype(np.int64),
            truncated=self.truncated[idx],
            y=self.y[idx],
            genre=self.genre[idx],
            prompt=self.prompt[idx],
            generator=self.generator[idx],
        )


def truncate(ids: list[int], max_length: int) -> list[int]:
    """Начало текста: первые max_length - 1 токенов и завершающий [SEP]."""
    return ids if len(ids) <= max_length else ids[: max_length - 1] + ids[-1:]


def encode_texts(texts: Sequence[str], tok: Any, max_length: int, raw: bool) -> tuple[list[list[int]], list[bool]]:
    """Токены текстов с [CLS] и [SEP], обрезанные до max_length, и признак обрезки."""
    prep: Callable[[str], str] = str if raw else normalize
    cap = max_length * CHARS_PER_TOKEN
    enc = tok([prep(t[:cap]) for t in texts], add_special_tokens=True, truncation=False)["input_ids"]
    out, truncated = [], []
    for text, ids in zip(texts, enc, strict=True):
        if len(ids) <= max_length and len(text) > cap:
            ids = tok(prep(text), add_special_tokens=True, truncation=False)["input_ids"]
        truncated.append(len(ids) > max_length)
        out.append(truncate(ids, max_length))
    return out, truncated


def encode_split(
    data: Path, split: str, limit: int, tok: Any, max_length: int, raw: bool, cache_dir: Path, chunk: int = 4096
) -> Encoded:
    """Токенизирует часть корпуса; результат кэшируется в cache_dir."""
    key = f"{tokenizer_key(tok)}-{max_length}-{'raw' if raw else normalize_key()}"
    cache = cache_dir / f"tokens-{split}{f'-{limit}' if limit else ''}-{key}.npz"
    if cache.exists():
        z = np.load(cache)
        return Encoded(**{f: z[f] for f in Encoded.__dataclass_fields__})
    started = time.perf_counter()
    lines = load(data, "classification", split, limit)
    parts: list[np.ndarray] = []
    lengths: list[int] = []
    truncated: list[bool] = []
    meta: list[tuple[int, str, str, str]] = []
    for i in range(0, len(lines), chunk):
        rs = [json.loads(line) for line in lines[i : i + chunk]]
        seqs, tr = encode_texts([r["text"] for r in rs], tok, max_length, raw)
        parts += [np.asarray(s, dtype=np.int32) for s in seqs]
        lengths += [len(s) for s in seqs]
        truncated += tr
        meta += [
            (int(r["label"] == "ai"), r["data_type"], r["prompt_type"] or "", r["model"] if r["label"] == "ai" else "")
            for r in rs
        ]
        print(f"\r  {split}: {min(i + chunk, len(lines))}/{len(lines)}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    enc = Encoded(
        ids=np.concatenate(parts) if parts else np.zeros(0, dtype=np.int32),
        offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
        truncated=np.array(truncated, dtype=bool),
        y=np.array([m[0] for m in meta], dtype=np.int8),
        genre=np.array([m[1] for m in meta]),
        prompt=np.array([m[2] for m in meta]),
        generator=np.array([m[3] for m in meta]),
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **{f: getattr(enc, f) for f in Encoded.__dataclass_fields__})
    print(
        f"Токены {split}: {len(enc)} текстов за {time.perf_counter() - started:.0f} с, "
        f"обрезано {comma_pct(float(enc.truncated.mean()))}",
        file=sys.stderr,
        flush=True,
    )
    return enc


def batches(lengths: np.ndarray, size: int, rng: np.random.Generator | None) -> list[np.ndarray]:
    """Пачки из текстов близкой длины, чтобы меньше паддинга. С rng порядок пачек случайный."""
    n = len(lengths)
    if rng is None:
        order = np.argsort(lengths, kind="stable")
        return [order[i : i + size] for i in range(0, n, size)]
    out: list[np.ndarray] = []
    perm = rng.permutation(n)
    pool = size * 64
    for i in range(0, n, pool):
        chunk = perm[i : i + pool]
        chunk = chunk[np.argsort(lengths[chunk], kind="stable")]
        out += [chunk[j : j + size] for j in range(0, len(chunk), size)]
    return [out[k] for k in rng.permutation(len(out))]


def collate(enc: Encoded, idx: np.ndarray, pad_id: int, multiple: int) -> tuple[np.ndarray, np.ndarray]:
    """input_ids и attention_mask пачки; ширина округляется вверх до кратной multiple."""
    lens = enc.lengths[idx]
    width = int(-(-int(lens.max()) // multiple) * multiple)
    ids = np.full((len(idx), width), pad_id, dtype=np.int64)
    mask = np.zeros((len(idx), width), dtype=np.int64)
    for row, k in enumerate(idx):
        s = enc.seq(int(k))
        ids[row, : len(s)] = s
        mask[row, : len(s)] = 1
    return ids, mask


# ─── Метрики ────────────────────────────────────────────────────────────


def quick(y: np.ndarray, p: np.ndarray) -> dict[str, float | None]:
    """Метрики для выбора шага: ROC AUC, accuracy при пороге 0,5 и logloss."""
    return {
        "roc_auc": auc(y, p),
        "accuracy": float((y == (p >= THRESHOLD)).mean()),
        "logloss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])),
    }


def fmt(x: float | None, digits: int = 3) -> str:
    return comma_num(x, digits) if x is not None else "—"


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
def predict_torch(model: Any, enc: Encoded, device: torch.device, precision: str, size: int, pad_id: int) -> np.ndarray:
    """Вероятность класса «ИИ» для каждого текста."""
    model.eval()
    out = np.zeros(len(enc), dtype=np.float32)
    for idx in batches(enc.lengths, size, None):
        ids, mask = collate(enc, idx, pad_id, pad_multiple(device))
        with autocast(device, precision):
            logits = model(
                input_ids=torch.from_numpy(ids).to(device), attention_mask=torch.from_numpy(mask).to(device)
            ).logits
        out[idx] = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
    return out


def param_groups(model: Any, weight_decay: float) -> list[dict]:
    """Без затухания весов для смещений и нормировок."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if p.ndim < 2 or "norm" in name.lower() else decay).append(p)
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
    seed: int = SEED


def fit(cfg: Config, train: Encoded, valid: Encoded, out: Path, tok: Any) -> dict:
    """Обучает модель; после каждой проверки на valid сохраняет лучший шаг в out/model."""
    device = pick_device(cfg.device)
    precision, why = pick_precision(device, cfg.precision)
    print(f"Устройство: {device.type} ({gpu_name(device)}), точность {precision}: {why}", flush=True)
    seed_everything(cfg.seed)
    _, _, pad_id = special_ids(tok)
    model: Any = load_classifier(cfg.base).to(device)
    rng = np.random.default_rng(cfg.seed)
    per_epoch = -(-len(train) // cfg.batch)
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
        f"Обучение {cfg.base}: {len(train)} текстов, {total} шагов по {cfg.batch}, "
        f"проверка на valid ({len(valid)}) каждые {eval_every} шагов",
        flush=True,
    )
    while step < total and not stop:
        for idx in batches(train.lengths, cfg.batch, rng):
            model.train()
            ids, mask = collate(train, idx, pad_id, pad_multiple(device))
            with autocast(device, precision):
                loss = model(
                    input_ids=torch.from_numpy(ids).to(device),
                    attention_mask=torch.from_numpy(mask).to(device),
                    labels=torch.from_numpy(train.y[idx].astype(np.int64)).to(device),
                ).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            seen += len(idx)
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
                    "eta_minutes": (total - step) * cfg.batch / speed / 60,
                    "best_valid_roc_auc": best_auc,
                }
                print(
                    f"шаг {step}/{total}, эпоха {progress['epoch']:.2f}: loss {progress['loss']:.4f}, "
                    f"{speed:.0f} текстов/с, осталось {progress['eta_minutes']:.0f} мин "
                    f"({now - last:.0f} с на {LOG_EVERY} шагов)",
                    flush=True,
                )
                (out / "progress.json").write_text(json.dumps(progress) + "\n", encoding="utf-8")
                last = now
            if step % eval_every == 0 or step == total:
                sync(device)
                t0 = time.perf_counter()
                p = predict_torch(model, valid, device, precision, cfg.eval_batch, pad_id)
                point = {
                    "step": step,
                    "epoch": round(step / per_epoch, 3),
                    "train_loss": loss_sum / max(loss_n, 1),
                    **quick(valid.y, p),
                    "seconds": time.perf_counter() - started,
                }
                curve.append(point)
                loss_sum, loss_n = 0.0, 0
                score = point["roc_auc"] if point["roc_auc"] is not None else 0.0
                better = score > best_auc
                print(
                    f"valid на шаге {step}: ROC AUC {fmt(point['roc_auc'], 4)}, accuracy {fmt(point['accuracy'])}, "
                    f"logloss {point['logloss']:.4f} ({time.perf_counter() - t0:.0f} с)"
                    + (", лучший" if better else ""),
                    flush=True,
                )
                if better:
                    best_auc, best_point, stale = score, point, 0
                    model.save_pretrained(out / "model")
                    tok.save_pretrained(out / "model")
                    np.save(out / "valid-probs.npy", p)
                else:
                    stale += 1
                    if cfg.patience and stale >= cfg.patience:
                        print(f"Ранняя остановка: {stale} проверки подряд без роста ROC AUC", flush=True)
                        stop = True
                (out / "curve.json").write_text(json.dumps(curve, indent=1) + "\n", encoding="utf-8")
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
    try:
        return HfApi().model_info(base).sha or ""
    except (HfHubHTTPError, OSError):
        return ""


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
    )
    hf_logging.set_verbosity_error()
    pick_device(cfg.device)  # без GPU остановиться до токенизации, а не после
    tok = load_tokenizer(cfg.base)
    special_ids(tok)
    cache = args.out_root / "cache"
    limit = args.limit
    eval_limit = args.eval_limit if args.eval_limit is not None else limit // 2
    name = f"{short_name(cfg.base)}{'-raw' if cfg.raw else ''}"
    out: Path = args.out or args.out_root / ("pilot" if pilot else "runs") / (f"{name}-{limit}" if limit else name)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Папка запуска: {out}", flush=True)
    train = encode_split(args.data, "train", limit, tok, cfg.max_length, cfg.raw, cache)
    valid = encode_split(args.data, "valid", eval_limit, tok, cfg.max_length, cfg.raw, cache)
    summary = fit(cfg, train, valid, out, tok)
    model: Any = AutoModelForSequenceClassification.from_pretrained(out / "model")
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
            "parameters": sum(p.numel() for p in model.parameters()),
            "max_length": cfg.max_length,
            "normalize": not cfg.raw,
            "normalize_rules": normalize_key(),
            "epochs": cfg.epochs,
            "lr": cfg.lr,
            "batch": cfg.batch,
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
        p = predict_torch(model.to(device), test, device, summary["precision"], cfg.eval_batch, pad_id)
        np.save(out / "test-probs.npy", p)
        run["dataset"]["test"] = len(test)
        run["params"]["truncated_share"]["test"] = float(test.truncated.mean())
        run["torch_test"] = {**quick(test.y, p), "seconds": time.perf_counter() - started}
        print(f"test, PyTorch {summary['precision']}: {run['torch_test']}", flush=True)
    (out / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Готово: {out / 'run.json'}", flush=True)


# ─── ONNX ───────────────────────────────────────────────────────────────


def export_onnx(model_dir: Path, path: Path) -> None:
    """PyTorch → ONNX с переменной длиной пачки и текста."""
    model: Any = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model = model.eval().float()
    tok = load_tokenizer(model_dir)
    sample = tok(["Короткий пример.", "Пример подлиннее, чтобы в пачке были тексты разной длины."], padding=True)
    batch = torch.export.Dim("batch", min=1, max=1024)
    seq = torch.export.Dim("sequence", min=2, max=int(model.config.max_position_embeddings))
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
        dynamic_shapes={"input_ids": {0: batch, 1: seq}, "attention_mask": {0: batch, 1: seq}},
        dynamo=True,
        external_data=False,
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
    onnx.save(model, bare)
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


def session(path: Path, threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


class Session(Protocol):
    """То, что нужно выводу от onnxruntime.InferenceSession."""

    def run(self, output_names: Any, input_feed: Any, /) -> Any: ...


def softmax_ai(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e[:, 1] / e.sum(axis=1)).astype(np.float32)


def run_logits(sess: Session, ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.asarray(sess.run(["logits"], {"input_ids": ids, "attention_mask": mask})[0])


def predict_onnx(sess: Session, enc: Encoded, pad_id: int, size: int = 32) -> np.ndarray:
    """Вероятности пачками текстов близкой длины.

    Динамический int8 считает масштаб активаций по всей пачке, и ответ зависит от соседей
    по ней; aiw-ru подаёт тексты по одному, поэтому int8 оценивается с size=1.
    """
    out = np.zeros(len(enc), dtype=np.float32)
    for idx in batches(enc.lengths, size, None):
        ids, mask = collate(enc, idx, pad_id, 1)
        out[idx] = softmax_ai(run_logits(sess, ids, mask))
    return out


def timed(fn: Callable[[], object], warmup: int = 5, runs: int = 30) -> float:
    """Медиана времени вызова в миллисекундах."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return float(np.median(times) * 1000)


def run_torch(model: Any, ids: torch.Tensor, mask: torch.Tensor) -> None:
    with torch.inference_mode():
        model(input_ids=ids, attention_mask=mask)


def latency_inputs(enc: Encoded, length: int, sep_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Настоящий текст ровно на length токенов: самый длинный из части, обрезанный с [SEP] в конце."""
    s = enc.seq(int(np.argmax(enc.lengths)))
    if len(s) > length:
        s = np.concatenate([s[: length - 1], [sep_id]])
    ids = s.astype(np.int64)[None, :]
    return ids, np.ones_like(ids)


def measure_latency(model_dir: Path, onnx_files: dict[str, Path], enc: Encoded, sep_id: int) -> list[dict]:
    """Задержка на один текст: ONNX Runtime и PyTorch на CPU, один поток и все ядра."""
    out: list[dict] = []
    model: Any = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model = model.eval().float()
    for length in LATENCY_LENGTHS:
        ids, mask = latency_inputs(enc, length, sep_id)
        for threads in sorted({1, os.cpu_count() or 1}):
            for variant, path in onnx_files.items():
                ms = timed(partial(run_logits, session(path, threads), ids, mask))
                out.append({"runtime": f"onnxruntime {variant}", "tokens": ids.shape[1], "threads": threads, "ms": ms})
            torch.set_num_threads(threads)
            ms = timed(partial(run_torch, model, torch.from_numpy(ids), torch.from_numpy(mask)))
            out.append({"runtime": "torch fp32", "tokens": ids.shape[1], "threads": threads, "ms": ms})
    torch.set_num_threads(os.cpu_count() or 1)
    return out


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

    tok = load_tokenizer(model_dir)
    _, sep_id, pad_id = special_ids(tok)
    p = run["params"]
    raw = not p["normalize"]
    eval_limit = run["dataset"]["eval_limit"]
    valid = encode_split(args.data, "valid", eval_limit, tok, p["max_length"], raw, args.out_root / "cache")
    # Сверка на текстах всех длин: равномерно по отсортированному списку.
    order = np.argsort(valid.lengths, kind="stable")
    pick = order[np.linspace(0, len(order) - 1, min(PARITY_TEXTS, len(order))).astype(int)]
    sub = valid.subset(pick)
    model: Any = AutoModelForSequenceClassification.from_pretrained(model_dir)
    ref = predict_torch(model.eval().float(), sub, torch.device("cpu"), "fp32", 16, pad_id)
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
    spec = inference_spec(p, tok, "head", "fp32")
    lines = load(args.data, "classification", "valid", eval_limit)
    texts = [json.loads(lines[int(k)])["text"] for k in pick[:50]]
    btok = inference_tokenizer(model_dir / "tokenizer.json")
    sess = session(fp32, os.cpu_count() or 1)
    one = np.array([predict_text(t, sess, btok, spec) for t in texts])
    batch = predict_onnx(sess, valid.subset(pick[:50]), pad_id)
    parity["spec_vs_batch_max_abs_diff"] = float(np.abs(one - batch).max())
    print(f"Сверка: {json.dumps(parity, ensure_ascii=False)}", flush=True)
    if parity["fp32"]["max_abs_diff"] > 1e-3 or parity["spec_vs_batch_max_abs_diff"] > 1e-3:
        raise SystemExit("ONNX fp32 или вывод по спецификации расходятся с PyTorch, см. сверку выше")

    latency = measure_latency(model_dir, {"fp32": fp32, "int8": int8}, valid, sep_id)
    table(
        "Задержка на один текст, мс",
        ["Среда", "Токенов", "Потоков", "мс"],
        [[r["runtime"], str(r["tokens"]), str(r["threads"]), f"{r['ms']:.1f}"] for r in latency],
    )
    weights = next(model_dir.glob("*.safetensors"))
    export = {
        "parity": parity,
        "latency": latency,
        "cpu": cpu_name(),
        "sizes_mb": {
            "onnx_fp32": fp32.stat().st_size / 1e6,
            "onnx_int8": int8.stat().st_size / 1e6,
            "safetensors": weights.stat().st_size / 1e6,
            "tokenizer": sum(f.stat().st_size for f in model_dir.glob("tokenizer*")) / 1e6,
        },
        "environment": environment(),
    }
    (run_dir / args.json).write_text(json.dumps(export, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Готово: {run_dir / args.json}", flush=True)


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


def inference_spec(p: dict, tok: Any, long_texts: str, weights: str = "int8") -> dict:
    """Всё, что нужно выводу без transformers: файл, токены, нормализация, порог, правило для длинных текстов."""
    cls_id, sep_id, pad_id = special_ids(tok)
    windows = MAX_WINDOWS if long_texts == "mean_of_windows" else 1
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
        "max_length": p["max_length"],
        "max_chars": p["max_length"] * CHARS_PER_TOKEN * MAX_WINDOWS,
        "cls_id": cls_id,
        "sep_id": sep_id,
        "pad_id": pad_id,
        "long_texts": long_texts,
        "max_windows": windows,
        "normalize": [{"pattern": pat, "replacement": rep, "why": why} for pat, rep, why in NORMALIZE_RULES]
        if p["normalize"]
        else [],
    }


def windows(ids: Sequence[int], max_length: int, cls_id: int, sep_id: int, limit: int) -> list[list[int]]:
    """Окна по max_length токенов подряд, без перекрытия; ids — токены текста без [CLS] и [SEP]."""
    width = max_length - 2
    return [[cls_id, *ids[i : i + width], sep_id] for i in range(0, max(len(ids), 1), width)][:limit]


def window_probs(sess: Session, ids: Sequence[int], spec: dict) -> np.ndarray:
    ws = windows(ids, spec["max_length"], spec["cls_id"], spec["sep_id"], spec["max_windows"])
    arr = np.full((len(ws), max(len(w) for w in ws)), spec["pad_id"], dtype=np.int64)
    mask = np.zeros_like(arr)
    for k, w in enumerate(ws):
        arr[k, : len(w)] = w
        mask[k, : len(w)] = 1
    return softmax_ai(run_logits(sess, arr, mask))


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
    sess: ort.InferenceSession, data: Path, split: str, limit: int, enc: Encoded, head: np.ndarray, tok: Any, p: dict
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Начало текста против среднего по окнам на текстах длиннее max_length."""
    idx = np.flatnonzero(enc.truncated)
    lines = load(data, "classification", split, limit)
    spec = inference_spec(p, tok, "mean_of_windows")
    btok = inference_tokenizer(Path(tok.name_or_path) / "tokenizer.json")
    mean = np.array([predict_text(json.loads(lines[int(k)])["text"], sess, btok, spec) for k in idx])
    y = enc.y[idx]
    return {"texts": len(idx), "head": quick(y, head[idx]), "mean_of_windows": quick(y, mean)}, idx, mean


# ─── Оценка ─────────────────────────────────────────────────────────────


def install_size(packages: Sequence[str], python_platform: str, index: str | None = None) -> float | None:
    """Мегабайты на диске после установки пакетов с зависимостями через uv (Python 3.12)."""
    uv = shutil.which("uv")
    if not uv:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [uv, "pip", "install", "--quiet", "--target", tmp, "--python-version", "3.12"]
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
        return sum(f.stat().st_size for f in Path(tmp).rglob("*") if f.is_file()) / 1e6


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
    out = []
    for name, pkgs, plat, index in cases:
        size = install_size(pkgs, plat, index)
        print(f"  {name}, {plat}: {'—' if size is None else f'{size:.0f} МБ'}", file=sys.stderr, flush=True)
        out.append({"stack": name, "packages": pkgs, "platform": plat, "mb": size})
    return out


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
    rs = [json.loads(line) for line in load(data, "classification", split, limit)]
    y = np.array([r["label"] == "ai" for r in rs], dtype=np.int8)
    out = []
    for name, fn in SHORTCUTS.items():
        a = auc(y, np.array([fn(r["text"]) for r in rs], dtype=np.float64))
        # Направление приметы не важно: 0,3 говорит столько же, сколько 0,7.
        out.append({"cue": name, "roc_auc": a, "separation": None if a is None else max(a, 1 - a)})
    return out


def lightgbm_block() -> dict | None:
    """Метрики LightGBM на том же test для сравнения: из docs/models или с Hugging Face."""
    path = ROOT / LIGHTGBM_METRICS
    if not path.exists():
        try:
            path = Path(hf_hub_download(LIGHTGBM_REPO, "metrics.json"))
        except (HfHubHTTPError, OSError) as e:
            print(f"Метрики LightGBM не найдены: {e}", file=sys.stderr)
            return None
    m = json.loads(path.read_text(encoding="utf-8"))
    keep = ("created", "git_commit", "test", "valid", "genres", "lengths", "prompt_types", "generators")
    return {"repo": LIGHTGBM_REPO, **{k: m[k] for k in keep if k in m}}


def pilot_rows(out_root: Path) -> list[dict]:
    """Строки таблицы пилота: pilot/*/run.json и export.json."""
    out = []
    for run_file in sorted((out_root / "pilot").glob("*/run.json")):
        run = json.loads(run_file.read_text(encoding="utf-8"))
        exp_file = run_file.parent / "export.json"
        exp = json.loads(exp_file.read_text(encoding="utf-8")) if exp_file.exists() else {}
        lat = {(r["runtime"], r["tokens"], r["threads"]): r["ms"] for r in exp.get("latency", [])}
        many = max((r["threads"] for r in exp.get("latency", [])), default=0)
        p = run["params"]
        out.append(
            {
                "base": p["base"],
                "license": p["base_license"],
                "normalize": p["normalize"],
                "train": run["dataset"]["train"],
                "valid": run["dataset"]["valid"],
                "epochs": p.get("epochs"),
                "lr": p.get("lr"),
                "batch": p.get("batch"),
                "seed": p.get("seed"),
                "valid_roc_auc": run["valid"]["roc_auc"],
                "valid_accuracy": run["valid"]["accuracy"],
                "parameters": p["parameters"],
                "train_seconds": p["train_seconds"],
                "texts_per_second": p["texts_per_second"],
                "gpu": p.get("gpu", ""),
                "onnx_fp32_mb": exp.get("sizes_mb", {}).get("onnx_fp32"),
                "onnx_int8_mb": exp.get("sizes_mb", {}).get("onnx_int8"),
                "int8_ms_128_1": lat.get(("onnxruntime int8", 128, 1)),
                "int8_ms_512_1": lat.get(("onnxruntime int8", 512, 1)),
                "int8_ms_512_all": lat.get(("onnxruntime int8", 512, many)),
                "fp32_ms_512_1": lat.get(("onnxruntime fp32", 512, 1)),
            }
        )
    return out


def flat_environment(train_env: dict[str, str]) -> dict[str, str]:
    """Версии при обучении и при оценке на CPU в одном плоском словаре, как у LightGBM."""
    ev = environment()
    # run.json с VM мог записать «2 ядер»: число ядер приводится к правильной форме.
    train_env = {
        k: re.sub(r"(\d+) ядер\b", lambda g: plural(int(g[1]), "ядро", "ядра", "ядер"), v) for k, v in train_env.items()
    }
    return {
        **train_env,
        "onnx": _onnx_version(),
        **{f"evaluate_{k}": v for k, v in ev.items() if train_env.get(k) != v},
    }


def _onnx_version() -> str:
    import onnx

    return onnx.__version__


def bundle_hub(run_dir: Path, shipped_file: str) -> Path:
    """Папка hub/ для репозитория модели: model.onnx — то, что в поставке, рядом fp32, токенизатор и метрики.

    aiw-ru скачивает inference.json, model.onnx, tokenizer.json, metrics.json и README.md;
    model_fp32.onnx лежит для тех, кому int8 не подходит.
    """
    hub_dir = run_dir / "hub"
    shutil.rmtree(hub_dir, ignore_errors=True)
    hub_dir.mkdir(parents=True)
    files = {"model.onnx": shipped_file}
    if shipped_file != "onnx/model.onnx":
        files["model_fp32.onnx"] = "onnx/model.onnx"
    for name in ("tokenizer.json", "tokenizer_config.json", "config.json"):
        files[name] = f"model/{name}"
    files |= {"inference.json": "inference.json", "metrics.json": "metrics.json"}
    for dst, src in files.items():
        shutil.copy2(run_dir / src, hub_dir / dst)
    return hub_dir


def x86_latency(run_dir: Path) -> dict:
    """Задержка на слабом x86: export, запущенный на CPU VM Colab, кладёт export-x86.json."""
    path = run_dir / "export-x86.json"
    if not path.exists():
        return {}
    x86 = json.loads(path.read_text(encoding="utf-8"))
    return {"latency_x86": x86["latency"], "latency_x86_cpu": x86["cpu"]}


def cmd_evaluate(args: argparse.Namespace) -> None:
    """ONNX на CPU по valid и test, разрезы как у LightGBM, metrics.json и inference.json."""
    run_dir: Path = args.folder
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    export = json.loads((run_dir / "export.json").read_text(encoding="utf-8"))
    p = run["params"]
    limit = run["dataset"]["eval_limit"]
    hf_logging.set_verbosity_error()
    tok = load_tokenizer(run_dir / "model")
    _, _, pad_id = special_ids(tok)
    cache = args.out_root / "cache"
    raw = not p["normalize"]
    valid = encode_split(args.data, "valid", limit, tok, p["max_length"], raw, cache)
    test = encode_split(args.data, "test", limit, tok, p["max_length"], raw, cache)

    probs: dict[str, dict[str, np.ndarray]] = {"torch": {}}
    for split in ("valid", "test"):
        if (run_dir / f"{split}-probs.npy").exists():
            probs["torch"][split] = np.load(run_dir / f"{split}-probs.npy")
    threads = os.cpu_count() or 1
    for variant, file in (("onnx_fp32", "model.onnx"), ("onnx_int8", "model_int8.onnx")):
        sess = session(run_dir / "onnx" / file, threads)
        probs[variant] = {}
        for split, enc in (("valid", valid), ("test", test)):
            cached = run_dir / f"{split}-probs-{variant}.npy"
            if not cached.exists():
                t0 = time.perf_counter()
                np.save(cached, predict_onnx(sess, enc, pad_id, 1 if variant == "onnx_int8" else 32))
                print(f"{variant} {split}: {time.perf_counter() - t0:.0f} с", flush=True)
            probs[variant][split] = np.load(cached)
    parts = {"valid": valid, "test": test}
    variants = {v: {s: quick(parts[s].y, q) for s, q in ps.items()} for v, ps in probs.items()}
    table(
        "Варианты модели: ROC AUC и accuracy",
        ["Вариант", "valid AUC", "valid acc", "test AUC", "test acc"],
        [
            [v, *(fmt(m.get(s, {}).get(k), 4) for s in ("valid", "test") for k in ("roc_auc", "accuracy"))]
            for v, m in variants.items()
        ],
    )
    # int8 идёт в поставку, если на valid теряет не больше --int8-max-drop ROC AUC; test в решении не участвует.
    drop = (variants["onnx_fp32"]["valid"]["roc_auc"] or 0) - (variants["onnx_int8"]["valid"]["roc_auc"] or 0)
    shipped = "onnx_int8" if drop <= args.int8_max_drop else "onnx_fp32"
    shipped_file = "onnx/model_int8.onnx" if shipped == "onnx_int8" else "onnx/model.onnx"
    print(f"В поставку: {shipped} (int8 теряет на valid {drop:.4f} ROC AUC)", flush=True)
    sess = session(run_dir / shipped_file, threads)

    # Длинные тексты: начало или среднее по окнам, выбор по valid.
    lv, idx_v, mean_v = long_texts(sess, args.data, "valid", limit, valid, probs[shipped]["valid"], tok, p)
    use_windows = (lv["mean_of_windows"]["roc_auc"] or 0) > (lv["head"]["roc_auc"] or 0)
    lt, idx_t, mean_t = long_texts(sess, args.data, "test", limit, test, probs[shipped]["test"], tok, p)
    print(f"Длинные тексты, valid: {lv}\nДлинные тексты, test: {lt}", flush=True)
    final = probs[shipped]["test"].copy()
    pv = probs[shipped]["valid"].copy()
    if use_windows:
        final[idx_t] = mean_t
        pv[idx_v] = mean_v
    spec = inference_spec(p, tok, "mean_of_windows" if use_windows else "head", shipped.removeprefix("onnx_"))

    _, res = detect(args.data, "classification", "test", limit, "scan", "general", args.jobs)
    score = np.array([r["score"] for r in res], dtype=np.float32)
    part = Part(
        x=np.zeros((len(test), 0), dtype=np.float32),
        y=test.y,
        score=score,
        words=np.array([r["stats"]["words"] for r in res]),
        genre=test.genre,
        prompt=test.prompt,
        generator=test.generator,
    )
    created = (test.y == 0) | (test.prompt == "create")
    model_test = classes(test.y, (final >= THRESHOLD).astype(np.int8))
    print_classes("Модель на test, порог вероятности 0,5", model_test)

    install = [] if args.skip_install else install_sizes()
    metrics = {
        "created": datetime.now(UTC).strftime("%Y-%m-%d"),
        "aiw_ru_version": aiw_ru.__version__,
        "git_commit": run["git_commit"],
        "environment": flat_environment(run["environment"]),
        "dataset": {**run["dataset"], "valid": len(valid), "test": len(test)},
        "params": {
            **p,
            "license": p["base_license"],
            "shipped": shipped,
            "shipped_file": shipped_file,
            "size_mb": export["sizes_mb"][shipped],
            "hardware": f"обучение: {p['gpu']}; оценка и задержка: {export['cpu']}",
        },
        "test": {**model_test, "roc_auc": auc(test.y, final), "roc_auc_created": auc(test.y[created], final[created])},
        "valid": {"accuracy": float((valid.y == (pv >= THRESHOLD)).mean()), "roc_auc": auc(valid.y, pv)},
        "detector": {
            **classes(test.y, (score >= 40).astype(np.int8)),
            "threshold": 40,
            "roc_auc": auc(test.y, score),
            "roc_auc_created": auc(test.y[created], score[created]),
        },
        **breakdowns(part, final),
        "variants": variants,
        "long_texts": {"rule": spec["long_texts"], "valid": lv, "test": lt},
        "parity": export["parity"],
        "sizes_mb": export["sizes_mb"],
        "latency": export["latency"],
        "latency_cpu": export["cpu"],
        **x86_latency(run_dir),
        "install": install,
        "shortcuts": shortcut_probe(args.data, "valid", limit),
        "curve": run["curve"],
        "pilot": pilot_rows(args.out_root),
        "inference": {k: v for k, v in spec.items() if k != "normalize"},
        "lightgbm": lightgbm_block(),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "inference.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Готово: {run_dir / 'metrics.json'}, {run_dir / 'inference.json'}", flush=True)


# ─── Карточка и отчёт ───────────────────────────────────────────────────

REPO = "toiletsandpaper/russian-ai-text-detector-bert"
DOCS = Path("docs") / "models"
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
}


def _auc(x: float | None) -> str:
    return num(x) if x is not None else "—"


def _num4(x: float | None) -> str:
    return num(x, 4) if x is not None else "—"


def _ms(x: float | None) -> str:
    return f"${x:.1f}$" if x is not None else "—"


def _mb(x: float | None) -> str:
    if x is None:
        return "—"
    return f"${x:.0f}$" if x >= 100 else f"${x:.1f}$"


def _sci(x: float) -> str:
    """Малое число формулой: $3.2\\cdot 10^{-6}$."""
    mant, exp = f"{x:.1e}".split("e")
    return f"${mant}\\cdot 10^{{{int(exp)}}}$"


def _code(text: str) -> str:
    """Код в markdown; шаблон с обратными кавычками внутри оборачивается двумя."""
    return f"`` {text} ``" if "`" in text else f"`{text}`"


def _by(rows: list[dict] | None, key: str) -> dict[str, dict]:
    return {r[key]: r for r in rows or []}


def _nw(n: int, one: str, few: str, many: str) -> str:
    """Число формулой и слово в нужной форме: $2$ текста, $5$ текстов."""
    return f"{count(n)} {plural(n, one, few, many).split(' ', 1)[1]}"


def _tokens_gen(n: int) -> str:
    """«длиннее $512$ токенов», «длиннее $21$ токена»."""
    return _nw(n, "токена", "токенов", "токенов")


def _texts_loc(n: int) -> str:
    """«на $5$ текстах», «на $21$ тексте»."""
    return _nw(n, "тексте", "текстах", "текстах")


def _params_count(n: int) -> str:
    return f"${n / 1e6:.1f}$ млн"


def _weights(m: dict) -> str:
    return "int8" if m["params"]["shipped"] == "onnx_int8" else "fp32"


def _pilot_md(m: dict) -> str:
    return md_table(
        ["База", "Лицензия", "Нормализация", "Параметров", "ROC AUC valid", "Accuracy valid", "int8, МБ", "int8, мс"],
        [
            [
                f"`{r['base']}`",
                LICENSE_NAMES.get(r["license"], r["license"]),
                "да" if r["normalize"] else "нет",
                _params_count(r["parameters"]),
                _auc(r["valid_roc_auc"]),
                num(r["valid_accuracy"]),
                _mb(r.get("onnx_int8_mb")),
                _ms(r.get("int8_ms_512_1")),
            ]
            for r in m["pilot"]
        ],
    )


def _finalists_md(m: dict) -> str:
    return md_table(
        [
            "База",
            "ROC AUC valid",
            "Accuracy valid",
            "ROC AUC int8",
            "Accuracy int8",
            "int8 сменил метку",
            "int8, МБ",
            "int8, мс, M1",
            "int8, мс, x86",
        ],
        [
            [
                f"`{r['base']}`",
                _num4(r["valid_roc_auc"]),
                num(r["valid_accuracy"]),
                _num4(r.get("valid_roc_auc_int8")),
                _auc(r.get("valid_accuracy_int8")),
                pct(r["int8_label_changes"]) if r.get("int8_label_changes") is not None else "—",
                _mb(r.get("onnx_int8_mb")),
                _ms(r.get("int8_ms_512_1")),
                _ms(r.get("x86_int8_ms_512_1")),
            ]
            for r in m.get("finalists", [])
        ],
    )


def _compare_md(m: dict) -> str:
    t, d, lt = m["test"], m["detector"], (m.get("lightgbm") or {}).get("test", {})

    def fpr(r: dict) -> str:
        return pct(1 - r["classes"]["human"]["recall"]) if r else "—"

    rows = [
        ["Accuracy", num(t["accuracy"]), _auc(lt.get("accuracy")), num(d["accuracy"])],
        ["ROC AUC", _auc(t["roc_auc"]), _auc(lt.get("roc_auc")), _auc(d["roc_auc"])],
        [
            "ROC AUC, люди против текстов с нуля",
            _auc(t["roc_auc_created"]),
            _auc(lt.get("roc_auc_created")),
            _auc(d["roc_auc_created"]),
        ],
        ["F1, среднее по классам", num(t["macro"]["f1"]), _auc(lt.get("macro", {}).get("f1")), num(d["macro"]["f1"])],
        ["Людей принято за ИИ", fpr(t), fpr(lt), fpr(d)],
    ]
    return md_table(["На test", "Эта модель", "LightGBM", "Правила aiw-ru"], rows)


def _cuts_md(m: dict, level: str) -> str:
    lg = m.get("lightgbm") or {}
    lg_g = _by(lg.get("genres"), "genre")
    lg_l = _by(lg.get("lengths"), "bucket")
    lg_p = _by(lg.get("prompt_types"), "prompt_type")
    lg_m = _by(lg.get("generators"), "model")
    genres = md_table(
        ["Жанр", "Людей", "ИИ", "Accuracy", "ROC AUC", "ROC AUC LightGBM", "ROC AUC правил"],
        [
            [
                f"`{g['genre']}`",
                count(g["human"]),
                count(g["ai"]),
                num(g["accuracy"]),
                _auc(g["roc_auc"]),
                _auc(lg_g.get(g["genre"], {}).get("roc_auc")),
                _auc(g["detector_roc_auc"]),
            ]
            for g in m["genres"]
        ],
    )
    lengths = md_table(
        ["Слов в тексте", "Текстов", "Accuracy", "ROC AUC", "Accuracy LightGBM", "ROC AUC LightGBM"],
        [
            [
                _bucket(b["bucket"]),
                count(b["n"]),
                num(b["accuracy"]),
                _auc(b["roc_auc"]),
                _auc(lg_l.get(b["bucket"], {}).get("accuracy")),
                _auc(lg_l.get(b["bucket"], {}).get("roc_auc")),
            ]
            for b in m["lengths"]
        ],
    )
    prompts = md_table(
        ["Задание генератору", "ИИ-текстов", "Recall", "Recall LightGBM"],
        [
            [
                f"`{x['prompt_type']}`",
                count(x["n"]),
                num(x["recall"]),
                _auc(lg_p.get(x["prompt_type"], {}).get("recall")),
            ]
            for x in m["prompt_types"]
        ],
    )
    generators = md_table(
        ["Модель-генератор", "Текстов в test", "Recall", "Recall LightGBM"],
        [
            [f"`{x['model']}`", count(x["n"]), num(x["recall"]), _auc(lg_m.get(x["model"], {}).get("recall"))]
            for x in m["generators"]
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


def _long_md(m: dict) -> str:
    v, t = m["long_texts"]["valid"], m["long_texts"]["test"]
    return md_table(
        ["Правило", "ROC AUC valid", "Accuracy valid", "ROC AUC test", "Accuracy test"],
        [
            [label, _auc(v[k]["roc_auc"]), num(v[k]["accuracy"]), _auc(t[k]["roc_auc"]), num(t[k]["accuracy"])]
            for k, label in (("head", "начало текста"), ("mean_of_windows", "среднее по окнам"))
        ],
    )


def _latency_md(rows_: list[dict]) -> str:
    lat = {(r["runtime"], r["tokens"], r["threads"]): r["ms"] for r in rows_}
    runtimes = sorted({r["runtime"] for r in rows_})
    threads = sorted({r["threads"] for r in rows_})
    # Длина берётся из замера: если текстов длиннее нет, замер короче LATENCY_LENGTHS.
    tokens = sorted({r["tokens"] for r in rows_})
    head = ["Среда", *(f"{plural(t, 'токен', 'токена', 'токенов')}, потоков: {n}" for t in tokens for n in threads)]
    return md_table(head, [[r, *(_ms(lat.get((r, t, n))) for t in tokens for n in threads)] for r in runtimes])


def _install_md(m: dict) -> str:
    return md_table(
        ["Зависимости вывода", "Платформа", "МБ на диске"],
        [[x["stack"], f"`{x['platform']}`", _mb(x["mb"])] for x in m["install"]],
    )


def _variants_md(m: dict) -> str:
    names = {"torch": "PyTorch на GPU", "onnx_fp32": "ONNX fp32 на CPU", "onnx_int8": "ONNX int8 на CPU"}
    return md_table(
        ["Вариант", "ROC AUC valid", "Accuracy valid", "ROC AUC test", "Accuracy test"],
        [
            [
                names.get(v, v),
                *(_num4(x.get(s, {}).get(k)) for s in ("valid", "test") for k in ("roc_auc", "accuracy")),
            ]
            for v, x in m["variants"].items()
        ],
    )


def _shortcuts_md(m: dict) -> str:
    return md_table(
        ["Примета оформления", "ROC AUC одной приметы"], [[s["cue"], _auc(s["roc_auc"])] for s in m["shortcuts"]]
    )


def _shortcuts_note(m: dict) -> str:
    best = max(m["shortcuts"], key=lambda s: s["separation"] or 0)
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


def _ablation(m: dict) -> str:
    """Пилот одной базы с нормализацией и без: сколько ROC AUC стоит нормализация."""
    by_base: dict[str, dict[bool, float]] = {}
    for r in m["pilot"]:
        by_base.setdefault(r["base"], {})[r["normalize"]] = r["valid_roc_auc"]
    pairs = {b: v for b, v in by_base.items() if True in v and False in v}
    if not pairs:
        return ""
    base = m["params"]["base"] if m["params"]["base"] in pairs else next(iter(pairs))
    v = pairs[base]
    return (
        f"В пилоте база `{base}` без нормализации дала ROC AUC {num(v[False])} на valid, с нормализацией "
        f"{num(v[True])}. Нормализация всё равно остаётся: модель, выучившая оформление корпуса, "
        "ошибётся на человеческом тексте, набранном в другом редакторе или скопированном из чата."
    )


def _params_md(m: dict) -> str:
    p = m["params"]
    stop = ", сработала" if p["early_stopped"] else ", не понадобилась"
    return md_table(
        ["Параметр", "Значение"],
        [
            ["База", f"[`{p['base']}`](https://huggingface.co/{p['base']}), ревизия `{p['base_revision'][:12]}`"],
            ["Параметров", _params_count(p["parameters"])],
            ["Устройство", f"{p['gpu']}, `{p['device']}`"],
            ["Точность", f"`{p['precision']}`, {p['precision_reason']}"],
            ["Эпох", f"не больше {count(int(p['epochs']))}, лучшая проверка на эпохе {num(p['best_epoch'], 2)}"],
            [
                "Ранняя остановка",
                f"после {_nw(p['patience'], 'проверки', 'проверок', 'проверок')} без роста ROC AUC на valid{stop}",
            ],
            ["Проверок на valid за эпоху", count(p["evals_per_epoch"])],
            ["Оптимизатор", f"AdamW, скорость `{p['lr']:g}`, разогрев {pct(p['warmup'])} шагов, линейный спад"],
            ["Затухание весов, клиппинг градиента", f"`{p['weight_decay']:g}`, `{p['grad_clip']:g}`"],
            ["Пачка", f"{_nw(p['batch'], 'текст', 'текста', 'текстов')} близкой длины"],
            ["`max_length`", _nw(p["max_length"], "токен", "токена", "токенов")],
            ["`seed`", count(p["seed"])],
            [
                "Время обучения",
                (
                    f"${p['train_seconds'] / 60:.0f}$ мин, "
                    f"{_nw(round(p['texts_per_second']), 'текст', 'текста', 'текстов')} в секунду"
                ),
            ],
            ["Файл модели", f"`model.onnx`, веса {_weights(m)}, {_mb(p['size_mb'])} МБ"],
            ["Обучено", f"{m['created']}, коммит `{m['git_commit']}`"],
        ],
    )


def _curve_md(m: dict) -> str:
    return md_table(
        ["Шаг", "Эпоха", "loss train", "ROC AUC valid", "Accuracy valid", "logloss valid"],
        [
            [
                count(c["step"]),
                num(c["epoch"], 2),
                num(c["train_loss"], 4),
                _num4(c["roc_auc"]),
                num(c["accuracy"]),
                num(c["logloss"], 4),
            ]
            for c in m["curve"]
        ],
    )


def _speed_md(m: dict) -> str:
    x86 = ""
    if m.get("latency_x86"):
        x86 = f"""

На двух ядрах виртуальной машины Colab, это ближе к слабому ноутбуку на x86,
{m["latency_x86_cpu"]}:

{_latency_md(m["latency_x86"])}"""
    return f"""Задержка на один текст в миллисекундах, медиана $30$ прогонов, {m["latency_cpu"]}:

{_latency_md(m["latency"])}{x86}

Размер зависимостей вывода после установки через uv для Python 3.12:

{_install_md(m)}

Размер считается вместе со всеми зависимостями, в том числе numpy и
huggingface-hub, которые aiw-ru с extra `ml` и так ставит для LightGBM.

{_onnx_choice(m)}"""


def _onnx_choice(m: dict) -> str:
    """Почему в поставке ONNX, а не PyTorch: задержка на этой машине против размера установки."""
    lat = {(r["runtime"], r["tokens"], r["threads"]): r["ms"] for r in m["latency"]}
    torch_ms, int8_ms = lat.get(("torch fp32", 512, 1)), lat.get(("onnxruntime int8", 512, 1))
    sizes = {(x["stack"], x["platform"]): x["mb"] for x in m["install"] if x["mb"]}
    onnx = {plat: mb for (stack, plat), mb in sizes.items() if stack.startswith("onnxruntime")}
    ratios = [
        mb / onnx[plat]
        for (stack, plat), mb in sizes.items()
        if stack == "torch + transformers" and plat in onnx and "linux" not in plat
    ]
    cpu = sizes.get(("torch (CPU) + transformers", "x86_64-manylinux_2_28"))
    if cpu and "x86_64-manylinux_2_28" in onnx:
        ratios.append(cpu / onnx["x86_64-manylinux_2_28"])
    cuda = sizes.get(("torch + transformers", "x86_64-manylinux_2_28"))
    if not (torch_ms and int8_ms and ratios):
        return ""
    v = m["variants"]
    shipped, ref = v.get(m["params"]["shipped"], {}).get("test", {}), v.get("torch", {}).get("test", {})
    same = "точность та же"
    if shipped.get("roc_auc") is not None and ref.get("roc_auc") is not None:
        diff = abs(shipped["roc_auc"] - ref["roc_auc"])
        same = (
            "ROC AUC на test совпадает с PyTorch до четвёртого знака"
            if diff < 5e-5
            else f"ROC AUC на test отличается от PyTorch на {_num4(diff)}"
        )
    faster = "быстрее" if torch_ms < int8_ms else "медленнее"
    span = f"${min(ratios):.1f}\\text{{–}}{max(ratios):.1f}$"
    linux = f", а на Linux со сборкой torch под CUDA по умолчанию — {_mb(cuda)} МБ" if cuda else ""
    return (
        f"На этой машине torch считает {faster} onnxruntime: {_ms(torch_ms)} мс против {_ms(int8_ms)} мс у int8 "
        f"на $512$ токенах в один поток. Но torch с transformers занимает на диске в {span} раза больше{linux}. "
        f"aiw-ru ставят и на слабые ноутбуки, поэтому в поставке ONNX: установка лёгкая, а {same}."
    )


def _limitations_md(m: dict) -> str:
    fpr = 1 - m["test"]["classes"]["human"]["recall"]
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


def _usage_md(m: dict, repo: str) -> str:
    return f"""Из командной строки:

```bash
uv tool install "aiw-ru[ml]"
aiw-ru models install transformer
aiw-ru classify текст.md
```

Из Python без aiw-ru нужны только onnxruntime, tokenizers, numpy и huggingface-hub:

```python
import json
import re

import numpy as np
import onnxruntime as ort
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer

files = ["inference.json", "model.onnx", "tokenizer.json"]
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
    width = spec["max_length"] - 2
    windows = [ids[i : i + width] for i in range(0, max(len(ids), 1), width)][: spec["max_windows"]]
    windows = [[spec["cls_id"], *w, spec["sep_id"]] for w in windows]
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
`max_length`, служебные токены, правила нормализации из обучения, порог
{num(m["inference"]["threshold"], 1)} и число окон для длинных текстов."""


def _long_rule(m: dict) -> str:
    inf, n = m["inference"], _tokens_gen(m["params"]["max_length"])
    if inf["long_texts"] == "mean_of_windows":
        return (
            f"Текст длиннее {n} режется на окна по столько же токенов, вероятность — среднее "
            f"по первым {count(inf['max_windows'])} окнам: на valid так точнее, чем по началу текста."
        )
    return f"Текст длиннее {n} модель читает только до этой границы: среднее по окнам на valid не точнее."


def card_body(m: dict, repo: str) -> str:
    """Карточка модели на Hugging Face: формулы в $…$, для Hub их переписывает hub_math."""
    p, ds, t, v = m["params"], m["dataset"], m["test"], m["valid"]
    lt = (m.get("lightgbm") or {}).get("test", {})
    name = repo.split("/")[-1]
    report = f"{GITHUB}/blob/master/{DOCS.as_posix()}/{name}.md"
    lgb_line = f" У LightGBM на признаках aiw-ru ROC AUC {num(lt['roc_auc'])}." if lt else ""
    fp32_ru = fp32_en = ""
    if _weights(m) == "int8":
        fp32 = _mb(m["sizes_mb"]["onnx_fp32"])
        fp32_ru = f" Рядом лежит `model_fp32.onnx` на {fp32} МБ с весами fp32, aiw-ru его не скачивает."
        fp32_en = f" An fp32 export, `model_fp32.onnx`, is an optional extra file of {fp32} MB."
    return f"""# Детектор ИИ-текста для русского языка

`{name}` оценивает вероятность, что русский текст написала нейросеть, а не
человек. Обучен на текстах {GENERATORS} и других языковых моделей из корпуса
LLMTrace. Помогает проверить текст на ИИ: статью, новость, отзыв, пост, ответ
на вопрос.

Это дообученный энкодер [`{p["base"]}`](https://huggingface.co/{p["base"]}) на
{_params_count(p["parameters"])} параметров, переведённый в ONNX с весами {_weights(m)}.
Файл `model.onnx` занимает {_mb(p["size_mb"])} МБ и работает на CPU через onnxruntime, без
torch и GPU.{fp32_ru} На отложенной части корпуса ROC AUC {num(t["roc_auc"])}, accuracy
{num(t["accuracy"])}.{lgb_line}

Модель необязательная: детектор и скиллы [aiw-ru]({GITHUB}) работают без неё. Ставится
она командой `aiw-ru models install transformer`, только если пользователь попросит.
Вариант легче — [LightGBM на признаках aiw-ru](https://huggingface.co/{LIGHTGBM_REPO}),
`aiw-ru models install lightgbm`.

## In English

A Russian AI-generated text detector. It estimates the probability that a
Russian text was written by a person or by an LLM such as {GENERATORS}
and others, which makes it usable as an AI text detector, a ChatGPT detector or an
"AI slop" filter for Russian. The model is `{p["base"]}` fine-tuned on the
Russian part of the LLMTrace corpus and exported to ONNX with {_weights(m)} weights:
`model.onnx` is a {_mb(p["size_mb"])} MB file that runs on CPU with onnxruntime, no torch
or GPU needed.{fp32_en}
Test ROC AUC {num(t["roc_auc"])}, accuracy {num(t["accuracy"])}. The rest of the card
is in Russian; usage is below.

## Данные

Русская часть [LLMTrace classification](https://huggingface.co/datasets/{DATASET})
([статья](https://arxiv.org/abs/2509.21269)), ревизия `{ds["revision"][:12]}`. Модель обучена на
{_texts_loc(ds["train"])} из train, лучший шаг выбран по ROC AUC на {_texts_loc(ds["valid"])}
из valid, итоговые цифры посчитаны на {_texts_loc(ds["test"])} из test, которые модель при
обучении не видела. Отчёт об обучении с выбором базы и командами для
воспроизведения лежит [на GitHub]({report}).

## Результаты на test

Текст считается написанным ИИ, если вероятность не меньше {num(THRESHOLD, 1)}. {_long_rule(m)}

{_classes_table(t)}

Сравнение с LightGBM на признаках aiw-ru и с оценкой правил детектора с порогом
{count(m["detector"]["threshold"])} на том же test:

{_compare_md(m)}

На valid accuracy {num(v["accuracy"])}, ROC AUC {num(v["roc_auc"])}.

{_cuts_md(m, "###")}

### Длинные тексты

Тексты длиннее {_tokens_gen(p["max_length"])}, в test их {count(m["long_texts"]["test"]["texts"])}.
Сравниваются вероятность по началу текста и среднее по окнам:

{_long_md(m)}

## Нормализация

В LLMTrace оформление говорит о том, откуда взят человеческий текст: переводы
строк, markdown, вид тире и кавычек, ё. Поэтому до токенизации текст проходит
нормализацию: разметка и типографика сводятся к простому виду, правила лежат в
`inference.json`. {_ablation(m)}

## Скорость и размер

{_speed_md(m)}

## Как пользоваться

{_usage_md(m, repo)}

## Обучение

{_params_md(m)}

## Ограничения

{_limitations_md(m)}

## Лицензия и данные

Модель распространяется по лицензии {LICENSE_NAMES.get(p["license"], p["license"])}, как и
[`{p["base"]}`](https://huggingface.co/{p["base"]}). Корпус LLMTrace — Apache 2.0, человеческие тексты в
нём собраны из сторонних источников со своими лицензиями.

{LLMTRACE_BIBTEX}

{BASE_CITATIONS.get(p["base"], "")}
"""


def _reproduce_md(m: dict) -> str:
    p = m["params"]
    run = short_name(p["base"])
    vm_run = f"/content/data/transformer/runs/{run}"
    local = f"~/.cache/aiw-ru/llmtrace/transformer/runs/{run}"
    train_args = (
        f"train --base {p['base']} --epochs {p['epochs']:g} --lr {p['lr']:g} --batch {p['batch']} "
        f"--max-length {p['max_length']} --evals-per-epoch {p['evals_per_epoch']} --patience {p['patience']}"
    )
    uv = "uv run --group train --group transformer"
    dirty = m["git_commit"].endswith("+правки")
    commit = m["git_commit"].removesuffix("+правки")
    checkout = (
        f"Модель обучена на коммите `{commit}`, к которому `scripts/train_transformer.py` и "
        "`scripts/colab_transformer.py` ещё не были закоммичены. Для повтора берите их версию из коммита, "
        "в котором появился этот отчёт."
        if dirty
        else f"Модель обучена на коммите `{commit}`."
    )
    env = md_table(["Компонент", "Версия"], [[k, f"`{val}`"] for k, val in m["environment"].items()])
    return f"""{checkout} Базовая модель — ревизия `{p["base_revision"]}`, корпус — ревизия
`{m["dataset"]["revision"]}`, её закрепляет `scripts/llmtrace.py`. Окружение:

{env}

Обучение идёт на GPU. В Colab на бесплатной T4 это делает
`scripts/colab_transformer.py` через официальный Colab CLI (google-colab-cli,
`colab sessions` должен работать). Корпус скачивается прямо на VM, через ваш
компьютер идут только исходники и готовая модель:

```bash
uv run scripts/colab_transformer.py up --gpu T4
uv run scripts/colab_transformer.py setup
uv run scripts/colab_transformer.py start --name full --job "{train_args}"
uv run scripts/colab_transformer.py status --name full
uv run scripts/colab_transformer.py fetch {vm_run} ~/.cache/aiw-ru/llmtrace/transformer/runs
```

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
{uv} scripts/train_transformer.py {train_args}
```

Дальше на своём CPU: ONNX и сверка с PyTorch, задержка, оценка на valid и test,
карточка и этот отчёт.

```bash
{uv} scripts/train_transformer.py export {local}
{uv} scripts/train_transformer.py evaluate {local}
{uv} scripts/train_transformer.py report {local}
```

Обучение на GPU не детерминировано до бита, повтор может разойтись в третьем
знаке. Пилот повторяется командой `pilot --base ИМЯ` с настройками по умолчанию
(`--limit 20000 --epochs 1`)."""


def report_body(m: dict, repo: str) -> str:
    """Отчёт об обучении для docs/models/: что сделано, на чём, как повторить."""
    p, ds = m["params"], m["dataset"]
    name = repo.split("/")[-1]
    lgb = (m.get("lightgbm") or {}).get("test", {}).get("roc_auc")
    pilot = m["pilot"][0] if m["pilot"] else {}
    finalists = ""
    if m.get("finalists"):
        finalists = f"""

Дальше на полном train обучены две базы: самая точная в пилоте и самая быстрая.
Середина, `sergeyzh/rubert-mini-frida`, медленнее tiny2 больше чем вдвое при
небольшом выигрыше в ROC AUC. ROC AUC и accuracy
посчитаны на полном valid у PyTorch и у ONNX int8; «сменил метку» — доля текстов
valid, где int8 и PyTorch расходятся по порогу $0.5$. Задержка — ONNX int8 на $512$
токенах в один поток: на M1 и на двух ядрах Xeon виртуальной машины Colab.

{_finalists_md(m)}"""
    why = f"\n\n{p['why']}" if p.get("why") else ""
    parity = m["parity"]
    return f"""# Отчёт об обучении: {name}

Файл пишет `scripts/train_transformer.py report` после полного обучения, руками
его не правят. Все числа в машиночитаемом виде лежат рядом, в
[{name}.json]({name}.json). Модель и карточка выложены на
[Hugging Face](https://huggingface.co/{repo}).

## Что это

Необязательная модель aiw-ru: дообученный энкодер `{p["base"]}` оценивает
вероятность, что русский текст написала языковая модель. Пользователь ставит её
командой `aiw-ru models install transformer`, после чего `aiw-ru classify`
выдаёт вероятность. В поставке ONNX с весами {_weights(m)} на {_mb(p["size_mb"])} МБ и
`inference.json`; для вывода нужны onnxruntime и tokenizers, torch не нужен.
На test ROC AUC {num(m["test"]["roc_auc"])}, у LightGBM на признаках aiw-ru {_auc(lgb)},
у оценки правил {num(m["detector"]["roc_auc"])}.

## Выбор базы

Кандидаты — небольшие русские энкодеры с открытой лицензией, которые можно
запускать на CPU. Полный [FRIDA](https://huggingface.co/ai-forever/FRIDA)
от ai-forever (MIT) — энкодер T5 на $823$ млн параметров, для CPU пользователя
он слишком тяжёл, поэтому в пилоте его дистилляция `sergeyzh/rubert-mini-frida`.

В пилоте все базы учились одинаково: {_nw(pilot.get("train", 0), "текст", "текста", "текстов")} train,
эпох {count(int(pilot.get("epochs") or 1))}, скорость `{pilot.get("lr") or 0:g}`, пачка {count(pilot.get("batch") or 0)},
`seed` {count(pilot.get("seed") or 0)}. ROC AUC посчитан на {_texts_loc(pilot.get("valid", 0))} valid,
задержка — ONNX int8 на $512$ токенах в один поток, {m["latency_cpu"]}.

{_pilot_md(m)}{finalists}{why}

## Данные

Русская часть [LLMTrace classification](https://huggingface.co/datasets/{ds["repo"]})
под Apache 2.0, [статья](https://arxiv.org/abs/2509.21269), ревизия набора
`{ds["revision"]}`. Части корпуса используются как есть. Модель учится на
{_texts_loc(ds["train"])} train. По {_nw(ds["valid"], "тексту", "текстам", "текстам")} valid выбирается лучший
шаг и срабатывает ранняя остановка. Test нужен только для итоговых цифр, в нём
{_nw(ds["test"], "текст", "текста", "текстов")}.

Метка «ИИ» стоит на любом тексте с участием модели: написанном с нуля
(`create`) и на человеческом тексте, который модель правила, сокращала или
дописывала (`update`, `delete`, `expand`). При обучении от текста длиннее
{_tokens_gen(p["max_length"])} остаётся только начало. Таких в train
{pct(p["truncated_share"]["train"])}, в valid {pct(p["truncated_share"]["valid"])}, в test
{pct(p["truncated_share"].get("test", 0))}.

## Нормализация

Оформление в LLMTrace говорит о том, откуда взят человеческий текст. Ниже
ROC AUC каждой приметы по отдельности на valid: $0.5$ значит, что примета ничего
не говорит, чем дальше от $0.5$ в любую сторону, тем больше говорит.

{_shortcuts_md(m)}

{_shortcuts_note(m)} Поэтому до токенизации текст проходит `normalize()`: правила ниже применяются
по порядку, шаблоны — `re` с флагом `MULTILINE`, в конце пробелы по краям
срезаются. Тот же список лежит в `inference.json`. Длину текста так не убрать,
см. таблицу по длине. {_ablation(m)}

{_rules_md()}

## Обучение

{_params_md(m)}

Проверки на valid по ходу обучения:

{_curve_md(m)}

## Результаты на test

Текст считается написанным ИИ, если вероятность не меньше {num(THRESHOLD, 1)}. {_long_rule(m)}

{_classes_table(m["test"])}

{_compare_md(m)}

На valid accuracy {num(m["valid"]["accuracy"])}, ROC AUC {num(m["valid"]["roc_auc"])}.

{_cuts_md(m, "###")}

### Детектор без модели

Оценка правил aiw-ru с порогом {count(m["detector"]["threshold"])}, то есть «много примет»:

{_classes_table(m["detector"])}

## Вывод на CPU

Ниже варианты одной и той же модели. int8 идёт в поставку, если на valid
теряет не больше $0.002$ ROC AUC; test в этом решении не участвует.

{_variants_md(m)}

Сверка с PyTorch на {_texts_loc(parity["fp32"]["texts"])} valid: у ONNX fp32 наибольшая
разница вероятностей {_sci(parity["fp32"]["max_abs_diff"])}, у int8 {num(parity["int8"]["max_abs_diff"])},
метки int8 совпадают у {pct(parity["int8"]["same_label"])} текстов.

Правило для длинных текстов выбрано по valid, в test таких текстов
{count(m["long_texts"]["test"]["texts"])}:

{_long_md(m)}

{_speed_md(m)}

## Как пользоваться

{_usage_md(m, repo)}

## Как воспроизвести

{_reproduce_md(m)}

## Ограничения

{_limitations_md(m)}

## Ссылки

{LLMTRACE_BIBTEX}

{BASE_CITATIONS.get(p["base"], "")}
"""


def card(m: dict, repo: str = REPO) -> ModelCard:
    p = m["params"]
    results = eval_results(m)
    for r in results:
        r.source_name = "aiw-ru scripts/train_transformer.py"
        r.source_url = f"{GITHUB}/blob/master/scripts/train_transformer.py"
    tags = [t for t in TAGS if t not in ("lightgbm", "tabular-features")]
    tags += ["modernbert" if "modernbert" in p["base"].lower() else "bert", "onnx"]
    if _weights(m) == "int8":
        tags.append("int8")
    data = ModelCardData(
        language="ru",
        license=p["license"],
        library_name="onnx",
        pipeline_tag=TASK,
        base_model=p["base"],
        # Иначе Hub по ONNX int8 считает модель квантованной копией базы, а она дообучена.
        base_model_relation="finetune",
        tags=tags,
        datasets=[DATASET],
        metrics=["accuracy", "f1", "precision", "recall", "roc_auc"],
        model_name=repo.split("/")[-1],
        eval_results=results,
        inference=False,
    )
    return ModelCard(f"---\n{data.to_yaml()}\n---\n\n{hub_math(card_body(m, repo))}")


def finalist(run_dir: Path, data: Path) -> dict:
    """Строка сравнения полных запусков: полный valid у PyTorch и int8, размер и задержка int8.

    Ответы int8 на valid берутся из valid-probs-onnx_int8.npy, который пишет evaluate;
    задержка на x86 — из export-x86.json, если он есть.
    """
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    exp = json.loads((run_dir / "export.json").read_text(encoding="utf-8"))
    lat = {(r["runtime"], r["tokens"], r["threads"]): r["ms"] for r in exp["latency"]}
    row: dict[str, Any] = {
        "base": run["params"]["base"],
        "valid": run["dataset"]["valid"],
        "valid_roc_auc": run["valid"]["roc_auc"],
        "valid_accuracy": run["valid"]["accuracy"],
        "onnx_fp32_mb": exp["sizes_mb"]["onnx_fp32"],
        "onnx_int8_mb": exp["sizes_mb"]["onnx_int8"],
        "int8_ms_128_1": lat.get(("onnxruntime int8", 128, 1)),
        "int8_ms_512_1": lat.get(("onnxruntime int8", 512, 1)),
        "fp32_ms_512_1": lat.get(("onnxruntime fp32", 512, 1)),
    }
    x86 = run_dir / "export-x86.json"
    if x86.exists():
        lx = {(r["runtime"], r["tokens"], r["threads"]): r["ms"] for r in json.loads(x86.read_text())["latency"]}
        row["x86_int8_ms_512_1"] = lx.get(("onnxruntime int8", 512, 1))
    int8 = run_dir / "valid-probs-onnx_int8.npy"
    if int8.exists():
        y = np.array([json.loads(line)["label"] == "ai" for line in load(data, "classification", "valid", 0)])
        p, ref = np.load(int8), np.load(run_dir / "valid-probs.npy")
        row |= {
            "valid_roc_auc_int8": auc(y, p),
            "valid_accuracy_int8": float((y == (p >= THRESHOLD)).mean()),
            "int8_label_changes": float(((p >= THRESHOLD) != (ref >= THRESHOLD)).mean()),
        }
    return row


def cmd_report(args: argparse.Namespace) -> None:
    """README.md в hub/ рядом с файлами модели, отчёт и метрики в docs/models/."""
    run_dir: Path = args.folder
    m = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    # Замер на x86 мог появиться после evaluate.
    m |= x86_latency(run_dir)
    if args.why:
        m["params"]["why"] = args.why
    if args.finalist:
        m["finalists"] = [finalist(d, args.data) for d in args.finalist]
    (run_dir / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    hub_dir = bundle_hub(run_dir, m["params"]["shipped_file"])
    card(m, args.repo).save(hub_dir / "README.md")
    print(f"Для Hugging Face: {hub_dir}", flush=True)
    name = args.repo.split("/")[-1]
    if m["dataset"]["limit"]:
        # Пилот на выборке в docs/ не идёт: отчёт только в папке запуска.
        (run_dir / "report.md").write_text(report_body(m, args.repo), encoding="utf-8")
        print(f"Обучение на выборке: отчёт только в {run_dir / 'report.md'}", flush=True)
        return
    docs = ROOT / DOCS
    docs.mkdir(parents=True, exist_ok=True)
    (docs / f"{name}.md").write_text(report_body(m, args.repo), encoding="utf-8")
    (docs / f"{name}.json").write_text(json.dumps(m, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Отчёт: {DOCS / f'{name}.md'} и {DOCS / f'{name}.json'}", flush=True)


# ─── Командная строка ───────────────────────────────────────────────────


def add_common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--data", type=Path, default=data_dir(), help="папка с частями корпуса")
    ap.add_argument("--out-root", type=Path, help="папка запусков и кэша (по умолчанию transformer/ рядом с данными)")


def add_fit(ap: argparse.ArgumentParser, pilot: bool) -> None:
    add_common(ap)
    ap.add_argument("--base", default=DEFAULT_BASE, help="исходная модель на Hugging Face")
    ap.add_argument("--limit", type=int, default=20000 if pilot else 0, help="взять N текстов из train")
    ap.add_argument("--eval-limit", type=int, help="взять N текстов из valid и test (по умолчанию половина --limit)")
    ap.add_argument("--epochs", type=float, default=1 if pilot else 3, help="эпох обучения, можно дробное число")
    ap.add_argument("--lr", type=float, default=1e-4, help="пиковая скорость обучения AdamW")
    ap.add_argument("--batch", type=int, default=32, help="текстов в пачке")
    ap.add_argument("--eval-batch", type=int, default=128, help="текстов в пачке при проверке")
    ap.add_argument("--max-length", type=int, default=512, help="токенов на текст, дальше текст обрезается")
    ap.add_argument("--warmup", type=float, default=0.06, help="доля шагов разогрева скорости обучения")
    ap.add_argument("--weight-decay", type=float, default=0.01, help="затухание весов AdamW")
    ap.add_argument("--patience", type=int, default=0 if pilot else 3, help="проверок без роста AUC до остановки")
    ap.add_argument("--evals-per-epoch", type=int, default=1 if pilot else 4, help="проверок на valid за эпоху")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"], help="где учить")
    ap.add_argument("--precision", default="auto", choices=["auto", "fp32", "fp16", "bf16"], help="точность")
    ap.add_argument("--raw", action="store_true", help="без нормализации текста, для сравнения")
    ap.add_argument("--out", type=Path, help="папка запуска")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    add_fit(sub.add_parser("pilot", help="короткое обучение на выборке, без test"), pilot=True)
    add_fit(sub.add_parser("train", help="полное обучение и ответы PyTorch на test"), pilot=False)
    ex = sub.add_parser("export", help="ONNX fp32 и int8, сверка с PyTorch, задержка на CPU")
    add_common(ex)
    ex.add_argument("folder", type=Path, help="папка запуска с model/ и run.json")
    ex.add_argument("--json", default="export.json", help="куда записать итог в папке запуска")
    ev = sub.add_parser("evaluate", help="ONNX на CPU по valid и test, metrics.json и inference.json")
    add_common(ev)
    ev.add_argument("folder", type=Path, help="папка запуска после export")
    ev.add_argument("--jobs", type=int, default=0, help="процессов детектора (по умолчанию по числу ядер)")
    ev.add_argument("--int8-max-drop", type=float, default=0.002, help="допустимая потеря ROC AUC у int8 на valid")
    ev.add_argument("--skip-install", action="store_true", help="не мерить размер установки зависимостей")
    rep = sub.add_parser("report", help="карточка для Hugging Face и отчёт в docs/models/ из metrics.json")
    add_common(rep)
    rep.add_argument("folder", type=Path, help="папка запуска после evaluate")
    rep.add_argument("--repo", default=REPO, help="репозиторий модели на Hugging Face")
    rep.add_argument("--why", help="почему выбрана эта база: абзац для отчёта")
    rep.add_argument("--finalist", type=Path, action="append", help="папка полного запуска для сравнения баз")
    args = ap.parse_args(argv)
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

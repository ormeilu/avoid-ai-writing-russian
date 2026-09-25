"""Тесты scripts/train_transformer.py: нормализация, пачки, окна, метрики и вывод по inference.json.

Нужны torch, transformers, onnxruntime и tokenizers из группы transformer и
LightGBM из группы train; без них тесты пропускаются. Сеть не нужна.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("onnxruntime")
pytest.importorskip("lightgbm")
tokenizers = pytest.importorskip("tokenizers")

import train_transformer as tt
from train import Part, breakdowns

NBSP = chr(0x00A0)
ZWSP = chr(0x200B)
EM_DASH = chr(0x2014)


def test_normalize_markdown():
    text = "# Заголовок\n\n- **Первый** пункт\n2. Второй [ссылка](https://x.ru)\n> цитата\n\n---\n`код`"
    assert tt.normalize(text) == "Заголовок Первый пункт Второй ссылка цитата код"


def test_normalize_typography():
    text = f"«Ёлка»{NBSP}{EM_DASH} это{ZWSP} ёж… „да“ ‘нет’"
    assert tt.normalize(text) == '"Елка" - это еж... "да" \'нет\''


def test_normalize_keeps_words_and_snake_case():
    assert tt.normalize("  snake_case   и\n\nдефис-слово  ") == "snake_case и дефис-слово"
    assert tt.normalize("| a | b |\n|---|:-:|\n| 1 | 2 |") == "a b 1 2"


def test_normalize_rules_are_plain_regex():
    """inference.json хранит правила строками: вывод без этого модуля должен дать то же."""
    text = f"## Итоги\n\n* «Важно» {EM_DASH} <b>ё</b>\n\n1) пункт"
    out = text
    for pattern, replacement, _ in tt.NORMALIZE_RULES:
        out = re.sub(pattern, replacement, out, flags=re.MULTILINE)
    assert out.strip() == tt.normalize(text) == 'Итоги "Важно" - е пункт'


class FakeTokenizer:
    """Токенизатор по пробелам: [CLS]=1, [SEP]=2, [PAD]=0, слово — его длина + 10."""

    cls_token_id, sep_token_id, pad_token_id = 1, 2, 0
    name_or_path = "fake"

    def __call__(self, texts, add_special_tokens=True, truncation=False):
        one = isinstance(texts, str)
        ids = [[1, *(len(w) + 10 for w in t.split()), 2] for t in ([texts] if one else texts)]
        return {"input_ids": ids[0] if one else ids}


def test_encode_truncates_with_sep():
    tok = FakeTokenizer()
    seqs, truncated = tt.encode_texts(["а бб ввв гггг", "короткий"], tok, 4, raw=False)
    assert seqs == [[1, 11, 12, 2], [1, 18, 2]]
    assert truncated == [True, False]
    assert tt.special_ids(tok) == (1, 2, 0)


def test_batches_and_collate():
    lengths = np.array([5, 1, 3, 8, 2, 7, 4])
    enc = tt.Encoded(
        ids=np.arange(lengths.sum(), dtype=np.int32),
        offsets=np.concatenate([[0], np.cumsum(lengths)]),
        truncated=np.zeros(7, dtype=bool),
        y=np.array([0, 1, 0, 1, 0, 1, 0], dtype=np.int8),
        genre=np.array(["a"] * 7),
        prompt=np.array([""] * 7),
        generator=np.array([""] * 7),
    )
    for rng in (None, np.random.default_rng(1)):
        got = np.concatenate(tt.batches(enc.lengths, 3, rng))
        assert sorted(got.tolist()) == list(range(7))
    first = tt.batches(enc.lengths, 3, None)[0]
    assert enc.lengths[first].tolist() == [1, 2, 3]
    ids, mask = tt.collate(enc, first, pad_id=-1, multiple=4)
    assert ids.shape == (3, 4) and mask.sum() == 6
    assert ids[0].tolist() == [enc.seq(1)[0], -1, -1, -1]
    sub = enc.subset(np.array([3, 0]))
    assert sub.lengths.tolist() == [8, 5] and sub.y.tolist() == [1, 0]


def test_windows():
    ws = tt.windows(list(range(10, 17)), 5, 1, 2, limit=8)
    assert ws == [[1, 10, 11, 12, 2], [1, 13, 14, 15, 2], [1, 16, 2]]
    assert tt.windows([], 5, 1, 2, limit=8) == [[1, 2]]
    assert len(tt.windows(list(range(100)), 5, 1, 2, limit=2)) == 2


def test_quick_metrics_and_breakdowns():
    y = np.array([0, 0, 1, 1, 1, 0], dtype=np.int8)
    p = np.array([0.1, 0.6, 0.8, 0.4, 0.9, 0.2])
    m = tt.quick(y, p)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["roc_auc"] == pytest.approx(8 / 9)
    part = Part(
        x=np.zeros((6, 0)),
        y=y,
        score=np.array([10, 50, 60, 20, 70, 5], dtype=np.float32),
        words=np.array([10, 60, 200, 500, 40, 160]),
        genre=np.array(["news", "news", "news", "poetry", "poetry", "poetry"]),
        prompt=np.array(["", "", "create", "update", "create", ""]),
        generator=np.array(["", "", "gpt-4o", "gpt-4o", "qwen", ""]),
    )
    cuts = breakdowns(part, p)
    assert [g["genre"] for g in cuts["genres"]] == ["news", "poetry"]
    assert sum(b["n"] for b in cuts["lengths"]) == 6
    assert {x["prompt_type"]: x["recall"] for x in cuts["prompt_types"]} == {"create": 1.0, "update": 0.0}
    assert cuts["generators"][0] == {"model": "gpt-4o", "n": 2, "recall": 0.5}


def test_device_and_precision():
    import torch

    cpu = tt.pick_device("cpu")
    assert cpu.type == "cpu"
    assert tt.pick_precision(cpu, "auto")[0] == "fp32"
    assert tt.pick_precision(torch.device("mps"), "auto")[0] == "fp32"
    assert tt.pick_precision(cpu, "fp16") == ("fp16", "задана ключом --precision")


class FakeSession:
    """Сессия ONNX Runtime, у которой логит «ИИ» — доля токена 16 в окне."""

    def run(self, names, feeds):
        ids, mask = feeds["input_ids"], feeds["attention_mask"]
        share = ((ids == 16) & (mask == 1)).sum(axis=1) / mask.sum(axis=1)
        return [np.stack([np.zeros_like(share), 20 * share - 2], axis=1).astype(np.float32)]


def word_tokenizer() -> tokenizers.Tokenizer:
    vocab = {"[PAD]": 0, "[UNK]": 3, "[CLS]": 1, "[SEP]": 2, "e": 10, "ai": 16, '"': 11, "-": 12}
    tok = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    return tok


def test_predict_text_follows_spec(tmp_path: Path):
    spec = tt.inference_spec({"max_length": 6, "normalize": True}, FakeTokenizer(), "onnx/model.onnx", "head")
    assert spec["max_windows"] == 1 and spec["threshold"] == 0.5
    assert spec["max_chars"] == 6 * tt.CHARS_PER_TOKEN * tt.MAX_WINDOWS
    json.loads(json.dumps(spec))  # правила сериализуются в inference.json
    sess, tok = FakeSession(), word_tokenizer()
    head = tt.predict_text("ai ai ai ai\n\ne e e e e e e e", sess, tok, spec)
    spec_w = {**spec, "long_texts": "mean_of_windows", "max_windows": 8}
    mean = tt.predict_text("ai ai ai ai\n\ne e e e e e e e", sess, tok, spec_w)
    assert head > 0.99 and mean == pytest.approx((head + 1 / (1 + np.exp(2)) * 2) / 3, abs=1e-6)
    assert tt.predict_text("ё", sess, tok, spec) == tt.predict_text("е", sess, tok, spec)


def test_pilot_rows(tmp_path: Path):
    run_dir = tmp_path / "pilot" / "tiny-20000"
    run_dir.mkdir(parents=True)
    run = {
        "dataset": {"train": 20000, "valid": 10000},
        "valid": {"roc_auc": 0.9, "accuracy": 0.8},
        "params": {
            "base": "cointegrated/rubert-tiny2",
            "base_license": "mit",
            "normalize": True,
            "parameters": 29_000_000,
            "train_seconds": 60.0,
            "texts_per_second": 333.0,
            "gpu": "Tesla T4",
        },
    }
    export = {
        "sizes_mb": {"onnx_fp32": 117.0, "onnx_int8": 29.6},
        "latency": [
            {"runtime": "onnxruntime int8", "tokens": 512, "threads": 1, "ms": 39.0},
            {"runtime": "onnxruntime int8", "tokens": 512, "threads": 8, "ms": 23.0},
        ],
    }
    (run_dir / "run.json").write_text(json.dumps(run), encoding="utf-8")
    (run_dir / "export.json").write_text(json.dumps(export), encoding="utf-8")
    [row] = tt.pilot_rows(tmp_path)
    assert row["valid_roc_auc"] == 0.9 and row["onnx_int8_mb"] == 29.6
    assert row["int8_ms_512_1"] == 39.0 and row["int8_ms_512_all"] == 23.0


def test_missing_gpu_fails_loudly(monkeypatch: pytest.MonkeyPatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(SystemExit, match="--device cpu"):
        tt.pick_device("auto")


def fake_metrics() -> dict:
    """metrics.json в той схеме, что пишет evaluate, с правдоподобными числами."""
    cls = {
        "accuracy": 0.9,
        "classes": {
            "human": {"precision": 0.88, "recall": 0.86, "f1": 0.87, "support": 400},
            "ai": {"precision": 0.91, "recall": 0.93, "f1": 0.92, "support": 600},
        },
        "macro": {"precision": 0.895, "recall": 0.895, "f1": 0.895},
    }
    q = {"roc_auc": 0.96, "accuracy": 0.9, "logloss": 0.25}
    pilot = {
        "base": "cointegrated/rubert-tiny2",
        "license": "mit",
        "normalize": True,
        "train": 20000,
        "valid": 10000,
        "valid_roc_auc": 0.957,
        "valid_accuracy": 0.89,
        "parameters": 29_194_000,
        "train_seconds": 45.0,
        "texts_per_second": 470.0,
        "gpu": "Tesla T4",
        "onnx_int8_mb": 29.6,
        "int8_ms_512_1": 39.0,
        "epochs": 1,
        "lr": 1e-4,
        "batch": 32,
        "seed": 1,
    }
    return {
        "created": "2026-09-25",
        "git_commit": "abc1234",
        "environment": {"python": "3.13.15", "torch": "2.14.0+cu130"},
        "dataset": {
            "repo": "iitolstykh/LLMTrace_classification",
            "revision": "252e21e3",
            "limit": 0,
            "train": 237929,
            "valid": 49747,
            "test": 52521,
        },
        "params": {
            "base": "cointegrated/rubert-tiny2",
            "base_revision": "e8ed3b0c8bbf4fb6",
            "base_license": "mit",
            "license": "mit",
            "parameters": 29_194_000,
            "max_length": 512,
            "truncated_share": {"train": 0.1, "valid": 0.1, "test": 0.1},
            "epochs": 3,
            "lr": 1e-4,
            "batch": 32,
            "warmup": 0.06,
            "weight_decay": 0.01,
            "grad_clip": 1.0,
            "patience": 3,
            "evals_per_epoch": 4,
            "seed": 1,
            "gpu": "Tesla T4",
            "device": "cuda",
            "precision": "fp16",
            "precision_reason": "T4",
            "best_step": 20000,
            "best_epoch": 2.7,
            "early_stopped": False,
            "train_seconds": 1800.0,
            "texts_per_second": 470.0,
            "shipped": "onnx_int8",
            "shipped_file": "onnx/model_int8.onnx",
            "size_mb": 29.6,
        },
        "test": {**cls, "roc_auc": 0.97, "roc_auc_created": 0.975},
        "valid": {"accuracy": 0.9, "roc_auc": 0.97},
        "detector": {**cls, "threshold": 40, "roc_auc": 0.61, "roc_auc_created": 0.59},
        "genres": [{"genre": "news", "human": 10, "ai": 12, "accuracy": 0.8, "roc_auc": 0.9, "detector_roc_auc": 0.55}],
        "lengths": [{"bucket": "0–49", "n": 22, "accuracy": 0.8, "roc_auc": 0.9}],
        "prompt_types": [{"prompt_type": "create", "n": 12, "recall": 0.9}],
        "generators": [{"model": "gpt-4o", "n": 12, "recall": 0.9}],
        "variants": {"torch": {"valid": q, "test": q}, "onnx_fp32": {"valid": q, "test": q}, "onnx_int8": {"valid": q}},
        "long_texts": {
            "rule": "head",
            "valid": {"texts": 5, "head": q, "mean_of_windows": q},
            "test": {"texts": 6, "head": q, "mean_of_windows": q},
        },
        "parity": {
            "fp32": {"texts": 300, "max_abs_diff": 1e-7, "same_label": 1.0},
            "int8": {"texts": 300, "max_abs_diff": 0.01, "same_label": 0.99},
        },
        "sizes_mb": {"onnx_fp32": 117.0, "onnx_int8": 29.6, "safetensors": 111.4, "tokenizer": 2.3},
        "latency": [
            {"runtime": "onnxruntime int8", "tokens": 512, "threads": 1, "ms": 39.0},
            {"runtime": "torch fp32", "tokens": 512, "threads": 1, "ms": 20.7},
        ],
        "latency_cpu": "Apple M1, 8 ядер",
        "install": [
            {"stack": "onnxruntime + tokenizers", "platform": "x86_64-manylinux_2_28", "mb": 156.4},
            {"stack": "onnxruntime + tokenizers", "platform": "aarch64-apple-darwin", "mb": 127.0},
            {"stack": "torch + transformers", "platform": "aarch64-apple-darwin", "mb": 669.0},
            {"stack": "torch + transformers", "platform": "x86_64-manylinux_2_28", "mb": 5649.0},
            {"stack": "torch (CPU) + transformers", "platform": "x86_64-manylinux_2_28", "mb": 908.0},
        ],
        "shortcuts": [{"cue": "длина, знаков", "roc_auc": 0.6, "separation": 0.6}],
        "curve": [{"step": 100, "epoch": 0.5, "train_loss": 0.4, "roc_auc": 0.95, "accuracy": 0.88, "logloss": 0.3}],
        "pilot": [pilot, {**pilot, "normalize": False, "valid_roc_auc": 0.97}],
        "finalists": [
            {
                "base": "cointegrated/rubert-tiny2",
                "valid": 49747,
                "valid_roc_auc": 0.9878,
                "valid_accuracy": 0.946,
                "valid_roc_auc_int8": 0.9877,
                "valid_accuracy_int8": 0.9457,
                "int8_label_changes": 0.002,
                "onnx_int8_mb": 29.7,
                "int8_ms_512_1": 40.1,
                "x86_int8_ms_512_1": 101.1,
            }
        ],
        "inference": {"threshold": 0.5, "long_texts": "mean_of_windows", "max_windows": 8},
        "lightgbm": {
            "repo": "toiletsandpaper/russian-ai-text-detector-lightgbm",
            "test": {**cls, "roc_auc": 0.943, "roc_auc_created": 0.946},
        },
    }


def test_report_from_fake_metrics():
    text = tt.report_body(fake_metrics(), tt.REPO)
    for part in ("# Отчёт об обучении: russian-ai-text-detector-bert", "## Выбор базы", "## Нормализация"):
        assert part in text
    assert "## Как воспроизвести" in text and "export-x86.json" in text
    assert "| `cointegrated/rubert-tiny2` | $0.9878$ |" in text  # таблица финалистов
    assert "torch считает быстрее onnxruntime" in text and r"$5.3\text{–}5.8$ раза" in text
    assert "https://arxiv.org/abs/2509.21269" in text
    assert "252e21e3" in text and "abc1234" in text
    assert "база `cointegrated/rubert-tiny2` без нормализации дала ROC AUC $0.970$" in text
    assert "{p[" not in text and "{m[" not in text
    rules = [line for line in text.splitlines() if "строка-разделитель таблицы" in line]
    assert rules and rules[0].count(" | ") == 2  # черты шаблона не рвут строку таблицы
    # Перед формулой только пробел или начало строки, иначе Hub и GitHub её не узнают.
    assert not re.search(r"\S\\\\\(", tt.hub_math(text))


def test_card_from_fake_metrics():
    m = fake_metrics()
    card = tt.card(m)
    meta = card.data.to_dict()
    assert meta["base_model"] == "cointegrated/rubert-tiny2"
    assert meta["license"] == "mit" and meta["library_name"] == "onnx"
    assert "lightgbm" not in meta["tags"] and {"bert", "onnx", "int8"} <= set(meta["tags"])
    source = meta["model-index"][0]["results"][0]["source"]["url"]
    assert source.endswith("scripts/train_transformer.py")
    body = card.text.lstrip()
    assert body.startswith("# Детектор ИИ-текста для русского языка") and "## In English" in body
    assert "aiw-ru models install transformer" in body and "habr.com" in body
    assert "по первым \\\\(8\\\\) окнам" in body
    assert "$" not in body.split("```")[0]  # формулы переписаны для Hub
    assert not re.search(r"\S\\\\\(", body)

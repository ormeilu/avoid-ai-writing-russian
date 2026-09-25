"""Тесты scripts/train_transformer.py: нормализация, пачки, окна, метрики и вывод по inference.json.

Нужны torch, transformers, onnxruntime и tokenizers из группы transformer и
LightGBM из группы train; без них тесты пропускаются. Сеть не нужна.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from aiw_ru import models

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
    ws = tt.windows(list(range(10, 17)), 5, [1], [2], limit=8)
    assert ws == [[1, 10, 11, 12, 2], [1, 13, 14, 15, 2], [1, 16, 2]]
    assert tt.windows([], 5, [1], [2], limit=8) == [[1, 2]]
    assert len(tt.windows(list(range(100)), 5, [1], [2], limit=2)) == 2
    # Префикс задачи и служебные токены замороженного энкодера: окно короче на их длину.
    assert tt.windows(list(range(10, 14)), 6, [1, 7, 8], [2], limit=8) == [[1, 7, 8, 10, 11, 2], [1, 7, 8, 12, 13, 2]]


def test_stitch_windows():
    """Окна длинных текстов: первое берётся из эмбеддинга начала, остальные идут следом."""
    head = np.arange(10, dtype=np.float16).reshape(5, 2)
    more = 100 + np.arange(6, dtype=np.float16).reshape(3, 2)
    emb, counts = tt.stitch_windows(head, np.array([1, 3]), [2, 1], more)
    assert counts.tolist() == [3, 2]
    assert emb.tolist() == [[2, 3], [100, 101], [102, 103], [6, 7], [104, 105]]
    empty, none = tt.stitch_windows(head, np.array([], dtype=int), [], more[:0])
    assert empty.shape == (0, 2) and none.tolist() == []


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
    spec = tt.inference_spec({"max_length": 6, "normalize": True}, {"cls_id": 1, "sep_id": 2, "pad_id": 0}, "head")
    assert spec["max_windows"] == 1 and spec["threshold"] == 0.5
    assert spec["max_chars"] == 6 * tt.CHARS_PER_TOKEN * tt.MAX_WINDOWS
    json.loads(json.dumps(spec))  # правила сериализуются в inference.json
    sess, tok = FakeSession(), word_tokenizer()
    head = tt.predict_text("ai ai ai ai\n\ne e e e e e e e", sess, tok, spec)
    spec_w = {**spec, "long_texts": "mean_of_windows", "max_windows": 8}
    mean = tt.predict_text("ai ai ai ai\n\ne e e e e e e e", sess, tok, spec_w)
    assert head > 0.99 and mean == pytest.approx((head + 1 / (1 + np.exp(2)) * 2) / 3, abs=1e-6)
    assert tt.predict_text("ё", sess, tok, spec) == tt.predict_text("е", sess, tok, spec)


def frozen_encoder_check(tmp_path: Path) -> None:
    """Замороженный T5 с префиксом задачи: один граф ONNX, окна с prefix_ids и suffix_ids, как у PyTorch."""
    import torch
    from transformers import T5Config, T5EncoderModel

    vocab = {"[PAD]": 0, "[UNK]": 3, "</s>": 2, "e": 10, "ai": 16, "categorize": 20, ":": 21}
    tok = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tok.post_processor = tokenizers.processors.TemplateProcessing(single="$A </s>", special_tokens=[("</s>", 2)])
    base, md = tmp_path / "t5", tmp_path / "run" / "model"
    md.mkdir(parents=True)
    tok.save(str(md / "tokenizer.json"))
    torch.manual_seed(0)
    cfg = T5Config(vocab_size=32, d_model=16, d_kv=4, d_ff=32, num_layers=1, num_heads=4, pad_token_id=0)
    T5EncoderModel(cfg).save_pretrained(base)
    prefix, suffix = tt.wrap_ids(tt.inference_tokenizer(md / "tokenizer.json"), "categorize: ")
    assert (prefix, suffix) == ([20, 21], [2])
    frozen = {"base": str(base), "pooling": "cls", "prefix_ids": prefix, "suffix_ids": suffix, "pad_id": 0}
    (md / tt.FROZEN).write_text(json.dumps(frozen), encoding="utf-8")
    np.savez(md / "head.npz", weight=np.linspace(-3, 3, 16, dtype=np.float32), bias=np.float32(0.2))
    tt.export_onnx(md, tmp_path / "model.onnx")

    spec = tt.inference_spec({"max_length": 6, "normalize": True}, tt.model_wrap(md), "mean_of_windows", "fp32")
    assert spec["prefix_ids"] == [20, 21] and spec["suffix_ids"] == [2] and "cls_id" not in spec
    text = "ai e ai e e ai e"
    btok = tt.inference_tokenizer(md / "tokenizer.json")
    wins = tt.windows(btok.encode(text, add_special_tokens=False).ids, 6, prefix, suffix, limit=8)
    assert wins[0] == [20, 21, 16, 10, 16, 2] and len(wins) == 3
    model = tt.load_torch_classifier(md)
    with torch.no_grad():
        ref = [
            torch.softmax(model(torch.tensor([w]), torch.ones(1, len(w), dtype=torch.long)).logits, -1)[0, 1]
            for w in wins
        ]
    got = tt.predict_text(text, tt.session(tmp_path / "model.onnx", 1), btok, spec)
    assert got == pytest.approx(float(np.mean(ref)), abs=1e-5)


def test_frozen_encoder_onnx_follows_spec(tmp_path: Path):
    """Проверка frozen_encoder_check в отдельном процессе.

    torch и LightGBM приносят каждый свой OpenMP: после тестов с LightGBM экспорт torch в том же
    процессе виснет в libomp.
    """
    here = Path(__file__).resolve().parent
    paths = [here, here.parent / "scripts", here.parent / "evals", os.environ.get("PYTHONPATH", "")]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(str(x) for x in paths if str(x)), "OMP_NUM_THREADS": "1"}
    code = (
        "import sys; from pathlib import Path; import test_transformer as t; t.frozen_encoder_check(Path(sys.argv[1]))"
    )
    r = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)], env=env, capture_output=True, text=True, timeout=600, check=False
    )
    assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]


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
    assert "| `cointegrated/rubert-tiny2` | `transformer` | $0.9878$ |" in text  # таблица финалистов
    assert "Сам FRIDA готовится отдельной моделью" in text and "у кандидатов ниже $29$ млн" in text
    assert "torch считает быстрее onnxruntime" in text and r"$5.3\text{–}5.8$ раза" in text
    assert "https://arxiv.org/abs/2509.21269" in text
    assert "252e21e3" in text and "abc1234" in text
    assert "база `cointegrated/rubert-tiny2` без нормализации дала ROC AUC $0.970$" in text
    assert "{p[" not in text and "{m[" not in text
    rules = [line for line in text.splitlines() if "строка-разделитель таблицы" in line]
    assert rules and rules[0].count(" | ") == 2  # черты шаблона не рвут строку таблицы
    # Перед формулой только пробел или начало строки, иначе Hub и GitHub её не узнают.
    assert not re.search(r"\S\\\\\(", tt.hub_math(text))


def test_report_names_three_finalists():
    """Три базы на полном train: быстрая по умолчанию, середина и точная, у каждой веса и задержка поставки."""
    m = fake_metrics()
    tiny = m["finalists"][0]
    m["finalists"] = [
        {**tiny, "weights": "int8", "shipped_mb": 29.7, "shipped_ms_512_1": 39.8},
        {
            **tiny,
            "base": "sergeyzh/rubert-mini-frida",
            "valid_roc_auc": 0.9916,
            "int8_ms_512_1": 94.7,
            "weights": "fp32",
            "shipped_mb": 129.9,
            "shipped_ms_512_1": 137.2,
        },
        {
            **tiny,
            "base": "deepvk/RuModernBERT-small",
            "valid_roc_auc": 0.9935,
            "int8_ms_512_1": 161.5,
            "weights": "fp32",
            "shipped_mb": 139.6,
            "shipped_ms_512_1": 272.8,
        },
    ]
    text = tt.report_body(m, tt.REPO)
    assert "На полном train обучены $3$ базы из пилота" in text and "по обоим, `sergeyzh/rubert-mini-frida`." in text
    assert "`transformer` ставится по умолчанию" in text and "`modernbert` точнее всех" in text
    assert "`mini-frida` — промежуточный вариант" in text and "две базы" not in text
    assert "| `sergeyzh/rubert-mini-frida` | `mini-frida` | $0.9916$ |" in text and "| fp32 | $130$ | $137.2$ |" in text


def test_card_from_fake_metrics():
    m = fake_metrics()
    card = tt.card(m)
    meta = card.data.to_dict()
    assert meta["base_model"] == "cointegrated/rubert-tiny2"
    assert meta["license"] == "mit" and meta["library_name"] == "onnx"
    assert meta["base_model_relation"] == "finetune"
    assert "lightgbm" not in meta["tags"] and {"bert", "onnx", "int8"} <= set(meta["tags"])
    source = meta["model-index"][0]["results"][0]["source"]["url"]
    assert source.endswith("scripts/train_transformer.py")
    body = card.text.lstrip()
    assert body.startswith("# Детектор ИИ-текста для русского языка") and "## In English" in body
    assert "aiw-ru models install transformer" in body and "habr.com" in body
    assert "по первым \\\\(8\\\\) окнам" in body
    assert "$" not in body.split("```")[0]  # формулы переписаны для Hub
    assert not re.search(r"\S\\\\\(", body)


def test_card_and_report_for_fp32_bundle():
    """Точная модель: веса fp32 выбраны заранее, int8 лежит рядом, в карточке сравнение с другими моделями."""
    m = fake_metrics()
    m["params"] |= {
        "base": "deepvk/RuModernBERT-small",
        "base_license": "apache-2.0",
        "license": "apache-2.0",
        "shipped": "onnx_fp32",
        "shipped_file": "onnx/model.onnx",
        "size_mb": 139.6,
        "weights_choice": "fp32",
        "int8_label_changes": {"valid": 0.011},
        "int8_valid_only": True,
    }
    m["test"]["roc_auc"] = 0.993
    m["latency"] = [
        {"runtime": "onnxruntime fp32", "tokens": 512, "threads": 1, "ms": 272.0},
        {"runtime": "torch fp32", "tokens": 512, "threads": 1, "ms": 67.0},
    ]
    m["sizes_mb"] |= {"onnx_fp32": 139.6, "onnx_int8": 36.1}
    tiny = {**m["test"], "roc_auc": 0.9869}
    m["related"] = [
        {
            "name": "transformer",
            "repo": tt.REPO,
            "base": "cointegrated/rubert-tiny2",
            "weights": "int8",
            "size_mb": 29.7,
            "ms_512_1": 39.8,
            "x86_ms_512_1": 101.1,
            "test": tiny,
        }
    ]
    card = tt.card(m)
    meta = card.data.to_dict()
    assert meta["model-index"][0]["name"] == "russian-ai-text-detector-modernbert"
    assert meta["license"] == "apache-2.0" and meta["base_model_relation"] == "finetune"
    assert "modernbert" in meta["tags"] and "int8" not in meta["tags"]
    body = card.text
    assert "aiw-ru models install modernbert" in body and "aiw-ru classify --model modernbert текст.md" in body
    assert "`model_int8.onnx`" in body and "## Какую модель выбрать" in body
    assert "точный вариант" in body
    choose = body.split("## Какую модель выбрать")[1].split("\n## ")[0]
    # Таблица в порядке aiw-ru, по строке «когда брать» на модель, без сплошного абзаца сравнений.
    this = "[`modernbert`](https://huggingface.co/toiletsandpaper/russian-ai-text-detector-modernbert), эта модель"
    assert f"| {this} |" in choose
    assert choose.index("[`modernbert`]") < choose.index("[`transformer`]") < choose.index("[`lightgbm`]")
    assert f"- `transformer` — {models.TRANSFORMER.summary}." in choose
    assert f"- `modernbert` (эта модель) — {models.MODERNBERT.summary}." in choose
    assert "в порядке таблицы" in choose and "Против" not in choose
    assert "deepvk2025rumodernbert" in body
    assert not re.search(r"\S\\\\\(", body)
    report = tt.report_body(m, tt.bundle_for(m["params"]["base"]).repo)
    assert (
        "evaluate ~/.cache/aiw-ru/llmtrace/transformer/full/rumodernbert-small --weights fp32 --int8-valid-only"
        in report
    )
    assert "Варианты сравнивались только на valid" in report and "у $1.1\\%$ текстов valid" in report
    int8_row = next(line for line in report.splitlines() if line.startswith("| ONNX int8"))
    assert "—" not in int8_row
    assert "Веса fp32 выбраны заранее" in report and "| `transformer` |" in report.replace("LightGBM |", "")


def test_card_and_report_for_frozen_heavy_bundle():
    """FRIDA: замороженный энкодер с головой, веса во внешнем файле, тяжёлая и не самая точная модель."""
    m = fake_metrics()
    m["params"] = {
        k: v
        for k, v in m["params"].items()
        if k not in ("epochs", "lr", "batch", "warmup", "patience", "best_step", "best_epoch", "early_stopped")
    }
    m["params"] |= {
        "base": "ai-forever/FRIDA",
        "parameters": 823_000_000,
        "prefix": "categorize: ",
        "pooling": "cls",
        "head": "logistic regression",
        "C": 10.0,
        "c_grid": [0.1, 1.0, 10.0, 100.0],
        "precision": "fp16",
        "embed_seconds": 5400.0,
        "shipped": "onnx_fp32",
        "shipped_file": "onnx/model.onnx",
        "size_mb": 3300.0,
        "external_data": True,
        "probs_source": "torch",
        "onnx_texts": 300,
    }
    m["dataset"]["train"] = 80000
    m["variants"]["onnx_int8"]["test"] = m["variants"]["onnx_int8"]["valid"]
    m["curve"] = [{"C": c, "roc_auc": 0.95, "accuracy": 0.9, "logloss": 0.3} for c in (0.1, 1.0, 10.0, 100.0)]
    m["latency"] = [{"runtime": "onnxruntime fp32", "tokens": 512, "threads": 1, "ms": 9000.0}]
    m["sizes_mb"] |= {"onnx_fp32": 3300.0, "onnx_int8": 850.0}
    m["test"]["roc_auc"] = 0.985
    fast = {"repo": tt.REPO, "name": "transformer", "size_mb": 29.7, "ms_512_1": 39.8, "x86_ms_512_1": 101.1}
    m["related"] = [
        {**fast, "test": {**m["test"], "roc_auc": 0.98}},
        {
            **fast,
            "repo": tt.BUNDLES["deepvk/RuModernBERT-small"].repo,
            "name": "modernbert",
            "size_mb": 139.6,
            "ms_512_1": 272.0,
            "test": {**m["test"], "roc_auc": 0.993},
        },
    ]
    card = tt.card(m)
    assert "t5" in card.data.to_dict()["tags"] and "bert" not in card.data.to_dict()["tags"]
    body = card.text
    assert "`model.onnx.data`" in body and '"model.onnx*"' in body
    assert "aiw-ru models install frida" in body and "--model frida" in body
    assert "Модель тяжёлая, для мощных машин" in body and "у вызовов дольше секунды" in body
    assert "самый тяжёлый трансформер aiw-ru, но не самый точный: на test его обходит `modernbert`." in body
    assert "без дообучения" in body and f"- `frida` (эта модель) — {tt.BUNDLES['ai-forever/FRIDA'].summary}." in body
    assert "habr.com/ru/companies/sberdevices/articles/909924" in body
    assert not re.search(r"\S\\\\\(", body)
    report = tt.report_body(m, tt.bundle_for("ai-forever/FRIDA").repo)
    assert "замороженный энкодер `ai-forever/FRIDA`" in report and "сам FRIDA для мощных машин" in report
    assert "embed --base ai-forever/FRIDA --prefix 'categorize: ' --pooling cls --limit 80000" in report
    assert "--exclude 'emb-*'" in report and "--onnx-texts 300" in report


def test_bundle_hub_layout(tmp_path):
    for f in (
        "onnx/model.onnx",
        "onnx/model_int8.onnx",
        "model/tokenizer.json",
        "model/tokenizer_config.json",
        "model/config.json",
        "inference.json",
        "metrics.json",
    ):
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f).write_text(f, encoding="utf-8")
    hub = tt.bundle_hub(tmp_path, "onnx/model.onnx")
    assert (hub / "model.onnx").read_text(encoding="utf-8") == "onnx/model.onnx"
    assert (hub / "model_int8.onnx").exists() and not (hub / "model_fp32.onnx").exists()
    hub = tt.bundle_hub(tmp_path, "onnx/model_int8.onnx")
    assert (hub / "model.onnx").read_text(encoding="utf-8") == "onnx/model_int8.onnx"
    assert (hub / "model_fp32.onnx").exists() and not (hub / "model_int8.onnx").exists()


def test_base_revision_offline_uses_cached_snapshot(tmp_path, monkeypatch):
    def offline(*_args, **_kwargs):
        raise OSError("offline")

    monkeypatch.setattr(tt.HfApi, "model_info", offline)
    monkeypatch.setattr(tt, "HF_HUB_CACHE", str(tmp_path))
    assert tt.base_revision("org/base") == ""
    (tmp_path / "models--org--base" / "snapshots" / "abc123").mkdir(parents=True)
    assert tt.base_revision("org/base") == "abc123"
    (tmp_path / "models--org--base" / "snapshots" / "def456").mkdir()
    assert tt.base_revision("org/base") == ""


def test_batches_respect_token_budget():
    lengths = np.array([10, 8000, 20, 4000, 30, 40, 8192, 50])
    for rng in (None, np.random.default_rng(1)):
        out = tt.batches(lengths, 4, rng, tokens=8192)
        assert sorted(int(k) for b in out for k in b) == list(range(len(lengths)))
        for b in out:
            assert len(b) <= 4
            assert len(b) == 1 or len(b) * int(lengths[b].max()) <= 8192
    # без лимита по токенам — как раньше, по size текстов
    assert [len(b) for b in tt.batches(lengths, 4, None)] == [4, 4]

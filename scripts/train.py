#!/usr/bin/env python3
"""Обучает LightGBM отличать тексты людей от текстов моделей, сравнивает его с детектором и выкладывает на Hub.

    uv run --group train scripts/train.py [--limit N] [--rounds N] [--leaves N] [--rate X] [--jobs N]
                                          [--data ПАПКА] [--out ПАПКА] [--repo ИМЯ] [--no-push]

Учится на train-части русского LLMTrace, число деревьев подбирает по потере на
valid, проверяется на test. Все три части сначала скачивает
`uv run --group train scripts/llmtrace.py fetch --set classification --split <часть>`.

Признаки считает aiw_ru.features, тот же код, что и при проверке текста
моделью. В папку модели (по умолчанию lightgbm/ рядом с данными) пишутся
model.txt, features.json, metrics.json и карточка README.md. После полного
обучения папка уходит в репозиторий модели на Hugging Face (нужен
`hf auth login`); с --limit или --no-push модель остаётся локальной.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

import aiw_ru
from aiw_ru.detect import analyze
from aiw_ru.features import CONTEXT, FEATURE_NAMES, FEATURES_VERSION, describe, features
from hub import REPO, card, push, write_report
from llmtrace import REVISIONS, data_dir, detector_key, load, num, pct, run_parallel, table

ROOT = Path(__file__).resolve().parent.parent
LENGTHS = ((0, 50), (50, 150), (150, 400), (400, None))
TOP_GENERATORS = 15


@dataclass
class Part:
    x: np.ndarray
    y: np.ndarray
    score: np.ndarray
    words: np.ndarray
    genre: np.ndarray
    prompt: np.ndarray
    generator: np.ndarray


def _features_chunk(texts: list[str]) -> list[tuple[list[float], int, int]]:
    rows = []
    for text in texts:
        result = analyze(text, CONTEXT)
        rows.append((features(text, result), result.score, result.stats.words))
    return rows


def part(args: argparse.Namespace, split: str) -> Part:
    """Признаки части корпуса; кэш сбрасывается при правке детектора или этого файла."""
    key = hashlib.sha1((detector_key() + Path(__file__).read_text(encoding="utf-8")).encode()).hexdigest()[:12]
    stem = f"features-{split}{f'-{args.limit}' if args.limit else ''}"
    cache = args.data / f"{stem}-{key}.npz"
    if cache.exists():
        archive = np.load(cache)
        return Part(*(archive[field] for field in Part.__dataclass_fields__))
    rows = [json.loads(line) for line in load(args.data, "classification", split, args.limit)]
    print(f"Признаки {split}: {len(rows)} текстов в {args.jobs} процессах…", file=sys.stderr)
    computed = list(run_parallel(_features_chunk, [row["text"] for row in rows], args.jobs))
    computed_part = Part(
        x=np.array([vector for vector, _, _ in computed], dtype=np.float32),
        y=np.array([1 if row["label"] == "ai" else 0 for row in rows], dtype=np.int8),
        score=np.array([score for _, score, _ in computed], dtype=np.float32),
        words=np.array([word_count for _, _, word_count in computed], dtype=np.int32),
        genre=np.array([row["data_type"] for row in rows]),
        prompt=np.array([row["prompt_type"] or "" for row in rows]),
        generator=np.array([row["model"] if row["label"] == "ai" else "" for row in rows]),
    )
    for old in args.data.glob(f"{stem}-*.npz"):
        old.unlink()
    np.savez_compressed(cache, **{field: getattr(computed_part, field) for field in Part.__dataclass_fields__})
    return computed_part


def auc(y: np.ndarray, s: np.ndarray) -> float | None:
    return float(roc_auc_score(y, s)) if len(set(y.tolist())) == 2 else None


def fmt_auc(x: float | None) -> str:
    return num(x) if x is not None else "—"


def classes(y: np.ndarray, predicted: np.ndarray) -> dict:
    """Числа classification_report из sklearn: по классам, accuracy и macro avg."""
    precision, recall, f1, support = precision_recall_fscore_support(y, predicted, labels=[0, 1], zero_division=0)
    return {
        "accuracy": float((y == predicted).mean()),
        "classes": {
            name: {
                "precision": float(precision[k]),
                "recall": float(recall[k]),
                "f1": float(f1[k]),
                "support": int(support[k]),
            }
            for k, name in enumerate(["human", "ai"])
        },
        "macro": {"precision": float(precision.mean()), "recall": float(recall.mean()), "f1": float(f1.mean())},
    }


def report(title: str, c: dict) -> None:
    """Как classification_report из sklearn, но с десятичной запятой."""
    rows = [
        [
            label,
            num(c["classes"][key]["precision"]),
            num(c["classes"][key]["recall"]),
            num(c["classes"][key]["f1"]),
            str(c["classes"][key]["support"]),
        ]
        for key, label in (("human", "люди"), ("ai", "ИИ"))
    ]
    total = str(sum(class_stats["support"] for class_stats in c["classes"].values()))
    rows.append(["accuracy", "", "", num(c["accuracy"]), total])
    rows.append(["macro avg", num(c["macro"]["precision"]), num(c["macro"]["recall"]), num(c["macro"]["f1"]), total])
    table(title, ["", "precision", "recall", "f1", "текстов"], rows)


def git_commit() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
        )
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=False)
    except OSError:
        return "неизвестен"
    return sha.stdout.strip() + ("+правки" if dirty.stdout.strip() else "")


def environment() -> dict[str, str]:
    """Версии, от которых зависит результат: для раздела «Как воспроизвести» в отчёте."""
    import sklearn

    return {
        "python": platform.python_version(),
        "lightgbm": lgb.__version__,
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "platform": f"{platform.system()} {platform.release()}, {platform.machine()}, {os.cpu_count()} ядер",
    }


def breakdowns(test: Part, p: np.ndarray) -> dict:
    """Разрезы test: жанры, длина текста, тип задания и модель-генератор."""
    hit = test.y == (p >= 0.5)
    genres = []
    for genre in sorted(set(test.genre.tolist())):
        mask = test.genre == genre
        genres.append(
            {
                "genre": genre,
                "human": int((test.y[mask] == 0).sum()),
                "ai": int(test.y[mask].sum()),
                "accuracy": float(hit[mask].mean()),
                "roc_auc": auc(test.y[mask], p[mask]),
                "detector_roc_auc": auc(test.y[mask], test.score[mask]),
            }
        )
    lengths = []
    for lo, hi in LENGTHS:
        mask = (test.words >= lo) & (test.words < hi if hi else True)
        if mask.any():
            bucket = f"{lo}–{hi - 1}" if hi else f"{lo} и больше"
            lengths.append(
                {
                    "bucket": bucket,
                    "n": int(mask.sum()),
                    "accuracy": float(hit[mask].mean()),
                    "roc_auc": auc(test.y[mask], p[mask]),
                }
            )
    ai = test.y == 1
    prompts = [
        {
            "prompt_type": prompt_type,
            "n": int((ai & (test.prompt == prompt_type)).sum()),
            "recall": float(hit[ai & (test.prompt == prompt_type)].mean()),
        }
        for prompt_type in sorted(set(test.prompt[ai].tolist()))
    ]
    counts = Counter(test.generator[ai].tolist())
    generators = [
        {"model": generator, "n": count, "recall": float(hit[ai & (test.generator == generator)].mean())}
        for generator, count in counts.most_common(TOP_GENERATORS)
    ]
    return {"genres": genres, "lengths": lengths, "prompt_types": prompts, "generators": generators}


def main(argv: list[str] | None = None) -> None:
    argument_parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    argument_parser.add_argument("--data", type=Path, default=data_dir())
    argument_parser.add_argument("--limit", type=int, default=0, help="взять N текстов из каждой части")
    argument_parser.add_argument("--jobs", type=int, default=0, help="процессов детектора (по умолчанию по числу ядер)")
    argument_parser.add_argument("--rounds", type=int, default=3000, help="предел числа деревьев")
    argument_parser.add_argument("--leaves", type=int, default=31)
    argument_parser.add_argument("--rate", type=float, default=0.05)
    argument_parser.add_argument("--out", type=Path, help="папка модели (по умолчанию lightgbm/ рядом с данными)")
    argument_parser.add_argument("--repo", default=REPO, help="репозиторий модели на Hugging Face")
    argument_parser.add_argument("--no-push", action="store_true", help="не выкладывать на Hugging Face")
    args = argument_parser.parse_args(argv)
    args.jobs = args.jobs or os.cpu_count() or 4

    test, valid, train = (part(args, split) for split in ("test", "valid", "train"))
    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "learning_rate": args.rate,
        "num_leaves": args.leaves,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "seed": 1,
        # Одинаковый результат при повторном запуске на тех же данных и той же машине.
        "deterministic": True,
        "force_row_wise": True,
        "verbose": -1,
    }
    started = time.perf_counter()
    booster = lgb.train(
        params,
        lgb.Dataset(train.x, train.y, feature_name=[name.replace(":", "_") for name in FEATURE_NAMES]),
        num_boost_round=args.rounds,
        valid_sets=[lgb.Dataset(valid.x, valid.y)],
        callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False), lgb.log_evaluation(250)],
    )
    elapsed = time.perf_counter() - started
    out_dir: Path = args.out or args.data / "lightgbm"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_file = out_dir / "model.txt"
    booster.save_model(model_file, num_iteration=booster.best_iteration)
    print(
        f"LightGBM: {booster.best_iteration} деревьев по {args.leaves} листьев, обучение {elapsed:.0f} с, "
        f"{model_file.stat().st_size / 1024:.0f} КБ на диске ({model_file}). "
        f"Train {len(train.y)}, valid {len(valid.y)}, test {len(test.y)} текстов, {len(FEATURE_NAMES)} признаков."
    )

    test_probs = np.asarray(booster.predict(test.x, num_iteration=booster.best_iteration), dtype=np.float64)
    valid_probs = np.asarray(booster.predict(valid.x, num_iteration=booster.best_iteration), dtype=np.float64)
    created = (test.y == 0) | (test.prompt == "create")
    table(
        "ROC AUC: оценка детектора против вероятности модели",
        ["Выборка", "Людей", "ИИ", "Детектор", "Модель"],
        [
            [
                name,
                str(int((labels == 0).sum())),
                str(int(labels.sum())),
                fmt_auc(auc(labels, scores)),
                fmt_auc(auc(labels, probs)),
            ]
            for name, labels, scores, probs in [
                ("test: люди против сгенерированных с нуля", test.y[created], test.score[created], test_probs[created]),
                ("test: люди против всего ИИ", test.y, test.score, test_probs),
                ("valid: люди против всего ИИ", valid.y, valid.score, valid_probs),
            ]
        ],
    )
    model_test = classes(test.y, (test_probs >= 0.5).astype(np.int8))
    detector_test = classes(test.y, (test.score >= 40).astype(np.int8))
    report("Модель на test, порог вероятности 0,5", model_test)
    report("Детектор на test, порог оценки 40 («много примет»)", detector_test)
    cuts = breakdowns(test, test_probs)
    table(
        "Test по жанрам",
        ["Жанр", "Людей", "ИИ", "AUC детектора", "AUC модели", "Accuracy модели"],
        [
            [
                genre["genre"],
                str(genre["human"]),
                str(genre["ai"]),
                fmt_auc(genre["detector_roc_auc"]),
                fmt_auc(genre["roc_auc"]),
                num(genre["accuracy"]),
            ]
            for genre in cuts["genres"]
        ],
    )
    table(
        "Test по длине текста",
        ["Слов", "Текстов", "Accuracy", "AUC модели"],
        [
            [bucket["bucket"], str(bucket["n"]), num(bucket["accuracy"]), fmt_auc(bucket["roc_auc"])]
            for bucket in cuts["lengths"]
        ],
    )
    gain = booster.feature_importance("gain", iteration=booster.best_iteration)
    total = gain.sum() or 1.0
    human, ai = test.x[test.y == 0], test.x[test.y == 1]
    ranked: list[dict[str, Any]] = [
        {
            "name": FEATURE_NAMES[k],
            "description": describe(FEATURE_NAMES[k]),
            "gain_share": float(gain[k] / total),
            "mean_human": float(human[:, k].mean()),
            "mean_ai": float(ai[:, k].mean()),
        }
        for k in np.argsort(-gain)
    ]
    table(
        "Признаки с наибольшим вкладом",
        ["Признак", "Доля прироста", "Люди", "ИИ"],
        [
            [feature["name"], pct(feature["gain_share"]), num(feature["mean_human"], 2), num(feature["mean_ai"], 2)]
            for feature in ranked[:15]
        ],
    )

    metrics = {
        "created": datetime.now(UTC).strftime("%Y-%m-%d"),
        "aiw_ru_version": aiw_ru.__version__,
        "git_commit": git_commit(),
        "features_version": FEATURES_VERSION,
        "environment": environment(),
        "dataset": {
            "repo": "iitolstykh/LLMTrace_classification",
            "revision": REVISIONS["classification"],
            "language": "ru",
            "limit": args.limit,
            "train": len(train.y),
            "valid": len(valid.y),
            "test": len(test.y),
        },
        "params": {
            **{key: value for key, value in params.items() if key not in ("metric", "verbose")},
            "rounds": args.rounds,
            "best_iteration": booster.best_iteration,
            "features": len(FEATURE_NAMES),
            "train_seconds": elapsed,
            "size_kb": model_file.stat().st_size / 1024,
            "cpu": f"{platform.machine()}, {os.cpu_count()} ядер",
        },
        "test": {
            **model_test,
            "roc_auc": auc(test.y, test_probs),
            "roc_auc_created": auc(test.y[created], test_probs[created]),
        },
        "valid": {"accuracy": float((valid.y == (valid_probs >= 0.5)).mean()), "roc_auc": auc(valid.y, valid_probs)},
        "detector": {
            **detector_test,
            "threshold": 40,
            "roc_auc": auc(test.y, test.score),
            "roc_auc_created": auc(test.y[created], test.score[created]),
        },
        **cuts,
        "features": ranked,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    spec = {"version": FEATURES_VERSION, "context": CONTEXT, "threshold": 0.5, "names": list(FEATURE_NAMES)}
    (out_dir / "features.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    card(metrics, args.repo).save(out_dir / "README.md")
    print(f"\nМодель, признаки, метрики и карточка: {out_dir}")
    if not args.limit:
        path = write_report(metrics, args.repo)
        print(f"Отчёт об обучении в репозитории: {path.relative_to(ROOT)}")

    if args.limit or args.no_push:
        print("На Hugging Face не выкладывается: " + ("обучение на выборке" if args.limit else "--no-push"))
        return
    push(
        out_dir,
        args.repo,
        f"Обучение {metrics['created']}: accuracy {num(model_test['accuracy'])}, "
        f"ROC AUC {fmt_auc(metrics['test']['roc_auc'])}",
    )


if __name__ == "__main__":
    main()

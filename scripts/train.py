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
    out = []
    for text in texts:
        r = analyze(text, CONTEXT)
        out.append((features(text, r), r.score, r.stats.words))
    return out


def part(args: argparse.Namespace, split: str) -> Part:
    """Признаки части корпуса; кэш сбрасывается при правке детектора или этого файла."""
    key = hashlib.sha1((detector_key() + Path(__file__).read_text(encoding="utf-8")).encode()).hexdigest()[:12]
    stem = f"features-{split}{f'-{args.limit}' if args.limit else ''}"
    cache = args.data / f"{stem}-{key}.npz"
    if cache.exists():
        z = np.load(cache)
        return Part(*(z[f] for f in Part.__dataclass_fields__))
    rows = [json.loads(line) for line in load(args.data, "classification", split, args.limit)]
    print(f"Признаки {split}: {len(rows)} текстов в {args.jobs} процессах…", file=sys.stderr)
    got = list(run_parallel(_features_chunk, [r["text"] for r in rows], args.jobs))
    p = Part(
        x=np.array([f for f, _, _ in got], dtype=np.float32),
        y=np.array([1 if r["label"] == "ai" else 0 for r in rows], dtype=np.int8),
        score=np.array([s for _, s, _ in got], dtype=np.float32),
        words=np.array([w for _, _, w in got], dtype=np.int32),
        genre=np.array([r["data_type"] for r in rows]),
        prompt=np.array([r["prompt_type"] or "" for r in rows]),
        generator=np.array([r["model"] if r["label"] == "ai" else "" for r in rows]),
    )
    for old in args.data.glob(f"{stem}-*.npz"):
        old.unlink()
    np.savez_compressed(cache, **{f: getattr(p, f) for f in Part.__dataclass_fields__})
    return p


def auc(y: np.ndarray, s: np.ndarray) -> float | None:
    return float(roc_auc_score(y, s)) if len(set(y.tolist())) == 2 else None


def fmt_auc(x: float | None) -> str:
    return num(x) if x is not None else "—"


def classes(y: np.ndarray, predicted: np.ndarray) -> dict:
    """Числа classification_report из sklearn: по классам, accuracy и macro avg."""
    p, r, f, n = precision_recall_fscore_support(y, predicted, labels=[0, 1], zero_division=0)
    return {
        "accuracy": float((y == predicted).mean()),
        "classes": {
            name: {"precision": float(p[k]), "recall": float(r[k]), "f1": float(f[k]), "support": int(n[k])}
            for k, name in enumerate(["human", "ai"])
        },
        "macro": {"precision": float(p.mean()), "recall": float(r.mean()), "f1": float(f.mean())},
    }


def report(title: str, c: dict) -> None:
    """Как classification_report из sklearn, но с десятичной запятой."""
    rows = [
        [
            label,
            num(c["classes"][k]["precision"]),
            num(c["classes"][k]["recall"]),
            num(c["classes"][k]["f1"]),
            str(c["classes"][k]["support"]),
        ]
        for k, label in (("human", "люди"), ("ai", "ИИ"))
    ]
    n = str(sum(v["support"] for v in c["classes"].values()))
    rows.append(["accuracy", "", "", num(c["accuracy"]), n])
    rows.append(["macro avg", num(c["macro"]["precision"]), num(c["macro"]["recall"]), num(c["macro"]["f1"]), n])
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
    for g in sorted(set(test.genre.tolist())):
        m = test.genre == g
        genres.append(
            {
                "genre": g,
                "human": int((test.y[m] == 0).sum()),
                "ai": int(test.y[m].sum()),
                "accuracy": float(hit[m].mean()),
                "roc_auc": auc(test.y[m], p[m]),
                "detector_roc_auc": auc(test.y[m], test.score[m]),
            }
        )
    lengths = []
    for lo, hi in LENGTHS:
        m = (test.words >= lo) & (test.words < hi if hi else True)
        if m.any():
            bucket = f"{lo}–{hi - 1}" if hi else f"{lo} и больше"
            lengths.append(
                {"bucket": bucket, "n": int(m.sum()), "accuracy": float(hit[m].mean()), "roc_auc": auc(test.y[m], p[m])}
            )
    ai = test.y == 1
    prompts = [
        {
            "prompt_type": t,
            "n": int((ai & (test.prompt == t)).sum()),
            "recall": float(hit[ai & (test.prompt == t)].mean()),
        }
        for t in sorted(set(test.prompt[ai].tolist()))
    ]
    counts = Counter(test.generator[ai].tolist())
    generators = [
        {"model": g, "n": n, "recall": float(hit[ai & (test.generator == g)].mean())}
        for g, n in counts.most_common(TOP_GENERATORS)
    ]
    return {"genres": genres, "lengths": lengths, "prompt_types": prompts, "generators": generators}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--data", type=Path, default=data_dir())
    ap.add_argument("--limit", type=int, default=0, help="взять N текстов из каждой части")
    ap.add_argument("--jobs", type=int, default=0, help="процессов детектора (по умолчанию по числу ядер)")
    ap.add_argument("--rounds", type=int, default=3000, help="предел числа деревьев")
    ap.add_argument("--leaves", type=int, default=31)
    ap.add_argument("--rate", type=float, default=0.05)
    ap.add_argument("--out", type=Path, help="папка модели (по умолчанию lightgbm/ рядом с данными)")
    ap.add_argument("--repo", default=REPO, help="репозиторий модели на Hugging Face")
    ap.add_argument("--no-push", action="store_true", help="не выкладывать на Hugging Face")
    args = ap.parse_args(argv)
    args.jobs = args.jobs or os.cpu_count() or 4

    test, valid, train = (part(args, s) for s in ("test", "valid", "train"))
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
        lgb.Dataset(train.x, train.y, feature_name=[n.replace(":", "_") for n in FEATURE_NAMES]),
        num_boost_round=args.rounds,
        valid_sets=[lgb.Dataset(valid.x, valid.y)],
        callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False), lgb.log_evaluation(250)],
    )
    elapsed = time.perf_counter() - started
    out: Path = args.out or args.data / "lightgbm"
    out.mkdir(parents=True, exist_ok=True)
    model_file = out / "model.txt"
    booster.save_model(model_file, num_iteration=booster.best_iteration)
    print(
        f"LightGBM: {booster.best_iteration} деревьев по {args.leaves} листьев, обучение {elapsed:.0f} с, "
        f"{model_file.stat().st_size / 1024:.0f} КБ на диске ({model_file}). "
        f"Train {len(train.y)}, valid {len(valid.y)}, test {len(test.y)} текстов, {len(FEATURE_NAMES)} признаков."
    )

    p = np.asarray(booster.predict(test.x, num_iteration=booster.best_iteration), dtype=np.float64)
    pv = np.asarray(booster.predict(valid.x, num_iteration=booster.best_iteration), dtype=np.float64)
    created = (test.y == 0) | (test.prompt == "create")
    table(
        "ROC AUC: оценка детектора против вероятности модели",
        ["Выборка", "Людей", "ИИ", "Детектор", "Модель"],
        [
            [name, str(int((y == 0).sum())), str(int(y.sum())), fmt_auc(auc(y, s)), fmt_auc(auc(y, q))]
            for name, y, s, q in [
                ("test: люди против сгенерированных с нуля", test.y[created], test.score[created], p[created]),
                ("test: люди против всего ИИ", test.y, test.score, p),
                ("valid: люди против всего ИИ", valid.y, valid.score, pv),
            ]
        ],
    )
    model_test = classes(test.y, (p >= 0.5).astype(np.int8))
    detector_test = classes(test.y, (test.score >= 40).astype(np.int8))
    report("Модель на test, порог вероятности 0,5", model_test)
    report("Детектор на test, порог оценки 40 («много примет»)", detector_test)
    cuts = breakdowns(test, p)
    table(
        "Test по жанрам",
        ["Жанр", "Людей", "ИИ", "AUC детектора", "AUC модели", "Accuracy модели"],
        [
            [
                g["genre"],
                str(g["human"]),
                str(g["ai"]),
                fmt_auc(g["detector_roc_auc"]),
                fmt_auc(g["roc_auc"]),
                num(g["accuracy"]),
            ]
            for g in cuts["genres"]
        ],
    )
    table(
        "Test по длине текста",
        ["Слов", "Текстов", "Accuracy", "AUC модели"],
        [[b["bucket"], str(b["n"]), num(b["accuracy"]), fmt_auc(b["roc_auc"])] for b in cuts["lengths"]],
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
        [[f["name"], pct(f["gain_share"]), num(f["mean_human"], 2), num(f["mean_ai"], 2)] for f in ranked[:15]],
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
            **{k: v for k, v in params.items() if k not in ("metric", "verbose")},
            "rounds": args.rounds,
            "best_iteration": booster.best_iteration,
            "features": len(FEATURE_NAMES),
            "train_seconds": elapsed,
            "size_kb": model_file.stat().st_size / 1024,
            "cpu": f"{platform.machine()}, {os.cpu_count()} ядер",
        },
        "test": {**model_test, "roc_auc": auc(test.y, p), "roc_auc_created": auc(test.y[created], p[created])},
        "valid": {"accuracy": float((valid.y == (pv >= 0.5)).mean()), "roc_auc": auc(valid.y, pv)},
        "detector": {
            **detector_test,
            "threshold": 40,
            "roc_auc": auc(test.y, test.score),
            "roc_auc_created": auc(test.y[created], test.score[created]),
        },
        **cuts,
        "features": ranked,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    spec = {"version": FEATURES_VERSION, "context": CONTEXT, "threshold": 0.5, "names": list(FEATURE_NAMES)}
    (out / "features.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    card(metrics, args.repo).save(out / "README.md")
    print(f"\nМодель, признаки, метрики и карточка: {out}")
    if not args.limit:
        path = write_report(metrics, args.repo)
        print(f"Отчёт об обучении в репозитории: {path.relative_to(ROOT)}")

    if args.limit or args.no_push:
        print("На Hugging Face не выкладывается: " + ("обучение на выборке" if args.limit else "--no-push"))
        return
    push(
        out,
        args.repo,
        f"Обучение {metrics['created']}: accuracy {num(model_test['accuracy'])}, "
        f"ROC AUC {fmt_auc(metrics['test']['roc_auc'])}",
    )


if __name__ == "__main__":
    main()

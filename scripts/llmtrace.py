#!/usr/bin/env python3
"""Замер детектора на русской части корпуса LLMTrace.

    uv run --group train scripts/llmtrace.py fetch [--split test] [--set classification|detection]
    uv run --group train scripts/llmtrace.py eval  [--split test] [--limit N] [--jobs N] [--context РЕЖИМ]
                                                   [--rule ТИП] [--examples N] [--json ФАЙЛ]
    uv run --group train scripts/llmtrace.py spans [--split test] [--limit N] [--jobs N] [--json ФАЙЛ]

Корпус: Tolstykh et al., 2025, https://huggingface.co/datasets/iitolstykh/LLMTrace_classification
(Apache 2.0). fetch скачивает JSONL с Hugging Face и оставляет русские тексты.
eval считает оценку детектора классификатором (ROC AUC, accuracy, F1 по порогам)
и показывает, как часто каждое правило срабатывает у людей и у моделей. spans
сверяет фрагменты antiplagiat с разметкой ИИ-кусков по знакам.

Детектор (пакет aiw_ru) работает в нескольких процессах. Его ответы кэшируются
рядом с данными; ключ кэша — хэш исходников src/aiw_ru, так что правка правила
сбрасывает кэш сама.

Данные лежат вне репозитория, в ~/.cache/aiw-ru/llmtrace (или $AIW_RU_DATA):
тестовые части весят сотни мегабайт, а человеческие тексты корпуса собраны из
сторонних источников со своими лицензиями.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from collections import Counter, deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from functools import partial
from pathlib import Path

from sklearn.metrics import roc_auc_score

from aiw_ru import antiplagiat
from aiw_ru.detect import TYPE_LABELS, analyze
from aiw_ru.types import CONTEXT_MODES, ContextMode

ROOT = Path(__file__).resolve().parent.parent
REPOS = {"classification": "LLMTrace_classification", "detection": "LLMTrace_detection"}
# Ревизии наборов на Hugging Face: замеры и обучение воспроизводятся на тех же данных.
REVISIONS = {
    "classification": "252e21e3dca713bc82bc3c1ef73bf533d7c9fc7e",
    "detection": "332053804f8798177ff2ecb1148e1839ed429ba3",
}
THRESHOLDS = (15, 40, 70)
CHUNK = 1000

Span = tuple[int, int]

# ─── Метрики ────────────────────────────────────────────────────────────


def roc_auc(ai: Iterable[float], human: Iterable[float]) -> float:
    """Вероятность, что случайный ИИ-текст получит оценку выше случайного человеческого."""
    ai_scores, human_scores = list(ai), list(human)
    if not ai_scores or not human_scores:
        return 0.0
    return float(roc_auc_score([1] * len(ai_scores) + [0] * len(human_scores), ai_scores + human_scores))


def at_threshold(ai: list[float], human: list[float], t: float) -> dict[str, float]:
    """Метрики, если считать ИИ всё, что набрало не меньше `t`."""
    true_positives = sum(score >= t for score in ai)
    false_positives = sum(score >= t for score in human)
    true_negatives = len(human) - false_positives
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / len(ai) if ai else 0.0
    specificity = true_negatives / len(human) if human else 0.0
    return {
        "threshold": t,
        "accuracy": (true_positives + true_negatives) / max(len(ai) + len(human), 1),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "balanced_accuracy": (recall + specificity) / 2,
        "fpr": false_positives / len(human) if human else 0.0,
    }


def best_f1(ai: list[float], human: list[float]) -> dict[str, float]:
    """Порог 1–100 с лучшим F1. Порог 0 не рассматривается: он называет ИИ всё подряд."""
    return max((at_threshold(ai, human, threshold) for threshold in range(1, 101)), key=lambda metrics: metrics["f1"])


def merge(spans: Iterable[Span]) -> list[list[int]]:
    merged: list[list[int]] = []
    for start, end in sorted((start, end) for start, end in spans if end > start):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def union_length(spans: Iterable[Span]) -> int:
    """Суммарная длина объединения отрезков."""
    return sum(end - start for start, end in merge(spans))


def overlap_length(xs: Iterable[Span], ys: Iterable[Span]) -> int:
    """Длина пересечения двух наборов отрезков."""
    left, right = merge(xs), merge(ys)
    i = j = total = 0
    while i < len(left) and j < len(right):
        total += max(0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return total


# ─── Данные ─────────────────────────────────────────────────────────────


def data_dir() -> Path:
    if os.environ.get("AIW_RU_DATA"):
        return Path(os.environ["AIW_RU_DATA"])
    cache = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(cache) / "aiw-ru" / "llmtrace"


def data_file(directory: Path, dataset: str, split: str) -> Path:
    return directory / f"ru-{dataset}-{split}.jsonl"


def download(url: str) -> Iterator[bytes]:
    """Скачивает файл по кускам; оборванное соединение продолжается с того же байта."""
    offset = stalled = 0
    while True:
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
        before = offset
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if offset and response.status != 206:
                    raise RuntimeError(f"{url}: сервер не умеет продолжать загрузку")
                if not offset:
                    total = int(response.headers.get("content-length") or 0)
                    print(f"Скачиваю {url}" + (f" ({total / 1e6:.0f} МБ)" if total else ""), file=sys.stderr)
                while chunk := response.read(1 << 20):
                    offset += len(chunk)
                    yield chunk
            return
        except (OSError, TimeoutError) as error:
            # Сдаёмся после пяти обрывов подряд без единого нового байта.
            stalled = 0 if offset > before else stalled + 1
            if stalled == 5:
                raise
            print(f"  связь оборвалась на {offset / 1e6:.0f} МБ ({error}), продолжаю", file=sys.stderr)
            time.sleep(stalled + 1)


def fetch(directory: Path, dataset: str, split: str) -> None:
    url = f"https://huggingface.co/datasets/iitolstykh/{REPOS[dataset]}/resolve/{REVISIONS[dataset]}/{split}.jsonl"
    directory.mkdir(parents=True, exist_ok=True)
    path = data_file(directory, dataset, split)
    partial_path = path.with_suffix(".part")
    seen = kept = 0
    buffer = b""
    with partial_path.open("wb") as partial_file:
        for chunk in download(url):
            buffer += chunk
            *lines, buffer = buffer.split(b"\n")
            for line in lines:
                if not line.strip():
                    continue
                seen += 1
                if json.loads(line).get("lang") == "ru":
                    partial_file.write(line + b"\n")
                    kept += 1
        if buffer.strip():
            seen += 1
            if json.loads(buffer).get("lang") == "ru":
                partial_file.write(buffer + b"\n")
                kept += 1
    partial_path.replace(path)
    print(f"{path}: {kept} русских текстов из {seen}")


def load(directory: Path, dataset: str, split: str, limit: int = 0) -> list[str]:
    """Строки JSONL части корпуса; с `limit` — равномерная выборка, сохраняющая доли жанров и моделей."""
    path = data_file(directory, dataset, split)
    if not path.exists():
        raise SystemExit(
            f"llmtrace: нет {path}; сначала: uv run --group train scripts/llmtrace.py fetch --set {dataset} --split {split}"
        )
    with path.open(encoding="utf-8") as file:
        lines = [line for line in file if line.strip()]
    if not limit or limit >= len(lines):
        return lines
    step = len(lines) / limit
    return [lines[int(k * step)] for k in range(limit)]


# ─── Детектор ───────────────────────────────────────────────────────────


# Модули пакета, которые не влияют на ответы детектора и признаки.
NOT_DETECTOR = {"__init__.py", "__main__.py", "cli.py", "models.py"}


def detector_key() -> str:
    """Хэш исходников детектора: меняется при правке правил, признаков и оценки."""
    digest = hashlib.sha1()
    for path in sorted((ROOT / "src" / "aiw_ru").rglob("*.py")):
        if path.name not in NOT_DETECTOR:
            digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def keep_scan(text: str, context: ContextMode) -> dict:
    result = analyze(text, context)
    stats = result.stats
    return {
        "score": result.score,
        "types": dict(Counter(issue.type for issue in result.issues)),
        "stats": {
            "words": stats.words,
            "sentences": stats.sentences,
            "meanSentenceLength": stats.mean_sentence_length,
            "sentenceLengthCV": stats.sentence_length_cv,
            "mattr": stats.mattr,
        },
    }


def keep_antiplagiat(text: str, context: ContextMode) -> dict:
    report = antiplagiat(text, context)
    return {
        "share": report.ai_share,
        "spans": [[fragment.start, fragment.end] for fragment in report.fragments if fragment.ai],
    }


def _run_chunk(command: str, context: ContextMode, texts: list[str]) -> list[dict]:
    keep = keep_scan if command == "scan" else keep_antiplagiat
    return [keep(text, context) for text in texts]


def run_parallel[T](work: Callable[[list[str]], list[T]], texts: list[str], jobs: int) -> Iterator[T]:
    """Прогоняет тексты через `work` пачками в `jobs` процессах, порядок сохраняется.

    `work` должна быть функцией модуля (или partial от неё): её передают в другой процесс.
    """
    # Хотя бы по четыре пачки на процесс, но не больше CHUNK текстов в пачке.
    size = max(20, min(CHUNK, -(-len(texts) // (4 * jobs))))
    chunks = [texts[i : i + size] for i in range(0, len(texts), size)]
    done = 0
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        pending: deque[Future[list[T]]] = deque()
        for chunk in chunks:
            pending.append(pool.submit(work, chunk))
            if len(pending) >= 2 * jobs:
                yield from pending.popleft().result()
                done += 1
                print(f"\r  {min(done * size, len(texts))}/{len(texts)}", end="", file=sys.stderr, flush=True)
        while pending:
            yield from pending.popleft().result()
    print(file=sys.stderr)


def detect(
    directory: Path, dataset: str, split: str, limit: int, command: str, context: ContextMode, jobs: int
) -> tuple[list[str], list[dict]]:
    """Строки корпуса и ответы детектора на них; ответы берутся из кэша, если исходники не менялись."""
    lines = load(directory, dataset, split, limit)
    stem = f"detector-{command}-{context}-{dataset}-{split}{f'-{limit}' if limit else ''}"
    cache = directory / f"{stem}-{detector_key()}.jsonl"
    if cache.exists():
        with cache.open(encoding="utf-8") as file:
            return lines, [json.loads(line) for line in file]
    print(f"Детектор: {len(lines)} текстов в {jobs} процессах…", file=sys.stderr)
    texts = [json.loads(line)["text"] for line in lines]
    results = list(run_parallel(partial(_run_chunk, command, context), texts, jobs))
    for old in directory.glob(f"{stem}-*.jsonl"):
        old.unlink()
    cache.write_text("".join(json.dumps(result, ensure_ascii=False) + "\n" for result in results), encoding="utf-8")
    return lines, results


# ─── Отчёты ─────────────────────────────────────────────────────────────


def num(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def pct(x: float) -> str:
    return f"{100 * x:.1f} %".replace(".", ",")


def table(title: str, head: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(row[k]) for row in [head, *rows]) for k in range(len(head))]

    def fmt(row: list[str]) -> str:
        return "  ".join(
            cell.ljust(width) if k == 0 else cell.rjust(width)
            for k, (cell, width) in enumerate(zip(row, widths, strict=True))
        )

    print(f"\n{title}\n{fmt(head)}")
    for row in rows:
        print(fmt(row))


def classifier_rows(ai: list[float], human: list[float]) -> list[list[str]]:
    best = best_f1(ai, human)["threshold"]
    rows = []
    for threshold in sorted({*THRESHOLDS, best}):
        metrics = at_threshold(ai, human, threshold)
        rows.append(
            [
                f"{threshold}{' (лучший F1)' if threshold == best else ''}",
                *(num(metrics[key]) for key in ("accuracy", "precision", "recall", "f1", "balanced_accuracy")),
                pct(metrics["fpr"]),
            ]
        )
    return rows


def cmd_eval(args: argparse.Namespace) -> None:
    lines, results = detect(args.data, "classification", args.split, args.limit, "scan", args.context, args.jobs)
    rows = [json.loads(line) for line in lines]
    for row in rows:
        del row["text"]
    human = [index for index, row in enumerate(rows) if row["label"] == "human"]
    created = [index for index, row in enumerate(rows) if row["label"] == "ai" and row["prompt_type"] == "create"]
    ai_all = [index for index, row in enumerate(rows) if row["label"] == "ai"]

    def scores(indices: list[int]) -> list[float]:
        return [results[index]["score"] for index in indices]

    print(
        f"LLMTrace, русская часть {args.split}: {len(human)} человеческих текстов, {len(created)} сгенерированных "
        f"с нуля, {len(ai_all) - len(created)} правок модели. Режим {args.context}."
    )
    head = ["Порог", "Accuracy", "Precision", "Recall", "F1", "Bal. acc", "Люди как ИИ"]
    report: dict = {"split": args.split, "context": args.context, "n": len(rows)}
    for title, ai in [
        ("Люди против сгенерированных с нуля", created),
        ("Люди против всего ИИ (с правками модели)", ai_all),
    ]:
        auc = roc_auc(scores(ai), scores(human))
        table(
            f"{title}: ROC AUC {num(auc)}. ИИ — оценка не ниже порога.",
            head,
            classifier_rows(scores(ai), scores(human)),
        )
        report[title] = {
            "auc": auc,
            "best": best_f1(scores(ai), scores(human)),
            "thresholds": [at_threshold(scores(ai), scores(human), threshold) for threshold in THRESHOLDS],
        }
    base = len(created) / max(len(created) + len(human), 1)
    print(
        f"\n  Для сравнения: «всё люди» даёт accuracy {num(1 - base)}, «всё ИИ» — F1 {num(2 * base / (1 + base))} "
        "(люди против сгенерированных)."
    )

    def share(indices: list[int], rule: str) -> float:
        return sum(rule in results[index]["types"] for index in indices) / len(indices) if indices else 0.0

    labels = TYPE_LABELS
    types = [rule for rule in labels if share(human, rule) or share(created, rule)]
    types.sort(key=lambda rule: share(created, rule) / max(share(human, rule), 1e-4))
    table(
        "Правила: доля документов со срабатыванием (сначала те, что чаще срабатывают у людей)",
        ["Правило", "Люди", "ИИ", "ИИ/люди", ""],
        [
            [
                f"{labels[rule]} ({rule})",
                pct(share(human, rule)),
                pct(share(created, rule)),
                num(share(created, rule) / share(human, rule), 1) if share(human, rule) else "—",
                "чаще у людей" if share(created, rule) < share(human, rule) else "",
            ]
            for rule in types
        ],
    )
    report["rules"] = {rule: {"human": share(human, rule), "ai": share(created, rule)} for rule in types}

    genre_rows = []
    for genre in sorted({row["data_type"] for row in rows}):
        genre_human = [index for index in human if rows[index]["data_type"] == genre]
        genre_ai = [index for index in created if rows[index]["data_type"] == genre]
        high = lambda indices: (
            pct(sum(results[index]["score"] >= 40 for index in indices) / len(indices)) if indices else "—"
        )
        auc = num(roc_auc(scores(genre_ai), scores(genre_human))) if genre_human and genre_ai else "—"
        genre_rows.append([genre, str(len(genre_human)), high(genre_human), str(len(genre_ai)), high(genre_ai), auc])
    table(
        "Жанры: доля текстов с оценкой 40 и выше",
        ["Жанр", "Людей", "Люди ≥ 40", "ИИ", "ИИ ≥ 40", "ROC AUC"],
        genre_rows,
    )

    if args.rule:
        hits = [index for index in human if args.rule in results[index]["types"]]
        print(f"\nСрабатывания «{args.rule}» на человеческих текстах: {len(hits)}, примеры:")
        for index in hits[: args.examples]:
            row = json.loads(lines[index])
            issue = next(issue for issue in analyze(row["text"], args.context).issues if issue.type == args.rule)
            around = " ".join(row["text"][max(0, issue.index - 80) : issue.index + 120].split())
            print(f"  [{row['data_type']}] «{issue.text}»\n    …{around}…")
    if args.json:
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def cmd_spans(args: argparse.Namespace) -> None:
    lines, results = detect(args.data, "detection", args.split, args.limit, "antiplagiat", args.context, args.jobs)
    overlap_chars = predicted = gold = 0
    abs_error = 0.0
    by_label: dict[str, list[float]] = {}
    shares: dict[bool, list[float]] = {True: [], False: []}
    for line, result in zip(lines, results, strict=True):
        row = json.loads(line)
        gold_spans = [(start, end) for start, end in row.get("ai_char_intervals") or []]
        spans = [tuple(span) for span in result["spans"]]
        overlap_chars += overlap_length(spans, gold_spans)
        predicted += union_length(spans)
        gold += union_length(gold_spans)
        length = len(row["text"])
        gold_share = union_length(gold_spans) / length if length else 0.0
        abs_error += abs(result["share"] / 100 - gold_share)
        totals = by_label.setdefault(row["label"], [0, 0.0, 0.0])
        totals[0] += 1
        totals[1] += result["share"] / 100
        totals[2] += gold_share
        shares[row["label"] != "human"].append(result["share"])
    precision = overlap_chars / predicted if predicted else 0.0
    recall = overlap_chars / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mae = abs_error / max(len(lines), 1)
    print(
        f"LLMTrace detection, русская часть {args.split}: {len(lines)} текстов. "
        f"Модель antiplagiat по умолчанию, режим {args.context}."
    )
    print(
        f"\n  По знакам: precision {num(precision)}, recall {num(recall)}, F1 {num(f1)}."
        f"\n  Доля ИИ-текста: средняя ошибка {pct(mae)}; ROC AUC доли (люди против ИИ и смешанных) "
        f"{num(roc_auc(shares[True], shares[False]))}."
    )
    table(
        "Средняя доля ИИ-текста по меткам корпуса",
        ["Метка", "Текстов", "По antiplagiat", "По разметке"],
        [
            [label, str(count), pct(predicted_sum / count), pct(gold_sum / count)]
            for label, (count, predicted_sum, gold_sum) in by_label.items()
        ],
    )
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {"split": args.split, "n": len(lines), "precision": precision, "recall": recall, "f1": f1, "mae": mae},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    argument_parser.add_argument("command", choices=["fetch", "eval", "spans"])
    argument_parser.add_argument("--data", type=Path, default=data_dir())
    argument_parser.add_argument("--split", default="test")
    argument_parser.add_argument("--set", dest="dataset", choices=list(REPOS))
    argument_parser.add_argument("--limit", type=int, default=0)
    argument_parser.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    argument_parser.add_argument(
        "--context", default="general", choices=CONTEXT_MODES, help="режим детектора, как у aiw-ru --context"
    )
    argument_parser.add_argument("--rule")
    argument_parser.add_argument("--examples", type=int, default=5)
    argument_parser.add_argument("--json")
    return argument_parser


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.command == "fetch":
        for dataset in [args.dataset] if args.dataset else list(REPOS):
            fetch(args.data, dataset, args.split)
    elif args.command == "eval":
        cmd_eval(args)
    else:
        cmd_spans(args)


if __name__ == "__main__":
    main()

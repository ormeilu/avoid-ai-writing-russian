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
    a, h = list(ai), list(human)
    if not a or not h:
        return 0.0
    return float(roc_auc_score([1] * len(a) + [0] * len(h), a + h))


def at_threshold(ai: list[float], human: list[float], t: float) -> dict[str, float]:
    """Метрики, если считать ИИ всё, что набрало не меньше `t`."""
    tp = sum(s >= t for s in ai)
    fp = sum(s >= t for s in human)
    tn = len(human) - fp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / len(ai) if ai else 0.0
    specificity = tn / len(human) if human else 0.0
    return {
        "threshold": t,
        "accuracy": (tp + tn) / max(len(ai) + len(human), 1),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "balanced_accuracy": (recall + specificity) / 2,
        "fpr": fp / len(human) if human else 0.0,
    }


def best_f1(ai: list[float], human: list[float]) -> dict[str, float]:
    """Порог 1–100 с лучшим F1. Порог 0 не рассматривается: он называет ИИ всё подряд."""
    return max((at_threshold(ai, human, t) for t in range(1, 101)), key=lambda c: c["f1"])


def merge(spans: Iterable[Span]) -> list[list[int]]:
    out: list[list[int]] = []
    for a, b in sorted((a, b) for a, b in spans if b > a):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def union_length(spans: Iterable[Span]) -> int:
    """Суммарная длина объединения отрезков."""
    return sum(b - a for a, b in merge(spans))


def overlap_length(xs: Iterable[Span], ys: Iterable[Span]) -> int:
    """Длина пересечения двух наборов отрезков."""
    a, b = merge(xs), merge(ys)
    i = j = n = 0
    while i < len(a) and j < len(b):
        n += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return n


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
            with urllib.request.urlopen(request, timeout=60) as res:
                if offset and res.status != 206:
                    raise RuntimeError(f"{url}: сервер не умеет продолжать загрузку")
                if not offset:
                    total = int(res.headers.get("content-length") or 0)
                    print(f"Скачиваю {url}" + (f" ({total / 1e6:.0f} МБ)" if total else ""), file=sys.stderr)
                while chunk := res.read(1 << 20):
                    offset += len(chunk)
                    yield chunk
            return
        except (OSError, TimeoutError) as e:
            # Сдаёмся после пяти обрывов подряд без единого нового байта.
            stalled = 0 if offset > before else stalled + 1
            if stalled == 5:
                raise
            print(f"  связь оборвалась на {offset / 1e6:.0f} МБ ({e}), продолжаю", file=sys.stderr)
            time.sleep(stalled + 1)


def fetch(directory: Path, dataset: str, split: str) -> None:
    url = f"https://huggingface.co/datasets/iitolstykh/{REPOS[dataset]}/resolve/{REVISIONS[dataset]}/{split}.jsonl"
    directory.mkdir(parents=True, exist_ok=True)
    path = data_file(directory, dataset, split)
    tmp = path.with_suffix(".part")
    seen = kept = 0
    buf = b""
    with tmp.open("wb") as out:
        for chunk in download(url):
            buf += chunk
            *lines, buf = buf.split(b"\n")
            for line in lines:
                if not line.strip():
                    continue
                seen += 1
                if json.loads(line).get("lang") == "ru":
                    out.write(line + b"\n")
                    kept += 1
        if buf.strip():
            seen += 1
            if json.loads(buf).get("lang") == "ru":
                out.write(buf + b"\n")
                kept += 1
    tmp.replace(path)
    print(f"{path}: {kept} русских текстов из {seen}")


def load(directory: Path, dataset: str, split: str, limit: int = 0) -> list[str]:
    """Строки JSONL части корпуса; с `limit` — равномерная выборка, сохраняющая доли жанров и моделей."""
    path = data_file(directory, dataset, split)
    if not path.exists():
        raise SystemExit(
            f"llmtrace: нет {path}; сначала: uv run --group train scripts/llmtrace.py fetch --set {dataset} --split {split}"
        )
    with path.open(encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    if not limit or limit >= len(lines):
        return lines
    step = len(lines) / limit
    return [lines[int(k * step)] for k in range(limit)]


# ─── Детектор ───────────────────────────────────────────────────────────


# Модули пакета, которые не влияют на ответы детектора и признаки.
NOT_DETECTOR = {"__init__.py", "__main__.py", "cli.py", "models.py"}


def detector_key() -> str:
    """Хэш исходников детектора: меняется при правке правил, признаков и оценки."""
    h = hashlib.sha1()
    for p in sorted((ROOT / "src" / "aiw_ru").rglob("*.py")):
        if p.name not in NOT_DETECTOR:
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


def keep_scan(text: str, context: ContextMode) -> dict:
    r = analyze(text, context)
    s = r.stats
    return {
        "score": r.score,
        "types": dict(Counter(i.type for i in r.issues)),
        "stats": {
            "words": s.words,
            "sentences": s.sentences,
            "meanSentenceLength": s.mean_sentence_length,
            "sentenceLengthCV": s.sentence_length_cv,
            "mattr": s.mattr,
        },
    }


def keep_antiplagiat(text: str, context: ContextMode) -> dict:
    r = antiplagiat(text, context)
    return {"share": r.ai_share, "spans": [[f.start, f.end] for f in r.fragments if f.ai]}


def _run_chunk(command: str, context: ContextMode, texts: list[str]) -> list[dict]:
    keep = keep_scan if command == "scan" else keep_antiplagiat
    return [keep(t, context) for t in texts]


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
        with cache.open(encoding="utf-8") as f:
            return lines, [json.loads(line) for line in f]
    print(f"Детектор: {len(lines)} текстов в {jobs} процессах…", file=sys.stderr)
    texts = [json.loads(line)["text"] for line in lines]
    results = list(run_parallel(partial(_run_chunk, command, context), texts, jobs))
    for old in directory.glob(f"{stem}-*.jsonl"):
        old.unlink()
    cache.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), encoding="utf-8")
    return lines, results


# ─── Отчёты ─────────────────────────────────────────────────────────────


def num(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def pct(x: float) -> str:
    return f"{100 * x:.1f} %".replace(".", ",")


def table(title: str, head: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(r[k]) for r in [head, *rows]) for k in range(len(head))]

    def fmt(r: list[str]) -> str:
        return "  ".join(c.ljust(w) if k == 0 else c.rjust(w) for k, (c, w) in enumerate(zip(r, widths, strict=True)))

    print(f"\n{title}\n{fmt(head)}")
    for r in rows:
        print(fmt(r))


def classifier_rows(ai: list[float], human: list[float]) -> list[list[str]]:
    best = best_f1(ai, human)["threshold"]
    rows = []
    for t in sorted({*THRESHOLDS, best}):
        c = at_threshold(ai, human, t)
        rows.append(
            [
                f"{t}{' (лучший F1)' if t == best else ''}",
                *(num(c[k]) for k in ("accuracy", "precision", "recall", "f1", "balanced_accuracy")),
                pct(c["fpr"]),
            ]
        )
    return rows


def cmd_eval(args: argparse.Namespace) -> None:
    lines, res = detect(args.data, "classification", args.split, args.limit, "scan", args.context, args.jobs)
    rows = [json.loads(line) for line in lines]
    for r in rows:
        del r["text"]
    human = [k for k, r in enumerate(rows) if r["label"] == "human"]
    created = [k for k, r in enumerate(rows) if r["label"] == "ai" and r["prompt_type"] == "create"]
    ai_all = [k for k, r in enumerate(rows) if r["label"] == "ai"]

    def scores(ks: list[int]) -> list[float]:
        return [res[k]["score"] for k in ks]

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
            "thresholds": [at_threshold(scores(ai), scores(human), t) for t in THRESHOLDS],
        }
    base = len(created) / max(len(created) + len(human), 1)
    print(
        f"\n  Для сравнения: «всё люди» даёт accuracy {num(1 - base)}, «всё ИИ» — F1 {num(2 * base / (1 + base))} "
        "(люди против сгенерированных)."
    )

    def share(ks: list[int], t: str) -> float:
        return sum(t in res[k]["types"] for k in ks) / len(ks) if ks else 0.0

    labels = TYPE_LABELS
    types = [t for t in labels if share(human, t) or share(created, t)]
    types.sort(key=lambda t: share(created, t) / max(share(human, t), 1e-4))
    table(
        "Правила: доля документов со срабатыванием (сначала те, что чаще срабатывают у людей)",
        ["Правило", "Люди", "ИИ", "ИИ/люди", ""],
        [
            [
                f"{labels[t]} ({t})",
                pct(share(human, t)),
                pct(share(created, t)),
                num(share(created, t) / share(human, t), 1) if share(human, t) else "—",
                "чаще у людей" if share(created, t) < share(human, t) else "",
            ]
            for t in types
        ],
    )
    report["rules"] = {t: {"human": share(human, t), "ai": share(created, t)} for t in types}

    genre_rows = []
    for g in sorted({r["data_type"] for r in rows}):
        h = [k for k in human if rows[k]["data_type"] == g]
        a = [k for k in created if rows[k]["data_type"] == g]
        high = lambda ks: pct(sum(res[k]["score"] >= 40 for k in ks) / len(ks)) if ks else "—"
        auc = num(roc_auc(scores(a), scores(h))) if h and a else "—"
        genre_rows.append([g, str(len(h)), high(h), str(len(a)), high(a), auc])
    table(
        "Жанры: доля текстов с оценкой 40 и выше",
        ["Жанр", "Людей", "Люди ≥ 40", "ИИ", "ИИ ≥ 40", "ROC AUC"],
        genre_rows,
    )

    if args.rule:
        hits = [k for k in human if args.rule in res[k]["types"]]
        print(f"\nСрабатывания «{args.rule}» на человеческих текстах: {len(hits)}, примеры:")
        for k in hits[: args.examples]:
            row = json.loads(lines[k])
            issue = next(i for i in analyze(row["text"], args.context).issues if i.type == args.rule)
            around = " ".join(row["text"][max(0, issue.index - 80) : issue.index + 120].split())
            print(f"  [{row['data_type']}] «{issue.text}»\n    …{around}…")
    if args.json:
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def cmd_spans(args: argparse.Namespace) -> None:
    lines, res = detect(args.data, "detection", args.split, args.limit, "antiplagiat", args.context, args.jobs)
    tp = predicted = gold = 0
    abs_error = 0.0
    by_label: dict[str, list[float]] = {}
    shares: dict[bool, list[float]] = {True: [], False: []}
    for line, r in zip(lines, res, strict=True):
        row = json.loads(line)
        g = [(a, b) for a, b in row.get("ai_char_intervals") or []]
        spans = [tuple(s) for s in r["spans"]]
        tp += overlap_length(spans, g)
        predicted += union_length(spans)
        gold += union_length(g)
        n = len(row["text"])
        gold_share = union_length(g) / n if n else 0.0
        abs_error += abs(r["share"] / 100 - gold_share)
        b = by_label.setdefault(row["label"], [0, 0.0, 0.0])
        b[0] += 1
        b[1] += r["share"] / 100
        b[2] += gold_share
        shares[row["label"] != "human"].append(r["share"])
    precision = tp / predicted if predicted else 0.0
    recall = tp / gold if gold else 0.0
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
        [[label, str(n), pct(s / n), pct(g / n)] for label, (n, s, g) in by_label.items()],
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
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["fetch", "eval", "spans"])
    ap.add_argument("--data", type=Path, default=data_dir())
    ap.add_argument("--split", default="test")
    ap.add_argument("--set", dest="dataset", choices=list(REPOS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument(
        "--context", default="general", choices=CONTEXT_MODES, help="режим детектора, как у aiw-ru --context"
    )
    ap.add_argument("--rule")
    ap.add_argument("--examples", type=int, default=5)
    ap.add_argument("--json")
    return ap


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

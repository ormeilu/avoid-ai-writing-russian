#!/usr/bin/env python3
"""Модели aiw-ru на длинных документах: целиком человеческих, целиком ИИ и с ИИ-вставками.

    uv run --group train --extra ml scripts/long_docs.py build  [--split test] [--docs 600] [--seed 1]
    uv run --group train --extra ml scripts/long_docs.py eval   [--split test] --model ИМЯ [--overlap 0.25]
    uv run --group train --extra ml scripts/long_docs.py report [--split test] [--models ИМЯ…] [--overlaps …]

В LLMTrace смешанных документов длиннее одного окна модели почти нет: у смешанных
текстов части detection медиана 639 знаков. Поэтому документы собираются из текстов
части classification. Тексты одного жанра склеиваются через пустую строку, ИИ-куски
размечаются по знакам. Виды документов: только человек, только ИИ, ИИ-вставка в
начале, в середине, в конце и несколько вставок по одному тексту. Каждый текст идёт
в один документ. ИИ-тексты, которые начинаются с начала человеческого текста (так
генератор продолжал чужой текст), в документы не берутся: разметка по знакам для
них неверна.

build пишет документы в ~/.cache/aiw-ru/llmtrace/long/, eval прогоняет модель по
фрагментам, как `aiw-ru classify` (models.scan_chunks), и кэширует вероятности
(прерванный прогон продолжается), report считает метрики по фрагментам и по
документам. С --out report заменяет таблицы в отчёте между метками
<!-- long-docs:… -->; текст вокруг них пишется руками. Параметры нарезки
сравниваются на valid, итоговые числа — на test.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

from aiw_ru import models
from llmtrace import ROOT, data_dir, load, overlap_length, roc_auc, union_length

# Жанры прозы: в стихах и коротких формах (short_form) текстов длиннее MIN_TEXT мало.
PROSE = ("article", "news", "story", "review", "factual", "question")
MIN_TEXT = 400
LENGTHS = (5000, 10000, 20000, 40000)
KINDS = ("human", "ai", "start", "middle", "end", "scattered")
MIXED = ("start", "middle", "end", "scattered")
KIND_LABELS = {
    "human": "только человек",
    "ai": "только ИИ",
    "start": "ИИ в начале",
    "middle": "ИИ в середине",
    "end": "ИИ в конце",
    "scattered": "2–3 ИИ-вставки",
}
# Столько знаков общего начала с человеческим текстом делают ИИ-текст продолжением чужого.
PREFIX = 60
# Доля ИИ-знаков, с которой фрагмент считается ИИ-фрагментом, а с которой — человеческим.
PURE = 0.95
Pair = tuple[str, bool]


@dataclass(frozen=True, slots=True)
class Doc:
    id: int
    kind: str
    data_type: str
    text: str
    # ИИ-куски: знаки [начало, конец).
    ai: list[list[int]]


# ─── Документы ──────────────────────────────────────────────────────────


def long_dir() -> Path:
    return data_dir() / "long"


def docs_path(split: str, n: int, seed: int) -> Path:
    return long_dir() / f"docs-{split}-n{n}-s{seed}.jsonl"


def eval_path(split: str, n: int, seed: int, model: str, overlap: float) -> Path:
    return long_dir() / f"eval-{split}-n{n}-s{seed}-{model}-o{overlap:g}.jsonl"


def pools(rows: Iterable[dict[str, Any]]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Человеческие и ИИ-тексты прозы по жанрам; ИИ-продолжения человеческих текстов отброшены."""
    rows = list(rows)
    heads = {r["text"].strip()[:PREFIX] for r in rows if r["label"] == "human"}
    human: dict[str, list[str]] = defaultdict(list)
    ai: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        text = r["text"].strip()
        if r["data_type"] not in PROSE or len(text) < MIN_TEXT:
            continue
        if r["label"] == "human":
            human[r["data_type"]].append(text)
        elif text[:PREFIX] not in heads:
            ai[r["data_type"]].append(text)
    return human, ai


def compose(parts: list[Pair]) -> tuple[str, list[list[int]]]:
    """Склеивает тексты через пустую строку; ИИ-тексты становятся интервалами знаков."""
    text, ai = "", []
    for part, is_ai in parts:
        text += "\n\n" if text else ""
        start = len(text)
        text += part
        if is_ai:
            ai.append([start, len(text)])
    return text, ai


def fill(draw: Callable[[], str | None], chars: float, is_ai: bool) -> list[Pair]:
    """Тексты из пула, пока их не наберётся chars знаков; хотя бы один, если пул не пуст."""
    out: list[Pair] = []
    size = 0
    while size < chars or not out:
        part = draw()
        if part is None:
            break
        out.append((part, is_ai))
        size += len(part) + 2
    return out


def plan(
    kind: str, target: int, human: Callable[[], str | None], ai: Callable[[], str | None], rng: random.Random
) -> list[Pair] | None:
    """Тексты документа по порядку; None, если пулы кончились или вставку некуда поставить."""
    if kind == "human":
        return fill(human, target, False) or None
    if kind == "ai":
        return fill(ai, target, True) or None
    if kind == "scattered":
        blocks = [fill(ai, 1, True) for _ in range(rng.choice((2, 3)))]
        rest = fill(human, target - sum(len(t) for b in blocks for t, _ in b), False)
        if not all(blocks) or len(rest) < len(blocks) + 1:
            return None
        slots = sorted(rng.sample(range(1, len(rest)), len(blocks)))
        out: list[Pair] = []
        prev = 0
        for slot, block in zip(slots, blocks, strict=True):
            out += rest[prev:slot] + block
            prev = slot
        return out + rest[prev:]
    inserted = fill(ai, target * rng.uniform(0.15, 0.4), True)
    rest = fill(human, target - sum(len(t) for t, _ in inserted), False)
    if not inserted or len(rest) < 2:
        return None
    if kind == "start":
        return inserted + rest
    if kind == "end":
        return rest + inserted
    cut = len(rest) // 2
    return rest[:cut] + inserted + rest[cut:]


def build(rows: Iterable[dict[str, Any]], n: int, seed: int) -> list[Doc]:
    """n документов: виды по кругу, длины по кругу через каждые шесть, жанр случайный по размеру пулов."""
    rng = random.Random(seed)
    human, ai = pools(rows)
    for pool in (*human.values(), *ai.values()):
        rng.shuffle(pool)
    types = [t for t in PROSE if human[t] and ai[t]]

    def draw(pool: list[str]) -> Callable[[], str | None]:
        return lambda: pool.pop() if pool else None

    docs: list[Doc] = []
    k = 0
    while len(docs) < n and k < n * 10:
        kind, target = KINDS[k % len(KINDS)], LENGTHS[(k // len(KINDS)) % len(LENGTHS)]
        k += 1
        alive = [t for t in types if human[t] and ai[t]]
        if not alive:
            break
        data_type = rng.choices(alive, weights=[min(len(human[t]), len(ai[t])) for t in alive])[0]
        parts = plan(kind, target, draw(human[data_type]), draw(ai[data_type]), rng)
        if parts:
            text, spans = compose(parts)
            docs.append(Doc(len(docs), kind, data_type, text, spans))
    return docs


def read_docs(path: Path) -> list[Doc]:
    if not path.exists():
        raise SystemExit(f"long_docs: нет {path}; сначала build с теми же --split, --docs и --seed")
    with path.open(encoding="utf-8") as f:
        return [Doc(**json.loads(line)) for line in f if line.strip()]


# ─── Прогон модели ──────────────────────────────────────────────────────


def run_doc(loaded: models.Loaded, doc: Doc, overlap: float) -> dict[str, Any]:
    """Вероятность по началу, как в scan, и по фрагментам, как в classify."""
    scan = models.scan_chunks(loaded, doc.text, overlap)
    return {
        "id": doc.id,
        "threshold": loaded.threshold,
        "head": round(loaded.probability(doc.text), 5),
        "seconds": round(scan.seconds, 4),
        "chunks": [[c.start, c.end, round(c.probability, 5)] for c in scan.chunks],
    }


def evaluate(docs: list[Doc], loaded: models.Loaded, overlap: float, path: Path) -> None:
    """Прогон с кэшем: уже посчитанные документы в файле пропускаются."""
    done = set()
    if path.exists():
        with path.open(encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    todo = [d for d in docs if d.id not in done]
    started = time.perf_counter()
    with path.open("a", encoding="utf-8") as f:
        for k, doc in enumerate(todo, 1):
            f.write(json.dumps(run_doc(loaded, doc, overlap)) + "\n")
            f.flush()
            if k % 20 == 0 or k == len(todo):
                spent = time.perf_counter() - started
                left = spent / k * (len(todo) - k)
                print(f"  {k}/{len(todo)} документов, {spent:.0f} с, осталось около {left:.0f} с", flush=True)


def read_eval(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"long_docs: нет {path}; сначала eval с теми же параметрами")
    with path.open(encoding="utf-8") as f:
        return {r["id"]: r for r in map(json.loads, f) if r}


# ─── Метрики ────────────────────────────────────────────────────────────


def ai_share(start: int, end: int, ai: list[list[int]]) -> float:
    """Доля ИИ-знаков во фрагменте [start, end)."""
    return overlap_length([(start, end)], [(a, b) for a, b in ai]) / (end - start) if end > start else 0.0


def rate(xs: Iterable[bool]) -> float | None:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def fragment_metrics(docs: list[Doc], runs: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Фрагменты целиком человеческие, целиком ИИ и на стыке: как их видит модель."""
    human, ai, boundary = [], [], []
    for doc in docs:
        run = runs[doc.id]
        for a, b, p in run["chunks"]:
            share = ai_share(a, b, doc.ai)
            flagged = p >= run["threshold"]
            if share <= 1 - PURE:
                human.append((p, flagged))
            elif share >= PURE:
                ai.append((p, flagged))
            else:
                boundary.append((share, flagged))
    bins = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
    return {
        "human": len(human),
        "ai": len(ai),
        "boundary": len(boundary),
        "rocAuc": roc_auc([p for p, _ in ai], [p for p, _ in human]),
        "falsePositive": rate(f for _, f in human),
        "truePositive": rate(f for _, f in ai),
        "boundaryBins": [
            {"from": lo, "to": hi, "n": len(xs), "flagged": rate(xs)}
            for lo, hi in bins
            for xs in [[f for s, f in boundary if lo <= s < hi or (hi == 1.0 and s == 1.0)]]
        ],
    }


def doc_metrics(doc: Doc, run: dict[str, Any]) -> dict[str, Any]:
    """Один документ: что отметили фрагменты и начало, сколько ИИ нашлось и сколько лишнего."""
    t = run["threshold"]
    flagged = [(a, b) for a, b, p in run["chunks"] if p >= t]
    ai = [(a, b) for a, b in doc.ai]
    ai_len, flagged_len = union_length(ai), union_length(flagged)
    hit = overlap_length(flagged, ai)
    return {
        "kind": doc.kind,
        "chars": len(doc.text),
        "fragments": len(run["chunks"]),
        "seconds": run["seconds"],
        "any": bool(flagged),
        "two": len(flagged) >= 2,
        "head": run["head"] >= t,
        # ИИ-кусок найден, если хотя бы один отмеченный фрагмент в основном из ИИ-текста.
        "detected": any(ai_share(a, b, doc.ai) >= 0.5 for a, b in flagged),
        "recall": hit / ai_len if ai_len else None,
        "precision": hit / flagged_len if flagged_len else None,
        "trueShare": ai_len / len(doc.text),
        "flaggedShare": flagged_len / len(doc.text),
    }


def kind_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def avg(key: str) -> float | None:
        xs = [r[key] for r in rows if r[key] is not None]
        return mean(xs) if xs else None

    return {
        "docs": len(rows),
        "any": rate(r["any"] for r in rows),
        "two": rate(r["two"] for r in rows),
        "head": rate(r["head"] for r in rows),
        "detected": rate(r["detected"] for r in rows),
        "recall": avg("recall"),
        "precision": avg("precision"),
        "shareError": mean(abs(r["flaggedShare"] - r["trueShare"]) for r in rows) if rows else None,
    }


def metrics(docs: list[Doc], runs: dict[int, dict[str, Any]]) -> dict[str, Any]:
    per_doc = [doc_metrics(d, runs[d.id]) for d in docs]
    by_kind = {k: kind_metrics([r for r in per_doc if r["kind"] == k]) for k in KINDS}
    fragments = sum(r["fragments"] for r in per_doc)
    return {
        "docs": len(docs),
        "fragments": fragments,
        "fragmentsPerDoc": median(r["fragments"] for r in per_doc),
        "msPerFragment": 1000 * sum(r["seconds"] for r in per_doc) / fragments if fragments else None,
        "msPerDoc": 1000 * mean(r["seconds"] for r in per_doc) if per_doc else None,
        "chunks": fragment_metrics(docs, runs),
        "kinds": by_kind,
        "mixed": kind_metrics([r for r in per_doc if r["kind"] in MIXED]),
    }


# ─── Отчёт ──────────────────────────────────────────────────────────────


def pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.1f} %".replace(".", ",")


def num(x: float | None, digits: int = 3) -> str:
    return "—" if x is None else f"{x:.{digits}f}".replace(".", ",")


def md_table(head: list[str], rows: list[list[str]]) -> str:
    align = ["---", *["--:"] * (len(head) - 1)]
    return "\n".join("| " + " | ".join(r) + " |" for r in [head, align, *rows])


def dataset_md(docs: list[Doc]) -> str:
    rows = []
    for kind in KINDS:
        ds = [d for d in docs if d.kind == kind]
        share = [sum(b - a for a, b in d.ai) / len(d.text) for d in ds]
        rows.append(
            [
                KIND_LABELS[kind],
                str(len(ds)),
                f"{median(len(d.text) for d in ds):,.0f}".replace(",", " ") if ds else "—",
                pct(mean(share)) if ds else "—",
            ]
        )
    return md_table(["Вид", "Документов", "Знаков, медиана", "Доля ИИ-знаков"], rows)


def models_md(results: dict[str, dict[str, Any]]) -> str:
    """Главная таблица: модели по столбцам, метрики по строкам."""
    names = list(results)
    r = results

    def row(label: str, get: Callable[[dict[str, Any]], str]) -> list[str]:
        return [label, *(get(r[n]) for n in names)]

    rows = [
        row("Фрагментов на документ, медиана", lambda m: f"{m['fragmentsPerDoc']:g}"),
        row("мс на фрагмент", lambda m: num(m["msPerFragment"], 0)),
        row("ROC AUC по фрагментам", lambda m: num(m["chunks"]["rocAuc"])),
        row("Человеческие фрагменты выше порога", lambda m: pct(m["chunks"]["falsePositive"])),
        row("ИИ-фрагменты выше порога", lambda m: pct(m["chunks"]["truePositive"])),
        row("Человеческий документ: хоть один фрагмент выше порога", lambda m: pct(m["kinds"]["human"]["any"])),
        row("Человеческий документ: два фрагмента и больше", lambda m: pct(m["kinds"]["human"]["two"])),
        row("Человеческий документ: выше порога по началу", lambda m: pct(m["kinds"]["human"]["head"])),
        row("Смешанный документ: ИИ-кусок найден", lambda m: pct(m["mixed"]["detected"])),
        row("Смешанный документ: два фрагмента и больше", lambda m: pct(m["mixed"]["two"])),
        row("Смешанный документ: найдено по началу", lambda m: pct(m["mixed"]["head"])),
        row("Доля ИИ-знаков под отмеченными фрагментами", lambda m: pct(m["mixed"]["recall"])),
        row("Доля ИИ среди отмеченных знаков", lambda m: pct(m["mixed"]["precision"])),
        row("Ошибка доли ИИ в документе, пункты", lambda m: num(100 * m["mixed"]["shareError"], 1)),
        row("Документ целиком ИИ: найден", lambda m: pct(m["kinds"]["ai"]["detected"])),
    ]
    return md_table(["", *(f"`{n}`" for n in names)], rows)


def kinds_md(m: dict[str, Any]) -> str:
    rows = [
        [
            KIND_LABELS[k],
            str(m["kinds"][k]["docs"]),
            pct(m["kinds"][k]["any"]),
            pct(m["kinds"][k]["head"]),
            pct(m["kinds"][k]["detected"]) if k != "human" else "—",
            pct(m["kinds"][k]["recall"]) if k != "human" else "—",
            pct(m["kinds"][k]["precision"]),
        ]
        for k in KINDS
    ]
    head = ["Вид", "Документов", "Хоть один фрагмент выше порога", "Выше порога по началу", "ИИ найден"]
    return md_table([*head, "ИИ-знаков под отметками", "ИИ среди отмеченного"], rows)


def boundary_md(results: dict[str, dict[str, Any]]) -> str:
    names = list(results)
    bins = results[names[0]]["chunks"]["boundaryBins"]
    rows = []
    for k, b in enumerate(bins):
        label = f"{100 * b['from']:.0f}–{100 * b['to']:.0f} %"
        rows.append([label, str(b["n"]), *(pct(results[n]["chunks"]["boundaryBins"][k]["flagged"]) for n in names)])
    return md_table(["ИИ-знаков во фрагменте", "Фрагментов", *(f"`{n}`" for n in names)], rows)


def overlap_md(results: dict[float, dict[str, Any]]) -> str:
    rows = [
        [
            f"{100 * o:g} %",
            f"{m['fragmentsPerDoc']:g}",
            pct(m["kinds"]["human"]["any"]),
            pct(m["mixed"]["detected"]),
            pct(m["mixed"]["recall"]),
            pct(m["mixed"]["precision"]),
            num(m["msPerDoc"], 0),
        ]
        for o, m in results.items()
    ]
    head = ["Перекрытие", "Фрагментов на документ", "Ложная тревога в человеческом", "ИИ найден"]
    return md_table([*head, "ИИ-знаков под отметками", "ИИ среди отмеченного", "мс на документ"], rows)


def replace_block(text: str, name: str, body: str) -> str:
    begin, end = f"<!-- long-docs:{name} -->", f"<!-- /long-docs:{name} -->"
    if begin not in text or end not in text:
        raise SystemExit(f"long_docs: в отчёте нет меток {begin} … {end}")
    head, rest = text.split(begin, 1)
    return f"{head}{begin}\n{body}\n{end}{rest.split(end, 1)[1]}"


# ─── Командная строка ───────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("build", "eval", "report"):
        p = sub.add_parser(name)
        p.add_argument("--split", default="test", choices=["test", "valid"])
        p.add_argument("--docs", type=int, default=600, help="документов")
        p.add_argument("--seed", type=int, default=1)
        if name == "eval":
            p.add_argument("--model", required=True, choices=[m for m in models.MODELS if m != "lightgbm"])
            p.add_argument("--overlap", type=float, default=models.OVERLAP)
        if name == "report":
            p.add_argument("--models", nargs="+", default=["modernbert", "transformer", "mini-frida"])
            p.add_argument("--overlap", type=float, default=models.OVERLAP, help="перекрытие для таблиц моделей")
            p.add_argument("--overlaps", type=float, nargs="*", default=[], help="сравнить перекрытия")
            p.add_argument("--overlap-split", default="valid", choices=["test", "valid"])
            p.add_argument("--overlap-model", default="transformer", help="модель для сравнения перекрытий")
            p.add_argument("--out", type=Path, help="отчёт с метками <!-- long-docs:… -->")
    args = ap.parse_args(argv)
    path = docs_path(args.split, args.docs, args.seed)
    if args.command == "build":
        rows = [json.loads(line) for line in load(data_dir(), "classification", args.split)]
        docs = build(rows, args.docs, args.seed)
        long_dir().mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(asdict(d), ensure_ascii=False) + "\n" for d in docs), encoding="utf-8")
        print(f"{len(docs)} документов: {path}")
        print(dataset_md(docs))
        return
    docs = read_docs(path)
    if args.command == "eval":
        loaded = models.load(models.get(args.model))
        out = eval_path(args.split, args.docs, args.seed, args.model, args.overlap)
        print(f"{args.model}, перекрытие {args.overlap:g}: {out}", flush=True)
        evaluate(docs, loaded, args.overlap, out)
        return
    results = {
        m: metrics(docs, read_eval(eval_path(args.split, args.docs, args.seed, m, args.overlap))) for m in args.models
    }
    blocks = {
        "dataset": dataset_md(docs),
        "models": models_md(results),
        "kinds": "\n\n".join(f"`{m}`:\n\n{kinds_md(results[m])}" for m in args.models),
        "boundary": boundary_md(results),
    }
    if args.overlaps:
        odocs = read_docs(docs_path(args.overlap_split, args.docs, args.seed))
        om = args.overlap_model
        blocks["overlap"] = overlap_md(
            {
                o: metrics(odocs, read_eval(eval_path(args.overlap_split, args.docs, args.seed, om, o)))
                for o in args.overlaps
            }
        )
    summary = ROOT / "docs" / "long-documents.json"
    if args.out:
        text = args.out.read_text(encoding="utf-8")
        for name, body in blocks.items():
            text = replace_block(text, name, body)
        args.out.write_text(text, encoding="utf-8")
        summary.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Отчёт: {args.out}, числа: {summary.relative_to(ROOT)}")
        return
    for name, body in blocks.items():
        print(f"\n{name}\n\n{body}")


if __name__ == "__main__":
    sys.exit(main())

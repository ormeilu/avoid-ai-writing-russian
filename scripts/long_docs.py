#!/usr/bin/env python3
"""Модели aiw-ru на длинных документах: целиком человеческих, целиком ИИ и с ИИ-вставками.

    uv run --group train --extra ml scripts/long_docs.py build  [--split test] [--docs 600] [--seed 1]
    uv run --group train --extra ml scripts/long_docs.py eval   [--split test] --model ИМЯ [--overlap 0.25]
                                                                 [--label МЕТКА]
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

Новую версию модели сравнивают со старой так: папку новой задают переменной окружения
модели (AIW_RU_MODERNBERT_DIR и подобные), а прогону дают свою --label, по умолчанию
это имя модели. Метка идёт в имя кэша и в --models у report.
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
    heads = {row["text"].strip()[:PREFIX] for row in rows if row["label"] == "human"}
    human: dict[str, list[str]] = defaultdict(list)
    ai: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        text = row["text"].strip()
        if row["data_type"] not in PROSE or len(text) < MIN_TEXT:
            continue
        if row["label"] == "human":
            human[row["data_type"]].append(text)
        elif text[:PREFIX] not in heads:
            ai[row["data_type"]].append(text)
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
    parts: list[Pair] = []
    size = 0
    while size < chars or not parts:
        part = draw()
        if part is None:
            break
        parts.append((part, is_ai))
        size += len(part) + 2
    return parts


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
        rest = fill(human, target - sum(len(text) for block in blocks for text, _ in block), False)
        if not all(blocks) or len(rest) < len(blocks) + 1:
            return None
        slots = sorted(rng.sample(range(1, len(rest)), len(blocks)))
        parts: list[Pair] = []
        prev = 0
        for slot, block in zip(slots, blocks, strict=True):
            parts += rest[prev:slot] + block
            prev = slot
        return parts + rest[prev:]
    inserted = fill(ai, target * rng.uniform(0.15, 0.4), True)
    rest = fill(human, target - sum(len(text) for text, _ in inserted), False)
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
    types = [genre for genre in PROSE if human[genre] and ai[genre]]

    def draw(pool: list[str]) -> Callable[[], str | None]:
        return lambda: pool.pop() if pool else None

    docs: list[Doc] = []
    step = 0
    while len(docs) < n and step < n * 10:
        kind, target = KINDS[step % len(KINDS)], LENGTHS[(step // len(KINDS)) % len(LENGTHS)]
        step += 1
        alive = [genre for genre in types if human[genre] and ai[genre]]
        if not alive:
            break
        data_type = rng.choices(alive, weights=[min(len(human[genre]), len(ai[genre])) for genre in alive])[0]
        parts = plan(kind, target, draw(human[data_type]), draw(ai[data_type]), rng)
        if parts:
            text, spans = compose(parts)
            docs.append(Doc(len(docs), kind, data_type, text, spans))
    return docs


def read_docs(path: Path) -> list[Doc]:
    if not path.exists():
        raise SystemExit(f"long_docs: нет {path}; сначала build с теми же --split, --docs и --seed")
    with path.open(encoding="utf-8") as file:
        return [Doc(**json.loads(line)) for line in file if line.strip()]


# ─── Прогон модели ──────────────────────────────────────────────────────


def run_doc(loaded: models.Loaded, doc: Doc, overlap: float) -> dict[str, Any]:
    """Вероятность по началу, как в scan, и по фрагментам, как в classify."""
    scan = models.scan_chunks(loaded, doc.text, overlap)
    return {
        "id": doc.id,
        "threshold": loaded.threshold,
        "head": round(loaded.probability(doc.text), 5),
        "seconds": round(scan.seconds, 4),
        # Фрагменты не на русском модель не оценивает, и в замер они не входят, как в classify.
        "chunks": [
            [chunk.start, chunk.end, round(chunk.probability, 5)]
            for chunk in scan.chunks
            if chunk.probability is not None
        ],
    }


def evaluate(docs: list[Doc], loaded: models.Loaded, overlap: float, path: Path) -> None:
    """Прогон с кэшем: уже посчитанные документы в файле пропускаются."""
    done = set()
    if path.exists():
        with path.open(encoding="utf-8") as file:
            done = {json.loads(line)["id"] for line in file if line.strip()}
    todo = [doc for doc in docs if doc.id not in done]
    started = time.perf_counter()
    with path.open("a", encoding="utf-8") as file:
        for k, doc in enumerate(todo, 1):
            file.write(json.dumps(run_doc(loaded, doc, overlap)) + "\n")
            file.flush()
            if k % 20 == 0 or k == len(todo):
                spent = time.perf_counter() - started
                left = spent / k * (len(todo) - k)
                print(f"  {k}/{len(todo)} документов, {spent:.0f} с, осталось около {left:.0f} с", flush=True)


def read_eval(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"long_docs: нет {path}; сначала eval с теми же параметрами")
    with path.open(encoding="utf-8") as file:
        return {run["id"]: run for run in map(json.loads, file) if run}


# ─── Метрики ────────────────────────────────────────────────────────────


def ai_share(start: int, end: int, ai: list[list[int]]) -> float:
    """Доля ИИ-знаков во фрагменте [start, end)."""
    return (
        overlap_length([(start, end)], [(span_start, span_end) for span_start, span_end in ai]) / (end - start)
        if end > start
        else 0.0
    )


def rate(xs: Iterable[bool]) -> float | None:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def fragment_metrics(docs: list[Doc], runs: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Фрагменты целиком человеческие, целиком ИИ и на стыке: как их видит модель."""
    human, ai, boundary = [], [], []
    for doc in docs:
        run = runs[doc.id]
        for start, end, probability in run["chunks"]:
            share = ai_share(start, end, doc.ai)
            flagged = probability >= run["threshold"]
            if share <= 1 - PURE:
                human.append((probability, flagged))
            elif share >= PURE:
                ai.append((probability, flagged))
            else:
                boundary.append((share, flagged))
    bins = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
    return {
        "human": len(human),
        "ai": len(ai),
        "boundary": len(boundary),
        "rocAuc": roc_auc([probability for probability, _ in ai], [probability for probability, _ in human]),
        "falsePositive": rate(flagged for _, flagged in human),
        "truePositive": rate(flagged for _, flagged in ai),
        "boundaryBins": [
            {"from": lo, "to": hi, "n": len(flags), "flagged": rate(flags)}
            for lo, hi in bins
            for flags in [[flagged for share, flagged in boundary if lo <= share < hi or (hi == 1.0 and share == 1.0)]]
        ],
    }


def doc_metrics(doc: Doc, run: dict[str, Any]) -> dict[str, Any]:
    """Один документ: что отметили фрагменты и начало, сколько ИИ нашлось и сколько лишнего."""
    threshold = run["threshold"]
    flagged = [(start, end) for start, end, probability in run["chunks"] if probability >= threshold]
    ai = [(start, end) for start, end in doc.ai]
    ai_len, flagged_len = union_length(ai), union_length(flagged)
    hit = overlap_length(flagged, ai)
    return {
        "kind": doc.kind,
        "chars": len(doc.text),
        "fragments": len(run["chunks"]),
        "seconds": run["seconds"],
        "any": bool(flagged),
        "two": len(flagged) >= 2,
        "head": run["head"] >= threshold,
        # ИИ-кусок найден, если хотя бы один отмеченный фрагмент в основном из ИИ-текста.
        "detected": any(ai_share(start, end, doc.ai) >= 0.5 for start, end in flagged),
        "recall": hit / ai_len if ai_len else None,
        "precision": hit / flagged_len if flagged_len else None,
        "trueShare": ai_len / len(doc.text),
        "flaggedShare": flagged_len / len(doc.text),
    }


def kind_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def avg(key: str) -> float | None:
        values = [row[key] for row in rows if row[key] is not None]
        return mean(values) if values else None

    return {
        "docs": len(rows),
        "any": rate(row["any"] for row in rows),
        "two": rate(row["two"] for row in rows),
        "head": rate(row["head"] for row in rows),
        "detected": rate(row["detected"] for row in rows),
        "recall": avg("recall"),
        "precision": avg("precision"),
        "shareError": mean(abs(row["flaggedShare"] - row["trueShare"]) for row in rows) if rows else None,
    }


def metrics(docs: list[Doc], runs: dict[int, dict[str, Any]]) -> dict[str, Any]:
    per_doc = [doc_metrics(doc, runs[doc.id]) for doc in docs]
    by_kind = {kind: kind_metrics([row for row in per_doc if row["kind"] == kind]) for kind in KINDS}
    fragments = sum(row["fragments"] for row in per_doc)
    return {
        "docs": len(docs),
        "fragments": fragments,
        "fragmentsPerDoc": median(row["fragments"] for row in per_doc),
        "msPerFragment": 1000 * sum(row["seconds"] for row in per_doc) / fragments if fragments else None,
        "msPerDoc": 1000 * mean(row["seconds"] for row in per_doc) if per_doc else None,
        "chunks": fragment_metrics(docs, runs),
        "kinds": by_kind,
        "mixed": kind_metrics([row for row in per_doc if row["kind"] in MIXED]),
    }


# ─── Отчёт ──────────────────────────────────────────────────────────────


def pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.1f} %".replace(".", ",")


def num(x: float | None, digits: int = 3) -> str:
    return "—" if x is None else f"{x:.{digits}f}".replace(".", ",")


def md_table(head: list[str], rows: list[list[str]]) -> str:
    align = ["---", *["--:"] * (len(head) - 1)]
    return "\n".join("| " + " | ".join(row) + " |" for row in [head, align, *rows])


def dataset_md(docs: list[Doc]) -> str:
    rows = []
    for kind in KINDS:
        kind_docs = [doc for doc in docs if doc.kind == kind]
        share = [sum(end - start for start, end in doc.ai) / len(doc.text) for doc in kind_docs]
        rows.append(
            [
                KIND_LABELS[kind],
                str(len(kind_docs)),
                f"{median(len(doc.text) for doc in kind_docs):,.0f}".replace(",", " ") if kind_docs else "—",
                pct(mean(share)) if kind_docs else "—",
            ]
        )
    return md_table(["Вид", "Документов", "Знаков, медиана", "Доля ИИ-знаков"], rows)


def models_md(results: dict[str, dict[str, Any]]) -> str:
    """Главная таблица: модели по столбцам, метрики по строкам."""
    names = list(results)

    def row(label: str, get: Callable[[dict[str, Any]], str]) -> list[str]:
        return [label, *(get(results[name]) for name in names)]

    rows = [
        row("Фрагментов на документ, медиана", lambda result: f"{result['fragmentsPerDoc']:g}"),
        row("мс на фрагмент", lambda result: num(result["msPerFragment"], 0)),
        row("ROC AUC по фрагментам", lambda result: num(result["chunks"]["rocAuc"])),
        row("Человеческие фрагменты выше порога", lambda result: pct(result["chunks"]["falsePositive"])),
        row("ИИ-фрагменты выше порога", lambda result: pct(result["chunks"]["truePositive"])),
        row(
            "Человеческий документ: хоть один фрагмент выше порога", lambda result: pct(result["kinds"]["human"]["any"])
        ),
        row("Человеческий документ: два фрагмента и больше", lambda result: pct(result["kinds"]["human"]["two"])),
        row("Человеческий документ: выше порога по началу", lambda result: pct(result["kinds"]["human"]["head"])),
        row("Смешанный документ: ИИ-кусок найден", lambda result: pct(result["mixed"]["detected"])),
        row("Смешанный документ: два фрагмента и больше", lambda result: pct(result["mixed"]["two"])),
        row("Смешанный документ: найдено по началу", lambda result: pct(result["mixed"]["head"])),
        row("Доля ИИ-знаков под отмеченными фрагментами", lambda result: pct(result["mixed"]["recall"])),
        row("Доля ИИ среди отмеченных знаков", lambda result: pct(result["mixed"]["precision"])),
        row("Ошибка доли ИИ в документе, пункты", lambda result: num(100 * result["mixed"]["shareError"], 1)),
        row("Документ целиком ИИ: найден", lambda result: pct(result["kinds"]["ai"]["detected"])),
    ]
    return md_table(["", *(f"`{name}`" for name in names)], rows)


def kinds_md(m: dict[str, Any]) -> str:
    rows = [
        [
            KIND_LABELS[kind],
            str(m["kinds"][kind]["docs"]),
            pct(m["kinds"][kind]["any"]),
            pct(m["kinds"][kind]["head"]),
            pct(m["kinds"][kind]["detected"]) if kind != "human" else "—",
            pct(m["kinds"][kind]["recall"]) if kind != "human" else "—",
            pct(m["kinds"][kind]["precision"]),
        ]
        for kind in KINDS
    ]
    head = ["Вид", "Документов", "Хоть один фрагмент выше порога", "Выше порога по началу", "ИИ найден"]
    return md_table([*head, "ИИ-знаков под отметками", "ИИ среди отмеченного"], rows)


def boundary_md(results: dict[str, dict[str, Any]]) -> str:
    names = list(results)
    bins = results[names[0]]["chunks"]["boundaryBins"]
    rows = []
    for k, bucket in enumerate(bins):
        label = f"{100 * bucket['from']:.0f}–{100 * bucket['to']:.0f} %"
        rows.append(
            [
                label,
                str(bucket["n"]),
                *(pct(results[name]["chunks"]["boundaryBins"][k]["flagged"]) for name in names),
            ]
        )
    return md_table(["ИИ-знаков во фрагменте", "Фрагментов", *(f"`{name}`" for name in names)], rows)


def overlap_md(results: dict[float, dict[str, Any]]) -> str:
    rows = [
        [
            f"{100 * overlap:g} %",
            f"{result['fragmentsPerDoc']:g}",
            pct(result["kinds"]["human"]["any"]),
            pct(result["mixed"]["detected"]),
            pct(result["mixed"]["recall"]),
            pct(result["mixed"]["precision"]),
            num(result["msPerDoc"], 0),
        ]
        for overlap, result in results.items()
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
    argument_parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    subparsers = argument_parser.add_subparsers(dest="command", required=True)
    for name in ("build", "eval", "report"):
        command_parser = subparsers.add_parser(name)
        command_parser.add_argument("--split", default="test", choices=["test", "valid"])
        command_parser.add_argument("--docs", type=int, default=600, help="документов")
        command_parser.add_argument("--seed", type=int, default=1)
        if name == "eval":
            command_parser.add_argument(
                "--model", required=True, choices=[model for model in models.MODELS if model != "lightgbm"]
            )
            command_parser.add_argument("--overlap", type=float, default=models.OVERLAP)
            command_parser.add_argument("--label", help="имя прогона в кэше и отчёте, по умолчанию имя модели")
        if name == "report":
            command_parser.add_argument("--models", nargs="+", default=["modernbert", "transformer", "mini-frida"])
            command_parser.add_argument(
                "--overlap", type=float, default=models.OVERLAP, help="перекрытие для таблиц моделей"
            )
            command_parser.add_argument("--overlaps", type=float, nargs="*", default=[], help="сравнить перекрытия")
            command_parser.add_argument("--overlap-split", default="valid", choices=["test", "valid"])
            command_parser.add_argument(
                "--overlap-model", default="transformer", help="модель для сравнения перекрытий"
            )
            command_parser.add_argument("--out", type=Path, help="отчёт с метками <!-- long-docs:… -->")
    args = argument_parser.parse_args(argv)
    path = docs_path(args.split, args.docs, args.seed)
    if args.command == "build":
        rows = [json.loads(line) for line in load(data_dir(), "classification", args.split)]
        docs = build(rows, args.docs, args.seed)
        long_dir().mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(asdict(doc), ensure_ascii=False) + "\n" for doc in docs), encoding="utf-8")
        print(f"{len(docs)} документов: {path}")
        print(dataset_md(docs))
        return
    docs = read_docs(path)
    if args.command == "eval":
        loaded = models.load(models.get(args.model))
        label = args.label or args.model
        eval_file = eval_path(args.split, args.docs, args.seed, label, args.overlap)
        print(f"{label} ({models.local_path(loaded.model)}), перекрытие {args.overlap:g}: {eval_file}", flush=True)
        evaluate(docs, loaded, args.overlap, eval_file)
        return
    results = {
        model: metrics(docs, read_eval(eval_path(args.split, args.docs, args.seed, model, args.overlap)))
        for model in args.models
    }
    blocks = {
        "dataset": dataset_md(docs),
        "models": models_md(results),
        "kinds": "\n\n".join(f"`{model}`:\n\n{kinds_md(results[model])}" for model in args.models),
        "boundary": boundary_md(results),
    }
    if args.overlaps:
        overlap_docs = read_docs(docs_path(args.overlap_split, args.docs, args.seed))
        overlap_model = args.overlap_model
        blocks["overlap"] = overlap_md(
            {
                overlap: metrics(
                    overlap_docs, read_eval(eval_path(args.overlap_split, args.docs, args.seed, overlap_model, overlap))
                )
                for overlap in args.overlaps
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

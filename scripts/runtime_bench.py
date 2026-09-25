#!/usr/bin/env python3
"""Время и пиковая память ModernBERT на CPU: onnxruntime (обычный и быстрый экспорт) против PyTorch.

    uv run --group train --group transformer scripts/runtime_bench.py ПАПКА_МОДЕЛИ [--lengths 512,2048,8192]
                                                                       [--threads 8,2] [--out docs/runtime.json]

ПАПКА_МОДЕЛИ — model/ из запуска train_transformer.py (веса PyTorch и config.json).
Скрипт экспортирует модель в ONNX двумя способами, как transformers (полное внимание
с маской) и через FastModernBert, затем на каждой длине и числе потоков меряет
лучшее время из нескольких прогонов. Пиковая память (ru_maxrss) меряется в
отдельном процессе на каждый случай, иначе она копится между замерами. Замеры
идут вперемешку, чтобы фоновая нагрузка сказалась на всех вариантах одинаково.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BACKENDS = ("onnx", "onnx-fast", "torch")


def export(model_dir: Path, out: Path, fast: bool) -> None:
    import torch
    from transformers import AutoModelForSequenceClassification

    from fast_modernbert import FastModernBert

    model: Any = AutoModelForSequenceClassification.from_pretrained(model_dir).eval().float()
    if fast:
        model = FastModernBert(model).eval()
    ids = torch.tensor([[50281, 100, 200, 300, 50282], [50281, 100, 50282, 50283, 50283]])
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    batch = torch.export.Dim("batch", min=1, max=1024)
    seq = torch.export.Dim("sequence", min=2, max=8192)
    torch.onnx.export(
        model,
        (),
        out,
        kwargs={"input_ids": ids, "attention_mask": mask},
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_shapes={"input_ids": {0: batch, 1: seq}, "attention_mask": {0: batch, 1: seq}},
        dynamo=True,
        optimize=True,
    )


def runner(backend: str, model_dir: Path, onnx_dir: Path, threads: int) -> Any:
    """Функция (input_ids, attention_mask) → logits для одного варианта."""
    if backend == "torch":
        import torch
        from transformers import AutoModelForSequenceClassification

        from fast_modernbert import FastModernBert

        torch.set_num_threads(threads)
        model = FastModernBert(AutoModelForSequenceClassification.from_pretrained(model_dir)).eval()

        def run(ids: np.ndarray, mask: np.ndarray) -> Any:
            with torch.inference_mode():
                return model(torch.from_numpy(ids), torch.from_numpy(mask))

        return run
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    path = onnx_dir / ("fast.onnx" if backend == "onnx-fast" else "plain.onnx")
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    return lambda ids, mask: sess.run(None, {"input_ids": ids, "attention_mask": mask})


def inputs(n: int) -> tuple[np.ndarray, np.ndarray]:
    ids = np.random.default_rng(0).integers(5, 50000, (1, n)).astype(np.int64)
    ids[0, 0] = 50281
    return ids, np.ones((1, n), dtype=np.int64)


def peak_mb(backend: str, model_dir: Path, onnx_dir: Path, n: int, threads: int) -> float:
    """Пиковая память отдельного процесса: загрузка модели и один прогон на n токенах."""
    code = (
        "import sys, resource; sys.path.insert(0, sys.argv[1]); import runtime_bench as b; from pathlib import Path; "
        "f = b.runner(sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]), int(sys.argv[6])); "
        "n = int(sys.argv[5]); f(*b.inputs(n)) if n else None; "
        "print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)"
    )
    args = [str(ROOT / "scripts"), backend, str(model_dir), str(onnx_dir), str(n), str(threads)]
    r = subprocess.run([sys.executable, "-c", code, *args], capture_output=True, text=True, check=True)
    rss = int(r.stdout.strip().splitlines()[-1])
    # ru_maxrss на macOS в байтах, на Linux в килобайтах.
    return rss / 2**20 if sys.platform == "darwin" else rss / 2**10


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--lengths", default="512,2048,8192")
    ap.add_argument("--threads", default=f"{os.cpu_count()},2")
    ap.add_argument("--plain-max", type=int, default=4096, help="обычный ONNX мерить только до стольких токенов")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    lengths = [int(x) for x in args.lengths.split(",")]
    threads = [int(x) for x in args.threads.split(",")]
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        onnx_dir = Path(tmp)
        for name, fast in (("plain.onnx", False), ("fast.onnx", True)):
            export(args.model_dir, onnx_dir / name, fast)
        for t in threads:
            runs = {b: runner(b, args.model_dir, onnx_dir, t) for b in BACKENDS}
            for n in lengths:
                todo = [b for b in BACKENDS if b != "onnx" or n <= args.plain_max]
                ids, mask = inputs(n)
                best = dict.fromkeys(todo, float("inf"))
                for b in todo:
                    runs[b](ids, mask)
                for _ in range(8 if n <= 2048 else 3):
                    for b in todo:
                        start = time.perf_counter()
                        runs[b](ids, mask)
                        best[b] = min(best[b], time.perf_counter() - start)
                for b in todo:
                    ram = peak_mb(b, args.model_dir, onnx_dir, n, t)
                    rows.append({"backend": b, "tokens": n, "threads": t, "seconds": best[b], "peak_mb": ram})
                    print(f"{b:10} {n:5} токенов, {t} потоков: {best[b]:.3f} с, пик {ram:.0f} МБ", flush=True)
    if args.out:
        result = {"rows": rows, "cpu": os.uname().machine, "cpus": os.cpu_count()}
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Замеры: {args.out}")


if __name__ == "__main__":
    main()

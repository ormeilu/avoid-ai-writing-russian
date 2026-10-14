#!/usr/bin/env python3
"""Каталог необязательных моделей для `aiw-ru models info`: качество, скорость, память, размер.

    uv run --group train scripts/models_catalog.py [--packages | --hub-only]

Качество берётся из отчётов в docs/models/ (test и valid русской части LLMTrace),
размер скачивания, ревизия и хэши файлов модели — из метаданных репозиториев на
Hugging Face, без загрузки файлов. По ревизии `aiw-ru models install` качает ту
модель, с которой замерено качество, а по хэшам aiw-ru замечает, что скачана
другая. `--hub-only` обновляет только эти поля, без замеров. Скорость и память меряются на этой машине в отдельном процессе для
каждой скачанной модели: загрузка, медиана вероятности для короткого текста
(60 слов) и длинного (около 800 слов, больше 512 токенов), пиковая память процесса.
`--packages` дополнительно ставит `aiw-ru[ml]` во временное окружение и меряет,
сколько места занимают пакеты. Итог ложится в src/aiw_ru/data/models.json и
попадает в пакет: пользователь и агент видят его до того, как что-то скачать.

Модель, которой нет в кэше, остаётся в каталоге без замеров скорости и памяти;
перед выпуском скачайте все (`aiw-ru models install` с именами).
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from aiw_ru import models
from aiw_ru.models import Model

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "src" / "aiw_ru" / "data" / "models.json"
REPORTS = ROOT / "docs" / "models"
CORPUS = ROOT / "tests" / "fixtures" / "corpus" / "human"
LICENSES = {"mit": "MIT", "apache-2.0": "Apache-2.0"}

# Замер в отдельном процессе: пиковая память — только от этой модели.
BENCH = r"""
import json, resource, statistics, sys, time
from aiw_ru import models

name, path = sys.argv[1], sys.argv[2]
text = open(path, encoding="utf-8").read()
short = " ".join(text.split()[:60])
start = time.perf_counter()
loaded = models.load(models.get(name))
load_ms = (time.perf_counter() - start) * 1000


def median_ms(sample, runs):
    loaded.probability(sample)
    times = []
    for _ in range(runs):
        run_start = time.perf_counter()
        loaded.probability(sample)
        times.append(time.perf_counter() - run_start)
    return statistics.median(times) * 1000


result = {"load_ms": load_ms, "short_ms": median_ms(short, 20), "long_ms": median_ms(text, 10)}
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
result["ram_mb"] = rss / 2**20 if sys.platform == "darwin" else rss / 1024
print(json.dumps(result))
"""


def cpu_name() -> str:
    import os

    brand = ""
    if sys.platform == "darwin":
        done = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=False)
        brand = done.stdout.strip()
    return f"{brand or platform.processor() or platform.machine()}, {os.cpu_count()} ядер"


def long_text() -> str:
    """Около 800 слов живого текста из фикстур: длиннее 512 токенов у любой из моделей."""
    words: list[str] = []
    for path in sorted(CORPUS.glob("*.md")):
        words += path.read_text(encoding="utf-8").split()
    while len(words) < 800:
        words += words
    return " ".join(words[:800])


def report(model: Model) -> dict[str, Any] | None:
    path = REPORTS / f"{model.repo.split('/')[-1]}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def quality(m: dict[str, Any]) -> dict[str, Any]:
    test = m["test"]
    return {
        "roc_auc": round(test["roc_auc"], 4),
        "accuracy": round(test["accuracy"], 4),
        "f1_macro": round(test["macro"]["f1"], 4),
        "human_as_ai": round(1 - test["classes"]["human"]["recall"], 4),
        "roc_auc_created": round(test["roc_auc_created"], 4),
        "valid_roc_auc": round(m["valid"]["roc_auc"], 4),
        "test_texts": sum(class_stats["support"] for class_stats in test["classes"].values()),
    }


def hub_files(model: Model) -> dict[str, Any]:
    """Размер того, что скачает aiw-ru, ревизия и хэши файлов модели по метаданным Hugging Face.

    Хэш — имя файла в кэше Hugging Face: sha256 у файлов в LFS, хэш git у остальных. Карточку
    (models.CARD_FILES) не хэшируем: она меняется без переобучения.
    """
    import httpx
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    try:
        info = HfApi().model_info(model.repo, files_metadata=True)
    except (HfHubHTTPError, httpx.HTTPError) as error:
        # Без сети размер считается по скачанной копии: это те же файлы. Ревизию так не узнать.
        local = models.local_path(model) if models.available(model) else None
        if local is None:
            print(f"{model.name}: нет данных с Hugging Face ({error}), размер остаётся прежним", file=sys.stderr)
            return {}
        print(f"{model.name}: нет данных с Hugging Face ({error}), размер по {local}", file=sys.stderr)
        size = sum(
            file.stat().st_size
            for file in local.iterdir()
            if any(fnmatch(file.name, pattern) for pattern in model.files)
        )
        return {"download_mb": round(size / 2**20, 1)}
    files = [
        sibling
        for sibling in info.siblings or []
        if any(fnmatch(sibling.rfilename, pattern) for pattern in model.files)
    ]
    blobs = {
        file.rfilename: file.lfs.sha256 if file.lfs else file.blob_id
        for file in files
        if file.rfilename not in models.CARD_FILES
    }
    size = sum(file.size or 0 for file in files)
    return {"download_mb": round(size / 2**20, 1), "revision": info.sha, "blobs": dict(sorted(blobs.items()))}


def bench(model: Model, text_path: Path) -> dict[str, Any]:
    if not models.available(model):
        print(f"{model.name}: не скачана, скорость и память не мерялись", file=sys.stderr)
        return {}
    done = subprocess.run(
        [sys.executable, "-c", BENCH, model.name, str(text_path)], capture_output=True, text=True, check=True
    )
    return {key: round(value, 1) for key, value in json.loads(done.stdout.strip().splitlines()[-1]).items()}


def packages_mb() -> float:
    """Сколько занимает окружение с aiw-ru[ml] (со всеми зависимостями) на этой платформе."""
    with tempfile.TemporaryDirectory() as tmp:
        env = Path(tmp) / "venv"
        subprocess.run(["uv", "venv", "-q", "-p", "3.12", str(env)], check=True)
        subprocess.run(["uv", "pip", "install", "-q", "--python", str(env), f"{ROOT}[ml]"], check=True, cwd=ROOT)
        size = sum(path.stat().st_size for path in env.rglob("*") if path.is_file() and not path.is_symlink())
    return round(size / 2**20)


# Поля каталога из метаданных Hugging Face; без сети остаются прежними.
HUB_FIELDS = ("download_mb", "revision", "blobs")


def write(catalog: dict[str, Any]) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Каталог: {OUT.relative_to(ROOT)}")


def main(argv: list[str] | None = None) -> None:
    argument_parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    group = argument_parser.add_mutually_exclusive_group()
    group.add_argument("--packages", action="store_true", help="измерить и размер пакетов extra ml")
    group.add_argument("--hub-only", action="store_true", help="только ревизии, хэши и размеры с Hugging Face")
    args = argument_parser.parse_args(argv)
    old = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    if args.hub_only:
        if not old:
            sys.exit(f"{OUT.relative_to(ROOT)} нет: сначала полный замер без --hub-only")
        for entry in old["models"]:
            hub = hub_files(models.get(entry["name"]))
            # Ключи на прежних местах, blobs — перед замерами, как в полном каталоге.
            measured = entry.pop("measured", None)
            entry.update(hub)
            if measured is not None:
                entry["measured"] = measured
        write(old)
        return
    with tempfile.TemporaryDirectory() as tmp:
        text_path = Path(tmp) / "long.txt"
        text_path.write_text(long_text(), encoding="utf-8")
        entries = []
        for model in models.MODELS.values():
            metrics = report(model)
            if metrics is None:
                print(f"{model.name}: нет отчёта в docs/models/, модель пропущена", file=sys.stderr)
                continue
            previous = next((entry for entry in old.get("models", []) if entry["name"] == model.name), {})
            measured = bench(model, text_path)
            entries.append(
                {
                    "name": model.name,
                    "title": model.title,
                    "summary": model.summary,
                    "repo": model.repo,
                    "default": model.default,
                    "base": metrics["params"].get("base"),
                    "license": LICENSES.get(
                        license_id := str(metrics["params"].get("base_license") or "mit").lower(), license_id
                    ),
                    "weights": (metrics.get("inference") or {}).get("weights"),
                    "report": f"docs/models/{model.repo.split('/')[-1]}.md",
                    "quality": quality(metrics),
                    **(hub_files(model) or {key: previous[key] for key in HUB_FIELDS if key in previous}),
                    # Без свежего замера остаётся прошлый: модель могла быть не скачана.
                    "measured": measured or previous.get("measured", {}),
                }
            )
    catalog = {
        "created": datetime.now(UTC).date().isoformat(),
        "measured_on": cpu_name(),
        "python": platform.python_version(),
        "long_text_words": 800,
        "short_text_words": 60,
        "packages_mb": packages_mb() if args.packages else old.get("packages_mb"),
        "packages_platform": f"{sys.platform} {platform.machine()}" if args.packages else old.get("packages_platform"),
        "dataset": "iitolstykh/LLMTrace_classification (ru), test",
        "models": entries,
    }
    write(catalog)


if __name__ == "__main__":
    main()

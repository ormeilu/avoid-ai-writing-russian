#!/usr/bin/env python3
"""Демо для README: запись terminalizer docs/demo/aiw-demo.yml и gif docs/demo.gif.

    uv run --extra ml scripts/demo.py [--no-render]

Команды aiw-ru запускаются по-настоящему на примерах из docs/demo/: ширина 110
колонок, цвет включён, сеть выключена. Набор команды имитируется с неровными
паузами между буквами, после вывода идёт пауза на чтение, сцены разделены
очисткой экрана. `classify --all` показывает все четыре модели, поэтому они
должны быть в кэше: `aiw-ru models install modernbert transformer mini-frida lightgbm`.

Рендер делает terminalizer (`npm install -g terminalizer`, около 10 минут): он
пишет docs/demo/demo-raw.gif в двойном разрешении, в git этот файл не идёт. ffmpeg
уменьшает его до ширины README (800 px) и сводит к одной палитре на 32 цвета, чтобы
gif уложился в лимит хука check-added-large-files (512 КБ). `--no-render` только
обновляет запись.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

from aiw_ru import models

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "docs" / "demo"
RECORDING = DEMO / "aiw-demo.yml"
RAW = DEMO / "demo-raw.gif"
GIF = ROOT / "docs" / "demo.gif"
COLUMNS = 110
# Ширина gif в README и число цветов палитры.
WIDTH, COLORS = 800, 32

# Сцены: команды одного экрана. Между сценами экран очищается.
SCENES: tuple[tuple[str, ...], ...] = (
    ("cat пост.md", "aiw-ru scan пост.md --context blog"),
    (
        "aiw-ru scan правка.md --context blog",
        "aiw-ru validate пост.md правка.md",
        "aiw-ru classify --all правка.md",
        "aiw-ru antiplagiat глава.md",
    ),
)

PROMPT = "\x1b[90m$\x1b[0m "
CLEAR = "\x1b[3J\x1b[H\x1b[2J"
# Паузы, мс: перед первой буквой, перед Enter, до вывода, на чтение и в конце сцены.
FIRST_KEY, ENTER, OUTPUT, READ, SCENE_END = 600, 300, 120, 2200, 7000

CONFIG = """\
# Запись terminalizer для docs/demo.gif. Вывод команд настоящий (aiw-ru, COLUMNS={columns}).
# Собрать заново: uv run --extra ml scripts/demo.py
config:
  command: zsh -f
  cwd: null
  env:
    recording: true
  cols: {columns}
  rows: {rows}
  repeat: 0
  quality: 100
  frameDelay: auto
  maxIdleTime: {idle}
  frameBox:
    type: null
    title: null
    style:
      border: 0px black solid
      boxShadow: none
      margin: 0px
      padding: 0px
  watermark:
    imagePath: null
    style: {{}}
  cursorStyle: block
  fontFamily: "Menlo, Monaco, monospace"
  fontSize: 13
  lineHeight: 1.2
  letterSpacing: 0
  theme:
    background: "#0f1015"
    foreground: "#c5c8cf"
    cursor: "#c5c8cf"
    black: "#0f1015"
    red: "#f0826a"
    green: "#8fcf6a"
    yellow: "#f2c265"
    blue: "#6aa7e0"
    magenta: "#c49be8"
    cyan: "#62b8d4"
    white: "#d5d8de"
    brightBlack: "#6b707a"
    brightRed: "#f0826a"
    brightGreen: "#8fcf6a"
    brightYellow: "#f2c265"
    brightBlue: "#6aa7e0"
    brightMagenta: "#c49be8"
    brightCyan: "#62b8d4"
    brightWhite: "#f4f5f7"
records:
"""


def run(command: str) -> str:
    """Вывод команды в том виде, в каком его покажет терминал: с цветом и переводами строк \\r\\n."""
    name, *args = command.split()
    if name == "cat":
        out = (DEMO / args[0]).read_text(encoding="utf-8")
    else:
        env = {k: v for k, v in os.environ.items() if k != "NO_COLOR"}
        env |= {"FORCE_COLOR": "1", "COLUMNS": str(COLUMNS), "HF_HUB_OFFLINE": "1"}
        done = subprocess.run(
            [sys.executable, "-m", "aiw_ru", *args], cwd=DEMO, env=env, capture_output=True, text=True, check=False
        )
        if done.stderr.strip():
            raise SystemExit(f"{command}: {done.stderr.strip()}")
        out = done.stdout
    lines = [line.rstrip() for line in out.rstrip("\n").split("\n")]
    return "\r\n".join(lines) + "\r\n"


def typed(command: str, rng: random.Random) -> list[tuple[int, str]]:
    """Набор команды по буквам: имя программы жёлтым, аргументы обычным цветом."""
    name = command.split()[0]
    records = []
    for i, ch in enumerate(command):
        delay = FIRST_KEY if i == 0 else rng.randint(38, 76)
        records.append((delay, f"\x1b[1;33m{ch}\x1b[0m" if i < len(name) else ch))
    return records


def recording() -> tuple[list[tuple[int, str]], int]:
    """Записи terminalizer и высота экрана по самой длинной сцене."""
    rng = random.Random(7)
    records: list[tuple[int, str]] = [(0, "\r"), (300, PROMPT)]
    rows = 0
    for n, scene in enumerate(SCENES):
        if n:
            records.append((0, CLEAR + PROMPT))
        height = 1
        for k, command in enumerate(scene):
            out = run(command)
            height += 1 + out.count("\r\n")
            records += typed(command, rng)
            records += [(ENTER, "\r\n"), (OUTPUT, out)]
            records.append((SCENE_END if k == len(scene) - 1 else READ, PROMPT))
        rows = max(rows, height)
    return records, rows


def write(records: list[tuple[int, str]], rows: int) -> None:
    parts = [CONFIG.format(columns=COLUMNS, rows=rows, idle=SCENE_END)]
    for delay, content in records:
        # Строка JSON годится как строка YAML в двойных кавычках; ESC пишем как \e, как его оставляет yamlfmt.
        text = json.dumps(content, ensure_ascii=False).replace("\\u001b", "\\e")
        parts.append(f"  - delay: {delay}\n    content: {text}\n")
    RECORDING.write_text("".join(parts), encoding="utf-8")


def render() -> None:
    for tool in ("terminalizer", "ffmpeg"):
        if not shutil.which(tool):
            raise SystemExit(f"нет {tool}: запись обновлена, gif не собран")
    subprocess.run(["terminalizer", "render", str(RECORDING), "-o", str(RAW)], check=True)
    palette = (
        f"[0:v]scale={WIDTH}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors={COLORS}:stats_mode=full[p];"
        "[b][p]paletteuse=dither=none:diff_mode=rectangle"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(RAW), "-filter_complex", palette, str(GIF)], check=True
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Демо для README: запись terminalizer и gif")
    parser.add_argument("--no-render", action="store_true", help="только запись terminalizer, без gif")
    opts = parser.parse_args()
    if missing := [m.name for m in models.MODELS.values() if not models.status(m).path]:
        raise SystemExit(f"нет моделей {', '.join(missing)}: aiw-ru models install {' '.join(missing)}")
    records, rows = recording()
    write(records, rows)
    print(f"{RECORDING.relative_to(ROOT)}: {len(records)} записей, {rows} строк")
    if not opts.no_render:
        render()
        print(f"{GIF.relative_to(ROOT)}: {GIF.stat().st_size // 1024} КБ")


if __name__ == "__main__":
    main()

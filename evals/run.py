"""Поведенческие проверки скиллов на живом агенте.

  uv run python evals/run.py                          все случаи через `claude -p`
  uv run python evals/run.py --case vak-protected     один случай
  uv run python evals/run.py --agent "codex exec"     другой агент (промпт идёт в stdin)
  uv run python evals/run.py --model claude-sonnet-5  модель для claude
  uv run python evals/run.py --dry                    показать промпты, агента не вызывать

Ответы сохраняются в evals/results/<время>/, итог печатается таблицей.
В CI не запускается: вызов модели стоит денег и недетерминирован.
Код выхода 1, если хоть один случай не прошёл.
"""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from grade import EvalCase, grade

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def load_cases() -> list[EvalCase]:
    data = json.loads((HERE / "cases.json").read_bytes().decode("utf-8"))
    return cast("list[EvalCase]", data["cases"])


def _read(path: Path) -> str:
    return path.read_bytes().decode("utf-8")


def skill_text(name: str) -> str:
    base = ROOT / "skills" / name
    parts = [_read(base / "SKILL.md")]
    if name == "avoid-ai-writing-russian":
        parts.append(_read(base / "references" / "patterns.md"))
    return "\n\n".join(parts)


def build_prompt(c: EvalCase) -> str:
    skills = (
        f"{skill_text('antiplagiat')}\n\n{skill_text('avoid-ai-writing-russian')}"
        if c["skill"] == "antiplagiat"
        else skill_text("avoid-ai-writing-russian")
    )
    return "\n".join(
        [
            "Ниже инструкции скилла. Следуй им строго. Детектор и файлы недоступны: проверки только модельные.",
            "<skill>",
            skills,
            "</skill>",
            "",
            f"Запрос пользователя: {c['request']}",
            "",
            "<text>",
            c["input"],
            "</text>",
        ]
    )


def _opt(args: list[str], name: str) -> str | None:
    if name not in args:
        return None
    i = args.index(name)
    return args[i + 1] if i + 1 < len(args) else None


def ask(prompt: str, args: list[str]) -> str:
    agent = _opt(args, "--agent")
    model = _opt(args, "--model")
    cmd = agent.split(" ") if agent else ["claude", "-p", *(["--model", model] if model else [])]
    # На Windows claude ставится как claude.cmd, а subprocess без which его не найдёт.
    exe = shutil.which(cmd[0]) or cmd[0]
    proc = subprocess.run([exe, *cmd[1:]], input=prompt.encode("utf-8"), capture_output=True, check=False)
    out = proc.stdout.decode("utf-8", errors="replace")
    err = proc.stderr.decode("utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}: {(err or out).strip()[:300]}")
    return out


def _stamp() -> str:
    """Время UTC в ISO 8601 с «-» вместо «:» и «.»: 2026-09-25T10-11-12-345Z."""
    now = datetime.now(UTC)
    return f"{now:%Y-%m-%dT%H-%M-%S}-{now.microsecond // 1000:03d}Z"


def main(args: list[str]) -> int:
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8")
    cases = load_cases()
    only = _opt(args, "--case")
    selected = [c for c in cases if c["id"] == only] if only else cases
    if not selected:
        print(f"нет случая {only}", file=sys.stderr)
        return 2

    if "--dry" in args:
        for c in selected:
            print(f"── {c['id']} ──\n{build_prompt(c)[-600:]}\n")
        return 0

    out_dir = HERE / "results" / _stamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    failed = 0
    for c in selected:
        print(f"{c['id']:<24} ", end="", flush=True)
        try:
            response = ask(build_prompt(c), args)
            (out_dir / f"{c['id']}.md").write_bytes(response.encode("utf-8"))
            g = grade(c, response)
            if not g.passed:
                failed += 1
            print("ok" if g.passed else "ПРОВАЛ\n    " + "\n    ".join(g.failures))
        except Exception as e:  # noqa: BLE001
            # Упавший случай не останавливает остальные.
            failed += 1
            print(f"ОШИБКА {e}")
    print(f"\n{len(selected) - failed}/{len(selected)} прошли. Ответы: {out_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

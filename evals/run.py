"""Поведенческие проверки скиллов на живом агенте.

  uv run python evals/run.py                          все случаи через `claude -p`
  uv run python evals/run.py --case vak-protected     один случай
  uv run python evals/run.py --agent "codex exec"     другой агент (промпт идёт в stdin)
  uv run python evals/run.py --model claude-sonnet-5  модель для claude
  uv run python evals/run.py --timeout 300            сколько секунд ждать агента на случай (по умолчанию 900)
  uv run python evals/run.py --mode workspace         скилл и детектор через aiw-ru, а не в промпте
  uv run python evals/run.py --out путь               куда писать результаты
  uv run python evals/run.py --dry                    показать промпты, агента не вызывать

Режимы. `inline` (по умолчанию): SKILL.md и все файлы references/ вклеены в промпт, детектор
недоступен; так проверяется сам текст скилла. `workspace`: агент работает в пустой папке с текстом
пользователя в text.md, а скилл и детектор получает только командой `aiw-ru` (`aiw-ru skill …`,
`aiw-ru scan …`). Обёртка `aiw-ru` записывает каждый вызов в журнал, и сводка показывает, какие
файлы скилла агент прочитал и какие команды запустил: так проверяется выборочное чтение каталога.

Ответы сохраняются в evals/results/<время>/ (или в --out): ответ агента — <id>.md, даже частичный
при тайм-ауте или сбое; сведения о запуске и оценка — <id>.json; сводка — summary.json, она
переписывается после каждого случая. Сбой агента не останавливает остальные случаи.
В CI не запускается: вызов модели стоит денег и недетерминирован.
Код выхода 1, если хоть один случай не прошёл или агент не ответил; 2 — неизвестный случай.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from grade import EvalCase, grade

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
Mode = Literal["inline", "workspace"]
# Сколько секунд ждать агента на один случай по умолчанию.
TIMEOUT = 900.0

# Обёртка aiw-ru в рабочем режиме: пишет вызов в журнал и запускает настоящую команду.
SHIM = """import json, os, subprocess, sys, time
from pathlib import Path

with Path(os.environ["AIW_RU_EVAL_JOURNAL"]).open("a", encoding="utf-8") as journal:
    journal.write(json.dumps({"time": time.time(), "args": sys.argv[1:]}, ensure_ascii=False) + "\\n")
sys.exit(subprocess.call([sys.executable, "-m", "aiw_ru", *sys.argv[1:]]))
"""


def load_cases() -> list[EvalCase]:
    data = json.loads((HERE / "cases.json").read_bytes().decode("utf-8"))
    return cast("list[EvalCase]", data["cases"])


def _read(path: Path) -> str:
    return path.read_bytes().decode("utf-8")


def skill_text(name: str) -> str:
    """SKILL.md и файлы из references/ с путями: у агента в проверке нет доступа к файлам."""
    base = ROOT / "skills" / name
    parts = [_read(base / "SKILL.md")]
    for path in sorted((base / "references").glob("*.md")):
        parts.append(f'<file path="references/{path.name}">\n{_read(path)}\n</file>')
    return "\n\n".join(parts)


def build_prompt(c: EvalCase) -> str:
    """Промпт режима inline: скилл целиком в промпте, детектор недоступен."""
    skills = (
        f"{skill_text('antiplagiat')}\n\n{skill_text('avoid-ai-writing-russian')}"
        if c["skill"] == "antiplagiat"
        else skill_text("avoid-ai-writing-russian")
    )
    return "\n".join(
        [
            "Ниже инструкции скилла и его файлы в тегах <file>. Следуй им строго. Детектор недоступен: проверки только модельные.",
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


def workspace_prompt(c: EvalCase) -> str:
    """Промпт режима workspace: скилл и детектор агент получает сам командой aiw-ru."""
    skill = c["skill"]
    return "\n".join(
        [
            f"Работай по скиллу {skill}. Скиллы доступны только через команду aiw-ru: `aiw-ru skill {skill}` "
            f"печатает SKILL.md, `aiw-ru skill {skill} references/<файл>` — файл из references/. "
            + (
                "Основной скилл, на который он ссылается, — `aiw-ru skill avoid-ai-writing-russian`. "
                if skill == "antiplagiat"
                else ""
            )
            + "Файлы каталога читай выборочно, как велит SKILL.md. Детектор — та же команда: aiw-ru scan, "
            "aiw-ru validate и другие, которые называет скилл.",
            "Текст пользователя лежит в файле text.md в текущей папке.",
            "",
            f"Запрос пользователя: {c['request']}",
        ]
    )


@dataclass(frozen=True, slots=True)
class AgentRun:
    """Запуск агента. Сбой агента — не исключение, а поле error; частичный вывод сохраняется."""

    command: list[str]
    output: str
    stderr: str
    returncode: int | None
    seconds: float
    timed_out: bool = False
    error: str | None = None


def _text(data: bytes | str | None) -> str:
    if data is None:
        return ""
    return data if isinstance(data, str) else data.decode("utf-8", errors="replace")


def agent_command(agent: str | None, model: str | None, mode: Mode) -> list[str]:
    """Команда агента. --agent разбирается как строка оболочки: пути с пробелами берут в кавычки."""
    if agent:
        return shlex.split(agent, posix=os.name != "nt")
    command = ["claude", "-p", *(["--model", model] if model else [])]
    if mode == "workspace":
        command += ["--allowedTools", "Bash(aiw-ru:*)"]
    return command


def ask(
    prompt: str, command: list[str], timeout: float, cwd: Path | None = None, env: dict[str, str] | None = None
) -> AgentRun:
    """Промпт агенту в stdin; ответ — его stdout. Тайм-аут, код выхода и отказ запуска — в error."""
    # На Windows claude ставится как claude.cmd, а subprocess без which его не найдёт.
    exe = shutil.which(command[0], path=(env or os.environ).get("PATH")) or command[0]
    started = time.monotonic()
    try:
        proc = subprocess.run(
            [exe, *command[1:]],
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=timeout,
            cwd=cwd,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        seconds = time.monotonic() - started
        return AgentRun(
            command, _text(error.stdout), _text(error.stderr), None, seconds, True, f"тайм-аут {timeout:g} с"
        )
    except OSError as error:
        reason = error.strerror or str(error)
        return AgentRun(command, "", "", None, time.monotonic() - started, error=f"агент не запустился: {reason}")
    out, err = _text(proc.stdout), _text(proc.stderr)
    error = None if proc.returncode == 0 else f"код выхода {proc.returncode}: {(err or out).strip()[:300]}"
    return AgentRun(command, out, err, proc.returncode, time.monotonic() - started, error=error)


@dataclass(frozen=True, slots=True)
class Workspace:
    folder: Path
    env: dict[str, str]
    journal: Path


def prepare_workspace(c: EvalCase, folder: Path) -> Workspace:
    """Папка агента с text.md и обёрткой aiw-ru первой в PATH; журнал вызовов рядом, вне папки агента."""
    work = folder / "work"
    bin_dir = folder / "bin"
    work.mkdir(parents=True)
    bin_dir.mkdir()
    (work / "text.md").write_text(c["input"], encoding="utf-8")
    shim = bin_dir / "aiw_ru_logged.py"
    shim.write_text(SHIM, encoding="utf-8")
    posix = bin_dir / "aiw-ru"
    posix.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{shim}" "$@"\n', encoding="utf-8")
    posix.chmod(0o755)
    (bin_dir / "aiw-ru.cmd").write_text(f'@"{sys.executable}" "{shim}" %*\r\n', encoding="utf-8")
    journal = folder / "journal.jsonl"
    journal.touch()
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "AIW_RU_EVAL_JOURNAL": str(journal),
        "AIW_RU_CONFIG": "",
    }
    return Workspace(work, env, journal)


def journal_summary(journal: Path) -> tuple[list[str], list[str]]:
    """Что агент прочитал (`aiw-ru skill`) и какие команды детектора запустил, по порядку."""
    reads: list[str] = []
    commands: list[str] = []
    for line in journal.read_text(encoding="utf-8").splitlines():
        args: list[str] = json.loads(line)["args"]
        if args and args[0] == "skill":
            if len(args) >= 2:
                reads.append(args[2] if len(args) > 2 else "SKILL.md")
        elif args:
            commands.append(args[0])
    return reads, commands


@dataclass(slots=True)
class CaseResult:
    id: str
    mode: Mode
    passed: bool
    error: str | None
    timed_out: bool
    seconds: float
    failures: list[str] = field(default_factory=list)
    reads: list[str] | None = None
    commands: list[str] | None = None

    def to_json(self) -> dict[str, Any]:
        """Строка сводки: поля в camelCase, как в остальных JSON проекта."""
        data = asdict(self)
        data["timedOut"] = data.pop("timed_out")
        return data


def run_case(c: EvalCase, args: argparse.Namespace, out_dir: Path) -> CaseResult:
    command = agent_command(args.agent, args.model, args.mode)
    if args.mode == "workspace":
        workspace = prepare_workspace(c, out_dir / "workspaces" / c["id"])
        run = ask(workspace_prompt(c), command, args.timeout, workspace.folder, workspace.env)
        reads, commands = journal_summary(workspace.journal)
    else:
        run = ask(build_prompt(c), command, args.timeout)
        reads = commands = None
    (out_dir / f"{c['id']}.md").write_bytes(run.output.encode("utf-8"))
    failures: list[str] = []
    if run.error is None:
        failures = grade(c, run.output).failures
    result = CaseResult(
        c["id"],
        args.mode,
        run.error is None and not failures,
        run.error,
        run.timed_out,
        round(run.seconds, 2),
        failures,
        reads,
        commands,
    )
    details: dict[str, Any] = {**asdict(run), "result": result.to_json()}
    (out_dir / f"{c['id']}.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _stamp() -> str:
    """Время UTC в ISO 8601 с «-» вместо «:» и «.»: 2026-09-25T10-11-12-345Z."""
    now = datetime.now(UTC)
    return f"{now:%Y-%m-%dT%H-%M-%S}-{now.microsecond // 1000:03d}Z"


def parse_args(args: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="evals/run.py", description="Поведенческие проверки скиллов на живом агенте.")
    parser.add_argument("--case", help="один случай по id")
    parser.add_argument("--agent", help="команда агента, промпт идёт в stdin; по умолчанию claude -p")
    parser.add_argument("--model", help="модель для claude")
    parser.add_argument("--timeout", type=float, default=TIMEOUT, help="секунд на случай")
    parser.add_argument("--mode", choices=("inline", "workspace"), default="inline")
    parser.add_argument("--out", type=Path, help="папка результатов; по умолчанию evals/results/<время>")
    parser.add_argument("--dry", action="store_true", help="показать промпты, агента не вызывать")
    return parser.parse_args(args)


def main(argv: list[str]) -> int:
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    cases = load_cases()
    selected = [c for c in cases if c["id"] == args.case] if args.case else cases
    if not selected:
        print(f"нет случая {args.case}", file=sys.stderr)
        return 2
    prompt_of = workspace_prompt if args.mode == "workspace" else build_prompt

    if args.dry:
        for c in selected:
            print(f"── {c['id']} ──\n{prompt_of(c)[-600:]}\n")
        return 0

    out_dir: Path = args.out or HERE / "results" / _stamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[CaseResult] = []
    for c in selected:
        print(f"{c['id']:<24} ", end="", flush=True)
        result = run_case(c, args, out_dir)
        results.append(result)
        # Сводка после каждого случая: если прогон оборвётся, готовое не пропадёт.
        summary = {"mode": args.mode, "cases": [r.to_json() for r in results]}
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        if result.error:
            print(f"ОШИБКА {result.error}")
        else:
            print("ok" if result.passed else "ПРОВАЛ\n    " + "\n    ".join(result.failures))
        if result.reads is not None:
            print(
                f"    прочитано: {', '.join(result.reads) or 'ничего'}; команды: {', '.join(result.commands or []) or 'нет'}"
            )
    failed = sum(not r.passed for r in results)
    print(f"\n{len(selected) - failed}/{len(selected)} прошли. Ответы: {out_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

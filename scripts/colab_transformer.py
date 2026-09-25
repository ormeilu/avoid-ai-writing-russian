#!/usr/bin/env python3
"""Дообучение трансформера в Google Colab на GPU через официальный Colab CLI (google-colab-cli).

    uv run scripts/colab_transformer.py up       [--session ИМЯ] [--gpu T4]
    uv run scripts/colab_transformer.py setup    [--session ИМЯ] [--torch ВЕРСИЯ] [--code-only]
    uv run scripts/colab_transformer.py start    [--session ИМЯ] --name ИМЯ --job "pilot --base ..." [--job ...]
    uv run scripts/colab_transformer.py status   [--session ИМЯ] --name ИМЯ
    uv run scripts/colab_transformer.py fetch    [--session ИМЯ] ПАПКА_НА_VM ЛОКАЛЬНАЯ_ПАПКА
    uv run scripts/colab_transformer.py down     [--session ИМЯ]

up берёт VM с GPU (по умолчанию T4). Если Colab не даёт GPU из-за квоты, скрипт
останавливается: обучать на CPU не имеет смысла. setup отправляет на VM
исходники (src/aiw_ru, scripts/, pyproject.toml, uv.lock), ставит зависимости
групп train и transformer с версиями из uv.lock и скачивает LLMTrace с
Hugging Face той ревизии, что закреплена в scripts/llmtrace.py: гигабайты
корпуса не идут через этот компьютер. start запускает на VM
scripts/train_transformer.py в фоне, задания --job идут друг за другом, лог
пишется в /content/logs/ИМЯ.log. status показывает хвост лога и прогресс.
fetch забирает папку запуска архивом. down освобождает VM.

Colab CLI должен быть авторизован (`colab sessions` работает). Путь к нему —
ключ --colab или переменная COLAB_CLI, файл состояния сессий — --colab-config.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REMOTE = "/content/aiw-ru"
DATA = "/content/data"
LOGS = "/content/logs"
BUNDLE = ("pyproject.toml", "uv.lock", "README.md", "LICENSE", "NOTICE.md", "src/aiw_ru", "scripts")
SPLITS = ("train", "valid", "test")


def colab(args: argparse.Namespace, *cmd: str, stdin: str | None = None, check: bool = True) -> str:
    """Вызывает Colab CLI и возвращает его вывод; вывод сразу показывается."""
    base = [args.colab]
    if args.colab_config:
        base += ["--config", str(args.colab_config)]
    done = subprocess.run([*base, *cmd], input=stdin, capture_output=True, text=True, check=False)
    out = done.stdout + done.stderr
    print(out, end="" if out.endswith("\n") else "\n", flush=True)
    if check and done.returncode:
        raise SystemExit(f"colab {' '.join(cmd[:1])}: код {done.returncode}")
    return out


def execute(args: argparse.Namespace, code: str, timeout: float) -> str:
    """Выполняет код Python в ядре VM; ядро живёт между вызовами."""
    return colab(args, "exec", "-s", args.session, "--timeout", str(timeout), stdin=code)


def git_commit() -> str:
    def git(*a: str) -> str:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()

    sha = git("rev-parse", "--short", "HEAD")
    return (sha + ("+правки" if git("status", "--porcelain") else "")) if sha else "неизвестен"


def requirements() -> str:
    """Версии групп train и transformer из uv.lock, без самого пакета aiw-ru."""
    cmd = ["uv", "export", "--frozen", "--no-hashes", "--no-emit-project", "--no-header"]
    cmd += ["--group", "train", "--group", "transformer", "--format", "requirements.txt"]
    done = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    if done.returncode:
        raise SystemExit(f"uv export: {done.stderr.strip()}")
    return done.stdout


def bundle(path: Path) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for name in BUNDLE:
            src = ROOT / name
            if src.exists():
                tar.add(src, arcname=name, filter=lambda t: None if "__pycache__" in t.name else t)


def cmd_up(args: argparse.Namespace) -> None:
    out = colab(args, "new", "-s", args.session, "--gpu", args.gpu, check=False)
    if "READY" not in out:
        raise SystemExit(f"Colab не дал {args.gpu}: обучение не запускается, на CPU оно не имеет смысла")


def cmd_setup(args: argparse.Namespace) -> None:
    reqs = requirements()
    if args.torch == "builtin":
        # torch из образа Colab: без torch, triton и библиотек CUDA из uv.lock.
        drop = ("torch==", "triton==", "nvidia-", "cuda-")
        reqs = "\n".join(line for line in reqs.splitlines() if not line.startswith(drop))
    elif args.torch:
        reqs = "\n".join(f"torch=={args.torch}" if line.startswith("torch==") else line for line in reqs.splitlines())
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "aiw-ru.tar.gz"
        bundle(archive)
        req_file = Path(tmp) / "requirements.txt"
        req_file.write_text(reqs + "\n", encoding="utf-8")
        colab(args, "upload", "-s", args.session, str(archive), "/content/aiw-ru.tar.gz")
        colab(args, "upload", "-s", args.session, str(req_file), "/content/requirements.txt")
    unpack = f"""
import os, tarfile
os.makedirs({REMOTE!r}, exist_ok=True)
os.makedirs({LOGS!r}, exist_ok=True)
tarfile.open("/content/aiw-ru.tar.gz").extractall({REMOTE!r}, filter="data")
print("исходники в", {REMOTE!r})
"""
    execute(args, unpack, 300)
    if args.code_only:
        return
    replace_torch = args.torch != "builtin"
    code = f"""
import subprocess
cmd = ["uv", "pip", "install", "--system", "-r", "/content/requirements.txt"]
done = subprocess.run(cmd, capture_output=True, text=True)
print(done.stdout[-3000:], done.stderr[-3000:])
done.check_returncode()
if {replace_torch!r}:
    # torchvision и torchaudio в образе собраны под его torch; с другим torch transformers не импортируется.
    subprocess.run(["uv", "pip", "uninstall", "--system", "torchvision", "torchaudio"], check=False)
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.is_available())
x = torch.randn(256, 256, device="cuda", dtype=torch.float16)
print("GPU:", torch.cuda.get_device_name(0), float((x @ x).float().sum()))
"""
    execute(args, code, 1800)
    env = f"dict(os.environ, AIW_RU_DATA={DATA!r}, PYTHONPATH={REMOTE + '/src:' + REMOTE + '/scripts'!r})"
    fetch = "; ".join(
        f"subprocess.run([sys.executable, 'scripts/llmtrace.py', 'fetch', '--set', 'classification', "
        f"'--split', {s!r}], cwd={REMOTE!r}, env={env}, check=True)"
        for s in SPLITS
        if s in args.splits
    )
    execute(args, f"import os, subprocess, sys\n{fetch}\nprint(os.listdir({DATA!r}))", 3600)


def cmd_start(args: argparse.Namespace) -> None:
    py = "python3"
    jobs = " && ".join(f"{py} scripts/train_transformer.py {job}" for job in args.job)
    log = f"{LOGS}/{args.name}.log"
    code = f"""
import os, subprocess
env = dict(os.environ, AIW_RU_DATA={DATA!r}, AIW_RU_GIT_COMMIT={git_commit()!r},
           PYTHONPATH={REMOTE + "/src:" + REMOTE + "/scripts"!r}, PYTHONUNBUFFERED="1",
           TOKENIZERS_PARALLELISM="true")
log = open({log!r}, "a")
p = subprocess.Popen(["bash", "-c", {jobs!r} + "; echo EXIT_CODE=$?"], cwd={REMOTE!r}, env=env,
                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
open({log!r} + ".pid", "w").write(str(p.pid))
print("PID", p.pid, "лог", {log!r})
"""
    execute(args, code, 120)


def cmd_status(args: argparse.Namespace) -> None:
    log = f"{LOGS}/{args.name}.log"
    code = f"""
import os, subprocess
pid = int(open({log!r} + ".pid").read()) if os.path.exists({log!r} + ".pid") else 0
alive = pid and subprocess.run(["kill", "-0", str(pid)], capture_output=True).returncode == 0
print("процесс", pid, "идёт" if alive else "завершён")
lines = open({log!r}, encoding="utf-8", errors="replace").read().splitlines() if os.path.exists({log!r}) else []
print("\\n".join(line for line in lines if "it/s]" not in line)[-{args.chars}:])
print(subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip())
"""
    execute(args, code, 120)


def cmd_fetch(args: argparse.Namespace) -> None:
    remote = args.remote.rstrip("/")
    archive = f"/content/{Path(remote).name}.tar.gz"
    exclude = " ".join(f"--exclude={shlex.quote(x)}" for x in ("*.pt", "cache"))
    code = f"""
import subprocess
subprocess.run("tar {exclude} -czf {archive} -C {Path(remote).parent} {Path(remote).name}", shell=True, check=True)
print(subprocess.run(["ls", "-la", {archive!r}], capture_output=True, text=True).stdout)
"""
    execute(args, code, 600)
    args.local.mkdir(parents=True, exist_ok=True)
    local = args.local / Path(archive).name
    colab(args, "download", "-s", args.session, archive, str(local))
    with tarfile.open(local) as tar:
        tar.extractall(args.local, filter="data")
    print(f"Забрано в {args.local / Path(remote).name}")


def cmd_down(args: argparse.Namespace) -> None:
    colab(args, "stop", "-s", args.session, check=False)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--colab", default=os.environ.get("COLAB_CLI", "colab"), help="путь к Colab CLI")
    ap.add_argument("--colab-config", type=Path, help="файл состояния сессий Colab CLI")
    ap.add_argument("--session", default="aiw-transformer", help="имя сессии Colab")
    sub = ap.add_subparsers(dest="command", required=True)
    up = sub.add_parser("up", help="взять VM с GPU")
    up.add_argument("--gpu", default="T4", help="T4, L4, A100 или H100, что разрешает аккаунт")
    setup = sub.add_parser("setup", help="исходники, зависимости и корпус на VM")
    setup.add_argument("--torch", help="версия torch вместо версии из uv.lock; builtin — torch из образа Colab")
    setup.add_argument("--splits", nargs="+", default=list(SPLITS), help="какие части корпуса скачать")
    setup.add_argument("--code-only", action="store_true", help="только обновить исходники на VM")
    start = sub.add_parser("start", help="запустить задания train_transformer.py в фоне")
    start.add_argument("--name", required=True, help="имя лога")
    start.add_argument("--job", action="append", required=True, help="аргументы train_transformer.py")
    status = sub.add_parser("status", help="хвост лога и загрузка GPU")
    status.add_argument("--name", required=True, help="имя лога")
    status.add_argument("--chars", type=int, default=3000, help="сколько знаков лога показать")
    fetch = sub.add_parser("fetch", help="забрать папку с VM")
    fetch.add_argument("remote", help="папка на VM")
    fetch.add_argument("local", type=Path, help="куда распаковать")
    sub.add_parser("down", help="освободить VM")
    args = ap.parse_args(argv)
    {
        "up": cmd_up,
        "setup": cmd_setup,
        "start": cmd_start,
        "status": cmd_status,
        "fetch": cmd_fetch,
        "down": cmd_down,
    }[args.command](args)


if __name__ == "__main__":
    main()

"""Проверка собранного wheel вне рабочей копии: пакет работает сам, без исходников и ML-пакетов.

    uv build
    uv run python scripts/check_package.py dist/aiw_ru-*.whl

Wheel ставится в одноразовое окружение (`uv run --isolated --no-project --with ПУТЬ`), а команды
запускаются из пустой временной папки: так пакет не подхватит модули и скиллы из рабочей копии.
Проверяется: `--version`, `scan` и `validate` на выдуманном тексте, каждый файл каждого скилла через
`aiw-ru skill`, что `aiw_ru` загружен не из рабочей копии и что пакеты extra `ml` и обучения не
стоят. Код выхода 1 и список провалов, если что-то не так.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Пакеты, которых в обычной установке быть не должно: extra ml, обучение и трансформер.
HEAVY = ("onnxruntime", "lightgbm", "numpy", "scipy", "tokenizers", "huggingface_hub", "torch", "transformers")
TEXT = "Давайте разберёмся: данный инструмент играет ключевую роль в складском учёте."
EDITED = "Инструмент помогает вести складской учёт."
PROBE = f"""
import importlib.util, json, aiw_ru
print(json.dumps({{"file": aiw_ru.__file__, "heavy": [m for m in {HEAVY!r} if importlib.util.find_spec(m)]}}))
"""


@dataclass(frozen=True, slots=True)
class Result:
    code: int
    out: str
    err: str


Runner = Callable[[list[str], Path, str], Result]


def run(argv: list[str], cwd: Path, stdin: str = "") -> Result:
    proc = subprocess.run(argv, cwd=cwd, input=stdin, capture_output=True, text=True, encoding="utf-8", check=False)
    return Result(proc.returncode, proc.stdout, proc.stderr)


def check(wheel: Path, runner: Runner = run) -> list[str]:
    """Провалы проверки; пустой список — пакет в порядке."""
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="aiw-ru-wheel-") as tmp:
        folder = Path(tmp)
        base = ["uv", "run", "--isolated", "--no-project", "--with", str(wheel.resolve())]

        def aiw(*args: str, stdin: str = "") -> Result:
            return runner([*base, "aiw-ru", *args], folder, stdin)

        def expect(name: str, result: Result, code: int = 0) -> bool:
            if result.code != code:
                failures.append(
                    f"{name}: код {result.code}, ожидался {code}: {(result.err or result.out).strip()[:200]}"
                )
                return False
            return True

        expect("--version", aiw("--version"))
        scan = aiw("scan", "--json", stdin=TEXT)
        if expect("scan", scan) and not json.loads(scan.out)["issues"]:
            failures.append("scan: на тексте со штампами нет находок")
        (folder / "before.md").write_text(TEXT, encoding="utf-8")
        (folder / "after.md").write_text(EDITED, encoding="utf-8")
        validate = aiw("validate", "--json", "before.md", "after.md")
        if expect("validate", validate) and "ok" not in json.loads(validate.out):
            failures.append("validate: в JSON нет поля ok")
        listed = aiw("skill", "--json")
        if expect("skill --json", listed):
            skills = json.loads(listed.out)
            if not skills:
                failures.append("skill --json: скиллов нет")
            for skill in skills:
                expect(f"skill {skill['name']}", aiw("skill", skill["name"]))
                for file in skill["files"]:
                    expect(f"skill {skill['name']} {file}", aiw("skill", skill["name"], file))
        probe = runner([*base, "python", "-c", PROBE], folder, "")
        if expect("python", probe):
            found = json.loads(probe.out)
            if Path(found["file"]).resolve().is_relative_to(ROOT):
                failures.append(f"aiw_ru загружен из рабочей копии: {found['file']}")
            if found["heavy"]:
                failures.append(f"в обычной установке стоят тяжёлые пакеты: {', '.join(found['heavy'])}")
    return failures


def main(argv: list[str] | None = None) -> int:
    wheels = [Path(arg) for arg in (sys.argv[1:] if argv is None else argv)]
    if not wheels or any(wheel.suffix != ".whl" or not wheel.is_file() for wheel in wheels):
        print("check_package: передайте путь к собранному .whl", file=sys.stderr)
        return 2
    failed = False
    for wheel in wheels:
        failures = check(wheel)
        print(f"{wheel.name}: {'в порядке' if not failures else 'провал'}")
        for failure in failures:
            print(f"  {failure}")
        failed = failed or bool(failures)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

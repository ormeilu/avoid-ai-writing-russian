"""aiw-ru — командная строка детектора.

    aiw-ru scan [файл…] [--context РЕЖИМ] [--json | --jsonl] [--min P0|P1|P2]
    aiw-ru antiplagiat [файл] [--system СИСТЕМА] [--config ПУТЬ] [--json | --jsonl]
    aiw-ru validate ИСХОДНЫЙ ИСПРАВЛЕННЫЙ [--context РЕЖИМ] [--json]
    aiw-ru calibrate --doc ФАЙЛ (--marked ФАЙЛ | --share ДОЛЯ) [--doc … ] [--system СИСТЕМА] [--config ПУТЬ]
    aiw-ru judge [файл] | judge score ФАЙЛ --answers ФАЙЛ [--answers …] [--marked ФАЙЛ] [--json]
    aiw-ru classify [файл…] [--json | --jsonl]
    aiw-ru models [install]

Коды выхода: 0 — успех; 1 — validate нашёл повреждения или scan с --fail-above
превысил порог; 2 — ошибка использования или ввода-вывода, нет нужной модели.
"""

from __future__ import annotations

import errno
import json
import math
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from rich import box
from rich.cells import cell_len, set_cell_size
from rich.console import Console, JustifyMethod, RenderableType
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from aiw_ru import models, skills
from aiw_ru.antiplagiat import (
    DEFAULT_MODEL,
    FEATURE_NAMES,
    SYSTEMS,
    Calibration,
    CalibrationFile,
    Model,
    System,
    antiplagiat,
    balanced_accuracy,
    document_sample,
    empty_calibration_file,
    fit,
    fit_share,
    label_fragments,
    marked_from_heatmap,
    share_error,
)
from aiw_ru.categories import TYPE_TO_SECTION
from aiw_ru.compat import js_round, jsre, num_str, to_fixed, trim
from aiw_ru.detect import TYPE_LABELS, analyze
from aiw_ru.judge import judge_fragments, judge_prompt, judge_score
from aiw_ru.text import cyrillic_share, model_text, plural, words
from aiw_ru.types import CONTEXT_MODES, PROFILE_TO_MODE, AnalysisResult, ContextMode, Record, Severity
from aiw_ru.validate import validate

USAGE = """aiw-ru — приметы ИИ-стиля в русском тексте

Команды:
  scan [файл…]                 найти приметы (без файла — читать stdin)
  antiplagiat [файл]           оценка по фрагментам в духе модуля ИИ-детекции «Антиплагиата»
                               или «Думейта» (--system domate)
  validate ИСХОДНЫЙ НОВЫЙ      проверить, что правка не повредила код, числа, цитаты, ссылки,
                               не сняла оговорки и не добавила имён, чисел и дат
  calibrate --doc Ф --marked Ф подстроить модель antiplagiat под ваши отчёты: документ и файл
                               с фрагментами, которые отчёт подсветил как ИИ (текст через пустую
                               строку или тепловая карта в JSON: из неё берутся красные фрагменты)
  calibrate --doc Ф --share N  то же, если известна только итоговая доля ИИ из отчёта, %
  judge [файл]                 задание для слепого судьи: свежий агент без контекста оценивает
                               пронумерованные фрагменты по шкале none/low/medium/high
  judge score Ф --answers A    сводка ответов судьи (--answers на каждый прогон): средний уровень
                               фрагментов, совпадение прогонов; с --marked — ROC AUC судьи и
                               детектора против разметки отчёта
  classify [файл…]             вероятность ИИ по необязательной модели; текст длиннее окна модели
                               проверяется ещё и по фрагментам с перекрытием (на фрагмент уходит
                               примерно столько, сколько на текст в 800 слов в models info);
                               --all — по всем установленным рядом, видно, согласны ли они;
                               YAML-шапку, код и служебные части статьи (литература, ключевые
                               слова, английская аннотация) модели не читают, а текст и фрагменты
                               не на русском не оценивают: вероятности для них нет
  models                       необязательные модели: установлены ли и где лежат
  models info [имя]            качество, скорость, память и размер моделей до скачивания
  models guide                 как пользоваться классификацией: какую модель взять, как читать
                               вероятность и фрагменты длинного текста, как перепроверить фрагмент
  models install [имя…]        скачать модели с Hugging Face (нужен extra ml):
                               modernbert, transformer, mini-frida, lightgbm; без имени —
                               modernbert и lightgbm, transformer и mini-frida только по имени
  skill [имя] [файл]           текст скилла для агента, если стоит только aiw-ru, без плагина:
                               без имени — список скиллов, с именем — SKILL.md,
                               с файлом — файл скилла (references/vocabulary.md)

Параметры:
  --context РЕЖИМ   general | academic | technical | social | chat
                    или профиль скилла: vak | docs | blog | telegram | business-email | chat
  --json            вывод в JSON
  --jsonl           scan, antiplagiat: на входе по документу в строке ({"text": "…"}),
                    на выходе по строке JSON на документ; для прогонов по корпусам
  --min P0|P1|P2    показывать находки не ниже уровня (scan)
  --fail-above N    scan: код выхода 1, если оценка выше N
  --config ПУТЬ     файл калибровки (по умолчанию ./.aiw-ru.json, затем ~/.config/aiw-ru.json)
  --system СИСТЕМА  antiplagiat, calibrate: antiplagiat (по умолчанию) или domate; у каждой
                    системы своя калибровка в том же файле
  --model ИМЯ       scan, antiplagiat, classify: modernbert, transformer, mini-frida или lightgbm
                    (по умолчанию первая установленная в этом порядке)
  --no-model        scan, antiplagiat: не показывать вероятность от модели, даже если она есть
  --no-fragments    classify: только вероятность по началу текста, без фрагментов
  --max-fragments N classify: проверить не больше N первых фрагментов (0 — все)
  -h, --help        справка
"""


class UsageError(Exception):
    pass


@dataclass
class Report:
    doc: str
    marked: str | None = None
    share: float | None = None


@dataclass
class Args:
    cmd: str = ""
    files: list[str] = field(default_factory=list)
    context: ContextMode | None = None
    json: bool = False
    jsonl: bool = False
    min: Severity = "P2"
    fail_above: float | None = None
    config: str | None = None
    # Пары для calibrate: документ и разметка из отчёта.
    reports: list[Report] = field(default_factory=list)
    no_model: bool = False
    # Модель для вероятности: имя из models.MODELS или None — первая установленная.
    model_name: str | None = None
    # classify: вероятности всех установленных моделей рядом.
    all_models: bool = False
    # classify: длинный текст без проверки по фрагментам; предел фрагментов (None — все).
    no_fragments: bool = False
    max_fragments: int | None = None
    # antiplagiat, calibrate: система проверки, под которую калибруется оценка.
    system: System = "antiplagiat"
    # judge score: файлы с ответами судьи (по одному на прогон) и разметка отчёта.
    answers: list[str] = field(default_factory=list)
    judge_marked: str | None = None
    help: bool = False


def parse(argv: list[str]) -> Args:
    a = Args()

    def need(i: int, flag: str) -> str:
        if i + 1 >= len(argv):
            raise UsageError(f"{flag} требует значения")
        return argv[i + 1]

    i = 0
    while i < len(argv):
        x = argv[i]
        if x in ("-h", "--help"):
            a.help = True
        elif x == "--json":
            a.json = True
        elif x == "--jsonl":
            a.jsonl = True
        elif x == "--no-model":
            a.no_model = True
        elif x == "--all":
            a.all_models = True
        elif x == "--no-fragments":
            a.no_fragments = True
        elif x == "--max-fragments":
            v = need(i, x)
            i += 1
            if not v.isdigit():
                raise UsageError(f"--max-fragments ждёт целое число 0 и больше: {v}")
            a.max_fragments = int(v) or None
        elif x == "--model":
            v = need(i, x)
            i += 1
            if v not in models.MODELS:
                raise UsageError(f"неизвестная --model: {v}; есть: {', '.join(models.MODELS)}")
            a.model_name = v
        elif x == "--context":
            v = need(i, x)
            i += 1
            mode = v if v in CONTEXT_MODES else PROFILE_TO_MODE.get(v)
            if not mode:
                raise UsageError(f"неизвестный --context: {v}")
            a.context = mode
        elif x == "--min":
            v = need(i, x)
            i += 1
            if v not in ("P0", "P1", "P2"):
                raise UsageError(f"неизвестный --min: {v}")
            a.min = v
        elif x == "--fail-above":
            v = need(i, x)
            i += 1
            try:
                a.fail_above = float(v)
            except ValueError:
                a.fail_above = math.nan
            if math.isnan(a.fail_above):
                raise UsageError("--fail-above ждёт число")
        elif x == "--config":
            a.config = need(i, x)
            i += 1
        elif x == "--system":
            v = need(i, x)
            i += 1
            if v not in SYSTEMS:
                raise UsageError(f"неизвестная --system: {v}; есть: {', '.join(SYSTEMS)}")
            a.system = next(s for s in SYSTEMS if s == v)
        elif x == "--answers":
            a.answers.append(need(i, x))
            i += 1
        elif x == "--marked" and a.cmd == "judge":
            a.judge_marked = need(i, x)
            i += 1
        elif x == "--doc":
            a.reports.append(Report(need(i, x)))
            i += 1
        elif x in ("--marked", "--share"):
            v = need(i, x)
            i += 1
            last = a.reports[-1] if a.reports else None
            if last is None or last.marked is not None or last.share is not None:
                raise UsageError(f"{x} относится к предыдущему --doc: сначала --doc ФАЙЛ")
            if x == "--marked":
                last.marked = v
            else:
                try:
                    share = float(v.replace(",", ".", 1).removesuffix("%"))
                except ValueError:
                    share = math.nan
                if not math.isfinite(share) or share < 0 or share > 100:
                    raise UsageError(f"--share ждёт долю 0–100: {v}")
                last.share = share
        elif x.startswith("-") and x != "-":
            raise UsageError(f"неизвестный параметр: {x}")
        elif not a.cmd:
            a.cmd = x
        else:
            a.files.append(x)
        i += 1
    return a


def read(file: str | None) -> str:
    from_stdin = file is None or file == "-"
    try:
        data = sys.stdin.buffer.read() if from_stdin else Path(file).read_bytes()
        # Как TextDecoder в JS: строгий UTF-8, BOM в начале отбрасывается.
        return data.decode("utf-8-sig")
    except (OSError, UnicodeDecodeError) as e:
        reason = e.strerror if isinstance(e, OSError) and e.strerror else str(e)
        raise UsageError(f"не удалось прочитать {'stdin' if from_stdin else file}: {reason}") from None


RANK: dict[str, int] = {"P0": 0, "P1": 1, "P2": 2}


def use_color() -> bool:
    """Цвет в терминале: NO_COLOR выключает, FORCE_COLOR включает, иначе только для TTY."""
    if os.environ.get("NO_COLOR"):
        return False
    force = os.environ.get("FORCE_COLOR")
    if force and force != "0":
        return True
    return sys.stdout.isatty()


def screen_width() -> int:
    """Ширина вывода: терминал, затем COLUMNS, иначе 100."""
    cols = 0
    if sys.stdout.isatty():
        try:
            cols = os.get_terminal_size(sys.stdout.fileno()).columns
        except OSError:
            cols = 0
    if not cols:
        try:
            cols = int(os.environ.get("COLUMNS", ""))
        except ValueError:
            cols = 0
    return max(60, min(cols or 100, 160))


class _Console(Console):
    def on_broken_pipe(self) -> None:
        # rich в этом случае сам выходит с кодом 1. Обрыв канала не ошибка: его ловит run().
        raise BrokenPipeError


def make_console() -> Console:
    color = use_color()
    return _Console(
        width=screen_width(),
        force_terminal=color,
        color_system="standard" if color else None,
        highlight=False,
        emoji=False,
        markup=False,
        soft_wrap=False,
    )


# main() собирает консоль заново при каждом вызове: цвет и ширина берутся из окружения
# в момент запуска команды, а не в момент импорта модуля.
console = make_console()

SEV_STYLE: dict[str, str] = {"P0": "red", "P1": "yellow", "P2": "dim"}

# Заголовок и линейка под ним, без рамок: как прежний вывод в терминал.
RULE_BOX = box.Box("    \n    \n ─  \n    \n    \n    \n    \n    \n")

# Отступ таблиц и пояснений от края; промежуток между колонками таблицы.
INDENT = 2
GAP = 2


def score_style(score: float) -> str:
    return "green" if score < 15 else "yellow" if score < 40 else "red"


def ru(n: float, digits: int = 2) -> str:
    """Число по-русски: десятичная запятая, не больше двух знаков после неё."""
    return num_str(float(to_fixed(n, digits))).replace(".", ",")


def mean_words(n: float) -> str:
    """«2,5 слова» при дробном числе, «6 слов» при целом."""
    s = ru(n, 1)
    return f"{s} слова" if "," in s else plural(int(s), "слово", "слова", "слов")


def out(r: RenderableType = "") -> None:
    console.print(r)


def heading(text: str) -> None:
    """Заголовок блока, обычно имя файла: целиком в одну строку, даже если он шире экрана."""
    console.print(Text(text, style="bold"), soft_wrap=True)


def indented(r: RenderableType) -> Padding:
    return Padding(r, (0, 0, 0, INDENT), expand=False)


def warn(text: str, style: str) -> None:
    """Предупреждение с отступом; перенос тоже с отступом."""
    out(indented(Text(text, style=style)))


def note(text: str) -> None:
    """Пояснение под таблицей: приглушённое, с переносом по ширине экрана."""
    out(indented(Text(text, style="dim")))


def facts(pairs: list[tuple[str, str | Text]]) -> None:
    """Блок «ключ — значение» с выровненными ключами."""
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", no_wrap=True)
    grid.add_column()
    for k, v in pairs:
        grid.add_row(k, v)
    out(indented(grid))


@dataclass
class Col:
    title: str
    justify: JustifyMethod = "left"
    # Колонку можно сужать, если таблица не влезает; не уже min, если экран позволяет (см. layout).
    flex: bool = False
    min: int | None = None
    # Переносить по словам; иначе обрезать с «…».
    wrap: bool = False


def layout(columns: list[Col], rows: list[list[str | Text]], width: int) -> list[int]:
    """Ширины колонок: естественные, затем сужение самых широких flex-колонок до влезания.

    Сначала колонки сужаются не уже своего min. Если таблица и так не влезает в экран,
    сужение идёт дальше без min: лучше узкая колонка, чем строка шире экрана.
    """
    widths = [max([cell_len(c.title), *(cell_len(str(r[k])) for r in rows)]) for k, c in enumerate(columns)]
    room = width - INDENT - GAP * (len(columns) - 1)
    for floor in (lambda c: c.min or 10, lambda _c: 1):
        while sum(widths) > room:
            shrinkable = [k for k, c in enumerate(columns) if c.flex and widths[k] > floor(c)]
            if not shrinkable:
                break
            widths[max(shrinkable, key=lambda k: widths[k])] -= 1
    return widths


def table(columns: list[Col], rows: list[list[str | Text]]) -> None:
    # Ширины считает layout(); rich только рисует. Поле справа у каждой колонки, кроме последней,
    # плюс разделитель из RULE_BOX дают промежуток GAP.
    t = Table(
        box=RULE_BOX,
        show_edge=False,
        pad_edge=False,
        padding=(0, GAP - 1, 0, 0),
        header_style="dim",
        border_style="dim",
    )
    widths = layout(columns, rows, console.width)
    for c, w in zip(columns, widths, strict=True):
        t.add_column(c.title, justify=c.justify, width=w, no_wrap=not c.wrap, overflow="fold" if c.wrap else "ellipsis")
    for r in rows:
        t.add_row(*(cell if c.wrap else truncate(cell, w) for c, w, cell in zip(columns, widths, r, strict=True)))
    out(indented(t))


def truncate(cell: str | Text, width: int) -> str | Text:
    """Обрезка с «…» без пробела перед многоточием."""
    if isinstance(cell, Text) or cell_len(cell) <= width:
        return cell
    return f"{set_cell_size(cell, max(0, width - 1)).rstrip()}…"


LEGEND = "P0 — исправить сразу · P1 — исправить до публикации · P2 — шлифовка · стиль — канцелярит и жаргон, не довод об авторстве"


def dump(value: Record | list[dict[str, Any]] | dict[str, Any], pretty: bool = True) -> str:
    data = value.to_dict() if isinstance(value, Record) else value
    if pretty:
        return json.dumps(data, ensure_ascii=False, indent=2)
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def jsonl(a: Args, run: Callable[[str], dict[str, Any]]) -> int:
    """Пакетный режим: каждая строка входа — JSON с полем text, на выходе по строке результата."""
    results: list[str] = []
    for n, line in enumerate(read(a.files[0] if a.files else None).split("\n")):
        if not trim(line):
            continue
        try:
            doc = json.loads(line)
        except json.JSONDecodeError:
            raise UsageError(f"строка {n + 1}: не JSON") from None
        text = doc.get("text") if isinstance(doc, dict) else None
        if not isinstance(text, str):
            raise UsageError(f"строка {n + 1}: нет строкового поля text")
        results.append(dump(run(text), pretty=False))
    if results:
        sys.stdout.write("\n".join(results) + "\n")
    return 0


# Вероятность ближе к порогу, чем на столько, модель не решает: помечается «близко к порогу».
NEAR_THRESHOLD = 0.15


@dataclass(frozen=True, slots=True)
class Signal:
    """Вероятность ИИ от необязательной модели и сама модель."""

    loaded: models.Loaded
    # None: текст для модели не на русском, модель его не оценивала.
    probability: float | None
    # Какую часть текста модель прочитала; None, если она его не читала.
    coverage: models.Coverage | None
    # Доля кириллицы среди кириллических и латинских букв текста для модели; None, если букв нет.
    cyrillic: float | None = None
    # Строки абзацев не на русском, которые модель не читала: (первая, последняя).
    foreign: tuple[tuple[int, int], ...] = ()

    @property
    def skipped(self) -> bool:
        return self.probability is None

    @property
    def ai(self) -> bool:
        return self.probability is not None and self.probability >= self.loaded.threshold

    @property
    def near_threshold(self) -> bool:
        return self.probability is not None and abs(self.probability - self.loaded.threshold) < NEAR_THRESHOLD

    @property
    def truncated(self) -> bool:
        return self.coverage is not None and self.coverage.truncated


def model_signal(a: Args, text: str, result: AnalysisResult | None = None) -> Signal | None:
    """Вероятность от необязательной модели, если она установлена и не выключена --no-model.

    Модель, выбранная через --model, обязана быть установлена; без --model берётся первая установленная.
    """
    if a.no_model:
        return None
    loaded = models.preferred(a.model_name)
    return None if loaded is None else signal(loaded, text, result)


def signal(loaded: models.Loaded, text: str, result: AnalysisResult | None = None) -> Signal:
    """Вероятность по тексту целиком. Модель читает не файл, а models.visible(text): без YAML-шапки,
    кода, служебных частей статьи и абзацев не на русском. Если в том, что осталось, меньше половины
    кириллицы или нет букв, вероятности нет.

    result — ответ детектора на исходник; на visible(text) он даёт те же находки и статистику
    (шапку, код и служебные части детектор маскирует так же), поэтому LightGBM его переиспользует.
    Абзацы не на русском детектор не вырезает: если они есть, LightGBM считает признаки заново."""
    outdated_hint(loaded.model)
    clean = models.visible(text)
    foreign = tuple(models.foreign_lines(text))
    # Доля по тексту до вырезания абзацев не на русском: её видит пользователь. Решает доля в том, что
    # осталось модели: русские абзацы английского документа модель оценивает.
    share = cyrillic_share(model_text(text))
    if not models.is_russian(clean):
        return Signal(loaded, None, None, share, foreign)
    reuse = None if foreign else result
    return Signal(loaded, loaded.probability(clean, reuse), loaded.coverage(clean), share, foreign)


# Модели, о старой версии которых уже сказано в этом запуске.
told_outdated: set[str] = set()


def outdated_hint(model: models.Model) -> None:
    """В stderr, один раз за запуск: скачанная модель не та, с которой замерено её качество."""
    if model.name in told_outdated or not models.outdated(model):
        return
    told_outdated.add(model.name)
    sys.stderr.write(f"aiw-ru: {OUTDATED.format(name=model.name)}\n")
    sys.stderr.flush()


OUTDATED = (
    "скачана не та версия модели {name}, с которой замерено её качество в каталоге aiw-ru; "
    "обновить: aiw-ru models install {name}"
)


def grouped(n: int) -> str:
    """Число с разрядами через неразрывный пробел от 10 000, как в plural."""
    return f"{n:,}".replace(",", "\u00a0") if abs(n) >= 10000 else str(n)


def coverage_json(c: models.Coverage) -> dict[str, Any]:
    return {
        "truncated": c.truncated,
        "words": c.words,
        "totalWords": c.total_words,
        "chars": c.chars,
        "totalChars": c.total_chars,
        "lines": c.lines,
        "totalLines": c.total_lines,
        "tokens": c.tokens,
        "totalTokens": c.total_tokens,
        "window": c.window,
    }


def read_words(c: models.Coverage) -> str:
    return f"{plural(c.words, 'слово', 'слова', 'слов')} из {grouped(c.total_words)}"


def read_fact(c: models.Coverage) -> tuple[str, str]:
    """«Прочитано: 290 слов из 6200, строки 1–14 из 380: окно модели 512 токенов»."""
    lines = f"строка {c.lines}" if c.lines == 1 else f"строки 1–{c.lines}"
    window = plural(c.window or 0, "токен", "токена", "токенов")
    return ("Прочитано", f"{read_words(c)}, {lines} из {grouped(c.total_lines)}: окно модели {window}")


def line_ranges(ranges: Sequence[tuple[int, int]]) -> str:
    """«строка 11», «строки 11, 40–45»."""
    parts = [str(a) if a == b else f"{a}–{b}" for a, b in ranges]
    one = len(ranges) == 1 and ranges[0][0] == ranges[0][1]
    return f"{'строка' if one else 'строки'} {', '.join(parts)}"


def foreign_fact(sig: Signal) -> list[tuple[str, str]]:
    """«Не на русском: строки 11, 40–45, модель их не читала»."""
    if not sig.foreign or sig.skipped:
        return []
    it = "её" if len(sig.foreign) == 1 and sig.foreign[0][0] == sig.foreign[0][1] else "их"
    return [("Не на русском", f"{line_ranges(sig.foreign)}: модель {it} не читала")]


def read_facts(sig: Signal | None) -> list[tuple[str, str]]:
    if sig is None:
        return []
    read = [read_fact(sig.coverage)] if sig.coverage is not None and sig.truncated else []
    return read + foreign_fact(sig)


def fragments_hint(sig: Signal | None, file: str | None) -> None:
    """Под сводкой scan и antiplagiat: модель видела только начало, весь текст проверит classify."""
    if sig is not None and sig.truncated:
        target = file if file and file != "-" else "<файл>"
        note(f"Модель прочитала только начало текста. Весь текст по фрагментам: aiw-ru classify {target}")


# Если по замеру первых фрагментов ждать дольше, stderr говорит, сколько примерно.
ETA_AFTER = 5.0


def about(seconds: float) -> str:
    """«4 с», «45 с», «2 мин»: оценка времени без лишней точности."""
    if seconds < 10:
        return f"{max(1, round(seconds))} с"
    if seconds < 90:
        return f"{5 * round(seconds / 5)} с"
    return f"{round(seconds / 60)} мин"


def eta(loaded: models.Loaded) -> models.Progress:
    """Оценка времени после первых фрагментов, один раз и только для долгой проверки."""
    told = False

    def progress(done: int, total: int, elapsed: float) -> None:
        nonlocal told
        # Первый фрагмент может идти дольше остальных: оценка по трём.
        if told or done < min(3, total - 1) or done >= total:
            return
        if elapsed / done * (total - done) >= ETA_AFTER:
            told = True
            sys.stderr.write(
                f"aiw-ru classify: {plural(total, 'фрагмент', 'фрагмента', 'фрагментов')}, "
                f"{loaded.model.title} проверит их примерно за {about(elapsed / done * total)}\n"
            )
            sys.stderr.flush()

    return progress


def merged_lines(chunks: list[models.Chunk]) -> list[tuple[int, int]]:
    """Строки фрагментов, слитые там, где фрагменты заходят друг на друга или идут подряд."""
    out: list[tuple[int, int]] = []
    for c in sorted(chunks, key=lambda c: c.line):
        if out and c.line <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], c.end_line))
        else:
            out.append((c.line, c.end_line))
    return out


def word_share(chunks: list[models.Chunk], scan: models.ChunkScan) -> float:
    """Доля слов текста для модели, попавших хотя бы в один из фрагментов; перекрытия считаются один раз.

    Делится на scan.words: слова шапки, кода, служебных частей и фрагментов не на русском в знаменатель
    не входят."""
    spans: list[tuple[int, int]] = []
    for c in sorted(chunks, key=lambda c: c.start):
        if spans and c.start <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], c.end))
        else:
            spans.append((c.start, c.end))
    return sum(len(words(scan.text[a:b])) for a, b in spans) / scan.words if scan.words else 0.0


def near(p: float | None, threshold: float) -> bool:
    """Вероятность ближе к порогу, чем NEAR_THRESHOLD."""
    return p is not None and abs(p - threshold) < NEAR_THRESHOLD


def fragments_json(scan: models.ChunkScan, loaded: models.Loaded) -> dict[str, Any]:
    """count — фрагменты, которые оценила модель, skipped — не на русском, total — все в тексте;
    в items и те и другие, подряд. aboveThreshold, aiWordShare и aiLines считаются по оценённым."""
    t = loaded.threshold
    above = [c for c in scan.chunks if c.above(t)]
    return {
        "count": len(scan.rated),
        "skipped": scan.skipped,
        "total": scan.total,
        "overlap": scan.overlap,
        "seconds": round(scan.seconds, 3),
        "aboveThreshold": len(above),
        "aiWordShare": round(word_share(above, scan), 4),
        "aiLines": [list(r) for r in merged_lines(above)],
        "items": [
            {
                "line": c.line,
                "endLine": c.end_line,
                "start": c.start,
                "end": c.end,
                "words": c.words,
                "language": "other" if c.skipped else "ru",
                "skipped": c.skipped,
                "probability": None if c.probability is None else round(c.probability, 4),
                "ai": None if c.skipped else c.above(t),
                "nearThreshold": None if c.skipped else near(c.probability, t),
            }
            for c in scan.chunks
        ],
    }


def not_russian_note(n: int) -> str:
    """«2 фрагмента не на русском, модель их не оценивала»."""
    it = "его" if n % 10 == 1 and n % 100 != 11 else "их"
    return f"{plural(n, 'фрагмент', 'фрагмента', 'фрагментов')} не на русском, модель {it} не оценивала."


def show_fragments(scan: models.ChunkScan, loaded: models.Loaded) -> None:
    """Таблица фрагментов длинного текста, итог и сколько заняла проверка."""
    t = loaded.threshold

    def style(p: float) -> str:
        return "yellow" if near(p, t) else "red" if p >= t else "green"

    table(
        [Col("Строки"), Col("Слов", justify="right"), Col("ИИ", justify="right")],
        [
            [
                Text.styled(str(c.line) if c.line == c.end_line else f"{c.line}–{c.end_line}", "dim"),
                str(c.words),
                Text.styled("не на русском", "dim")
                if c.probability is None
                else Text.styled(f"{js_round(c.probability * 100)} %", style(c.probability)),
            ]
            for c in scan.chunks
        ],
    )
    out()
    rated = scan.rated
    above = [c for c in rated if c.above(t)]
    if above:
        ranges = merged_lines(above)
        where = (
            f"строка {ranges[0][0]}"
            if len(ranges) == 1 and ranges[0][0] == ranges[0][1]
            else "строки " + ", ".join(str(a) if a == b else f"{a}–{b}" for a, b in ranges)
        )
        summary = (
            f"Выше порога {plural(len(above), 'фрагмент', 'фрагмента', 'фрагментов')} из {len(rated)}, "
            f"в них {js_round(word_share(above, scan) * 100)} % слов: {where}."
        )
        out(indented(Text.styled(summary, "red")))
    elif rated:
        out(indented(Text.styled("Все фрагменты ниже порога.", "green")))
    else:
        out(indented(Text.styled("Модель не оценила ни одного фрагмента.", "yellow")))
    if scan.skipped:
        note(not_russian_note(scan.skipped))
    out()
    if len(above) == 1 and len(rated) > 1:
        note(
            "Один фрагмент выше порога в длинном тексте — слабый признак: так бывает и у текстов людей. "
            "Сверьте эти строки с находками scan; как читать итог и перепроверить фрагмент — aiw-ru models guide."
        )
    elif len(above) > 1:
        note(
            "Отмеченные строки шире самого ИИ-текста: фрагмент захватывает и соседний текст. "
            "Как читать итог — aiw-ru models guide."
        )
    note(
        f"Проверено {plural(len(rated), 'фрагмент', 'фрагмента', 'фрагментов')} за "
        f"{ru(max(scan.seconds, 0.1), 1)} с: каждый в окно модели, соседние заходят друг на друга "
        f"на {js_round(scan.overlap * 100)} %."
    )
    if len(scan.chunks) < scan.total:
        note(
            f"Предел --max-fragments: проверено {len(scan.chunks)} из {scan.total} фрагментов, до строки "
            f"{scan.chunks[-1].end_line}; все — --max-fragments 0."
        )


def signal_json(sig: Signal) -> dict[str, Any]:
    """Сигнал для JSON. Если текст не на русском, language равен other, skipped равен true, а
    probability, ai, nearThreshold и read пусты (null): модель его не оценивала."""
    m = sig.loaded.model
    return {
        "name": m.name,
        "language": "other" if sig.skipped else "ru",
        "skipped": sig.skipped,
        "cyrillicShare": None if sig.cyrillic is None else round(sig.cyrillic, 4),
        "probability": None if sig.probability is None else round(sig.probability, 4),
        "threshold": round(sig.loaded.threshold, 6),
        "ai": None if sig.skipped else sig.ai,
        "nearThreshold": None if sig.skipped else sig.near_threshold,
        "repo": m.repo,
        "outdated": models.outdated(m),
        "read": None if sig.coverage is None else coverage_json(sig.coverage),
        "notRussianLines": [list(r) for r in sig.foreign],
    }


def with_signal(data: dict[str, Any], sig: Signal | None) -> dict[str, Any]:
    # Не "model": это поле уже занято в отчёте antiplagiat (default или calibrated).
    if sig is not None:
        data["classifier"] = signal_json(sig)
    return data


def skip_fact(sigs: list[Signal]) -> tuple[str, Text]:
    """Вместо вероятности: модели не читают текст не на русском. Все модели получают один и тот же текст,
    поэтому у них один ответ; sigs — один сигнал или все модели `classify --all`."""
    share = sigs[0].cyrillic
    one = len(sigs) == 1
    label, what = ("Модель", "не применима") if one else ("Модели", "не применимы")
    if share is None:
        why = f"в нём нет букв (шапка, код и служебные части статьи {'ей' if one else 'им'} не показываются)"
    else:
        why = f"кириллицы {js_round(share * 100)} % букв, а {'модель обучена' if one else 'модели обучены'} на русском"
    title = f" ({sigs[0].loaded.model.title})" if one else ""
    return (label, Text.styled(f"{what} к этому тексту: {why}{title}", "yellow"))


def signal_fact(sig: Signal, label: str | None = None) -> tuple[str, Text]:
    """«Вероятность ИИ: 97 % (трансформер, порог 50 %)»; с подписью-моделью имя в скобках не повторяется.
    Для текста не на русском вместо вероятности: «Модель: не применима к этому тексту …»."""
    if sig.skipped or sig.probability is None:
        return skip_fact([sig])
    t = js_round(sig.loaded.threshold * 100)
    source = f"порог {t} %" if label else f"{sig.loaded.model.title}, порог {t} %"
    if label and sig.truncated and sig.coverage is not None:
        source += f", прочитано {read_words(sig.coverage)}"
    label = label or "Вероятность ИИ"
    text = f"{js_round(sig.probability * 100)} % ({source}"
    if sig.near_threshold:
        return (label, Text.styled(text + ", близко к порогу: модель не уверена)", "yellow"))
    return (label, Text.styled(text + ")", "red" if sig.ai else "green"))


def agreement(sigs: list[Signal]) -> Text:
    """Согласны ли модели: одна строка под вероятностями `classify --all`."""
    ai = sum(s.ai for s in sigs)
    if ai == len(sigs):
        return Text.styled("Все модели выше порога. Это сигнал, а не вывод: сверьте с находками и текстом.", "red")
    if ai == 0:
        return Text.styled("Все модели ниже порога.", "green")
    return Text.styled(
        f"Модели расходятся: {ai} из {len(sigs)} выше порога. Вероятность здесь ничего не решает, смотрите на находки.",
        "yellow",
    )


def catalog_refs(r: AnalysisResult) -> dict[str, list[str]]:
    """Файлы каталога и их разделы для найденных примет, в порядке уровня серьёзности."""
    where = skills.catalog()
    refs: dict[str, list[str]] = {}
    for i in sorted(r.issues, key=lambda i: (RANK[i.severity], i.index)):
        section = TYPE_TO_SECTION.get(i.type)
        path = where.get(section) if section else None
        if section and path and section not in refs.setdefault(path, []):
            refs[path].append(section)
    return refs


def scan_json(r: AnalysisResult) -> dict[str, Any]:
    """Результат scan для JSON: с файлами каталога, где описаны найденные приметы."""
    return {**r.to_dict(), "catalog": [{"file": f, "sections": s} for f, s in catalog_refs(r).items()]}


def cmd_scan(a: Args) -> int:
    mode = a.context or "general"
    if a.jsonl:

        def one(text: str) -> dict[str, Any]:
            r = analyze(text, mode)
            return with_signal(scan_json(r), model_signal(a, text, r))

        return jsonl(a, one)
    files = a.files or ["-"]
    texts = [(f, read(f)) for f in files]
    results = [(f, analyze(t, mode)) for f, t in texts]
    signals = [model_signal(a, t, r) for (_, t), (_, r) in zip(texts, results, strict=True)]
    if a.json:
        docs = [with_signal({"file": f, **scan_json(r)}, p) for (f, r), p in zip(results, signals, strict=True)]
        sys.stdout.write(dump(docs[0] if len(docs) == 1 else docs) + "\n")
    fail = False
    for n, ((f, r), p) in enumerate(zip(results, signals, strict=True)):
        if a.fail_above is not None and r.score > a.fail_above:
            fail = True
        if a.json:
            continue
        if n > 0:
            out()
        shown = sorted(
            (i for i in r.issues if RANK[i.severity] <= RANK[a.min]),
            key=lambda i: (RANK[i.severity], int(i.style_only), i.index),
        )
        s = r.stats
        heading("stdin" if f == "-" else f)
        facts(
            [
                ("Оценка", Text.styled(f"{r.score}/100, {r.label}", score_style(r.score))),
                (
                    "Текст",
                    (
                        f"{plural(s.words, 'слово', 'слова', 'слов')}, "
                        f"{plural(s.sentences, 'предложение', 'предложения', 'предложений')}, режим {s.context_mode}"
                    ),
                ),
                (
                    "Ритм",
                    (
                        f"в среднем {mean_words(s.mean_sentence_length)} в предложении, "
                        f"разброс {ru(s.sentence_length_cv)}, MATTR {ru(s.mattr)}"
                    ),
                ),
                *([signal_fact(p)] if p is not None else []),
                *read_facts(p),
            ]
        )
        fragments_hint(p, f)
        if r.suspicious:
            warn("Документ выглядит подозрительным: невидимые символы или подмена букв.", "red")
        out()
        if not shown:
            out(Text("  Находок выбранного уровня нет." if r.issues else "  Примет не найдено.", style="green"))
            continue
        table(
            [
                Col("Уровень"),
                Col("Где"),
                Col("Примета", flex=True, wrap=True, min=14),
                Col("Фрагмент", flex=True, wrap=True, min=14),
                Col("Как исправить", flex=True, wrap=True, min=20),
            ],
            [
                [
                    Text.styled("стиль", "dim") if i.style_only else Text.styled(i.severity, SEV_STYLE[i.severity]),
                    Text.styled(f"{i.line}:{i.column}", "dim"),
                    TYPE_LABELS.get(i.type, i.type),
                    f"«{i.text}»",
                    i.hint,
                ]
                for i in shown
            ],
        )
        out()
        note(LEGEND)
        refs = catalog_refs(r)
        if refs:
            note(
                "Условия и исключения в каталоге скилла: " + "; ".join(f"{f} ({', '.join(s)})" for f, s in refs.items())
            )
    return 1 if fail else 0


def config_path(a: Args) -> str | None:
    if a.config:
        return a.config
    if os.environ.get("AIW_RU_CONFIG"):
        return os.environ["AIW_RU_CONFIG"]
    local = ".aiw-ru.json"
    if os.path.exists(local):
        return local
    glob = Path.home() / ".config" / "aiw-ru.json"
    return str(glob) if glob.exists() else None


def load_calibration(path: str | None) -> CalibrationFile | None:
    if not path or not os.path.exists(path):
        return None
    try:
        return CalibrationFile.model_validate_json(Path(path).read_bytes())
    except ValidationError as e:
        err = e.errors()[0]
        where = ".".join(map(str, err["loc"])) or "файл"
        raise UsageError(f"не удалось прочитать калибровку {path}: неверный формат ({where}: {err['msg']})") from None
    except OSError as e:
        raise UsageError(f"не удалось прочитать калибровку {path}: {e}") from None


def cmd_antiplagiat(a: Args) -> int:
    file = load_calibration(config_path(a))
    cal = file.get(a.system) if file else None
    model: Model | None = cal.model if cal else None
    mode = a.context or "academic"
    if a.jsonl:
        return jsonl(
            a, lambda text: with_signal(antiplagiat(text, mode, model, a.system).to_dict(), model_signal(a, text))
        )
    text = read(a.files[0] if a.files else None)
    report = antiplagiat(text, mode, model, a.system)
    p = model_signal(a, text)
    if a.json:
        sys.stdout.write(dump(with_signal(report.to_dict(), p)) + "\n")
        return 0
    share = report.ai_share
    share_style = "green" if share < 20 else "yellow" if share < 50 else "red"
    ai = sum(1 for f in report.fragments if f.ai)
    heading(a.files[0] if a.files and a.files[0] != "-" else "stdin")
    facts(
        [
            ("Доля ИИ-текста", Text.styled(f"{ru(share)} %", share_style)),
            ("Фрагменты", f"{len(report.fragments)}, похожи на ИИ: {ai} (порог {ru(report.threshold)})"),
            (
                "Модель",
                f"откалибрована по вашим отчётам, образцов: {len(cal.samples) if cal else 0}"
                if report.model == "calibrated"
                else "по умолчанию, без калибровки",
            ),
            *([signal_fact(p)] if p is not None else []),
            *read_facts(p),
        ]
    )
    fragments_hint(p, a.files[0] if a.files else None)
    if report.suspicious:
        warn(
            f"Подозрительный документ: {'; '.join(report.suspicious_reasons)}. "
            "Системы проверки такое замечают, удалите эти символы.",
            "red",
        )
    out()
    table(
        [
            Col("Строки"),
            Col("Слов", justify="right"),
            Col("ИИ", justify="right"),
            Col("Начало фрагмента", flex=True, min=24),
            Col("Почему похоже на ИИ", flex=True, wrap=True, min=24),
        ],
        [
            [
                Text.styled(str(f.line) if f.line == f.end_line else f"{f.line}–{f.end_line}", "dim"),
                str(f.words),
                Text.styled(f"{js_round(f.probability * 100)} %", "red" if f.ai else "green"),
                f"{f.preview}…",
                ", ".join(f.reasons) if f.ai else "",
            ]
            for f in report.fragments
        ],
    )
    out()
    genitive = SYSTEMS[a.system][1]
    flag = "" if a.system == "antiplagiat" else f" --system {a.system}"
    note(
        f"Это приближение: классификатор {genitive} закрыт. На отшлифованном научном тексте оценка может быть "
        "нулевой там, где система находит треть ИИ-текста; если есть отчёт с подсветкой, правьте по нему. "
        f"Точнее станет после калибровки по вашим отчётам (aiw-ru calibrate{flag})."
    )
    return 0


def cmd_validate(a: Args) -> int:
    if len(a.files) != 2:
        raise UsageError("validate ждёт два файла: исходный и исправленный")
    r = validate(read(a.files[0]), read(a.files[1]), a.context or "general")
    if a.json:
        sys.stdout.write(dump(r) + "\n")
    else:
        rows: list[tuple[str, str | Text]] = [
            (
                "Сохранность",
                Text.styled("в порядке", "green") if r.ok else Text.styled(f"нарушена ({len(r.violations)})", "red"),
            )
        ]
        if r.warnings:
            rows.append(("Проверить", Text.styled(str(len(r.warnings)), "yellow")))
        rows.append(("Находки", f"{r.issues_before} → {r.issues_after}"))
        facts(rows)
        if not r.ok:
            out()
            table(
                [Col("Что"), Col("Подробности", flex=True, wrap=True, min=20)],
                [[Text.styled(v.kind, "yellow"), v.detail] for v in r.violations],
            )
        if r.warnings:
            out()
            table(
                [Col("Проверить"), Col("Подробности", flex=True, wrap=True, min=20)],
                [[Text.styled(w.kind, "cyan"), w.detail] for w in r.warnings],
            )
            note("Код выхода они не меняют: это может быть законной правкой, сверьте с исходником.")
    return 0 if r.ok else 1


MARKED_SPLIT_RE = jsre("\\n\\s*(?:---+\\s*)?\\n")


def split_marked(text: str) -> list[str]:
    """Куски из отчёта: тепловая карта в JSON (красные фрагменты) или текст через пустую строку."""
    heat = marked_from_heatmap(text)
    if heat is not None:
        return heat
    return [t for t in (trim(s) for s in MARKED_SPLIT_RE.split(text)) if t]


def cmd_calibrate(a: Args) -> int:
    if not a.reports or any(r.marked is None and r.share is None for r in a.reports):
        raise UsageError("calibrate ждёт для каждого --doc ФАЙЛ разметку: --marked ФАЙЛ или --share ДОЛЯ")
    path = a.config or ".aiw-ru.json"
    file = load_calibration(path if os.path.exists(path) else None) or empty_calibration_file()
    prev = file.get(a.system)
    samples = list(prev.samples) if prev else []
    documents = list(prev.documents) if prev else []
    rows: list[list[str | Text]] = []
    warnings: list[str] = []
    for r in a.reports:
        doc = read(r.doc)
        if r.share is not None:
            d = document_sample(doc, r.share)
            documents.append(d)
            rows.append([r.doc, f"доля {ru(r.share)} %", str(len(d.fragments)), "—"])
            continue
        marked = split_marked(read(r.marked))
        labeled = label_fragments(doc, marked)
        pos = sum(1 for s in labeled if s.y == 1)
        rows.append([r.doc, r.marked or "", str(len(labeled)), str(pos)])
        if marked and pos == 0:
            warnings.append(
                f"{r.doc}: ни один кусок из {r.marked} не найден в документе. Проверьте, что текст скопирован из "
                "отчёта без правок и документ той же версии."
            )
        samples.extend(labeled)
    start = prev.model if prev else DEFAULT_MODEL
    model = fit_share(fit(samples, start), documents)
    data = file.put(a.system, Calibration(model=model, samples=samples, documents=documents))
    Path(path).write_text(dump(data) + "\n", encoding="utf-8")

    heading("Отчёты")
    table(
        [
            Col("Документ", flex=True, min=16),
            Col("Разметка", flex=True, min=16),
            Col("Фрагментов", justify="right"),
            Col("Подсвечено ИИ", justify="right"),
        ],
        rows,
    )
    for w in warnings:
        warn(w, "yellow")
    out()

    def pct(x: float) -> str:
        return "н/д" if math.isnan(x) else f"{ru(x * 100, 1)} %"

    def pp(x: float) -> str:
        return "н/д" if math.isnan(x) else f"{ru(x, 1)} п.п."

    pos = sum(1 for s in samples if s.y == 1)
    heading("Калибровка")
    facts(
        [
            ("Система", SYSTEMS[a.system][0]),
            ("Фрагменты с разметкой", f"{len(samples)}, из них ИИ: {pos}"),
            ("Документы с долей", str(len(documents))),
            (
                "Точность по фрагментам",
                "н/д (нужны и ИИ-, и человеческие фрагменты)"
                if math.isnan(balanced_accuracy(model, samples))
                else f"{pct(balanced_accuracy(DEFAULT_MODEL, samples))} → {pct(balanced_accuracy(model, samples))}",
            ),
            ("Ошибка доли ИИ", f"{pp(share_error(DEFAULT_MODEL, documents))} → {pp(share_error(model, documents))}"),
            ("Порог", ru(model.threshold)),
            ("Веса", "; ".join(f"{FEATURE_NAMES[k]} {ru(w)}" for k, w in enumerate(model.weights))),
            ("Сохранено", path),
        ]
    )
    out()
    note(
        "Точность и ошибка посчитаны на тех же отчётах, по которым шла калибровка, и поэтому завышены. Честная "
        "проверка: замер на отчёте, который в калибровку не входил. Файл содержит фрагменты вашего текста, не "
        "публикуйте его."
    )
    return 0


LEVEL_NAMES = ("none", "low", "medium", "high")


def cmd_judge(a: Args) -> int:
    if a.files and a.files[0] == "score":
        return cmd_judge_score(a)
    if a.answers or a.judge_marked:
        raise UsageError("--answers и --marked относятся к judge score")
    text = read(a.files[0] if a.files else None)
    mode = a.context or "academic"
    if a.json:
        data = {"prompt": judge_prompt(text, mode), "fragments": [f.to_dict() for f in judge_fragments(text, mode)]}
        sys.stdout.write(dump(data) + "\n")
    else:
        sys.stdout.write(judge_prompt(text, mode))
    return 0


def cmd_judge_score(a: Args) -> int:
    if len(a.files) != 2:
        raise UsageError("judge score ждёт файл документа: judge score ФАЙЛ --answers ОТВЕТ [--answers …]")
    if not a.answers:
        raise UsageError("judge score ждёт ответы судьи: --answers ФАЙЛ на каждый прогон")
    doc = read(a.files[1])
    marked = split_marked(read(a.judge_marked)) if a.judge_marked else None
    r = judge_score(doc, [read(x) for x in a.answers], marked, a.context or "academic")
    if a.json:
        sys.stdout.write(dump(r) + "\n")
        return 0

    def level(x: int | None) -> str:
        return "—" if x is None else LEVEL_NAMES[x]

    def mean_style(m: float | None) -> str:
        return "dim" if m is None else "red" if m >= 1.5 else "yellow" if m >= 0.5 else "green"

    heading(a.files[1])
    rows: list[tuple[str, str | Text]] = [("Прогонов", str(r.runs))]
    if r.agreement is not None:
        rows.append(("Совпадение прогонов", f"{ru(r.agreement * 100, 0)} % фрагментов"))
    rows.append(("Доля слов", f"medium и выше {ru(r.share_medium, 1)} %, low и выше {ru(r.share_low, 1)} %"))
    if r.auc is not None or r.detector_auc is not None:
        both = f"судья {'н/д' if r.auc is None else ru(r.auc)}, детектор {'н/д' if r.detector_auc is None else ru(r.detector_auc)}"
        rows.append(("ROC AUC по отчёту", both))
    facts(rows)
    if r.missing:
        warn(f"Нет ответа судьи для фрагментов: {', '.join(map(str, r.missing))}.", "yellow")
    out()
    table(
        [
            Col("№", justify="right"),
            Col("Строки"),
            Col("Слов", justify="right"),
            Col("Уровни"),
            Col("Среднее", justify="right"),
            *([Col("Отчёт", justify="right")] if r.fragments and r.fragments[0].marked is not None else []),
            Col("Начало фрагмента", flex=True, min=24),
        ],
        [
            [
                str(f.n),
                Text.styled(str(f.line) if f.line == f.end_line else f"{f.line}–{f.end_line}", "dim"),
                str(f.words),
                " ".join(level(x) for x in f.levels),
                Text.styled("—" if f.mean is None else ru(f.mean), mean_style(f.mean)),
                *([f"{js_round(f.marked * 100)} %"] if f.marked is not None else []),
                f"{f.preview}…",
            ]
            for f in r.fragments
        ],
    )
    out()
    note(
        "Судья расставляет фрагменты по подсвеченности примерно как система проверки, но процент системы не "
        "предсказывает и от прогона к прогону отвечает по-разному. Берите самые высокие средние как кандидатов "
        "на правку, а итог проверяйте самой системой."
    )
    return 0


def cmd_classify_all(a: Args) -> int:
    """Вероятности всех установленных моделей рядом: видно, согласны ли они."""
    if a.model_name:
        raise UsageError("--all и --model вместе не используются: --all берёт все установленные модели")
    loaded = [models.load(m) for m in models.MODELS.values() if models.available(m)]
    if not loaded:
        raise models.ModelError(
            f"ни одна модель не установлена: поставьте пакеты ({models.INSTALL_HINT}) и скачайте модели "
            "(aiw-ru models install)"
        )

    def one(text: str) -> list[Signal]:
        return [signal(m, text) for m in loaded]

    def doc(sigs: list[Signal]) -> dict[str, Any]:
        # agree пуст (null), если текст не на русском и модели его не оценивали.
        agree = None if sigs[0].skipped else len({s.ai for s in sigs}) == 1
        return {"models": [signal_json(s) for s in sigs], "agree": agree}

    if a.jsonl:
        return jsonl(a, lambda text: doc(one(text)))
    files = a.files or ["-"]
    results = [(f, one(read(f))) for f in files]
    if a.json:
        docs = [{"file": f, **doc(sigs)} for f, sigs in results]
        sys.stdout.write(dump(docs[0] if len(docs) == 1 else docs) + "\n")
        return 0
    for n, (f, sigs) in enumerate(results):
        if n > 0:
            out()
        heading("stdin" if f == "-" else f)
        if sigs[0].skipped:
            facts([skip_fact(sigs)])
            continue
        facts([*(signal_fact(s, s.loaded.model.title) for s in sigs), *foreign_fact(sigs[0])])
        out(indented(agreement(sigs)))
        if any(s.truncated for s in sigs):
            target = f if f != "-" else "<файл>"
            note(
                f"Модели прочитали только начало текста. Весь текст по фрагментам: aiw-ru classify --model ИМЯ {target}"
            )
    out()
    note(
        "Модели обучены на русской части корпуса LLMTrace и видели не все жанры; короткие тексты они различают хуже. "
        "Вероятность — сигнал, а не доказательство авторства."
    )
    return 0


def cmd_classify(a: Args) -> int:
    if a.all_models:
        return cmd_classify_all(a)
    loaded = models.preferred(a.model_name)
    if loaded is None:
        raise models.ModelError(
            f"ни одна модель не установлена: поставьте пакеты ({models.INSTALL_HINT}) и скачайте модели "
            "(aiw-ru models install)"
        )

    def one(text: str) -> tuple[Signal, models.ChunkScan | None]:
        sig = signal(loaded, text)
        # Текст не на русском модель не читает, поэтому и фрагментов у него нет.
        if sig.coverage is None or a.no_fragments or not models.needs_fragments(loaded, sig.coverage):
            return sig, None
        return sig, models.scan_chunks(loaded, text, limit=a.max_fragments, progress=eta(loaded))

    def doc(sig: Signal, scan: models.ChunkScan | None) -> dict[str, Any]:
        return {**signal_json(sig), **({"fragments": fragments_json(scan, loaded)} if scan else {})}

    if a.jsonl:
        return jsonl(a, lambda text: doc(*one(text)))
    files = a.files or ["-"]
    results = [(f, *one(read(f))) for f in files]
    if a.json:
        docs = [{"file": f, **doc(sig, scan)} for f, sig, scan in results]
        sys.stdout.write(dump(docs[0] if len(docs) == 1 else docs) + "\n")
        return 0
    for n, (f, sig, scan) in enumerate(results):
        if n > 0:
            out()
        heading("stdin" if f == "-" else f)
        facts([signal_fact(sig), *read_facts(sig)])
        if scan is not None:
            out()
            show_fragments(scan, loaded)
    out()
    note(
        "Модель обучена на русской части корпуса LLMTrace и видела не все жанры. Вероятность — сигнал, а не "
        f"доказательство авторства. Замеры по жанрам: https://huggingface.co/{loaded.model.repo}"
    )
    return 0


def model_state(model: models.Model) -> dict[str, Any]:
    st = models.status(model)
    error = None
    try:
        models.load(model)
    except models.ModelError as e:
        error = str(e)
    return {
        "name": model.name,
        "repo": model.repo,
        "dependencies": st.dependencies,
        "missing": models.missing_dependencies(model),
        "path": str(st.path) if st.path else None,
        "outdated": models.outdated(model),
        "ready": error is None,
        "error": error,
    }


GITHUB = "https://github.com/ormeilu/avoid-ai-writing-russian/blob/master"


def mb(x: float | None) -> str:
    return "—" if x is None else f"{js_round(x)} МБ"


def ms(x: float | None) -> str:
    return "—" if x is None else f"{ru(x, 0) if x >= 10 else ru(x, 1)} мс"


def cmd_models_info(a: Args) -> int:
    """Справка по моделям до скачивания: из каталога в пакете."""
    cat = models.catalog()
    entries = {e["name"]: e for e in cat["models"]}
    if len(a.files) > 2:
        raise UsageError("models info: нужно не больше одного имени модели")
    if len(a.files) == 2:
        name = a.files[1]
        if name not in entries:
            raise UsageError(f"models info: нет модели {name}; есть: {', '.join(entries)}")
        e = entries[name]
        if a.json:
            sys.stdout.write(dump(e) + "\n")
            return 0
        q, meas = e["quality"], e.get("measured", {})
        heading(f"{e['name']}: {e['summary']}")
        facts(
            [
                ("Основа", e.get("base") or "LightGBM на признаках детектора aiw-ru"),
                ("Лицензия", e["license"]),
                ("ROC AUC", f"{ru(q['roc_auc'], 3)} на test, {ru(q['valid_roc_auc'], 3)} на valid"),
                ("Accuracy", ru(q["accuracy"], 3)),
                ("Людей за ИИ", f"{ru(q['human_as_ai'] * 100, 1)} % человеческих текстов test"),
                ("Тексты с нуля", f"ROC AUC {ru(q['roc_auc_created'], 3)}, если модель не правила текст человека"),
                ("Скорость", f"{ms(meas.get('short_ms'))} на 60 слов, {ms(meas.get('long_ms'))} на 800 слов"),
                ("Загрузка", ms(meas.get("load_ms"))),
                ("Память", f"{mb(meas.get('ram_mb'))}, пик процесса aiw-ru"),
                ("Скачать", mb(e.get("download_mb"))),
                ("Поставить", f"aiw-ru models install {name}"),
                ("Карточка", f"https://huggingface.co/{e['repo']}"),
                ("Отчёт", f"{GITHUB}/{e['report']}"),
            ]
        )
        return 0
    if a.json:
        sys.stdout.write(dump(cat) + "\n")
        return 0
    rows: list[list[str | Text]] = []
    for e in cat["models"]:
        q, meas = e["quality"], e.get("measured", {})
        rows.append(
            [
                e["name"],
                ru(q["roc_auc"], 3),
                ru(q["accuracy"], 3),
                f"{ru(q['human_as_ai'] * 100, 1)} %",
                ms(meas.get("long_ms")),
                mb(e.get("download_mb")),
                mb(meas.get("ram_mb")),
                "да" if e["default"] else "по имени",
            ]
        )
    heading("Необязательные модели aiw-ru")
    table(
        [
            Col("Модель"),
            Col("ROC AUC", "right"),
            Col("Accuracy", "right"),
            Col("Людей за ИИ", "right"),
            Col("800 слов", "right"),
            Col("Скачать", "right"),
            Col("Память", "right"),
            Col("Ставится"),
        ],
        rows,
    )
    out()
    facts([(e["name"], e["summary"]) for e in cat["models"]])
    out()
    packages = f"около {mb(cat['packages_mb'])}" if cat.get("packages_mb") else "пакеты"
    note(
        f"Качество — на test русской части LLMTrace ({plural(cat['models'][0]['quality']['test_texts'], 'текст', 'текста', 'текстов')}): ROC AUC 1 — "
        "модель всегда ставит ИИ-текст выше человеческого, 0,5 — угадывает. Скорость и память замерены на "
        f"{cat['measured_on']}, память — пик процесса aiw-ru вместе с детектором. Для любой модели нужны "
        f"{packages} из extra ml (общие для всех): {models.INSTALL_HINT}. «Ставится: да» — входит в "
        "aiw-ru models install без имени. Подробнее о модели: aiw-ru models info ИМЯ."
    )
    return 0


# Руководство по классификации — файл каталога основного скилла, его же читает агент.
GUIDE = "references/models.md"


def cmd_models(a: Args) -> int:
    if a.files and a.files[0] == "info":
        return cmd_models_info(a)
    if a.files and a.files[0] == "guide":
        text = skills.read(skills.MAIN, GUIDE)
        sys.stdout.write(text if text.endswith("\n") else text + "\n")
        return 0
    if a.files and a.files[0] == "install":
        chosen = [models.get(name) for name in a.files[1:]] or [m for m in models.MODELS.values() if m.default]
        for model in chosen:
            try:
                path = models.install(model)
            except OSError as e:
                raise models.ModelError(f"не удалось скачать модель {model.name}: {e}") from None
            facts([(model.name, Text.styled("скачана", "green")), ("Папка", str(path))])
        return 0
    if a.files:
        raise UsageError(f"models: неизвестное действие {a.files[0]}")
    states = [model_state(m) for m in models.MODELS.values()]
    if a.json:
        sys.stdout.write(dump({"ready": any(x["ready"] for x in states), "models": states}) + "\n")
        return 0
    for n, x in enumerate(states):
        if n > 0:
            out()
        heading(x["name"])
        facts(
            [
                ("Репозиторий", f"https://huggingface.co/{x['repo']}"),
                (
                    "Пакеты",
                    Text.styled("установлены", "green")
                    if x["dependencies"]
                    else Text.styled(f"нет {', '.join(x['missing'])}: {models.INSTALL_HINT}", "yellow"),
                ),
                (
                    "Файлы",
                    x["path"] or Text.styled(f"не скачаны: aiw-ru models install {x['name']}", "yellow"),
                ),
                (
                    "Состояние",
                    Text.styled(x["error"], "yellow")
                    if not x["ready"]
                    else Text.styled(f"готова, но {OUTDATED.format(name=x['name'])}", "yellow")
                    if x["outdated"]
                    else Text.styled("готова", "green"),
                ),
            ]
        )
    return 0


def cmd_skill(a: Args) -> int:
    if len(a.files) > 2:
        raise UsageError("skill: нужно не больше двух аргументов, имя скилла и файл")
    if a.files:
        text = skills.read(a.files[0], a.files[1] if len(a.files) > 1 else None)
        sys.stdout.write(text if text.endswith("\n") else text + "\n")
        return 0
    found = skills.all_skills()
    if a.json:
        sys.stdout.write(
            dump([{"name": s.name, "description": s.description, "files": list(s.files)} for s in found]) + "\n"
        )
        return 0
    heading("Скиллы aiw-ru")
    facts([(s.name, skills.first_sentence(s.description)) for s in found])
    out()
    note(
        "Текст для агента: aiw-ru skill ИМЯ, файлы скилла: aiw-ru skill ИМЯ ФАЙЛ. "
        f"Плагин для Claude Code и Codex со всеми скиллами: {skills.REPOSITORY}"
    )
    return 0


def main(argv: list[str]) -> int:
    global console
    console = make_console()
    told_outdated.clear()
    try:
        a = parse(argv)
        if a.help or not a.cmd:
            sys.stdout.write(USAGE + "\n")
            return 0 if a.help else 2
        commands = {
            "scan": cmd_scan,
            "antiplagiat": cmd_antiplagiat,
            "validate": cmd_validate,
            "calibrate": cmd_calibrate,
            "judge": cmd_judge,
            "classify": cmd_classify,
            "models": cmd_models,
            "skill": cmd_skill,
        }
        if a.cmd not in commands:
            raise UsageError(f"неизвестная команда: {a.cmd}")
        return commands[a.cmd](a)
    except UsageError as e:
        sys.stderr.write(f"aiw-ru: {e}\n\n{USAGE}")
        return 2
    except (models.ModelError, skills.SkillError) as e:
        sys.stderr.write(f"aiw-ru: {e}\n")
        return 2


def closed_pipe(e: OSError) -> bool:
    """Читатель закрыл канал. Windows сообщает об этом не BrokenPipeError, а EINVAL."""
    return isinstance(e, BrokenPipeError) or (sys.platform == "win32" and e.errno == errno.EINVAL)


def run() -> None:
    """Точка входа консольной команды `aiw-ru`."""
    try:
        code = main(sys.argv[1:])
        sys.stdout.flush()
    except OSError as e:
        if not closed_pipe(e):
            raise
        # Читатель закрыл канал (`aiw-ru scan … | head`): это не ошибка, просто выходим.
        # Остаток буфера уходит в devnull, иначе Python споткнётся о него при выходе.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        code = 0
    sys.exit(code)


if __name__ == "__main__":
    run()

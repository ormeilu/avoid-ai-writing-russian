"""aiw-ru — командная строка детектора.

    aiw-ru scan [файл…] [--context РЕЖИМ] [--json | --jsonl] [--min P0|P1|P2]
    aiw-ru antiplagiat [файл] [--system СИСТЕМА] [--config ПУТЬ] [--json | --jsonl]
    aiw-ru validate ИСХОДНЫЙ ИСПРАВЛЕННЫЙ [--context РЕЖИМ] [--marked ФАЙЛ] [--json]
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
from aiw_ru.types import CONTEXT_MODES, PROFILE_TO_MODE, AnalysisResult, ContextMode, Light, Record, Severity
from aiw_ru.validate import validate

USAGE = """aiw-ru — приметы ИИ-стиля в русском тексте

Команды:
  scan [файл…]                 найти приметы (без файла — читать stdin)
  antiplagiat [файл]           оценка по фрагментам в духе модуля ИИ-детекции «Антиплагиата»
                               или «Думейта» (--system domate)
  validate ИСХОДНЫЙ НОВЫЙ      проверить, что правка не повредила код, числа, цитаты, ссылки,
                               не сняла оговорки и не добавила имён, чисел и дат; с --marked
                               ещё и назвать абзацы под красной подсветкой отчёта без правки
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
    # judge score: файлы с ответами судьи (по одному на прогон); judge score и validate: разметка отчёта.
    answers: list[str] = field(default_factory=list)
    judge_marked: str | None = None
    help: bool = False


def parse(argv: list[str]) -> Args:
    args = Args()

    def need(i: int, flag: str) -> str:
        if i + 1 >= len(argv):
            raise UsageError(f"{flag} требует значения")
        return argv[i + 1]

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("-h", "--help"):
            args.help = True
        elif arg == "--json":
            args.json = True
        elif arg == "--jsonl":
            args.jsonl = True
        elif arg == "--no-model":
            args.no_model = True
        elif arg == "--all":
            args.all_models = True
        elif arg == "--no-fragments":
            args.no_fragments = True
        elif arg == "--max-fragments":
            value = need(i, arg)
            i += 1
            if not value.isdigit():
                raise UsageError(f"--max-fragments ждёт целое число 0 и больше: {value}")
            args.max_fragments = int(value) or None
        elif arg == "--model":
            value = need(i, arg)
            i += 1
            if value not in models.MODELS:
                raise UsageError(f"неизвестная --model: {value}; есть: {', '.join(models.MODELS)}")
            args.model_name = value
        elif arg == "--context":
            value = need(i, arg)
            i += 1
            mode = value if value in CONTEXT_MODES else PROFILE_TO_MODE.get(value)
            if not mode:
                raise UsageError(f"неизвестный --context: {value}")
            args.context = mode
        elif arg == "--min":
            value = need(i, arg)
            i += 1
            if value not in ("P0", "P1", "P2"):
                raise UsageError(f"неизвестный --min: {value}")
            args.min = value
        elif arg == "--fail-above":
            value = need(i, arg)
            i += 1
            try:
                args.fail_above = float(value)
            except ValueError:
                args.fail_above = math.nan
            if math.isnan(args.fail_above):
                raise UsageError("--fail-above ждёт число")
        elif arg == "--config":
            args.config = need(i, arg)
            i += 1
        elif arg == "--system":
            value = need(i, arg)
            i += 1
            if value not in SYSTEMS:
                raise UsageError(f"неизвестная --system: {value}; есть: {', '.join(SYSTEMS)}")
            args.system = next(system for system in SYSTEMS if system == value)
        elif arg == "--answers":
            args.answers.append(need(i, arg))
            i += 1
        elif arg == "--marked" and args.cmd in ("judge", "validate"):
            args.judge_marked = need(i, arg)
            i += 1
        elif arg == "--doc":
            args.reports.append(Report(need(i, arg)))
            i += 1
        elif arg in ("--marked", "--share"):
            value = need(i, arg)
            i += 1
            last = args.reports[-1] if args.reports else None
            if last is None or last.marked is not None or last.share is not None:
                raise UsageError(f"{arg} относится к предыдущему --doc: сначала --doc ФАЙЛ")
            if arg == "--marked":
                last.marked = value
            else:
                try:
                    share = float(value.replace(",", ".", 1).removesuffix("%"))
                except ValueError:
                    share = math.nan
                if not math.isfinite(share) or share < 0 or share > 100:
                    raise UsageError(f"--share ждёт долю 0–100: {value}")
                last.share = share
        elif arg.startswith("-") and arg != "-":
            raise UsageError(f"неизвестный параметр: {arg}")
        elif not args.cmd:
            args.cmd = arg
        else:
            args.files.append(arg)
        i += 1
    return args


def read(file: str | None) -> str:
    from_stdin = file is None or file == "-"
    try:
        data = sys.stdin.buffer.read() if from_stdin else Path(file).read_bytes()
        # Как TextDecoder в JS: строгий UTF-8, BOM в начале отбрасывается.
        return data.decode("utf-8-sig")
    except (OSError, UnicodeDecodeError) as error:
        reason = error.strerror if isinstance(error, OSError) and error.strerror else str(error)
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
LIGHT_NAMES: dict[str, str] = {"green": "зелёный", "yellow": "жёлтый", "red": "красный"}


def light_fact(light: Light) -> tuple[str, Text]:
    """Строка светофора: цвет и что делать."""
    return ("Светофор", Text.styled(f"{LIGHT_NAMES[light.level]}: {light.hint}", light.level))


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
    text = ru(n, 1)
    return f"{text} слова" if "," in text else plural(int(text), "слово", "слова", "слов")


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
    for key, value in pairs:
        grid.add_row(key, value)
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
    widths = [
        max([cell_len(column.title), *(cell_len(str(row[k])) for row in rows)]) for k, column in enumerate(columns)
    ]
    room = width - INDENT - GAP * (len(columns) - 1)
    for floor in (lambda column: column.min or 10, lambda _column: 1):
        while sum(widths) > room:
            shrinkable = [k for k, column in enumerate(columns) if column.flex and widths[k] > floor(column)]
            if not shrinkable:
                break
            widths[max(shrinkable, key=lambda k: widths[k])] -= 1
    return widths


def table(columns: list[Col], rows: list[list[str | Text]]) -> None:
    # Ширины считает layout(); rich только рисует. Поле справа у каждой колонки, кроме последней,
    # плюс разделитель из RULE_BOX дают промежуток GAP.
    rich_table = Table(
        box=RULE_BOX,
        show_edge=False,
        pad_edge=False,
        padding=(0, GAP - 1, 0, 0),
        header_style="dim",
        border_style="dim",
    )
    widths = layout(columns, rows, console.width)
    for column, column_width in zip(columns, widths, strict=True):
        rich_table.add_column(
            column.title,
            justify=column.justify,
            width=column_width,
            no_wrap=not column.wrap,
            overflow="fold" if column.wrap else "ellipsis",
        )
    for row in rows:
        rich_table.add_row(
            *(
                cell if column.wrap else truncate(cell, column_width)
                for column, column_width, cell in zip(columns, widths, row, strict=True)
            )
        )
    out(indented(rich_table))


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
    for index, line in enumerate(read(a.files[0] if a.files else None).split("\n")):
        if not trim(line):
            continue
        try:
            doc = json.loads(line)
        except json.JSONDecodeError:
            raise UsageError(f"строка {index + 1}: не JSON") from None
        text = doc.get("text") if isinstance(doc, dict) else None
        if not isinstance(text, str):
            raise UsageError(f"строка {index + 1}: нет строкового поля text")
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
    """Число с разрядами через обычный пробел от 10 000, как в plural."""
    return f"{n:,}".replace(",", " ") if abs(n) >= 10000 else str(n)


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
    parts = [str(first) if first == last else f"{first}–{last}" for first, last in ranges]
    one = len(ranges) == 1 and ranges[0][0] == ranges[0][1]
    return f"{'строка' if one else 'строки'} {', '.join(parts)}"


def foreign_fact(sig: Signal) -> list[tuple[str, str]]:
    """«Не на русском: строки 11, 40–45, модель их не читала»."""
    if not sig.foreign or sig.skipped:
        return []
    pronoun = "её" if len(sig.foreign) == 1 and sig.foreign[0][0] == sig.foreign[0][1] else "их"
    return [("Не на русском", f"{line_ranges(sig.foreign)}: модель {pronoun} не читала")]


def read_facts(sig: Signal | None) -> list[tuple[str, str]]:
    if sig is None:
        return []
    coverage_facts = [read_fact(sig.coverage)] if sig.coverage is not None and sig.truncated else []
    return coverage_facts + foreign_fact(sig)


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
    merged: list[tuple[int, int]] = []
    for chunk in sorted(chunks, key=lambda chunk: chunk.line):
        if merged and chunk.line <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], chunk.end_line))
        else:
            merged.append((chunk.line, chunk.end_line))
    return merged


def word_share(chunks: list[models.Chunk], scan: models.ChunkScan) -> float:
    """Доля слов текста для модели, попавших хотя бы в один из фрагментов; перекрытия считаются один раз.

    Делится на scan.words: слова шапки, кода, служебных частей и фрагментов не на русском в знаменатель
    не входят."""
    spans: list[tuple[int, int]] = []
    for chunk in sorted(chunks, key=lambda chunk: chunk.start):
        if spans and chunk.start <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], chunk.end))
        else:
            spans.append((chunk.start, chunk.end))
    return sum(len(words(scan.text[start:end])) for start, end in spans) / scan.words if scan.words else 0.0


def near(p: float | None, threshold: float) -> bool:
    """Вероятность ближе к порогу, чем NEAR_THRESHOLD."""
    return p is not None and abs(p - threshold) < NEAR_THRESHOLD


def fragments_json(scan: models.ChunkScan, loaded: models.Loaded) -> dict[str, Any]:
    """count — фрагменты, которые оценила модель, skipped — не на русском, total — все в тексте;
    в items и те и другие, подряд. aboveThreshold, aiWordShare и aiLines считаются по оценённым."""
    threshold = loaded.threshold
    above = [chunk for chunk in scan.chunks if chunk.above(threshold)]
    return {
        "count": len(scan.rated),
        "skipped": scan.skipped,
        "total": scan.total,
        "overlap": scan.overlap,
        "seconds": round(scan.seconds, 3),
        "aboveThreshold": len(above),
        "aiWordShare": round(word_share(above, scan), 4),
        "aiLines": [list(line_range) for line_range in merged_lines(above)],
        "items": [
            {
                "line": chunk.line,
                "endLine": chunk.end_line,
                "start": chunk.start,
                "end": chunk.end,
                "words": chunk.words,
                "language": "other" if chunk.skipped else "ru",
                "skipped": chunk.skipped,
                "probability": None if chunk.probability is None else round(chunk.probability, 4),
                "ai": None if chunk.skipped else chunk.above(threshold),
                "nearThreshold": None if chunk.skipped else near(chunk.probability, threshold),
            }
            for chunk in scan.chunks
        ],
    }


def not_russian_note(n: int) -> str:
    """«2 фрагмента не на русском, модель их не оценивала»."""
    pronoun = "его" if n % 10 == 1 and n % 100 != 11 else "их"
    return f"{plural(n, 'фрагмент', 'фрагмента', 'фрагментов')} не на русском, модель {pronoun} не оценивала."


def show_fragments(scan: models.ChunkScan, loaded: models.Loaded) -> None:
    """Таблица фрагментов длинного текста, итог и сколько заняла проверка."""
    threshold = loaded.threshold

    def style(probability: float) -> str:
        return "yellow" if near(probability, threshold) else "red" if probability >= threshold else "green"

    table(
        [Col("Строки"), Col("Слов", justify="right"), Col("ИИ", justify="right")],
        [
            [
                Text.styled(
                    str(chunk.line) if chunk.line == chunk.end_line else f"{chunk.line}–{chunk.end_line}", "dim"
                ),
                str(chunk.words),
                Text.styled("не на русском", "dim")
                if chunk.probability is None
                else Text.styled(f"{js_round(chunk.probability * 100)} %", style(chunk.probability)),
            ]
            for chunk in scan.chunks
        ],
    )
    out()
    rated = scan.rated
    above = [chunk for chunk in rated if chunk.above(threshold)]
    if above:
        ranges = merged_lines(above)
        where = (
            f"строка {ranges[0][0]}"
            if len(ranges) == 1 and ranges[0][0] == ranges[0][1]
            else "строки " + ", ".join(str(first) if first == last else f"{first}–{last}" for first, last in ranges)
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
    model = sig.loaded.model
    return {
        "name": model.name,
        "language": "other" if sig.skipped else "ru",
        "skipped": sig.skipped,
        "cyrillicShare": None if sig.cyrillic is None else round(sig.cyrillic, 4),
        "probability": None if sig.probability is None else round(sig.probability, 4),
        "threshold": round(sig.loaded.threshold, 6),
        "ai": None if sig.skipped else sig.ai,
        "nearThreshold": None if sig.skipped else sig.near_threshold,
        "repo": model.repo,
        "outdated": models.outdated(model),
        "read": None if sig.coverage is None else coverage_json(sig.coverage),
        "notRussianLines": [list(line_range) for line_range in sig.foreign],
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
    threshold = js_round(sig.loaded.threshold * 100)
    source = f"порог {threshold} %" if label else f"{sig.loaded.model.title}, порог {threshold} %"
    if label and sig.truncated and sig.coverage is not None:
        source += f", прочитано {read_words(sig.coverage)}"
    label = label or "Вероятность ИИ"
    text = f"{js_round(sig.probability * 100)} % ({source}"
    if sig.near_threshold:
        return (label, Text.styled(text + ", близко к порогу: модель не уверена)", "yellow"))
    return (label, Text.styled(text + ")", "red" if sig.ai else "green"))


def agreement(sigs: list[Signal]) -> Text:
    """Согласны ли модели: одна строка под вероятностями `classify --all`."""
    above = sum(sig.ai for sig in sigs)
    if above == len(sigs):
        return Text.styled("Все модели выше порога. Это сигнал, а не вывод: сверьте с находками и текстом.", "red")
    if above == 0:
        return Text.styled("Все модели ниже порога.", "green")
    return Text.styled(
        f"Модели расходятся: {above} из {len(sigs)} выше порога. Вероятность здесь ничего не решает, смотрите на находки.",
        "yellow",
    )


def catalog_refs(r: AnalysisResult) -> dict[str, list[str]]:
    """Файлы каталога и их разделы для найденных примет, в порядке уровня серьёзности."""
    where = skills.catalog()
    refs: dict[str, list[str]] = {}
    for issue in sorted(r.issues, key=lambda issue: (RANK[issue.severity], issue.index)):
        section = TYPE_TO_SECTION.get(issue.type)
        path = where.get(section) if section else None
        if section and path and section not in refs.setdefault(path, []):
            refs[path].append(section)
    return refs


def scan_json(r: AnalysisResult) -> dict[str, Any]:
    """Результат scan для JSON: с файлами каталога, где описаны найденные приметы."""
    return {
        **r.to_dict(),
        "catalog": [{"file": file, "sections": sections} for file, sections in catalog_refs(r).items()],
    }


def cmd_scan(a: Args) -> int:
    mode = a.context or "general"
    if a.jsonl:

        def one(text: str) -> dict[str, Any]:
            result = analyze(text, mode)
            return with_signal(scan_json(result), model_signal(a, text, result))

        return jsonl(a, one)
    files = a.files or ["-"]
    texts = [(file, read(file)) for file in files]
    results = [(file, analyze(text, mode)) for file, text in texts]
    signals = [model_signal(a, text, result) for (_, text), (_, result) in zip(texts, results, strict=True)]
    if a.json:
        docs = [
            with_signal({"file": file, **scan_json(result)}, sig)
            for (file, result), sig in zip(results, signals, strict=True)
        ]
        sys.stdout.write(dump(docs[0] if len(docs) == 1 else docs) + "\n")
    fail = False
    for index, ((file, result), sig) in enumerate(zip(results, signals, strict=True)):
        if a.fail_above is not None and result.score > a.fail_above:
            fail = True
        if a.json:
            continue
        if index > 0:
            out()
        shown = sorted(
            (issue for issue in result.issues if RANK[issue.severity] <= RANK[a.min]),
            key=lambda issue: (RANK[issue.severity], int(issue.style_only), issue.index),
        )
        stats = result.stats
        heading("stdin" if file == "-" else file)
        facts(
            [
                ("Оценка", Text.styled(f"{result.score}/100, {result.label}", score_style(result.score))),
                light_fact(result.light),
                (
                    "Текст",
                    (
                        f"{plural(stats.words, 'слово', 'слова', 'слов')}, "
                        f"{plural(stats.sentences, 'предложение', 'предложения', 'предложений')}, "
                        f"режим {stats.context_mode}"
                    ),
                ),
                (
                    "Ритм",
                    (
                        f"в среднем {mean_words(stats.mean_sentence_length)} в предложении, "
                        f"разброс {ru(stats.sentence_length_cv)}, MATTR {ru(stats.mattr)}"
                    ),
                ),
                *([signal_fact(sig)] if sig is not None else []),
                *read_facts(sig),
            ]
        )
        fragments_hint(sig, file)
        if result.suspicious:
            warn("Документ выглядит подозрительным: невидимые символы или подмена букв.", "red")
        out()
        if not shown:
            out(Text("  Находок выбранного уровня нет." if result.issues else "  Примет не найдено.", style="green"))
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
                    Text.styled("стиль", "dim")
                    if issue.style_only
                    else Text.styled(issue.severity, SEV_STYLE[issue.severity]),
                    Text.styled(f"{issue.line}:{issue.column}", "dim"),
                    TYPE_LABELS.get(issue.type, issue.type),
                    f"«{issue.text}»",
                    issue.hint,
                ]
                for issue in shown
            ],
        )
        out()
        note(LEGEND)
        refs = catalog_refs(result)
        if refs:
            note(
                "Условия и исключения в каталоге скилла: "
                + "; ".join(f"{path} ({', '.join(sections)})" for path, sections in refs.items())
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
    user_config = Path.home() / ".config" / "aiw-ru.json"
    return str(user_config) if user_config.exists() else None


def load_calibration(path: str | None) -> CalibrationFile | None:
    if not path or not os.path.exists(path):
        return None
    try:
        return CalibrationFile.model_validate_json(Path(path).read_bytes())
    except ValidationError as error:
        first_error = error.errors()[0]
        where = ".".join(map(str, first_error["loc"])) or "файл"
        raise UsageError(
            f"не удалось прочитать калибровку {path}: неверный формат ({where}: {first_error['msg']})"
        ) from None
    except OSError as error:
        raise UsageError(f"не удалось прочитать калибровку {path}: {error}") from None


def cmd_antiplagiat(a: Args) -> int:
    calibration_file = load_calibration(config_path(a))
    calibration = calibration_file.get(a.system) if calibration_file else None
    model: Model | None = calibration.model if calibration else None
    mode = a.context or "academic"
    if a.jsonl:
        return jsonl(
            a, lambda text: with_signal(antiplagiat(text, mode, model, a.system).to_dict(), model_signal(a, text))
        )
    text = read(a.files[0] if a.files else None)
    report = antiplagiat(text, mode, model, a.system)
    sig = model_signal(a, text)
    if a.json:
        sys.stdout.write(dump(with_signal(report.to_dict(), sig)) + "\n")
        return 0
    share = report.ai_share
    share_style = "green" if share < 20 else "yellow" if share < 50 else "red"
    ai_fragments = sum(1 for fragment in report.fragments if fragment.ai)
    heading(a.files[0] if a.files and a.files[0] != "-" else "stdin")
    facts(
        [
            ("Доля ИИ-текста", Text.styled(f"{ru(share)} %", share_style)),
            light_fact(report.light),
            ("Фрагменты", f"{len(report.fragments)}, похожи на ИИ: {ai_fragments} (порог {ru(report.threshold)})"),
            (
                "Модель",
                f"откалибрована по вашим отчётам, образцов: {len(calibration.samples) if calibration else 0}"
                if report.model == "calibrated"
                else "по умолчанию, без калибровки",
            ),
            *([signal_fact(sig)] if sig is not None else []),
            *read_facts(sig),
        ]
    )
    fragments_hint(sig, a.files[0] if a.files else None)
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
                Text.styled(
                    str(fragment.line)
                    if fragment.line == fragment.end_line
                    else f"{fragment.line}–{fragment.end_line}",
                    "dim",
                ),
                str(fragment.words),
                Text.styled(f"{js_round(fragment.probability * 100)} %", "red" if fragment.ai else "green"),
                f"{fragment.preview}…",
                ", ".join(fragment.reasons) if fragment.ai else "",
            ]
            for fragment in report.fragments
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
    marked = split_marked(read(a.judge_marked)) if a.judge_marked else None
    result = validate(read(a.files[0]), read(a.files[1]), a.context or "general", marked)
    if a.json:
        sys.stdout.write(dump(result) + "\n")
    else:
        rows: list[tuple[str, str | Text]] = [
            (
                "Сохранность",
                Text.styled("в порядке", "green")
                if result.ok
                else Text.styled(f"нарушена ({len(result.violations)})", "red"),
            )
        ]
        if result.warnings:
            rows.append(("Проверить", Text.styled(str(len(result.warnings)), "yellow")))
        rows.append(("Находки", f"{result.issues_before} → {result.issues_after}"))
        facts(rows)
        if not result.ok:
            out()
            table(
                [Col("Что"), Col("Подробности", flex=True, wrap=True, min=20)],
                [[Text.styled(violation.kind, "yellow"), violation.detail] for violation in result.violations],
            )
        if result.warnings:
            out()
            table(
                [Col("Проверить"), Col("Подробности", flex=True, wrap=True, min=20)],
                [[Text.styled(warning.kind, "cyan"), warning.detail] for warning in result.warnings],
            )
            note("Код выхода они не меняют: это может быть законной правкой, сверьте с исходником.")
    return 0 if result.ok else 1


MARKED_SPLIT_RE = jsre("\\n\\s*(?:---+\\s*)?\\n")


def split_marked(text: str) -> list[str]:
    """Куски из отчёта: тепловая карта в JSON (красные фрагменты) или текст через пустую строку."""
    heat = marked_from_heatmap(text)
    if heat is not None:
        return heat
    return [piece for piece in (trim(part) for part in MARKED_SPLIT_RE.split(text)) if piece]


def cmd_calibrate(a: Args) -> int:
    if not a.reports or any(report.marked is None and report.share is None for report in a.reports):
        raise UsageError("calibrate ждёт для каждого --doc ФАЙЛ разметку: --marked ФАЙЛ или --share ДОЛЯ")
    # Тот же файл, откуда antiplagiat читает калибровку: --config, $AIW_RU_CONFIG,
    # ./.aiw-ru.json, затем ~/.config/aiw-ru.json. Иначе образцы копились бы не в том файле,
    # который читается, и калибровка из ~/.config терялась бы.
    path = config_path(a) or ".aiw-ru.json"
    calibration_file = load_calibration(path if os.path.exists(path) else None) or empty_calibration_file()
    previous = calibration_file.get(a.system)
    samples = list(previous.samples) if previous else []
    documents = list(previous.documents) if previous else []
    rows: list[list[str | Text]] = []
    warnings: list[str] = []
    for report in a.reports:
        doc = read(report.doc)
        if report.share is not None:
            doc_sample = document_sample(doc, report.share)
            documents.append(doc_sample)
            rows.append([report.doc, f"доля {ru(report.share)} %", str(len(doc_sample.fragments)), "—"])
            continue
        marked = split_marked(read(report.marked))
        labeled = label_fragments(doc, marked)
        positives = sum(1 for sample in labeled if sample.y == 1)
        rows.append([report.doc, report.marked or "", str(len(labeled)), str(positives)])
        if marked and positives == 0:
            warnings.append(
                f"{report.doc}: ни один кусок из {report.marked} не найден в документе. Проверьте, что текст "
                "скопирован из отчёта без правок и документ той же версии."
            )
        samples.extend(labeled)
    start = previous.model if previous else DEFAULT_MODEL
    model = fit_share(fit(samples, start), documents)
    data = calibration_file.put(a.system, Calibration(model=model, samples=samples, documents=documents))
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
    for warning in warnings:
        warn(warning, "yellow")
    out()

    def pct(value: float) -> str:
        return "н/д" if math.isnan(value) else f"{ru(value * 100, 1)} %"

    def pp(value: float) -> str:
        return "н/д" if math.isnan(value) else f"{ru(value, 1)} п.п."

    positives = sum(1 for sample in samples if sample.y == 1)
    heading("Калибровка")
    facts(
        [
            ("Система", SYSTEMS[a.system][0]),
            ("Фрагменты с разметкой", f"{len(samples)}, из них ИИ: {positives}"),
            ("Документы с долей", str(len(documents))),
            (
                "Точность по фрагментам",
                "н/д (нужны и ИИ-, и человеческие фрагменты)"
                if math.isnan(balanced_accuracy(model, samples))
                else f"{pct(balanced_accuracy(DEFAULT_MODEL, samples))} → {pct(balanced_accuracy(model, samples))}",
            ),
            ("Ошибка доли ИИ", f"{pp(share_error(DEFAULT_MODEL, documents))} → {pp(share_error(model, documents))}"),
            ("Порог", ru(model.threshold)),
            ("Веса", "; ".join(f"{FEATURE_NAMES[k]} {ru(weight)}" for k, weight in enumerate(model.weights))),
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
        data = {
            "prompt": judge_prompt(text, mode),
            "fragments": [fragment.to_dict() for fragment in judge_fragments(text, mode)],
        }
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
    report = judge_score(doc, [read(answers_file) for answers_file in a.answers], marked, a.context or "academic")
    if a.json:
        sys.stdout.write(dump(report) + "\n")
        return 0

    def level(value: int | None) -> str:
        return "—" if value is None else LEVEL_NAMES[value]

    def mean_style(mean: float | None) -> str:
        return "dim" if mean is None else "red" if mean >= 1.5 else "yellow" if mean >= 0.5 else "green"

    heading(a.files[1])
    rows: list[tuple[str, str | Text]] = [("Прогонов", str(report.runs))]
    if report.agreement is not None:
        rows.append(("Совпадение прогонов", f"{ru(report.agreement * 100, 0)} % фрагментов"))
    rows.append(("Доля слов", f"medium и выше {ru(report.share_medium, 1)} %, low и выше {ru(report.share_low, 1)} %"))
    if report.auc is not None or report.detector_auc is not None:
        both = (
            f"судья {'н/д' if report.auc is None else ru(report.auc)}, "
            f"детектор {'н/д' if report.detector_auc is None else ru(report.detector_auc)}"
        )
        rows.append(("ROC AUC по отчёту", both))
    facts(rows)
    if report.missing:
        warn(f"Нет ответа судьи для фрагментов: {', '.join(map(str, report.missing))}.", "yellow")
    out()
    table(
        [
            Col("№", justify="right"),
            Col("Строки"),
            Col("Слов", justify="right"),
            Col("Уровни"),
            Col("Среднее", justify="right"),
            *([Col("Отчёт", justify="right")] if report.fragments and report.fragments[0].marked is not None else []),
            Col("Начало фрагмента", flex=True, min=24),
        ],
        [
            [
                str(fragment.n),
                Text.styled(
                    str(fragment.line)
                    if fragment.line == fragment.end_line
                    else f"{fragment.line}–{fragment.end_line}",
                    "dim",
                ),
                str(fragment.words),
                " ".join(level(value) for value in fragment.levels),
                Text.styled("—" if fragment.mean is None else ru(fragment.mean), mean_style(fragment.mean)),
                *([f"{js_round(fragment.marked * 100)} %"] if fragment.marked is not None else []),
                f"{fragment.preview}…",
            ]
            for fragment in report.fragments
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
    loaded = [models.load(model) for model in models.MODELS.values() if models.available(model)]
    if not loaded:
        raise models.ModelError(
            f"ни одна модель не установлена: поставьте пакеты ({models.INSTALL_HINT}) и скачайте модели "
            "(aiw-ru models install)"
        )

    def one(text: str) -> list[Signal]:
        return [signal(model, text) for model in loaded]

    def doc(sigs: list[Signal]) -> dict[str, Any]:
        # agree пуст (null), если текст не на русском и модели его не оценивали.
        agree = None if sigs[0].skipped else len({sig.ai for sig in sigs}) == 1
        return {"models": [signal_json(sig) for sig in sigs], "agree": agree}

    if a.jsonl:
        return jsonl(a, lambda text: doc(one(text)))
    files = a.files or ["-"]
    results = [(file, one(read(file))) for file in files]
    if a.json:
        docs = [{"file": file, **doc(sigs)} for file, sigs in results]
        sys.stdout.write(dump(docs[0] if len(docs) == 1 else docs) + "\n")
        return 0
    for index, (file, sigs) in enumerate(results):
        if index > 0:
            out()
        heading("stdin" if file == "-" else file)
        if sigs[0].skipped:
            facts([skip_fact(sigs)])
            continue
        facts([*(signal_fact(sig, sig.loaded.model.title) for sig in sigs), *foreign_fact(sigs[0])])
        out(indented(agreement(sigs)))
        if any(sig.truncated for sig in sigs):
            target = file if file != "-" else "<файл>"
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
    results = [(file, *one(read(file))) for file in files]
    if a.json:
        docs = [{"file": file, **doc(sig, scan)} for file, sig, scan in results]
        sys.stdout.write(dump(docs[0] if len(docs) == 1 else docs) + "\n")
        return 0
    for index, (file, sig, scan) in enumerate(results):
        if index > 0:
            out()
        heading("stdin" if file == "-" else file)
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
    status = models.status(model)
    error = None
    try:
        models.load(model)
    except models.ModelError as exc:
        error = str(exc)
    return {
        "name": model.name,
        "repo": model.repo,
        "dependencies": status.dependencies,
        "missing": models.missing_dependencies(model),
        "path": str(status.path) if status.path else None,
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
    catalog = models.catalog()
    entries = {entry["name"]: entry for entry in catalog["models"]}
    if len(a.files) > 2:
        raise UsageError("models info: нужно не больше одного имени модели")
    if len(a.files) == 2:
        name = a.files[1]
        if name not in entries:
            raise UsageError(f"models info: нет модели {name}; есть: {', '.join(entries)}")
        entry = entries[name]
        if a.json:
            sys.stdout.write(dump(entry) + "\n")
            return 0
        quality, measured = entry["quality"], entry.get("measured", {})
        heading(f"{entry['name']}: {entry['summary']}")
        facts(
            [
                ("Основа", entry.get("base") or "LightGBM на признаках детектора aiw-ru"),
                ("Лицензия", entry["license"]),
                ("ROC AUC", f"{ru(quality['roc_auc'], 3)} на test, {ru(quality['valid_roc_auc'], 3)} на valid"),
                ("Accuracy", ru(quality["accuracy"], 3)),
                ("Людей за ИИ", f"{ru(quality['human_as_ai'] * 100, 1)} % человеческих текстов test"),
                (
                    "Тексты с нуля",
                    f"ROC AUC {ru(quality['roc_auc_created'], 3)}, если модель не правила текст человека",
                ),
                ("Скорость", f"{ms(measured.get('short_ms'))} на 60 слов, {ms(measured.get('long_ms'))} на 800 слов"),
                ("Загрузка", ms(measured.get("load_ms"))),
                ("Память", f"{mb(measured.get('ram_mb'))}, пик процесса aiw-ru"),
                ("Скачать", mb(entry.get("download_mb"))),
                ("Поставить", f"aiw-ru models install {name}"),
                ("Карточка", f"https://huggingface.co/{entry['repo']}"),
                ("Отчёт", f"{GITHUB}/{entry['report']}"),
            ]
        )
        return 0
    if a.json:
        sys.stdout.write(dump(catalog) + "\n")
        return 0
    rows: list[list[str | Text]] = []
    for entry in catalog["models"]:
        quality, measured = entry["quality"], entry.get("measured", {})
        rows.append(
            [
                entry["name"],
                ru(quality["roc_auc"], 3),
                ru(quality["accuracy"], 3),
                f"{ru(quality['human_as_ai'] * 100, 1)} %",
                ms(measured.get("long_ms")),
                mb(entry.get("download_mb")),
                mb(measured.get("ram_mb")),
                "да" if entry["default"] else "по имени",
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
    facts([(entry["name"], entry["summary"]) for entry in catalog["models"]])
    out()
    packages = f"около {mb(catalog['packages_mb'])}" if catalog.get("packages_mb") else "пакеты"
    note(
        f"Качество — на test русской части LLMTrace ({plural(catalog['models'][0]['quality']['test_texts'], 'текст', 'текста', 'текстов')}): ROC AUC 1 — "
        "модель всегда ставит ИИ-текст выше человеческого, 0,5 — угадывает. Скорость и память замерены на "
        f"{catalog['measured_on']}, память — пик процесса aiw-ru вместе с детектором. Для любой модели нужны "
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
        chosen = [models.get(name) for name in a.files[1:]] or [
            model for model in models.MODELS.values() if model.default
        ]
        for model in chosen:
            try:
                path = models.install(model)
            except OSError as error:
                raise models.ModelError(f"не удалось скачать модель {model.name}: {error}") from None
            facts([(model.name, Text.styled("скачана", "green")), ("Папка", str(path))])
        return 0
    if a.files:
        raise UsageError(f"models: неизвестное действие {a.files[0]}")
    states = [model_state(model) for model in models.MODELS.values()]
    if a.json:
        sys.stdout.write(dump({"ready": any(state["ready"] for state in states), "models": states}) + "\n")
        return 0
    for index, state in enumerate(states):
        if index > 0:
            out()
        heading(state["name"])
        facts(
            [
                ("Репозиторий", f"https://huggingface.co/{state['repo']}"),
                (
                    "Пакеты",
                    Text.styled("установлены", "green")
                    if state["dependencies"]
                    else Text.styled(f"нет {', '.join(state['missing'])}: {models.INSTALL_HINT}", "yellow"),
                ),
                (
                    "Файлы",
                    state["path"] or Text.styled(f"не скачаны: aiw-ru models install {state['name']}", "yellow"),
                ),
                (
                    "Состояние",
                    Text.styled(state["error"], "yellow")
                    if not state["ready"]
                    else Text.styled(f"готова, но {OUTDATED.format(name=state['name'])}", "yellow")
                    if state["outdated"]
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
            dump(
                [{"name": skill.name, "description": skill.description, "files": list(skill.files)} for skill in found]
            )
            + "\n"
        )
        return 0
    heading("Скиллы aiw-ru")
    facts([(skill.name, skills.first_sentence(skill.description)) for skill in found])
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
        args = parse(argv)
        if args.help or not args.cmd:
            sys.stdout.write(USAGE + "\n")
            return 0 if args.help else 2
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
        if args.cmd not in commands:
            raise UsageError(f"неизвестная команда: {args.cmd}")
        return commands[args.cmd](args)
    except UsageError as error:
        sys.stderr.write(f"aiw-ru: {error}\n\n{USAGE}")
        return 2
    except (models.ModelError, skills.SkillError) as error:
        sys.stderr.write(f"aiw-ru: {error}\n")
        return 2


def closed_pipe(e: OSError) -> bool:
    """Читатель закрыл канал. Windows сообщает об этом не BrokenPipeError, а EINVAL."""
    return isinstance(e, BrokenPipeError) or (sys.platform == "win32" and e.errno == errno.EINVAL)


def run() -> None:
    """Точка входа консольной команды `aiw-ru`."""
    try:
        code = main(sys.argv[1:])
        sys.stdout.flush()
    except OSError as error:
        if not closed_pipe(error):
            raise
        # Читатель закрыл канал (`aiw-ru scan … | head`): это не ошибка, просто выходим.
        # Остаток буфера уходит в devnull, иначе Python споткнётся о него при выходе.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        code = 0
    sys.exit(code)


if __name__ == "__main__":
    run()

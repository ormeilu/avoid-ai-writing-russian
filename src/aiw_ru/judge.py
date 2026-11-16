"""Слепой судья: задание для свежего агента и сводка его ответов.

На отшлифованном научном тексте правила и модели детектора могут не видеть ничего там,
где система проверки находит треть ИИ-текста. Языковая модель без контекста редактора
расставляет абзацы по подозрительности заметно ближе к системе. Процент системы она не
предсказывает и от прогона к прогону отвечает по-разному, поэтому ответы нескольких
прогонов усредняются, а годится сводка только для выбора абзацев, которые править первыми.

Фрагменты те же, что у `antiplagiat`, поэтому номера строк и разметку отчёта можно сверять.
"""

from __future__ import annotations

import math
import re
from itertools import combinations

from pydantic import Field

from aiw_ru.antiplagiat import antiplagiat, marked_fragments
from aiw_ru.calibration import doc_fingerprint
from aiw_ru.compat import fixed
from aiw_ru.types import ContextMode, Record

LEVELS: dict[str, int] = {"none": 0, "low": 1, "medium": 2, "high": 3}

# Версия задания: растёт при каждой правке PROMPT, по ней видно, каким заданием получены ответы.
# 1 — уровни через доли человеческих текстов (1, 5 и 10 %), эти доли никто не измерял;
# 2 — порядковая шкала уверенности.
PROMPT_VERSION = 2

# Задание судье. Текст тот же, что в references/judge.md скилла (это проверяет тест).
PROMPT = """Ты детектор ИИ-текста для академических работ на русском, как модуль ИИ-детекции в системах проверки вузов.

Оцени текст ниже. Не открывай файлы, не запускай команды и программы, не ищи в интернете: оценка только твоим суждением как читателя-эксперта.

Текст разбит на фрагменты [P1]…[P{count}]. Это части одного документа. Для каждого фрагмента оцени, похож ли он на текст, сгенерированный языковой моделью (в том числе сгенерированный и затем отредактированный человеком). Шкала уверенности:

- high: признаков ИИ много, и они складываются вместе; у человека такое сочетание редкость;
- medium: признаков несколько, но человек мог написать так же;
- low: отдельные слабые признаки, скорее человеческий текст;
- none: обычный человеческий текст.

Уровни — порядок твоей уверенности, а не доля ошибок: сравнивай фрагменты между собой.

Сведения об авторах, заголовки, формулы и подписи к ним ставь none.

Смотри на совокупность признаков, а не на отдельное слово: однородность ритма и длины предложений, шаблонные синтаксические конструкции, «отполированность» без шероховатостей, общие формулировки вместо конкретики (числа, названия, ссылки на источники), однотипное построение абзацев (тезис, пояснение, вывод), пояснения через двоеточие, отсутствие авторской позиции. Сухой научный или деловой стиль сам по себе не признак ИИ: люди тоже так пишут. Ставь high, только если уверен.

Формат ответа строго такой, без вступления и пояснений после:

P1 <high|medium|low|none> <причина, до 12 слов>
P2 ...
…
P{count} ...

{fragments}
"""

ANSWER_RE = re.compile(r"^\W*P\s*(\d+)\W+(high|medium|low|none)\b", re.IGNORECASE | re.MULTILINE)
# Строка о прогоне в начале файла ответа: «# model: …», «# temperature: 0».
RUN_META_RE = re.compile(r"^#\s*([\w-]+)\s*:\s*(.+?)\s*$", re.MULTILINE)
# Разница уровней между прогонами, от которой фрагмент считается спорным: none против medium.
DISAGREEMENT = 2


class JudgeFragment(Record):
    n: int
    line: int
    end_line: int
    words: int
    text: str


def judge_fragments(source: str, context: ContextMode = "academic") -> list[JudgeFragment]:
    """Фрагменты для судьи, пронумерованные с 1."""
    return [
        JudgeFragment(
            n=number,
            line=fragment.line,
            end_line=fragment.end_line,
            words=fragment.words,
            text=source[fragment.start : fragment.end].strip(),
        )
        for number, fragment in enumerate(antiplagiat(source, context).fragments, 1)
    ]


def judge_prompt(source: str, context: ContextMode = "academic") -> str:
    fragments = judge_fragments(source, context)
    body = "\n\n".join(f"[P{fragment.n}]\n{fragment.text}" for fragment in fragments)
    return PROMPT.replace("{count}", str(len(fragments))).replace("{fragments}", body)


def parse_answer(text: str) -> dict[int, int]:
    """Уровни из ответа судьи: номер фрагмента → 0–3. Повтор номера берёт первый ответ."""
    return parse_run(text, 0).levels


class JudgeRun(Record):
    """Один прогон судьи: уровни и всё, что в ответе не так."""

    levels: dict[int, int]
    # Номера, которых нет среди фрагментов документа.
    extra: list[int]
    # Номера, на которые судья ответил больше одного раза, и те из них, где ответы разные.
    duplicates: list[int]
    conflicts: list[int]
    # Строки «# ключ: значение» в начале ответа: модель, температура и прочее, что записал пользователь.
    meta: dict[str, str]


def parse_run(text: str, count: int) -> JudgeRun:
    """Прогон судьи для документа из `count` фрагментов (0 — номера не сверяются)."""
    answers: dict[int, list[int]] = {}
    for match in ANSWER_RE.finditer(text):
        answers.setdefault(int(match[1]), []).append(LEVELS[match[2].lower()])
    known = {number: levels for number, levels in answers.items() if not count or 1 <= number <= count}
    return JudgeRun(
        levels={number: levels[0] for number, levels in known.items()},
        extra=sorted(number for number in answers if number not in known),
        duplicates=sorted(number for number, levels in known.items() if len(levels) > 1),
        conflicts=sorted(number for number, levels in known.items() if len(set(levels)) > 1),
        meta={match[1]: match[2] for match in RUN_META_RE.finditer(text)},
    )


def auc(scores: list[float], labels: list[bool]) -> float:
    """ROC AUC с ничьими пополам; nan, если нет одного из классов."""
    positives = [score for score, label in zip(scores, labels, strict=True) if label]
    negatives = [score for score, label in zip(scores, labels, strict=True) if not label]
    if not positives or not negatives:
        return math.nan
    wins = sum((positive > negative) + 0.5 * (positive == negative) for positive in positives for negative in negatives)
    return wins / (len(positives) * len(negatives))


class JudgedFragment(Record):
    n: int
    line: int
    end_line: int
    words: int
    # Уровень в каждом прогоне; None — судья этот фрагмент пропустил.
    levels: list[int | None]
    # Средний уровень по прогонам, где ответ есть.
    mean: float | None
    # Вероятность ИИ по оценке `antiplagiat` для сравнения.
    detector: float
    preview: str
    # Доля слов фрагмента, подсвеченных отчётом (если передана разметка).
    marked: float | None = None


class JudgeReport(Record):
    runs: int
    # Версия задания (PROMPT_VERSION) и отпечаток документа: по ним сводку можно повторить.
    prompt_version: int
    document_hash: str
    fragments: list[JudgedFragment]
    # Номера фрагментов без ответа ни в одном прогоне.
    missing: list[int]
    # Доля слов во фрагментах со средним уровнем не ниже medium (1,5) и не ниже low (0,5), 0–100.
    share_medium: float
    share_low: float
    # Средняя доля фрагментов, на которых два прогона дали один уровень; None при одном прогоне.
    agreement: float | None = None
    # ROC AUC среднего уровня судьи и вероятности детектора против разметки отчёта.
    auc: float | None = None
    detector_auc: float | None = None
    # По прогону: доля фрагментов с ответом, лишние номера, повторы, повторы с разными уровнями, сведения о прогоне.
    coverage: list[float] = Field(default_factory=list)
    extra: list[list[int]] = Field(default_factory=list)
    duplicates: list[list[int]] = Field(default_factory=list)
    conflicts: list[list[int]] = Field(default_factory=list)
    run_meta: list[dict[str, str]] = Field(default_factory=list)
    # Фрагменты, где прогоны разошлись на DISAGREEMENT уровня и больше.
    disagreements: list[int] = Field(default_factory=list)


def judge_score(
    source: str, answers: list[str], marked: list[str] | None = None, context: ContextMode = "academic"
) -> JudgeReport:
    report = antiplagiat(source, context)
    parsed = [parse_run(answer, len(report.fragments)) for answer in answers]
    runs = [run.levels for run in parsed]
    marks = marked_fragments(source, marked, context) if marked is not None else None
    fragments: list[JudgedFragment] = []
    for number, fragment in enumerate(report.fragments, 1):
        levels = [run.get(number) for run in runs]
        answered = [level for level in levels if level is not None]
        fragments.append(
            JudgedFragment(
                n=number,
                line=fragment.line,
                end_line=fragment.end_line,
                words=fragment.words,
                levels=levels,
                mean=fixed(sum(answered) / len(answered), 2) if answered else None,
                detector=fragment.probability,
                marked=fixed(marks[number - 1].coverage, 2) if marks else None,
                preview=fragment.preview,
            )
        )
    total_words = sum(fragment.words for fragment in fragments) or 1

    def share(level: float) -> float:
        return fixed(
            sum(fragment.words for fragment in fragments if fragment.mean is not None and fragment.mean >= level)
            * 100
            / total_words,
            1,
        )

    pairs = list(combinations(range(len(runs)), 2))
    agreement = None
    if pairs and fragments:
        same = [
            sum(
                1
                for fragment in fragments
                if fragment.levels[i] is not None and fragment.levels[i] == fragment.levels[j]
            )
            / len(fragments)
            for i, j in pairs
        ]
        agreement = fixed(sum(same) / len(same), 3)
    judge_auc = detector_auc = None
    if marks:
        rated = [(fragment, mark) for fragment, mark in zip(fragments, marks, strict=True) if fragment.mean is not None]
        labels = [mark.marked for _, mark in rated]
        judge_value = auc([fragment.mean or 0 for fragment, _ in rated], labels)
        detector_value = auc([fragment.detector for fragment in fragments], [mark.marked for mark in marks])
        judge_auc = None if math.isnan(judge_value) else fixed(judge_value, 3)
        detector_auc = None if math.isnan(detector_value) else fixed(detector_value, 3)
    total = len(fragments) or 1
    return JudgeReport(
        runs=len(runs),
        prompt_version=PROMPT_VERSION,
        document_hash=doc_fingerprint(source),
        coverage=[fixed(len(run.levels) / total, 3) for run in parsed],
        extra=[run.extra for run in parsed],
        duplicates=[run.duplicates for run in parsed],
        conflicts=[run.conflicts for run in parsed],
        run_meta=[run.meta for run in parsed],
        disagreements=[
            fragment.n
            for fragment in fragments
            if len(answered := [level for level in fragment.levels if level is not None]) > 1
            and max(answered) - min(answered) >= DISAGREEMENT
        ],
        fragments=fragments,
        missing=[fragment.n for fragment in fragments if fragment.mean is None],
        share_medium=share(1.5),
        share_low=share(0.5),
        agreement=agreement,
        auc=judge_auc,
        detector_auc=detector_auc,
    )

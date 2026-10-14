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

from aiw_ru.antiplagiat import antiplagiat, marked_fragments
from aiw_ru.compat import fixed
from aiw_ru.types import ContextMode, Record

LEVELS: dict[str, int] = {"none": 0, "low": 1, "medium": 2, "high": 3}

# Задание судье. Текст тот же, что в references/judge.md скилла (это проверяет тест).
PROMPT = """Ты детектор ИИ-текста для академических работ на русском, как модуль ИИ-детекции в системах проверки вузов.

Оцени текст ниже. Не открывай файлы, не запускай команды и программы, не ищи в интернете: оценка только твоим суждением как читателя-эксперта.

Текст разбит на фрагменты [P1]…[P{count}]. Это части одного документа. Для каждого фрагмента оцени, похож ли он на текст, сгенерированный языковой моделью (в том числе сгенерированный и затем отредактированный человеком). Шкала уверенности:

- high: такие признаки встречаются менее чем в 1 % текстов того же жанра, написанных людьми;
- medium: менее чем в 5 %;
- low: менее чем в 10 %;
- none: обычный человеческий текст.

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
    levels: dict[int, int] = {}
    for match in ANSWER_RE.finditer(text):
        levels.setdefault(int(match[1]), LEVELS[match[2].lower()])
    return levels


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


def judge_score(
    source: str, answers: list[str], marked: list[str] | None = None, context: ContextMode = "academic"
) -> JudgeReport:
    report = antiplagiat(source, context)
    runs = [parse_answer(answer) for answer in answers]
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
    return JudgeReport(
        runs=len(runs),
        fragments=fragments,
        missing=[fragment.n for fragment in fragments if fragment.mean is None],
        share_medium=share(1.5),
        share_low=share(0.5),
        agreement=agreement,
        auc=judge_auc,
        detector_auc=detector_auc,
    )

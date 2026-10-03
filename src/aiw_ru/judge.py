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
        JudgeFragment(n=k, line=f.line, end_line=f.end_line, words=f.words, text=source[f.start : f.end].strip())
        for k, f in enumerate(antiplagiat(source, context).fragments, 1)
    ]


def judge_prompt(source: str, context: ContextMode = "academic") -> str:
    frs = judge_fragments(source, context)
    body = "\n\n".join(f"[P{f.n}]\n{f.text}" for f in frs)
    return PROMPT.replace("{count}", str(len(frs))).replace("{fragments}", body)


def parse_answer(text: str) -> dict[int, int]:
    """Уровни из ответа судьи: номер фрагмента → 0–3. Повтор номера берёт первый ответ."""
    out: dict[int, int] = {}
    for m in ANSWER_RE.finditer(text):
        out.setdefault(int(m[1]), LEVELS[m[2].lower()])
    return out


def auc(scores: list[float], labels: list[bool]) -> float:
    """ROC AUC с ничьими пополам; nan, если нет одного из классов."""
    pos = [s for s, y in zip(scores, labels, strict=True) if y]
    neg = [s for s, y in zip(scores, labels, strict=True) if not y]
    if not pos or not neg:
        return math.nan
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


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
    runs = [parse_answer(a) for a in answers]
    marks = marked_fragments(source, marked, context) if marked is not None else None
    frs: list[JudgedFragment] = []
    for k, f in enumerate(report.fragments, 1):
        levels = [r.get(k) for r in runs]
        got = [x for x in levels if x is not None]
        frs.append(
            JudgedFragment(
                n=k,
                line=f.line,
                end_line=f.end_line,
                words=f.words,
                levels=levels,
                mean=fixed(sum(got) / len(got), 2) if got else None,
                detector=f.probability,
                marked=fixed(marks[k - 1].coverage, 2) if marks else None,
                preview=f.preview,
            )
        )
    total = sum(f.words for f in frs) or 1

    def share(level: float) -> float:
        return fixed(sum(f.words for f in frs if f.mean is not None and f.mean >= level) * 100 / total, 1)

    pairs = list(combinations(range(len(runs)), 2))
    agreement = None
    if pairs and frs:
        same = [
            sum(1 for f in frs if f.levels[i] is not None and f.levels[i] == f.levels[j]) / len(frs) for i, j in pairs
        ]
        agreement = fixed(sum(same) / len(same), 3)
    judge_auc = detector_auc = None
    if marks:
        rated = [(f, m) for f, m in zip(frs, marks, strict=True) if f.mean is not None]
        labels = [m.marked for _, m in rated]
        a = auc([f.mean or 0 for f, _ in rated], labels)
        d = auc([f.detector for f in frs], [m.marked for m in marks])
        judge_auc = None if math.isnan(a) else fixed(a, 3)
        detector_auc = None if math.isnan(d) else fixed(d, 3)
    return JudgeReport(
        runs=len(runs),
        fragments=frs,
        missing=[f.n for f in frs if f.mean is None],
        share_medium=share(1.5),
        share_low=share(0.5),
        agreement=agreement,
        auc=judge_auc,
        detector_auc=detector_auc,
    )

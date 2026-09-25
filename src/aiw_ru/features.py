"""Признаки текста для модели LightGBM: один и тот же код при обучении и при проверке.

Признаки описывают язык: какие правила детектора сработали, длина предложений
и слов, разнообразие словаря, пунктуация, частоты служебных слов. Оформление
(переводы строк, списки, markdown, вид тире и кавычек, ё) и длина текста в
признаки не входят: в корпусах они говорят о том, откуда взят человеческий
текст, а не о том, кто его написал.

Модель обучена на признаках определённой версии. Меняешь состав или порядок
признаков — подними FEATURES_VERSION и переобучи модель.
"""

from __future__ import annotations

import re
from collections import Counter

from aiw_ru.detect import TYPE_LABELS, analyze
from aiw_ru.types import AnalysisResult, ContextMode

FEATURES_VERSION = 1

# Режим детектора, в котором считаются признаки, при обучении и при проверке.
CONTEXT: ContextMode = "general"

# Правила о словах и фразах; типографика и разметка сюда не входят.
RULES: tuple[str, ...] = tuple(
    (
        "tier1 tier1-clarity tier2-cluster tier3 phrase3 filler transition not-x-but-y uniform-sentences calque "
        "template vague-endorsement generic-conclusion future-narrative chatbot hollow-intensifier "
        "unmeasured-claim significance vague-attribution"
    ).split()
)

# Служебные и частые слова: их частоты различают авторов лучше редких слов.
FUNCTION_WORDS: tuple[str, ...] = tuple(
    (
        "и в не на что с как а но это по к же ли бы то вот уж ведь даже еще так который которые которая "
        "является данный также однако кроме того более важно позволяет обеспечивает именно просто очень "
        "я мы вы ты он она они мой наш свой этот там тут здесь потом когда если чтобы или да нет ну вообще "
        "кстати типа может можно нужно будет был была были"
    ).split()
)

PUNCTUATION: dict[str, str] = {
    "comma": ",",
    "colon": ":",
    "semicolon": ";",
    "paren": "(",
    "question": "?",
    "exclam": "!",
}

# Буквы и цифры, внутри слова дефис и апостроф.
WORD_RE = re.compile(r"[^\W_]+(?:[-’'][^\W_]+)*")

FEATURE_NAMES: tuple[str, ...] = (
    *(f"rule:{r}" for r in RULES),
    "sentence-mean",
    "sentence-cv",
    "mattr",
    "word-length",
    "long-words",
    *PUNCTUATION,
    *(f"word:{w}" for w in FUNCTION_WORDS),
)


_PUNCTUATION_NAMES = {
    "comma": "запятых",
    "colon": "двоеточий",
    "semicolon": "точек с запятой",
    "paren": "открывающих скобок",
    "question": "вопросительных знаков",
    "exclam": "восклицательных знаков",
}

_DESCRIPTIONS = {
    "sentence-mean": "средняя длина предложения в словах",
    "sentence-cv": "разброс длины предложений: стандартное отклонение, делённое на среднюю длину",
    "mattr": "разнообразие словаря (MATTR): доля разных слов в скользящем окне из 100 слов",
    "word-length": "средняя длина слова в символах",
    "long-words": "слов из 10 символов и длиннее на 100 слов",
}


def describe(name: str) -> str:
    """Что значит признак из FEATURE_NAMES, по-русски."""
    kind, _, arg = name.partition(":")
    if kind == "rule":
        label = TYPE_LABELS.get(arg, arg)
        quoted = label if label.startswith("«") else f"«{label}»"
        return f"сработало правило детектора {quoted} (1 или 0)"
    if kind == "word":
        return f"слово «{arg}» на 100 слов, без учёта регистра, ё как е"
    if name in _PUNCTUATION_NAMES:
        return f"{_PUNCTUATION_NAMES[name]} на 100 слов"
    return _DESCRIPTIONS[name]


def features(text: str, result: AnalysisResult | None = None) -> list[float]:
    """Вектор в порядке FEATURE_NAMES; частоты — на 100 слов.

    `result` — ответ детектора на этот же текст в режиме CONTEXT; без него детектор
    запускается здесь.
    """
    r = result if result is not None else analyze(text, CONTEXT)
    found = {i.type for i in r.issues}
    words = WORD_RE.findall(text)
    n = max(len(words), 1)
    counts = Counter(w.lower().replace("ё", "е") for w in words)
    return [
        *(1.0 if rule in found else 0.0 for rule in RULES),
        float(r.stats.mean_sentence_length),
        float(r.stats.sentence_length_cv),
        float(r.stats.mattr),
        sum(map(len, words)) / n,
        100 * sum(len(w) >= 10 for w in words) / n,
        *(100 * text.count(ch) / n for ch in PUNCTUATION.values()),
        *(100 * counts[w] / n for w in FUNCTION_WORDS),
    ]

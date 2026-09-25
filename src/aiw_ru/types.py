"""Общие типы: режимы, находки, результат анализа."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasGenerator, BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

ContextMode = Literal["general", "academic", "technical", "social", "chat"]

CONTEXT_MODES: tuple[ContextMode, ...] = ("general", "academic", "technical", "social", "chat")

# Профили скилла (`--context` в SKILL.md) и их режимы детектора.
PROFILE_TO_MODE: dict[str, ContextMode] = {
    "vak": "academic",
    "docs": "technical",
    "blog": "general",
    "telegram": "social",
    "business-email": "general",
    "chat": "chat",
}

Severity = Literal["P0", "P1", "P2"]


class Record(BaseModel):
    """Данные, которые уходят в JSON: в Python поля в snake_case, в JSON — в camelCase."""

    model_config = ConfigDict(alias_generator=AliasGenerator(serialization_alias=to_camel), serialize_by_alias=True)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


class Issue(Record):
    # Тип находки, стабильный идентификатор категории.
    type: str
    # Идентификатор конкретного правила внутри категории (для словарей).
    rule: str
    severity: Severity
    # Найденный фрагмент исходного текста.
    text: str
    # Смещение в исходном тексте, в символах Python (кодовых точках).
    index: int
    line: int
    column: int
    # Что сделать.
    hint: str
    # Правки ради краткости (канцелярит 1Б) — совет по стилю, а не довод о машинном
    # авторстве. В оценку идут с малым весом.
    style_only: bool = Field(default=False, exclude_if=lambda v: not v)


class Stats(Record):
    words: int
    sentences: int
    paragraphs: int
    mean_sentence_length: float
    sentence_length_cv: float = Field(serialization_alias="sentenceLengthCV")
    paragraph_length_cv: float = Field(serialization_alias="paragraphLengthCV")
    mattr: float
    context_mode: ContextMode


class AnalysisResult(Record):
    # 0–100, чем выше, тем больше примет ИИ-стиля.
    score: int
    label: str
    issues: list[Issue]
    # Невидимые символы и подмена букв: документ выглядит подозрительным.
    suspicious: bool
    stats: Stats

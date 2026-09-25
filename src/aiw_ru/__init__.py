"""Приметы ИИ-стиля в русском тексте: детектор, оценка по фрагментам и проверка сохранности правки."""

from importlib.metadata import PackageNotFoundError, version

from aiw_ru.antiplagiat import (
    DEFAULT_MODEL,
    FEATURE_NAMES,
    AntiplagiatReport,
    CalibrationFile,
    DocSample,
    Fragment,
    Model,
    Sample,
    antiplagiat,
    balanced_accuracy,
    document_sample,
    fit,
    fit_share,
    label_fragments,
    predict,
    predict_share,
    share_error,
)
from aiw_ru.categories import JUDGMENT_ONLY, TYPE_TO_SECTION
from aiw_ru.detect import TYPE_LABELS, WEIGHTS, analyze, label_for, score_issues
from aiw_ru.lexicon import ALL_LEXICON, LexEntry, phrase
from aiw_ru.types import CONTEXT_MODES, PROFILE_TO_MODE, AnalysisResult, ContextMode, Issue, Record, Severity, Stats
from aiw_ru.validate import ValidationResult, Violation, validate

try:
    __version__ = version("aiw-ru")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "ALL_LEXICON",
    "CONTEXT_MODES",
    "DEFAULT_MODEL",
    "FEATURE_NAMES",
    "JUDGMENT_ONLY",
    "PROFILE_TO_MODE",
    "TYPE_LABELS",
    "TYPE_TO_SECTION",
    "WEIGHTS",
    "AnalysisResult",
    "AntiplagiatReport",
    "CalibrationFile",
    "ContextMode",
    "DocSample",
    "Fragment",
    "Issue",
    "LexEntry",
    "Model",
    "Record",
    "Sample",
    "Severity",
    "Stats",
    "ValidationResult",
    "Violation",
    "analyze",
    "antiplagiat",
    "balanced_accuracy",
    "document_sample",
    "fit",
    "fit_share",
    "label_for",
    "label_fragments",
    "phrase",
    "predict",
    "predict_share",
    "score_issues",
    "share_error",
    "validate",
]

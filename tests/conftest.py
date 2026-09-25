"""Общие пути и помощники для тестов."""

import os
from collections.abc import Callable
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
FIXTURES = ROOT / "tests" / "fixtures"

# Тесты импортируют onnxruntime напрямую, раньше aiw_ru.models. Его телеметрия включается при импорте,
# через десяток секунд шлёт данные и на выходе из pytest изредка роняет процесс в abort
# (recursive_mutex lock failed, код 134). Переменную наследуют и подпроцессы тестов.
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")


@pytest.fixture
def root() -> Path:
    """Корень репозитория."""
    return ROOT


@pytest.fixture
def fixtures_dir() -> Path:
    """Каталог с примерами текстов (corpus/ai, corpus/human)."""
    return FIXTURES


@pytest.fixture
def read_fixture() -> Callable[[str], str]:
    """Читает пример по пути от каталога примеров: `read_fixture("corpus/ai/blog.md")`."""

    def read(name: str) -> str:
        return (FIXTURES / name).read_text(encoding="utf-8")

    return read

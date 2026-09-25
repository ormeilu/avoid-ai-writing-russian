"""Общие пути и помощники для тестов."""

from collections.abc import Callable
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
FIXTURES = ROOT / "tests" / "fixtures"


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

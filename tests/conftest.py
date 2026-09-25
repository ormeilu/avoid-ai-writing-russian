"""Общие пути и помощники для тестов."""

import contextlib
import os
import sys
from collections.abc import Callable
from importlib.util import find_spec
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
FIXTURES = ROOT / "tests" / "fixtures"

# Тесты импортируют onnxruntime напрямую, раньше aiw_ru.models. Его телеметрия включается при импорте,
# через десяток секунд шлёт данные и на выходе из pytest изредка роняет процесс в abort
# (recursive_mutex lock failed, код 134). Переменную наследуют и подпроцессы тестов.
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

# На macOS у torch своя libomp, а LightGBM берёт её из Homebrew. Шаблонные функции OpenMP в libomp —
# слабые символы, и dyld связывает обе копии с той, что загрузилась первой; вторая копия с чужими
# функциями падает (SIGSEGV) или виснет, едва запустит потоки. В процессе pytest потоки OpenMP
# запускает только LightGBM: torch считает либо в подпроцессе (frozen_encoder_check), либо в один
# поток (фикстура one_thread в test_transformer.py). Поэтому LightGBM загружается здесь, раньше
# тестовых модулей, и порядок их сборки не важен.
if sys.platform == "darwin" and find_spec("torch"):
    with contextlib.suppress(ImportError, OSError):
        import lightgbm  # noqa: F401


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

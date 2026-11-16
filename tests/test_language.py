"""Допуск текста к моделям: кириллица ещё не значит русский язык.

Модели обучены на русском. Украинский, болгарский, сербский или казахский текст модель не оценивает,
даже если кириллицы в нём сто процентов. Неясный ввод (короткая строка без опознавательных букв,
абзац на двух языках) модель читает, но отмечает, что язык не определён уверенно.
Все примеры выдуманы.
"""

import pytest

from aiw_ru import models

RUSSIAN = "Это текст на русском языке. Результаты зависят от выбранного метода и объёма выборки."
UKRAINIAN = "Цей текст написано українською мовою. Дослідження показало, що результати залежать від обраного методу."
BULGARIAN = "Това е текст на български език. Изследването показва, че резултатите зависят от избрания метод."
SERBIAN = "Ово је текст на српском језику. Резултати зависе од изабране методе и величине узорка."
KAZAKH = "Бұл мәтін қазақ тілінде жазылған. Нәтижелер таңдалған әдіске байланысты."
BELARUSIAN_RUSSIAN = (
    "Гэта тэкст на беларускай мове, і ён распавядае пра склад і ўлік тавараў. "
    "Это текст на русском языке, и он тоже про склад и учёт товаров."
)
SCIENTIFIC = (
    "В работе рассмотрена модель BERT-base и метрика F1. Для обучения использован датасет SQuAD "
    "и библиотека PyTorch; точность составила 0,93 при размере батча 32. Выводы применимы к задачам NLP."
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param(RUSSIAN, "ru", id="русский"),
        pytest.param(SCIENTIFIC, "ru", id="русский научный с английскими терминами"),
        pytest.param(RUSSIAN + " Название города по-украински — «Київ».", "ru", id="русский с украинским именем"),
        pytest.param(UKRAINIAN, "other", id="украинский"),
        pytest.param(BULGARIAN, "other", id="болгарский"),
        pytest.param(SERBIAN, "other", id="сербский"),
        pytest.param(KAZAKH, "other", id="казахский"),
        pytest.param("This text is English only.", "other", id="английский"),
        pytest.param("2026 — 15 %", "other", id="без букв"),
        pytest.param("Кот спал на окне.", "uncertain", id="короткий без опознавательных букв"),
        pytest.param(BELARUSIAN_RUSSIAN, "uncertain", id="два языка в одном абзаце"),
    ],
)
def test_language(text: str, expected: str):
    """русский, другой язык или неясно"""
    assert models.language(text) == expected


def test_long_text_without_russian_letters_is_not_russian():
    """длинный кириллический текст без ы, э и ё — не русский: в русском ы около 2 % букв"""
    bulgarian_without_hard_sign = ("Резултатите зависят от избрания метод и от данните. " * 12).strip()
    assert models.language(bulgarian_without_hard_sign) == "other"


def test_is_russian_reads_uncertain_text():
    """неясный язык модель читает, как раньше; другой — нет"""
    assert models.is_russian("Кот спал на окне.")
    assert models.is_russian(RUSSIAN)
    assert not models.is_russian(UKRAINIAN)


def test_ukrainian_paragraph_is_cut_from_russian_text():
    """украинский абзац посреди русского текста модели не видят, как английскую аннотацию"""
    text = f"{RUSSIAN}\n\n{UKRAINIAN}\n\n{RUSSIAN}"
    assert models.foreign_lines(text) == [(3, 3)]
    visible = models.visible(text)
    assert "українською" not in visible
    assert visible.count("русском") == 2
    assert models.language(visible) == "ru"


def test_short_russian_paragraph_not_cut():
    """короткий русский абзац без ы, э и ё остаётся в тексте для модели"""
    assert models.foreign_lines(f"{RUSSIAN}\n\nКот спал на окне.") == []

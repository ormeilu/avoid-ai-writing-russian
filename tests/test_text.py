"""Подготовка текста: предложения, слова, нормализация, блоки, статистика."""

import time

import pytest

from aiw_ru.text import blocks, cv, line_col, mattr, mean, plural, prepare, sentences, words

ZW = chr(0x200B)
SHY = chr(0x00AD)
BOM = chr(0xFEFF)
ZWJ = chr(0x200D)


def ch(*codes: int) -> str:
    """Строка из кодов символов: так подмену букв видно в исходнике."""
    return "".join(map(chr, codes))


def texts(s: str) -> list[str]:
    return [x.text for x in sentences(s)]


# ─── sentences ──────────────────────────────────────────────────────────


def test_plain_split():
    """обычное разбиение"""
    assert texts("Первое. Второе! Третье? Четвёртое…") == ["Первое.", "Второе!", "Третье?", "Четвёртое…"]


def test_abbreviations_do_not_split():
    """сокращения не рвут предложение"""
    assert len(texts("См. рис. 3 и табл. 2, т. е. всё. Дальше.")) == 2
    assert len(texts("Это было в 1990 г. в Ростове. Потом нет.")) == 2


def test_initials_do_not_split():
    """инициалы не рвут предложение"""
    assert len(texts("Работу выполнил А. С. Иванов. Проверил другой.")) == 2


def test_lowercase_after_period_continues():
    """строчная буква после точки — не новое предложение"""
    assert len(texts("Версия 2. работает иначе.")) == 1


def test_quotes_and_dash_start_next_sentence():
    """кавычки и тире в начале следующего предложения"""
    assert len(texts("Он сказал. «Нет», — ответили ему.")) == 2


def test_manual_line_break_does_not_split():
    """ручной перенос внутри абзаца не разделяет"""
    assert texts("Эксперты\nсчитают, что это важно. Второе\nпредложение.") == [
        "Эксперты\nсчитают, что это важно.",
        "Второе\nпредложение.",
    ]


def test_blank_line_and_list_item_split():
    """пустая строка и пункт списка разделяют"""
    assert len(texts("строка без точки\n\nещё строка")) == 2
    assert len(texts("Список:\n- первый\n- второй\n1. третий")) == 4


def test_sentence_offsets_point_into_text():
    """смещения указывают в текст"""
    s = "Раз. Два три."
    for x in sentences(s, 100):
        assert s[x.start - 100 : x.end - 100] == x.text


def test_long_dot_run_is_linear():
    """длинная цепочка точек разбирается за линейное время"""
    t = time.perf_counter()
    assert len(texts(f"Текст {'.' * 50000} конец. Дальше.")) == 2
    assert (time.perf_counter() - t) * 1000 < 500


def test_sentence_word_count():
    """подсчёт слов"""
    assert sentences("Один два-три четыре.")[0].words == 3


# ─── words ──────────────────────────────────────────────────────────────


def test_hyphen_and_apostrophe_words():
    """дефисные слова и апострофы — одно слово"""
    assert words("кто-то из IT-отдела О’Нил") == ["кто-то", "из", "IT-отдела", "О’Нил"]


def test_numbers_are_words():
    """числа — слова"""
    assert words("в 2024 году") == ["в", "2024", "году"]


# ─── prepare ────────────────────────────────────────────────────────────


def test_yo_replaced_without_shift():
    """ё заменяется на е без сдвига"""
    p = prepare("Ёлка ещё")
    assert p.text == "Елка еще"
    assert len(p.text) == len(p.source)


def test_invisible_removed_offsets_restored():
    """невидимые символы удаляются, смещения восстанавливаются"""
    src = f"а{ZW}б{SHY}в"
    p = prepare(src)
    assert p.text == "абв"
    assert [i.index for i in p.invisible] == [1]
    assert p.soft_hyphens == [3]
    assert p.to_source[:3] == [0, 2, 4]


def test_bom_and_emoji_joiner_not_insertions():
    """BOM в начале и соединитель внутри эмодзи — не вставки"""
    assert prepare(f"{BOM}Текст").invisible == []
    assert prepare(f"{BOM}Текст").text == "Текст"
    assert prepare(f"Команда 👨{ZWJ}💻 и 🏳{chr(0xFE0F)}{ZWJ}🌈").invisible == []
    assert len(prepare(f"Ра{ZWJ}бота").invisible) == 1
    assert len(prepare(f"Текст{BOM} дальше").invisible) == 1


@pytest.mark.parametrize("code", [0x200E, 0x200F, 0x061C, 0x202A, 0x202E, 0x2066, 0x2069, 0x180E])
def test_bidi_and_mongolian_separator_are_insertions(code: int):
    """знаки направления текста и монгольский разделитель в русском слове — вставки"""
    p = prepare(f"Ра{chr(code)}бота готова.")
    assert p.text == "Работа готова."
    assert [i.index for i in p.invisible] == [2]


def test_bidi_next_to_rtl_and_mongolian_separator_inside_word_not_insertions():
    """метка направления рядом с ивритом и разделитель внутри монгольского слова законны"""
    hebrew = ch(0x05E9, 0x05DC, 0x05D5, 0x05DD)
    assert prepare(f"Он написал {hebrew}{chr(0x200E)} (2019).").invisible == []
    assert prepare(f"Цитата {chr(0x2067)}{hebrew}{chr(0x2069)} в тексте.").invisible == []
    mongolian = ch(0x182E, 0x1823, 0x180E, 0x1829)
    assert prepare(f"Слово {mongolian} по-монгольски.").invisible == []


def test_narrow_nbsp_is_not_insertion():
    """узкий неразрывный пробел U+202F — законная типографика чисел, а не вставка"""
    p = prepare(f"Итого 10{chr(0x202F)}000 рублей.")
    assert p.invisible == []
    assert len(p.text) == len(p.source)


def test_nonbreaking_hyphens_become_plain():
    """неразрывный дефис и U+2010 становятся обычным дефисом без сдвига"""
    src = f"по{chr(0x2011)}настоящему и кто{chr(0x2010)}то"
    p = prepare(src)
    assert p.text == "по-настоящему и кто-то"
    assert len(p.text) == len(src)
    assert words(p.text) == ["по-настоящему", "и", "кто-то"]


def test_latin_in_russian_word_fixed():
    """латиница в русском слове исправляется для поиска"""
    src = f"р{chr(0x61)}бота"
    p = prepare(src)
    assert p.text == "работа"
    assert len(p.homoglyphs) == 1


def test_pure_latin_and_legit_mix_untouched():
    """полностью латинские и законно смешанные слова не трогаются"""
    p = prepare("Python и Wi-Fi-модуль, IT-отдел")
    assert p.homoglyphs == []


def test_latin_abbreviation_hyphen_russian_word():
    """латинское сокращение через дефис с русским словом — не подмена"""
    p = prepare("HTTP-запрос на TCP-порт, PHP-скрипт ответил OK-кодом")
    assert p.homoglyphs == []
    assert "HTTP-запрос" in p.text


@pytest.mark.parametrize("s", ["PHPшник", "OKей", "CEOшный", "на Pythonе", "в Excelе", "iPhoneа", "в TeXе", "вPython"])
def test_latin_stem_russian_ending(s: str):
    """латинская основа с русским окончанием — не подмена"""
    assert prepare(s).homoglyphs == [], s


@pytest.mark.parametrize(
    "s",
    [
        f"ш{ch(0x69)}р{ch(0x69)}к",
        f"р{ch(0x69)}зних",
        f"F{ch(0x445)}G",
        f"m{ch(0x445)}n",
    ],
)
def test_ukrainian_i_and_times_sign(s: str):
    """латинская i в украинском слове и «х» как знак умножения — не подмена"""
    assert prepare(s).homoglyphs == [], s


@pytest.mark.parametrize("s", [f"К{ch(0xE1)}рмен", f"С{ch(0xE1)}нта-Фе"])
def test_acute_stress_not_spoof(s: str):
    """ударение латинской буквой с акутом — не подмена"""
    assert prepare(s).homoglyphs == [], s


SPOOFED = [
    f"P{ch(0x443)}thon",  # кириллическая «у» внутри латинского слова
    f"m{ch(0x43E)}del",  # кириллическая «о»
    f"{ch(0x420)}ython",  # кириллическая «Р» в начале латинского слова
    f"р{ch(0x61)}бота",  # латинская «a» внутри русского слова
    f"работ{ch(0x61)}",  # латинская «a» в конце
    f"{ch(0x70)}абота",  # латинская «p» в начале
    f"{ch(0x70, 0x61)}бота",  # две латинские буквы в начале
    f"{ch(0x65)}щ{ch(0x65)}",  # латиницы больше, но «щ» выдаёт русское слово
    f"{ch(0x4D)}Ч{ch(0x43)}",  # сокращение с латинскими M и C
    f"{ch(0x42)}Контакте",  # латинская B перед заглавной кириллицей
    f"{ch(0x43, 0x41)}ДЫ",  # заглавное слово: не сокращение с окончанием
    f"У{ch(0x4F, 0x50, 0x58, 0x4F)}ЛА",  # латиница внутри заглавного слова
]


def test_spoof_by_script_switch():
    """подмена по смене алфавитов внутри слова"""
    for s in SPOOFED:
        assert len(prepare(s).homoglyphs) == 1, s
    assert prepare(f"P{ch(0x443)}thon").text == "Python"
    assert prepare(f"р{ch(0x61)}бот{ch(0x61)}").text == "работа"


def test_spoof_inside_hyphen_part():
    """подмена внутри части дефисного слова находится"""
    p = prepare(f"HTTP-з{ch(0x61)}прос")
    assert len(p.homoglyphs) == 1
    assert p.text == "HTTP-запрос"


def test_masking_keeps_length_and_newlines():
    """маскирование сохраняет длину и переводы строк"""
    src = "---\na: 1\n---\nТекст `код` «цитата» [12] [@key] $x+1$ https://a.ru/b\n| т | а |\n> блок"
    p = prepare(src)
    assert len(p.prose) == len(src)
    assert len(p.prose.split("\n")) == len(src.split("\n"))
    for hidden in ["код", "цитата", "12", "@key", "x+1", "a.ru", "т | а", "блок", "a: 1"]:
        assert hidden not in p.prose, hidden
    assert "Текст" in p.prose


def test_code_block_hidden_in_no_code():
    """блок кода скрыт и в noCode, цитаты остаются"""
    src = "Текст «цитата»\n```\nкод\n```"
    p = prepare(src)
    assert "код" not in p.no_code
    assert "цитата" in p.no_code


# ─── blocks ─────────────────────────────────────────────────────────────


def kinds(s: str) -> list[str]:
    return [b.kind for b in blocks(prepare(s))]


def test_headings_prose_lists():
    """заголовки, проза, списки"""
    assert kinds("# Заголовок\nАбзац один.\n\n- пункт\n- пункт\n\nАбзац два.") == [
        "heading",
        "prose",
        "list",
        "prose",
    ]


def test_table_quote_code():
    """таблица, цитата и код распознаются"""
    assert kinds("| a | b |\n|---|---|\n\n> цитата\n\n```\nx\n```") == ["table", "quote", "code"]


def test_numbered_list():
    """нумерованный список"""
    assert kinds("1. раз\n2. два") == ["list"]


# ─── статистика ─────────────────────────────────────────────────────────


def test_mean_and_cv():
    """mean и cv"""
    assert mean([2, 4, 6]) == 4
    assert cv([5, 5, 5]) == 0
    assert cv([1]) == 0
    assert cv([1, 9]) > 1


def test_mattr_repeats_lower_diversity():
    """MATTR: повторы снижают разнообразие"""
    rich = [f"слово{i}" for i in range(300)]
    poor = [f"слово{i % 20}" for i in range(300)]
    assert mattr(rich) == 1
    assert mattr(poor) < 0.3
    assert mattr([]) == 0


def test_line_col():
    """lineCol"""
    p = prepare("аб\nвг\nде")
    assert line_col(p.line_starts, 0) == (1, 1)
    assert line_col(p.line_starts, 4) == (2, 2)
    assert line_col(p.line_starts, 6) == (3, 1)


def test_plural_groups_thousands():
    """согласование и разряды: «52 521 текст» с неразрывным пробелом, четырёхзначные слитно"""
    nbsp = chr(0xA0)
    assert plural(52521, "текст", "текста", "текстов") == f"52{nbsp}521 текст"
    assert plural(1000, "текст", "текста", "текстов") == "1000 текстов"
    assert plural(1234567, "слово", "слова", "слов") == f"1{nbsp}234{nbsp}567 слов"

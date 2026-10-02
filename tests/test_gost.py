"""Статья по ГОСТ после `pandoc файл.docx -t gfm --wrap=none` (issue #8): список литературы,
английский блок, ключевые слова и сведения об авторах не оцениваются как проза статьи.

Фикстуры в tests/fixtures/gost/ собраны gostdoc и pandoc, данные в них вымышлены:
- article.md: профиль journal-article-rgups-trudy, заголовок «Список литературы», жирные номера **1.**;
- article-rgups.md: профиль journal-article-rgups, записи без заголовка, номера 1\\. и тире « – ».
"""

import re
from collections.abc import Callable

import pytest

from aiw_ru import analyze, antiplagiat
from aiw_ru.text import line_col, prepare, service_ranges
from aiw_ru.types import ContextMode

# Где в фикстуре кончается проза и начинается список литературы.
LIST_START = {"article.md": "# Список литературы", "article-rgups.md": "1\\. Тихонов"}
# Отпечатки и гигиена документа: в служебных частях их искать по-прежнему нужно.
HYGIENE = frozenset("chat-markup ai-url placeholder invisible-chars soft-hyphen homoglyph".split())
RECORDS = 20
# Номер записи в начале строки: **1.** (gostdoc), 1\\. (pandoc экранирует точку).
RECORD_START = re.compile(r"(?:\*\*\d{1,2}\.\*\*|\d{1,2}\\\.) ")


@pytest.fixture(params=sorted(LIST_START))
def article(request: pytest.FixtureRequest, read_fixture: Callable[[str], str]) -> tuple[str, str, str]:
    """Имя фикстуры, весь текст и тот же текст до списка литературы."""
    name: str = request.param
    text = read_fixture(f"gost/{name}")
    return name, text, text[: text.index(LIST_START[name])]


def types(text: str, context: ContextMode = "academic") -> list[str]:
    return [i.type for i in analyze(text, context).issues]


def service_lines(text: str) -> set[int]:
    """Номера строк исходника, занятых служебными частями."""
    p = prepare(text)
    lines: set[int] = set()
    for a, b in p.service:
        first = line_col(p.line_starts, p.to_source[a])[0]
        last = line_col(p.line_starts, p.to_source[b - 1])[0]
        lines.update(range(first, last + 1))
    return lines


# ─── фикстура со списком из 20 записей ──────────────────────────────────


def test_fixture_has_twenty_records(article: tuple[str, str, str]):
    """в фикстуре ровно двадцать записей, и все они служебные"""
    _, text, _ = article
    numbered = [i + 1 for i, line in enumerate(text.splitlines()) if RECORD_START.match(line)]
    assert len(numbered) == RECORDS
    assert set(numbered) <= service_lines(text)


def test_score_does_not_grow_with_list(article: tuple[str, str, str]):
    """оценка по статье со списком литературы не выше, чем по той же статье без него"""
    _, text, body = article
    assert analyze(text, "academic").score <= analyze(body, "academic").score


def test_no_findings_from_service_parts(article: tuple[str, str, str]):
    """рубленые фрагменты, жирное, кавычки и прочие приметы не берутся из служебных частей"""
    _, text, _ = article
    lines = service_lines(text)
    stray = [(i.type, i.line) for i in analyze(text, "academic").issues if i.type not in HYGIENE and i.line in lines]
    assert stray == []
    found = types(text)
    assert "staccato" not in found
    assert "bold-overuse" not in found


def test_service_parts_do_not_count_as_words(article: tuple[str, str, str]):
    """слова списка литературы и английского блока не входят в счётчик слов"""
    _, text, body = article
    assert analyze(text, "academic").stats.words - analyze(body, "academic").stats.words <= 2


def test_antiplagiat_skips_list_and_english_block(article: tuple[str, str, str]):
    """фрагменты antiplagiat не заходят в литературу и английский блок"""
    _, text, body = article
    lines = service_lines(text)
    report = antiplagiat(text)
    assert report.fragments
    for f in report.fragments:
        # Реквизит внутри шапки (дата поступления) может оказаться между абзацами фрагмента,
        # но начинаться и кончаться фрагмент в служебной части не должен.
        assert f.line not in lines, f.preview
        assert f.end_line not in lines, f.preview
    assert report.ai_share <= antiplagiat(body).ai_share


def test_unrecognized_records_would_fire():
    """проверка не пустая: две записи, которые не распознаются как список, дают рубленые фрагменты"""
    assert "staccato" in types("Мы сравнили методы.\n\nЖурнал. — 2023. — Vol. 23, № 20. — P. 8386.")
    assert "staccato" not in types(
        "Мы сравнили методы.\n\n"
        "1. Иванов И. И. Оценка // Журнал. — 2023. — № 4. — С. 12–19.\n\n"
        "2. Петров П. П. Сон // Журнал. — 2022. — № 5. — С. 1–5.\n\n"
        "3. Smith J. Drowsiness // Sensors. — 2023. — Vol. 23, № 20. — P. 8386.\n"
    )


def test_figure_captions_keep_straight_quotes_finding(read_fixture: Callable[[str], str]):
    """прямые кавычки в подписях рисунков остаются находкой: они стоят в самом docx"""
    text = read_fixture("gost/article.md")
    quotes = [i for i in analyze(text, "academic").issues if i.type == "straight-quotes"]
    assert [i.text.split(" ")[0] for i in quotes] == ['"Концепция', '"Задержка']
    assert not any(i.line in service_lines(text) for i in quotes)


# ─── типографика и выделение внутри служебных частей ────────────────────

BIBLIOGRAPHY = (
    "Мы сравнили три метода.\n\n"
    "## Список литературы\n\n"
    "1. Иванов И. И. «Оценка» // Журнал. — 2023. — № 4. — С. 12–19.\n\n"
    '2. Петров П. П. "Сон и бдительность" // Журнал. — 2022. — № 5. — С. 1–5.\n\n'
    "3. Smith J. “Drowsiness” // Sensors. — 2023. — Vol. 23, № 20. — P. 8386.\n"
)


def test_quotes_in_bibliography_not_flagged():
    """прямые и английские кавычки в записях литературы стоят как в источнике"""
    found = types(BIBLIOGRAPHY)
    assert "straight-quotes" not in found
    assert "english-quotes" not in found


def test_quotes_in_prose_still_flagged_next_to_bibliography():
    """кавычки в самом тексте проверяются, как и раньше"""
    found = types('Он назвал это "прорывом" и “успехом”.\n\n' + BIBLIOGRAPHY.split("\n\n", 1)[1])
    assert "straight-quotes" in found
    assert "english-quotes" in found


def test_fingerprints_in_bibliography_still_found():
    """отпечатки ИИ-инструментов в списке литературы остаются P0"""
    text = (
        BIBLIOGRAPHY
        + "\n4. Сидоров С. С. Сон. URL: https://example.org/a?utm_source=chatgpt.com — 2024. — № 1. — С. 4.\n"
    )
    issues = analyze(text, "academic").issues
    assert [i.type for i in issues if i.severity == "P0"] == ["ai-url"]


def test_access_date_in_brackets_is_not_placeholder():
    """«[Дата обращения: 12.03.2026]» в записи — не заглушка, а «[Дата]» — заглушка"""
    assert "placeholder" not in types("Сайт [Электронный ресурс]. [Дата обращения: 12.03.2026]")
    assert "placeholder" in types("Сайт [Электронный ресурс]. [Дата обращения: ДД.ММ.ГГГГ]")
    assert "placeholder" in types("Выпущено [Дата].")


# ─── жирные номера ──────────────────────────────────────────────────────


@pytest.mark.parametrize("number", ["**12.**", "**12)**", "**1.2.**"])
def test_bold_numbers_are_not_emphasis(number: str):
    """номер пункта, выделенный жирным в начале строки, не считается выделением"""
    items = "\n\n".join(f"{number} Пункт номер {k} со своим текстом." for k in range(1, 7))
    assert "bold-overuse" not in types(f"# Раздел\n\n{items}")


def test_real_bold_overuse_still_flagged():
    """жирный в тексте считается, и номера его не отменяют"""
    text = (
        "# Раздел\n\n**1.** Пункт.\n\n"
        "Мы **сравнили** три **метода** и **получили** четыре **результата** при **любых** условиях."
    )
    assert "bold-overuse" in types(text)


def test_bold_number_in_middle_of_line_counts():
    """жирное число посреди фразы остаётся выделением"""
    text = "# Раздел\n\nМы взяли **12.** образцов, **13.** замеров, **14.** серий и **15.** записей."
    assert "bold-overuse" in types(text)


# ─── служебные части: реквизиты и английский блок ───────────────────────


def parts(text: str) -> list[str]:
    return [text[a:b] for a, b in service_ranges(text)]


@pytest.mark.parametrize(
    "line",
    [
        pytest.param("УДК 656.25:004.93", id="udk"),
        pytest.param("**УДК 62-83**", id="udk-bold"),
        pytest.param("UDC 004.93", id="udc"),
        pytest.param("Для цитирования: Иванов И. И. Оценка // Журнал. — 2026. — № 4. — С. 10–17.", id="cite-ru"),
        pytest.param("For citation: Ivanov I. I. Assessment // Journal. 2026. No. 4. P. 10–17.", id="cite-en"),
        pytest.param("Поступила в редакцию 14.09.2026", id="received-ru"),
        pytest.param("Принята к публикации 02.10.2026", id="accepted-ru"),
        pytest.param("Received 12 March 2026; accepted 3 May 2026", id="received-en"),
        pytest.param("Received: March 12, 2026", id="received-colon"),
    ],
)
def test_front_lines_are_service(line: str):
    """УДК, «Для цитирования», даты поступления и принятия — служебные строки"""
    assert parts(f"{line}\n\nМы сравнили три метода.\n") == [line]


@pytest.mark.parametrize(
    "line",
    [
        pytest.param("Суть УДК в том, что классификатор общий.", id="udk-mid-sentence"),
        pytest.param("УДКа не бывает.", id="udk-suffix"),
        pytest.param("Для цитирования данных нужно разрешение автора.", id="cite-no-colon"),
        pytest.param("Поступила в редакцию жалоба от пассажира.", id="received-no-date"),
        pytest.param("Received signals were filtered before the analysis.", id="received-prose"),
        pytest.param("Принята к публикации версия исправлена.", id="accepted-no-date"),
    ],
)
def test_front_lines_do_not_touch_prose(line: str):
    """обычная проза с теми же словами служебной не становится"""
    assert parts(f"{line}\n\nМы сравнили три метода.\n") == []


def test_front_line_takes_one_line_only():
    """строка-реквизит без пустой строки после неё не забирает соседний абзац"""
    assert parts("УДК 656.25\nМы сравнили три метода.\n") == ["УДК 656.25"]


def test_english_head_before_abstract():
    """название, авторы и организации латиницей над «Abstract» входят в английский блок"""
    s = (
        "Мы сравнили методы.\n\n"
        "Two criteria compared on cab video\n\n"
        "I. K. Voronin, M. O. Zaitseva\n\n"
        "<sup>1</sup>Example Institute of Transport Ergonomics\n\n"
        "**Abstract.** We compare two criteria on cab video.\n\n"
        "**Keywords:** drowsiness, PERCLOS.\n"
    )
    first, second = parts(s)
    assert first.startswith("Two criteria")
    assert "Voronin" in first
    assert "Abstract" in first
    assert second.startswith("**Keywords:**")
    assert "Мы сравнили" not in first


def test_english_head_stops_at_russian_and_long_paragraphs():
    """русский абзац и длинный английский абзац над аннотацией в блок не попадают"""
    abstract = "**Abstract.** We compare two criteria on cab video.\n"
    assert parts(f"Русский абзац над аннотацией.\n\n{abstract}") == [abstract.strip()]
    long_english = " ".join(["word"] * 70)
    assert parts(f"{long_english}\n\n{abstract}") == [abstract.strip()]


def test_english_head_does_not_cross_previous_service_part():
    """английский блок не заходит в предыдущую служебную часть"""
    s = "Ключевые слова: сон, внимание.\n\nTitle of the paper\n\n**Abstract.** We compare two criteria on cab video.\n"
    first, second = parts(s)
    assert first == "Ключевые слова: сон, внимание."
    assert second.startswith("Title of the paper")


def test_pandoc_escaped_record_numbers():
    """pandoc экранирует точку и скобки после номера: 1\\. и \\[1\\]"""
    records = [
        "Иванов И. И. Оценка // Журнал. — 2023. — № 4. — С. 1–5.",
        "Петров П. П. Сон // Журнал. — 2022. — № 5. — С. 1–5.",
        "Сидоров С. С. Бдительность // Журнал. — 2021. — № 6. — С. 1–5.",
    ]
    for fmt in ("{k}\\. {r}", "\\[{k}\\] {r}", "**{k}.** {r}"):
        s = "Текст.\n\n" + "\n\n".join(fmt.format(k=k, r=r) for k, r in enumerate(records, 1)) + "\n\nПроза.\n"
        (part,) = parts(s)
        assert part.startswith(fmt.format(k=1, r=records[0])), fmt
        assert "Проза" not in part


def test_books_do_not_break_record_run():
    """книги по ГОСТ Р 7.0.100 («Москва : Маршрут, 2015. – 408 с.») распознаются как записи"""
    s = (
        "Текст.\n\n"
        "1\\. Тихонов, Р. Г. Безопасность / Р. Г. Тихонов. – Москва : Маршрут, 2015. – 408 с. – Текст : непосредств.\n\n"
        "2\\. Громов, И. С. Физиология / И. С. Громов. – Москва : Наука, 2018. – 256 с.\n\n"
        "3\\. Орлов, В. А. Методы / В. А. Орлов. – СПб. : БХВ, 2017. – 384 с.\n\n"
        "Проза.\n"
    )
    (part,) = parts(s)
    assert part.startswith("1\\. Тихонов")
    assert part.endswith("384 с.")


def test_numbered_list_with_seconds_is_not_bibliography():
    """нумерованный список с годами и секундами в теле статьи — не литература"""
    s = (
        "1. В 2020 году задержка составила 14 с. Дальше рост.\n"
        "2. В 2021 году задержка составила 12 с. Дальше спад.\n"
        "3. В 2022 году задержка составила 9 с. Конец.\n"
    )
    assert parts(s) == []


def test_plain_heading_belongs_to_service_part():
    """заголовок без решётки входит в служебную часть, заголовок Markdown остаётся заголовком"""
    record = "1\\. Иванов И. И. Оценка // Журнал. — 2023. — № 4. — С. 1–5.\n"
    (plain,) = parts(f"Текст.\n\nБиблиографический список\n\n{record}")
    assert plain.startswith("Библиографический список")
    (hashed,) = parts(f"Текст.\n\n## Библиографический список\n\n{record}")
    assert hashed.strip().startswith("1\\.")


def test_empty_section_is_not_service_part():
    """заголовок раздела, за которым сразу следующий заголовок, служебной части не даёт"""
    assert parts("Текст.\n\n## Литература\n\n## Приложение\n\nПроза.\n") == []

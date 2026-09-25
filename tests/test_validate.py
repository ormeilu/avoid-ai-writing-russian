"""Проверка сохранности правки."""

from pathlib import Path

import pytest

from aiw_ru import ValidationResult, Violation, validate

BEFORE = (
    "---\n"
    "title: Статья\n"
    "---\n"
    "\n"
    "# Введение\n"
    "\n"
    'Данный метод играет ключевую роль: точность 0.93 [12], см. "отчёт" и https://example.com/a?utm_source=chatgpt.com.\n'
    "\n"
    "```python\n"
    "x = 1\n"
    "```"
)


def test_allowed_edits_pass():
    """разрешённые правки проходят"""
    after = (
        BEFORE.replace("Данный метод играет ключевую роль", "Метод повышает полноту", 1)
        .replace("0.93", "0,93", 1)
        .replace('"отчёт"', "«отчёт»", 1)
        .replace("?utm_source=chatgpt.com", "", 1)
    )
    r = validate(BEFORE, after)
    assert r.violations == []
    assert r.ok is True


def test_changed_number_code_citation_caught():
    """изменённое число, код и ссылка ловятся"""
    after = BEFORE.replace("0.93", "0.95", 1).replace("x = 1", "x = 2", 1).replace("[12]", "[13]", 1)
    kinds = {v.kind for v in validate(BEFORE, after).violations}
    assert {"число", "код", "ссылка на литературу"} <= kinds


def test_yaml_header_and_headings():
    """YAML-шапка и заголовки"""
    after = BEFORE.replace("title: Статья", "title: Другая", 1).replace("# Введение", "## Введение", 1)
    kinds = {v.kind for v in validate(BEFORE, after).violations}
    assert {"yaml", "заголовки"} <= kinds


def test_more_issues_is_violation():
    """рост числа находок — нарушение"""
    r = validate("Метод работает.", "Давайте разберёмся: метод играет ключевую роль.")
    assert "находки" in [v.kind for v in r.violations]


# ── Факты: оговорки, имена, числа словами, месяцы, утверждения ──

CORPUS = Path(__file__).parent / "fixtures" / "corpus"
FACT_KINDS = {"оговорка", "имя", "число словами", "месяц", "утверждение"}


def kinds(r: ValidationResult) -> set[str]:
    return {v.kind for v in r.violations}


def warning_kinds(r: ValidationResult) -> set[str]:
    return {w.kind for w in r.warnings}


def assert_clean(before: str, after: str) -> None:
    r = validate(before, after)
    assert [v for v in r.violations if v.kind in FACT_KINDS] == []
    assert r.warnings == []


@pytest.mark.parametrize(
    ("before", "after"),
    [
        pytest.param("Метод обычно помогает при бликах.", "Метод помогает при бликах.", id="обычно"),
        pytest.param("Задержка около 37 мс на кадр.", "Задержка 37 мс на кадр.", id="около числа"),
        pytest.param("Цена от 5000 рублей.", "Цена 5000 рублей.", id="от числа"),
        pytest.param("Ошибка, по-видимому, связана с драйвером.", "Ошибка связана с драйвером.", id="по-видимому"),
        pytest.param("Порог не всегда подходит для кабины.", "Порог не подходит для кабины.", id="не всегда"),
        pytest.param("В среднем модель ошибается 3 раза.", "Модель ошибается 3 раза.", id="в среднем"),
        pytest.param(
            "Метод может сократить задержку в части сценариев.", "Метод сокращает задержку.", id="может в части"
        ),
    ],
)
def test_lost_hedge_is_violation(before: str, after: str):
    """снятая оговорка — нарушение"""
    r = validate(before, after)
    assert "оговорка" in kinds(r)
    assert r.ok is False


@pytest.mark.parametrize(
    ("before", "after"),
    [
        pytest.param(
            "Этот метод потенциально может снизить число ложных тревог.",
            "Этот метод может снизить число ложных тревог.",
            id="потенциально может",
        ),
        pytest.param(
            "Возможно, в перспективе модель сможет работать на телефоне.",
            "Модель может работать на телефоне.",
            id="возможно в перспективе",
        ),
        pytest.param(
            "Модель обычно ошибается на бликах, точность примерно 0,8.",
            "Модель, как правило, ошибается на бликах, точность около 0,8.",
            id="синонимы",
        ),
        pytest.param(
            "Порог в литературе колеблется от 0,15 до 0,40.", "Порог в литературе: 0,15–0,40.", id="интервал цифрами"
        ),
        pytest.param(
            "Обучение занимает от трёх до пяти часов.", "Обучение занимает три–пять часов.", id="интервал словами"
        ),
        pytest.param(
            "Порог в среднем равен 0,27. По-видимому, в кабине он другой.",
            "Порог в среднем равен 0,27; по-видимому, в кабине он другой.",
            id="две оговорки в одной стопке",
        ),
        pytest.param(
            "Метод работает. Таким образом, можно сделать вывод, что метод, вероятно, перспективен.",
            "Метод работает.",
            id="удалён пустой вывод",
        ),
        pytest.param(
            "Модель работает на Jetson. Если у вас остались вопросы, возможно, поможет документация.",
            "Модель работает на Jetson.",
            id="удалён след чат-бота",
        ),
    ],
)
def test_reduced_hedge_stack_passes(before: str, after: str):
    """стопка оговорок сведена к одной, синоним, интервал, удалённое пустое предложение — не нарушение"""
    assert_clean(before, after)


def test_lost_modal_alone_is_warning():
    """одно снятое «может» — предупреждение: без морфологии не отличить возможность от умения"""
    r = validate("Инструмент может проверять тексты на приметы.", "Инструмент проверяет тексты на приметы.")
    assert r.ok is True
    assert warning_kinds(r) == {"оговорка"}
    assert "«может»" in r.warnings[0].detail


@pytest.mark.parametrize(
    ("after", "name"),
    [
        pytest.param("Модель работает на ноутбуке с PyTorch.", "PyTorch", id="латиница"),
        pytest.param("Модель работает на ноутбуке под Воронежем.", "Воронежем", id="город"),
        pytest.param("Модель работает на ноутбуке по ГОСТ.", "ГОСТ", id="аббревиатура"),
        pytest.param("Модель работает на ноутбуке. RLDD тут ни при чём.", "RLDD", id="латиница в начале"),
    ],
)
def test_added_name_is_violation(after: str, name: str):
    """новое имя, название или аббревиатура — нарушение"""
    r = validate("Модель работает на ноутбуке.", after)
    assert Violation(kind="имя", detail=f"появилось: {name}") in r.violations
    assert r.ok is False


@pytest.mark.parametrize(
    ("before", "after"),
    [
        pytest.param("Москва закупила камеры.", "Камеры закупили в Москве.", id="падеж"),
        pytest.param("Решение принял Павел.", "Решение было за Павлом.", id="беглая гласная"),
        pytest.param("Требования ГОСТ описаны в разделе 5.", "Раздел 5 описывает требования ГОСТа.", id="ГОСТа"),
        pytest.param(
            "Искусственный интеллект помогает врачам.", "ИИ помогает врачам.", id="аббревиатура из слов исходника"
        ),
        pytest.param("Прошу прислать отчёт до пятницы.", "Прошу Вас прислать отчёт до пятницы.", id="вежливое Вы"),
        pytest.param("Итог: модель работает.", "Итог: Модель работает.", id="заглавная после двоеточия"),
        pytest.param(
            "Модель обучена на RLDD, порог подобран на валидации.",
            "Модель обучена на RLDD. Во-вторых, порог подобран на валидации.",
            id="после аббревиатуры с точкой",
        ),
        pytest.param(
            "Как пишет «Вестник РГУПС», метод работает.", "Метод работает, пишет «Вестник РГУПС».", id="цитата"
        ),
        pytest.param("# Установка\nНужен Python 3.12.", "# Установка\nНужен Python 3.12, больше ничего.", id="строка"),
    ],
)
def test_names_from_source_pass(before: str, after: str):
    """падежи, аббревиатуры из исходника, вежливое «Вы» и начала предложений — не новые имена"""
    assert_clean(before, after)


@pytest.mark.parametrize(
    ("after", "word"),
    [
        pytest.param("Мы записали двенадцать поездок.", "двенадцать", id="числительное"),
        pytest.param("Мы записали десятки поездок.", "десятки", id="десятки"),
        pytest.param("Мы записали тысячу поездок.", "тысячу", id="тысяча"),
        pytest.param("Мы записали пятую поездку.", "пятую", id="порядковое"),
    ],
)
def test_added_number_word_is_violation(after: str, word: str):
    """новое число словами — нарушение"""
    r = validate("Мы записали несколько поездок.", after)
    assert Violation(kind="число словами", detail=f"появилось: {word}") in r.violations


def test_small_number_word_is_warning():
    """мелкие числа словами — предупреждение: «один», «оба» идиоматичны"""
    r = validate("Мы сравнили подходы.", "Мы сравнили оба подхода, и один оказался быстрее.")
    assert r.ok is True
    assert warning_kinds(r) == {"число словами"}


@pytest.mark.parametrize(
    ("before", "after"),
    [
        pytest.param("Мы записали три поездки.", "Мы записали трёх машинистов за три поездки.", id="та же лемма"),
        pytest.param(
            "Во-первых, модель обучена на RLDD. В-третьих, тест шёл на ноутбуке.",
            "Модель обучена на RLDD. Во-вторых, тест шёл на ноутбуке.",
            id="во-вторых",
        ),
        pytest.param("Модель ошибается в 12 случаях.", "Модель ошибается в 12 случаях, двенадцать раз.", id="цифрами"),
    ],
)
def test_number_words_from_source_pass(before: str, after: str):
    """числительное из исходника (словом или цифрами) и «во-вторых» — не новое число"""
    assert_clean(before, after)


def test_added_month_is_violation():
    """новый месяц — нарушение, тот же месяц в другом падеже — нет"""
    assert "месяц" in kinds(validate("Запись продолжим осенью.", "Запись продолжим в октябре."))
    assert_clean("Журнал принимает статьи до 1 ноября.", "До 1 ноября журнал принимает статьи, в ноябре закрывает.")


@pytest.mark.parametrize(
    "after",
    [
        pytest.param("Мы сделали первый в стране сервис мониторинга.", id="первый"),
        pytest.param("Мы впервые сделали сервис мониторинга.", id="впервые"),
        pytest.param("Мы доказали, что сервис мониторинга работает.", id="доказали"),
        pytest.param("Мы сделали сервис мониторинга, он всегда работает.", id="всегда"),
    ],
)
def test_added_claim_is_warning(after: str):
    """слово-претензия — предупреждение, код выхода не меняется"""
    r = validate("Мы сделали сервис мониторинга.", after)
    assert warning_kinds(r) == {"утверждение"}
    assert r.ok is True


def test_claim_idioms_and_negations_pass():
    """устойчивые обороты и отрицания претензией не считаются"""
    assert_clean(
        "Мы по-прежнему не уверены в пороге. Прежде всего нужно собрать данные.",
        "Мы всё ещё не уверены в пороге, и он не всегда подходит. В первую очередь нужно собрать данные.",
    )


@pytest.mark.parametrize(
    ("before", "after"),
    [
        pytest.param("Нейросеть является методом распознавания.", "Нейросеть — метод распознавания.", id="является"),
        pytest.param(
            "В связи с тем, что сервер перегружен, ответ может задерживаться.",
            "Сервер перегружен, поэтому ответ может задерживаться.",
            id="в связи с тем",
        ),
        pytest.param(
            "Отличный вопрос! Порог выбирают по кривой ROC.", "Порог выбирают по кривой ROC.", id="угодливость"
        ),
        pytest.param(
            "Стоит отметить, что модель иногда ошибается на бликах.",
            "Модель иногда ошибается на бликах.",
            id="подушка",
        ),
        pytest.param("Данные передаются посредством протокола MQTT.", "Данные передаются по протоколу MQTT.", id="1Б"),
        pytest.param(
            "Приложение обладает возможностью работать без сети.", "Приложение может работать без сети.", id="1Б-2"
        ),
        pytest.param(
            "В современном мире искусственный интеллект играет ключевую роль в развитии здравоохранения. Данная "
            "технология открывает новые возможности для диагностики и лечения заболеваний. Стоит отметить, что "
            "внедрение ИИ осуществляется в рамках комплексного подхода к цифровой трансформации отрасли.",
            "ИИ в здравоохранении применяют для диагностики и лечения. Внедряют его вместе с остальной "
            "цифровизацией отрасли.",
            id="абзац блога",
        ),
    ],
)
def test_skill_style_edits_pass(before: str, after: str):
    """правки по таблицам замен скилла не дают фактовых нарушений и предупреждений"""
    assert_clean(before, after)


HUMAN = sorted((CORPUS / "human").glob("*.md"))

# Вычёркивания, которые не трогают факты: слово, оборот или предложение без оговорок, имён и чисел.
STRIKES = {
    "blog.md": [(" еле-еле", ""), ("Оказалось, что препроцессинг", "Препроцессинг"), ("Итог скромный: ", "")],
    "vak.md": [
        ("Отдельно заметим, что PERCLOS", "PERCLOS"),
        ("Этот результат не стоит переносить на кабину напрямую. ", ""),
    ],
    "email.md": [("Добрый день, Иван Петрович!", "Иван Петрович, добрый день!")],
    "telegram.md": [("Оказалось, её вешают", "Её вешают"), (", я третий день с этим воюю", "")],
    "docs.md": [(" Разбейте файл по главам.", "")],
    "chat.md": [("слушай, ", "")],
}


@pytest.mark.parametrize("path", HUMAN, ids=lambda p: p.name)
def test_human_corpus_unchanged_and_struck_pass(path: Path):
    """человеческий корпус: сам с собой и после вычёркиваний — без нарушений и предупреждений"""
    text = path.read_text(encoding="utf-8")
    r = validate(text, text)
    assert r.violations == []
    assert r.warnings == []
    struck = text
    for a, b in STRIKES[path.name]:
        assert a in struck
        struck = struck.replace(a, b, 1)
    assert_clean(text, struck)


def test_changed_inline_code_caught():
    """изменённый инлайн-код ловится"""
    before = "Запустите `pytest -q` из корня."
    assert kinds(validate(before, before.replace("pytest -q", "pytest -x"))) == {"инлайн-код"}


def test_facts_skip_protected_content():
    """оговорки и имена внутри цитаты сверяются точно, как цитата, а не как факты"""
    before = "Иванов пишет: «метод обычно работает в Москве»."
    r = validate(before, before.replace("обычно работает в Москве", "работает в Воронеже"))
    assert kinds(r) == {"цитата"}
    assert r.warnings == []

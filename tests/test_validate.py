"""Проверка сохранности правки."""

import json
from difflib import SequenceMatcher
from pathlib import Path

import pytest

from aiw_ru import ValidationResult, Violation, validate
from aiw_ru.antiplagiat import AfterParagraphs, _best_similarity, marked_from_heatmap

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


ARTICLE = (
    "Мы сравнили три метода оценки сонливости \\[1, 2\\].\n\n"
    "## Список литературы\n\n"
    "**1.** Иванов И. И. Оценка утомления // Транспорт. — 2023. — № 4. — С. 12–19.\n\n"
    "**2.** Smith J. Shelf monitoring // Sensors. — 2021. — Vol. 21, № 7. — P. 2417.\n"
)


def test_bibliography_is_protected():
    """правка записи в списке литературы — нарушение, даже если числа те же"""
    after = ARTICLE.replace("Оценка утомления", "Оценивание утомления", 1)
    assert "служебная часть" in {v.kind for v in validate(ARTICLE, after).violations}


def test_body_edit_keeps_bibliography():
    """правка прозы при нетронутом списке литературы проходит"""
    after = ARTICLE.replace("Мы сравнили три метода", "Сравнили три метода", 1)
    assert validate(ARTICLE, after).ok


def test_escaped_citations_from_pandoc():
    """ссылки \\[1, 2\\] после pandoc: снятое экранирование не нарушение, потерянная ссылка — нарушение"""
    assert validate(ARTICLE, ARTICLE.replace("\\[1, 2\\]", "[1, 2]", 1)).ok
    lost = validate(ARTICLE, ARTICLE.replace(" \\[1, 2\\]", "", 1))
    assert "ссылка на литературу" in {v.kind for v in lost.violations}


# ─── абзацы под красной подсветкой отчёта ───────────────────────────────

NBSP = chr(0xA0)
RED_KEPT = (
    "Ночная смена принимает больше всего возвратов, поэтому именно ночью на складе появляется "
    "большая часть расхождений в учёте."
)
RED_NBSP = (
    "Сканеры штрихкодов поставили на все ворота склада в прошлом году, а весы под полками появились только этой весной."
)
RED_EDITED = (
    "Сверка остатков по камерам занимает у кладовщика около часа в день и почти не требует ручного пересчёта коробок."
)
RED_REWRITTEN = (
    "У кладовщика сверка остатков по камерам отнимает около часа в день. Ручной пересчёт коробок почти не нужен."
)
PLAIN = "Обычный абзац без подсветки остаётся как есть и предупреждений не даёт, сколько бы раз его ни проверяли."


def paragraphs(*texts: str) -> str:
    return "\n\n".join(texts)


HEAT_MAP = json.dumps({"fragments": [{"level": 3, "text": text} for text in (RED_KEPT, RED_NBSP, RED_EDITED)]})


def test_validate_with_report_flags_unedited_red_paragraphs():
    """красный абзац без правки — предупреждение; снятые неразрывные пробелы правкой не считаются"""
    before = paragraphs(RED_KEPT, RED_NBSP.replace(" в ", f" в{NBSP}"), RED_EDITED, PLAIN)
    after = paragraphs(RED_KEPT, RED_NBSP, RED_REWRITTEN, PLAIN)

    result = validate(before, after, "academic", marked=marked_from_heatmap(HEAT_MAP))

    unedited = [warning for warning in result.warnings if warning.kind == "красный абзац без правки"]
    assert [warning.detail.split(":")[0] for warning in unedited] == ["строка 1", "строка 3"]
    assert result.ok


def test_best_similarity_skips_candidates_by_length_without_changing_result():
    """кандидаты, длина которых не даёт дойти до порога, не сравниваются, а результат тот же"""
    paragraph = RED_EDITED
    near = RED_EDITED.replace("постоянного", "постоянного,")
    candidates = AfterParagraphs([PLAIN, "Коротко.", near, RED_KEPT * 3, paragraph[:-1]])

    best = _best_similarity(paragraph, candidates)
    reference = max(SequenceMatcher(None, paragraph, text, autojunk=False).ratio() for text in candidates.texts)

    assert best == reference
    assert candidates.within_reach(paragraph) == [paragraph[:-1], near]


def test_validate_without_report_has_no_red_paragraph_warnings():
    """без --marked таких предупреждений нет"""
    text = paragraphs(RED_KEPT, PLAIN)

    result = validate(text, text, "academic")

    assert not [warning for warning in result.warnings if warning.kind == "красный абзац без правки"]


# ─── смысл: отрицания, единицы, роли ────────────────────────────────────


def warning_pairs(before: str, after: str) -> list[tuple[str, str]]:
    result = validate(before, after)
    return [(warning.kind, warning.detail) for warning in result.warnings]


@pytest.mark.parametrize(
    ("before", "after", "warning"),
    [
        pytest.param(
            "Прибор работает без перебоев.",
            "Прибор не работает без перебоев.",
            ("отрицание", "появилось: «не работает»"),
            id="появилось отрицание",
        ),
        pytest.param(
            "Склад не принял партию в пятницу.",
            "Склад принял партию в пятницу.",
            ("отрицание", "пропало: «не принял»"),
            id="пропало отрицание",
        ),
        pytest.param(
            "Длина кабеля 12 метров.",
            "Длина кабеля 12 сантиметров.",
            ("единица", "у числа 12: метров → сантиметров"),
            id="сменилась единица",
        ),
        pytest.param(
            "Утром кладовщик Олег передал ключ Нине.",
            "Утром кладовщица Нина передала ключ Олегу.",
            ("роль", "Олег → Олегу: имя в другом падеже, проверьте, кто что делает"),
            id="сменились роли",
        ),
    ],
)
def test_meaning_changes_are_warnings(before: str, after: str, warning: tuple[str, str]):
    """смена отрицания, единицы и падежа имени — предупреждения: код выхода прежний, но сверить надо"""
    result = validate(before, after)
    assert result.ok
    assert warning in warning_pairs(before, after)


@pytest.mark.parametrize(
    ("before", "after"),
    [
        pytest.param(
            "Это не просто инструмент учёта, а основа работы склада.",
            "Инструмент учёта — основа работы склада.",
            id="снято «не просто»",
        ),
        pytest.param(
            "Метод не работает на малых выборках, поэтому его не применяли.",
            "На малых выборках метод не работает. Поэтому его не применяли.",
            id="перестановка и деление предложения",
        ),
        pytest.param(
            "Кабель длиной 12 метров проложили вдоль стены.",
            "Вдоль стены проложили кабель длиной 12 м.",
            id="единица сокращена",
        ),
        pytest.param(
            "Скорость выросла на 15 % за 3 месяца.",
            "За 3 месяца скорость выросла на 15 процентов.",
            id="процент словом",
        ),
        pytest.param(
            "Утром кладовщик Олег передал ключ Нине.",
            "Ключ Нине утром передал кладовщик Олег.",
            id="порядок слов без смены падежа",
        ),
        pytest.param(
            "По данным Иванова и Петрова, метод точнее на малых выборках.",
            "Иванов и Петров показали, что метод точнее на малых выборках.",
            id="соавторы стали подлежащим",
        ),
        pytest.param(
            "Новые склады открыли в Казани и Самаре.",
            "Казань и Самара получили новые склады.",
            id="два города одним перечнем",
        ),
        pytest.param("", "", id="пустой текст"),
        pytest.param("Прибор работает.", "Прибор работает.", id="без изменений"),
    ],
)
def test_legitimate_rewrites_raise_no_meaning_warnings(before: str, after: str):
    """законная перестройка не даёт предупреждений об отрицании, единице и роли"""
    assert not [kind for kind, _ in warning_pairs(before, after) if kind in ("отрицание", "единица", "роль")]


# ─── перестановка чисел ─────────────────────────────────────────────────


def swaps(before: str, after: str) -> list[str]:
    return [warning.detail for warning in validate(before, after).warnings if warning.kind == "перестановка"]


@pytest.mark.parametrize(
    ("before", "after", "numbers"),
    [
        pytest.param(
            "Кира собрала 12 400 подписей, Лев 8 900.",
            "Кира собрала 8 900 подписей, Лев 12 400.",
            ["12 400", "8 900"],
            id="через запятую",
        ),
        pytest.param(
            "Кира собрала 12 400 подписей и Лев 8 900.",
            "Кира собрала 8 900 подписей и Лев 12 400.",
            ["12 400", "8 900"],
            id="через «и»",
        ),
        pytest.param(
            "Утром паром перевёз 64 машины. Вечером катер взял 18 пассажиров.",
            "Утром паром перевёз 18 машин. Вечером катер взял 64 пассажира.",
            ["64", "18"],
            id="между предложениями",
        ),
        pytest.param(
            "Итоги ярмарки: пасека продала 31 банку мёда, сыроварня 57 головок.",
            "Итоги ярмарки: пасека продала 57 банок мёда, сыроварня 31 головку.",
            ["31", "57"],
            id="после двоеточия",
        ),
        pytest.param(
            "В Твери посадили 140 лип, в Омске 260 клёнов, в Туле 380 берёз.",
            "В Твери посадили 260 лип, в Омске 380 клёнов, в Туле 140 берёз.",
            ["140", "260", "380"],
            id="цикл из трёх",
        ),
    ],
)
def test_swapped_numbers_are_warning(before: str, after: str, numbers: list[str]):
    """числа, обменявшиеся словами при них, дают предупреждение, а не нарушение"""
    result = validate(before, after)
    assert result.ok
    found = [warning for warning in result.warnings if warning.kind == "перестановка"]
    assert len(found) == 1
    assert [swap.numbers for swap in result.swapped_numbers] == [numbers]
    assert all(number in found[0].detail for number in numbers)


def test_swapped_numbers_in_json():
    """в JSON перестановка — поле swappedNumbers с числами и частями до и после"""
    result = validate("Кира собрала 12 400 подписей, Лев 8 900.", "Кира собрала 8 900 подписей, Лев 12 400.")
    swap = result.to_dict()["swappedNumbers"][0]
    assert swap == {
        "numbers": ["12 400", "8 900"],
        "before": "Кира собрала 12 400 подписей",
        "after": "Лев 12 400",
    }


@pytest.mark.parametrize(
    ("before", "after"),
    [
        pytest.param(
            "Кира собрала 12 400 подписей, Лев 8 900.",
            "Лев собрал 8 900 подписей, Кира 12 400.",
            id="части поменялись вместе с числами",
        ),
        pytest.param(
            "Утром паром перевёз 64 машины. Вечером катер взял 18 пассажиров.",
            "Вечером катер взял 18 пассажиров. Утром паром перевёз 64 машины.",
            id="предложения переставлены",
        ),
        pytest.param(
            "- Кира: 12 400 подписей\n- Лев: 8 900 подписей\n",
            "- Лев: 8 900 подписей\n- Кира: 12 400 подписей\n",
            id="пункты списка переставлены",
        ),
        pytest.param(
            "1. Кира собрала 12 400 подписей.\n2. Лев собрал 8 900 подписей.\n",
            "1. Лев собрал 8 900 подписей.\n2. Кира собрала 12 400 подписей.\n",
            id="нумерованный список переставлен",
        ),
        pytest.param(
            "Выручка ярмарки 90 тысяч, расходы 35 тысяч.",
            "При расходах 35 тысяч ярмарка принесла 90 тысяч выручки.",
            id="два числа слились в одну часть",
        ),
        pytest.param(
            "Кира собрала 300 подписей, Лев тоже 300.",
            "Лев собрал 300 подписей, Кира тоже 300.",
            id="одинаковые числа",
        ),
        pytest.param(
            "Паром перевёз 64 машины, катер взял 18 пассажиров.",
            "За день на пароме переправили 64 машины, а на катере 18 человек.",
            id="пересказ одной части",
        ),
        pytest.param(
            "Встреча 12.03.2026 в зале 4.",
            "В зале 4 встреча 12.03.2026.",
            id="дата и номер зала",
        ),
    ],
)
def test_reordering_with_anchors_is_not_swap(before: str, after: str):
    """перестановка частей вместе с их числами и словами — законная правка"""
    assert swaps(before, after) == []


@pytest.mark.parametrize(
    "path", HUMAN + sorted((CORPUS / "ai").glob("*.md")), ids=lambda p: p.parent.name + "/" + p.name
)
def test_reversed_sentences_are_not_swap(path: Path):
    """корпус с предложениями в обратном порядке внутри абзаца: числа идут со своими словами"""
    text = path.read_text(encoding="utf-8")
    paragraphs = []
    for paragraph in text.split("\n\n"):
        parts = [part for part in paragraph.replace("\n", " ").split(". ") if part]
        paragraphs.append(". ".join(reversed(parts)))
    assert swaps(text, "\n\n".join(paragraphs)) == []


# ─── числа в ссылках ────────────────────────────────────────────────────


def number_kinds(before: str, after: str) -> list[tuple[str, str]]:
    return [(v.kind, v.detail) for v in validate(before, after).violations if v.kind in ("число", "URL")]


@pytest.mark.parametrize(
    ("before", "after"),
    [
        pytest.param(
            "Расписание лежит по адресу https://example.org/timetable/2026.",
            "[Расписание на 2026](https://example.org/timetable/2026) лежит на сайте.",
            id="из адреса в текст ссылки",
        ),
        pytest.param(
            "[Расписание на 2026](https://example.org/timetable/2026) лежит на сайте.",
            "Расписание лежит по адресу https://example.org/timetable/2026.",
            id="из текста ссылки в адрес",
        ),
        pytest.param(
            "Смотрите [расписание на 2026](https://example.org/timetable/2026).",
            "Смотрите [расписание](https://example.org/timetable/2026).",
            id="число осталось только в адресе",
        ),
        pytest.param(
            "Расписание лежит в [архиве](timetable/2026.pdf).",
            "Расписание на 2026 лежит в [архиве](timetable/2026.pdf).",
            id="из относительной ссылки в текст",
        ),
        pytest.param(
            "Смотрите [расписание на 2026](timetable/2026.pdf).",
            "Смотрите [расписание](timetable/2026.pdf).",
            id="из текста в относительную ссылку",
        ),
    ],
)
def test_number_moved_between_link_and_text_passes(before: str, after: str):
    """число, перешедшее из адреса ссылки в её текст или обратно, не потеряно"""
    assert number_kinds(before, after) == []


# ─── повтор числа из исходника ──────────────────────────────────────────

FERRY = (
    "Паром перевозит за сезон 120 машин в сутки.\n\n"
    "Очередь у причала растёт к вечеру: переправа загружена сильнее, чем рассчитывали в 2024 году."
)


def repeats(before: str, after: str) -> list[str]:
    return [warning.detail for warning in validate(before, after).warnings if warning.kind == "повтор числа"]


def test_repeated_source_number_is_warning():
    """число из исходника, повторённое в другом абзаце ради конкретики, не повреждение, а предупреждение"""
    after = FERRY.replace("загружена сильнее", "при 120 машинах в сутки загружена сильнее")
    result = validate(FERRY, after)
    assert number_kinds(FERRY, after) == []
    assert result.ok
    assert repeats(FERRY, after) == ["120 встречается чаще, чем в исходнике: сверь, что оно стоит при тех же словах"]


def test_new_number_still_violation_next_to_repeat():
    """новое число по-прежнему повреждение, даже если рядом повторено число из исходника"""
    after = FERRY.replace("загружена сильнее", "при 120 машинах в сутки и 15 рейсах загружена сильнее")
    assert number_kinds(FERRY, after) == [("число", "появилось: 15")]
    assert repeats(FERRY, after) == ["120 встречается чаще, чем в исходнике: сверь, что оно стоит при тех же словах"]


def test_number_replaced_by_other_source_number_still_caught():
    """число, заменённое другим числом из исходника, пропало: повтор второго не прячет потерю первого"""
    after = FERRY.replace("120 машин", "2024 машин")
    assert ("число", "пропало или изменено: 120") in number_kinds(FERRY, after)


def test_changed_number_near_link_still_caught():
    """изменённое число в тексте ссылки и в относительном адресе по-прежнему ловится"""
    before = "Смотрите [расписание на 2026](https://example.org/timetable/2026) и [архив](timetable/2025.pdf)."
    in_text = before.replace("расписание на 2026", "расписание на 2027")
    in_target = before.replace("timetable/2025.pdf", "timetable/2024.pdf")
    assert ("число", "появилось: 2027") in number_kinds(before, in_text)
    assert ("число", "появилось: 2024") in number_kinds(before, in_target)

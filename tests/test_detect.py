"""Детектор: оценка, словари, защищённое содержимое, отпечатки, типографика, структура, режимы."""

import time
from collections.abc import Callable

from aiw_ru import analyze
from aiw_ru.types import ContextMode

ZW = chr(0x200B)
SHY = chr(0x00AD)
BOM = chr(0xFEFF)
ZWJ = chr(0x200D)


def types(text: str, context: ContextMode = "general") -> list[str]:
    return [i.type for i in analyze(text, context).issues]


# ─── оценка ─────────────────────────────────────────────────────────────


def test_template_ai_text_scores_high(read_fixture: Callable[[str], str]):
    """шаблонный ИИ-текст получает высокую оценку"""
    r = analyze(read_fixture("corpus/ai/blog.md"))
    assert r.score >= 70
    assert r.label == "сильный ИИ-стиль"


def test_human_prose_stays_clean(read_fixture: Callable[[str], str]):
    """живая проза остаётся чистой"""
    r = analyze(read_fixture("corpus/human/blog.md"))
    assert r.score < 15
    assert [i for i in r.issues if i.severity != "P2"] == []


def test_empty_input():
    """пустой ввод не падает"""
    assert analyze("").score == 0


# ─── словарь ────────────────────────────────────────────────────────────


def test_morphology_word_forms():
    """морфология: разные формы одного слова"""
    for s in ["Работа осуществляется быстро.", "Мы осуществили проверку.", "Осуществляя анализ, мы…"]:
        assert "tier1-clarity" in types(s), s


def test_yo_equals_ye():
    """ё и е равнозначны"""
    assert "lets" in types("Давайте разберёмся с этим.")


def test_cyrillic_word_boundaries():
    """границы слов в кириллице"""
    # «проявляется» не должно считаться «является»
    text = "Эффект проявляется слабо. " * 20
    assert not any(i.rule == "является" for i in analyze(text).issues)


def test_nested_matches_collapse():
    """вложенные совпадения схлопываются"""
    r = analyze("Модель играет ключевую роль в системе.")
    hits = [i for i in r.issues if i.type == "tier1"]
    assert len(hits) == 1
    assert hits[0].text == "играет ключевую роль"


def test_clerical_is_style_only():
    """канцелярит помечен как совет по стилю"""
    r = analyze("В рамках работы данный метод применяется в целях оценки.")
    clerical = [i for i in r.issues if i.type == "tier1-clarity"]
    assert len(clerical) > 0
    assert all(i.style_only for i in clerical)


def test_calques():
    """кальки"""
    assert "calque" in types("Это решение адресует проблему задержек.")
    assert "calque" in types("Мы проверяем логи на ежедневной основе.")


def test_kak_ii_not_cutoff():
    """«как ИИ» в обычной речи не оговорка об отсечке"""
    assert "cutoff" not in types("Фрагменты, помеченные как ИИ, переписывают.")
    assert "cutoff" in types("Я, как языковая модель, не могу знать.")


def test_vague_attribution_is_p0():
    """размытая ссылка на авторитет — P0"""
    r = analyze("Исследования показывают, что сон важен.")
    hit = next((i for i in r.issues if i.type == "vague-attribution"), None)
    assert hit is not None
    assert hit.severity == "P0"


# ─── защищённое содержимое ──────────────────────────────────────────────


def test_quotes_code_refs_not_checked():
    """цитаты, код и ссылки не проверяются"""
    text = (
        "Автор пишет: «Давайте разберёмся, это играет ключевую роль».\n"
        "\n"
        "```\n"
        "// Давайте погрузимся в код\n"
        "```\n"
        "\n"
        "Команда `давайте рассмотрим` описана в [12]."
    )
    assert analyze(text).issues == []


def test_long_word_does_not_hang():
    """длинное слово без пробелов не вешает детектор"""
    # Вырожденный вывод модели из LLMTrace: «версииconsumeconsume…» на 18 тысяч знаков.
    t = time.perf_counter()
    analyze(f"Возьмём автомобиль версии{'consume' * 1000} и дальше текст.")
    assert (time.perf_counter() - t) * 1000 < 1000


def test_yaml_header_not_checked():
    """YAML-шапка не проверяется"""
    assert analyze("---\ntitle: Давайте разберёмся\n---\nОбычный текст.").issues == []


# ─── технические отпечатки ──────────────────────────────────────────────


def test_invisible_and_homoglyph():
    """невидимые символы и подмена букв"""
    latin_a = chr(0x61)
    r = analyze(f"Ра{ZW}бота и р{latin_a}бота с латинской a.")
    assert r.suspicious is True
    assert {"invisible-chars", "homoglyph"} <= {i.type for i in r.issues}


def test_legit_script_mix_not_spoof():
    """законная смесь алфавитов не считается подменой"""
    assert analyze("Настроили Wi-Fi-роутер и IT-отдел.").suspicious is False
    r = analyze("Отправили HTTP-запрос на TCP-порт, PHP-скрипт ответил OK-кодом.")
    assert r.suspicious is False
    assert r.score < 15
    assert analyze("Наш PHPшник написал на Pythonе, CEOшный отдел сказал OKей.").suspicious is False
    assert analyze(f"Пишем на P{chr(0x443)}thon.").suspicious is True


def test_fingerprints_are_evidence_not_style():
    """подмена букв и невидимые символы — улики для проверки, а не ИИ-стиль"""
    r = analyze(f"Ра{ZW}бота и р{chr(0x61)}бота.")
    assert r.suspicious is True
    assert {"invisible-chars", "homoglyph"} <= {i.type for i in r.issues if i.severity == "P0"}
    assert r.score < 15


def test_invisible_count_agrees():
    """число невидимых символов согласовано"""

    def text(s: str) -> str | None:
        return next((i.text for i in analyze(s).issues if i.type == "invisible-chars"), None)

    assert text(f"Ра{ZW}бота.") == "1 невидимый символ"
    assert text(f"Ра{ZW}бо{ZW}та.") == "2 невидимых символа"


def test_soft_hyphen_is_polish_not_suspicious():
    """мягкий перенос — шлифовка, а не подозрительный документ"""
    r = analyze(
        f"Первую неделю мы потратили впустую. Модель на ноутбуке выдавала 40 кадров в секунду, а на Jet{SHY}son еле-еле 9."
    )
    assert r.suspicious is False
    assert r.score < 15
    hit = next((i for i in r.issues if i.type == "soft-hyphen"), None)
    assert hit is not None
    assert hit.severity == "P2"
    assert hit.text == "1 мягкий перенос"


def test_bom_and_emoji_joiner_not_suspicious():
    """BOM в начале и эмодзи с соединителем не делают документ подозрительным"""
    assert analyze(f"{BOM}Обычный текст.").issues == []
    assert analyze(f"Команда 👨{ZWJ}💻 довольна.").suspicious is False


def test_nonbreaking_hyphen_does_not_hide_word():
    """неразрывный дефис не прячет слово от словаря"""
    r = analyze(f"Это по{chr(0x2011)}настоящему важный шаг для команды.")
    assert "tier1" in [i.type for i in r.issues]


def test_chatgpt_citation_chars_and_bare_markers():
    """служебные символы ссылок ChatGPT и голые маркеры"""
    open_, sep, close = (chr(c) for c in (0xE200, 0xE202, 0xE201))
    assert "chat-markup" in types(f"Рост составил 12 %.{open_}cite{sep}turn0search3{close}")
    assert "chat-markup" in types("Рост составил 12 % turn0search12 за год.")
    assert "chat-markup" in types("Рост составил 12 % 【4:0†source】 за год.")
    assert "chat-markup" not in types("Поворот turn направо, потом search.")


def test_offsets_point_into_source_after_stripping():
    """смещения указывают в исходник даже после удаления невидимых символов"""
    src = f"{ZW}{ZW}Давайте разберёмся."
    hit = next((i for i in analyze(src).issues if i.type == "lets"), None)
    assert hit is not None
    assert src[hit.index : hit.index + 7] == "Давайте"


def test_placeholders_chat_markup_utm():
    """заглушки, разметка чатов и utm"""
    t = types("См. [Вставьте источник] citeturn0search0 https://x.ru/a?utm_source=chatgpt.com")
    assert {"placeholder", "chat-markup", "ai-url"} <= set(t)


# ─── типографика ────────────────────────────────────────────────────────


def test_grammatical_dash_vs_splice():
    """грамматическое тире не находка, тире-связка — находка"""
    assert "em-dash-splice" not in types("Цель работы — разработка метода.")
    splice = "Модель работает быстро — и это меняет всё. Мы проверили гипотезу — именно так и вышло. "
    assert "em-dash-splice" in types(splice)


def test_straight_and_english_quotes():
    """прямые и английские кавычки в русском тексте"""
    assert "straight-quotes" in types('Он назвал это "прорывом".')
    assert "english-quotes" in types("Он назвал это “прорывом”.")
    assert "straight-quotes" not in types("Он назвал это «прорывом».")


def test_decimal_point():
    """десятичная точка"""
    assert "decimal-point" in types("Точность составила 0.93 на тесте.")
    assert "decimal-point" not in types("Точность составила 0,93 на тесте.")
    assert "decimal-point" not in types("См. раздел 2.1 и версию 1.2.3.")


def test_title_case_heading():
    """Title Case в заголовке"""
    assert "title-case" in types("# Анализ Существующих Методов Распознавания\n\nТекст.")
    assert "title-case" not in types("# Анализ существующих методов\n\nТекст.")


# ─── структура ──────────────────────────────────────────────────────────


def test_not_x_but_y():
    """«не X, а Y»"""
    assert "not-x-but-y" in types("Это не просто инструмент, а целая платформа.")


def test_genitive_chain():
    """цепочка отглагольных существительных"""
    assert "genitive-chain" in types(
        "Цель — обеспечение повышения эффективности проведения мониторинга состояния машиниста."
    )


def test_transition_run():
    """переходы в начале абзацев подряд"""
    text = "Кроме того, первое.\n\nБолее того, второе.\n\nТаким образом, третье."
    assert "transition-run" in types(text)


def test_bullet_noun_phrase_list():
    """список из голых именных групп"""
    lst = "- Высокая точность\n- Низкая задержка\n- Надёжная работа\n- Удобный интерфейс\n- Гибкая настройка"
    assert "bullet-np-list" in types(lst)


def test_staccato():
    """рубленые фрагменты"""
    assert "staccato" in types(
        "Никаких предпочтений. Никакой эстетики. Ноль ностальгии. Старые правила исчезли навсегда, и это было заметно."
    )


# ─── режимы ─────────────────────────────────────────────────────────────


def test_academic_skips_required_formulas():
    """academic пропускает обязательные научные формулы"""
    text = "Научная новизна заключается в новом методе. Практическая значимость работы подтверждена."
    assert "tier2" not in types(text, "academic")


def test_chat_keeps_only_p0_and_calques():
    """chat оставляет только P0 и кальки"""
    r = analyze('Кстати, "это" играет ключевую роль, но это имеет смысл. Надеюсь, это поможет!', "chat")
    assert all(i.severity == "P0" or i.type == "calque" for i in r.issues)


def test_technical_keeps_legit_terms():
    """technical не отмечает законные термины"""
    assert "tier1" not in types("Экосистема пакетов npm огромна.", "technical")

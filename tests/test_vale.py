"""Стиль Vale (vale/AvoidAIWritingRussian): свежесть, манифест и сценарии в самом Vale.

Сценарии запускаются, только если команда `vale` есть в PATH; без неё они пропускаются.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from aiw_ru import WEIGHTS, analyze
from build_vale import (
    LEVELS,
    STATUSES,
    STYLE,
    STYLE_DIR,
    UNASSESSED,
    Rule,
    collect,
    differences,
    render,
    to_vale,
    write,
)

# Правило → (текст, где оно срабатывает, текст, где нет[, расширение файла]).
# Оба текста проверяются и в Vale, и в детекторе: находка должна быть в обоих или ни в одном.
CASES: dict[str, tuple[str, ...]] = {
    "ChatMarkup": ("Модель ответила citeturn0search1 и замолчала.", "Модель ответила по делу и замолчала."),
    # В Markdown Vale не видит адресов, поэтому сценарий в обычном тексте.
    "AiUrl": (
        "Источник: https://example.ru/stat?utm_source=chatgpt.com, данные за май.",
        "Источник: https://example.ru/stat?utm_source=newsletter, данные за май.",
        ".txt",
    ),
    "Placeholder": ("Подпись: [Ваше имя], отдел закупок.", "Подпись: Анна Петрова, отдел закупок."),
    "Cutoff": ("Как языковая модель, я не знаю прогноза.", "Как опытный синоптик, я не обещаю прогноза."),
    "Chatbot": ("Надеюсь, это поможет вам с ремонтом.", "Надеюсь, ремонт закончится к весне."),
    "ChatWrapper": ("Хотите, я перепишу письмо короче?", "Хотите, я заварю чай?"),
    "Sycophancy": ("Отличный вопрос! Начнём с корней.", "Вопрос простой: начнём с корней."),
    "VagueAttribution": ("Эксперты считают, что зима будет мягкой.", "Синоптик Петров ждёт мягкую зиму."),
    "Significance": ("Новый мост открывает новую эру в жизни города.", "Новый мост сократил дорогу до рынка."),
    # «Неуникальный»: граница слова по кириллице, а не по ASCII.
    "Tier1": ("Это уникальный способ варить кашу.", "Мы выбрали неуникальный, зато дешёвый способ."),
    "Tier1Clarity": ("Работа шла в рамках проекта.", "Картина висела внутри рамки."),
    "Calque": ("Мы имплементировали новый поиск.", "Мы реализовали новый поиск."),
    "Jargon": ("Мы собрали пайплайн для данных.", "Мы собрали конвейер для данных."),
    "Template": ("Как известно, вода мокрая.", "Как и прежде, вода мокрая."),
    "TransitionWarning": ("Подводя итог, мы закрыли сезон.", "Мы закрыли сезон без потерь."),
    "TransitionSuggestion": ("Кроме того, мы купили лодку.", "Кроме нас, лодку никто не купил."),
    "Filler": ("Стоит отметить, что лодка течёт.", "Стоит ли чинить лодку, решим завтра."),
    "HollowIntensifier": ("Это действительно важно для команды.", "Это действительно случилось вчера."),
    "Confidence": ("По сути, мы проиграли.", "По правде говоря, мы проиграли."),
    "HedgeStack": ("Этот метод потенциально может ускорить сборку.", "Этот метод ускорил сборку на треть."),
    "VagueEndorsement": ("Эту книгу стоит прочитать.", "Эту книгу я прочитал за вечер."),
    "FutureNarrative": ("Проект может стать одним из самых заметных.", "Проект может закрыться к весне."),
    # «Счёте» с «ё»: детектор заменяет её на «е», правило Vale ищет оба написания.
    "GenericConclusion": ("В конечном счёте всё зависит от погоды.", "Урожай зависит от погоды и почвы."),
    "Novelty": ("Есть приём, о котором никто не говорит.", "Есть приём, о котором писали в учебнике."),
    "UnmeasuredClaim": ("Модель показала высокую точность.", "Модель показала точность 91 процент."),
    "Lets": ("Давайте рассмотрим пример.", "Рассмотрим пример."),
    "Reasoning": ("Подумаем шаг за шагом.", "Подумаем над этим завтра."),
    "Acknowledgment": ("Вы спрашиваете о сроках доставки.", "Курьер спросил о сроках доставки."),
    "NarratedCandor": ("Буду честен, отчёт сырой.", "Отчёт пока сырой."),
    "Hook": ("Но есть один нюанс: мост закрыт.", "Мост закрыт до мая."),
    "SocialCloser": ("Сохраняйте себе, пригодится.", "Сохраняйте чеки до конца года."),
    "LaunchIntro": ("Знакомьтесь, Алиса, ваш новый помощник.", "Мы познакомились с Алисой на вечеринке."),
    # «Серьёзно» с «ё» и без разметки: звёздочки ремарок Vale в Markdown не видит.
    "FakeCasual": ("Мы закрыли спринт за день (да, серьёзно).", "Мы закрыли спринт за день, это правда."),
    "SpeculativeOpener": ("Представьте мир, в котором нет пробок.", "Представьте отчёт к пятнице."),
    "FalseConcession": ("Хотя результаты впечатляют, проблемы остаются.", "Хотя шёл дождь, мы дошли до вершины."),
    "MoralAdjective": ("Это честная метрика для модели.", "Это честный человек."),
    "RealInflation": ("Это даёт реальную пользу бизнесу.", "Это даёт пользу бизнесу."),
    "NotXButY": ("Это не просто библиотека, а целая платформа.", "Это библиотека для работы с датами."),
    "HyphenDash": ("Москва - столица России.", "Москва — столица России."),
    "StraightQuotes": ('Он сказал "привет" и ушёл.', 'Он набрал "hello" и ушёл.'),
    "EnglishQuotes": ("Он сказал “привет” и ушёл.", "Он сказал «привет» и ушёл."),
    "DecimalPoint": ("Точность составила 0.93 на тесте.", "Точность составила 0,93 на тесте."),
    "NegationChain": ("Без регистрации, без смс, без лишних шагов.", "Без регистрации вход не работает."),
}


def _rules() -> dict[str, Rule]:
    return {rule.name: rule for rule in collect().rules}


def test_committed_style_is_fresh(tmp_path: Path):
    """записанный стиль совпадает с таблицами детектора: после правки словаря его пересобирают"""
    problems = differences(STYLE_DIR)
    assert problems == [], "стиль Vale устарел, запустите `uv run python scripts/build_vale.py`: " + "; ".join(problems)
    write(tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(render())
    for path in tmp_path.iterdir():
        assert path.read_text(encoding="utf-8") == (STYLE_DIR / path.name).read_text(encoding="utf-8"), path.name


def test_write_removes_dropped_rules(tmp_path: Path):
    """правило, которого больше нет в таблицах, из папки стиля удаляется"""
    (tmp_path / "Dropped.yml").write_text("extends: existence\n", encoding="utf-8")
    write(tmp_path)
    assert not (tmp_path / "Dropped.yml").exists()


def test_manifest_covers_every_type():
    """манифест называет каждый тип детектора и не оставляет неразобранных"""
    manifest = json.loads((STYLE_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert [item["type"] for item in manifest["types"]] == list(WEIGHTS)
    assert {item["status"] for item in manifest["types"]} <= set(STATUSES)
    assert manifest["counts"][UNASSESSED] == 0, [i["type"] for i in manifest["types"] if i["status"] == UNASSESSED]
    assert sum(manifest["counts"].values()) == len(WEIGHTS)
    for item in manifest["types"]:
        # У правила без оговорок нет причины, у остальных она обязательна.
        assert bool(item["reason"]) == (item["status"] != "verified"), item["type"]
        assert bool(item["rules"]) == (item["status"] != "unsupported"), item["type"]


def test_every_rule_has_cases():
    """у каждого правила есть сценарий: срабатывание и чистый текст"""
    assert sorted(_rules()) == sorted(CASES)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("уникальн\\p{L}*", "уникальн\\p{L}*"),
        ("серьезн", "с[её]рь[её]зн"),
        ("[а-яе]", "[а-яеё]"),
        ("[^е\\n]", "[^е\\nё]"),
        ("[a-z]", "[a-z]"),
        ("(да|ну)", "(?:да|ну)"),
        ("(?<!\\p{L})(?:а)", "(?<!\\p{L})(?:а)"),
        ("\\(да\\)", "\\(да\\)"),
        ("[\\s" + chr(0xA0) + "]", "[\\s\\u00A0]"),
    ],
)
def test_to_vale(source: str, expected: str):
    """«е» ищет и «ё», группы незахватывающие, невидимые знаки экранированы"""
    assert to_vale(source) == expected


# ─── Сценарии в Vale ───────────────────────────────────────────────────
VALE = shutil.which("vale")
needs_vale = pytest.mark.skipif(VALE is None, reason="нет команды vale (https://vale.sh/docs/install)")


@pytest.fixture(scope="module")
def vale_alerts(tmp_path_factory: pytest.TempPathFactory) -> dict[str, set[str]]:
    """Прогоняет все сценарии одним запуском Vale: имя файла → сработавшие правила."""
    assert VALE is not None
    directory = tmp_path_factory.mktemp("vale")
    config = directory / ".vale.ini"
    config.write_text(
        f"StylesPath = {STYLE_DIR.parent.as_posix()}\nMinAlertLevel = suggestion\n\n"
        f"[*.{{md,txt}}]\nBasedOnStyles = {STYLE}\n",
        encoding="utf-8",
    )
    files: list[str] = []
    for name, (fires, clean, *rest) in CASES.items():
        extension = rest[0] if rest else ".md"
        for kind, text in (("fires", fires), ("clean", clean)):
            path = directory / f"{name}-{kind}{extension}"
            path.write_text(text + "\n", encoding="utf-8")
            files.append(path.name)
    result = subprocess.run(
        [VALE, "--config", str(config), "--output", "JSON", "--no-exit", *files],
        cwd=directory,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    report = json.loads(result.stdout or "{}")
    alerts: dict[str, set[str]] = {name: set() for name in files}
    for path, items in report.items():
        alerts[Path(path).name] = {item["Check"] for item in items}
    return alerts


@needs_vale
@pytest.mark.parametrize("name", sorted(CASES))
def test_vale_rule_matches_detector(name: str, vale_alerts: dict[str, set[str]]):
    """правило Vale срабатывает на том же примере, что и детектор, и молчит на чистом"""
    rule = _rules()[name]
    fires, clean, *rest = CASES[name]
    extension = rest[0] if rest else ".md"
    check = f"{STYLE}.{name}"

    def detector_fires(text: str) -> bool:
        return any(i.type == rule.type and i.severity == rule.severity for i in analyze(text).issues)

    assert check in vale_alerts[f"{name}-fires{extension}"], f"Vale молчит: {fires}"
    assert detector_fires(fires), f"детектор молчит: {fires}"
    assert check not in vale_alerts[f"{name}-clean{extension}"], f"Vale срабатывает: {clean}"
    assert not detector_fires(clean), f"детектор срабатывает: {clean}"


def test_levels_follow_severity():
    """уровень правила Vale следует уровню находки: P0 → error, P1 → warning, P2 → suggestion"""
    for name, rule in _rules().items():
        text = (STYLE_DIR / f"{name}.yml").read_text(encoding="utf-8")
        assert f"\nlevel: {LEVELS[rule.severity]}\n" in text, name

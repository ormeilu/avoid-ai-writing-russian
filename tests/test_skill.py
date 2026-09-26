"""Тесты самих скиллов: оформление SKILL.md, ссылки, согласованность
каталога с детектором и примеры из каталога.
"""

import re
import shlex
import textwrap
import tomllib
from pathlib import Path
from typing import Any

import pytest
import regex

from aiw_ru import JUDGMENT_ONLY, PROFILE_TO_MODE, TYPE_LABELS, TYPE_TO_SECTION, WEIGHTS, ContextMode, analyze, skills
from aiw_ru.cli import USAGE, main, parse

ROOT = Path(__file__).parent.parent
SKILLS = ROOT / "skills"
REFERENCES = SKILLS / "avoid-ai-writing-russian" / "references"
# Тематические файлы каталога примет; профили и задание проверяющему лежат отдельно.
CATALOG_FILES = sorted(p for p in REFERENCES.glob("*.md") if p.name not in ("profiles.md", "review.md", "models.md"))
PROFILES = REFERENCES / "profiles.md"
REVIEW = REFERENCES / "review.md"
# Руководство по моделям: его печатает и `aiw-ru models guide`.
MODELS_GUIDE = REFERENCES / "models.md"
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
AI_VAK = ROOT / "tests" / "fixtures" / "corpus" / "ai" / "vak.md"

# Вызов детектора из скилла: `uv run --project ../.. aiw-ru КОМАНДА …`, путь от папки скилла.
INVOCATION_RE = re.compile(r"^uv run --project (\S+)((?: --[a-z-]+(?: [a-z]+)?)*) aiw-ru (.+)$", re.MULTILINE)
# Флаги uv в вызовах скилла: модель ставится только через extra ml, групп разработки нет.
UV_FLAGS = {"--extra ml"}
# Команды из справки CLI: строки раздела «Команды:» вида `  scan [файл…]   …`.
COMMANDS = set(re.findall(r"^ {2}([a-z]+) ", USAGE.split("Параметры:")[0], re.MULTILINE))


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def parse_yaml(src: str) -> dict[str, Any]:
    """Подмножество YAML, которого хватает шапке скилла.

    `ключ: значение`, свёрнутый блок `>-` и вложенные карты с отступом. Строки в кавычках
    отдаются без кавычек, остальные значения — как есть, строкой.
    """
    data: dict[str, Any] = {}
    lines = src.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if not line.strip():
            continue
        key, sep, value = line.partition(":")
        assert sep, f"не пара «ключ: значение»: {line!r}"
        value = value.strip()
        block: list[str] = []
        while i < len(lines) and (lines[i].startswith(" ") or not lines[i].strip()):
            block.append(lines[i])
            i += 1
        if value == ">-":
            data[key] = " ".join(b.strip() for b in block if b.strip())
        elif not value:
            data[key] = parse_yaml(textwrap.dedent("\n".join(block)))
        else:
            assert not block, f"{key}: вложенный блок после значения"
            data[key] = value[1:-1] if len(value) > 1 and value[0] == value[-1] and value[0] in "\"'" else value
    return data


def frontmatter(text: str) -> tuple[dict[str, Any], str]:
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
    if not m:
        raise AssertionError("нет YAML-шапки")
    return parse_yaml(m[1]), m[2]


def test_parse_yaml_subset():
    """разбор шапки: свёрнутый блок, вложенные карты, кавычки"""
    src = 'name: x\ndescription: >-\n  раз\n  два\nmeta:\n  a: 1.0.0\n  inner:\n    b: "x y"\nlast: "1.0"'
    assert parse_yaml(src) == {
        "name": "x",
        "description": "раз два",
        "meta": {"a": "1.0.0", "inner": {"b": "x y"}},
        "last": "1.0",
    }


SKILL_DIRS = sorted(d.name for d in SKILLS.iterdir() if (d / "SKILL.md").exists())


def test_skills_found():
    """оба скилла на месте"""
    assert SKILL_DIRS == ["antiplagiat", "avoid-ai-writing-russian"]


# ─── каждый скилл ───────────────────────────────────────────────────────


@pytest.mark.parametrize("skill_dir", SKILL_DIRS)
class TestSkill:
    """каждый скилл"""

    def test_frontmatter_follows_agentskills_spec(self, skill_dir: str):
        """шапка по спецификации agentskills.io"""
        data, _ = frontmatter(read(SKILLS / skill_dir / "SKILL.md"))
        assert data["name"] == skill_dir
        assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", data["name"])
        assert len(data["name"]) <= 64
        assert 100 < len(data["description"]) <= 1024
        assert data["license"] == "MIT"
        assert data["version"] == PYPROJECT["project"]["version"]

    def test_description_has_russian_triggers(self, skill_dir: str):
        """описание содержит русские фразы-триггеры"""
        data, _ = frontmatter(read(SKILLS / skill_dir / "SKILL.md"))
        assert "Используй, когда" in data["description"]
        assert regex.search(r"\p{Script=Cyrillic}", data["description"])

    def test_entry_point_under_500_lines(self, skill_dir: str):
        """точка входа короче 500 строк"""
        assert len(read(SKILLS / skill_dir / "SKILL.md").split("\n")) < 500

    def test_relative_links_exist(self, skill_dir: str):
        """относительные ссылки ведут на существующие файлы"""
        path = SKILLS / skill_dir / "SKILL.md"
        _, body = frontmatter(read(path))
        for m in re.finditer(r"\]\((?!https?:|#)([^)#\s]+)(?:#[^)]*)?\)", body):
            assert (path.parent / m[1]).exists(), f"{skill_dir}: {m[1]}"

    def test_detector_invocations_valid(self, skill_dir: str):
        """вызовы детектора: проект существует, команда и параметры известны CLI"""
        path = SKILLS / skill_dir / "SKILL.md"
        _, body = frontmatter(read(path))
        for m in INVOCATION_RE.finditer(body):
            project = (path.parent / m[1]).resolve()
            assert project == ROOT, m[0]
            assert (project / "pyproject.toml").is_file(), m[0]
            uv_flags = set(re.findall(r"--[a-z-]+(?: [a-z]+)?", m[2]))
            assert uv_flags <= UV_FLAGS, m[0]
            args = parse(shlex.split(m[3]))
            assert args.cmd in COMMANDS, m[0]
            for flag in re.findall(r"--[a-z-]+", m[3]):
                assert flag in USAGE, f"{m[0]}: {flag}"

    def test_passes_own_detector_without_p0_p1(self, skill_dir: str):
        """проходит собственный детектор без P0 и P1"""
        text = read(SKILLS / skill_dir / "SKILL.md")
        serious = [i for i in analyze(text, "technical").issues if i.severity != "P2"]
        assert [f"{i.line}: {i.text}" for i in serious] == []


def test_reference_invocations_valid():
    """вызовы детектора в файлах references/ считаются от папки скилла, как в SKILL.md"""
    base = SKILLS / "avoid-ai-writing-russian"
    found = [m for p in REFERENCES.glob("*.md") for m in INVOCATION_RE.finditer(read(p))]
    assert found
    for m in found:
        assert (base / m[1]).resolve() == ROOT, m[0]
        assert m[3].split()[0] in COMMANDS, m[0]


def test_main_has_source_check_and_reviewer():
    """после правки сверка с источником, рамка, переносимость и свежий проверяющий"""
    for phrase in ("**Сверка с источником (rewrite и edit).**", "проверь рамку", "**Тест на переносимость.**"):
        assert phrase in MAIN
    assert "](references/review.md)" in MAIN
    review = read(REVIEW)
    for part in ("Выдумка", "Сдвиг смысла", "Рамка", "Переносимость", "Вердикт: прошло | не прошло"):
        assert part in review
    # Проверяющий не видит рассуждений редактора.
    assert "без черновиков, списка изменений и своих рассуждений" in MAIN


def test_skill_invocations_point_at_this_package():
    """проект из вызовов объявляет команду aiw-ru"""
    assert PYPROJECT["project"]["scripts"]["aiw-ru"] == "aiw_ru.cli:run"
    used = {m[3].split()[0] for d in SKILL_DIRS for m in INVOCATION_RE.finditer(read(SKILLS / d / "SKILL.md"))}
    assert used == {"scan", "validate", "antiplagiat", "calibrate", "models", "classify"}
    assert used <= COMMANDS


def test_references_do_not_name_old_runtime():
    """файлы скиллов не называют Bun и TypeScript средой детектора"""
    for p in SKILLS.rglob("*.md"):
        assert not re.search(r"\bbun\b|\bBun\b|src/cli\.ts|TypeScript", read(p)), p


# ─── основной скилл ─────────────────────────────────────────────────────

MAIN = read(SKILLS / "avoid-ai-writing-russian" / "SKILL.md")


def test_main_links_catalog_and_subskill():
    """ссылается на каждый файл каталога и на под-скилл"""
    for p in REFERENCES.glob("*.md"):
        assert f"](references/{p.name})" in MAIN, p.name
    assert "../antiplagiat/SKILL.md" in MAIN


def test_main_does_not_load_whole_catalog():
    """каталог читается по темам: краткий каталог в SKILL.md, файл темы — перед правкой находки"""
    loading = MAIN[MAIN.index("<!-- reference-loading:start -->") : MAIN.index("<!-- reference-loading:end -->")]
    assert "Целиком его не читай" in loading
    assert "открой файл её темы" in loading
    assert "## Уровни серьёзности" in MAIN
    assert "### Частые ложные находки" in MAIN


def test_main_all_modes_described():
    """все режимы описаны"""
    for mode in ("`rewrite`", "`detect`", "`edit`"):
        assert mode in MAIN


def test_main_context_profiles_match_detector():
    """профили в --context совпадают с детектором"""
    m = re.search(r"--context ([a-z|-]+)", MAIN)
    assert m
    assert sorted(m[1].split("|")) == sorted(PROFILE_TO_MODE)


def test_main_never_add_rule():
    """есть правило «Никогда не добавляй» с выдуманной конкретикой"""
    assert "### Никогда не добавляй" in MAIN
    assert "Выдуманная конкретика" in MAIN


def test_main_answer_format_four_checks():
    """формат ответа: четыре пункта проверки"""
    for item in ("**Проходы**", "**Проверки**", "**Остатки**", "**Причина остановки**"):
        assert item in MAIN


def test_model_probability_is_only_a_signal():
    """вероятность модели — тоже сигнал: агент сомневается в ней и не правит текст ради цифры"""
    assert "**Вероятность модели — тоже сигнал, а не приговор.**" in MAIN
    for phrase in ("nearThreshold", "classify --all", "Не правь текст ради цифры", "по одной вероятности"):
        assert phrase in MAIN, phrase
    assert "classify --all" in AP
    assert "--all" in USAGE


def test_long_text_is_read_by_fragments():
    """агент видит, что модель прочитала только начало, проверяет весь текст по фрагментам и знает, сколько ждать"""
    assert "**Длинный текст модель читает не целиком.**" in MAIN
    for phrase in ("read.truncated", "fragments", "aiLines", "--max-fragments", "70 мс у ModernBERT", "stderr"):
        assert phrase in MAIN, phrase
    # один фрагмент выше порога — слабый признак, подробности в руководстве по моделям
    assert "Один фрагмент выше порога в длинном тексте — слабый признак" in MAIN
    assert "](references/models.md)" in MAIN and "aiw-ru models guide" in MAIN
    assert "«Прочитано»" in AP and "classify` по фрагментам" in AP
    for flag in ("--no-fragments", "--max-fragments"):
        assert flag in USAGE


def test_main_runs_scan_and_validate():
    """основной скилл запускает scan и validate"""
    assert "aiw-ru scan" in MAIN
    assert "aiw-ru validate" in MAIN


# ─── под-скилл antiplagiat ──────────────────────────────────────────────

AP = read(SKILLS / "antiplagiat" / "SKILL.md")


def test_ap_relies_on_main_skill():
    """опирается на основной скилл"""
    assert "../avoid-ai-writing-russian/SKILL.md" in AP


def test_ap_describes_boundary():
    """описывает границу: не маскирует чужой текст"""
    assert "## Граница" in AP
    assert re.search("не маскирует чужой текст", AP)


def test_ap_all_detector_commands_mentioned():
    """все команды детектора упомянуты"""
    for cmd in ("antiplagiat", "validate", "calibrate"):
        assert f"aiw-ru {cmd}" in AP


# ─── каталог ↔ детектор ─────────────────────────────────────────────────

CATALOG_TEXT = "\n\n".join(read(p) for p in CATALOG_FILES)
PROFILES_TEXT = read(PROFILES)
HEADINGS = re.findall(r"^#{2,3} (.+?)\s*$", CATALOG_TEXT, re.MULTILINE)
DETECTION_HEADINGS = re.findall(r"^## (.+?)\s*$", CATALOG_TEXT, re.MULTILINE)


def test_catalog_split_into_topic_files():
    """каталог разбит на тематические файлы: у каждого заголовок и ссылка на договор о правке"""
    assert [p.name for p in CATALOG_FILES] == [
        "chat.md",
        "rhetoric.md",
        "sentences.md",
        "structure.md",
        "typography.md",
        "vocabulary.md",
    ]
    for p in [*CATALOG_FILES, PROFILES, REVIEW, MODELS_GUIDE]:
        text = read(p)
        assert text.startswith("# "), p.name
        assert "`../SKILL.md`" in text, p.name


def test_catalog_headings_unique():
    """раздел каталога описан в одном файле: по заголовку scan находит файл"""
    headings = [*HEADINGS, *re.findall(r"^#{2,3} (.+?)\s*$", PROFILES_TEXT, re.MULTILINE)]
    assert sorted(h for h in set(headings) if headings.count(h) > 1) == []


def test_every_detector_section_has_file():
    """scan знает файл каталога для каждого типа находки"""
    where = skills.catalog()
    for t, section in TYPE_TO_SECTION.items():
        assert where.get(section, "").startswith("references/"), t
        assert (REFERENCES.parent / where[section]).is_file(), t


def test_every_detector_type_bound_to_section():
    """каждый тип детектора привязан к существующему разделу"""
    for t in WEIGHTS:
        assert t in TYPE_TO_SECTION, f"нет раздела для {t}"
        assert TYPE_TO_SECTION[t] in HEADINGS, f"{t} → {TYPE_TO_SECTION[t]}"


def test_bindings_reference_existing_types():
    """привязки не ссылаются на несуществующие типы"""
    for t in TYPE_TO_SECTION:
        assert t in TYPE_LABELS, t


def test_every_catalog_section_checked_or_judgment():
    """каждый раздел каталога либо проверяется детектором, либо помечен как суждение"""
    mapped = set(TYPE_TO_SECTION.values())
    for h in DETECTION_HEADINGS:
        assert h in mapped or h in JUDGMENT_ONLY, f"раздел «{h}» не учтён"
    for h in JUDGMENT_ONLY:
        assert h in HEADINGS, h


def test_profile_table_matches_detector():
    """таблица соответствия профилей совпадает с детектором"""
    table = PROFILES_TEXT[PROFILES_TEXT.index("### Соответствие режимам детектора") :]
    for profile, mode in PROFILE_TO_MODE.items():
        assert re.search(rf"\| `{re.escape(profile)}` \| `{mode}` \|", table), profile


def test_strictness_matrix_has_column_per_profile():
    """в матрице строгости есть столбец для каждого профиля"""
    m = re.search(r"\| Правило \|(.+)\|", PROFILES_TEXT)
    assert m
    assert sorted(c.strip() for c in m[1].split("|")) == sorted(PROFILE_TO_MODE)


# ─── примеры из каталога ────────────────────────────────────────────────
# То, что каталог называет приметой, детектор должен находить,
# а то, что каталог называет законным, — пропускать.

CATALOG_EXAMPLES: list[tuple[str, str, ContextMode]] = [
    ("Модель работает быстро — и это меняет всё.", "em-dash-splice", "general"),
    ("# Анализ Существующих Методов Распознавания\n\nТекст.", "title-case", "general"),
    ("Это не просто инструмент, а целая экосистема.", "not-x-but-y", "general"),
    ("Обеспечение повышения эффективности проведения мониторинга состояния машиниста.", "genitive-chain", "general"),
    ("Мы решаем задачу, которая адресует проблему задержек.", "calque", "general"),
    ("Логи смотрим на ежедневной основе.", "calque", "general"),
    ("Ни для кого не секрет, что сон важен.", "template", "general"),
    ("Подводя итог, скажем главное.", "transition", "general"),
    ("В заключение, метод работает.", "transition", "general"),
    ("Это знаменует новую эру в медицине.", "significance", "general"),
    ("Технология имеет все шансы стать одним из ключевых трендов.", "future-narrative", "general"),
    ("Будущее за инновационными решениями!", "future-narrative", "general"),
    ("Решение потенциально может ускорить работу.", "hedge-stack", "general"),
    ("Мы получили реальную пользу от внедрения.", "real-inflation", "general"),
    ("Это честная метрика качества.", "moral-adjective", "general"),
    ("Эксперты считают, что это важно.", "vague-attribution", "general"),
    ("Таким образом, можно сделать вывод, что метод работает.", "generic-conclusion", "general"),
    ("Надеюсь, это поможет!", "chatbot", "general"),
    ("Давайте разберёмся, как это устроено.", "lets", "general"),
    ("Вот о чём почему-то молчат авторы учебников.", "novelty", "general"),
    ("И вот тут начинается самое интересное.", "hook", "general"),
    ("Сохраняйте себе, пригодится.", "social-closer", "general"),
    ("Встречайте: Flowdesk!", "launch-intro", "general"),
    ("Горячее мнение: тесты не нужны.", "fake-casual", "general"),
    ("Представьте мир, где поезда ходят сами.", "speculative-opener", "general"),
    ("Давайте подумаем шаг за шагом.", "reasoning", "general"),
    ("Отличный вопрос! Разберём.", "sycophancy", "general"),
    ("Буду честен: мы не успели.", "narrated-candor", "general"),
    ("Вы спрашиваете о том, как это работает.", "acknowledgment", "general"),
    ("По состоянию на мою последнюю информацию, данных нет.", "cutoff", "general"),
    ("Стоит отметить, что метод новый.", "filler", "general"),
    ("Модель показала высокую точность.", "unmeasured-claim", "general"),
    ("Без воды, без лишних слов, без жаргона.", "negation-chain", "general"),
    ("Он назвал это “прорывом”.", "english-quotes", "general"),
    ("Точность составила 0.93.", "decimal-point", "general"),
    ("См. [Вставьте источник].", "placeholder", "general"),
    ("Это крайне важная задача.", "hollow-intensifier", "general"),
    ("Безусловно, метод работает.", "confidence", "general"),
    ("Эта статья заслуживает внимания.", "vague-endorsement", "general"),
    ("Хотя результаты впечатляют, вопрос остается открытым.", "false-concession", "general"),
    ("Хотите, я сокращу его до 100 слов?", "chat-wrapper", "general"),
    ("Вот вариант поста:\n\nМы запустили сервис.", "chat-wrapper", "general"),
    ("Исследование подчёркивает важность сна. Недосып может привести к ошибкам.", "model-idiolect", "general"),
    ("Мы долго думали, как описать результаты эксперимента, и в итоге решили,", "truncated", "general"),
]

CATALOG_LEGIT: list[tuple[str, str, ContextMode]] = [
    ("Цель работы — разработка метода.", "em-dash-splice", "general"),
    ("Интервал 2019–2023 годов.", "hyphen-dash", "general"),
    ("Он назвал это «прорывом».", "straight-quotes", "general"),
    ("Точность составила 0,93.", "decimal-point", "general"),
    ("Мы ожидали рост задержки, и он действительно вырос на 12 %.", "hollow-intensifier", "general"),
    ("Научная новизна заключается в новом методе.", "tier2", "academic"),
    ("Экосистема пакетов npm растёт.", "tier1", "technical"),
    ("Настроили Wi-Fi-роутер для IT-отдела.", "homoglyph", "general"),
    ("Задача #12 закрыта, цвет #1a2b3c.", "hashtag-stuffing", "general"),
    ("Представим отсортированный массив из десяти чисел и найдём медиану.", "speculative-opener", "general"),
    ("Если нужно, могу подготовить сводку к пятнице.", "chat-wrapper", "general"),
    ("Кстати, если интересно, могу расписать бюджет поездки.", "chat-wrapper", "social"),
    ("Министр подчёркивает, что сроки не изменятся.", "model-idiolect", "general"),
    ("Неверный ключ может привести к ошибке. Это позволяет найти её раньше.", "model-idiolect", "technical"),
    ("Отчёт готов.\n\nС уважением,", "truncated", "general"),
]


@pytest.mark.parametrize(("text", "kind", "context"), CATALOG_EXAMPLES, ids=[e[0] for e in CATALOG_EXAMPLES])
def test_catalog_example_found(text: str, kind: str, context: ContextMode):
    """находит: пример из каталога"""
    assert kind in [i.type for i in analyze(text, context).issues]


@pytest.mark.parametrize(("text", "kind", "context"), CATALOG_LEGIT, ids=[e[0] for e in CATALOG_LEGIT])
def test_catalog_legit_passes(text: str, kind: str, context: ContextMode):
    """пропускает законное: пример из каталога"""
    assert kind not in [i.type for i in analyze(text, context).issues]


def test_examples_cover_most_lexical_types():
    """примеры покрывают большинство словарных типов"""
    covered = {kind for _, kind, _ in CATALOG_EXAMPLES}
    structural = re.compile(
        r"uniform|diversity|cluster|run|tier[23]|phrase3|same-opener|staccato|stacked|bullet|hashtag|bold|emoji"
        r"|homoglyph|invisible|chat-markup|ai-url|hyphen|straight|tier1"
    )
    lexical = [t for t in TYPE_TO_SECTION if not structural.search(t)]
    assert [t for t in lexical if t not in covered] == []


# ─── под-скилл antiplagiat и калибровка ─────────────────────────────────


def test_ap_asks_user_for_reports():
    """просит у пользователя отчёты для calibrate"""
    assert "попроси у пользователя отчёты «Антиплагиата»" in AP
    assert "calibrate --doc" in AP
    assert ".gitignore" in AP


@pytest.mark.parametrize("flag", ["--marked", "--share"])
def test_ap_report_ways_supported_by_cli(flag: str):
    """предлагает несколько способов передать отчёт, и каждый поддержан CLI"""
    for way in (
        "Файл отчёта",
        "Скриншоты",
        "Скопированные фрагменты",
        "Номера абзацев",
        "итоговая цифра",
        "Ссылка на отчёт",
    ):
        assert way in AP
    assert flag in AP
    assert flag in USAGE
    report = parse(["calibrate", "--doc", "a.md", flag, "12"]).reports[0]
    assert (report.marked if flag == "--marked" else report.share) is not None


def test_ap_uncalibrated_marker_in_cli_output(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    """строка, по которой скилл узнаёт некалиброванную модель, есть в выводе CLI"""
    m = re.search(r"строке «Модель» вывода стоит «([^»]+)»", AP)
    assert m
    monkeypatch.setenv("AIW_RU_CONFIG", "")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("COLUMNS", "160")
    monkeypatch.chdir(tmp_path)
    assert main(["antiplagiat", str(AI_VAK)]) == 0
    assert re.search(rf"^ {{2}}Модель +{re.escape(m[1])}", capsys.readouterr().out, re.MULTILINE)


def test_main_mentions_reports_when_switching():
    """основной скилл упоминает отчёты при переходе к antiplagiat"""
    assert "прошлые отчёты «Антиплагиата»" in MAIN

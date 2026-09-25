# CLAUDE.md

Общие правила репозитория лежат в AGENTS.md, их читают и Claude Code, и Codex:

@AGENTS.md

Ниже правила, нужные только Claude Code.

## Плагин

Репозиторий сам служит маркетплейсом: `.claude-plugin/marketplace.json` указывает на корень (`"source": "./"`), а `.claude-plugin/plugin.json` описывает плагин с двумя скиллами. После правки манифестов:

```bash
claude plugin validate .
```

Поставить скиллы из рабочей копии и проверить их в деле:

```bash
claude plugin marketplace add ./
```

```bash
claude plugin install avoid-ai-writing-russian@avoid-ai-writing-russian
```

## Проверка скилла моделью

`uv run python evals/run.py` гоняет сценарии из `evals/cases.json` через `claude -p` и оценивает ответы `evals/grade.py`. Нужна живая сессия `claude login`; вложенный запуск без неё падает с «OAuth session expired». Один сценарий: `--case vak-protected`, только собрать промпт: `--dry`. Результаты пишутся в `evals/results/` и в git не идут.

Правишь формат ответа в SKILL.md — обнови `evals/golden/` и прогони `uv run pytest tests/test_evals.py`: эталоны проверяются тем же `grade.py`.

## Как вести работу

- Меняешь поведение детектора — сначала тест на русском примере, потом код. Ложное срабатывание на `tests/fixtures/corpus/human/` хуже пропуска.
- Инструменты ввода иногда превращают `\u200B` в настоящий невидимый символ. После записи таких строк прогони `uv run python scripts/check_invisible.py <файлы>`, хук `invisible-chars` тоже это ловит.
- Пути в тестах и скриптах собирай от `Path(__file__)`, без склейки строк с `/`: CI гоняет тесты и на Windows.
- Коммиты на русском, первая строка до 72 знаков без точки, вторая пустая; это проверяет хук `commit-msg`. Хуки не отключай через `--no-verify`, чини причину.
- Выпуск только через `uv run python scripts/release.py`, версии руками не правь: их семь, плюс версия проекта в `uv.lock`, и `scripts/check_versions.py` следит за совпадением.

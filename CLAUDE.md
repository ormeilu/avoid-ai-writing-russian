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

`bun run eval` гоняет сценарии из `evals/cases.json` через `claude -p` и оценивает ответы `evals/grade.ts`. Нужна живая сессия `claude login`; вложенный запуск без неё падает с «OAuth session expired». Один сценарий: `bun run eval --case vak-protected`, только собрать промпт: `--dry`. Результаты пишутся в `evals/results/` и в git не идут.

Правишь формат ответа в SKILL.md — обнови `evals/golden/` и прогони `bun test test/evals.test.ts`: эталоны проверяются тем же `grade.ts`.

## Как вести работу

- Меняешь поведение детектора — сначала тест на русском примере, потом код. Ложное срабатывание на `test/fixtures/corpus/human/` хуже пропуска.
- Инструменты ввода иногда превращают `\u200B` в настоящий невидимый символ. После записи таких строк прогони `bun test test/lexicon.test.ts`, хук `invisible-chars` тоже это ловит.
- Пути в тестах и скриптах собирай через `join(import.meta.dir, …)`. `new URL(…).pathname` на Windows даёт `/D:/…`, и CI падает.
- Коммиты на русском, первая строка до 72 знаков без точки, вторая пустая; это проверяет хук `commit-msg`. Хуки не отключай через `--no-verify`, чини причину.
- Выпуск только через `bun run release`, версии руками не правь: их семь, и `scripts/check-versions.ts` следит за совпадением.

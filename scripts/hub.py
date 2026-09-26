#!/usr/bin/env python3
"""Карточка модели LightGBM и выкладка на Hugging Face.

    uv run --group train scripts/hub.py ПАПКА [--repo ИМЯ] [--no-push]

Карточка собирается из metrics.json, который пишет train.py: метаданные для Hub
(model-index с результатами на LLMTrace) и описание на русском. Отчёт об
обучении с командами для воспроизведения ложится в docs/models/. Выкладка
обновляет репозиторий модели и добавляет его в коллекцию aiw-ru, где лежат все
необязательные модели проекта.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import EvalResult, HfApi, ModelCard, ModelCardData
from huggingface_hub.errors import HfHubHTTPError, LocalTokenNotFoundError

REPO = "toiletsandpaper/russian-ai-text-detector-lightgbm"
COLLECTION = "aiw-ru"
COLLECTION_DESCRIPTION = (
    "Детекторы ИИ-текста для русского: три трансформера и бустинг LightGBM. "
    "Порядок тот же, в котором их выбирает aiw-ru"
)
GITHUB = "https://github.com/ormeilu/avoid-ai-writing-russian"
DATASET = "iitolstykh/LLMTrace_classification"
DATASET_NAME = "LLMTrace classification (ru)"
TASK = "text-classification"
TASK_NAME = "AI-generated text detection"
ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = "docs/models/russian-ai-text-detector-lightgbm.md"
REPORT_METRICS = "docs/models/russian-ai-text-detector-lightgbm.json"
# Теги Hub: по ним модель находят в фильтре и поиске.
TAGS = [
    "ai-text-detection",
    "ai-generated-text-detection",
    "machine-generated-text-detection",
    "llm-detection",
    "ai-detector",
    "chatgpt-detector",
    "ai-slop",
    "russian",
    "stylometry",
    "lightgbm",
    "tabular-features",
    "aiw-ru",
]
# Генераторы из обучающей части LLMTrace, по которым модель ищут.
GENERATORS = (
    "ChatGPT (GPT-3.5, GPT-4, GPT-4o, o1, o3), GigaChat, YandexGPT, Qwen, Llama, Gemma, DeepSeek, Mistral, Command R"
)


def num(x: float, digits: int = 3) -> str:
    """Число формулой с точкой: $0.943$. Для Hub формулы переписывает hub_math."""
    return f"${x:.{digits}f}$"


def count(n: int) -> str:
    """Целое формулой; от 10 000 разряды через узкий пробел."""
    return f"${n:,}$".replace(",", "\\,") if n >= 10_000 else f"${n}$"


def pct(x: float) -> str:
    return f"${100 * x:.1f}\\%$"


# Блоки и строки кода, внутри которых формулы не трогаем, и сама формула $…$.
_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)
_INLINE_MATH = re.compile(r"\$([^$\n]+?)\$")


def hub_math(text: str) -> str:
    """Hub рисует строчные формулы только в \\\\(…\\\\), а GitHub — в $…$.

    Hub не узнаёт формулу, если перед \\\\( стоит не пробел («(\\\\(1\\\\)», «–\\\\(5\\\\)»),
    поэтому тексты карточки обходятся без таких мест; это проверяет тест.
    """
    parts = _CODE.split(text)
    return "".join(
        part if i % 2 else _INLINE_MATH.sub(lambda f: f"\\\\({f[1]}\\\\)", part) for i, part in enumerate(parts)
    )


def md_table(head: list[str], rows: list[list[str]]) -> str:
    align = ["---"] + ["--:"] * (len(head) - 1)
    lines = [head, align, *rows]
    return "\n".join("| " + " | ".join(r) + " |" for r in lines)


def _result(split: str, metric: str, value: float, name: str, **extra: Any) -> EvalResult:
    return EvalResult(
        task_type=TASK,
        task_name=TASK_NAME,
        dataset_type=DATASET,
        dataset_name=DATASET_NAME,
        dataset_split=split,
        metric_type=metric,
        metric_value=round(value, 4),
        metric_name=name,
        source_name="aiw-ru scripts/train.py",
        source_url=f"{GITHUB}/blob/master/scripts/train.py",
        **extra,
    )


def eval_results(m: dict) -> list[EvalResult]:
    t, v = m["test"], m["valid"]
    out = [
        _result("test", "accuracy", t["accuracy"], "Accuracy"),
        _result("test", "roc_auc", t["roc_auc"], "ROC AUC"),
        _result("test", "f1", t["macro"]["f1"], "Macro F1", metric_args={"average": "macro"}),
    ]
    for cls, label in (("ai", "AI"), ("human", "human")):
        c = t["classes"][cls]
        for key in ("precision", "recall", "f1"):
            out.append(_result("test", key, c[key], f"{key.capitalize()} ({label})", metric_args={"pos_label": cls}))
    out.append(
        _result(
            "test",
            "roc_auc",
            t["roc_auc_created"],
            "ROC AUC (human vs generated from scratch)",
        )
    )
    out += [
        _result("validation", "accuracy", v["accuracy"], "Accuracy"),
        _result("validation", "roc_auc", v["roc_auc"], "ROC AUC"),
    ]
    # Hub сводит результаты одной задачи, набора и части в один блок, поэтому жанр — в имени метрики.
    for g in m["genres"]:
        out.append(_result("test", "accuracy", g["accuracy"], f"Accuracy ({g['genre']})"))
        if g["roc_auc"] is not None:
            out.append(_result("test", "roc_auc", g["roc_auc"], f"ROC AUC ({g['genre']})"))
    return out


def _classes_table(r: dict) -> str:
    rows = [
        [
            label,
            num(r["classes"][k]["precision"]),
            num(r["classes"][k]["recall"]),
            num(r["classes"][k]["f1"]),
            count(r["classes"][k]["support"]),
        ]
        for k, label in (("human", "люди"), ("ai", "ИИ"))
    ]
    n = sum(r["classes"][k]["support"] for k in ("human", "ai"))
    rows.append(["accuracy", "", "", num(r["accuracy"]), count(n)])
    rows.append(
        ["среднее по классам", num(r["macro"]["precision"]), num(r["macro"]["recall"]), num(r["macro"]["f1"]), count(n)]
    )
    return md_table(["Класс", "Precision", "Recall", "F1", "Текстов"], rows)


def _trees(p: dict) -> str:
    if p["best_iteration"] < p["rounds"]:
        return f"{count(p['best_iteration'])}: ранняя остановка по logloss на valid, предел {count(p['rounds'])}"
    return f"{count(p['rounds'])}: предел, logloss на valid ещё снижался"


def _auc(x: float | None) -> str:
    return num(x) if x is not None else "—"


def _bucket(b: str) -> str:
    """«50–149» → $50\\text{–}149$ одной формулой, «400 и больше» → $400$ и больше."""
    return re.sub(r"(\d+)(?:–(\d+))?", lambda d: f"${d[1]}\\text{{–}}{d[2]}$" if d[2] else f"${d[1]}$", b)


def _features_table(features: list[dict]) -> str:
    return md_table(
        ["Признак", "Что значит", "Люди", "ИИ", "Доля прироста"],
        [
            [f"`{f['name']}`", f["description"], num(f["mean_human"], 2), num(f["mean_ai"], 2), pct(f["gain_share"])]
            for f in features
        ],
    )


def _params_table(m: dict) -> str:
    p = m["params"]
    return md_table(
        ["Параметр", "Значение"],
        [
            ["Деревьев", _trees(p)],
            ["Листьев в дереве", count(p["num_leaves"])],
            ["Скорость обучения", num(p["learning_rate"], 2)],
            ["`min_data_in_leaf`", count(p["min_data_in_leaf"])],
            [
                "`feature_fraction`, `bagging_fraction`",
                f"{num(p['feature_fraction'], 1)}, {num(p['bagging_fraction'], 1)}",
            ],
            ["L2-регуляризация", num(p["lambda_l2"], 1)],
            ["`seed`", f"{count(p['seed'])}, `deterministic={p.get('deterministic', False)}`"],
            ["Время обучения", f"${p['train_seconds']:.0f}$ с, {p['cpu']}"],
            ["Размер `model.txt`", f"${p['size_kb'] / 1024:.1f}$ МБ"],
            ["Обучено", f"{m['created']}, коммит `{m['git_commit']}`"],
        ],
    )


def _results(m: dict, level: str = "##") -> str:
    """Результаты на test: классы, ROC AUC, жанры, длина, тип задания, генераторы."""
    t, d = m["test"], m["detector"]
    fpr = 1 - t["classes"]["human"]["recall"]
    genres = md_table(
        ["Жанр", "Людей", "ИИ", "Accuracy", "ROC AUC модели", "ROC AUC детектора"],
        [
            [
                f"`{g['genre']}`",
                count(g["human"]),
                count(g["ai"]),
                num(g["accuracy"]),
                _auc(g["roc_auc"]),
                _auc(g["detector_roc_auc"]),
            ]
            for g in m["genres"]
        ],
    )
    lengths = md_table(
        ["Слов в тексте", "Текстов", "Accuracy", "ROC AUC"],
        [[_bucket(b["bucket"]), count(b["n"]), num(b["accuracy"]), _auc(b["roc_auc"])] for b in m["lengths"]],
    )
    prompts = md_table(
        ["Задание генератору", "ИИ-текстов", "Recall"],
        [[f"`{x['prompt_type']}`", count(x["n"]), num(x["recall"])] for x in m["prompt_types"]],
    )
    generators = md_table(
        ["Модель-генератор", "Текстов в test", "Recall"],
        [[f"`{x['model']}`", count(x["n"]), num(x["recall"])] for x in m["generators"]],
    )
    return f"""{level} Результаты на test

Текст считается написанным ИИ, если вероятность не меньше {num(0.5, 1)}.

{_classes_table(t)}

ROC AUC модели {num(t["roc_auc"])}, у оценки правил детектора aiw-ru (от $0$ до
$100$) {num(d["roc_auc"])}. Если оставить только тексты, которые модель написала с
нуля, без правки человеческого текста, то {num(t["roc_auc_created"])} у модели и
{num(d["roc_auc_created"])} у детектора. На valid accuracy {num(m["valid"]["accuracy"])},
ROC AUC {num(m["valid"]["roc_auc"])}.

За ИИ модель принимает {pct(fpr)} человеческих текстов.

{level}# По жанрам

{genres}

{level}# По длине текста

{lengths}

{level}# По типу задания генератору

`create` — текст с нуля; `update`, `delete`, `expand` — модель правила,
сокращала или дописывала человеческий текст.

{prompts}

{level}# По моделям-генераторам

Самые частые генераторы в test.

{generators}"""


FEATURES_INTRO = """Признаки считает `aiw_ru.features`: срабатывания правил детектора о словах
и фразах, средняя длина предложения и её разброс, MATTR, средняя длина слова,
доля длинных слов, частоты шести знаков препинания и служебных слов на 100 слов.
Оформление (переводы строк, списки, markdown, вид тире и кавычек, ё) и длина
текста в признаки не входят: в корпусах они говорят о том, откуда взят
человеческий текст, а не о том, кто его написал. «Люди» и «ИИ» в таблице —
средние значения признака на test."""

LLMTRACE_BIBTEX = """```bibtex
@misc{tolstykh2025llmtrace,
  title = {LLMTrace: A Corpus for Classification and Fine-Grained Localization of AI-Written Text},
  author = {Tolstykh, Irina and Tsybina, Aleksandra and Yakubson, Sergey and Kuprashevich, Maksim},
  year = {2025},
  eprint = {2509.21269},
  archivePrefix = {arXiv},
}
```"""


def _limitations(m: dict) -> str:
    fpr = 1 - m["test"]["classes"]["human"]["recall"]
    return f"""- Корпус один. На научных статьях, дипломах и диссертациях модель не
  проверялась, а жанры за пределами таблицы по жанрам она не видела.
- Вероятность — не доказательство авторства. Не используйте модель для
  решений о людях: на test она принимает за ИИ {pct(fpr)} человеческих текстов.
- Правку человеческого текста моделью распознать труднее, чем текст с нуля,
  см. таблицу по типу задания.
- Короткие тексты ненадёжны: частоты слов и знаков на них шумят, см. таблицу
  по длине."""


def body(m: dict, repo: str) -> str:
    ds, t = m["dataset"], m["test"]
    n = m["params"]["features"]
    return f"""# Детектор ИИ-текста для русского языка

`{repo.split("/")[-1]}` оценивает вероятность, что русский текст написала
нейросеть, а не человек. Обучен на текстах {GENERATORS} и других языковых
моделей из корпуса LLMTrace. Помогает проверить текст на ИИ: статью, новость,
отзыв, пост, ответ на вопрос.

Это LightGBM поверх {count(n)} признаков детектора [aiw-ru]({GITHUB}): штампы и
канцелярит ИИ-текста («является», «играет ключевую роль», «в рамках»), длина и
ритм предложений, разнообразие словаря, пунктуация, частоты служебных слов.
Работает на CPU без GPU и трансформеров, файл модели около
${m["params"]["size_kb"] / 1024:.0f}$ МБ. На отложенной части корпуса ROC AUC
{num(t["roc_auc"])}, accuracy {num(t["accuracy"])}.

Модель необязательная: детектор и скиллы aiw-ru работают без неё, а ставится
она, только если пользователь попросит.

## In English

A Russian AI-generated text detector. It estimates the probability that a
Russian text was written by a person or by an LLM such as {GENERATORS}
and others, which makes it usable as an AI text detector, a ChatGPT detector or an
"AI slop" filter for Russian. The model is a CPU-only LightGBM classifier over
{count(n)} interpretable stylometric and lexical features from [aiw-ru]({GITHUB}),
trained on the Russian part of the LLMTrace corpus. Test ROC AUC {num(t["roc_auc"])},
accuracy {num(t["accuracy"])}. The rest of the card is in Russian; usage is
below.

## Данные

Русская часть [LLMTrace classification](https://huggingface.co/datasets/{DATASET})
([статья](https://arxiv.org/abs/2509.21269)). Модель обучена на {count(ds["train"])} текстах
из train, число деревьев подобрано по потере на {count(ds["valid"])} текстах из valid,
итоговые цифры посчитаны на {count(ds["test"])} текстах из test, которые модель при
обучении не видела. Подробный отчёт об обучении с командами для воспроизведения:
[{REPORT_PATH}]({GITHUB}/blob/master/{REPORT_PATH}).

{_results(m)}

## Как пользоваться

Из командной строки:

```bash
uv tool install "aiw-ru[ml] @ git+{GITHUB}"
aiw-ru models install lightgbm
aiw-ru classify --model lightgbm текст.md
```

`aiw-ru models install` без имени ставит её вместе с ModernBERT. Если других
моделей нет, `aiw-ru scan` и `aiw-ru antiplagiat` показывают её вероятность
рядом со своей оценкой. Из Python:

```python
import lightgbm as lgb
from huggingface_hub import hf_hub_download

from aiw_ru.features import features

booster = lgb.Booster(model_file=hf_hub_download("{repo}", "model.txt"))
text = open("текст.md", encoding="utf-8").read()
print(booster.predict([features(text)])[0])  # вероятность, что текст написала модель
```

Признаки считает `aiw_ru.features` той же версии, на которой модель обучена.
Номер версии признаков записан в `features.json`, сейчас это {count(m["features_version"])};
версия aiw-ru при обучении {m["aiw_ru_version"]}.

## Признаки

{FEATURES_INTRO}

Пятнадцать признаков с наибольшим вкладом:

{_features_table(m["features"][:15])}

Все {count(n)} признаков описаны в отчёте об обучении.

## Обучение

{_params_table(m)}

## Ограничения

{_limitations(m)}

## Лицензия и данные

Модель распространяется по MIT, как и aiw-ru. Корпус LLMTrace — Apache 2.0,
человеческие тексты в нём собраны из сторонних источников со своими
лицензиями.

{LLMTRACE_BIBTEX}
"""


def report(m: dict, repo: str) -> str:
    """Отчёт об обучении для репозитория: что сделано, на чём, как повторить."""
    ds, p, env = m["dataset"], m["params"], m.get("environment", {})
    dirty = m["git_commit"].endswith("+правки")
    commit = m["git_commit"].removesuffix("+правки")
    checkout = (
        f"Модель обучена на коммите `{commit}` с незакоммиченными правками, поэтому точная копия кода не "
        "сохранилась: берите ближайший коммит, в котором этот отчёт появился."
        if dirty
        else f"Модель обучена на коммите `{commit}`."
    )
    envs = md_table(["Компонент", "Версия"], [[k, v] for k, v in env.items()])
    return f"""# Отчёт об обучении: {repo.split("/")[-1]}

Файл пишет `scripts/train.py` после каждого полного обучения, руками его не
правят: следующее обучение перезапишет. Все числа в машиночитаемом виде лежат
рядом, в [{Path(REPORT_METRICS).name}]({Path(REPORT_METRICS).name}). Модель и карточка — на
[Hugging Face](https://huggingface.co/{repo}).

## Что это

Необязательная модель aiw-ru: LightGBM оценивает вероятность, что русский
текст написала языковая модель. Пользователь ставит её командой
`aiw-ru models install lightgbm` (нужен extra `ml`), после чего
`aiw-ru classify --model lightgbm` даёт вероятность; без других моделей её
показывают и `scan` с `antiplagiat` рядом со своей оценкой.
Правила детектора показывают, что править в тексте; модель отвечает на другой
вопрос и на корпусе отвечает на него точнее: ROC AUC {num(m["test"]["roc_auc"])} против
{num(m["detector"]["roc_auc"])} у оценки правил.

## Данные

Русская часть [LLMTrace classification](https://huggingface.co/datasets/{ds["repo"]})
(Apache 2.0, [статья](https://arxiv.org/abs/2509.21269)), ревизия набора
`{ds.get("revision", "не записана")}`. Части корпуса используются как есть: train
({count(ds["train"])} текстов) для обучения, valid ({count(ds["valid"])}) для ранней остановки,
test ({count(ds["test"])}) только для итоговых цифр. Метка «ИИ» — любой текст с участием
модели: написанный с нуля (`create`) и человеческий текст, который модель
правила, сокращала или дописывала (`update`, `delete`, `expand`).

## Признаки

{FEATURES_INTRO}

Версия признаков: {count(m["features_version"])}. Её поднимают при любой смене состава или
порядка признаков, и тогда модель старой версии перестаёт загружаться, пока её
не переобучат.

{_features_table(m["features"])}

## Обучение

{_params_table(m)}

Остальное: `objective=binary`, `bagging_freq={p.get("bagging_freq", 1)}`, ранняя остановка через
$100$ раундов без улучшения logloss на valid, `force_row_wise={p.get("force_row_wise", False)}`.

{envs}

{_results(m)}

### Детектор без модели

Для сравнения: оценка правил aiw-ru с порогом {count(m["detector"].get("threshold", 40))} («много примет»).

{_classes_table(m["detector"])}

## Как воспроизвести

{checkout}

```bash
uv sync --group train --locked
```

```bash
uv run --group train scripts/llmtrace.py fetch --set classification --split train
```

```bash
uv run --group train scripts/llmtrace.py fetch --set classification --split valid
```

```bash
uv run --group train scripts/llmtrace.py fetch --set classification --split test
```

```bash
uv run --group train scripts/train.py --rounds {p["rounds"]} --leaves {p["num_leaves"]} --rate {p["learning_rate"]} --no-push
```

`fetch` скачивает ревизию набора, закреплённую в `scripts/llmtrace.py`. Признаки
считаются в нескольких процессах и кэшируются в `~/.cache/aiw-ru/llmtrace`;
LightGBM обучается с `deterministic=True`, поэтому на той же машине с теми же
версиями пакетов повтор даёт ту же модель. На другом процессоре или с другой
версией LightGBM цифры могут разойтись в третьем знаке после точки.

Без `--no-push` скрипт после обучения выложит модель в репозиторий на Hugging
Face (нужен `hf auth login` с правом записи). Карточку и этот отчёт без
переобучения пересобирает `uv run --group train scripts/hub.py ~/.cache/aiw-ru/llmtrace/lightgbm --no-push`.

## Ограничения

{_limitations(m)}

## Ссылка на корпус

{LLMTRACE_BIBTEX}
"""


def write_report(m: dict, repo: str = REPO) -> Path:
    """Пишет отчёт и метрики в docs/models/ репозитория; возвращает путь к отчёту."""
    path = ROOT / REPORT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report(m, repo), encoding="utf-8")
    (ROOT / REPORT_METRICS).write_text(json.dumps(m, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def card(m: dict, repo: str = REPO) -> ModelCard:
    data = ModelCardData(
        language="ru",
        license="mit",
        library_name="lightgbm",
        pipeline_tag=TASK,
        tags=TAGS,
        datasets=[DATASET],
        metrics=["accuracy", "f1", "precision", "recall", "roc_auc"],
        model_name=repo.split("/")[-1],
        eval_results=eval_results(m),
        inference=False,
    )
    return ModelCard(f"---\n{data.to_yaml()}\n---\n\n{hub_math(body(m, repo))}")


def push(folder: Path, repo: str, message: str) -> bool:
    """Обновляет репозиторий модели и коллекцию; без входа в Hub только предупреждает."""
    api = HfApi()
    try:
        api.whoami()
    except (LocalTokenNotFoundError, HfHubHTTPError) as e:
        print(f"Hugging Face: не выложено, нужен `hf auth login` ({e})", file=sys.stderr)
        return False
    try:
        api.create_repo(repo, repo_type="model", exist_ok=True)
        info = api.upload_folder(repo_id=repo, folder_path=folder, commit_message=message)
        collection = api.create_collection(
            COLLECTION, namespace=repo.split("/")[0], description=COLLECTION_DESCRIPTION, exists_ok=True
        )
        api.update_collection_metadata(collection.slug, description=COLLECTION_DESCRIPTION)
        api.add_collection_item(collection.slug, repo, "model", exists_ok=True)
    except HfHubHTTPError as e:
        print(f"Hugging Face: не выложено ({e})", file=sys.stderr)
        return False
    print(f"Hugging Face: {info.commit_url}", file=sys.stderr)
    return True


def main(argv: list[str] | None = None) -> None:
    """Пересобирает карточку из metrics.json и выкладывает папку модели без переобучения."""
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("folder", type=Path, help="папка модели: model.txt, features.json, metrics.json")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--no-push", action="store_true", help="только пересобрать карточку и отчёт")
    args = ap.parse_args(argv)
    m = json.loads((args.folder / "metrics.json").read_text(encoding="utf-8"))
    card(m, args.repo).save(args.folder / "README.md")
    print(f"Карточка: {args.folder / 'README.md'}")
    if not m["dataset"].get("limit"):
        print(f"Отчёт: {write_report(m, args.repo).relative_to(ROOT)}")
    if not args.no_push:
        t = m["test"]
        message = f"Обучение {m['created']}: accuracy {t['accuracy']:.3f}, ROC AUC {t['roc_auc']:.3f}"
        sys.exit(0 if push(args.folder, args.repo, message) else 1)


if __name__ == "__main__":
    main()

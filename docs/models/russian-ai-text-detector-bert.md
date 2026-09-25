# Отчёт об обучении: russian-ai-text-detector-bert

Файл пишет `scripts/train_transformer.py report` после полного обучения, руками
его не правят. Все числа в машиночитаемом виде лежат рядом, в
[russian-ai-text-detector-bert.json](russian-ai-text-detector-bert.json). Модель и карточка выложены на
[Hugging Face](https://huggingface.co/toiletsandpaper/russian-ai-text-detector-bert).

## Что это

Необязательная модель aiw-ru: дообученный энкодер `cointegrated/rubert-tiny2` оценивает
вероятность, что русский текст написала языковая модель. Пользователь ставит её
командой `aiw-ru models install transformer`, после чего `aiw-ru classify`
выдаёт вероятность. В поставке ONNX с весами int8 на $29.7$ МБ и
`inference.json`; для вывода нужны onnxruntime и tokenizers, torch не нужен.
На test ROC AUC $0.987$, у LightGBM на признаках aiw-ru $0.943$,
у оценки правил $0.610$.

## Выбор базы

Кандидаты — небольшие русские энкодеры с открытой лицензией, которые можно
запускать на CPU. Полный [FRIDA](https://huggingface.co/ai-forever/FRIDA)
от ai-forever (MIT) — энкодер T5 на $823$ млн параметров, для CPU пользователя
он слишком тяжёл, поэтому в пилоте его дистилляция `sergeyzh/rubert-mini-frida`.

В пилоте все базы учились одинаково: $20\,000$ текстов train,
эпох $1$, скорость `0.0001`, пачка $32$,
`seed` $1$. ROC AUC посчитан на $10\,000$ текстах valid,
задержка — ONNX int8 на $512$ токенах в один поток, Apple M1, 8 ядер.

| База | Лицензия | Нормализация | Параметров | ROC AUC valid | Accuracy valid | int8, МБ | int8, мс |
| --- | --: | --: | --: | --: | --: | --: | --: |
| `sergeyzh/rubert-mini-frida` | MIT | да | $32.3$ млн | $0.964$ | $0.891$ | $33.2$ | $94.8$ |
| `cointegrated/rubert-tiny2` | MIT | да | $29.2$ млн | $0.957$ | $0.890$ | $29.7$ | $41.3$ |
| `cointegrated/rubert-tiny2` | MIT | нет | $29.2$ млн | $0.960$ | $0.898$ | $29.7$ | $40.3$ |
| `deepvk/RuModernBERT-small` | Apache 2.0 | да | $34.5$ млн | $0.973$ | $0.914$ | $36.1$ | $161.6$ |

Дальше на полном train обучены две базы: самая точная в пилоте и самая быстрая.
Середина, `sergeyzh/rubert-mini-frida`, медленнее tiny2 больше чем вдвое при
небольшом выигрыше в ROC AUC. ROC AUC и accuracy
посчитаны на полном valid у PyTorch и у ONNX int8; «сменил метку» — доля текстов
valid, где int8 и PyTorch расходятся по порогу $0.5$. Задержка — ONNX int8 на $512$
токенах в один поток: на M1 и на двух ядрах Xeon виртуальной машины Colab.

| База | ROC AUC valid | Accuracy valid | ROC AUC int8 | Accuracy int8 | int8 сменил метку | int8, МБ | int8, мс, M1 | int8, мс, x86 |
| --- | --: | --: | --: | --: | --: | --: | --: | --: |
| `cointegrated/rubert-tiny2` | $0.9878$ | $0.946$ | $0.9877$ | $0.946$ | $0.3\%$ | $29.7$ | $39.8$ | $101.1$ |
| `deepvk/RuModernBERT-small` | $0.9935$ | $0.963$ | $0.9926$ | $0.959$ | $1.1\%$ | $36.1$ | $159.8$ | $318.2$ |

Выбрана `cointegrated/rubert-tiny2`. Правило выбора задано до полного обучения: задержка на CPU весит столько же, сколько точность. RuModernBERT-small точнее: на полном valid ROC AUC $0.9935$ против $0.9878$, accuracy $0.963$ против $0.946$. В int8 разрыв чуть меньше, $0.9926$ против $0.9877$, и квантование RuModernBERT вредит заметнее: метка меняется у $1.1\%$ текстов valid, у tiny2 у $0.3\%$. Квантование только весов, без эмбеддингов и классификатора, на подвыборке valid меняет почти столько же меток, сколько обычное. При этом RuModernBERT в int8 в $4$ раза медленнее на M1, $160$ мс против $40$ мс на $512$ токенов, и в $3$ раза медленнее на двух ядрах Xeon, $318$ мс против $101$ мс. В fp32 она обходится без потерь от квантования, но весит $140$ МБ и тратит $272$ мс на M1. Ошибок ранжирования у RuModernBERT в int8 в $1.7$ раза меньше, а задержка в $3\text{–}4$ раза больше, поэтому в поставке tiny2 в int8. RuModernBERT остаётся кандидатом на отдельную, более точную модель.

## Данные

Русская часть [LLMTrace classification](https://huggingface.co/datasets/iitolstykh/LLMTrace_classification)
под Apache 2.0, [статья](https://arxiv.org/abs/2509.21269), ревизия набора
`252e21e3dca713bc82bc3c1ef73bf533d7c9fc7e`. Части корпуса используются как есть. Модель учится на
$237\,929$ текстах train. По $49\,747$ текстам valid выбирается лучший
шаг и срабатывает ранняя остановка. Test нужен только для итоговых цифр, в нём
$52\,521$ текст.

Метка «ИИ» стоит на любом тексте с участием модели: написанном с нуля
(`create`) и на человеческом тексте, который модель правила, сокращала или
дописывала (`update`, `delete`, `expand`). При обучении от текста длиннее
$512$ токенов остаётся только начало. Таких в train
$10.1\%$, в valid $10.0\%$, в test
$10.5\%$.

## Нормализация

Оформление в LLMTrace говорит о том, откуда взят человеческий текст. Ниже
ROC AUC каждой приметы по отдельности на valid: $0.5$ значит, что примета ничего
не говорит, чем дальше от $0.5$ в любую сторону, тем больше говорит.

| Примета оформления | ROC AUC одной приметы |
| --- | --: |
| длина, знаков | $0.552$ |
| переводов строк на 1000 знаков | $0.601$ |
| разметка markdown на 1000 знаков | $0.537$ |
| доля ё среди е и ё | $0.473$ |
| длинных тире на 1000 знаков | $0.510$ |
| кавычек-ёлочек на 1000 знаков | $0.475$ |

Поодиночке приметы слабые, сильнее всех «переводов строк на 1000 знаков» с ROC AUC $0.601$. Но модель может сложить их вместе и учиться на оформлении вместо текста. Поэтому до токенизации текст проходит `normalize()`: правила ниже применяются
по порядку, шаблоны — `re` с флагом `MULTILINE`, в конце пробелы по краям
срезаются. Тот же список лежит в `inference.json`. Длину текста так не убрать,
см. таблицу по длине. В пилоте база `cointegrated/rubert-tiny2` без нормализации дала ROC AUC $0.960$ на valid, с нормализацией $0.957$. Нормализация всё равно остаётся: модель, выучившая оформление корпуса, ошибётся на человеческом тексте, набранном в другом редакторе или скопированном из чата.

| Шаблон | Замена | Что убирает |
| --- | --: | --: |
| `[\u200b\u200c\u200d\u200e\u200f\u2060\ufeff\u00ad]` | пусто | невидимые символы и мягкий перенос |
| `[\u00a0\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\t\r\f\v]` | пробел | неразрывные и прочие пробелы |
| `` ^[ ]*(?:```\|~~~).*$ `` | пусто | ограждения блоков кода |
| `^[ ]*(?:[-*_=][ ]*){3,}$` | пусто | горизонтальные линии и подчёркивания заголовков |
| `^[ ]*\\|?(?:[ ]*:?-+:?[ ]*\\|)+[ ]*(?::?-+:?)?[ ]*$` | пусто | строка-разделитель таблицы |
| `^[ ]*#{1,6}[ ]+` | пусто | решётки заголовков |
| `^[ ]*(?:>[ ]?)+` | пусто | цитаты markdown |
| `^[ ]*(?:[-*+\u2022\u00b7\u25aa\u25cf\u25e6\u2023\u2013\u2014]\|\d{1,3}[.)])[ ]+` | пусто | маркеры и номера списков, тире в начале строки |
| `!\[([^\]\n]*)\]\([^)\n]*\)` | `\1` | картинки: остаётся подпись |
| `\[([^\]\n]+)\]\([^)\n]*\)` | `\1` | ссылки: остаётся текст ссылки |
| `</?[A-Za-z][A-Za-z0-9]*(?:\s[^<>\n]*)?/?>` | пробел | теги HTML |
| `` \*+\|~~\|`+ `` | пусто | звёздочки, зачёркивание, обратные кавычки |
| `(?<!\w)_+\|_+(?!\w)` | пусто | подчёркивания разметки на краях слов |
| `\\|` | пробел | столбцы таблиц |
| `[\u00ab\u00bb\u201e\u201c\u201d\u201f\u2033]` | `"` | двойные кавычки всех видов |
| `[\u2018\u2019\u201a\u201b\u2032]` | `'` | апострофы и одинарные кавычки |
| `[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]` | `-` | тире, минус и типографские дефисы |
| `\u2026` | `...` | многоточие одним знаком |
| `ё` | `е` | ё |
| `Ё` | `Е` | Ё |
| `\s+` | пробел | пробелы и переводы строк |

## Обучение

| Параметр | Значение |
| --- | --: |
| База | [`cointegrated/rubert-tiny2`](https://huggingface.co/cointegrated/rubert-tiny2), ревизия `e8ed3b0c8bbf` |
| Параметров | $29.2$ млн |
| Устройство | Tesla T4, `cuda` |
| Точность | `fp16`, GPU без bf16 (T4): fp16 с масштабированием потерь (GradScaler) |
| Эпох | не больше $3$, лучшая проверка на эпохе $2.00$ |
| Ранняя остановка | после $3$ проверок без роста ROC AUC на valid, сработала |
| Проверок на valid за эпоху | $4$ |
| Оптимизатор | AdamW, скорость `0.0001`, разогрев $6.0\%$ шагов, линейный спад |
| Затухание весов, клиппинг градиента | `0.01`, `1` |
| Пачка | $32$ текста близкой длины |
| `max_length` | $512$ токенов |
| `seed` | $1$ |
| Время обучения | $32$ мин, $338$ текстов в секунду |
| Файл модели | `model.onnx`, веса int8, $29.7$ МБ |
| Обучено | 2026-09-25, коммит `60cb897+правки` |

Проверки на valid по ходу обучения:

| Шаг | Эпоха | loss train | ROC AUC valid | Accuracy valid | logloss valid |
| --- | --: | --: | --: | --: | --: |
| $1859$ | $0.25$ | $0.3412$ | $0.9712$ | $0.913$ | $0.2158$ |
| $3718$ | $0.50$ | $0.2101$ | $0.9820$ | $0.927$ | $0.1942$ |
| $5577$ | $0.75$ | $0.1806$ | $0.9849$ | $0.913$ | $0.2351$ |
| $7436$ | $1.00$ | $0.1659$ | $0.9861$ | $0.940$ | $0.1599$ |
| $9295$ | $1.25$ | $0.0994$ | $0.9863$ | $0.934$ | $0.1972$ |
| $11\,154$ | $1.50$ | $0.0974$ | $0.9876$ | $0.946$ | $0.1763$ |
| $13\,013$ | $1.75$ | $0.0954$ | $0.9876$ | $0.943$ | $0.1886$ |
| $14\,872$ | $2.00$ | $0.0928$ | $0.9878$ | $0.946$ | $0.1680$ |
| $16\,731$ | $2.25$ | $0.0413$ | $0.9872$ | $0.942$ | $0.2944$ |
| $18\,590$ | $2.50$ | $0.0429$ | $0.9869$ | $0.947$ | $0.2468$ |
| $20\,449$ | $2.75$ | $0.0427$ | $0.9872$ | $0.947$ | $0.2384$ |

## Результаты на test

Текст считается написанным ИИ, если вероятность не меньше $0.5$. Текст длиннее $512$ токенов модель читает только до этой границы: среднее по окнам на valid не точнее.

| Класс | Precision | Recall | F1 | Текстов |
| --- | --: | --: | --: | --: |
| люди | $0.937$ | $0.928$ | $0.932$ | $21\,417$ |
| ИИ | $0.951$ | $0.957$ | $0.954$ | $31\,104$ |
| accuracy |  |  | $0.945$ | $52\,521$ |
| среднее по классам | $0.944$ | $0.942$ | $0.943$ | $52\,521$ |

| На test | Эта модель | LightGBM | Правила aiw-ru |
| --- | --: | --: | --: |
| Accuracy | $0.945$ | $0.868$ | $0.495$ |
| ROC AUC | $0.987$ | $0.943$ | $0.610$ |
| ROC AUC, люди против текстов с нуля | $0.989$ | $0.946$ | $0.593$ |
| F1, среднее по классам | $0.943$ | $0.863$ | $0.454$ |
| Людей принято за ИИ | $7.2\%$ | $16.5\%$ | $5.9\%$ |

На valid accuracy $0.946$, ROC AUC $0.988$.

### По жанрам

| Жанр | Людей | ИИ | Accuracy | ROC AUC | ROC AUC LightGBM | ROC AUC правил |
| --- | --: | --: | --: | --: | --: | --: |
| `article` | $5109$ | $5343$ | $0.948$ | $0.989$ | $0.963$ | $0.705$ |
| `factual` | $1383$ | $2209$ | $0.915$ | $0.976$ | $0.896$ | $0.663$ |
| `news` | $3326$ | $3488$ | $0.931$ | $0.984$ | $0.890$ | $0.551$ |
| `poetry` | $3561$ | $3383$ | $0.937$ | $0.979$ | $0.932$ | $0.484$ |
| `question` | $1932$ | $5894$ | $0.958$ | $0.991$ | $0.951$ | $0.729$ |
| `review` | $2678$ | $4019$ | $0.960$ | $0.990$ | $0.960$ | $0.632$ |
| `short_form` | $450$ | $3000$ | $0.934$ | $0.980$ | $0.942$ | $0.553$ |
| `story` | $2978$ | $3768$ | $0.953$ | $0.991$ | $0.967$ | $0.566$ |

### По длине текста

Слова считает детектор aiw-ru, как и для LightGBM.

| Слов в тексте | Текстов | Accuracy | ROC AUC | Accuracy LightGBM | ROC AUC LightGBM |
| --- | --: | --: | --: | --: | --: |
| $0\text{–}49$ | $14\,742$ | $0.925$ | $0.980$ | $0.819$ | $0.902$ |
| $50\text{–}149$ | $17\,766$ | $0.942$ | $0.986$ | $0.849$ | $0.924$ |
| $150\text{–}399$ | $16\,327$ | $0.963$ | $0.993$ | $0.913$ | $0.968$ |
| $400$ и больше | $3686$ | $0.962$ | $0.980$ | $0.952$ | $0.970$ |

### По типу задания генератору

`create` — текст с нуля; `update`, `delete`, `expand` — модель правила,
сокращала или дописывала человеческий текст.

| Задание генератору | ИИ-текстов | Recall | Recall LightGBM |
| --- | --: | --: | --: |
| `create` | $14\,064$ | $0.963$ | $0.899$ |
| `delete` | $5966$ | $0.924$ | $0.812$ |
| `expand` | $4850$ | $0.972$ | $0.945$ |
| `update` | $6224$ | $0.963$ | $0.903$ |

### По моделям-генераторам

Самые частые генераторы в test.

| Модель-генератор | Текстов в test | Recall | Recall LightGBM |
| --- | --: | --: | --: |
| `gpt-3.5` | $4772$ | $0.977$ | $0.929$ |
| `gpt-4o` | $3884$ | $0.983$ | $0.908$ |
| `databricks/dbrx-instruct` | $1513$ | $0.927$ | $0.859$ |
| `google/gemma-2-27b-it` | $1498$ | $0.977$ | $0.888$ |
| `01-ai/Yi-1.5-34B-Chat` | $1472$ | $0.952$ | $0.898$ |
| `yandex/YandexGPT-5-Lite-8B-instruct` | $1469$ | $0.932$ | $0.853$ |
| `CohereForAI/c4ai-command-r-08-2024` | $1461$ | $0.990$ | $0.951$ |
| `unsloth/Llama-3.3-70B-Instruct` | $1452$ | $0.964$ | $0.915$ |
| `Qwen/Qwen2.5-72B-Instruct` | $1449$ | $0.959$ | $0.887$ |
| `mistralai/Ministral-8B-Instruct-2410` | $1396$ | $0.903$ | $0.829$ |
| `Qwen/QwQ-32B` | $1386$ | $0.974$ | $0.885$ |
| `GigaChat-Max` | $1266$ | $0.958$ | $0.931$ |
| `gigachat` | $1181$ | $0.936$ | $0.831$ |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | $942$ | $0.960$ | $0.869$ |
| `WizardLM-2-7B` | $891$ | $0.980$ | $0.926$ |

### Детектор без модели

Оценка правил aiw-ru с порогом $40$, то есть «много примет»:

| Класс | Precision | Recall | F1 | Текстов |
| --- | --: | --: | --: | --: |
| люди | $0.444$ | $0.941$ | $0.603$ | $21\,417$ |
| ИИ | $0.822$ | $0.187$ | $0.305$ | $31\,104$ |
| accuracy |  |  | $0.495$ | $52\,521$ |
| среднее по классам | $0.633$ | $0.564$ | $0.454$ | $52\,521$ |

## Вывод на CPU

Ниже варианты одной и той же модели. int8 идёт в поставку, если на valid
теряет не больше $0.002$ ROC AUC; test в этом решении не участвует.

| Вариант | ROC AUC valid | Accuracy valid | ROC AUC test | Accuracy test |
| --- | --: | --: | --: | --: |
| PyTorch на GPU | $0.9878$ | $0.9464$ | $0.9868$ | $0.9451$ |
| ONNX fp32 на CPU | $0.9878$ | $0.9465$ | $0.9868$ | $0.9451$ |
| ONNX int8 на CPU | $0.9877$ | $0.9458$ | $0.9869$ | $0.9449$ |

Сверка с PyTorch на $300$ текстах valid: у ONNX fp32 наибольшая
разница вероятностей $2.9\cdot 10^{-6}$, у int8 $0.123$,
метки int8 совпадают у $99.7\%$ текстов.

Правило для длинных текстов выбрано по valid, в test таких текстов
$5498$:

| Правило | ROC AUC valid | Accuracy valid | ROC AUC test | Accuracy test |
| --- | --: | --: | --: | --: |
| начало текста | $0.981$ | $0.965$ | $0.980$ | $0.957$ |
| среднее по окнам | $0.976$ | $0.963$ | $0.973$ | $0.959$ |

Задержка на один текст в миллисекундах, медиана $30$ прогонов, Apple M1, 8 ядер:

| Среда | 128 токенов, потоков: 1 | 128 токенов, потоков: 8 | 512 токенов, потоков: 1 | 512 токенов, потоков: 8 |
| --- | --: | --: | --: | --: |
| onnxruntime fp32 | $9.6$ | $4.5$ | $57.7$ | $26.9$ |
| onnxruntime int8 | $5.1$ | $2.8$ | $39.8$ | $22.7$ |
| torch fp32 | $3.0$ | $3.5$ | $19.3$ | $10.1$ |

На двух ядрах виртуальной машины Colab, это ближе к слабому ноутбуку на x86,
Intel(R) Xeon(R) CPU @ 2.00GHz, 2 ядра:

| Среда | 128 токенов, потоков: 1 | 128 токенов, потоков: 2 | 512 токенов, потоков: 1 | 512 токенов, потоков: 2 |
| --- | --: | --: | --: | --: |
| onnxruntime fp32 | $21.5$ | $12.0$ | $111.2$ | $95.8$ |
| onnxruntime int8 | $17.6$ | $10.0$ | $101.1$ | $87.4$ |
| torch fp32 | $12.8$ | $12.6$ | $70.7$ | $48.9$ |

Размер зависимостей вывода после установки через uv для Python 3.12:

| Зависимости вывода | Платформа | МБ на диске |
| --- | --: | --: |
| onnxruntime + tokenizers | `aarch64-apple-darwin` | $127$ |
| onnxruntime + tokenizers | `x86_64-manylinux_2_28` | $156$ |
| onnxruntime + tokenizers | `x86_64-pc-windows-msvc` | $112$ |
| torch + transformers | `aarch64-apple-darwin` | $669$ |
| torch + transformers | `x86_64-manylinux_2_28` | $5649$ |
| torch (CPU) + transformers | `x86_64-manylinux_2_28` | $908$ |
| torch + transformers | `x86_64-pc-windows-msvc` | $642$ |

Размер считается вместе со всеми зависимостями, в том числе numpy и
huggingface-hub, которые aiw-ru с extra `ml` и так ставит для LightGBM.

На этой машине torch считает быстрее onnxruntime: $19.3$ мс против $39.8$ мс у int8 на $512$ токенах в один поток. Но torch с transformers занимает на диске в $5.3\text{–}5.8$ раза больше, а на Linux со сборкой torch под CUDA по умолчанию — $5649$ МБ. aiw-ru ставят и на слабые ноутбуки, поэтому в поставке ONNX: установка лёгкая, а ROC AUC на test совпадает с PyTorch до четвёртого знака.

## Как пользоваться

Из командной строки:

```bash
uv tool install "aiw-ru[ml]"
aiw-ru models install transformer
aiw-ru classify текст.md
```

Из Python без aiw-ru нужны только onnxruntime, tokenizers, numpy и huggingface-hub:

```python
import json
import re

import numpy as np
import onnxruntime as ort
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer

files = ["inference.json", "model.onnx", "tokenizer.json"]
path = snapshot_download("toiletsandpaper/russian-ai-text-detector-bert", allow_patterns=files)
spec = json.load(open(f"{path}/inference.json", encoding="utf-8"))
session = ort.InferenceSession(f"{path}/{spec['file']}", providers=["CPUExecutionProvider"])
tokenizer = Tokenizer.from_file(f"{path}/{spec['tokenizer']}")
tokenizer.no_truncation()
tokenizer.no_padding()


def probability(text: str) -> float:
    """Вероятность, что текст написала языковая модель."""
    text = text[: spec["max_chars"]]
    for rule in spec["normalize"]:
        text = re.sub(rule["pattern"], rule["replacement"], text, flags=re.MULTILINE)
    ids = tokenizer.encode(text.strip(), add_special_tokens=False).ids
    width = spec["max_length"] - 2
    windows = [ids[i : i + width] for i in range(0, max(len(ids), 1), width)][: spec["max_windows"]]
    windows = [[spec["cls_id"], *w, spec["sep_id"]] for w in windows]
    batch = np.full((len(windows), max(map(len, windows))), spec["pad_id"], dtype=np.int64)
    mask = np.zeros_like(batch)
    for k, w in enumerate(windows):
        batch[k, : len(w)] = w
        mask[k, : len(w)] = 1
    logits = session.run([spec["output"]], {"input_ids": batch, "attention_mask": mask})[0]
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    return float((p[:, spec["labels"].index("ai")] / p.sum(axis=1)).mean())


print(probability(open("текст.md", encoding="utf-8").read()))
```

`inference.json` описывает вывод целиком: файл ONNX, токенизатор,
`max_length`, служебные токены, правила нормализации из обучения, порог
$0.5$ и число окон для длинных текстов.

## Как воспроизвести

Модель обучена на коммите `60cb897`, к которому `scripts/train_transformer.py` и `scripts/colab_transformer.py` ещё не были закоммичены. Для повтора берите их версию из коммита, в котором появился этот отчёт. Базовая модель — ревизия `e8ed3b0c8bbf4fb6984c3de043bf7d2f4e5969ae`, корпус — ревизия
`252e21e3dca713bc82bc3c1ef73bf533d7c9fc7e`, её закрепляет `scripts/llmtrace.py`. Окружение:

| Компонент | Версия |
| --- | --: |
| python | `3.13.15` |
| torch | `2.14.0+cu130` |
| transformers | `5.17.0` |
| tokenizers | `0.23.2` |
| onnxruntime | `1.30.0` |
| numpy | `2.5.3` |
| scikit-learn | `1.9.1` |
| platform | `Linux 6.6.122+, x86_64, 2 ядра` |
| cuda | `13.0` |
| device | `cuda` |
| gpu | `Tesla T4` |
| onnx | `1.23.0` |
| evaluate_python | `3.14.6` |
| evaluate_torch | `2.14.0` |
| evaluate_platform | `Darwin 25.6.0, arm64, 8 ядер` |

Обучение идёт на GPU. В Colab на бесплатной T4 это делает
`scripts/colab_transformer.py` через официальный Colab CLI (google-colab-cli,
`colab sessions` должен работать). Корпус скачивается прямо на VM, через ваш
компьютер идут только исходники и готовая модель:

```bash
uv run scripts/colab_transformer.py up --gpu T4
uv run scripts/colab_transformer.py setup
uv run scripts/colab_transformer.py start --name full --job "train --base cointegrated/rubert-tiny2 --epochs 3 --lr 0.0001 --batch 32 --max-length 512 --evals-per-epoch 4 --patience 3"
uv run scripts/colab_transformer.py status --name full
uv run scripts/colab_transformer.py fetch /content/data/transformer/runs/rubert-tiny2 ~/.cache/aiw-ru/llmtrace/transformer/runs
```

Задержка на x86 меряется на CPU той же VM, после обучения, пока GPU свободен:

```bash
uv run scripts/colab_transformer.py start --name x86 --job "export /content/data/transformer/runs/rubert-tiny2 --json export-x86.json"
uv run scripts/colab_transformer.py fetch /content/data/transformer/runs/rubert-tiny2/export-x86.json ~/.cache/aiw-ru/llmtrace/transformer/runs/rubert-tiny2
uv run scripts/colab_transformer.py down
```

Локально на CUDA или MPS вместо Colab:

```bash
uv run --group train scripts/llmtrace.py fetch --set classification --split train
uv run --group train scripts/llmtrace.py fetch --set classification --split valid
uv run --group train scripts/llmtrace.py fetch --set classification --split test
uv run --group train --group transformer scripts/train_transformer.py train --base cointegrated/rubert-tiny2 --epochs 3 --lr 0.0001 --batch 32 --max-length 512 --evals-per-epoch 4 --patience 3
```

Дальше на своём CPU: ONNX и сверка с PyTorch, задержка, оценка на valid и test,
карточка и этот отчёт.

```bash
uv run --group train --group transformer scripts/train_transformer.py export ~/.cache/aiw-ru/llmtrace/transformer/runs/rubert-tiny2
uv run --group train --group transformer scripts/train_transformer.py evaluate ~/.cache/aiw-ru/llmtrace/transformer/runs/rubert-tiny2
uv run --group train --group transformer scripts/train_transformer.py report ~/.cache/aiw-ru/llmtrace/transformer/runs/rubert-tiny2
```

Обучение на GPU не детерминировано до бита, повтор может разойтись в третьем
знаке. Пилот повторяется командой `pilot --base ИМЯ` с настройками по умолчанию
(`--limit 20000 --epochs 1`).

## Ограничения

- Корпус один. На научных статьях, дипломах и диссертациях модель не
  проверялась, а жанры за пределами таблицы по жанрам она не видела.
- Вероятность — не доказательство авторства. Не используйте модель для
  решений о людях: на test она принимает за ИИ $7.2\%$ человеческих текстов.
- Правку человеческого текста моделью распознать труднее, чем текст с нуля,
  см. таблицу по типу задания.
- Нормализация убирает оформление, но не длину: короткие тексты модель
  различает хуже, см. таблицу по длине.
- В корпусе генераторы до 2025 года; тексты новых моделей могут
  распознаваться хуже.

## Ссылки

```bibtex
@misc{tolstykh2025llmtrace,
  title = {LLMTrace: A Corpus for Classification and Fine-Grained Localization of AI-Written Text},
  author = {Tolstykh, Irina and Tsybina, Aleksandra and Yakubson, Sergey and Kuprashevich, Maksim},
  year = {2025},
  eprint = {2509.21269},
  archivePrefix = {arXiv},
}
```

`cointegrated/rubert-tiny2` описан в [посте автора на Хабре](https://habr.com/ru/post/669674/).

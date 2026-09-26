# Отчёт об обучении: russian-ai-text-detector-modernbert

Файл пишет `scripts/train_transformer.py report` после полного обучения, руками
его не правят. Все числа в машиночитаемом виде лежат рядом, в
[russian-ai-text-detector-modernbert.json](russian-ai-text-detector-modernbert.json). Модель и карточка выложены на
[Hugging Face](https://huggingface.co/toiletsandpaper/russian-ai-text-detector-modernbert).

## Что это

Необязательная модель aiw-ru: дообученный энкодер `deepvk/RuModernBERT-small` оценивает
вероятность, что русский текст написала языковая модель. Пользователь ставит её
командой `aiw-ru models install modernbert`, после чего `aiw-ru classify текст.md`
выдаёт вероятность. В поставке ONNX с весами fp32 на $140$ МБ и
`inference.json`; для вывода нужны onnxruntime и tokenizers, torch не нужен.
На test ROC AUC $0.993$, у LightGBM на признаках aiw-ru $0.943$,
у оценки правил $0.615$.

Это точный вариант среди трансформеров aiw-ru, за точность он платит скоростью и размером.

| Модель | Людей принято за ИИ | ROC AUC | Accuracy | Файл, МБ | M1, мс | x86, мс |
| --- | --: | --: | --: | --: | --: | --: |
| [`modernbert`](https://huggingface.co/toiletsandpaper/russian-ai-text-detector-modernbert), эта модель | $3.4\%$ | $0.9930$ | $0.963$ | $140$ | $207.2$ | — |
| [`transformer`](https://huggingface.co/toiletsandpaper/russian-ai-text-detector-bert) | $7.2\%$ | $0.9869$ | $0.945$ | $29.7$ | $39.8$ | $101.1$ |
| [`mini-frida`](https://huggingface.co/toiletsandpaper/russian-ai-text-detector-mini-frida) | $9.5\%$ | $0.9909$ | $0.948$ | $130$ | $137.2$ | $216.2$ |
| [`lightgbm`](https://huggingface.co/toiletsandpaper/russian-ai-text-detector-lightgbm) | $16.5\%$ | $0.9431$ | $0.868$ | $10.0$ | — | — |

- `modernbert` (эта модель) — самая точная: выбор по умолчанию, но тяжелее трансформера и в несколько раз медленнее.
- `transformer` — почти так же точна, лёгкая и в несколько раз быстрее ModernBERT: для слабой машины.
- `mini-frida` — ROC AUC выше, чем у трансформера, но при пороге 50 % чаще принимает людей за ИИ; в 4 раза тяжелее и в 3 раза медленнее.
- `lightgbm` — самая лёгкая, работает и без onnxruntime (Mac на Intel), но заметно менее точна.

Если установлено несколько моделей, aiw-ru берёт вероятность у первой из них
в порядке таблицы. Порядок задаёт доля людей, принятых за ИИ, а не ROC AUC:
ошибиться в человеке дороже, чем пропустить ИИ-текст.

Задержка — один текст на $512$ токенов в один поток на M1 и на двух ядрах Xeon
виртуальной машины Colab (x86). Размер у LightGBM — файл модели, признаки для неё
считает сам детектор aiw-ru.

## Выбор базы

Кандидаты — небольшие русские энкодеры с открытой лицензией, которые можно
запускать на CPU. Полный [FRIDA](https://huggingface.co/ai-forever/FRIDA)
от ai-forever (MIT) — энкодер T5 на $823$ млн параметров, а у кандидатов ниже $29\text{–}35$ млн. В пилоте вместо
FRIDA взята его дистилляция `sergeyzh/rubert-mini-frida`.

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

На полном train обучены $3$ базы из пилота: самая быстрая, `cointegrated/rubert-tiny2`, самая точная, `deepvk/RuModernBERT-small`, и средняя по обоим, `sergeyzh/rubert-mini-frida`. Все они выпущены как модели aiw-ru: `modernbert` точнее всех и ставится по умолчанию, `transformer` считает быстрее всех, `mini-frida` — промежуточный вариант.

ROC AUC и accuracy посчитаны на полном valid у PyTorch, «ROC AUC int8» — у ONNX int8;
«сменил метку» — доля текстов valid, где int8 и PyTorch расходятся по порогу $0.5$.
Размер и задержка — у варианта из поставки, на $512$ токенах в один поток: на M1 и на
двух ядрах Xeon виртуальной машины Colab.

| База | Имя в aiw-ru | ROC AUC valid | Accuracy valid | ROC AUC int8 | int8 сменил метку | Веса в поставке | Файл, МБ | мс, M1 | мс, x86 |
| --- | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| `cointegrated/rubert-tiny2` | `transformer` | $0.9878$ | $0.946$ | $0.9877$ | $0.3\%$ | int8 | $29.7$ | $39.8$ | $101.1$ |
| `sergeyzh/rubert-mini-frida` | `mini-frida` | $0.9916$ | $0.948$ | $0.9913$ | $0.6\%$ | fp32 | $130$ | $137.2$ | $216.2$ |
| `deepvk/RuModernBERT-small` | `modernbert` | $0.9935$ | $0.963$ | $0.9926$ | $1.1\%$ | fp32 | $140$ | $272.8$ | $367.9$ |

RuModernBERT-small — самая точная из трёх баз, обученных на полном train, и модель aiw-ru по умолчанию. Эта ревизия обучена заново на окне в $8192$ токена вместо $512$: она читает целиком $99.9\%$ текстов test против $86.5\%$ у прежней, а на текстах от $400$ слов её accuracy $0.982$ против $0.966$. Локальное внимание — $\pm 64$ токена, как в исходном ModernBERT; transformers по умолчанию читает окно этой базы как $\pm 128$. На пилотах ($20\,000$ текстов, одна эпоха, два сида) ни длина окна ($512$, $2048$, $8192$), ни ширина локального окна не дали разницы больше разброса между сидами, а при $\pm 64$ локальные слои считают внимание блоками вдвое меньше. Квантование модели вредит: int8 меняет ответ у $1.0\%$ текстов valid. Поэтому в поставке fp32.

## Данные

Русская часть [LLMTrace classification](https://huggingface.co/datasets/iitolstykh/LLMTrace_classification)
под Apache 2.0, [статья](https://arxiv.org/abs/2509.21269), ревизия набора
`252e21e3dca713bc82bc3c1ef73bf533d7c9fc7e`. Части корпуса используются как есть. Модель обучена на $237\,929$ текстах из train, лучший шаг и ранняя остановка — по ROC AUC на $49\,747$ текстах из valid. Test нужен только для
итоговых цифр, в нём $52\,521$ текст.

Метка «ИИ» стоит на любом тексте с участием модели: написанном с нуля
(`create`) и на человеческом тексте, который модель правила, сокращала или
дописывала (`update`, `delete`, `expand`). При обучении от текста длиннее
$8192$ токенов остаётся только начало. Таких в train
$0.1\%$, в valid $0.1\%$, в test
$0.1\%$.

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
| База | [`deepvk/RuModernBERT-small`](https://huggingface.co/deepvk/RuModernBERT-small), ревизия `06d09cc59b2c` |
| Параметров | $34.5$ млн |
| Устройство | NVIDIA GeForce RTX 4090, `cuda` |
| Точность | `bf16`, GPU умеет bf16: диапазон как у fp32, масштабировать потери не нужно |
| Эпох | не больше $3$, лучшая проверка на эпохе $2.00$ |
| Ранняя остановка | после $3$ проверок без роста ROC AUC на valid, сработала |
| Проверок на valid за эпоху | $2$ |
| Оптимизатор | AdamW, скорость `0.0001`, разогрев $6.0\%$ шагов, линейный спад |
| Затухание весов, клиппинг градиента | `0.01`, `1` |
| Пачка | $32$ текста близкой длины, лимит пачки — $16\,384$ токена |
| `max_length` | $8192$ токена |
| Локальное окно внимания | ±$64$ токена |
| `seed` | $1$ |
| Время обучения | $16$ мин, $742$ текста в секунду |
| Файл модели | `model.onnx`, веса fp32, $140$ МБ |
| Обучено | 2026-09-26, коммит `5cd5f2d` |

Проверки на valid по ходу обучения:

| Шаг | Эпоха | loss train | ROC AUC valid | Accuracy valid | logloss valid |
| --- | --: | --: | --: | --: | --: |
| $4301$ | $0.50$ | $0.2778$ | $0.9891$ | $0.947$ | $0.1499$ |
| $8602$ | $1.00$ | $0.1346$ | $0.9916$ | $0.958$ | $0.1125$ |
| $12\,903$ | $1.50$ | $0.0763$ | $0.9925$ | $0.960$ | $0.1371$ |
| $17\,204$ | $2.00$ | $0.0741$ | $0.9934$ | $0.964$ | $0.1179$ |
| $21\,505$ | $2.50$ | $0.0319$ | $0.9923$ | $0.962$ | $0.2010$ |
| $25\,806$ | $3.00$ | $0.0201$ | $0.9928$ | $0.965$ | $0.1758$ |
| $25\,809$ | $3.00$ | $0.0005$ | $0.9928$ | $0.965$ | $0.1758$ |

## Результаты на test

Текст считается написанным ИИ, если вероятность не меньше $0.5$. Текст длиннее $8192$ токенов модель читает только до этой границы: среднее по окнам на valid не точнее. `aiw-ru classify` вдобавок проверяет длинный текст фрагментами по $512$ токенов: на окне целиком ИИ-вставка тонет в человеческом тексте.

| Класс | Precision | Recall | F1 | Текстов |
| --- | --: | --: | --: | --: |
| люди | $0.945$ | $0.966$ | $0.956$ | $21\,417$ |
| ИИ | $0.976$ | $0.962$ | $0.969$ | $31\,104$ |
| accuracy |  |  | $0.963$ | $52\,521$ |
| среднее по классам | $0.961$ | $0.964$ | $0.962$ | $52\,521$ |

| На test | Эта модель | `mini-frida` | `transformer` | LightGBM | Правила aiw-ru |
| --- | --: | --: | --: | --: | --: |
| Accuracy | $0.963$ | $0.948$ | $0.945$ | $0.868$ | $0.503$ |
| ROC AUC | $0.993$ | $0.991$ | $0.987$ | $0.943$ | $0.615$ |
| ROC AUC, люди против текстов с нуля | $0.994$ | $0.992$ | $0.989$ | $0.946$ | $0.597$ |
| F1, среднее по классам | $0.962$ | $0.945$ | $0.943$ | $0.863$ | $0.466$ |
| Людей принято за ИИ | $3.4\%$ | $9.5\%$ | $7.2\%$ | $16.5\%$ | $5.9\%$ |

На valid accuracy $0.964$, ROC AUC $0.993$.

### По жанрам

| Жанр | Людей | ИИ | Accuracy | ROC AUC | ROC AUC LightGBM | ROC AUC правил |
| --- | --: | --: | --: | --: | --: | --: |
| `article` | $5109$ | $5343$ | $0.966$ | $0.994$ | $0.963$ | $0.709$ |
| `factual` | $1383$ | $2209$ | $0.943$ | $0.986$ | $0.896$ | $0.673$ |
| `news` | $3326$ | $3488$ | $0.955$ | $0.993$ | $0.890$ | $0.566$ |
| `poetry` | $3561$ | $3383$ | $0.960$ | $0.989$ | $0.932$ | $0.488$ |
| `question` | $1932$ | $5894$ | $0.969$ | $0.994$ | $0.951$ | $0.729$ |
| `review` | $2678$ | $4019$ | $0.979$ | $0.995$ | $0.960$ | $0.638$ |
| `short_form` | $450$ | $3000$ | $0.941$ | $0.989$ | $0.942$ | $0.556$ |
| `story` | $2978$ | $3768$ | $0.973$ | $0.995$ | $0.967$ | $0.570$ |

### По длине текста

Слова считает детектор aiw-ru, как и для LightGBM.

| Слов в тексте | Текстов | Accuracy | ROC AUC | Accuracy LightGBM | ROC AUC LightGBM |
| --- | --: | --: | --: | --: | --: |
| $0\text{–}49$ | $14\,742$ | $0.940$ | $0.985$ | $0.819$ | $0.902$ |
| $50\text{–}149$ | $17\,766$ | $0.963$ | $0.994$ | $0.849$ | $0.924$ |
| $150\text{–}399$ | $16\,327$ | $0.981$ | $0.997$ | $0.913$ | $0.968$ |
| $400$ и больше | $3686$ | $0.982$ | $0.988$ | $0.952$ | $0.970$ |

### По типу задания генератору

`create` — текст с нуля; `update`, `delete`, `expand` — модель правила,
сокращала или дописывала человеческий текст.

| Задание генератору | ИИ-текстов | Recall | Recall LightGBM |
| --- | --: | --: | --: |
| `create` | $14\,064$ | $0.964$ | $0.899$ |
| `delete` | $5966$ | $0.934$ | $0.812$ |
| `expand` | $4850$ | $0.979$ | $0.945$ |
| `update` | $6224$ | $0.970$ | $0.903$ |

### По моделям-генераторам

Самые частые генераторы в test.

| Модель-генератор | Текстов в test | Recall | Recall LightGBM |
| --- | --: | --: | --: |
| `gpt-3.5` | $4772$ | $0.977$ | $0.929$ |
| `gpt-4o` | $3884$ | $0.976$ | $0.908$ |
| `databricks/dbrx-instruct` | $1513$ | $0.941$ | $0.859$ |
| `google/gemma-2-27b-it` | $1498$ | $0.978$ | $0.888$ |
| `01-ai/Yi-1.5-34B-Chat` | $1472$ | $0.963$ | $0.898$ |
| `yandex/YandexGPT-5-Lite-8B-instruct` | $1469$ | $0.936$ | $0.853$ |
| `CohereForAI/c4ai-command-r-08-2024` | $1461$ | $0.992$ | $0.951$ |
| `unsloth/Llama-3.3-70B-Instruct` | $1452$ | $0.963$ | $0.915$ |
| `Qwen/Qwen2.5-72B-Instruct` | $1449$ | $0.970$ | $0.887$ |
| `mistralai/Ministral-8B-Instruct-2410` | $1396$ | $0.920$ | $0.829$ |
| `Qwen/QwQ-32B` | $1386$ | $0.980$ | $0.885$ |
| `GigaChat-Max` | $1266$ | $0.974$ | $0.931$ |
| `gigachat` | $1181$ | $0.929$ | $0.831$ |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | $942$ | $0.966$ | $0.869$ |
| `WizardLM-2-7B` | $891$ | $0.989$ | $0.926$ |

### Детектор без модели

Оценка правил aiw-ru с порогом $40$, то есть «много примет»:

| Класс | Precision | Recall | F1 | Текстов |
| --- | --: | --: | --: | --: |
| люди | $0.448$ | $0.941$ | $0.607$ | $21\,417$ |
| ИИ | $0.832$ | $0.202$ | $0.325$ | $31\,104$ |
| accuracy |  |  | $0.503$ | $52\,521$ |
| среднее по классам | $0.640$ | $0.572$ | $0.466$ | $52\,521$ |

## Вывод на CPU

Ниже варианты одной и той же модели. int8 идёт в поставку, если на valid теряет не больше $0.002$ ROC AUC и меняет ответ не больше чем у $0.5\%$ текстов; test в этом решении не участвует. Здесь в поставке fp32: int8 меняет ответ у $1.0\%$ текстов valid.

| Вариант | ROC AUC valid | Accuracy valid |
| --- | --: | --: |
| PyTorch на GPU | $0.9934$ | $0.9642$ |
| ONNX fp32 на CPU | $0.9934$ | $0.9642$ |
| ONNX int8 на CPU | $0.9932$ | $0.9636$ |

Варианты сравнивались только на valid, по нему выбираются веса. На test посчитан только вариант из поставки, ONNX fp32.

Сверка с PyTorch на $300$ текстах valid: у ONNX fp32 наибольшая
разница вероятностей $1.0\cdot 10^{-6}$, у int8 $0.832$,
метки int8 совпадают у $99.0\%$ текстов. Вывод по `inference.json`, текст за текстом, как в aiw-ru, расходится с итоговыми вероятностями оценки на $300$ текстах valid не больше чем на $1.4\cdot 10^{-6}$.

Правило для длинных текстов выбрано по valid, в test таких текстов
$63$:

| Правило | ROC AUC valid | Accuracy valid | ROC AUC test | Accuracy test |
| --- | --: | --: | --: | --: |
| начало текста | $1.000$ | $1.000$ | $0.988$ | $0.968$ |
| среднее по окнам | $1.000$ | $0.958$ | $0.960$ | $0.984$ |

Задержка на один текст в миллисекундах, медиана $30$ прогонов, у вызовов дольше секунды $5$, Apple M1, 8 ядер:

| Среда | 128 токенов, потоков: 1 | 128 токенов, потоков: 8 | 512 токенов, потоков: 1 | 512 токенов, потоков: 8 | 2048 токенов, потоков: 1 | 2048 токенов, потоков: 8 | 8192 токена, потоков: 1 | 8192 токена, потоков: 8 |
| --- | --: | --: | --: | --: | --: | --: | --: | --: |
| onnxruntime fp32 | $57.6$ | $43.5$ | $207.2$ | $99.9$ | $1081.6$ | $427.0$ | $7851.4$ | $2966.6$ |
| onnxruntime int8 | $27.9$ | $23.1$ | $107.8$ | $64.3$ | $641.1$ | $320.5$ | $6061.6$ | $2611.3$ |
| torch fp32 | $16.8$ | $18.0$ | $70.9$ | $48.3$ | $632.7$ | $284.6$ | $8664.1$ | $3106.6$ |

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

На этой машине torch считает быстрее onnxruntime: $70.9$ мс против $207.2$ мс у ONNX fp32 на $512$ токенах в один поток. Но torch с transformers занимает на диске в $5.3\text{–}5.8$ раза больше, а на Linux со сборкой torch под CUDA по умолчанию — $5649$ МБ. aiw-ru ставят и на слабые ноутбуки, поэтому в поставке ONNX: установка лёгкая, а ROC AUC на test совпадает с PyTorch до четвёртого знака.

## Как пользоваться

Из командной строки:

```bash
uv tool install "aiw-ru[ml]"
aiw-ru models install modernbert
aiw-ru classify текст.md
```

Из Python без aiw-ru нужны только onnxruntime, tokenizers и numpy:

```python
import json
import os
import re

# Без этой строки onnxruntime 1.30 при импорте включает телеметрию Microsoft,
# и процесс изредка падает на выходе (recursive_mutex lock failed, код 134).
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

import numpy as np
import onnxruntime as ort
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer

files = ["inference.json", "model.onnx*", "tokenizer.json"]
path = snapshot_download("toiletsandpaper/russian-ai-text-detector-modernbert", allow_patterns=files)
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
    prefix, suffix = [spec["cls_id"]], [spec["sep_id"]]
    width = spec.get("read_length", spec["max_length"]) - len(prefix) - len(suffix)
    windows = [ids[i : i + width] for i in range(0, max(len(ids), 1), width)][: spec["max_windows"]]
    windows = [[*prefix, *w, *suffix] for w in windows]
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
`max_length`, токены вокруг окна, правила нормализации из обучения, порог
$0.5$ и число окон для длинных текстов.

## Как воспроизвести

Модель обучена на коммите `5cd5f2d`. Базовая модель — ревизия `06d09cc59b2c4f61cb84e398e96e17aeead5b8b8`, корпус — ревизия
`252e21e3dca713bc82bc3c1ef73bf533d7c9fc7e`, её закрепляет `scripts/llmtrace.py`. Окружение:

| Компонент | Версия |
| --- | --: |
| python | `3.12.12` |
| torch | `2.14.0+cu130` |
| transformers | `5.17.0` |
| tokenizers | `0.23.2` |
| onnxruntime | `1.30.0` |
| numpy | `2.5.3` |
| scikit-learn | `1.9.1` |
| platform | `Linux, x86_64, 32 ядра` |
| cuda | `13.0` |
| device | `cuda` |
| gpu | `NVIDIA GeForce RTX 4090` |
| onnx | `1.23.0` |

Модель обучена на NVIDIA GeForce RTX 4090, `bf16`. На той же машине считались ONNX,
сверка с PyTorch и оценка на valid и test. Задержку в карточке мерил `export`
на M1 в копии папки запуска с ключом `--json export-latency.json`: файл
кладётся в папку запуска, и `report` берёт задержку из него. Абзац о выборе базы передаётся ключом `--why`, его текст приведён в этом отчёте.

```bash
uv run --group train scripts/llmtrace.py fetch --set classification --split train
uv run --group train scripts/llmtrace.py fetch --set classification --split valid
uv run --group train scripts/llmtrace.py fetch --set classification --split test
uv run --group train --group transformer scripts/train_transformer.py train --base deepvk/RuModernBERT-small --epochs 3 --lr 0.0001 --batch 32 --max-length 8192 --evals-per-epoch 2 --patience 3 --batch-tokens 16384 --sliding-window 64 --out ~/.cache/aiw-ru/llmtrace/transformer/full/rumodernbert-small-8k
uv run --group train --group transformer scripts/train_transformer.py export ~/.cache/aiw-ru/llmtrace/transformer/full/rumodernbert-small-8k
uv run --group train --group transformer scripts/train_transformer.py evaluate ~/.cache/aiw-ru/llmtrace/transformer/full/rumodernbert-small-8k --int8-valid-only
uv run --group train --group transformer scripts/train_transformer.py report ~/.cache/aiw-ru/llmtrace/transformer/full/rumodernbert-small-8k --finalist ~/.cache/aiw-ru/llmtrace/transformer/full/rubert-tiny2 --finalist ~/.cache/aiw-ru/llmtrace/transformer/full/rubert-mini-frida --finalist ~/.cache/aiw-ru/llmtrace/transformer/full/rumodernbert-small --related ~/.cache/aiw-ru/llmtrace/transformer/full/rubert-tiny2 --related ~/.cache/aiw-ru/llmtrace/transformer/full/rubert-mini-frida
```

Обучение на GPU не детерминировано до бита, повтор может разойтись в третьем
знаке. Пилот повторяется командой `pilot --base ИМЯ` с настройками по умолчанию
(`--limit 20000 --epochs 1`).

## Ограничения

- Корпус один. На научных статьях, дипломах и диссертациях модель не
  проверялась, а жанры за пределами таблицы по жанрам она не видела.
- Вероятность — не доказательство авторства. Не используйте модель для
  решений о людях: на test она принимает за ИИ $3.4\%$ человеческих текстов.
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

RuModernBERT построен по схеме [ModernBERT](https://arxiv.org/abs/2412.13663).

```bibtex
@misc{deepvk2025rumodernbert,
  title = {RuModernBERT: Modernized BERT for Russian},
  author = {Spirin, Egor and Malashenko, Boris and Sokolov, Andrey},
  url = {https://huggingface.co/deepvk/rumodernbert-base},
  publisher = {Hugging Face},
  year = {2025},
}
```

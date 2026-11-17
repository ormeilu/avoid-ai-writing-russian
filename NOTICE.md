# Происхождение и авторство

Проект — русская адаптация [avoid-ai-writing](https://github.com/conorbronsdon/avoid-ai-writing) Конора Бронсдона (Conor Bronsdon), распространяемого по лицензии MIT. Copyright (c) 2026 Conor Bronsdon.

Из исходного проекта взяты устройство скилла (режимы `rewrite`, `detect`, `edit`; договор о правке; уровни серьёзности P0–P2; трёхуровневый словарь; профили контекста и голоса; правило «никогда не добавляй»; самоисключение для текстов о приметах) и значительная часть каталога примет, которые переведены и переосмыслены для русского языка.

Основой послужила версия 3.36.0. Из версий 3.37–3.39 взяты идеи, код под них написан заново: маркеры `ignore-start`/`ignore-end`, которые исключают фрагмент из оценки; стиль для Vale, собранный из таблиц правил детектора, с манифестом проверенных и приблизительных правил; запись о постановочном открытии в каталоге; утверждения о том, как часто модели пишут оборот, подкреплены замером или помечены как наблюдение; совет не менять тире-связку на двоеточие.

Написано заново для этой адаптации:

- каталог русских примет: канцелярит, кальки с английского, цепочки родительного падежа, русская типографика (тире как грамматика, «ёлочки», десятичная запятая, ГОСТ-списки);
- профили `vak`, `telegram`, `business-email`, `chat` и исключения для научного текста;
- детектор на Python (исходный детектор на JavaScript рассчитан на английский и сюда не переносился);
- под-скилл `antiplagiat` с оценкой по фрагментам и калибровкой по отчётам.

Исходный проект, в свою очередь, ссылается на работы, из которых взяты отдельные приметы: `blader/humanizer`, `brandonwise/humanizer`, `Aboudjem/humanizer-skill`, `isatimur/de-slop`, `welttowelt/stop-slop-refined`, LLM cliché highlighter Саймона Уиллисона и tropes.fyi. Благодарности им переходят и сюда.

## Данные и источники

Правила детектора проверялись и уточнялись по этим работам:

- LLMTrace: A Corpus for Classification and Fine-Grained Localization of AI-Written Text (Irina Tolstykh, Aleksandra Tsybina, Sergey Yakubson, Maksim Kuprashevich; Findings of EMNLP 2026, с. 41585–41622, [ACL Anthology](https://aclanthology.org/2026.findings-emnlp.1951/), [arXiv:2509.21269](https://arxiv.org/abs/2509.21269)). Корпус текстов людей и моделей на русском и английском, [датасет](https://huggingface.co/datasets/iitolstykh/LLMTrace_classification) под лицензией Apache 2.0. Набор с разметкой ИИ-фрагментов [LLMTrace_detection](https://huggingface.co/datasets/iitolstykh/LLMTrace_detection) (тоже Apache 2.0) нужен для сверки фрагментов `antiplagiat` (`scripts/llmtrace.py spans`). На русской части тестового набора (21 417 человеческих текстов и 14 064 сгенерированных с нуля) мы мерили, как часто правила срабатывают на людях. Отсюда новое правило подмены букв и решение не учитывать технические отпечатки в оценке. Данные в репозиторий не входят: человеческие тексты корпуса собраны из сторонних источников со своими лицензиями.
- GigaCheck: Detecting LLM-generated Content via Object-Centric Span Localization (те же авторы, Aleksandr Gordeev, Vladimir Dokholyan; Findings of ACL 2026, [arXiv:2410.23728](https://arxiv.org/abs/2410.23728)). Работа, из которой мы узнали о LLMTrace и о разметке ИИ-фрагментов с точностью до знака.
- Pangram 4 Technical Report (Ben Glickenhaus, Katherine Thai, Jenna Russell, Elyas Masrour, Yue Han, Max Spero, Bradley Emi; Pangram Labs, 2026, [arXiv:2607.27183](https://arxiv.org/abs/2607.27183)). Идея держать технический мусор (невидимые символы, следы PDF и распознавания) отдельно от оценки стиля.
- [Wikipedia: Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing) и её подстраница о разметке [Signs of AI writing/Markup](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing/Markup): разметка ссылок ChatGPT (`turn0search0`, `oaicite`, `contentReference`), следы Gemini, Perplexity и Grok, корректирующие рамки моделей 2026 года и сведения о том, что избыток длинного тире уходит. Русская страница [Признаки текстов, сгенерированных искусственным интеллектом](https://ru.wikipedia.org/wiki/Википедия:Признаки_сгенерированности_текста) (ВП:ИИПГИИ) в основном переводит английскую. Тексты страниц распространяются под CC BY-SA, мы их не копируем: берём описанные приметы и пишем свои примеры. [Описание служебных символов U+E200–U+E204](https://github.com/sanand0/openai-conversations/blob/main/private-unicode-control-characters.md) — из репозитория sanand0/openai-conversations.
- [AI Tells](https://graphite.io/research/ai-tells) (Gregory Druck, Jose Luis Paredes, Ethan Smith; Graphite, 16 сентября 2026). Частоты оборотов GPT-6, Claude Opus 5 и Gemini 3.1 Pro на английских веб-статьях и замер длинного тире по моделям, на них опираются раздел «Почерк модели» в `vocabulary.md` и запись о тире-связке в `typography.md`. Лицензия данных в публикации не указана, мы её не нашли. Проект некоммерческий, поэтому мы ссылаемся на опубликованные цифры так, как если бы это было разрешено; сами данные в репозиторий не входят.

## Модели

Модели для `classify` дообучены на русской части LLMTrace из открытых моделей. Веса базовых моделей в репозиторий не входят, дообученные лежат на Hugging Face в коллекции `aiw-ru` с карточками, где названа база:

- `transformer` — из [cointegrated/rubert-tiny2](https://huggingface.co/cointegrated/rubert-tiny2) Давида Дале (David Dale), лицензия MIT;
- `mini-frida` — из [sergeyzh/rubert-mini-frida](https://huggingface.co/sergeyzh/rubert-mini-frida), лицензия MIT; она получена дистилляцией [ai-forever/FRIDA](https://huggingface.co/ai-forever/FRIDA) (MIT) в [sergeyzh/rubert-mini-sts](https://huggingface.co/sergeyzh/rubert-mini-sts) (MIT);
- `modernbert` — из [deepvk/RuModernBERT-small](https://huggingface.co/deepvk/RuModernBERT-small) (Egor Spirin, Boris Malashenko, Andrey Sokolov), лицензия Apache 2.0; RuModernBERT построен по схеме ModernBERT ([arXiv:2412.13663](https://arxiv.org/abs/2412.13663)).

LightGBM обучен с нуля на признаках детектора.

## Идеи из других проектов

- [humanizer-ru](https://github.com/ilyautov/humanizer-ru) Ильи Утова (Ilya Utov), лицензия MIT, Copyright (c) 2026 Ilya Utov. Из его «факт-замка» (`facts.py`) взята идея сверки фактов в `aiw-ru validate`: снятые оговорки, новые имена, числа словами, месяцы и слова-претензии. Код написан заново и без морфологии (pymorphy3 и razdel детектору недоступны): на словарях и регистре букв, со стопками оговорок и семействами синонимов. Оттуда же идеи проверки правки свежим агентом, который видит только исходник, итог и задание (`references/review.md`), теста на переносимость в SKILL.md и примет детектора «почерк модели» (`model-idiolect`), «обвязка ответа чата» (`chat-wrapper`) и «текст оборван на полуслове» (`truncated`). Тексты, словари и замеры на LLMTrace свои. Из версий 3.29–3.32 и PR #184 (ноябрь 2026) взяты идеи предупреждения о переставленных числах в `aiw-ru validate`, правила «после текста ничего», когда пользователь просит только текст, и примета «швы правки» в `references/review.md`.
- [humanizer-skill](https://github.com/Aboudjem/humanizer-skill) (Aboudjem), лицензия MIT. Оттуда приметы «спор с безымянным оппонентом» и «оговорка рядом с уверенностью» (P54, P55) и проверка, что число, переехавшее в ссылку, `validate` не считает потерянным.
- [stop-slop-refined](https://github.com/welttowelt/stop-slop-refined) (welttowelt), лицензия MIT. Оттуда примета «квалификация вместо довода» и замечание, что двоеточие на месте тире сохраняет тот же рывок.
- [humanizer](https://github.com/blader/humanizer) (blader), лицензия MIT. Оттуда требование к режиму `edit` называть, сколько разделов файла проверено.

«Антиплагиат» — товарный знак АО «Антиплагиат». Проект с ним не связан, использует только публично известные сведения о том, как устроен отчёт, и не воспроизводит его классификатор. Уровни признаков ИИ, версия модели в отчёте и цифры точности взяты из новостей разработчика за 2026 год, ссылки на них — в `skills/antiplagiat/references/antiplagiat.md`.

С «Думейтом» (domate.ru) проект тоже не связан: сведения о его тепловой карте взяты из публикаций разработчика и проверок автора проекта, они собраны в `docs/domate.md` и `skills/antiplagiat/references/domate.md`.

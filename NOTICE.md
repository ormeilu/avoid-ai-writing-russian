# Происхождение и авторство

Проект — русская адаптация [avoid-ai-writing](https://github.com/conorbronsdon/avoid-ai-writing) Конора Бронсдона (Conor Bronsdon), распространяемого по лицензии MIT. Copyright (c) 2026 Conor Bronsdon.

Из исходного проекта взяты устройство скилла (режимы `rewrite`, `detect`, `edit`; договор о правке; уровни серьёзности P0–P2; трёхуровневый словарь; профили контекста и голоса; правило «никогда не добавляй»; самоисключение для текстов о приметах) и значительная часть каталога примет, которые переведены и переосмыслены для русского языка.

Написано заново для этой адаптации:

- каталог русских примет: канцелярит, кальки с английского, цепочки родительного падежа, русская типографика (тире как грамматика, «ёлочки», десятичная запятая, ГОСТ-списки);
- профили `vak`, `telegram`, `business-email`, `chat` и исключения для научного текста;
- детектор на Python (исходный детектор на JavaScript рассчитан на английский и сюда не переносился);
- под-скилл `antiplagiat` с оценкой по фрагментам и калибровкой по отчётам.

Исходный проект, в свою очередь, ссылается на работы, из которых взяты отдельные приметы: `blader/humanizer`, `brandonwise/humanizer`, `Aboudjem/humanizer-skill`, `isatimur/de-slop`, `welttowelt/stop-slop-refined`, LLM cliché highlighter Саймона Уиллисона и tropes.fyi. Благодарности им переходят и сюда.

## Данные и источники

Правила детектора проверялись и уточнялись по этим работам:

- LLMTrace: A Corpus for Classification and Fine-Grained Localization of AI-Written Text (Irina Tolstykh, Aleksandra Tsybina, Sergey Yakubson, Maksim Kuprashevich; Findings of EMNLP 2026, [arXiv:2509.21269](https://arxiv.org/abs/2509.21269)). Корпус текстов людей и моделей на русском и английском, [датасет](https://huggingface.co/datasets/iitolstykh/LLMTrace_classification) под лицензией Apache 2.0. На русской части тестового набора (21 417 человеческих текстов и 14 064 сгенерированных) мы мерили, как часто правила срабатывают на людях. Отсюда новое правило подмены букв и решение не учитывать технические отпечатки в оценке. Данные в репозиторий не входят: человеческие тексты корпуса собраны из сторонних источников со своими лицензиями.
- GigaCheck: Detecting LLM-generated Content via Object-Centric Span Localization (те же авторы, Aleksandr Gordeev, Vladimir Dokholyan; Findings of ACL 2026, [arXiv:2410.23728](https://arxiv.org/abs/2410.23728)). Работа, из которой мы узнали о LLMTrace и о разметке ИИ-фрагментов с точностью до знака.
- [Технический отчёт Pangram 4](https://pangram-public.s3.us-east-1.amazonaws.com/pdf/pangram_4_technical_report.pdf) (Pangram Labs, 2026). Идея держать технический мусор (невидимые символы, следы PDF и распознавания) отдельно от оценки стиля.
- [Wikipedia: Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing), раздел о разметке ссылок ChatGPT (`turn0search0`, `oaicite`, `contentReference`), и [описание служебных символов U+E200–U+E204](https://github.com/sanand0/openai-conversations/blob/main/private-unicode-control-characters.md) из репозитория sanand0/openai-conversations.

«Антиплагиат» — товарный знак АО «Антиплагиат». Проект с ним не связан, использует только публично известные сведения о том, как устроен отчёт, и не воспроизводит его классификатор.

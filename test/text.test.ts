import { describe, expect, test } from "bun:test";
import { blocks, cv, lineCol, mattr, mean, prepare, sentences, words } from "../src/text.ts";

const ZW = String.fromCharCode(0x200b);
const SHY = String.fromCharCode(0x00ad);

describe("sentences", () => {
  const texts = (s: string): string[] => sentences(s).map((x) => x.text);

  test("обычное разбиение", () => {
    expect(texts("Первое. Второе! Третье? Четвёртое…")).toEqual(["Первое.", "Второе!", "Третье?", "Четвёртое…"]);
  });

  test("сокращения не рвут предложение", () => {
    expect(texts("См. рис. 3 и табл. 2, т. е. всё. Дальше.")).toHaveLength(2);
    expect(texts("Это было в 1990 г. в Ростове. Потом нет.")).toHaveLength(2);
  });

  test("инициалы не рвут предложение", () => {
    expect(texts("Работу выполнил А. С. Иванов. Проверил другой.")).toHaveLength(2);
  });

  test("строчная буква после точки — не новое предложение", () => {
    expect(texts("Версия 2. работает иначе.")).toHaveLength(1);
  });

  test("кавычки и тире в начале следующего предложения", () => {
    expect(texts("Он сказал. «Нет», — ответили ему.")).toHaveLength(2);
  });

  test("ручной перенос внутри абзаца не разделяет", () => {
    expect(texts("Эксперты\nсчитают, что это важно. Второе\nпредложение.")).toEqual([
      "Эксперты\nсчитают, что это важно.",
      "Второе\nпредложение.",
    ]);
  });

  test("пустая строка и пункт списка разделяют", () => {
    expect(texts("строка без точки\n\nещё строка")).toHaveLength(2);
    expect(texts("Список:\n- первый\n- второй\n1. третий")).toHaveLength(4);
  });

  test("смещения указывают в текст", () => {
    const s = "Раз. Два три.";
    for (const x of sentences(s, 100)) expect(s.slice(x.start - 100, x.end - 100)).toBe(x.text);
  });

  test("длинная цепочка точек разбирается за линейное время", () => {
    const t = performance.now();
    expect(texts(`Текст ${".".repeat(50000)} конец. Дальше.`)).toHaveLength(2);
    expect(performance.now() - t).toBeLessThan(500);
  });

  test("подсчёт слов", () => {
    expect(sentences("Один два-три четыре.")[0]?.words).toBe(3);
  });
});

describe("words", () => {
  test("дефисные слова и апострофы — одно слово", () => {
    expect(words("кто-то из IT-отдела О’Нил")).toEqual(["кто-то", "из", "IT-отдела", "О’Нил"]);
  });

  test("числа — слова", () => {
    expect(words("в 2024 году")).toEqual(["в", "2024", "году"]);
  });
});

describe("prepare", () => {
  test("ё заменяется на е без сдвига", () => {
    const p = prepare("Ёлка ещё");
    expect(p.text).toBe("Елка еще");
    expect(p.text.length).toBe(p.source.length);
  });

  test("невидимые символы удаляются, смещения восстанавливаются", () => {
    const src = `а${ZW}б${SHY}в`;
    const p = prepare(src);
    expect(p.text).toBe("абв");
    expect(p.invisible.map((i) => i.index)).toEqual([1]);
    expect(p.softHyphens).toEqual([3]);
    expect(p.toSource.slice(0, 3)).toEqual([0, 2, 4]);
  });

  test("BOM в начале и соединитель внутри эмодзи — не вставки", () => {
    const bom = String.fromCharCode(0xfeff);
    const zwj = String.fromCharCode(0x200d);
    expect(prepare(`${bom}Текст`).invisible).toEqual([]);
    expect(prepare(`${bom}Текст`).text).toBe("Текст");
    expect(prepare(`Команда 👨${zwj}💻 и 🏳${String.fromCharCode(0xfe0f)}${zwj}🌈`).invisible).toEqual([]);
    expect(prepare(`Ра${zwj}бота`).invisible).toHaveLength(1);
    expect(prepare(`Текст${bom} дальше`).invisible).toHaveLength(1);
  });

  test("неразрывный дефис и U+2010 становятся обычным дефисом без сдвига", () => {
    const src = `по${String.fromCharCode(0x2011)}настоящему и кто${String.fromCharCode(0x2010)}то`;
    const p = prepare(src);
    expect(p.text).toBe("по-настоящему и кто-то");
    expect(p.text.length).toBe(src.length);
    expect(words(p.text)).toEqual(["по-настоящему", "и", "кто-то"]);
  });

  test("латиница в русском слове исправляется для поиска", () => {
    const src = `р${String.fromCharCode(0x61)}бота`;
    const p = prepare(src);
    expect(p.text).toBe("работа");
    expect(p.homoglyphs).toHaveLength(1);
  });

  test("полностью латинские и законно смешанные слова не трогаются", () => {
    const p = prepare("Python и Wi-Fi-модуль, IT-отдел");
    expect(p.homoglyphs).toEqual([]);
  });

  test("латинское сокращение через дефис с русским словом — не подмена", () => {
    const p = prepare("HTTP-запрос на TCP-порт, PHP-скрипт ответил OK-кодом");
    expect(p.homoglyphs).toEqual([]);
    expect(p.text).toContain("HTTP-запрос");
  });

  test("латинская основа с русским окончанием — не подмена", () => {
    for (const s of ["PHPшник", "OKей", "CEOшный", "на Pythonе", "в Excelе", "iPhoneа", "в TeXе", "вPython"]) {
      expect(prepare(s).homoglyphs, s).toEqual([]);
    }
  });

  test("латинская i в украинском слове и «х» как знак умножения — не подмена", () => {
    const i = String.fromCharCode(0x69);
    const kha = String.fromCharCode(0x445);
    for (const s of [`ш${i}р${i}к`, `р${i}зних`, `F${kha}G`, `m${kha}n`]) expect(prepare(s).homoglyphs, s).toEqual([]);
  });

  test("ударение латинской буквой с акутом — не подмена", () => {
    const a = String.fromCharCode(0xe1);
    for (const s of [`К${a}рмен`, `С${a}нта-Фе`]) expect(prepare(s).homoglyphs, s).toEqual([]);
  });

  test("подмена по смене алфавитов внутри слова", () => {
    const ch = (...codes: number[]): string => String.fromCharCode(...codes);
    const spoofed = [
      `P${ch(0x443)}thon`, // кириллическая «у» внутри латинского слова
      `m${ch(0x43e)}del`, // кириллическая «о»
      `${ch(0x420)}ython`, // кириллическая «Р» в начале латинского слова
      `р${ch(0x61)}бота`, // латинская «a» внутри русского слова
      `работ${ch(0x61)}`, // латинская «a» в конце
      `${ch(0x70)}абота`, // латинская «p» в начале
      `${ch(0x70, 0x61)}бота`, // две латинские буквы в начале
      `${ch(0x65)}щ${ch(0x65)}`, // латиницы больше, но «щ» выдаёт русское слово
      `${ch(0x4d)}Ч${ch(0x43)}`, // сокращение с латинскими M и C
      `${ch(0x42)}Контакте`, // латинская B перед заглавной кириллицей
      `${ch(0x43, 0x41)}ДЫ`, // заглавное слово: не сокращение с окончанием
      `У${ch(0x4f, 0x50, 0x58, 0x4f)}ЛА`, // латиница внутри заглавного слова
    ];
    for (const s of spoofed) expect(prepare(s).homoglyphs, s).toHaveLength(1);
    expect(prepare(`P${ch(0x443)}thon`).text).toBe("Python");
    expect(prepare(`р${ch(0x61)}бот${ch(0x61)}`).text).toBe("работа");
  });

  test("подмена внутри части дефисного слова находится", () => {
    const p = prepare(`HTTP-з${String.fromCharCode(0x61)}прос`);
    expect(p.homoglyphs).toHaveLength(1);
    expect(p.text).toBe("HTTP-запрос");
  });

  test("маскирование сохраняет длину и переводы строк", () => {
    const src = "---\na: 1\n---\nТекст `код` «цитата» [12] [@key] $x+1$ https://a.ru/b\n| т | а |\n> блок";
    const p = prepare(src);
    expect(p.prose.length).toBe(src.length);
    expect(p.prose.split("\n").length).toBe(src.split("\n").length);
    for (const hidden of ["код", "цитата", "12", "@key", "x+1", "a.ru", "т | а", "блок", "a: 1"]) {
      expect(p.prose.includes(hidden), hidden).toBe(false);
    }
    expect(p.prose).toContain("Текст");
  });

  test("блок кода скрыт и в noCode, цитаты остаются", () => {
    const src = "Текст «цитата»\n```\nкод\n```";
    const p = prepare(src);
    expect(p.noCode).not.toContain("код");
    expect(p.noCode).toContain("цитата");
  });
});

describe("blocks", () => {
  const kinds = (s: string): string[] => blocks(prepare(s)).map((b) => b.kind);

  test("заголовки, проза, списки", () => {
    expect(kinds("# Заголовок\nАбзац один.\n\n- пункт\n- пункт\n\nАбзац два.")).toEqual([
      "heading",
      "prose",
      "list",
      "prose",
    ]);
  });

  test("таблица, цитата и код распознаются", () => {
    expect(kinds("| a | b |\n|---|---|\n\n> цитата\n\n```\nx\n```")).toEqual(["table", "quote", "code"]);
  });

  test("нумерованный список", () => {
    expect(kinds("1. раз\n2. два")).toEqual(["list"]);
  });
});

describe("статистика", () => {
  test("mean и cv", () => {
    expect(mean([2, 4, 6])).toBe(4);
    expect(cv([5, 5, 5])).toBe(0);
    expect(cv([1])).toBe(0);
    expect(cv([1, 9])).toBeGreaterThan(1);
  });

  test("MATTR: повторы снижают разнообразие", () => {
    const rich = Array.from({ length: 300 }, (_, i) => `слово${i}`);
    const poor = Array.from({ length: 300 }, (_, i) => `слово${i % 20}`);
    expect(mattr(rich)).toBe(1);
    expect(mattr(poor)).toBeLessThan(0.3);
    expect(mattr([])).toBe(0);
  });

  test("lineCol", () => {
    const p = prepare("аб\nвг\nде");
    expect(lineCol(p.lineStarts, 0)).toEqual({ line: 1, column: 1 });
    expect(lineCol(p.lineStarts, 4)).toEqual({ line: 2, column: 2 });
    expect(lineCol(p.lineStarts, 6)).toEqual({ line: 3, column: 1 });
  });
});

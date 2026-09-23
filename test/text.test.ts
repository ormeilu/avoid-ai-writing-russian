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

  test("перевод строки разделяет", () => {
    expect(texts("строка без точки\nещё строка")).toHaveLength(2);
  });

  test("смещения указывают в текст", () => {
    const s = "Раз. Два три.";
    for (const x of sentences(s, 100)) expect(s.slice(x.start - 100, x.end - 100)).toBe(x.text);
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
    expect(p.invisible.map((i) => i.index)).toEqual([1, 3]);
    expect(p.toSource.slice(0, 3)).toEqual([0, 2, 4]);
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

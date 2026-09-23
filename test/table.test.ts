import { describe, expect, test } from "bun:test";
import { layout, table, truncate, wrapText } from "../src/table.ts";

describe("таблицы", () => {
  test("перенос по словам", () => {
    expect(wrapText("один два три четыре", 9)).toEqual(["один два", "три", "четыре"]);
    expect(wrapText("", 5)).toEqual([""]);
    expect(wrapText("сверхдлинноеслово", 6)).toEqual(["сверхд", "линное", "слово"]);
  });

  test("обрезка с многоточием", () => {
    expect(truncate("короткий", 20)).toBe("короткий");
    expect(truncate("довольно длинная строка", 10)).toBe("довольно…");
  });

  test("сужаются только flex-колонки и не уже min", () => {
    const cols = [{ title: "A" }, { title: "Текст", flex: true, min: 8 }];
    const rows = [["x", "очень длинный текст ячейки, который не влезает"]];
    expect(layout(cols, rows, { width: 30 })).toEqual([1, 25]);
    expect(layout(cols, rows, { width: 5 })).toEqual([1, 8]);
  });

  test("строки не длиннее ширины, перенос держит колонки", () => {
    const lines = table(
      [{ title: "Уровень" }, { title: "Что", flex: true, wrap: true, min: 10 }],
      [
        ["P0", "первая длинная подсказка, которой нужен перенос"],
        ["P1", "коротко"],
      ],
      { width: 32 },
    );
    for (const l of lines) expect(l.length).toBeLessThanOrEqual(32);
    expect(lines[0]).toBe("  Уровень  Что");
    expect(lines[1]).toMatch(/^ {2}─+ {2}─+$/);
    expect(lines[2]).toStartWith("  P0       первая");
    expect(lines[3]).toStartWith("           ");
    expect(lines.at(-1)).toBe("  P1       коротко");
  });

  test("раскраска не захватывает пробелы выравнивания", () => {
    const lines = table([{ title: "A" }, { title: "B" }], [["x", "y"]], {
      width: 40,
      paint: (_r, _c, text) => `<${text}>`,
    });
    expect(lines[2]).toBe("  <x>  <y>");
  });

  test("выравнивание по правому краю", () => {
    const lines = table([{ title: "Слов", align: "right" }], [["7"]], { width: 40 });
    expect(lines[2]).toBe("     7");
  });
});

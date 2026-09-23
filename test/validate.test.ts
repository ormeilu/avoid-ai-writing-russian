import { describe, expect, test } from "bun:test";
import { validate } from "../src/index.ts";

describe("validate", () => {
  const before = [
    "---",
    "title: Статья",
    "---",
    "",
    "# Введение",
    "",
    'Данный метод играет ключевую роль: точность 0.93 [12], см. "отчёт" и https://example.com/a?utm_source=chatgpt.com.',
    "",
    "```python",
    "x = 1",
    "```",
  ].join("\n");

  test("разрешённые правки проходят", () => {
    const after = before
      .replace("Данный метод играет ключевую роль", "Метод повышает полноту")
      .replace("0.93", "0,93")
      .replace('"отчёт"', "«отчёт»")
      .replace("?utm_source=chatgpt.com", "");
    const r = validate(before, after);
    expect(r.violations).toEqual([]);
    expect(r.ok).toBe(true);
  });

  test("изменённое число, код и ссылка ловятся", () => {
    const after = before.replace("0.93", "0.95").replace("x = 1", "x = 2").replace("[12]", "[13]");
    const kinds = validate(before, after).violations.map((v) => v.kind);
    expect(kinds).toEqual(expect.arrayContaining(["число", "код", "ссылка на литературу"]));
  });

  test("YAML-шапка и заголовки", () => {
    const after = before.replace("title: Статья", "title: Другая").replace("# Введение", "## Введение");
    const kinds = validate(before, after).violations.map((v) => v.kind);
    expect(kinds).toEqual(expect.arrayContaining(["yaml", "заголовки"]));
  });

  test("рост числа находок — нарушение", () => {
    const r = validate("Метод работает.", "Давайте разберёмся: метод играет ключевую роль.");
    expect(r.violations.map((v) => v.kind)).toContain("находки");
  });
});

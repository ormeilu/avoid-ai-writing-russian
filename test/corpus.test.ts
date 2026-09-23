/**
 * Корпус: живые тексты разных жанров не должны давать серьёзных находок,
 * шаблонные тексты моделей в тех же жанрах — должны. Каждый новый
 * детектор проверяется здесь на ложные срабатывания.
 */

import { describe, expect, test } from "bun:test";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { analyze, antiplagiat, type ContextMode } from "../src/index.ts";

const DIR = join(import.meta.dir, "fixtures/corpus");
const MODE: Record<string, ContextMode> = {
  vak: "academic",
  docs: "technical",
  blog: "general",
  telegram: "social",
  email: "general",
  chat: "chat",
};
const load = (kind: string): { genre: string; text: string }[] =>
  readdirSync(join(DIR, kind))
    .filter((f) => f.endsWith(".md"))
    .map((f) => ({ genre: f.replace(/\.md$/, ""), text: readFileSync(join(DIR, kind, f), "utf8") }));

const human = load("human");
const ai = load("ai");

describe("живые тексты", () => {
  test.each(human)("$genre: нет находок P0 и P1", ({ genre, text }) => {
    const r = analyze(text, { context: MODE[genre] });
    expect(r.issues.filter((i) => i.severity !== "P2").map((i) => `${i.type}: ${i.text}`)).toEqual([]);
    expect(r.score).toBeLessThan(15);
    expect(r.suspicious).toBe(false);
  });

  test.each(human)("$genre: antiplagiat не метит ни одного фрагмента", ({ text }) => {
    expect(antiplagiat(text).aiShare).toBe(0);
  });

  test("то же в общем режиме: жанровые поблажки не маскируют ошибки", () => {
    for (const { genre, text } of human) {
      if (genre === "chat") continue;
      const p0 = analyze(text).issues.filter((i) => i.severity === "P0");
      expect(p0, genre).toEqual([]);
    }
  });
});

describe("шаблонные тексты моделей", () => {
  test.each(ai)("$genre: высокая оценка", ({ genre, text }) => {
    expect(analyze(text, { context: MODE[genre] }).score).toBeGreaterThanOrEqual(60);
  });

  test.each(ai.filter((t) => t.genre !== "telegram"))("$genre: antiplagiat метит больше половины", ({ text }) => {
    expect(antiplagiat(text).aiShare).toBeGreaterThan(50);
  });

  test("каждый жанр живых текстов оценён ниже любого текста модели", () => {
    const maxHuman = Math.max(...human.map((h) => analyze(h.text, { context: MODE[h.genre] }).score));
    const minAi = Math.min(...ai.map((a) => analyze(a.text, { context: MODE[a.genre] }).score));
    expect(maxHuman).toBeLessThan(minAi);
  });

  test("у каждого жанра есть пара", () => {
    expect(ai.map((a) => a.genre).sort()).toEqual(
      human
        .map((h) => h.genre)
        .filter((g) => g !== "chat")
        .sort(),
    );
  });
});

describe("устойчивость", () => {
  test("перестановка абзацев не меняет набор находок", () => {
    const text = ai.find((a) => a.genre === "blog")?.text ?? "";
    const paras = text.split("\n\n");
    const shuffled = [paras[0], ...paras.slice(1).reverse()].join("\n\n");
    const types = (t: string): string[] => [...new Set(analyze(t).issues.map((i) => i.type))].sort();
    expect(types(shuffled)).toEqual(types(text));
  });

  test("ё и е дают одинаковый результат", () => {
    const text = ai.find((a) => a.genre === "vak")?.text ?? "";
    const yo = text.replace(/е/g, (c, i: number) => (i % 7 === 0 ? "ё" : c));
    expect(analyze(yo).issues.length).toBe(analyze(text).issues.length);
  });

  test("CRLF обрабатывается как LF", () => {
    const text = ai.find((a) => a.genre === "vak")?.text ?? "";
    expect(analyze(text.replace(/\n/g, "\r\n")).score).toBe(analyze(text).score);
  });

  test("большой документ обрабатывается быстро", () => {
    const big = ai
      .map((a) => a.text)
      .join("\n\n")
      .repeat(10);
    const t0 = performance.now();
    analyze(big);
    antiplagiat(big);
    expect(performance.now() - t0).toBeLessThan(4000);
  });
});

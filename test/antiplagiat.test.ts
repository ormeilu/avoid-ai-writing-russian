import { describe, expect, test } from "bun:test";
import { antiplagiat, balancedAccuracy, DEFAULT_MODEL, fit, labelFragments, type Sample } from "../src/index.ts";

const fixture = (name: string): Promise<string> => Bun.file(new URL(`./fixtures/${name}`, import.meta.url)).text();

describe("antiplagiat", () => {
  test("ИИ-текст получает большую долю, живой — нулевую", async () => {
    expect((await antiplagiat(await fixture("ai.md"))).aiShare).toBeGreaterThan(50);
    expect((await antiplagiat(await fixture("human.md"))).aiShare).toBe(0);
  });

  test("фрагменты покрывают прозу и указывают строки", async () => {
    const r = antiplagiat(await fixture("ai.md"));
    expect(r.fragments.length).toBeGreaterThan(0);
    for (const f of r.fragments) {
      expect(f.line).toBeGreaterThan(0);
      expect(f.endLine).toBeGreaterThanOrEqual(f.line);
      expect(f.probability).toBeGreaterThanOrEqual(0);
      expect(f.probability).toBeLessThanOrEqual(1);
    }
  });

  test("подозрительный документ", () => {
    const zw = String.fromCharCode(0x200b);
    expect(antiplagiat(`Обычный${zw} текст.`).suspicious).toBe(true);
  });

  test("разметка по подсвеченным кускам", async () => {
    const doc = `${await fixture("ai.md")}\n\n${await fixture("human.md")}`;
    const marked = ["Более того, использование ИИ способствует оптимизации процессов и повышению эффективности работы медицинских учреждений."];
    const samples = labelFragments(doc, marked);
    expect(samples.filter((s) => s.y === 1)).toHaveLength(1);
  });

  test("калибровка не ухудшает точность на обучающих данных", () => {
    const samples: Sample[] = [
      { f: [2, 0.8, 0.9, 1, 0.9, 0.3, 0.2], y: 1 },
      { f: [1.5, 0.7, 0.8, 0.5, 0.8, 0.2, 0.1], y: 1 },
      { f: [0.1, 0.4, 0.6, 0.2, 0.9, 0.1, 0.1], y: 1 },
      { f: [0, 0.2, 0.3, 0, 0.2, 0, 0], y: 0 },
      { f: [0.2, 0.3, 0.5, 0.1, 0.3, 0, 0.1], y: 0 },
      { f: [0, 0.1, 0.2, 0, 0.1, 0, 0], y: 0 },
    ];
    const model = fit(samples);
    expect(balancedAccuracy(model, samples)).toBeGreaterThanOrEqual(balancedAccuracy(DEFAULT_MODEL, samples));
  });
});

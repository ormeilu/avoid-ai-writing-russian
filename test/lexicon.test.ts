import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { ALL_LEXICON, phrase, TYPE_LABELS, WEIGHTS } from "../src/index.ts";

describe("контракт словаря", () => {
  test("у каждого типа есть подпись и вес", () => {
    for (const e of ALL_LEXICON) {
      expect(TYPE_LABELS[e.type], e.type).toBeDefined();
      expect(WEIGHTS[e.type], e.type).toBeDefined();
    }
    for (const t of Object.keys(WEIGHTS)) expect(TYPE_LABELS[t], t).toBeDefined();
  });

  test("идентификаторы правил уникальны внутри типа", () => {
    const seen = new Set<string>();
    for (const e of ALL_LEXICON) {
      const key = `${e.type}/${e.id}`;
      expect(seen.has(key), key).toBe(false);
      seen.add(key);
    }
  });

  test("мини-язык фраз", () => {
    const re = new RegExp(phrase("игра* (ключев*|важн*)( роль)?"), "iu");
    expect(re.test("играет ключевую роль")).toBe(true);
    expect(re.test("играла важную")).toBe(true);
    expect(re.test("переиграет важную")).toBe(false);
    expect(new RegExp(phrase("\\*шепотом\\*"), "iu").test("*шепотом*")).toBe(true);
  });

  test("регулярные выражения не зацикливаются на длинном тексте", () => {
    const long = "слово ".repeat(20000);
    const t0 = performance.now();
    for (const e of ALL_LEXICON) {
      e.re.lastIndex = 0;
      long.match(e.re);
    }
    expect(performance.now() - t0).toBeLessThan(3000);
  });

  test("в репозитории нет невидимых символов", () => {
    const root = join(import.meta.dir, "..");
    const glob = new Bun.Glob("{src,test,skills,examples}/**/*.{ts,md,json}");
    const files = [
      ...glob.scanSync(root),
      "README.md",
      "CONTRIBUTING.md",
      "AGENTS.md",
      "CLAUDE.md",
      "NOTICE.md",
      "CHANGELOG.md",
    ];
    expect(files.length).toBeGreaterThan(10);
    for (const f of files) {
      const s = readFileSync(join(root, f), "utf8");
      expect(/[\u200B-\u200D\u2060\uFEFF]/.test(s), f).toBe(false);
    }
  });
});

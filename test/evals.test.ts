import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { type EvalCase, extractFinal, grade } from "../evals/grade.ts";
import { buildPrompt } from "../evals/run.ts";

const DIR = join(import.meta.dir, "../evals");
const { cases } = JSON.parse(readFileSync(join(DIR, "cases.json"), "utf8")) as { cases: EvalCase[] };
const byId = (id: string): EvalCase => cases.find((c) => c.id === id) as EvalCase;
const golden = (name: string): string => readFileSync(join(DIR, "golden", name), "utf8");

describe("cases.json", () => {
  test("идентификаторы уникальны, у каждого случая есть проверки", () => {
    expect(new Set(cases.map((c) => c.id)).size).toBe(cases.length);
    for (const c of cases) {
      expect(Object.keys(c.checks).length, c.id).toBeGreaterThan(0);
      expect(["avoid-ai-writing-russian", "antiplagiat"]).toContain(c.skill);
      expect(c.input.length, c.id).toBeGreaterThan(20);
    }
  });

  test("регулярные выражения в проверках компилируются", () => {
    for (const c of cases) {
      for (const re of [...(c.checks.finalMatches ?? []), ...(c.checks.responseMatches ?? [])]) {
        expect(() => new RegExp(re, "iu"), `${c.id}: ${re}`).not.toThrow();
      }
    }
  });

  test("запрещённые строки действительно есть в исходнике, сохраняемые — тоже", () => {
    for (const c of cases) {
      for (const s of c.checks.forbid ?? []) {
        if (c.id === "injection") continue; // проверяет, что агент не дописал своё
        expect(c.input, `${c.id}: forbid «${s}»`).toContain(s.replace("0,93", "0.93"));
      }
      for (const s of c.checks.preserve ?? []) {
        if (c.id === "vak-typography") continue; // ожидается исправленная форма
        expect(c.input, `${c.id}: preserve «${s}»`).toContain(s);
      }
    }
  });

  test("есть случаи на все режимы и оба скилла", () => {
    const ids = cases.map((c) => c.id);
    expect(ids).toEqual(expect.arrayContaining(["detect-only", "clean-noop", "injection", "invented-specifics"]));
    expect(cases.some((c) => c.skill === "antiplagiat")).toBe(true);
  });

  test("промпт содержит скилл, запрос и текст", () => {
    const p = buildPrompt(byId("antiplagiat-borrowing"));
    expect(p).toContain("name: antiplagiat");
    expect(p).toContain("name: avoid-ai-writing-russian");
    expect(p).toContain("<text>");
    expect(p).toContain("перефразируй синонимами");
  });
});

describe("extractFinal", () => {
  test("жирный заголовок", () => {
    expect(extractFinal("**Итоговый текст**\n\nТекст.\n\n**Изменения**\n\nх")).toBe("Текст.");
  });

  test("заголовок Markdown и цитата", () => {
    expect(extractFinal("## Итоговый текст\n\n> Строка один.\n> Строка два.\n\n## Проверка\n")).toBe(
      "Строка один.\nСтрока два.",
    );
  });

  test("блок кода", () => {
    expect(extractFinal("**Итоговый текст:**\n```\nТекст.\n```\n**Проверка**")).toBe("Текст.");
  });

  test("нет раздела", () => {
    expect(extractFinal("Просто ответ")).toBeUndefined();
  });
});

describe("эталонные ответы", () => {
  test("хорошая правка проходит", () => {
    expect(grade(byId("blog-rewrite"), golden("blog-rewrite.md")).failures).toEqual([]);
  });

  test("правка с выдуманными числами и оставленной подушкой — провал", () => {
    const g = grade(byId("blog-rewrite"), golden("blog-rewrite.bad.md"));
    expect(g.pass).toBe(false);
    expect(g.failures.join("\n")).toContain("выдуманные числа: 23, 2024");
    expect(g.failures.join("\n")).toContain("осталось: Стоит отметить");
  });

  test("чистый текст без изменений проходит", () => {
    expect(grade(byId("clean-noop"), golden("clean-noop.md")).failures).toEqual([]);
  });

  test("режим detect проходит", () => {
    expect(grade(byId("detect-only"), golden("detect-only.md")).failures).toEqual([]);
  });

  test("изменённая цитата — провал", () => {
    const g = grade(byId("protected-quote"), golden("protected-quote.bad.md"));
    expect(g.pass).toBe(false);
    expect(g.failures.join("\n")).toContain("пропало: «Данный подход");
  });

  test("ответ без итогового текста — провал для правки", () => {
    expect(grade(byId("vak-protected"), "Всё хорошо.").failures).toContain("в ответе нет раздела «Итоговый текст»");
  });

  test("эталон antiplagiat-asks-reports: агент просит отчёты для калибровки", () => {
    expect(grade(byId("antiplagiat-asks-reports"), golden("antiplagiat-asks-reports.md")).failures).toEqual([]);
  });
});

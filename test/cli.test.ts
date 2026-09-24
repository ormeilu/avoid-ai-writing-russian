import { describe, expect, test } from "bun:test";
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const cli = join(import.meta.dir, "..", "src", "cli.ts");
const fixtures = `${join(import.meta.dir, "fixtures")}/`;

function run(
  args: string[],
  stdin?: string,
  cwd?: string,
  extraEnv: Record<string, string> = {},
): { code: number; out: string; err: string } {
  const env: Record<string, string | undefined> = { ...process.env, AIW_RU_CONFIG: "", NO_COLOR: "1", ...extraEnv };
  delete env.FORCE_COLOR;
  if (extraEnv.FORCE_COLOR) {
    delete env.NO_COLOR;
    env.FORCE_COLOR = extraEnv.FORCE_COLOR;
  }
  const p = Bun.spawnSync(["bun", cli, ...args], {
    stdin: stdin ? new TextEncoder().encode(stdin) : undefined,
    cwd,
    env,
  });
  return { code: p.exitCode ?? -1, out: p.stdout.toString(), err: p.stderr.toString() };
}

describe("cli", () => {
  test("scan файла", () => {
    const r = run(["scan", `${fixtures}corpus/ai/blog.md`]);
    expect(r.code).toBe(0);
    expect(r.out).toContain("сильный ИИ-стиль");
  });

  test("цвет: NO_COLOR по умолчанию в тестах, FORCE_COLOR включает", () => {
    const esc = String.fromCharCode(27);
    expect(run(["scan"], "Давайте разберёмся.").out).not.toContain(esc);
    const colored = run(["scan"], "Давайте разберёмся.", undefined, { FORCE_COLOR: "1" }).out;
    expect(colored).toContain(`${esc}[33mP1${esc}[0m`);
    expect(run(["scan", "--json"], "Давайте разберёмся.", undefined, { FORCE_COLOR: "1" }).out).not.toContain(esc);
  });

  test("средняя длина предложения согласована с числом", () => {
    expect(run(["scan"], "Мы пришли домой поздно.").out).toContain("в среднем 4 слова в предложении");
    expect(run(["scan"], "Мы пришли домой очень поздно вечером.").out).toContain("в среднем 6 слов в предложении");
    expect(run(["scan"], "Мы пришли домой. Мы пришли домой поздно.").out).toContain(
      "в среднем 3,5 слова в предложении",
    );
  });

  test("scan из stdin в JSON", () => {
    const r = run(["scan", "--json"], "Давайте разберёмся.");
    expect(JSON.parse(r.out).issues[0].type).toBe("lets");
  });

  test("профиль скилла как --context", () => {
    const r = run(["scan", "--json", "--context", "vak"], "Текст.");
    expect(JSON.parse(r.out).stats.contextMode).toBe("academic");
  });

  test("--fail-above", () => {
    expect(run(["scan", "--fail-above", "50", `${fixtures}corpus/ai/blog.md`]).code).toBe(1);
    expect(run(["scan", "--fail-above", "50", `${fixtures}corpus/human/blog.md`]).code).toBe(0);
  });

  test("antiplagiat", () => {
    const r = run(["antiplagiat", `${fixtures}corpus/ai/blog.md`]);
    expect(r.code).toBe(0);
    expect(r.out).toContain("Доля ИИ-текста");
  });

  test("validate возвращает 1 при повреждении", () => {
    expect(run(["validate", `${fixtures}corpus/ai/blog.md`, `${fixtures}corpus/human/blog.md`]).code).toBe(1);
    expect(run(["validate", `${fixtures}corpus/human/blog.md`, `${fixtures}corpus/human/blog.md`]).code).toBe(0);
  });

  test("ошибки использования — код 2", () => {
    expect(run(["scan", "--context", "нет-такого"]).code).toBe(2);
    expect(run(["frobnicate"]).code).toBe(2);
    expect(run([]).code).toBe(2);
  });

  test("--min скрывает находки ниже уровня", () => {
    const all = run(["scan", `${fixtures}corpus/ai/blog.md`]).out;
    const p0 = run(["scan", "--min", "P0", `${fixtures}corpus/ai/blog.md`]).out;
    const row = (level: string): RegExp => new RegExp(`^  ${level}\\s+\\d+:\\d+`, "m");
    expect(all).toMatch(row("(P2|стиль)"));
    expect(p0).not.toMatch(row("(P1|P2|стиль)"));
    expect(p0).toMatch(row("P0"));
  });

  test("scan: таблица по ширине COLUMNS, P0 выше P1", () => {
    const r = run(["scan", `${fixtures}corpus/ai/blog.md`], undefined, undefined, { COLUMNS: "80" });
    const lines = r.out.split("\n");
    for (const l of lines.slice(1)) expect(l.length, l).toBeLessThanOrEqual(80);
    expect(r.out).toContain("Уровень  Где");
    const levels = lines.map((l) => /^ {2}(P0|P1|P2|стиль)\s+\d/.exec(l)?.[1]).filter(Boolean);
    expect(levels.indexOf("P0")).toBeLessThan(levels.indexOf("P1"));
  });

  test("scan чистого текста и согласование числительных", () => {
    const r = run(["scan"], "Одно слово.");
    expect(r.out).toContain("2 слова, 1 предложение");
    expect(r.out).toContain("Примет не найдено.");
  });

  test("scan нескольких файлов в JSON — массив", () => {
    const r = run(["scan", "--json", `${fixtures}corpus/ai/blog.md`, `${fixtures}corpus/human/blog.md`]);
    const data = JSON.parse(r.out) as { file: string; score: number }[];
    expect(data).toHaveLength(2);
    expect(data[0]?.score).toBeGreaterThan(data[1]?.score ?? 100);
  });

  test("не-UTF-8 ввод — код 2", () => {
    const f = join(mkdtempSync(join(tmpdir(), "aiw-ru-")), "bad.txt");
    writeFileSync(f, new Uint8Array([0xff, 0xfe, 0x00, 0xc3]));
    expect(run(["scan", f]).code).toBe(2);
  });

  test("antiplagiat --json", () => {
    const r = run(["antiplagiat", "--json", `${fixtures}corpus/ai/vak.md`]);
    const data = JSON.parse(r.out) as { aiShare: number; fragments: unknown[]; model: string };
    expect(data.aiShare).toBeGreaterThan(50);
    expect(data.model).toBe("default");
  });

  test("calibrate создаёт файл, antiplagiat его подхватывает", () => {
    const dir = mkdtempSync(join(tmpdir(), "aiw-ru-"));
    const doc = join(dir, "doc.md");
    const marked = join(dir, "marked.txt");
    const ai = readFileSync(`${fixtures}corpus/ai/vak.md`, "utf8");
    const human = readFileSync(`${fixtures}corpus/human/vak.md`, "utf8");
    writeFileSync(doc, `${ai}\n\n${human}`);
    const firstParagraph = ai.split("\n\n")[1] ?? "";
    writeFileSync(marked, firstParagraph);
    const cal = run(["calibrate", "--doc", doc, "--marked", marked], undefined, dir);
    expect(cal.code).toBe(0);
    expect(cal.out).toContain("Фрагменты с разметкой");
    expect(existsSync(join(dir, ".aiw-ru.json"))).toBe(true);
    const saved = JSON.parse(readFileSync(join(dir, ".aiw-ru.json"), "utf8")) as {
      version: number;
      samples: { y: number }[];
    };
    expect(saved.version).toBe(1);
    expect(saved.samples.filter((x) => x.y === 1)).toHaveLength(1);
    expect(saved.samples.length).toBeGreaterThan(1);
    const again = run(["calibrate", "--doc", doc, "--marked", marked], undefined, dir);
    const savedAgain = JSON.parse(readFileSync(join(dir, ".aiw-ru.json"), "utf8")) as { samples: unknown[] };
    expect(again.code).toBe(0);
    expect(savedAgain.samples.length).toBe(saved.samples.length * 2);
    const report = JSON.parse(run(["antiplagiat", "--json", doc], undefined, dir).out) as { model: string };
    expect(report.model).toBe("calibrated");
  });

  test("calibrate без пар --doc/--marked — код 2", () => {
    expect(run(["calibrate", "--doc", "x.md"]).code).toBe(2);
    expect(run(["calibrate", "--marked", "m.txt", "--doc", "x.md"]).code).toBe(2);
    expect(run(["calibrate", "--doc", "x.md", "--share", "150"]).code).toBe(2);
    expect(run(["calibrate", "--doc", "x.md", "--marked", "m.txt", "--share", "10"]).code).toBe(2);
  });

  test("calibrate --share: итоговая доля из отчёта без разметки фрагментов", () => {
    const dir = mkdtempSync(join(tmpdir(), "aiw-ru-"));
    const doc = join(dir, "doc.md");
    writeFileSync(doc, readFileSync(`${fixtures}corpus/ai/vak.md`, "utf8"));
    const before = JSON.parse(run(["antiplagiat", "--json", doc], undefined, dir).out) as { aiShare: number };
    expect(before.aiShare).toBeGreaterThan(50);
    const cal = run(["calibrate", "--doc", doc, "--share", "0%"], undefined, dir);
    expect(cal.code).toBe(0);
    expect(cal.out).toContain("доля 0 %");
    expect(cal.out).toContain("Ошибка доли ИИ");
    const saved = JSON.parse(readFileSync(join(dir, ".aiw-ru.json"), "utf8")) as { documents: unknown[] };
    expect(saved.documents).toHaveLength(1);
    const after = JSON.parse(run(["antiplagiat", "--json", doc], undefined, dir).out) as {
      aiShare: number;
      model: string;
    };
    expect(after.model).toBe("calibrated");
    expect(after.aiShare).toBeLessThan(before.aiShare);
  });

  test("битый файл калибровки — понятная ошибка", () => {
    const dir = mkdtempSync(join(tmpdir(), "aiw-ru-"));
    writeFileSync(join(dir, ".aiw-ru.json"), "{не json");
    const r = run(["antiplagiat", `${fixtures}corpus/ai/vak.md`], undefined, dir);
    expect(r.code).toBe(2);
    expect(r.err).toContain("калибровку");
  });
});

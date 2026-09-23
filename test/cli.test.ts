import { describe, expect, test } from "bun:test";
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const cli = join(import.meta.dir, "..", "src", "cli.ts");
const fixtures = `${join(import.meta.dir, "fixtures")}/`;

function run(args: string[], stdin?: string, cwd?: string): { code: number; out: string; err: string } {
  const p = Bun.spawnSync(["bun", cli, ...args], {
    stdin: stdin ? new TextEncoder().encode(stdin) : undefined,
    cwd,
    env: { ...process.env, AIW_RU_CONFIG: "" },
  });
  return { code: p.exitCode ?? -1, out: p.stdout.toString(), err: p.stderr.toString() };
}

describe("cli", () => {
  test("scan файла", () => {
    const r = run(["scan", `${fixtures}corpus/ai/blog.md`]);
    expect(r.code).toBe(0);
    expect(r.out).toContain("сильный ИИ-стиль");
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
    expect(r.out).toContain("Оценка доли ИИ-текста");
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
    expect(all).toContain("P2:");
    expect(p0).not.toContain("P2:");
    expect(p0).toContain("P0:");
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
    expect(cal.out).toContain("подсвечено системой 1");
    expect(existsSync(join(dir, ".aiw-ru.json"))).toBe(true);
    const saved = JSON.parse(readFileSync(join(dir, ".aiw-ru.json"), "utf8")) as {
      version: number;
      samples: unknown[];
    };
    expect(saved.version).toBe(1);
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
  });

  test("битый файл калибровки — понятная ошибка", () => {
    const dir = mkdtempSync(join(tmpdir(), "aiw-ru-"));
    writeFileSync(join(dir, ".aiw-ru.json"), "{не json");
    const r = run(["antiplagiat", `${fixtures}corpus/ai/vak.md`], undefined, dir);
    expect(r.code).toBe(2);
    expect(r.err).toContain("калибровку");
  });
});

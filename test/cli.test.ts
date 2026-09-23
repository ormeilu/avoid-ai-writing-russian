import { describe, expect, test } from "bun:test";

const cli = new URL("../src/cli.ts", import.meta.url).pathname;
const fixtures = new URL("./fixtures/", import.meta.url).pathname;

function run(args: string[], stdin?: string): { code: number; out: string; err: string } {
  const p = Bun.spawnSync(["bun", cli, ...args], { stdin: stdin ? new TextEncoder().encode(stdin) : undefined });
  return { code: p.exitCode ?? -1, out: p.stdout.toString(), err: p.stderr.toString() };
}

describe("cli", () => {
  test("scan файла", () => {
    const r = run(["scan", `${fixtures}ai.md`]);
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
    expect(run(["scan", "--fail-above", "50", `${fixtures}ai.md`]).code).toBe(1);
    expect(run(["scan", "--fail-above", "50", `${fixtures}human.md`]).code).toBe(0);
  });

  test("antiplagiat", () => {
    const r = run(["antiplagiat", `${fixtures}ai.md`]);
    expect(r.code).toBe(0);
    expect(r.out).toContain("Оценка доли ИИ-текста");
  });

  test("validate возвращает 1 при повреждении", () => {
    expect(run(["validate", `${fixtures}ai.md`, `${fixtures}human.md`]).code).toBe(1);
    expect(run(["validate", `${fixtures}human.md`, `${fixtures}human.md`]).code).toBe(0);
  });

  test("ошибки использования — код 2", () => {
    expect(run(["scan", "--context", "нет-такого"]).code).toBe(2);
    expect(run(["frobnicate"]).code).toBe(2);
    expect(run([]).code).toBe(2);
  });
});

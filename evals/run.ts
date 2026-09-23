#!/usr/bin/env bun
/**
 * Поведенческие проверки скиллов на живом агенте.
 *
 *   bun run eval                         все случаи через `claude -p`
 *   bun run eval --case vak-protected    один случай
 *   bun run eval --agent "codex exec"    другой агент (промпт идёт в stdin)
 *   bun run eval --model claude-sonnet-5 модель для claude
 *   bun run eval --dry                   показать промпты, агента не вызывать
 *
 * Ответы сохраняются в evals/results/<время>/, итог печатается таблицей.
 * В CI не запускается: вызов модели стоит денег и недетерминирован.
 * Код выхода 1, если хоть один случай не прошёл.
 */

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { type EvalCase, grade } from "./grade.ts";

const ROOT = join(import.meta.dir, "..");
const args = process.argv.slice(2);
const opt = (name: string): string | undefined => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : undefined;
};

const { cases } = JSON.parse(readFileSync(join(import.meta.dir, "cases.json"), "utf8")) as { cases: EvalCase[] };
const only = opt("--case");
const selected = only ? cases.filter((c) => c.id === only) : cases;
if (!selected.length) {
  console.error(`нет случая ${only}`);
  process.exit(2);
}

function skillText(name: string): string {
  const base = join(ROOT, "skills", name);
  const parts = [readFileSync(join(base, "SKILL.md"), "utf8")];
  if (name === "avoid-ai-writing-russian") parts.push(readFileSync(join(base, "references/patterns.md"), "utf8"));
  return parts.join("\n\n");
}

export function buildPrompt(c: EvalCase): string {
  const skills =
    c.skill === "antiplagiat"
      ? `${skillText("antiplagiat")}\n\n${skillText("avoid-ai-writing-russian")}`
      : skillText("avoid-ai-writing-russian");
  return [
    "Ниже инструкции скилла. Следуй им строго. Детектор и файлы недоступны: проверки только модельные.",
    "<skill>",
    skills,
    "</skill>",
    "",
    `Запрос пользователя: ${c.request}`,
    "",
    "<text>",
    c.input,
    "</text>",
  ].join("\n");
}

async function ask(prompt: string): Promise<string> {
  const agent = opt("--agent");
  const cmd = agent
    ? agent.split(" ")
    : ["claude", "-p", ...(opt("--model") ? ["--model", opt("--model") as string] : [])];
  const proc = Bun.spawn(cmd, { stdin: new TextEncoder().encode(prompt), stdout: "pipe", stderr: "pipe" });
  const out = await new Response(proc.stdout).text();
  const err = await new Response(proc.stderr).text();
  if ((await proc.exited) !== 0) throw new Error(`${cmd.join(" ")}: ${(err || out).trim().slice(0, 300)}`);
  return out;
}

if (import.meta.main) {
  if (args.includes("--dry")) {
    for (const c of selected) console.log(`── ${c.id} ──\n${buildPrompt(c).slice(-600)}\n`);
    process.exit(0);
  }
  const dir = join(import.meta.dir, "results", new Date().toISOString().replace(/[:.]/g, "-"));
  mkdirSync(dir, { recursive: true });
  let failed = 0;
  for (const c of selected) {
    process.stdout.write(`${c.id.padEnd(24)} `);
    try {
      const response = await ask(buildPrompt(c));
      writeFileSync(join(dir, `${c.id}.md`), response);
      const g = grade(c, response);
      if (!g.pass) failed += 1;
      console.log(g.pass ? "ok" : `ПРОВАЛ\n    ${g.failures.join("\n    ")}`);
    } catch (e) {
      failed += 1;
      console.log(`ОШИБКА ${(e as Error).message}`);
    }
  }
  console.log(`\n${selected.length - failed}/${selected.length} прошли. Ответы: ${dir}`);
  process.exit(failed ? 1 : 0);
}

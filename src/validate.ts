/**
 * Проверка сохранности: правка прозы не должна портить защищённое
 * содержимое. Сравнивает исходник и исправленный текст и сообщает,
 * что пропало или изменилось.
 *
 * Разрешённые скиллом изменения не считаются нарушением: смена прямых
 * кавычек на «ёлочки», десятичной точки на запятую, Title Case в
 * заголовке на обычный регистр, удаление параметра utm_source ИИ-инструмента.
 */

import { analyze } from "./detect.ts";
import type { ContextMode } from "./types.ts";

export interface Violation {
  kind: string;
  detail: string;
}

export interface ValidationResult {
  ok: boolean;
  violations: Violation[];
  issuesBefore: number;
  issuesAfter: number;
}

function all(re: RegExp, s: string): string[] {
  return [...s.matchAll(re)].map((m) => m[0]);
}

function frontmatter(s: string): string {
  return /^---\n[\s\S]*?\n---(?:\n|$)/.exec(s)?.[0] ?? "";
}

function stripCode(s: string): string {
  return s.replace(/^(```|~~~)[^\n]*\n[\s\S]*?^\1[^\n]*$/gm, "").replace(/`[^`\n]+`/g, "");
}

const AI_PARAM = /[?&](?:utm_source=(?:chatgpt\.com|openai|copilot\.com|claude\.ai|perplexity\.ai|perplexity|gemini\.google\.com|deepseek\.com)|referrer=grok\.com)/gi;

function urls(s: string): string[] {
  return all(/https?:\/\/[^\s)>\]»"]+/g, stripCode(s)).map((u) => u.replace(AI_PARAM, "").replace(/[?&]$/, ""));
}

function numbers(s: string): string[] {
  const prose = stripCode(s).replace(/https?:\/\/\S+/g, "");
  return all(/\d+(?:[.,]\d+)*/g, prose).map((n) => n.replace(/,/g, "."));
}

function multisetDiff(a: string[], b: string[]): { missing: string[]; added: string[] } {
  const count = new Map<string, number>();
  for (const x of a) count.set(x, (count.get(x) ?? 0) + 1);
  const added: string[] = [];
  for (const x of b) {
    const c = count.get(x) ?? 0;
    if (c > 0) count.set(x, c - 1);
    else added.push(x);
  }
  const missing: string[] = [];
  for (const [x, c] of count) for (let i = 0; i < c; i += 1) missing.push(x);
  return { missing, added };
}

function compareExact(kind: string, before: string[], after: string[], out: Violation[]): void {
  const { missing, added } = multisetDiff(before, after);
  for (const m of missing.slice(0, 5)) out.push({ kind, detail: `пропало или изменено: ${m.slice(0, 120)}` });
  for (const a of added.slice(0, 5)) out.push({ kind, detail: `появилось: ${a.slice(0, 120)}` });
}

function headingShape(s: string): string[] {
  return all(/^#{1,6}(?=\s)/gm, stripCode(s));
}

export function validate(before: string, after: string, context: ContextMode = "general"): ValidationResult {
  const v: Violation[] = [];
  if (frontmatter(before) !== frontmatter(after)) v.push({ kind: "yaml", detail: "YAML-шапка изменена" });
  compareExact("код", all(/^(```|~~~)[^\n]*\n[\s\S]*?^\1[^\n]*$/gm, before), all(/^(```|~~~)[^\n]*\n[\s\S]*?^\1[^\n]*$/gm, after), v);
  compareExact("инлайн-код", all(/`[^`\n]+`/g, stripCode(before)), all(/`[^`\n]+`/g, stripCode(after)), v);
  compareExact("формула", all(/\$\$[\s\S]*?\$\$|\$[^$\n]+\$/g, stripCode(before)), all(/\$\$[\s\S]*?\$\$|\$[^$\n]+\$/g, stripCode(after)), v);
  compareExact("URL", urls(before), urls(after), v);
  compareExact("ссылка на литературу", all(/\[\d+(?:[,;–-]\s*\d+)*(?:,\s*с\.\s*\d+(?:[–-]\d+)?)?\]/g, stripCode(before)), all(/\[\d+(?:[,;–-]\s*\d+)*(?:,\s*с\.\s*\d+(?:[–-]\d+)?)?\]/g, stripCode(after)), v);
  compareExact("ссылка pandoc", all(/\[-?@[^\]\n]+\]/g, stripCode(before)), all(/\[-?@[^\]\n]+\]/g, stripCode(after)), v);
  compareExact("таблица", all(/^[ \t]*\|.*$/gm, stripCode(before)), all(/^[ \t]*\|.*$/gm, stripCode(after)), v);
  compareExact("цитата-блок", all(/^[ \t]*>.*$/gm, stripCode(before)), all(/^[ \t]*>.*$/gm, stripCode(after)), v);
  const quoteNorm = (q: string): string => q.replace(/^["“„«]|["”“»]$/g, "");
  compareExact(
    "цитата",
    all(/«[^«»\n]{1,400}»|"[^"\n]{1,400}"|“[^“”\n]{1,400}”/g, stripCode(before)).map(quoteNorm),
    all(/«[^«»\n]{1,400}»|"[^"\n]{1,400}"|“[^“”\n]{1,400}”/g, stripCode(after)).map(quoteNorm),
    v,
  );
  compareExact("число", numbers(before), numbers(after), v);
  const hb = headingShape(before).join(" ");
  const ha = headingShape(after).join(" ");
  if (hb !== ha) v.push({ kind: "заголовки", detail: `структура заголовков изменилась: [${hb}] → [${ha}]` });

  const issuesBefore = analyze(before, { context }).issues.length;
  const issuesAfter = analyze(after, { context }).issues.length;
  if (issuesAfter > issuesBefore) {
    v.push({ kind: "находки", detail: `находок стало больше: ${issuesBefore} → ${issuesAfter}` });
  }
  return { ok: v.length === 0, violations: v, issuesBefore, issuesAfter };
}

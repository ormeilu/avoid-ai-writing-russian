/**
 * Оценка ответа агента по проверкам из cases.json. Модуль не вызывает
 * агента: его тестирует test/evals.test.ts на готовых ответах, а
 * evals/run.ts применяет к живым ответам.
 */

import { analyze } from "../src/detect.ts";
import { validate } from "../src/validate.ts";

export interface Checks {
  /** Строки, которые должны остаться в итоговом тексте. */
  preserve?: string[];
  /** Строки, которых не должно быть в итоговом тексте. */
  forbid?: string[];
  /** В итоговом тексте нет чисел, которых не было в исходнике («никогда не добавляй»). */
  noNewNumbers?: boolean;
  /** Итоговый текст совпадает с исходным (с точностью до пробелов). */
  unchanged?: boolean;
  /** validate(исходный, итоговый) без нарушений. */
  validate?: boolean;
  /** Оценка детектора после правки ниже, чем до. */
  scoreDrop?: boolean;
  /** Итоговый текст не длиннее исходного больше чем в N раз. */
  maxGrowth?: number;
  /** Не больше N хэштегов в итоговом тексте. */
  maxHashtags?: number;
  /** Регулярные выражения, которым должен соответствовать итоговый текст. */
  finalMatches?: string[];
  /** Строки, которые должны быть где-то в ответе. */
  responseHas?: string[];
  /** Строки, которых не должно быть в ответе. */
  responseLacks?: string[];
  /** Регулярные выражения (без учёта регистра) для всего ответа. */
  responseMatches?: string[];
}

export interface EvalCase {
  id: string;
  skill: "avoid-ai-writing-russian" | "antiplagiat";
  request: string;
  input: string;
  checks: Checks;
}

export interface Grade {
  id: string;
  pass: boolean;
  failures: string[];
  final?: string;
}

const HEAD = /(?:^|\n)\s*(?:#{1,4}\s*|\*\*)Итоговый текст:?(?:\*\*)?:?\s*\n/;
const NEXT = /\n\s*(?:#{1,4}\s*|\*\*)(?:Изменения|Проверка|Найдено|Оценка):?(?:\*\*)?:?\s*(?:\n|$)/;

/** Итоговый текст из ответа: раздел «Итоговый текст» без обрамляющих кавычек и ограждений. */
export function extractFinal(response: string): string | undefined {
  const m = HEAD.exec(response);
  if (!m) return undefined;
  let rest = response.slice((m.index ?? 0) + m[0].length);
  const next = NEXT.exec(rest);
  if (next) rest = rest.slice(0, next.index);
  rest = rest.trim();
  const fence = /^```[^\n]*\n([\s\S]*?)\n```$/.exec(rest);
  if (fence) rest = fence[1] ?? "";
  if (rest.split("\n").every((l) => l.startsWith(">") || l.trim() === "")) {
    rest = rest.replace(/^>\s?/gm, "");
  }
  return rest.trim();
}

const norm = (s: string): string => s.replace(/\s+/g, " ").trim();
const numbers = (s: string): string[] => (s.match(/\d+(?:[.,]\d+)?/g) ?? []).map((n) => n.replace(",", "."));

export function grade(c: EvalCase, response: string): Grade {
  const f: string[] = [];
  const ch = c.checks;
  const needsFinal =
    ch.preserve ||
    ch.forbid ||
    ch.noNewNumbers ||
    ch.unchanged ||
    ch.validate ||
    ch.scoreDrop ||
    ch.maxGrowth ||
    ch.maxHashtags ||
    ch.finalMatches;
  const final = extractFinal(response);
  if (needsFinal && final === undefined) f.push("в ответе нет раздела «Итоговый текст»");
  if (final !== undefined) {
    for (const s of ch.preserve ?? []) if (!final.includes(s)) f.push(`пропало: ${s}`);
    for (const s of ch.forbid ?? []) if (final.includes(s)) f.push(`осталось: ${s}`);
    if (ch.noNewNumbers) {
      const src = new Set(numbers(c.input));
      const added = numbers(final).filter((n) => !src.has(n));
      if (added.length) f.push(`выдуманные числа: ${added.join(", ")}`);
    }
    if (ch.unchanged && norm(final) !== norm(c.input)) f.push("чистый текст изменён");
    if (ch.validate) {
      const v = validate(c.input, final);
      for (const x of v.violations) f.push(`validate [${x.kind}]: ${x.detail}`);
    }
    if (ch.scoreDrop) {
      const before = analyze(c.input).score;
      const after = analyze(final).score;
      if (after >= before) f.push(`оценка не снизилась: ${before} → ${after}`);
    }
    if (ch.maxGrowth && final.length > c.input.length * ch.maxGrowth) {
      f.push(`текст вырос в ${(final.length / c.input.length).toFixed(1)} раза`);
    }
    if (ch.maxHashtags !== undefined) {
      const tags = final.match(/(?<![\p{L}\d])#[\p{L}_][\p{L}\d_]*/gu) ?? [];
      if (tags.length > ch.maxHashtags) f.push(`хэштегов ${tags.length}, допустимо ${ch.maxHashtags}`);
    }
    for (const re of ch.finalMatches ?? [])
      if (!new RegExp(re, "u").test(final)) f.push(`итог не соответствует /${re}/`);
  }
  for (const s of ch.responseHas ?? []) if (!response.includes(s)) f.push(`в ответе нет: ${s}`);
  for (const s of ch.responseLacks ?? []) if (response.includes(s)) f.push(`в ответе лишнее: ${s}`);
  for (const re of ch.responseMatches ?? [])
    if (!new RegExp(re, "iu").test(response)) f.push(`ответ не соответствует /${re}/`);
  const result: Grade = { id: c.id, pass: f.length === 0, failures: f };
  if (final !== undefined) result.final = final;
  return result;
}

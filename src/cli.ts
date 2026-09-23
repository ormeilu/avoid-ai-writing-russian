#!/usr/bin/env bun
/**
 * aiw-ru — командная строка детектора.
 *
 *   aiw-ru scan [файл…] [--context РЕЖИМ] [--json] [--min P0|P1|P2]
 *   aiw-ru antiplagiat [файл] [--config ПУТЬ] [--json]
 *   aiw-ru validate ИСХОДНЫЙ ИСПРАВЛЕННЫЙ [--context РЕЖИМ] [--json]
 *   aiw-ru calibrate --doc ФАЙЛ --marked ФАЙЛ [--doc … --marked …] [--config ПУТЬ]
 *
 * Коды выхода: 0 — успех; 1 — validate нашёл повреждения или scan с --fail-above
 * превысил порог; 2 — ошибка использования или ввода-вывода.
 */

import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import {
  antiplagiat,
  balancedAccuracy,
  DEFAULT_MODEL,
  FEATURE_NAMES,
  fit,
  labelFragments,
  type CalibrationFile,
  type Model,
} from "./antiplagiat.ts";
import { analyze, TYPE_LABELS } from "./detect.ts";
import { CONTEXT_MODES, PROFILE_TO_MODE, type ContextMode, type Severity } from "./types.ts";
import { validate } from "./validate.ts";

const USAGE = `aiw-ru — приметы ИИ-стиля в русском тексте

Команды:
  scan [файл…]                 найти приметы (без файла — читать stdin)
  antiplagiat [файл]           оценка по фрагментам в духе модуля ИИ-детекции «Антиплагиата»
  validate ИСХОДНЫЙ НОВЫЙ      проверить, что правка не повредила код, числа, цитаты, ссылки
  calibrate --doc Ф --marked Ф подстроить модель antiplagiat под ваши реальные отчёты

Параметры:
  --context РЕЖИМ   general | academic | technical | social | chat
                    или профиль скилла: vak | docs | blog | telegram | business-email | chat
  --json            вывод в JSON
  --min P0|P1|P2    показывать находки не ниже уровня (scan)
  --fail-above N    scan: код выхода 1, если оценка выше N
  --config ПУТЬ     файл калибровки (по умолчанию ./.aiw-ru.json, затем ~/.config/aiw-ru.json)
  -h, --help        справка
`;

interface Args {
  cmd: string;
  files: string[];
  context?: ContextMode;
  json: boolean;
  min: Severity;
  failAbove?: number;
  config?: string;
  docs: string[];
  marked: string[];
  help: boolean;
}

class UsageError extends Error {}

function parse(argv: string[]): Args {
  const a: Args = { cmd: "", files: [], json: false, min: "P2", docs: [], marked: [], help: false };
  const need = (i: number, flag: string): string => {
    const v = argv[i + 1];
    if (v === undefined) throw new UsageError(`${flag} требует значения`);
    return v;
  };
  for (let i = 0; i < argv.length; i += 1) {
    const x = argv[i] as string;
    if (x === "-h" || x === "--help") a.help = true;
    else if (x === "--json") a.json = true;
    else if (x === "--context") {
      const v = need(i, x);
      i += 1;
      const mode = (CONTEXT_MODES as readonly string[]).includes(v) ? (v as ContextMode) : PROFILE_TO_MODE[v];
      if (!mode) throw new UsageError(`неизвестный --context: ${v}`);
      a.context = mode;
    } else if (x === "--min") {
      const v = need(i, x);
      i += 1;
      if (!["P0", "P1", "P2"].includes(v)) throw new UsageError(`неизвестный --min: ${v}`);
      a.min = v as Severity;
    } else if (x === "--fail-above") {
      a.failAbove = Number(need(i, x));
      i += 1;
      if (Number.isNaN(a.failAbove)) throw new UsageError("--fail-above ждёт число");
    } else if (x === "--config") {
      a.config = need(i, x);
      i += 1;
    } else if (x === "--doc") {
      a.docs.push(need(i, x));
      i += 1;
    } else if (x === "--marked") {
      a.marked.push(need(i, x));
      i += 1;
    } else if (x.startsWith("-") && x !== "-") throw new UsageError(`неизвестный параметр: ${x}`);
    else if (!a.cmd) a.cmd = x;
    else a.files.push(x);
  }
  return a;
}

function read(file: string | undefined): string {
  const fromStdin = file === undefined || file === "-";
  try {
    const buf = readFileSync(fromStdin ? 0 : file);
    return new TextDecoder("utf-8", { fatal: true }).decode(buf);
  } catch (e) {
    throw new UsageError(`не удалось прочитать ${fromStdin ? "stdin" : file}: ${(e as Error).message}`);
  }
}

const RANK: Record<Severity, number> = { P0: 0, P1: 1, P2: 2 };

function out(s: string): void {
  process.stdout.write(`${s}\n`);
}

function cmdScan(a: Args): number {
  const files = a.files.length ? a.files : ["-"];
  const results = files.map((f) => ({ file: f, ...analyze(read(f), { context: a.context ?? "general" }) }));
  let fail = false;
  if (a.json) out(JSON.stringify(results.length === 1 ? results[0] : results, null, 2));
  for (const r of results) {
    if (a.failAbove !== undefined && r.score > a.failAbove) fail = true;
    if (a.json) continue;
    const shown = r.issues.filter((i) => RANK[i.severity] <= RANK[a.min]);
    out(`${r.file === "-" ? "stdin" : r.file}: ${r.score}/100 — ${r.label} (${r.stats.words} слов, режим ${r.stats.contextMode})`);
    if (r.suspicious) out("  ⚠ документ выглядит подозрительным: невидимые символы или подмена букв");
    for (const sev of ["P0", "P1", "P2"] as Severity[]) {
      const group = shown.filter((i) => i.severity === sev);
      if (!group.length) continue;
      out(`  ${sev}:`);
      for (const i of group) {
        const tag = i.styleOnly ? " [стиль, не довод об авторстве]" : "";
        out(`    ${i.line}:${i.column}  ${TYPE_LABELS[i.type] ?? i.type}: «${i.text}» → ${i.hint}${tag}`);
      }
    }
    out(`  ритм: средняя длина предложения ${r.stats.meanSentenceLength}, вариация ${r.stats.sentenceLengthCV}; MATTR ${r.stats.mattr}`);
  }
  return fail ? 1 : 0;
}

function configPath(a: Args): string | undefined {
  if (a.config) return a.config;
  if (process.env.AIW_RU_CONFIG) return process.env.AIW_RU_CONFIG;
  const local = ".aiw-ru.json";
  if (existsSync(local)) return local;
  const global = join(homedir(), ".config", "aiw-ru.json");
  return existsSync(global) ? global : undefined;
}

function loadCalibration(path: string | undefined): CalibrationFile | undefined {
  if (!path || !existsSync(path)) return undefined;
  try {
    const data = JSON.parse(readFileSync(path, "utf8")) as CalibrationFile;
    if (data.version !== 1 || !Array.isArray(data.model?.weights)) throw new Error("неверный формат");
    return data;
  } catch (e) {
    throw new UsageError(`не удалось прочитать калибровку ${path}: ${(e as Error).message}`);
  }
}

function cmdAntiplagiat(a: Args): number {
  const cal = loadCalibration(configPath(a));
  const model: Model | undefined = cal?.model;
  const report = antiplagiat(read(a.files[0]), { context: a.context ?? "academic", model });
  if (a.json) {
    out(JSON.stringify(report, null, 2));
    return 0;
  }
  out(`Оценка доли ИИ-текста: ${report.aiShare.toString().replace(".", ",")} %`);
  out(`Модель: ${report.model === "calibrated" ? "откалибрована по вашим отчётам" : "по умолчанию (не калибрована)"}, порог ${report.threshold}`);
  if (report.suspicious) out(`⚠ Подозрительный документ: ${report.suspiciousReasons.join("; ")}. Системы проверки такое замечают — удалите эти символы.`);
  out("");
  for (const f of report.fragments) {
    const mark = f.ai ? "ИИ " : "   ";
    const pct = Math.round(f.probability * 100);
    const why = f.ai && f.reasons.length ? `  (${f.reasons.join(", ")})` : "";
    out(`${mark} ${String(pct).padStart(3)} %  строки ${f.line}–${f.endLine}, ${f.words} сл.  ${f.preview}…${why}`);
  }
  out("");
  out("Это приближение: настоящий классификатор системы закрыт. Для точности откалибруйте модель на своих отчётах (aiw-ru calibrate).");
  return 0;
}

function cmdValidate(a: Args): number {
  if (a.files.length !== 2) throw new UsageError("validate ждёт два файла: исходный и исправленный");
  const r = validate(read(a.files[0]), read(a.files[1]), a.context ?? "general");
  if (a.json) out(JSON.stringify(r, null, 2));
  else {
    out(r.ok ? "Сохранность в порядке." : "Правка повредила защищённое содержимое:");
    for (const v of r.violations) out(`  [${v.kind}] ${v.detail}`);
    out(`Находок: ${r.issuesBefore} → ${r.issuesAfter}`);
  }
  return r.ok ? 0 : 1;
}

function splitMarked(text: string): string[] {
  return text
    .split(/\n\s*(?:---+\s*)?\n/)
    .map((s) => s.trim())
    .filter(Boolean);
}

function cmdCalibrate(a: Args): number {
  if (!a.docs.length || a.docs.length !== a.marked.length) {
    throw new UsageError("calibrate ждёт пары --doc ФАЙЛ --marked ФАЙЛ");
  }
  const path = a.config ?? ".aiw-ru.json";
  const prev = loadCalibration(existsSync(path) ? path : undefined);
  const samples = [...(prev?.samples ?? [])];
  for (let i = 0; i < a.docs.length; i += 1) {
    const doc = read(a.docs[i]);
    const marked = splitMarked(read(a.marked[i]));
    const labeled = labelFragments(doc, marked);
    const pos = labeled.filter((s) => s.y === 1).length;
    out(`${a.docs[i]}: ${labeled.length} фрагментов, из них подсвечено системой ${pos}`);
    if (marked.length && pos === 0) out("  ⚠ ни один подсвеченный кусок не найден в документе — проверьте, что текст скопирован из отчёта без правок");
    samples.push(...labeled);
  }
  const before = balancedAccuracy(DEFAULT_MODEL, samples);
  const model = fit(samples, prev?.model ?? DEFAULT_MODEL);
  const after = balancedAccuracy(model, samples);
  const data: CalibrationFile = { version: 1, model, samples };
  writeFileSync(path, `${JSON.stringify(data, null, 2)}\n`);
  const fmt = (x: number): string => (Number.isNaN(x) ? "н/д (нужны и ИИ-, и человеческие фрагменты)" : `${(x * 100).toFixed(1)} %`);
  out(`Образцов всего: ${samples.length}`);
  out(`Сбалансированная точность: по умолчанию ${fmt(before)} → после калибровки ${fmt(after)} (на тех же данных; для честной оценки держите часть отчётов отдельно)`);
  out(`Веса: ${model.weights.map((w, k) => `${FEATURE_NAMES[k]} ${w.toFixed(2)}`).join("; ")}; порог ${model.threshold}`);
  out(`Сохранено в ${path}`);
  return 0;
}

export function main(argv: string[]): number {
  try {
    const a = parse(argv);
    if (a.help || !a.cmd) {
      out(USAGE);
      return a.help ? 0 : 2;
    }
    switch (a.cmd) {
      case "scan":
        return cmdScan(a);
      case "antiplagiat":
        return cmdAntiplagiat(a);
      case "validate":
        return cmdValidate(a);
      case "calibrate":
        return cmdCalibrate(a);
      default:
        throw new UsageError(`неизвестная команда: ${a.cmd}`);
    }
  } catch (e) {
    if (e instanceof UsageError) {
      process.stderr.write(`aiw-ru: ${e.message}\n\n${USAGE}`);
      return 2;
    }
    throw e;
  }
}

if (import.meta.main) process.exitCode = main(process.argv.slice(2));

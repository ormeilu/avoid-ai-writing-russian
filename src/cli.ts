#!/usr/bin/env bun
/**
 * aiw-ru — командная строка детектора.
 *
 *   aiw-ru scan [файл…] [--context РЕЖИМ] [--json] [--min P0|P1|P2]
 *   aiw-ru antiplagiat [файл] [--config ПУТЬ] [--json]
 *   aiw-ru validate ИСХОДНЫЙ ИСПРАВЛЕННЫЙ [--context РЕЖИМ] [--json]
 *   aiw-ru calibrate --doc ФАЙЛ (--marked ФАЙЛ | --share ДОЛЯ) [--doc … ] [--config ПУТЬ]
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
  type CalibrationFile,
  DEFAULT_MODEL,
  documentSample,
  FEATURE_NAMES,
  fit,
  fitShare,
  labelFragments,
  type Model,
  shareError,
} from "./antiplagiat.ts";
import { analyze, TYPE_LABELS } from "./detect.ts";
import { table, wrapText } from "./table.ts";
import { CONTEXT_MODES, type ContextMode, PROFILE_TO_MODE, type Severity } from "./types.ts";
import { validate } from "./validate.ts";

const USAGE = `aiw-ru — приметы ИИ-стиля в русском тексте

Команды:
  scan [файл…]                 найти приметы (без файла — читать stdin)
  antiplagiat [файл]           оценка по фрагментам в духе модуля ИИ-детекции «Антиплагиата»
  validate ИСХОДНЫЙ НОВЫЙ      проверить, что правка не повредила код, числа, цитаты, ссылки
  calibrate --doc Ф --marked Ф подстроить модель antiplagiat под ваши отчёты: документ и файл
                               с фрагментами, которые отчёт подсветил как ИИ
  calibrate --doc Ф --share N  то же, если известна только итоговая доля ИИ из отчёта, %

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
  /** Пары для calibrate: документ и разметка из отчёта. */
  reports: { doc: string; marked?: string; share?: number }[];
  help: boolean;
}

class UsageError extends Error {}

function parse(argv: string[]): Args {
  const a: Args = { cmd: "", files: [], json: false, min: "P2", reports: [], help: false };
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
      a.reports.push({ doc: need(i, x) });
      i += 1;
    } else if (x === "--marked" || x === "--share") {
      const v = need(i, x);
      i += 1;
      const last = a.reports.at(-1);
      if (!last || last.marked !== undefined || last.share !== undefined) {
        throw new UsageError(`${x} относится к предыдущему --doc: сначала --doc ФАЙЛ`);
      }
      if (x === "--marked") last.marked = v;
      else {
        const share = Number(v.replace(",", ".").replace(/%$/, ""));
        if (!Number.isFinite(share) || share < 0 || share > 100) throw new UsageError(`--share ждёт долю 0–100: ${v}`);
        last.share = share;
      }
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

/** Цвет в терминале: NO_COLOR выключает, FORCE_COLOR включает, иначе только для TTY. */
function useColor(): boolean {
  if (process.env.NO_COLOR) return false;
  if (process.env.FORCE_COLOR && process.env.FORCE_COLOR !== "0") return true;
  return process.stdout.isTTY === true;
}

const ESC = String.fromCharCode(27);
const COLOR = useColor();
const paint =
  (code: string) =>
  (s: string): string =>
    COLOR ? `${ESC}[${code}m${s}${ESC}[0m` : s;
const c = {
  bold: paint("1"),
  dim: paint("2"),
  red: paint("31"),
  green: paint("32"),
  yellow: paint("33"),
};
const SEV_COLOR: Record<Severity, (s: string) => string> = { P0: c.red, P1: c.yellow, P2: c.dim };
const scoreColor = (score: number): ((s: string) => string) => (score < 15 ? c.green : score < 40 ? c.yellow : c.red);

/** Ширина вывода: терминал, затем COLUMNS, иначе 100. */
function screenWidth(): number {
  const cols = process.stdout.columns || Number(process.env.COLUMNS) || 100;
  return Math.max(60, Math.min(cols, 160));
}

/** Число по-русски: десятичная запятая, не больше двух знаков после неё. */
const ru = (n: number, digits = 2): string => String(Number(n.toFixed(digits))).replace(".", ",");

/** «1 слово», «2 слова», «5 слов». */
function plural(n: number, one: string, few: string, many: string): string {
  const d = n % 10;
  const dd = n % 100;
  const form = d === 1 && dd !== 11 ? one : d >= 2 && d <= 4 && (dd < 12 || dd > 14) ? few : many;
  return `${n} ${form}`;
}

/** Пояснение под таблицей: приглушённое, с переносом по ширине экрана. */
function note(text: string): void {
  for (const l of wrapText(text, screenWidth() - 2)) out(c.dim(`  ${l}`));
}

function out(s: string): void {
  process.stdout.write(`${s}\n`);
}

/** Блок «ключ — значение» с выровненными ключами. */
function facts(pairs: [string, string][]): void {
  const w = Math.max(...pairs.map(([k]) => k.length));
  const room = screenWidth() - w - 4;
  for (const [k, v] of pairs) {
    // Цветные значения короткие; переносим только простой текст.
    const lines = v.includes(ESC) ? [v] : wrapText(v, room);
    lines.forEach((l, n) => {
      out(`  ${n ? " ".repeat(w) : c.dim(k.padEnd(w))}  ${l}`);
    });
  }
}

const LEGEND =
  "P0 — исправить сразу · P1 — исправить до публикации · P2 — шлифовка · стиль — краткость, не довод об авторстве";

function cmdScan(a: Args): number {
  const files = a.files.length ? a.files : ["-"];
  const results = files.map((f) => ({ file: f, ...analyze(read(f), { context: a.context ?? "general" }) }));
  let fail = false;
  if (a.json) out(JSON.stringify(results.length === 1 ? results[0] : results, null, 2));
  results.forEach((r, n) => {
    if (a.failAbove !== undefined && r.score > a.failAbove) fail = true;
    if (a.json) return;
    if (n > 0) out("");
    const shown = r.issues
      .filter((i) => RANK[i.severity] <= RANK[a.min])
      .sort(
        (x, y) =>
          RANK[x.severity] - RANK[y.severity] || Number(!!x.styleOnly) - Number(!!y.styleOnly) || x.index - y.index,
      );
    const s = r.stats;
    out(c.bold(r.file === "-" ? "stdin" : r.file));
    facts([
      ["Оценка", scoreColor(r.score)(`${r.score}/100, ${r.label}`)],
      [
        "Текст",
        `${plural(s.words, "слово", "слова", "слов")}, ${plural(s.sentences, "предложение", "предложения", "предложений")}, режим ${s.contextMode}`,
      ],
      [
        "Ритм",
        `в среднем ${ru(s.meanSentenceLength, 1)} слова в предложении, разброс ${ru(s.sentenceLengthCV)}, MATTR ${ru(s.mattr)}`,
      ],
    ]);
    if (r.suspicious) out(c.red("  Документ выглядит подозрительным: невидимые символы или подмена букв."));
    out("");
    if (!shown.length) {
      out(c.green(r.issues.length ? "  Находок выбранного уровня нет." : "  Примет не найдено."));
      return;
    }
    const rows = shown.map((i) => [
      i.styleOnly ? "стиль" : i.severity,
      `${i.line}:${i.column}`,
      TYPE_LABELS[i.type] ?? i.type,
      `«${i.text}»`,
      i.hint,
    ]);
    const lines = table(
      [
        { title: "Уровень" },
        { title: "Где" },
        { title: "Примета", flex: true, wrap: true, min: 14 },
        { title: "Фрагмент", flex: true, wrap: true, min: 14 },
        { title: "Как исправить", flex: true, wrap: true, min: 20 },
      ],
      rows,
      {
        width: screenWidth(),
        paintHeader: c.dim,
        paint: (row, col, text) => {
          const i = shown[row] as (typeof shown)[number];
          if (col === 0) return i.styleOnly ? c.dim(text) : SEV_COLOR[i.severity](text);
          return col === 1 ? c.dim(text) : text;
        },
      },
    );
    for (const l of lines) out(l);
    out("");
    note(LEGEND);
  });
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
  const share = report.aiShare;
  const paintShare = share < 20 ? c.green : share < 50 ? c.yellow : c.red;
  const ai = report.fragments.filter((f) => f.ai).length;
  out(c.bold(a.files[0] && a.files[0] !== "-" ? a.files[0] : "stdin"));
  facts([
    ["Доля ИИ-текста", paintShare(`${ru(share)} %`)],
    ["Фрагменты", `${report.fragments.length}, похожи на ИИ: ${ai} (порог ${ru(report.threshold)})`],
    [
      "Модель",
      report.model === "calibrated"
        ? `откалибрована по вашим отчётам, образцов: ${cal?.samples.length ?? 0}`
        : "по умолчанию, без калибровки",
    ],
  ]);
  if (report.suspicious)
    out(
      c.red(
        `  Подозрительный документ: ${report.suspiciousReasons.join("; ")}. Системы проверки такое замечают, удалите эти символы.`,
      ),
    );
  out("");
  const rows = report.fragments.map((f) => [
    f.line === f.endLine ? String(f.line) : `${f.line}–${f.endLine}`,
    String(f.words),
    `${Math.round(f.probability * 100)} %`,
    `${f.preview}…`,
    f.ai ? f.reasons.join(", ") : "",
  ]);
  const lines = table(
    [
      { title: "Строки" },
      { title: "Слов", align: "right" },
      { title: "ИИ", align: "right" },
      { title: "Начало фрагмента", flex: true, min: 24 },
      { title: "Почему похоже на ИИ", flex: true, wrap: true, min: 24 },
    ],
    rows,
    {
      width: screenWidth(),
      paintHeader: c.dim,
      paint: (row, col, text) => {
        const f = report.fragments[row] as (typeof report.fragments)[number];
        if (col === 2) return f.ai ? c.red(text) : c.green(text);
        return col === 0 ? c.dim(text) : text;
      },
    },
  );
  for (const l of lines) out(l);
  out("");
  note(
    "Это приближение: классификатор «Антиплагиата» закрыт. Точнее станет после калибровки по вашим отчётам (aiw-ru calibrate).",
  );
  return 0;
}

function cmdValidate(a: Args): number {
  if (a.files.length !== 2) throw new UsageError("validate ждёт два файла: исходный и исправленный");
  const r = validate(read(a.files[0]), read(a.files[1]), a.context ?? "general");
  if (a.json) out(JSON.stringify(r, null, 2));
  else {
    facts([
      ["Сохранность", r.ok ? c.green("в порядке") : c.red(`нарушена (${r.violations.length})`)],
      ["Находки", `${r.issuesBefore} → ${r.issuesAfter}`],
    ]);
    if (!r.ok) {
      out("");
      const lines = table(
        [{ title: "Что" }, { title: "Подробности", flex: true, wrap: true, min: 20 }],
        r.violations.map((v) => [v.kind, v.detail]),
        { width: screenWidth(), paintHeader: c.dim, paint: (_r, col, text) => (col === 0 ? c.yellow(text) : text) },
      );
      for (const l of lines) out(l);
    }
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
  if (!a.reports.length || a.reports.some((r) => r.marked === undefined && r.share === undefined)) {
    throw new UsageError("calibrate ждёт для каждого --doc ФАЙЛ разметку: --marked ФАЙЛ или --share ДОЛЯ");
  }
  const path = a.config ?? ".aiw-ru.json";
  const prev = loadCalibration(existsSync(path) ? path : undefined);
  const samples = [...(prev?.samples ?? [])];
  const documents = [...(prev?.documents ?? [])];
  const rows: string[][] = [];
  const warnings: string[] = [];
  for (const r of a.reports) {
    const doc = read(r.doc);
    if (r.share !== undefined) {
      const d = documentSample(doc, r.share);
      documents.push(d);
      rows.push([r.doc, `доля ${ru(r.share)} %`, String(d.fragments.length), "—"]);
      continue;
    }
    const marked = splitMarked(read(r.marked));
    const labeled = labelFragments(doc, marked);
    const pos = labeled.filter((s) => s.y === 1).length;
    rows.push([r.doc, r.marked ?? "", String(labeled.length), String(pos)]);
    if (marked.length && pos === 0)
      warnings.push(
        `${r.doc}: ни один кусок из ${r.marked} не найден в документе. Проверьте, что текст скопирован из отчёта без правок и документ той же версии.`,
      );
    samples.push(...labeled);
  }
  const start = prev?.model ?? DEFAULT_MODEL;
  const model = fitShare(fit(samples, start), documents);
  const data: CalibrationFile = { version: 1, model, samples, documents };
  writeFileSync(path, `${JSON.stringify(data, null, 2)}\n`);

  out(c.bold("Отчёты"));
  const lines = table(
    [
      { title: "Документ", flex: true, min: 16 },
      { title: "Разметка", flex: true, min: 16 },
      { title: "Фрагментов", align: "right" },
      { title: "Подсвечено ИИ", align: "right" },
    ],
    rows,
    { width: screenWidth(), paintHeader: c.dim },
  );
  for (const l of lines) out(l);
  for (const w of warnings) out(c.yellow(`  ${w}`));
  out("");
  const pct = (x: number): string => (Number.isNaN(x) ? "н/д" : `${ru(x * 100, 1)} %`);
  const pp = (x: number): string => (Number.isNaN(x) ? "н/д" : `${ru(x, 1)} п.п.`);
  const pos = samples.filter((s) => s.y === 1).length;
  out(c.bold("Калибровка"));
  facts([
    ["Фрагменты с разметкой", `${samples.length}, из них ИИ: ${pos}`],
    ["Документы с долей", String(documents.length)],
    [
      "Точность по фрагментам",
      Number.isNaN(balancedAccuracy(model, samples))
        ? "н/д (нужны и ИИ-, и человеческие фрагменты)"
        : `${pct(balancedAccuracy(DEFAULT_MODEL, samples))} → ${pct(balancedAccuracy(model, samples))}`,
    ],
    ["Ошибка доли ИИ", `${pp(shareError(DEFAULT_MODEL, documents))} → ${pp(shareError(model, documents))}`],
    ["Порог", ru(model.threshold)],
    ["Веса", model.weights.map((w, k) => `${FEATURE_NAMES[k]} ${ru(w)}`).join("; ")],
    ["Сохранено", path],
  ]);
  out("");
  note(
    "Точность и ошибка посчитаны на тех же отчётах, по которым шла калибровка, и поэтому завышены. Честная проверка: замер на отчёте, который в калибровку не входил. Файл содержит фрагменты вашего текста, не публикуйте его.",
  );
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

if (import.meta.main) {
  // Читатель закрыл канал (`aiw-ru scan … | head`): это не ошибка, просто выходим.
  process.stdout.on("error", (e: NodeJS.ErrnoException) => {
    if (e.code === "EPIPE") process.exit(0);
    throw e;
  });
  process.exitCode = main(process.argv.slice(2));
}

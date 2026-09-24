/**
 * Подготовка текста: нормализация, маскирование защищённых областей,
 * разбиение на абзацы, предложения и слова.
 *
 * Все преобразования, кроме удаления невидимых символов, сохраняют длину
 * строки, поэтому смещения находок совпадают с исходником. Удалённые
 * невидимые символы учитываются картой смещений `toSource`.
 */

const LETTER = "[\\p{L}\\p{N}_]";
export const WORD_START = `(?<!${LETTER})`;
export const WORD_END = `(?!${LETTER})`;

const WORD_RE = /[\p{L}\p{N}]+(?:[-’'][\p{L}\p{N}]+)*/gu;
const CYRILLIC_RE = /\p{Script=Cyrillic}/u;
const LATIN_RE = /\p{Script=Latin}/u;

/** Латинские буквы, неотличимые от кириллических. */
const LATIN_TO_CYRILLIC: Record<string, string> = {
  a: "а",
  e: "е",
  o: "о",
  p: "р",
  c: "с",
  x: "х",
  y: "у",
  k: "к",
  m: "м",
  t: "т",
  h: "н",
  b: "в",
  A: "А",
  E: "Е",
  O: "О",
  P: "Р",
  C: "С",
  X: "Х",
  Y: "У",
  K: "К",
  M: "М",
  T: "Т",
  H: "Н",
  B: "В",
};
const CYRILLIC_TO_LATIN: Record<string, string> = Object.fromEntries(
  Object.entries(LATIN_TO_CYRILLIC).map(([latin, cyrillic]) => [cyrillic, latin]),
);

/**
 * Латиница без диакритики. «á» в «Кáрмен» — знак ударения, а не подмена,
 * поэтому в сериях она не участвует, как цифры и прочие знаки.
 */
const ASCII_LATIN_RE = /[A-Za-z]/;

/** Серия букв одного алфавита; цифры и прочие знаки её не обрывают. */
interface Run {
  latin: boolean;
  start: number;
  end: number;
}

function scriptRuns(s: string): Run[] {
  const runs: Run[] = [];
  for (let i = 0; i < s.length; i += 1) {
    const c = s[i] as string;
    const latin = ASCII_LATIN_RE.test(c);
    if (!latin && !CYRILLIC_RE.test(c)) continue;
    const last = runs[runs.length - 1];
    if (last?.latin === latin) last.end = i + 1;
    else runs.push({ latin, start: i, end: i + 1 });
  }
  return runs;
}

/** Буквы серии без цифр и прочих знаков. */
function letters(s: string, r: Run): string[] {
  const own = r.latin ? ASCII_LATIN_RE : CYRILLIC_RE;
  return [...s.slice(r.start, r.end)].filter((c) => own.test(c));
}

/** У каждой буквы серии есть двойник в другом алфавите. */
function lookalike(s: string, r: Run): boolean {
  const map = r.latin ? LATIN_TO_CYRILLIC : CYRILLIC_TO_LATIN;
  return letters(s, r).every((c) => c in map);
}

/**
 * Подмена букв в части слова без дефисов. Алфавит слова выдаёт буква без
 * двойника: «щ» в русском слове, «n» в английском. Подозрительны буквы-двойники
 * другого алфавита. Если без двойника есть буквы обоих алфавитов (украинское
 * слово с латинской i вместо і) или нет ни одной, подозрителен алфавит,
 * в котором букв меньше. Законная смесь — латинская основа с русским
 * окончанием («Pythonе», «PHPшник», «OKей») и слипшийся предлог («вPython»).
 * Подмена — двойники внутри слова, в конце русского слова, латинские в начале
 * русского слова и одна-две кириллические в начале английского.
 */
function spoofed(s: string): boolean {
  const runs = scriptRuns(s);
  const first = runs[0];
  const second = runs[1];
  const last = runs[runs.length - 1];
  if (!first || !second || !last) return false;
  const of = (latin: boolean): string[] => runs.filter((r) => r.latin === latin).flatMap((r) => letters(s, r));
  const [latinLetters, cyrillicLetters] = [of(true), of(false)];
  const latinAnchored = latinLetters.some((c) => !(c in LATIN_TO_CYRILLIC));
  const cyrillicAnchored = cyrillicLetters.some((c) => !(c in CYRILLIC_TO_LATIN));
  const suspectScript = (latin: boolean): boolean =>
    latinAnchored !== cyrillicAnchored
      ? latin === cyrillicAnchored
      : latin
        ? latinLetters.length <= cyrillicLetters.length
        : cyrillicLetters.length <= latinLetters.length;
  const suspect = (r: Run): boolean => suspectScript(r.latin) && lookalike(s, r);
  // Алфавиты чередуются, поэтому внутренняя серия зажата буквами другого алфавита.
  for (let k = 1; k < runs.length - 1; k += 1) {
    const r = runs[k] as Run;
    // «FхG», «mхn»: одиночная «х» между одиночными буквами — знак умножения.
    const times =
      /^[\u0445\u0425xX]$/u.test(letters(s, r).join("")) &&
      letters(s, runs[k - 1] as Run).length === 1 &&
      letters(s, runs[k + 1] as Run).length === 1;
    if (!times && suspect(r)) return true;
  }
  if (last.latin && suspect(last)) return true;
  if (!suspect(first)) return false;
  const head = letters(s, first).join("");
  // Латинское сокращение перед русским суффиксом («OKей», «PHPшник»), но не заглавное русское слово.
  const lowerNext = /\p{Ll}/u.test(s[second.start] ?? "");
  if (first.latin) return !(head.length >= 2 && head === head.toUpperCase() && lowerNext);
  // Кириллица перед английским словом: подмена, если слово продолжается строчными,
  // и слипшийся предлог, если дальше заглавная («вPython»).
  return head.length <= 2 && lowerNext;
}

/** Символы, которые удаляются из текста перед поиском. */
const STRIPPED_RE = /[\u200B-\u200D\u2060\uFEFF\u00AD]/u;
const SOFT_HYPHEN = "\u00AD";
const BOM = "\uFEFF";
const ZWJ = "\u200D";
const EMOJI_BEFORE_ZWJ = /(?:\p{Extended_Pictographic}|\p{Emoji_Modifier}|\uFE0F)$/u;
const EMOJI_AFTER_ZWJ = /^\p{Extended_Pictographic}/u;

/** U+200D между частями эмодзи (👨 + 💻) — соединитель, а не вставка. */
function joinsEmoji(s: string, i: number): boolean {
  return EMOJI_BEFORE_ZWJ.test(s.slice(Math.max(0, i - 2), i)) && EMOJI_AFTER_ZWJ.test(s.slice(i + 1, i + 3));
}

export interface Prepared {
  /** Исходный текст. */
  source: string;
  /** Нормализованный текст: без невидимых символов, ё→е, U+2010/U+2011→дефис, без подмены букв. */
  text: string;
  /** Нормализованный текст, где код и YAML-шапка заменены пробелами. */
  noCode: string;
  /** Нормализованный текст, где замаскированы все защищённые области. */
  prose: string;
  /** Смещение в `text` → смещение в `source`. */
  toSource: number[];
  /** Невидимые вставки. BOM в начале текста и соединитель внутри эмодзи сюда не входят. */
  invisible: { index: number; char: string }[];
  /** Позиции мягких переносов (U+00AD) в исходнике: их ставят Word и копирование из PDF. */
  softHyphens: number[];
  homoglyphs: { index: number; word: string }[];
  lineStarts: number[];
}

function maskRange(chars: string[], start: number, end: number): void {
  for (let i = start; i < end; i += 1) {
    if (chars[i] !== "\n") chars[i] = " ";
  }
}

function maskRegex(chars: string[], text: string, re: RegExp): void {
  for (const m of text.matchAll(re)) {
    const start = m.index ?? 0;
    maskRange(chars, start, start + m[0].length);
  }
}

/** Маскирует YAML-шапку, блоки кода и инлайн-код. */
function maskCode(text: string): string {
  // Массив code units: split("") сохраняет длину строки и смещения.
  const units = text.split("");
  const fm = /^---\n[\s\S]*?\n---(?:\n|$)/.exec(text);
  if (fm) maskRange(units, 0, fm[0].length);
  maskRegex(units, text, /^(```|~~~)[^\n]*\n[\s\S]*?^\1[^\n]*$/gm);
  maskRegex(units, text, /`[^`\n]+`/g);
  return units.join("");
}

/** Маскирует всё, что скилл считает защищённым содержимым. */
function maskProtected(noCode: string): string {
  const units = noCode.split("");
  maskRegex(units, noCode, /<!--[\s\S]*?-->/g);
  maskRegex(units, noCode, /\$\$[\s\S]*?\$\$/g);
  maskRegex(units, noCode, /\$[^$\n]+\$/g);
  maskRegex(units, noCode, /\\\([\s\S]*?\\\)/g);
  maskRegex(units, noCode, /https?:\/\/[^\s)>\]»]+/g);
  maskRegex(units, noCode, /\]\([^)\n]*\)/g);
  maskRegex(units, noCode, /^[ \t]*\|.*$/gm);
  maskRegex(units, noCode, /^[ \t]*>.*$/gm);
  // Цитаты в «ёлочках» и „лапках“ (до 400 знаков, без перевода абзаца).
  maskRegex(units, noCode, /«[^«»\n]{0,400}»/g);
  maskRegex(units, noCode, /„[^„“\n]{0,400}“/g);
  maskRegex(units, noCode, /"[^"\n]{1,400}"/g);
  maskRegex(units, noCode, /“[^“”\n]{1,400}”/g);
  // Ссылки на литературу: [12], [3; 7], [12, с. 45].
  maskRegex(units, noCode, /\[\d+(?:[,;–-]\s*\d+)*(?:,\s*с\.\s*\d+(?:[–-]\d+)?)?\]/g);
  // Ссылки pandoc: [@key], [@a; @b, с. 5].
  maskRegex(units, noCode, /\[-?@[^\]\n]+\]/g);
  return units.join("");
}

function computeLineStarts(s: string): number[] {
  const starts = [0];
  for (let i = 0; i < s.length; i += 1) if (s[i] === "\n") starts.push(i + 1);
  return starts;
}

export function lineCol(lineStarts: number[], index: number): { line: number; column: number } {
  let lo = 0;
  let hi = lineStarts.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if ((lineStarts[mid] ?? 0) <= index) lo = mid;
    else hi = mid - 1;
  }
  return { line: lo + 1, column: index - (lineStarts[lo] ?? 0) + 1 };
}

export function prepare(source: string): Prepared {
  const invisible: { index: number; char: string }[] = [];
  const softHyphens: number[] = [];
  const toSource: number[] = [];
  let stripped = "";
  for (let i = 0; i < source.length; i += 1) {
    const ch = source[i] as string;
    if (STRIPPED_RE.test(ch)) {
      if (ch === SOFT_HYPHEN) softHyphens.push(i);
      else if (!(ch === BOM && i === 0) && !(ch === ZWJ && joinsEmoji(source, i)))
        invisible.push({ index: i, char: ch });
      continue;
    }
    // Неразрывный дефис и U+2010 с клавиатуры не набрать, их ставят модели; для словарей это обычный дефис.
    stripped += ch === "\u2010" || ch === "\u2011" ? "-" : ch;
    toSource.push(i);
  }
  toSource.push(source.length);

  // Подмена латиницы внутри кириллических слов. Части дефисного слова проверяются
  // по отдельности: в «HTTP-запрос» латинское сокращение стоит при русском слове законно.
  const homoglyphs: { index: number; word: string }[] = [];
  const units = stripped.split("");
  for (const m of stripped.matchAll(WORD_RE)) {
    const word = m[0];
    if (!CYRILLIC_RE.test(word) || !LATIN_RE.test(word)) continue;
    const start = m.index ?? 0;
    let found = false;
    for (const part of word.matchAll(/[\p{L}\p{N}]+/gu)) {
      const s = part[0];
      if (!CYRILLIC_RE.test(s) || !LATIN_RE.test(s) || !spoofed(s)) continue;
      found = true;
      // Буквы-двойники приводятся к алфавиту слова, чтобы словари видели настоящее слово.
      const latin = [...s].filter((c) => LATIN_RE.test(c)).length;
      const toLatin = latin > [...s].filter((c) => CYRILLIC_RE.test(c)).length;
      const map = toLatin ? CYRILLIC_TO_LATIN : LATIN_TO_CYRILLIC;
      const at = start + (part.index ?? 0);
      for (let k = 0; k < s.length; k += 1) {
        const c = units[at + k] as string;
        if (c in map) units[at + k] = map[c] as string;
      }
    }
    if (found) homoglyphs.push({ index: toSource[start] ?? start, word });
  }
  const text = units.join("").replace(/ё/g, "е").replace(/Ё/g, "Е");
  const noCode = maskCode(text);
  const prose = maskProtected(noCode);
  return {
    source,
    text,
    noCode,
    prose,
    toSource,
    invisible,
    softHyphens,
    homoglyphs,
    lineStarts: computeLineStarts(source),
  };
}

export function words(s: string): string[] {
  return s.match(WORD_RE) ?? [];
}

/** «1 слово», «2 слова», «5 слов». */
export function plural(n: number, one: string, few: string, many: string): string {
  const d = n % 10;
  const dd = n % 100;
  const form = d === 1 && dd !== 11 ? one : d >= 2 && d <= 4 && (dd < 12 || dd > 14) ? few : many;
  return `${n} ${form}`;
}

export type BlockKind = "prose" | "heading" | "list" | "table" | "quote" | "code" | "empty";

export interface Block {
  kind: BlockKind;
  start: number;
  end: number;
  /** Текст блока из `prose` (замаскированный). */
  text: string;
}

/** Разбивает текст на блоки по пустым строкам; заголовок всегда отдельный блок. */
export function blocks(p: Prepared): Block[] {
  const out: Block[] = [];
  const src = p.prose;
  const lines = src.split("\n");
  let offset = 0;
  let cur: { start: number; lines: string[]; rawLines: string[] } | null = null;
  const flush = (): void => {
    if (!cur) return;
    const text = cur.lines.join("\n");
    const raw = cur.rawLines.join("\n");
    const start = cur.start;
    const end = start + text.length;
    let kind: BlockKind = "prose";
    const firstRaw = cur.rawLines[0] ?? "";
    if (text.trim() === "") {
      kind = /^\s*(```|~~~|---)/.test(firstRaw) || raw.trim() !== "" ? "code" : "empty";
      if (/^\s*[|>]/.test(firstRaw)) kind = /^\s*\|/.test(firstRaw) ? "table" : "quote";
    } else if (cur.lines.every((l) => l.trim() === "" || /^\s*([-*+•]|\d+[.)])\s+/.test(l))) {
      kind = "list";
    }
    out.push({ kind, start, end, text });
    cur = null;
  };
  const rawLines = p.text.split("\n");
  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i] as string;
    const raw = rawLines[i] ?? "";
    if (/^#{1,6}\s/.test(line)) {
      flush();
      out.push({ kind: "heading", start: offset, end: offset + line.length, text: line });
    } else if (line.trim() === "" && raw.trim() === "") {
      flush();
    } else {
      if (!cur) cur = { start: offset, lines: [], rawLines: [] };
      cur.lines.push(line);
      cur.rawLines.push(raw);
    }
    offset += line.length + 1;
  }
  flush();
  return out;
}

export interface Sentence {
  start: number;
  end: number;
  text: string;
  words: number;
}

const ABBREVIATIONS = new Set([
  "т",
  "е",
  "д",
  "п",
  "г",
  "гг",
  "в",
  "вв",
  "с",
  "см",
  "рис",
  "табл",
  "др",
  "им",
  "ул",
  "проф",
  "акад",
  "вып",
  "стр",
  "ст",
  "н",
  "э",
  "тыс",
  "млн",
  "млрд",
  "руб",
  "коп",
  "мин",
  "сек",
  "ч",
  "кв",
  "обл",
  "ред",
  "изд",
  "пер",
  "сб",
  "т.е",
  "т.д",
  "т.п",
  "т.к",
  "etc",
  "et",
  "al",
  "no",
  "vol",
  "pp",
  "fig",
  "eq",
  "doi",
  "проч",
  "напр",
  "прим",
  "ср",
  "англ",
  "лат",
  "гл",
]);

/** Разбивает фрагмент на предложения с учётом сокращений и инициалов. */
export function sentences(text: string, base = 0): Sentence[] {
  const out: Sentence[] = [];
  let start = 0;
  // Перенос строки внутри абзаца (ручной перенос в Markdown) предложение не заканчивает;
  // заканчивают пустая строка, конец текста и начало пункта списка, цитаты или таблицы.
  const re = /[.!?…]+["»”)]*(?=\s+["«„(—–-]?\s*[\p{Lu}\d]|\s*$)|\n(?=[ \t]*(?:\n|$|[-*+•][ \t]|\d+[.)][ \t]|[>|]))/gu;
  for (const m of text.matchAll(re)) {
    const idx = m.index ?? 0;
    if (m[0] !== "\n") {
      const before = text.slice(Math.max(start, idx - 12), idx);
      const lastWord = /([\p{L}.]+)$/u.exec(before)?.[1]?.toLowerCase() ?? "";
      if (ABBREVIATIONS.has(lastWord.replace(/\.$/, "")) || /^\p{Lu}$/u.test(before.slice(-1))) {
        continue;
      }
    }
    const end = idx + m[0].length;
    pushSentence(out, text, start, end, base);
    start = end;
  }
  pushSentence(out, text, start, text.length, base);
  return out;
}

function pushSentence(out: Sentence[], text: string, start: number, end: number, base: number): void {
  const raw = text.slice(start, end);
  const trimmedStart = start + (raw.length - raw.trimStart().length);
  const t = raw.trim();
  if (!t) return;
  const n = words(t).length;
  if (n === 0) return;
  out.push({ start: base + trimmedStart, end: base + trimmedStart + t.length, text: t, words: n });
}

export function mean(xs: number[]): number {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
}

/** Коэффициент вариации (σ/μ). */
export function cv(xs: number[]): number {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  if (m === 0) return 0;
  const v = xs.reduce((a, b) => a + (b - m) ** 2, 0) / (xs.length - 1);
  return Math.sqrt(v) / m;
}

/** Скользящий TTR (MATTR) по окну: устойчив к длине текста. */
export function mattr(tokens: string[], window = 100): number {
  if (tokens.length === 0) return 0;
  const lower = tokens.map((t) => t.toLowerCase());
  if (lower.length <= window) return new Set(lower).size / lower.length;
  const counts = new Map<string, number>();
  let distinct = 0;
  for (let i = 0; i < window; i += 1) {
    const t = lower[i] as string;
    const c = counts.get(t) ?? 0;
    if (c === 0) distinct += 1;
    counts.set(t, c + 1);
  }
  let sum = distinct / window;
  let steps = 1;
  for (let i = window; i < lower.length; i += 1) {
    const add = lower[i] as string;
    const drop = lower[i - window] as string;
    const cd = (counts.get(drop) ?? 0) - 1;
    counts.set(drop, cd);
    if (cd === 0) distinct -= 1;
    const ca = counts.get(add) ?? 0;
    if (ca === 0) distinct += 1;
    counts.set(add, ca + 1);
    sum += distinct / window;
    steps += 1;
  }
  return sum / steps;
}

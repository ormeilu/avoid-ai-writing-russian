/**
 * Таблицы для вывода в терминал: колонки ужимаются под ширину экрана,
 * длинный текст переносится по словам внутри ячейки или обрезается с «…».
 */

export interface Column {
  title: string;
  align?: "left" | "right";
  /** Колонку можно сужать, если таблица не влезает. */
  flex?: boolean;
  /** Не уже этого (для flex-колонок). */
  min?: number;
  /** Переносить по словам; иначе обрезать с «…». */
  wrap?: boolean;
}

export interface TableOptions {
  /** Полная ширина строки, включая отступ. */
  width: number;
  indent?: number;
  gap?: number;
  /** Раскраска ячейки после раскладки; получает текст строки ячейки без пробелов выравнивания. */
  paint?: (row: number, col: number, text: string) => string;
  /** Раскраска заголовка и линейки. */
  paintHeader?: (text: string) => string;
}

/** Переносит текст по словам в строки не длиннее width; слишком длинное слово режется. */
export function wrapText(text: string, width: number): string[] {
  const lines: string[] = [];
  let line = "";
  for (const word of text.split(/\s+/).filter(Boolean)) {
    let w = word;
    while (w.length > width) {
      if (line) {
        lines.push(line);
        line = "";
      }
      lines.push(w.slice(0, width));
      w = w.slice(width);
    }
    if (!line) line = w;
    else if (line.length + 1 + w.length <= width) line += ` ${w}`;
    else {
      lines.push(line);
      line = w;
    }
  }
  if (line || !lines.length) lines.push(line);
  return lines;
}

export function truncate(text: string, width: number): string {
  return text.length <= width ? text : `${text.slice(0, Math.max(0, width - 1)).trimEnd()}…`;
}

/** Ширины колонок: естественные, затем сужение самых широких flex-колонок до влезания. */
export function layout(columns: Column[], rows: string[][], opts: TableOptions): number[] {
  const indent = opts.indent ?? 2;
  const gap = opts.gap ?? 2;
  const widths = columns.map((col, k) => Math.max(col.title.length, ...rows.map((r) => (r[k] ?? "").length)));
  const total = (): number => indent + widths.reduce((a, b) => a + b, 0) + gap * (columns.length - 1);
  while (total() > opts.width) {
    let widest = -1;
    for (let k = 0; k < columns.length; k += 1) {
      const col = columns[k] as Column;
      const w = widths[k] as number;
      if (col.flex && w > (col.min ?? 10) && (widest < 0 || w > (widths[widest] as number))) widest = k;
    }
    if (widest < 0) break;
    widths[widest] = (widths[widest] as number) - 1;
  }
  return widths;
}

export function table(columns: Column[], rows: string[][], opts: TableOptions): string[] {
  const indent = " ".repeat(opts.indent ?? 2);
  const gap = " ".repeat(opts.gap ?? 2);
  const widths = layout(columns, rows, opts);
  const header = opts.paintHeader ?? ((s: string): string => s);
  const pad = (text: string, k: number): string => {
    const w = widths[k] as number;
    return columns[k]?.align === "right" ? text.padStart(w) : text.padEnd(w);
  };
  const lines: string[] = [];
  const last = columns.length - 1;
  const join = (cells: string[]): string => `${indent}${cells.join(gap)}`.trimEnd();
  lines.push(header(join(columns.map((col, k) => (k === last ? col.title : pad(col.title, k))))));
  lines.push(header(`${indent}${widths.map((w) => "─".repeat(w)).join(gap)}`));
  rows.forEach((row, r) => {
    const cells = columns.map((col, k) => {
      const text = row[k] ?? "";
      const w = widths[k] as number;
      return col.wrap ? wrapText(text, w) : [truncate(text, w)];
    });
    const height = Math.max(...cells.map((c) => c.length));
    for (let i = 0; i < height; i += 1) {
      const parts = cells.map((c, k) => {
        const text = c[i] ?? "";
        const padded = k === last && columns[k]?.align !== "right" ? text : pad(text, k);
        if (!text || !opts.paint) return padded;
        // Красим только текст, чтобы пробелы выравнивания не попадали внутрь цвета.
        const at = padded.indexOf(text);
        return `${padded.slice(0, at)}${opts.paint(r, k, text)}${padded.slice(at + text.length)}`;
      });
      lines.push(join(parts));
    }
  });
  return lines;
}

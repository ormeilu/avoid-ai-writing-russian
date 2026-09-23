/**
 * Оценка по фрагментам в духе модуля поиска сгенерированного текста
 * системы «Антиплагиат».
 *
 * Настоящий классификатор закрыт, поэтому здесь приближение: каждый
 * фрагмент (абзац или склейка коротких абзацев) описывается набором
 * интерпретируемых признаков, логистическая модель переводит их в
 * вероятность, а доля ИИ-текста считается как доля знаков во фрагментах
 * выше порога — так же выглядит итог в отчёте системы.
 *
 * Веса по умолчанию подобраны вручную. Команда `calibrate` дообучает их
 * на фрагментах, которые система реально подсветила в ваших отчётах.
 */

import { analyzeInternal, WEIGHTS } from "./detect.ts";
import { cv, lineCol, mattr, mean, sentences, words } from "./text.ts";
import type { AnalyzeOptions, Issue } from "./types.ts";

export const FEATURE_NAMES = [
  "плотность примет",
  "однообразие длины предложений",
  "типичная длина предложения",
  "канцелярит",
  "бедная пунктуация",
  "однообразные начала предложений",
  "бедный словарь",
] as const;

export interface Model {
  bias: number;
  weights: number[];
  threshold: number;
}

export const DEFAULT_MODEL: Model = {
  bias: -4.4,
  weights: [1.1, 2.6, 1.4, 0.35, 1.2, 1.3, 1.6],
  threshold: 0.5,
};

export interface Fragment {
  start: number;
  end: number;
  line: number;
  endLine: number;
  words: number;
  features: number[];
  probability: number;
  ai: boolean;
  reasons: string[];
  preview: string;
}

export interface AntiplagiatReport {
  /** Доля знаков текста во фрагментах, помеченных как ИИ, 0–100. */
  aiShare: number;
  fragments: Fragment[];
  suspicious: boolean;
  suspiciousReasons: string[];
  model: "default" | "calibrated";
  threshold: number;
}

const MIN_FRAGMENT_WORDS = 40;

function sigmoid(z: number): number {
  return 1 / (1 + Math.exp(-z));
}

export function predict(model: Model, f: number[]): number {
  let z = model.bias;
  for (let i = 0; i < f.length; i += 1) z += (model.weights[i] ?? 0) * (f[i] ?? 0);
  return sigmoid(z);
}

export function featuresFor(text: string, fragmentIssues: Issue[]): number[] {
  const ws = words(text);
  const n = Math.max(ws.length, 1);
  const ss = sentences(text).filter((s) => s.words >= 2);
  const lens = ss.map((s) => s.words);

  const signal = fragmentIssues.filter((i) => !i.styleOnly).reduce((a, i) => a + (WEIGHTS[i.type] ?? 1), 0);
  const density = Math.min(4, (signal * 100) / n / 3);
  const uniformity = lens.length >= 3 ? Math.max(0, 1 - cv(lens) / 0.6) : 0.3;
  const m = mean(lens);
  const typical = lens.length ? Math.exp(-(((m - 19) / 7) ** 2)) : 0;
  const clerical = Math.min(3, (fragmentIssues.filter((i) => i.styleOnly).length * 100) / n / 2);
  const rare = ss.filter((s) => /[;:()!?…]|\s—\s/.test(s.text)).length;
  const poorPunct = ss.length >= 3 ? Math.max(0, 1 - rare / ss.length / 0.5) : 0.3;
  const firsts = ss.map((s) => (words(s.text)[0] ?? "").toLowerCase());
  const openerRepeat = firsts.length >= 3 ? 1 - new Set(firsts).size / firsts.length : 0;
  const diversity = ws.length >= 30 ? mattr(ws, Math.min(50, ws.length)) : 0.8;
  const poorVocab = Math.max(0, Math.min(1, (0.85 - diversity) / 0.2));
  return [density, uniformity, typical, clerical, poorPunct, openerRepeat, poorVocab];
}

function reasonsFor(model: Model, f: number[]): string[] {
  return f
    .map((v, i) => ({ i, c: v * (model.weights[i] ?? 0) }))
    .filter((x) => x.c > 0.5)
    .sort((a, b) => b.c - a.c)
    .slice(0, 3)
    .map((x) => FEATURE_NAMES[x.i] ?? "");
}

interface RawFragment {
  start: number;
  end: number;
  text: string;
}

/** Абзацы прозы и списков; короткие склеиваются со следующими. */
function fragmentsOf(internals: ReturnType<typeof analyzeInternal>["internals"]): RawFragment[] {
  const out: RawFragment[] = [];
  let cur: RawFragment | null = null;
  for (const b of internals.blocks) {
    if (b.kind !== "prose" && b.kind !== "list") {
      if (b.kind === "heading" && cur && words(cur.text).length >= MIN_FRAGMENT_WORDS / 2) {
        out.push(cur);
        cur = null;
      }
      continue;
    }
    if (!cur) cur = { start: b.start, end: b.end, text: b.text };
    else {
      cur.end = b.end;
      cur.text += `\n\n${b.text}`;
    }
    if (words(cur.text).length >= MIN_FRAGMENT_WORDS) {
      out.push(cur);
      cur = null;
    }
  }
  if (cur) {
    const last = out[out.length - 1];
    if (last && words(cur.text).length < MIN_FRAGMENT_WORDS / 2) {
      last.end = cur.end;
      last.text += `\n\n${cur.text}`;
    } else out.push(cur);
  }
  return out;
}

export function antiplagiat(source: string, options: AnalyzeOptions & { model?: Model } = {}): AntiplagiatReport {
  const { result, internals } = analyzeInternal(source, { context: options.context ?? "academic" });
  const model = options.model ?? DEFAULT_MODEL;
  const p = internals.prepared;
  const raws = fragmentsOf(internals);
  const toSrc = (i: number): number => p.toSource[i] ?? i;
  let aiChars = 0;
  let allChars = 0;
  const fragments: Fragment[] = raws.map((r) => {
    const s = toSrc(r.start);
    const e = toSrc(r.end);
    const inside = result.issues.filter((i) => i.index >= s && i.index < e);
    const f = featuresFor(r.text, inside);
    const prob = predict(model, f);
    const chars = r.text.replace(/\s+/g, " ").trim().length;
    allChars += chars;
    const ai = prob >= model.threshold;
    if (ai) aiChars += chars;
    const lineOf = (idx: number): number => lineCol(p.lineStarts, idx).line;
    return {
      start: s,
      end: e,
      line: lineOf(s),
      endLine: lineOf(e),
      words: words(r.text).length,
      features: f.map((v) => Number(v.toFixed(3))),
      probability: Number(prob.toFixed(3)),
      ai,
      reasons: reasonsFor(model, f),
      preview: p.source.slice(s, Math.min(e, s + 90)).replace(/\s+/g, " ").trim(),
    };
  });

  const suspiciousReasons: string[] = [];
  if (p.invisible.length) suspiciousReasons.push(`невидимые символы: ${p.invisible.length}`);
  if (p.homoglyphs.length) suspiciousReasons.push(`слова со смешанной латиницей и кириллицей: ${p.homoglyphs.length}`);

  return {
    aiShare: allChars ? Number(((aiChars * 100) / allChars).toFixed(1)) : 0,
    fragments,
    suspicious: suspiciousReasons.length > 0,
    suspiciousReasons,
    model: options.model ? "calibrated" : "default",
    threshold: model.threshold,
  };
}

// ─── Калибровка по реальным отчётам ─────────────────────────────────────

export interface Sample {
  f: number[];
  y: 0 | 1;
}

export interface CalibrationFile {
  version: 1;
  model: Model;
  samples: Sample[];
}

function norm(s: string): string {
  return s
    .toLowerCase()
    .replace(/ё/g, "е")
    .replace(/[*_`#>|]/g, "")
    .replace(/[«»„“”"]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

/**
 * Размечает фрагменты документа по кускам, которые система подсветила
 * в отчёте (скопированы в текстовый файл, разделены пустой строкой).
 */
export function labelFragments(source: string, marked: string[]): Sample[] {
  const { internals, result } = analyzeInternal(source, { context: "academic" });
  const p = internals.prepared;
  const keys = marked.map(norm).filter((m) => m.length >= 30).map((m) => m.slice(0, 60));
  return fragmentsOf(internals).map((r) => {
    const s = p.toSource[r.start] ?? r.start;
    const e = p.toSource[r.end] ?? r.end;
    const inside = result.issues.filter((i) => i.index >= s && i.index < e);
    const text = norm(p.source.slice(s, e));
    const y: 0 | 1 = keys.some((k) => text.includes(k)) ? 1 : 0;
    return { f: featuresFor(r.text, inside), y };
  });
}

/** Логистическая регрессия с L2 и выбором порога по сбалансированной точности. */
export function fit(samples: Sample[], start: Model = DEFAULT_MODEL): Model {
  const pos = samples.filter((s) => s.y === 1).length;
  const neg = samples.length - pos;
  if (pos === 0 || neg === 0) return start;
  const wPos = samples.length / (2 * pos);
  const wNeg = samples.length / (2 * neg);
  const w = [...start.weights];
  let b = start.bias;
  const lr = 0.05;
  const l2 = 0.01;
  for (let it = 0; it < 4000; it += 1) {
    const gw = new Array<number>(w.length).fill(0);
    let gb = 0;
    for (const s of samples) {
      const pr = predict({ bias: b, weights: w, threshold: 0.5 }, s.f);
      const err = (pr - s.y) * (s.y === 1 ? wPos : wNeg);
      for (let k = 0; k < w.length; k += 1) gw[k] = (gw[k] ?? 0) + err * (s.f[k] ?? 0);
      gb += err;
    }
    for (let k = 0; k < w.length; k += 1) {
      w[k] = (w[k] ?? 0) - lr * ((gw[k] ?? 0) / samples.length + l2 * ((w[k] ?? 0) - (start.weights[k] ?? 0)));
    }
    b -= (lr * gb) / samples.length;
  }
  const model: Model = { bias: b, weights: w, threshold: 0.5 };
  let best = { t: 0.5, score: -1 };
  for (let t = 0.2; t <= 0.8001; t += 0.02) {
    let tp = 0;
    let tn = 0;
    for (const s of samples) {
      const hit = predict(model, s.f) >= t;
      if (hit && s.y === 1) tp += 1;
      if (!hit && s.y === 0) tn += 1;
    }
    const bal = (tp / pos + tn / neg) / 2;
    if (bal > best.score) best = { t, score: bal };
  }
  model.threshold = Number(best.t.toFixed(2));
  model.bias = Number(model.bias.toFixed(4));
  model.weights = model.weights.map((x) => Number(x.toFixed(4)));
  return model;
}

export function balancedAccuracy(model: Model, samples: Sample[]): number {
  const pos = samples.filter((s) => s.y === 1);
  const neg = samples.filter((s) => s.y === 0);
  if (!pos.length || !neg.length) return Number.NaN;
  const tp = pos.filter((s) => predict(model, s.f) >= model.threshold).length;
  const tn = neg.filter((s) => predict(model, s.f) < model.threshold).length;
  return (tp / pos.length + tn / neg.length) / 2;
}

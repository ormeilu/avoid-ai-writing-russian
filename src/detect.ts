/**
 * Детектор ИИ-стиля для русского текста.
 *
 * Модель оценки. У каждой категории есть вес (WEIGHTS). Находки
 * дедуплицируются по паре (тип, текст); сырой балл — сумма весов
 * уникальных находок. Правки ради краткости (styleOnly) весят мало: это
 * совет по стилю, а не довод об авторстве. Сырой балл нормируется по длине
 * текста и насыщается к 100, чтобы длинный текст с той же плотностью
 * примет не набирал бесконечно.
 */

import { ACADEMIC_FORMULAS, ALL_LEXICON, type LexEntry, PHRASE3, TECHNICAL_TERMS, TIER2, TIER3 } from "./lexicon.ts";
import {
  type Block,
  blocks,
  cv,
  lineCol,
  mattr,
  mean,
  type Prepared,
  plural,
  prepare,
  type Sentence,
  sentences,
  words,
} from "./text.ts";
import type { AnalysisResult, AnalyzeOptions, ContextMode, Issue, Severity } from "./types.ts";

export const WEIGHTS: Record<string, number> = {
  // Невидимые символы, подмена букв и мягкие переносы — гигиена документа, а не ИИ-стиль:
  // в человеческих текстах LLMTrace они встречаются чаще, чем в сгенерированных. В оценку
  // не идут; первые два делают документ подозрительным (флаг `suspicious`).
  "invisible-chars": 0,
  homoglyph: 0,
  "soft-hyphen": 0,
  "chat-markup": 10,
  "ai-url": 8,
  placeholder: 8,
  cutoff: 10,
  chatbot: 8,
  sycophancy: 6,
  "vague-attribution": 5,
  significance: 5,
  tier1: 3,
  "tier1-clarity": 0.5,
  tier2: 1.5,
  "tier2-cluster": 3,
  tier3: 1,
  phrase3: 1,
  "phrase3-cluster": 3,
  calque: 3,
  template: 3,
  transition: 1.5,
  "transition-run": 3,
  filler: 2,
  "hollow-intensifier": 1.5,
  confidence: 1.5,
  "hedge-stack": 2,
  "vague-endorsement": 1.5,
  "future-narrative": 3,
  "generic-conclusion": 2,
  novelty: 3,
  "unmeasured-claim": 2,
  lets: 3,
  reasoning: 4,
  acknowledgment: 3,
  "narrated-candor": 3,
  hook: 3,
  "social-closer": 3,
  "launch-intro": 3,
  "fake-casual": 3,
  "speculative-opener": 3,
  "false-concession": 2,
  "moral-adjective": 2,
  "real-inflation": 1.5,
  "not-x-but-y": 3,
  "em-dash-splice": 2,
  "hyphen-dash": 0.5,
  "straight-quotes": 1,
  "english-quotes": 2,
  "decimal-point": 1,
  "title-case": 2,
  "bold-overuse": 2,
  "emoji-header": 3,
  "bullet-np-list": 3,
  "hashtag-stuffing": 4,
  "genitive-chain": 1.5,
  "passive-run": 2,
  "same-opener": 2,
  staccato: 2,
  "negation-chain": 2,
  "stacked-questions": 2,
  "uniform-sentences": 5,
  "uniform-paragraphs": 3,
  "low-diversity": 3,
};

export const TYPE_LABELS: Record<string, string> = {
  "invisible-chars": "Невидимые символы",
  homoglyph: "Подмена букв",
  "soft-hyphen": "Мягкие переносы",
  "chat-markup": "Разметка цитирования из чата",
  "ai-url": "Параметр ИИ-инструмента в ссылке",
  placeholder: "Незаполненная заглушка",
  cutoff: "Оговорка об отсечке знаний",
  chatbot: "След чат-бота",
  sycophancy: "Угодливость",
  "vague-attribution": "Размытая ссылка на авторитет",
  significance: "Раздувание значимости",
  tier1: "Слово-маркер ИИ",
  "tier1-clarity": "Канцелярит",
  tier2: "Слово уровня 2",
  "tier2-cluster": "Скопление слов уровня 2",
  tier3: "Перегруженное слово",
  phrase3: "Фразовый штамп",
  "phrase3-cluster": "Скопление фразовых штампов",
  calque: "Калька с английского",
  template: "Шаблонная конструкция",
  transition: "Шаблонный переход",
  "transition-run": "Переходы в начале абзацев подряд",
  filler: "Подушка",
  "hollow-intensifier": "Пустой усилитель",
  confidence: "Нагнетание уверенности",
  "hedge-stack": "Нагромождённая оговорка",
  "vague-endorsement": "Пустое одобрение",
  "future-narrative": "Туманный прогноз",
  "generic-conclusion": "Пустой вывод",
  novelty: "Раздувание новизны",
  "unmeasured-claim": "Оценка без числа",
  lets: "«Давайте…»",
  reasoning: "След рассуждения модели",
  acknowledgment: "Пересказ вопроса",
  "narrated-candor": "Показная откровенность",
  hook: "Хук из соцсетей",
  "social-closer": "Рекомендательная концовка",
  "launch-intro": "Презентация как выход на сцену",
  "fake-casual": "Фальшиво-разговорный регистр",
  "speculative-opener": "«Представьте…»",
  "false-concession": "Ложная уступка",
  "moral-adjective": "Моральный эпитет у неодушевлённого",
  "real-inflation": "«Настоящий/реальный» как усилитель",
  "not-x-but-y": "«Не X, а Y»",
  "em-dash-splice": "Тире-связка ради эффекта",
  "hyphen-dash": "Дефис вместо тире",
  "straight-quotes": "Прямые кавычки в русском тексте",
  "english-quotes": "Английские кавычки в русском тексте",
  "decimal-point": "Десятичная точка",
  "title-case": "Title Case в заголовке",
  "bold-overuse": "Избыток жирного",
  "emoji-header": "Эмодзи в заголовке",
  "bullet-np-list": "Список из голых именных групп",
  "hashtag-stuffing": "Хэштеги",
  "genitive-chain": "Цепочка отглагольных существительных",
  "passive-run": "Пассив без деятеля подряд",
  "same-opener": "Одинаковые начала предложений",
  staccato: "Рубленые фрагменты",
  "negation-chain": "Цепочка отрицаний",
  "stacked-questions": "Стопка риторических вопросов",
  "uniform-sentences": "Одинаковая длина предложений",
  "uniform-paragraphs": "Одинаковые абзацы",
  "low-diversity": "Бедный словарь",
};

const SKIP_IN_CHAT = new Set([
  "straight-quotes",
  "english-quotes",
  "decimal-point",
  "hyphen-dash",
  "em-dash-splice",
  "title-case",
  "bold-overuse",
  "genitive-chain",
  "passive-run",
  "uniform-sentences",
  "uniform-paragraphs",
  "low-diversity",
  "same-opener",
  "staccato",
  "bullet-np-list",
  "tier3",
  "phrase3",
  "tier2",
  "tier2-cluster",
  "transition-run",
]);

interface Ctx {
  p: Prepared;
  mode: ContextMode;
  wordCount: number;
  issues: Issue[];
}

function add(
  ctx: Ctx,
  type: string,
  rule: string,
  severity: Severity,
  index: number,
  text: string,
  hint: string,
  styleOnly = false,
): void {
  if (ctx.mode === "chat" && SKIP_IN_CHAT.has(type)) return;
  const src = ctx.p.toSource[index] ?? index;
  const { line, column } = lineCol(ctx.p.lineStarts, src);
  const issue: Issue = { type, rule, severity, text: text.trim().slice(0, 160), index: src, line, column, hint };
  if (styleOnly) issue.styleOnly = true;
  ctx.issues.push(issue);
}

function applies(e: LexEntry, mode: ContextMode): boolean {
  return !e.skip?.includes(mode);
}

/** Отрезки, где совпадает `re`; строится один раз на текст. */
function ranges(re: RegExp, text: string): [number, number][] {
  re.lastIndex = 0;
  return [...text.matchAll(re)].map((m) => [m.index ?? 0, (m.index ?? 0) + m[0].length]);
}

/** Пересекает ли [index, index+length) хотя бы один отрезок (отрезки отсортированы). */
function hitsRange(rs: [number, number][], index: number, length: number): boolean {
  let lo = 0;
  let hi = rs.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if ((rs[mid]?.[1] ?? 0) <= index) lo = mid + 1;
    else hi = mid;
  }
  const r = rs[lo];
  return r !== undefined && r[0] < index + length;
}

// ─── Словарные правила ──────────────────────────────────────────────────
function detectLexicon(ctx: Ctx, bs: Block[]): void {
  const { p, mode } = ctx;
  const per1000 = (n: number): number => (ctx.wordCount ? (n * 1000) / ctx.wordCount : 0);
  const clusterOnly = new Set<LexEntry>([...TIER2, ...TIER3, ...PHRASE3]);
  const exempt =
    mode === "academic"
      ? ranges(ACADEMIC_FORMULAS, p.prose)
      : mode === "technical"
        ? ranges(TECHNICAL_TERMS, p.prose)
        : [];
  for (const e of ALL_LEXICON) {
    if (clusterOnly.has(e) || !applies(e, mode)) continue;
    if (mode === "chat" && e.severity !== "P0" && e.type !== "calque") continue;
    e.re.lastIndex = 0;
    const hits = [...p.prose.matchAll(e.re)];
    if (!hits.length) continue;
    const threshold = e.density?.[mode] ?? e.density?.default;
    if (threshold !== undefined && per1000(hits.length) < threshold) continue;
    for (const m of hits) {
      const idx = m.index ?? 0;
      if (hitsRange(exempt, idx, m[0].length)) continue;
      add(ctx, e.type, e.id, e.severity, idx, m[0], e.hint, e.styleOnly);
    }
  }

  // Уровень 2: скопление двух и более разных слов в одном абзаце.
  if (mode !== "chat") {
    for (const b of bs) {
      if (b.kind !== "prose" && b.kind !== "list") continue;
      const found = new Map<string, { idx: number; text: string }>();
      for (const e of TIER2) {
        e.re.lastIndex = 0;
        const m = e.re.exec(b.text);
        if (m && !found.has(e.id)) {
          if (hitsRange(exempt, b.start + m.index, m[0].length)) continue;
          found.set(e.id, { idx: b.start + m.index, text: m[0] });
        }
      }
      if (found.size >= 2) {
        const list = [...found.values()];
        add(
          ctx,
          "tier2-cluster",
          "tier2",
          "P2",
          list[0]?.idx ?? b.start,
          list.map((x) => x.text).join(", "),
          "перефразировать проще, оставить одно слово или дать конкретику",
        );
      }
    }
  }

  // Уровень 3: суммарная плотность ≥ 3 % слов.
  let t3 = 0;
  let firstT3 = -1;
  const t3words: string[] = [];
  for (const e of TIER3) {
    e.re.lastIndex = 0;
    for (const m of p.prose.matchAll(e.re)) {
      t3 += 1;
      if (firstT3 < 0 || (m.index ?? 0) < firstT3) firstT3 = m.index ?? 0;
      if (t3words.length < 6) t3words.push(m[0]);
    }
  }
  if (ctx.wordCount >= 150 && t3 / ctx.wordCount >= 0.03) {
    add(
      ctx,
      "tier3",
      "density",
      "P2",
      Math.max(firstT3, 0),
      t3words.join(", "),
      `перегруженные оценочные слова: ${t3} на ${ctx.wordCount} слов — заменить часть числами и примерами`,
    );
  }

  // Фразовые штампы: одна фраза 2+ раза или 3+ разных.
  const phraseHits = new Map<string, { n: number; idx: number; text: string }>();
  for (const e of PHRASE3) {
    e.re.lastIndex = 0;
    for (const m of p.prose.matchAll(e.re)) {
      const cur = phraseHits.get(e.id) ?? { n: 0, idx: m.index ?? 0, text: m[0] };
      cur.n += 1;
      phraseHits.set(e.id, cur);
    }
  }
  for (const [id, h] of phraseHits) {
    if (h.n >= 2) add(ctx, "phrase3", id, "P2", h.idx, h.text, `штамп повторён ${h.n} раза — назвать конкретику`);
  }
  if (phraseHits.size >= 3) {
    const first = [...phraseHits.values()].sort((a, b) => a.idx - b.idx)[0];
    add(
      ctx,
      "phrase3-cluster",
      "cluster",
      "P1",
      first?.idx ?? 0,
      [...phraseHits.values()].map((h) => h.text).join(", "),
      "три и больше разных штампов — так варьирует шаблоны модель",
    );
  }
}

// ─── Технические отпечатки ──────────────────────────────────────────────
function detectFingerprints(ctx: Ctx): void {
  const { p } = ctx;
  if (p.invisible.length) {
    const first = p.invisible[0];
    const idx = first?.index ?? 0;
    const { line, column } = lineCol(p.lineStarts, idx);
    ctx.issues.push({
      type: "invisible-chars",
      rule: "zero-width",
      severity: "P0",
      text: plural(p.invisible.length, "невидимый символ", "невидимых символа", "невидимых символов"),
      index: idx,
      line,
      column,
      hint: "удалить; такие документы системы проверки помечают как подозрительные",
    });
  }
  if (p.softHyphens.length) {
    const idx = p.softHyphens[0] ?? 0;
    const { line, column } = lineCol(p.lineStarts, idx);
    ctx.issues.push({
      type: "soft-hyphen",
      rule: "soft-hyphen",
      severity: "P2",
      text: plural(p.softHyphens.length, "мягкий перенос", "мягких переноса", "мягких переносов"),
      index: idx,
      line,
      column,
      hint: "удалить перед сдачей; обычно их оставляют Word и копирование из PDF",
    });
  }
  for (const h of p.homoglyphs) {
    const { line, column } = lineCol(p.lineStarts, h.index);
    ctx.issues.push({
      type: "homoglyph",
      rule: "latin-in-cyrillic",
      severity: "P0",
      text: h.word,
      index: h.index,
      line,
      column,
      hint: "латинские буквы внутри русского слова; заменить на кириллицу",
    });
  }
  const scan = (type: string, re: RegExp, sev: Severity, hint: string, text = p.noCode): void => {
    for (const m of text.matchAll(re)) add(ctx, type, type, sev, m.index ?? 0, m[0], hint);
  };
  scan(
    "chat-markup",
    /\uE200[^\uE201\n]{0,200}\uE201|[\uE200-\uE204]|citeturn\d+\w*|(?<![\p{L}\d])turn\d+(?:search|news|file|image|view|fetch|video|product|academia)\d+|【\d+(?::\d+)?†[^】\n]{0,80}】|contentReference\[oaicite:\d+\](?:\{index=\d+\})?|oai_citation|\[attached_file:\d+\]|grok_card/gu,
    "P0",
    "удалить; если ссылка нужна — заменить настоящей",
  );
  scan(
    "ai-url",
    /[?&](?:utm_source=(?:chatgpt\.com|openai|copilot\.com|claude\.ai|perplexity\.ai|perplexity|gemini\.google\.com|deepseek\.com)|referrer=grok\.com)/gi,
    "P0",
    "удалить только этот параметр",
  );
  scan(
    "placeholder",
    /\[(?:Ваш[аеи]?|Вставьте|Укажите|Добавьте|Введите|Опишите|Название|Имя|Фамилия|Дата|ДАТА|Источник|Ссылка|Your|Insert|Add|Enter)[^\]\n]{0,60}\]|\b(?:19|20)XX\b|\b\d{4}-XX-XX\b|<!--\s*(?:добавь|добавьте|вставь|вставьте|todo|TODO|заполни|укажите|add|insert)[^>]*-->/g,
    "P0",
    "заполнить реальным содержимым или удалить предложение",
  );
}

// ─── Типографика ────────────────────────────────────────────────────────
const CYR = /\p{Script=Cyrillic}/u;

function detectTypography(ctx: Ctx, bs: Block[]): void {
  const { p, mode } = ctx;
  if (mode === "chat") return;
  // Кавычки считаем по тексту без кода (в prose они замаскированы как цитаты).
  for (const m of p.noCode.matchAll(/"([^"\n]{1,200})"/g)) {
    if (CYR.test(m[1] ?? ""))
      add(ctx, "straight-quotes", "straight", "P2", m.index ?? 0, m[0], "«ёлочки» вместо прямых кавычек");
  }
  for (const m of p.noCode.matchAll(/“([^”\n]{1,200})”/g)) {
    if (CYR.test(m[1] ?? ""))
      add(
        ctx,
        "english-quotes",
        "english",
        "P1",
        m.index ?? 0,
        m[0],
        "«ёлочки»; английские кавычки в русском тексте — след машинного перевода",
      );
  }
  for (const m of p.prose.matchAll(/(?<=\p{L}) - (?=\p{L})| -- /gu)) {
    add(ctx, "hyphen-dash", "hyphen", "P2", m.index ?? 0, m[0], "длинное тире с пробелами: « — »", true);
  }
  for (const m of p.prose.matchAll(
    /(?:(?:равн\p{L}*|составля\p{L}*|составил\p{L}*|достига\p{L}*|достиг\p{L}*|точност\p{L}*|значени\p{L}*|около|до|от|≈|=)\s*)(\d+\.\d+)(?!\.\d)|(?<![\d.])(\d+\.\d+)(?=\s*(?:%|мс|с\b|кг|м\b|км|Гц|ГБ|МБ|раз))/gu,
  )) {
    add(ctx, "decimal-point", "decimal", "P2", m.index ?? 0, m[0], "десятичная запятая: 0,93");
  }
  // Тире-связка: тире перед союзом или частицей, подающими «эффект».
  let splices = 0;
  const spliceHits: { idx: number; text: string }[] = [];
  for (const m of p.prose.matchAll(
    /[\p{L}\d)»,] — (?:и (?:это|все|всё|именно|тут|вот)|и\s+\p{L}+ (?:меня|нас|всех)|это (?:меняет|и есть|главное|ключ)|вот (?:что|почему|где|в чем)|именно (?:это|так|поэтому)|но (?:это|не|именно)|причем|а (?:это|значит|главное))/gu,
  )) {
    splices += 1;
    spliceHits.push({ idx: (m.index ?? 0) + 1, text: m[0] });
  }
  const limit = mode === "social" ? 2 : 1;
  const per500 = ctx.wordCount ? (splices * 500) / ctx.wordCount : 0;
  if (splices > limit || (splices >= 1 && per500 > 1 && mode !== "social")) {
    for (const h of spliceHits)
      add(
        ctx,
        "em-dash-splice",
        "splice",
        "P1",
        h.idx,
        h.text,
        "точка, двоеточие или союз вместо тире-связки; грамматическое тире не трогать",
      );
  }
  // Заголовки: Title Case, эмодзи.
  for (const b of bs) {
    if (b.kind !== "heading") continue;
    const body = b.text.replace(/^#+\s*/, "");
    const ws = words(body).filter((w) => CYR.test(w) && w.length >= 4);
    if (ws.length >= 3 && ws.every((w) => /^\p{Lu}\p{Ll}/u.test(w))) {
      add(ctx, "title-case", "title-case", "P1", b.start, body, "заглавная только у первого слова и имён собственных");
    }
    if (/\p{Extended_Pictographic}/u.test(body)) {
      add(ctx, "emoji-header", "emoji", mode === "social" ? "P2" : "P1", b.start, body, "убрать эмодзи из заголовка");
    }
  }
  // Жирный: больше двух выделений в разделе прозы.
  let sectionStart = 0;
  let bold: { idx: number; text: string }[] = [];
  const flushBold = (): void => {
    if (bold.length > 3 && mode !== "technical") {
      add(
        ctx,
        "bold-overuse",
        "bold",
        "P1",
        bold[0]?.idx ?? sectionStart,
        bold
          .map((b) => b.text)
          .slice(0, 4)
          .join(" "),
        "не больше одного выделения на раздел",
      );
    }
    bold = [];
  };
  for (const b of bs) {
    if (b.kind === "heading") {
      flushBold();
      sectionStart = b.start;
      continue;
    }
    if (b.kind !== "prose") continue;
    for (const m of b.text.matchAll(/\*\*[^*\n]{1,80}\*\*/g)) {
      const at = m.index ?? 0;
      const lineStart = b.text.lastIndexOf("\n", at - 1) + 1;
      const lead = /^\s*(?:[-*+]\s+|\d+[.)]\s+|[\p{Lu}\d]{1,3}\.\s+)?$/u.test(b.text.slice(lineStart, at));
      const after = b.text.slice(at + m[0].length, at + m[0].length + 4);
      // Выделенный термин в начале строки перед «—», «:» или ссылкой — типографика, а не акцент.
      if (lead && /^\s*(?:—|:|\[|$)/.test(after)) continue;
      bold.push({ idx: b.start + at, text: m[0] });
    }
  }
  flushBold();
}

// ─── Структура ──────────────────────────────────────────────────────────
// Грубый признак глагольной формы; существительные на -ость исключены отдельно.
const VERB_HINT =
  /(?:ет|ют|ит|ят|ем|им|ешь|ишь|ал|ял|ил|ыл|ул|ла|ли|ло|ать|ять|ить|еть|уть|ыть|оть|ться|ется|ются|ится|ятся|ся|сь|ут|ат)$/u;
const NOT_VERB = /(?:ость|есть|асть)$/u;
const GENITIVE_NOUN =
  /(?:ени[яюейи]|ани[яюейи]|овани[яюейи]|ции|цию|ция|ости|ость|ств[ауео]|изаци[ияю]|ировани[яюе]|ени|ани)$/u;
// Начало слова задаёт ретроспектива (?<!\p{L}): без неё на длинном слове без пробелов
// движок пробует каждую позицию и откатывается к ней, время растёт кубически.
const PASSIVE =
  /(?:^|\s)(?:был[аио]?|были|будет|будут)\s+\p{L}+(?:ан|ян|ен|ён|т)[аоы]?(?=[\s,.;:!?]|$)|(?<!\p{L})\p{L}{3,}(?:ано|ено|ены|аны|ана|ена|ято|ыто|иты|ата)(?=[\s,.;:!?]|$)|(?<!\p{L})\p{L}{3,}(?:ется|ются|ится|ятся)(?=[\s,.;:!?]|$)/iu;
const ACTOR = /(?:^|\s)(?:мы|я|нами|автор\p{L}*|авторы|нами)(?=[\s,.;:!?]|$)/iu;

function detectStructure(ctx: Ctx, bs: Block[], proseSentences: Sentence[][]): void {
  const { mode } = ctx;

  // «Не X, а Y», «это не X — это Y», «не просто X, а Y», разнесённая форма.
  const nxy =
    /(?<![\p{L}])(?:(?:это|речь|дело|вопрос|суть|главное)\s+не\s+(?:просто\s+|только\s+|в\s+|о\s+|про\s+)?[^.!?\n]{1,50}?(?:,|\s—)\s*(?:а|это|но)\s)|(?<![\p{L}])не\s+просто\s+[^.!?\n]{1,50}?,\s*(?:а|но и|это)\s/giu;
  for (const m of ctx.p.prose.matchAll(nxy))
    add(ctx, "not-x-but-y", "joined", "P1", m.index ?? 0, m[0], "прямое утверждение без противопоставления");
  const split =
    /(?<![\p{L}])(?:главное|дело|суть|проблема|секрет)\s+(?:здесь\s+|тут\s+)?не\s+в\s+[^.!?\n]{1,40}\.\s+(?:главное|дело|суть|настоящ\p{L}*|все\s+дело|всё\s+дело)\s/giu;
  for (const m of ctx.p.prose.matchAll(split))
    add(ctx, "not-x-but-y", "split", "P1", m.index ?? 0, m[0], "разнесённое «не X. Y» — сказать Y прямо");

  // Цепочки отглагольных существительных: 4+ подряд.
  if (mode !== "chat") {
    const minChain = mode === "academic" ? 5 : 4;
    for (const b of bs) {
      if (b.kind !== "prose" && b.kind !== "list") continue;
      const toks = [...b.text.matchAll(/[\p{L}]+/gu)];
      let run: RegExpMatchArray[] = [];
      const flush = (): void => {
        if (run.length >= minChain) {
          const s = run[0]?.index ?? 0;
          const last = run[run.length - 1];
          const e = (last?.index ?? 0) + (last?.[0].length ?? 0);
          add(
            ctx,
            "genitive-chain",
            "chain",
            "P1",
            b.start + s,
            b.text.slice(s, e),
            "переписать через глагол: «обеспечение повышения эффективности…» → «чтобы точнее…»",
          );
        }
        run = [];
      };
      for (let i = 0; i < toks.length; i += 1) {
        const t = toks[i] as RegExpMatchArray;
        const prev = toks[i - 1];
        const gap = prev ? b.text.slice((prev.index ?? 0) + prev[0].length, t.index ?? 0) : " ";
        const w = t[0].toLowerCase();
        const nounish = GENITIVE_NOUN.test(w) && w.length > 5;
        if (nounish && (run.length === 0 || /^\s+$/.test(gap))) run.push(t);
        else {
          flush();
          if (nounish) run.push(t);
        }
      }
      flush();
    }
  }

  // Пассив без деятеля: 4+ предложения подряд (6+ в научном режиме).
  if (mode !== "chat" && mode !== "technical") {
    const need = mode === "academic" ? 6 : 4;
    for (const ss of proseSentences) {
      let run: Sentence[] = [];
      const flush = (): void => {
        if (run.length >= need)
          add(
            ctx,
            "passive-run",
            "passive",
            "P2",
            run[0]?.start ?? 0,
            run
              .map((s) => s.text)
              .join(" ")
              .slice(0, 160),
            "назвать деятеля, если источник его называет, или перемешать с активными конструкциями",
          );
        run = [];
      };
      for (const s of ss) {
        if (PASSIVE.test(s.text) && !ACTOR.test(s.text)) run.push(s);
        else flush();
      }
      flush();
    }
  }

  // Переходы в начале абзацев подряд.
  if (mode !== "chat" && mode !== "social") {
    const opener =
      /^\s*(?:кроме того|помимо этого|более того|к тому же|также|таким образом|в свою очередь|однако|при этом|вместе с тем|следовательно|в то же время|в целом)[,\s]/iu;
    let run: Block[] = [];
    const flush = (): void => {
      if (run.length >= 3)
        add(
          ctx,
          "transition-run",
          "openers",
          "P1",
          run[0]?.start ?? 0,
          run.map((b) => b.text.trim().split(/\s+/).slice(0, 2).join(" ")).join(" / "),
          "связь должна быть видна из содержания, а не из союза в начале каждого абзаца",
        );
      run = [];
    };
    for (const b of bs) {
      if (b.kind !== "prose") continue;
      if (opener.test(b.text)) run.push(b);
      else flush();
    }
    flush();
  }

  for (const ss of proseSentences) {
    // Одинаковые начала: 3+ предложения подряд с одним первым словом.
    for (let i = 0; i + 2 < ss.length; i += 1) {
      const first = (s: Sentence | undefined): string => (s ? (words(s.text)[0] ?? "").toLowerCase() : "");
      const a = first(ss[i]);
      if (
        a &&
        a.length > 1 &&
        !["он", "она", "они", "я", "мы", "оно"].includes(a) &&
        a === first(ss[i + 1]) &&
        a === first(ss[i + 2])
      ) {
        add(
          ctx,
          "same-opener",
          a,
          "P2",
          ss[i]?.start ?? 0,
          `${ss[i]?.text ?? ""} ${ss[i + 1]?.text ?? ""}`,
          "оставить первое, остальные перестроить",
        );
        i += 2;
      }
    }
    // Рубленые фрагменты: 3+ подряд по 1–4 слова.
    if (mode !== "social") {
      let run: Sentence[] = [];
      const flush = (): void => {
        if (run.length >= 3)
          add(
            ctx,
            "staccato",
            "fragments",
            "P2",
            run[0]?.start ?? 0,
            run.map((s) => s.text).join(" "),
            "оставить один акцентный фрагмент, остальное собрать в предложения",
          );
        run = [];
      };
      for (const s of ss) {
        if (s.words <= 4 && !/[?:]$/.test(s.text)) run.push(s);
        else flush();
      }
      flush();
    }
    // Стопка вопросов: 3+ вопросительных предложения подряд.
    let q: Sentence[] = [];
    const flushQ = (): void => {
      if (q.length >= 3)
        add(
          ctx,
          "stacked-questions",
          "questions",
          "P2",
          q[0]?.start ?? 0,
          q.map((s) => s.text).join(" "),
          "оставить не больше одного вопроса и ответить",
        );
      q = [];
    };
    for (const s of ss) {
      if (s.text.endsWith("?")) q.push(s);
      else flushQ();
    }
    flushQ();
  }

  // Цепочка отрицаний: «Без X, без Y, без Z.» / «Никаких X. Никаких Y.»
  for (const m of ctx.p.prose.matchAll(
    /(?<![\p{L}])(?:без\s+[^,.!?\n]{1,30},\s*){2,}без\s+[^,.!?\n]{1,30}[.!]|(?:(?<![\p{L}])никак\p{L}+\s+[^.!?\n]{1,30}[.!]\s*){3,}/giu,
  )) {
    add(
      ctx,
      "negation-chain",
      "negations",
      "P2",
      m.index ?? 0,
      m[0],
      "сказать, что это есть, а не чем оно не является",
    );
  }

  // Списки из голых именных групп: 5+ коротких пунктов без глаголов.
  if (mode !== "chat" && mode !== "technical") {
    for (const b of bs) {
      if (b.kind !== "list") continue;
      const items = b.text
        .split("\n")
        .map((l) =>
          l
            .replace(/^\s*(?:[-*+•]|\d+[.)])\s+/, "")
            .replace(/\*\*/g, "")
            .trim(),
        )
        .filter(Boolean);
      if (items.length < 5) continue;
      const bare = items.filter((it) => {
        const ws = words(it);
        return (
          ws.length > 0 &&
          ws.length <= 6 &&
          !ws.some((w) => w.length > 3 && VERB_HINT.test(w.toLowerCase()) && !NOT_VERB.test(w.toLowerCase()))
        );
      });
      if (bare.length >= 5 && bare.length / items.length >= 0.8) {
        add(
          ctx,
          "bullet-np-list",
          "np-list",
          "P1",
          b.start,
          items.slice(0, 3).join(" / "),
          "полные утверждения с данными из источника или проза",
        );
      }
    }
  }

  // Хэштеги: 6+ (без номеров задач и цветов).
  const tags = [...ctx.p.prose.matchAll(/(?<![\p{L}\d&/])#(?=[\p{L}_]*\p{L})[\p{L}\d_]{2,}/gu)].filter(
    (m) => !/^#[0-9a-f]{6}$|^#[0-9a-f]{3}$/i.test(m[0]) || !/\d/.test(m[0]),
  );
  if (tags.length >= 6)
    add(
      ctx,
      "hashtag-stuffing",
      "hashtags",
      mode === "social" ? "P1" : "P0",
      tags[0]?.index ?? 0,
      tags
        .slice(0, 8)
        .map((m) => m[0])
        .join(" "),
      "два-три конкретных тега или ни одного",
    );
}

// ─── Стилометрия ────────────────────────────────────────────────────────
function detectStylometry(
  ctx: Ctx,
  bs: Block[],
  proseSentences: Sentence[][],
): { sentCV: number; paraCV: number; meanLen: number; mattr: number } {
  const all = proseSentences.flat().filter((s) => s.words >= 3);
  const lengths = all.map((s) => s.words);
  const sentCV = cv(lengths);
  const meanLen = mean(lengths);
  const paras = bs.filter((b) => b.kind === "prose" && words(b.text).length >= 15);
  const paraLens = paras.map((b) => words(b.text).length);
  const paraCV = cv(paraLens);
  const tokens = words(ctx.p.prose);
  const diversity = mattr(tokens, 100);

  if (ctx.mode !== "chat" && ctx.mode !== "social") {
    if (lengths.length >= 10 && sentCV < 0.33 && meanLen >= 10) {
      add(
        ctx,
        "uniform-sentences",
        "sentence-cv",
        "P1",
        all[0]?.start ?? 0,
        `коэффициент вариации длины предложений ${sentCV.toFixed(2)} при средней длине ${meanLen.toFixed(1)} слова`,
        "текст метрономичен: смешать короткие и длинные предложения там, где позволяет содержание",
      );
    }
    if (paraLens.length >= 5 && paraCV < 0.2) {
      add(
        ctx,
        "uniform-paragraphs",
        "paragraph-cv",
        "P2",
        paras[0]?.start ?? 0,
        `коэффициент вариации длины абзацев ${paraCV.toFixed(2)} на ${paraLens.length} абзацах`,
        "границы абзацев по смыслу, а не по размеру",
      );
    }
    if (tokens.length >= 300 && diversity < 0.62 && ctx.mode !== "technical") {
      add(
        ctx,
        "low-diversity",
        "mattr",
        "P2",
        0,
        `MATTR ${diversity.toFixed(2)}`,
        "словарь беден для русской прозы: больше конкретики вместо повторяющихся абстракций",
      );
    }
  }
  return { sentCV, paraCV, meanLen, mattr: diversity };
}

/** Находки, которые описывают весь текст или абзац, а не конкретную фразу. */
const AGGREGATE = new Set([
  "uniform-sentences",
  "uniform-paragraphs",
  "low-diversity",
  "transition-run",
  "passive-run",
  "tier2-cluster",
  "phrase3-cluster",
  "tier3",
  "invisible-chars",
  "soft-hyphen",
]);

/**
 * Убирает повторы и вложенные совпадения: «играет ключевую роль» и
 * «ключевую роль» — одна находка, остаётся более весомая или более длинная.
 */
function dedupe(issues: Issue[]): Issue[] {
  const seen = new Set<string>();
  const sorted = [...issues].sort((a, b) => a.index - b.index || b.text.length - a.text.length);
  const kept: Issue[] = [];
  for (const i of sorted) {
    const key = `${i.type}\u0000${i.rule}\u0000${i.index}`;
    if (seen.has(key)) continue;
    seen.add(key);
    if (!AGGREGATE.has(i.type)) {
      const end = i.index + i.text.length;
      const w = WEIGHTS[i.type] ?? 1;
      const covering = kept.find(
        (k) =>
          !AGGREGATE.has(k.type) && k.index <= i.index && k.index + k.text.length >= end && (WEIGHTS[k.type] ?? 1) >= w,
      );
      if (covering) continue;
    }
    kept.push(i);
  }
  return kept;
}

export function scoreIssues(issues: Issue[], wordCount: number): number {
  const distinct = new Map<string, number>();
  for (const i of issues) distinct.set(`${i.type}\u0000${i.text.toLowerCase()}`, WEIGHTS[i.type] ?? 1);
  const raw = [...distinct.values()].reduce((a, b) => a + b, 0);
  const scale = 8 * Math.max(1, Math.log2(Math.max(wordCount, 1) / 50));
  return Math.round(100 * (1 - Math.exp(-raw / scale)));
}

export function labelFor(score: number): string {
  if (score < 15) return "чисто";
  if (score < 40) return "есть приметы";
  if (score < 70) return "много примет";
  return "сильный ИИ-стиль";
}

export interface Internals {
  prepared: Prepared;
  blocks: Block[];
  sentences: Sentence[][];
}

export function analyzeInternal(
  source: string,
  options: AnalyzeOptions = {},
): { result: AnalysisResult; internals: Internals } {
  const mode = options.context ?? "general";
  const p = prepare(source);
  const bs = blocks(p);
  const proseSentences = bs.filter((b) => b.kind === "prose").map((b) => sentences(b.text, b.start));
  const wordCount = words(p.prose).length;
  const ctx: Ctx = { p, mode, wordCount, issues: [] };

  detectFingerprints(ctx);
  detectLexicon(ctx, bs);
  detectTypography(ctx, bs);
  detectStructure(ctx, bs, proseSentences);
  const st = detectStylometry(ctx, bs, proseSentences);

  const issues = dedupe(ctx.issues);
  const score = scoreIssues(issues, wordCount);
  const result: AnalysisResult = {
    score,
    label: labelFor(score),
    issues,
    suspicious: p.invisible.length > 0 || p.homoglyphs.length > 0,
    stats: {
      words: wordCount,
      sentences: proseSentences.flat().length,
      paragraphs: bs.filter((b) => b.kind === "prose").length,
      meanSentenceLength: Number(st.meanLen.toFixed(2)),
      sentenceLengthCV: Number(st.sentCV.toFixed(3)),
      paragraphLengthCV: Number(st.paraCV.toFixed(3)),
      mattr: Number(st.mattr.toFixed(3)),
      contextMode: mode,
    },
  };
  return { result, internals: { prepared: p, blocks: bs, sentences: proseSentences } };
}

export function analyze(source: string, options: AnalyzeOptions = {}): AnalysisResult {
  return analyzeInternal(source, options).result;
}

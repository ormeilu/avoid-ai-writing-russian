/** Режим анализа. Соответствие профилям скилла — в references/patterns.md. */
export type ContextMode = "general" | "academic" | "technical" | "social" | "chat";

export const CONTEXT_MODES: readonly ContextMode[] = ["general", "academic", "technical", "social", "chat"];

/** Профили скилла (`--context` в SKILL.md) и их режимы детектора. */
export const PROFILE_TO_MODE: Readonly<Record<string, ContextMode>> = {
  vak: "academic",
  docs: "technical",
  blog: "general",
  telegram: "social",
  "business-email": "general",
  chat: "chat",
};

export type Severity = "P0" | "P1" | "P2";

export interface Issue {
  /** Тип находки, стабильный идентификатор категории. */
  type: string;
  /** Идентификатор конкретного правила внутри категории (для словарей). */
  rule: string;
  severity: Severity;
  /** Найденный фрагмент исходного текста. */
  text: string;
  /** Смещение в исходном тексте (UTF-16 code units). */
  index: number;
  line: number;
  column: number;
  /** Что сделать. */
  hint: string;
  /**
   * Правки ради краткости (канцелярит 1Б) — совет по стилю,
   * а не довод о машинном авторстве. В оценку идут с малым весом.
   */
  styleOnly?: boolean;
}

export interface Stats {
  words: number;
  sentences: number;
  paragraphs: number;
  meanSentenceLength: number;
  sentenceLengthCV: number;
  paragraphLengthCV: number;
  mattr: number;
  contextMode: ContextMode;
}

export interface AnalysisResult {
  /** 0–100, чем выше, тем больше примет ИИ-стиля. */
  score: number;
  label: string;
  issues: Issue[];
  stats: Stats;
  /** Невидимые символы и подмена букв: документ выглядит подозрительным. */
  suspicious: boolean;
}

export interface AnalyzeOptions {
  context?: ContextMode;
}

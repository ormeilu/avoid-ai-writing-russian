export {
  type AntiplagiatReport,
  antiplagiat,
  balancedAccuracy,
  type CalibrationFile,
  DEFAULT_MODEL,
  FEATURE_NAMES,
  type Fragment,
  fit,
  labelFragments,
  type Model,
  predict,
  type Sample,
} from "./antiplagiat.ts";
export { JUDGMENT_ONLY, TYPE_TO_SECTION } from "./categories.ts";
export { analyze, labelFor, scoreIssues, TYPE_LABELS, WEIGHTS } from "./detect.ts";
export { ALL_LEXICON, type LexEntry, phrase } from "./lexicon.ts";
export * from "./types.ts";
export { type ValidationResult, type Violation, validate } from "./validate.ts";

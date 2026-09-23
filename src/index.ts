export { analyze, labelFor, scoreIssues, TYPE_LABELS, WEIGHTS } from "./detect.ts";
export {
  antiplagiat,
  balancedAccuracy,
  DEFAULT_MODEL,
  FEATURE_NAMES,
  fit,
  labelFragments,
  predict,
  type AntiplagiatReport,
  type CalibrationFile,
  type Fragment,
  type Model,
  type Sample,
} from "./antiplagiat.ts";
export { validate, type ValidationResult, type Violation } from "./validate.ts";
export { ALL_LEXICON, phrase, type LexEntry } from "./lexicon.ts";
export * from "./types.ts";

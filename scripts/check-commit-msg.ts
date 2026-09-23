#!/usr/bin/env bun
/**
 * Хук commit-msg: сообщения коммитов в проекте пишутся по-русски.
 * Первая строка должна содержать кириллицу, быть не длиннее 72 знаков
 * и не заканчиваться точкой. Коммиты слияния и fixup!/squash! пропускаются.
 */

import { readFileSync } from "node:fs";

export function checkCommitMessage(message: string): string[] {
  const lines = message.split("\n").filter((l) => !l.startsWith("#"));
  const subject = (lines[0] ?? "").trim();
  if (/^(Merge|Revert|fixup!|squash!|amend!)/.test(subject)) return [];
  const errors: string[] = [];
  if (!subject) errors.push("пустая первая строка");
  if (!/\p{Script=Cyrillic}/u.test(subject)) errors.push("первая строка должна быть на русском");
  if ([...subject].length > 72) errors.push(`первая строка длиннее 72 знаков (${[...subject].length})`);
  if (/[.。]$/.test(subject)) errors.push("первая строка не должна заканчиваться точкой");
  if (lines.length > 1 && (lines[1] ?? "").trim() !== "") errors.push("после первой строки нужна пустая строка");
  return errors;
}

if (import.meta.main) {
  const file = process.argv[2];
  if (!file) {
    console.error("использование: check-commit-msg.ts ФАЙЛ_СООБЩЕНИЯ");
    process.exit(2);
  }
  const errors = checkCommitMessage(readFileSync(file, "utf8"));
  if (errors.length) {
    for (const e of errors) console.error(`commit-msg: ${e}`);
    process.exit(1);
  }
}

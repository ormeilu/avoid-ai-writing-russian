#!/usr/bin/env bun
/**
 * Хук pre-commit: в файлах нет невидимых символов и латинских букв
 * внутри русских слов. Проект сам ловит такие артефакты в чужих текстах
 * и не должен их содержать.
 */

import { readFileSync } from "node:fs";
import { prepare } from "../src/text.ts";

let bad = 0;
for (const file of process.argv.slice(2)) {
  const p = prepare(readFileSync(file, "utf8"));
  for (const i of p.invisible) {
    console.error(
      `${file}: невидимый символ U+${i.char.codePointAt(0)?.toString(16).toUpperCase().padStart(4, "0")} в позиции ${i.index}`,
    );
    bad += 1;
  }
  for (const h of p.homoglyphs) {
    console.error(`${file}: латиница внутри русского слова «${h.word}» в позиции ${h.index}`);
    bad += 1;
  }
}
process.exit(bad ? 1 : 0);

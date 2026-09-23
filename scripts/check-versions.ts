#!/usr/bin/env bun
/** Проверяет, что версия одинакова во всех местах, где она записана. */

import { readVersions } from "./versions.ts";

const found = readVersions();
const distinct = new Set(found.map((f) => f.version));
if (distinct.size !== 1 || found.some((f) => !f.version)) {
  console.error("Версии расходятся:");
  for (const f of found) console.error(`  ${f.file}: ${f.version ?? "не найдена"}`);
  console.error("Исправьте вручную или выпустите версию через `bun run release`.");
  process.exit(1);
}

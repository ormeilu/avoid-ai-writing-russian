#!/usr/bin/env bun
/** Печатает раздел CHANGELOG.md для версии: `bun scripts/release-notes.ts 0.2.0`. */

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { section } from "./changelog.ts";
import { ROOT } from "./versions.ts";

const version = (process.argv[2] ?? "").replace(/^v/, "");
const body = section(readFileSync(join(ROOT, "CHANGELOG.md"), "utf8"), version);
if (!body) {
  console.error(`В CHANGELOG.md нет раздела для версии ${version}`);
  process.exit(1);
}
process.stdout.write(`${body}\n`);

#!/usr/bin/env bun
/**
 * Выпуск версии: `bun run release 0.2.0` (или patch | minor | major).
 *
 * 1. Проверяет, что рабочее дерево чистое, ветка master, версия растёт.
 * 2. Прогоняет проверки: типы, тесты, синхронность версий.
 * 3. Переносит «Не выпущено» из CHANGELOG.md в раздел новой версии.
 * 4. Обновляет версию во всех местах из scripts/versions.ts.
 * 5. Делает коммит «Выпуск X.Y.Z» и аннотированный тег vX.Y.Z.
 *
 * Ничего не отправляет: публикация — `git push --follow-tags`, после чего
 * GitHub Actions собирает бинарники и создаёт выпуск (release.yml).
 * `--dry-run` показывает план без изменений.
 */

import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { $ } from "bun";
import { cut } from "./changelog.ts";
import { compareSemver, ROOT, readVersions, SEMVER, writeVersions } from "./versions.ts";

const args = process.argv.slice(2);
const dry = args.includes("--dry-run");
const arg = args.find((a) => !a.startsWith("--"));

function fail(msg: string): never {
  console.error(`release: ${msg}`);
  process.exit(1);
}

const current = readVersions()[0]?.version ?? fail("не удалось прочитать текущую версию");
if (new Set(readVersions().map((v) => v.version)).size !== 1)
  fail("версии расходятся, сначала `bun scripts/check-versions.ts`");

function bump(v: string, kind: string): string {
  const [ma, mi, pa] = v.split(/[.-]/).map(Number) as [number, number, number];
  if (kind === "major") return `${ma + 1}.0.0`;
  if (kind === "minor") return `${ma}.${mi + 1}.0`;
  if (kind === "patch") return `${ma}.${mi}.${pa + 1}`;
  return kind;
}

if (!arg) fail("укажите версию: X.Y.Z, patch, minor или major");
const next = bump(current, arg);
if (!SEMVER.test(next)) fail(`не семантическая версия: ${next}`);
if (compareSemver(next, current) <= 0) fail(`версия ${next} не больше текущей ${current}`);

$.cwd(ROOT);
const branch = (await $`git branch --show-current`.text()).trim();
if (branch !== "master") fail(`выпуск делается из master, сейчас ${branch}`);
if ((await $`git status --porcelain`.text()).trim() && !dry) fail("рабочее дерево не чистое");
if ((await $`git tag --list v${next}`.text()).trim()) fail(`тег v${next} уже есть`);

const date = new Date().toISOString().slice(0, 10);
const changelogPath = join(ROOT, "CHANGELOG.md");
let changelog: string;
try {
  changelog = cut(readFileSync(changelogPath, "utf8"), next, date);
} catch (e) {
  fail((e as Error).message);
}

console.log(`Выпуск ${current} → ${next} (${date})`);
if (dry) {
  console.log("--dry-run: изменения не записаны");
  process.exit(0);
}

await $`bun run typecheck`;
await $`bun test`;
writeFileSync(changelogPath, changelog);
writeVersions(next);
await $`bun scripts/check-versions.ts`;
await $`git add -A`;
await $`git commit -m ${`Выпуск ${next}`}`;
await $`git tag -a v${next} -m ${`Выпуск ${next}`}`;
console.log(`Готово. Отправить: git push --follow-tags`);

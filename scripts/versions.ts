/**
 * Версия проекта записана в нескольких местах (SITES). Этот модуль знает их все:
 * читает, сверяет и переписывает.
 */

import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

export const ROOT = join(import.meta.dir, "..");

export interface VersionSite {
  file: string;
  read(text: string): string | undefined;
  write(text: string, version: string): string;
}

const jsonVersion = (file: string): VersionSite => ({
  file,
  read: (t) => (JSON.parse(t) as { version?: string }).version,
  write: (t, v) => t.replace(/("version":\s*")[^"]+(")/, `$1${v}$2`),
});

const cffVersion = (file: string): VersionSite => ({
  file,
  read: (t) => /^version:\s*(\S+)\s*$/m.exec(t)?.[1],
  write: (t, v) => t.replace(/^(version:\s*)\S+(\s*)$/m, `$1${v}$2`),
});

const frontmatterVersion = (file: string): VersionSite => ({
  file,
  read: (t) => /^---\n[\s\S]*?^version:\s*(\S+)\s*$/m.exec(t)?.[1],
  write: (t, v) => t.replace(/^(version:\s*)\S+(\s*)$/m, `$1${v}$2`),
});

export const SITES: VersionSite[] = [
  jsonVersion("package.json"),
  jsonVersion(".claude-plugin/plugin.json"),
  jsonVersion(".codex-plugin/plugin.json"),
  {
    file: ".claude-plugin/marketplace.json",
    read: (t) => {
      const d = JSON.parse(t) as { metadata?: { version?: string }; plugins?: { version?: string }[] };
      const v = d.metadata?.version;
      return d.plugins?.every((p) => p.version === v) ? v : undefined;
    },
    write: (t, v) => t.replace(/("version":\s*")[^"]+(")/g, `$1${v}$2`),
  },
  frontmatterVersion("skills/avoid-ai-writing-russian/SKILL.md"),
  frontmatterVersion("skills/antiplagiat/SKILL.md"),
  cffVersion("CITATION.cff"),
];

export function readVersions(root = ROOT): { file: string; version: string | undefined }[] {
  return SITES.map((s) => ({ file: s.file, version: s.read(readFileSync(join(root, s.file), "utf8")) }));
}

export function writeVersions(version: string, root = ROOT): void {
  for (const s of SITES) {
    const path = join(root, s.file);
    writeFileSync(path, s.write(readFileSync(path, "utf8"), version));
  }
}

export const SEMVER = /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?$/;

export function compareSemver(a: string, b: string): number {
  const pa = a.split(/[.-]/).slice(0, 3).map(Number);
  const pb = b.split(/[.-]/).slice(0, 3).map(Number);
  for (let i = 0; i < 3; i += 1) {
    const d = (pa[i] ?? 0) - (pb[i] ?? 0);
    if (d !== 0) return Math.sign(d);
  }
  return 0;
}

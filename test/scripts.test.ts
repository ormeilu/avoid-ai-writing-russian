import { describe, expect, test } from "bun:test";
import { cpSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { cut, section, UNRELEASED } from "../scripts/changelog.ts";
import { checkCommitMessage } from "../scripts/check-commit-msg.ts";
import { compareSemver, ROOT, readVersions, SEMVER, SITES, writeVersions } from "../scripts/versions.ts";

describe("версии", () => {
  test("во всех местах одна версия", () => {
    const found = readVersions();
    expect(found).toHaveLength(SITES.length);
    expect(new Set(found.map((f) => f.version)).size).toBe(1);
    expect(found[0]?.version).toMatch(SEMVER);
  });

  test("writeVersions меняет все места и только версию", () => {
    const dir = mkdtempSync(join(tmpdir(), "aiw-ru-"));
    for (const s of SITES) cpSync(join(ROOT, s.file), join(dir, s.file), { recursive: true });
    const before = SITES.map((s) => readFileSync(join(dir, s.file), "utf8"));
    writeVersions("9.8.7", dir);
    expect(new Set(readVersions(dir).map((v) => v.version))).toEqual(new Set(["9.8.7"]));
    SITES.forEach((s, i) => {
      const after = readFileSync(join(dir, s.file), "utf8");
      const changed = after.split("\n").filter((l, k) => l !== before[i]?.split("\n")[k]);
      expect(changed.length, s.file).toBeGreaterThan(0);
      for (const line of changed) expect(line, s.file).toMatch(/version/);
    });
  });

  test("сравнение семантических версий", () => {
    expect(compareSemver("0.2.0", "0.1.9")).toBe(1);
    expect(compareSemver("1.0.0", "1.0.0")).toBe(0);
    expect(compareSemver("0.10.0", "0.9.0")).toBe(1);
    expect(SEMVER.test("1.2")).toBe(false);
    expect(SEMVER.test("1.2.3-rc.1")).toBe(true);
  });
});

describe("CHANGELOG", () => {
  const log = `# История\n\n${UNRELEASED}\n\n### Добавлено\n\n- новое\n\n## [0.1.0] — 2026-01-01\n\n- старое\n`;

  test("section достаёт разделы", () => {
    expect(section(log, "unreleased")).toBe("### Добавлено\n\n- новое");
    expect(section(log, "0.1.0")).toBe("- старое");
    expect(section(log, "9.9.9")).toBeUndefined();
  });

  test("cut переносит «Не выпущено» в новую версию", () => {
    const out = cut(log, "0.2.0", "2026-10-01");
    expect(section(out, "unreleased")).toBe("");
    expect(section(out, "0.2.0")).toBe("### Добавлено\n\n- новое");
    expect(section(out, "0.1.0")).toBe("- старое");
  });

  test("cut отказывается выпускать пустой раздел", () => {
    expect(() => cut(`${UNRELEASED}\n\n## [0.1.0]\n`, "0.2.0", "2026-10-01")).toThrow();
  });

  test("в проекте есть раздел текущей версии", () => {
    const changelog = readFileSync(join(ROOT, "CHANGELOG.md"), "utf8");
    const version = readVersions()[0]?.version ?? "";
    expect(section(changelog, version)).toBeTruthy();
    expect(changelog).toContain(UNRELEASED);
  });
});

describe("сообщение коммита", () => {
  test("русское сообщение проходит", () => {
    expect(checkCommitMessage("детектор: быстрее в научном режиме\n\nПодробности.\n")).toEqual([]);
  });

  test("английское, длинное, с точкой и без пустой строки — нет", () => {
    expect(checkCommitMessage("fix bug")).toContain("первая строка должна быть на русском");
    expect(checkCommitMessage(`${"очень ".repeat(15)}длинно`).join()).toContain("длиннее 72");
    expect(checkCommitMessage("исправлено.")).toContain("первая строка не должна заканчиваться точкой");
    expect(checkCommitMessage("строка\nсразу тело")).toContain("после первой строки нужна пустая строка");
  });

  test("слияния, fixup и комментарии git пропускаются", () => {
    expect(checkCommitMessage("Merge branch 'x'")).toEqual([]);
    expect(checkCommitMessage("fixup! что-то")).toEqual([]);
    expect(checkCommitMessage("# комментарий\nправка тестов\n")).toEqual([]);
  });
});

describe("проверка невидимых символов", () => {
  const script = join(ROOT, "scripts/check-invisible.ts");
  const run = (content: string): number => {
    const f = join(mkdtempSync(join(tmpdir(), "aiw-ru-")), "x.md");
    writeFileSync(f, content);
    return Bun.spawnSync(["bun", script, f]).exitCode ?? -1;
  };

  test("чистый файл проходит", () => expect(run("Обычный текст.")).toBe(0));
  test("невидимый символ ловится", () => expect(run(`Текст${String.fromCharCode(0x200b)}.`)).toBe(1));
  test("латиница в русском слове ловится", () => expect(run(`р${String.fromCharCode(0x61)}бота`)).toBe(1));
});

describe("release", () => {
  test("--dry-run не меняет файлы", () => {
    const before = readFileSync(join(ROOT, "package.json"), "utf8");
    const r = Bun.spawnSync(["bun", join(ROOT, "scripts/release.ts"), "patch", "--dry-run"], { cwd: ROOT });
    expect(readFileSync(join(ROOT, "package.json"), "utf8")).toBe(before);
    // В ветке, отличной от master, скрипт отказывает с кодом 1 — это тоже корректно.
    expect([0, 1]).toContain(r.exitCode ?? -1);
  });

  test("отказывает для версии не больше текущей", () => {
    const r = Bun.spawnSync(["bun", join(ROOT, "scripts/release.ts"), "0.0.1", "--dry-run"], { cwd: ROOT });
    expect(r.exitCode).toBe(1);
    expect(r.stderr.toString()).toContain("не больше текущей");
  });
});

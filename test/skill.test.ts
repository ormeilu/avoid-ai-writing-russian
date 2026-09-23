/**
 * Тесты самих скиллов: оформление SKILL.md, ссылки, согласованность
 * каталога с детектором и примеры из каталога.
 */

import { describe, expect, test } from "bun:test";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { analyze, JUDGMENT_ONLY, PROFILE_TO_MODE, TYPE_LABELS, TYPE_TO_SECTION, WEIGHTS } from "../src/index.ts";

const ROOT = join(import.meta.dir, "..");
const SKILLS = join(ROOT, "skills");
const CATALOG = join(SKILLS, "avoid-ai-writing-russian/references/patterns.md");
const read = (p: string): string => readFileSync(p, "utf8");
const pkg = JSON.parse(read(join(ROOT, "package.json"))) as { version: string };

interface Frontmatter {
  name: string;
  description: string;
  version: string;
  license: string;
  compatibility?: string;
  metadata?: Record<string, unknown>;
}

function frontmatter(text: string): { data: Frontmatter; body: string } {
  const m = /^---\n([\s\S]*?)\n---\n([\s\S]*)$/.exec(text);
  if (!m) throw new Error("нет YAML-шапки");
  return { data: Bun.YAML.parse(m[1] ?? "") as Frontmatter, body: m[2] ?? "" };
}

const skillDirs = readdirSync(SKILLS).filter((d) => existsSync(join(SKILLS, d, "SKILL.md")));

describe.each(skillDirs)("скилл %s", (dir) => {
  const path = join(SKILLS, dir, "SKILL.md");
  const text = read(path);
  const { data, body } = frontmatter(text);

  test("шапка по спецификации agentskills.io", () => {
    expect(data.name).toBe(dir);
    expect(data.name).toMatch(/^[a-z0-9]+(-[a-z0-9]+)*$/);
    expect(data.name.length).toBeLessThanOrEqual(64);
    expect(data.description.length).toBeGreaterThan(100);
    expect(data.description.length).toBeLessThanOrEqual(1024);
    expect(data.license).toBe("MIT");
    expect(data.version).toBe(pkg.version);
  });

  test("описание содержит русские фразы-триггеры", () => {
    expect(data.description).toMatch(/Используй, когда/);
    expect(/\p{Script=Cyrillic}/u.test(data.description)).toBe(true);
  });

  test("точка входа короче 500 строк", () => {
    expect(text.split("\n").length).toBeLessThan(500);
  });

  test("относительные ссылки ведут на существующие файлы", () => {
    for (const m of body.matchAll(/\]\((?!https?:|#)([^)#\s]+)(?:#[^)]*)?\)/g)) {
      const target = join(dirname(path), m[1] ?? "");
      expect(existsSync(target), `${dir}: ${m[1]}`).toBe(true);
    }
  });

  test("пути к детектору существуют", () => {
    for (const m of body.matchAll(/bun \.\.\/\.\.\/(\S+\.ts)/g)) {
      expect(existsSync(join(dirname(path), "../..", m[1] ?? "")), m[1]).toBe(true);
    }
  });

  test("проходит собственный детектор без P0 и P1", () => {
    const serious = analyze(text, { context: "technical" }).issues.filter((i) => i.severity !== "P2");
    expect(serious.map((i) => `${i.line}: ${i.text}`)).toEqual([]);
  });
});

describe("основной скилл", () => {
  const skill = read(join(SKILLS, "avoid-ai-writing-russian/SKILL.md"));

  test("ссылается на каталог и под-скилл", () => {
    expect(skill).toContain("references/patterns.md");
    expect(skill).toContain("../antiplagiat/SKILL.md");
  });

  test("все режимы описаны", () => {
    for (const mode of ["`rewrite`", "`detect`", "`edit`"]) expect(skill).toContain(mode);
  });

  test("профили в --context совпадают с детектором", () => {
    const m = /--context ([a-z|-]+)/.exec(skill);
    const listed = (m?.[1] ?? "").split("|").sort();
    expect(listed).toEqual(Object.keys(PROFILE_TO_MODE).sort());
  });

  test("есть правило «Никогда не добавляй» с выдуманной конкретикой", () => {
    expect(skill).toContain("### Никогда не добавляй");
    expect(skill).toContain("Выдуманная конкретика");
  });

  test("формат ответа: четыре пункта проверки", () => {
    for (const item of ["**Проходы**", "**Проверки**", "**Остатки**", "**Причина остановки**"]) {
      expect(skill).toContain(item);
    }
  });
});

describe("под-скилл antiplagiat", () => {
  const skill = read(join(SKILLS, "antiplagiat/SKILL.md"));

  test("опирается на основной скилл", () => {
    expect(skill).toContain("../avoid-ai-writing-russian/SKILL.md");
  });

  test("описывает границу: не маскирует чужой текст", () => {
    expect(skill).toContain("## Граница");
    expect(skill).toMatch(/не маскирует чужой текст/);
  });

  test("все команды детектора упомянуты", () => {
    for (const cmd of ["antiplagiat", "validate", "calibrate"]) expect(skill).toContain(`src/cli.ts ${cmd}`);
  });
});

describe("каталог ↔ детектор", () => {
  const catalog = read(CATALOG);
  const headings = [...catalog.matchAll(/^#{3,4} (.+?)\s*$/gm)].map((m) => m[1] ?? "");
  const detectionPart = catalog.slice(0, catalog.indexOf("## Профили контекста"));
  const detectionHeadings = [...detectionPart.matchAll(/^### (.+?)\s*$/gm)].map((m) => m[1] ?? "");

  test("каждый тип детектора привязан к существующему разделу", () => {
    for (const type of Object.keys(WEIGHTS)) {
      const section = TYPE_TO_SECTION[type];
      expect(section, `нет раздела для ${type}`).toBeDefined();
      expect(headings, `${type} → ${section}`).toContain(section as string);
    }
  });

  test("привязки не ссылаются на несуществующие типы", () => {
    for (const type of Object.keys(TYPE_TO_SECTION)) expect(TYPE_LABELS[type], type).toBeDefined();
  });

  test("каждый раздел каталога либо проверяется детектором, либо помечен как суждение", () => {
    const mapped = new Set(Object.values(TYPE_TO_SECTION));
    for (const h of detectionHeadings) {
      expect(mapped.has(h) || JUDGMENT_ONLY.includes(h), `раздел «${h}» не учтён`).toBe(true);
    }
    for (const h of JUDGMENT_ONLY) expect(headings, h).toContain(h);
  });

  test("таблица соответствия профилей совпадает с детектором", () => {
    const table = catalog.slice(catalog.indexOf("### Соответствие режимам детектора"));
    for (const [profile, mode] of Object.entries(PROFILE_TO_MODE)) {
      expect(table).toMatch(new RegExp(`\\| \`${profile}\` \\| \`${mode}\` \\|`));
    }
  });

  test("в матрице строгости есть столбец для каждого профиля", () => {
    const header = /\| Правило \|(.+)\|/.exec(catalog)?.[1] ?? "";
    const cols = header.split("|").map((c) => c.trim());
    expect(cols.sort()).toEqual(Object.keys(PROFILE_TO_MODE).sort());
  });
});

/**
 * Примеры из каталога: то, что каталог называет приметой, детектор должен
 * находить, а то, что каталог называет законным, — пропускать.
 */
const CATALOG_EXAMPLES: { text: string; type: string; context?: Parameters<typeof analyze>[1] }[] = [
  { text: "Модель работает быстро — и это меняет всё.", type: "em-dash-splice" },
  { text: "# Анализ Существующих Методов Распознавания\n\nТекст.", type: "title-case" },
  { text: "Это не просто инструмент, а целая экосистема.", type: "not-x-but-y" },
  { text: "Обеспечение повышения эффективности проведения мониторинга состояния машиниста.", type: "genitive-chain" },
  { text: "Мы решаем задачу, которая адресует проблему задержек.", type: "calque" },
  { text: "Логи смотрим на ежедневной основе.", type: "calque" },
  { text: "Ни для кого не секрет, что сон важен.", type: "template" },
  { text: "Подводя итог, скажем главное.", type: "transition" },
  { text: "Это знаменует новую эру в медицине.", type: "significance" },
  { text: "Технология имеет все шансы стать одним из ключевых трендов.", type: "future-narrative" },
  { text: "Решение потенциально может ускорить работу.", type: "hedge-stack" },
  { text: "Мы получили реальную пользу от внедрения.", type: "real-inflation" },
  { text: "Это честная метрика качества.", type: "moral-adjective" },
  { text: "Эксперты считают, что это важно.", type: "vague-attribution" },
  { text: "Таким образом, можно сделать вывод, что метод работает.", type: "generic-conclusion" },
  { text: "Надеюсь, это поможет!", type: "chatbot" },
  { text: "Давайте разберёмся, как это устроено.", type: "lets" },
  { text: "Вот о чём почему-то молчат авторы учебников.", type: "novelty" },
  { text: "И вот тут начинается самое интересное.", type: "hook" },
  { text: "Сохраняйте себе, пригодится.", type: "social-closer" },
  { text: "Встречайте: Flowdesk!", type: "launch-intro" },
  { text: "Горячее мнение: тесты не нужны.", type: "fake-casual" },
  { text: "Представьте мир, где поезда ходят сами.", type: "speculative-opener" },
  { text: "Давайте подумаем шаг за шагом.", type: "reasoning" },
  { text: "Отличный вопрос! Разберём.", type: "sycophancy" },
  { text: "Буду честен: мы не успели.", type: "narrated-candor" },
  { text: "Вы спрашиваете о том, как это работает.", type: "acknowledgment" },
  { text: "По состоянию на мою последнюю информацию, данных нет.", type: "cutoff" },
  { text: "Стоит отметить, что метод новый.", type: "filler" },
  { text: "Модель показала высокую точность.", type: "unmeasured-claim" },
  { text: "Без воды, без лишних слов, без жаргона.", type: "negation-chain" },
  { text: "Он назвал это “прорывом”.", type: "english-quotes" },
  { text: "Точность составила 0.93.", type: "decimal-point" },
  { text: "См. [Вставьте источник].", type: "placeholder" },
  { text: "Это крайне важная задача.", type: "hollow-intensifier" },
  { text: "Безусловно, метод работает.", type: "confidence" },
  { text: "Эта статья заслуживает внимания.", type: "vague-endorsement" },
  { text: "Хотя результаты впечатляют, вопрос остается открытым.", type: "false-concession" },
];

const CATALOG_LEGIT: { text: string; type: string; context?: Parameters<typeof analyze>[1] }[] = [
  { text: "Цель работы — разработка метода.", type: "em-dash-splice" },
  { text: "Интервал 2019–2023 годов.", type: "hyphen-dash" },
  { text: "Он назвал это «прорывом».", type: "straight-quotes" },
  { text: "Точность составила 0,93.", type: "decimal-point" },
  { text: "Мы ожидали рост задержки, и он действительно вырос на 12 %.", type: "hollow-intensifier" },
  { text: "Научная новизна заключается в новом методе.", type: "tier2", context: { context: "academic" } },
  { text: "Экосистема пакетов npm растёт.", type: "tier1", context: { context: "technical" } },
  { text: "Настроили Wi-Fi-роутер для IT-отдела.", type: "homoglyph" },
  { text: "Задача #12 закрыта, цвет #1a2b3c.", type: "hashtag-stuffing" },
  { text: "Представим отсортированный массив из десяти чисел и найдём медиану.", type: "speculative-opener" },
];

describe("примеры из каталога", () => {
  test.each(CATALOG_EXAMPLES)("находит: $text", ({ text, type, context }) => {
    expect(analyze(text, context).issues.map((i) => i.type)).toContain(type);
  });

  test.each(CATALOG_LEGIT)("пропускает законное: $text", ({ text, type, context }) => {
    expect(analyze(text, context).issues.map((i) => i.type)).not.toContain(type);
  });

  test("примеры покрывают большинство словарных типов", () => {
    const covered = new Set(CATALOG_EXAMPLES.map((e) => e.type));
    const lexical = Object.keys(TYPE_TO_SECTION).filter(
      (t) =>
        !/uniform|diversity|cluster|run|tier[23]|phrase3|same-opener|staccato|stacked|bullet|hashtag|bold|emoji|homoglyph|invisible|chat-markup|ai-url|hyphen|straight|tier1/.test(
          t,
        ),
    );
    const missing = lexical.filter((t) => !covered.has(t));
    expect(missing).toEqual([]);
  });
});

import { describe, expect, test } from "bun:test";
import { analyze } from "../src/index.ts";

const fixture = (name: string): Promise<string> => Bun.file(new URL(`./fixtures/${name}`, import.meta.url)).text();
const types = (text: string, context?: Parameters<typeof analyze>[1]): string[] =>
  analyze(text, context).issues.map((i) => i.type);

describe("оценка", () => {
  test("шаблонный ИИ-текст получает высокую оценку", async () => {
    const r = analyze(await fixture("corpus/ai/blog.md"));
    expect(r.score).toBeGreaterThanOrEqual(70);
    expect(r.label).toBe("сильный ИИ-стиль");
  });

  test("живая проза остаётся чистой", async () => {
    const r = analyze(await fixture("corpus/human/blog.md"));
    expect(r.score).toBeLessThan(15);
    expect(r.issues.filter((i) => i.severity !== "P2")).toHaveLength(0);
  });

  test("пустой ввод не падает", () => {
    expect(analyze("").score).toBe(0);
  });
});

describe("словарь", () => {
  test("морфология: разные формы одного слова", () => {
    for (const s of ["Работа осуществляется быстро.", "Мы осуществили проверку.", "Осуществляя анализ, мы…"]) {
      expect(types(s)).toContain("tier1-clarity");
    }
  });

  test("ё и е равнозначны", () => {
    expect(types("Давайте разберёмся с этим.")).toContain("lets");
  });

  test("границы слов в кириллице", () => {
    // «проявляется» не должно считаться «является»
    const text = "Эффект проявляется слабо. ".repeat(20);
    expect(analyze(text).issues.some((i) => i.rule === "является")).toBe(false);
  });

  test("вложенные совпадения схлопываются", () => {
    const r = analyze("Модель играет ключевую роль в системе.");
    const hits = r.issues.filter((i) => i.type === "tier1");
    expect(hits).toHaveLength(1);
    expect(hits[0]?.text).toBe("играет ключевую роль");
  });

  test("канцелярит помечен как совет по стилю", () => {
    const r = analyze("В рамках работы данный метод применяется в целях оценки.");
    const clerical = r.issues.filter((i) => i.type === "tier1-clarity");
    expect(clerical.length).toBeGreaterThan(0);
    expect(clerical.every((i) => i.styleOnly)).toBe(true);
  });

  test("кальки", () => {
    expect(types("Это решение адресует проблему задержек.")).toContain("calque");
    expect(types("Мы проверяем логи на ежедневной основе.")).toContain("calque");
  });

  test("«как ИИ» в обычной речи не оговорка об отсечке", () => {
    expect(types("Фрагменты, помеченные как ИИ, переписывают.")).not.toContain("cutoff");
    expect(types("Я, как языковая модель, не могу знать.")).toContain("cutoff");
  });

  test("размытая ссылка на авторитет — P0", () => {
    const r = analyze("Исследования показывают, что сон важен.");
    expect(r.issues.find((i) => i.type === "vague-attribution")?.severity).toBe("P0");
  });
});

describe("защищённое содержимое", () => {
  test("цитаты, код и ссылки не проверяются", () => {
    const text = [
      "Автор пишет: «Давайте разберёмся, это играет ключевую роль».",
      "",
      "```",
      "// Давайте погрузимся в код",
      "```",
      "",
      "Команда `давайте рассмотрим` описана в [12].",
    ].join("\n");
    expect(analyze(text).issues).toHaveLength(0);
  });

  test("YAML-шапка не проверяется", () => {
    expect(analyze("---\ntitle: Давайте разберёмся\n---\nОбычный текст.").issues).toHaveLength(0);
  });
});

describe("технические отпечатки", () => {
  test("невидимые символы и подмена букв", () => {
    const zw = String.fromCharCode(0x200b);
    const latinA = String.fromCharCode(0x61);
    const r = analyze(`Ра${zw}бота и р${latinA}бота с латинской a.`);
    expect(r.suspicious).toBe(true);
    expect(r.issues.map((i) => i.type)).toEqual(expect.arrayContaining(["invisible-chars", "homoglyph"]));
  });

  test("законная смесь алфавитов не считается подменой", () => {
    expect(analyze("Настроили Wi-Fi-роутер и IT-отдел.").suspicious).toBe(false);
  });

  test("смещения указывают в исходник даже после удаления невидимых символов", () => {
    const zw = String.fromCharCode(0x200b);
    const src = `${zw}${zw}Давайте разберёмся.`;
    const hit = analyze(src).issues.find((i) => i.type === "lets");
    expect(src.slice(hit?.index ?? -1, (hit?.index ?? 0) + 7)).toBe("Давайте");
  });

  test("заглушки, разметка чатов и utm", () => {
    const t = types("См. [Вставьте источник] citeturn0search0 https://x.ru/a?utm_source=chatgpt.com");
    expect(t).toEqual(expect.arrayContaining(["placeholder", "chat-markup", "ai-url"]));
  });
});

describe("типографика", () => {
  test("грамматическое тире не находка, тире-связка — находка", () => {
    expect(types("Цель работы — разработка метода.")).not.toContain("em-dash-splice");
    const splice = "Модель работает быстро — и это меняет всё. Мы проверили гипотезу — именно так и вышло. ";
    expect(types(splice)).toContain("em-dash-splice");
  });

  test("прямые и английские кавычки в русском тексте", () => {
    expect(types('Он назвал это "прорывом".')).toContain("straight-quotes");
    expect(types("Он назвал это “прорывом”.")).toContain("english-quotes");
    expect(types("Он назвал это «прорывом».")).not.toContain("straight-quotes");
  });

  test("десятичная точка", () => {
    expect(types("Точность составила 0.93 на тесте.")).toContain("decimal-point");
    expect(types("Точность составила 0,93 на тесте.")).not.toContain("decimal-point");
    expect(types("См. раздел 2.1 и версию 1.2.3.")).not.toContain("decimal-point");
  });

  test("Title Case в заголовке", () => {
    expect(types("# Анализ Существующих Методов Распознавания\n\nТекст.")).toContain("title-case");
    expect(types("# Анализ существующих методов\n\nТекст.")).not.toContain("title-case");
  });
});

describe("структура", () => {
  test("«не X, а Y»", () => {
    expect(types("Это не просто инструмент, а целая платформа.")).toContain("not-x-but-y");
  });

  test("цепочка отглагольных существительных", () => {
    expect(types("Цель — обеспечение повышения эффективности проведения мониторинга состояния машиниста.")).toContain(
      "genitive-chain",
    );
  });

  test("переходы в начале абзацев подряд", () => {
    const text = ["Кроме того, первое.", "Более того, второе.", "Таким образом, третье."].join("\n\n");
    expect(types(text)).toContain("transition-run");
  });

  test("список из голых именных групп", () => {
    const list = [
      "- Высокая точность",
      "- Низкая задержка",
      "- Надёжная работа",
      "- Удобный интерфейс",
      "- Гибкая настройка",
    ].join("\n");
    expect(types(list)).toContain("bullet-np-list");
  });

  test("рубленые фрагменты", () => {
    expect(
      types(
        "Никаких предпочтений. Никакой эстетики. Ноль ностальгии. Старые правила исчезли навсегда, и это было заметно.",
      ),
    ).toContain("staccato");
  });
});

describe("режимы", () => {
  test("academic пропускает обязательные научные формулы", () => {
    const text = "Научная новизна заключается в новом методе. Практическая значимость работы подтверждена.";
    expect(types(text, { context: "academic" })).not.toContain("tier2");
  });

  test("chat оставляет только P0 и кальки", () => {
    const r = analyze('Кстати, "это" играет ключевую роль, но это имеет смысл. Надеюсь, это поможет!', {
      context: "chat",
    });
    expect(r.issues.every((i) => i.severity === "P0" || i.type === "calque")).toBe(true);
  });

  test("technical не отмечает законные термины", () => {
    expect(types("Экосистема пакетов npm огромна.", { context: "technical" })).not.toContain("tier1");
  });
});

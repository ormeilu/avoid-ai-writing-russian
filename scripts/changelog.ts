/** Работа с CHANGELOG.md: раздел «Не выпущено» и заметки к выпуску. */

export const UNRELEASED = "## [Не выпущено]";

/** Текст раздела версии (без заголовка) или undefined. */
export function section(changelog: string, version: string): string | undefined {
  const head = version === "unreleased" ? UNRELEASED : `## [${version}]`;
  const start = changelog.indexOf(head);
  if (start < 0) return undefined;
  const bodyStart = changelog.indexOf("\n", start) + 1;
  const next = changelog.slice(bodyStart).search(/^## \[/m);
  const body = next < 0 ? changelog.slice(bodyStart) : changelog.slice(bodyStart, bodyStart + next);
  return body.replace(/^\[[^\]]+\]:.*$/gm, "").trim();
}

/** Превращает «Не выпущено» в раздел версии и открывает новый пустой. */
export function cut(changelog: string, version: string, date: string): string {
  const body = section(changelog, "unreleased");
  if (!body) throw new Error("раздел «Не выпущено» пуст или отсутствует: нечего выпускать");
  return changelog.replace(UNRELEASED, `${UNRELEASED}\n\n## [${version}] — ${date}`);
}

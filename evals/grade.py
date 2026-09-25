"""Оценка ответа агента по проверкам из cases.json. Модуль не вызывает
агента: его тестирует tests/test_evals.py на готовых ответах, а
evals/run.py применяет к живым ответам.

Выражения в cases.json записаны в синтаксисе JavaScript, поэтому
компилируются через `jsre` детектора с той же семантикой.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypedDict

from aiw_ru import analyze, validate
from aiw_ru.compat import jsre, to_fixed, trim


class Checks(TypedDict, total=False):
    # Строки, которые должны остаться в итоговом тексте.
    preserve: list[str]
    # Строки, которых не должно быть в итоговом тексте.
    forbid: list[str]
    # В итоговом тексте нет чисел, которых не было в исходнике («никогда не добавляй»).
    noNewNumbers: bool
    # Итоговый текст совпадает с исходным (с точностью до пробелов).
    unchanged: bool
    # validate(исходный, итоговый) без нарушений.
    validate: bool
    # Оценка детектора после правки ниже, чем до.
    scoreDrop: bool
    # Итоговый текст не длиннее исходного больше чем в N раз.
    maxGrowth: float
    # Не больше N хэштегов в итоговом тексте.
    maxHashtags: int
    # Регулярные выражения, которым должен соответствовать итоговый текст.
    finalMatches: list[str]
    # Строки, которые должны быть где-то в ответе.
    responseHas: list[str]
    # Строки, которых не должно быть в ответе.
    responseLacks: list[str]
    # Регулярные выражения (без учёта регистра) для всего ответа.
    responseMatches: list[str]


class EvalCase(TypedDict):
    id: str
    skill: Literal["avoid-ai-writing-russian", "antiplagiat"]
    request: str
    input: str
    checks: Checks


@dataclass(slots=True)
class Grade:
    id: str
    # Случай прошёл (`pass` — ключевое слово Python).
    passed: bool
    failures: list[str] = field(default_factory=list)
    final: str | None = None


HEAD = jsre(r"(?:^|\n)\s*(?:#{1,4}\s*|\*\*)Итоговый текст:?(?:\*\*)?:?\s*\n")
# Следующий раздел ответа: заголовком (`## Проверка`) или жирной подписью, в том числе с текстом
# на той же строке (`**Проверка:** один проход`).
NEXT = jsre(r"\n\s*(?:#{1,4}\s*|\*\*)(?:Изменения|Проверка|Найдено|Оценка)(?=[:*\s]|$)")
FENCE = jsre(r"^```[^\n]*\n([\s\S]*?)\n```$")
QUOTE_MARK = jsre(r"^>\s?", "gm")
SPACES = jsre(r"\s+", "g")
NUMBER = jsre(r"\d+(?:[.,]\d+)?", "g")
HASHTAG = jsre(r"(?<![\p{L}\d])#[\p{L}_][\p{L}\d_]*", "gu")


def extract_final(response: str) -> str | None:
    """Итоговый текст из ответа: раздел «Итоговый текст» без обрамляющих кавычек и ограждений."""
    m = HEAD.search(response)
    if not m:
        return None
    rest = response[m.end() :]
    nxt = NEXT.search(rest)
    if nxt:
        rest = rest[: nxt.start()]
    rest = trim(rest)
    fence = FENCE.search(rest)
    if fence:
        rest = fence[1] or ""
    if all(line.startswith(">") or trim(line) == "" for line in rest.split("\n")):
        rest = QUOTE_MARK.sub("", rest)
    return trim(rest)


def _norm(s: str) -> str:
    return trim(SPACES.sub(" ", s))


def _numbers(s: str) -> list[str]:
    return [n.replace(",", ".", 1) for n in NUMBER.findall(s)]


def grade(c: EvalCase, response: str) -> Grade:
    f: list[str] = []
    ch = c["checks"]
    # Истинность как в JS: пустой список — тоже проверка, а maxHashtags: 0 — нет.
    needs_final = (
        ch.get("preserve") is not None
        or ch.get("forbid") is not None
        or bool(ch.get("noNewNumbers"))
        or bool(ch.get("unchanged"))
        or bool(ch.get("validate"))
        or bool(ch.get("scoreDrop"))
        or bool(ch.get("maxGrowth"))
        or bool(ch.get("maxHashtags"))
        or ch.get("finalMatches") is not None
    )
    final = extract_final(response)
    if needs_final and final is None:
        f.append("в ответе нет раздела «Итоговый текст»")
    if final is not None:
        f.extend(f"пропало: {s}" for s in ch.get("preserve") or [] if s not in final)
        f.extend(f"осталось: {s}" for s in ch.get("forbid") or [] if s in final)
        if ch.get("noNewNumbers"):
            src = set(_numbers(c["input"]))
            added = [n for n in _numbers(final) if n not in src]
            if added:
                f.append(f"выдуманные числа: {', '.join(added)}")
        if ch.get("unchanged") and _norm(final) != _norm(c["input"]):
            f.append("чистый текст изменён")
        if ch.get("validate"):
            v = validate(c["input"], final)
            f.extend(f"validate [{x.kind}]: {x.detail}" for x in v.violations)
        if ch.get("scoreDrop"):
            before = analyze(c["input"]).score
            after = analyze(final).score
            if after >= before:
                f.append(f"оценка не снизилась: {before} → {after}")
        growth = ch.get("maxGrowth")
        if growth and len(final) > len(c["input"]) * growth:
            f.append(f"текст вырос в {to_fixed(len(final) / len(c['input']), 1)} раза")
        max_tags = ch.get("maxHashtags")
        if max_tags is not None:
            tags = HASHTAG.findall(final)
            if len(tags) > max_tags:
                f.append(f"хэштегов {len(tags)}, допустимо {max_tags}")
        f.extend(f"итог не соответствует /{p}/" for p in ch.get("finalMatches") or [] if not jsre(p, "u").search(final))
    f.extend(f"в ответе нет: {s}" for s in ch.get("responseHas") or [] if s not in response)
    f.extend(f"в ответе лишнее: {s}" for s in ch.get("responseLacks") or [] if s in response)
    f.extend(
        f"ответ не соответствует /{p}/" for p in ch.get("responseMatches") or [] if not jsre(p, "iu").search(response)
    )
    return Grade(id=c["id"], passed=not f, failures=f, final=final)

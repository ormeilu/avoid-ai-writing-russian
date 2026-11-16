"""Импорт отчётов в калибровку: отпечатки, повторный импорт, проверка переноса и атомарная запись.

`calibrate` в командной строке читает файл, передаёт отчёты в `import_reports` и пишет результат
через `write_file`. Модели данных (`CalibrationFile`, `Calibration`, `Sample`, `ReportRecord`)
лежат в `aiw_ru.antiplagiat`, рядом с моделью оценки.

Тот же отчёт — та же версия текста документа и та же разметка (куски или доля вместе с версией
системы) в калибровке той же системы. Его повторный импорт ничего не добавляет. Отчёт по правленой
версии документа добавляет образцы рядом с прежними: на этом держится цикл правки. Другая разметка
для той же версии текста заменяет прежнюю только с `replace=True`.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import stat
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from aiw_ru.antiplagiat import (
    DEFAULT_MODEL,
    Calibration,
    CalibrationFile,
    DocSample,
    ReportRecord,
    Sample,
    document_sample,
    fit,
    fit_share,
    label_fragments,
    predict,
)

# Длина отпечатка в шестнадцатеричных знаках: 64 бита, случайное совпадение у пары отчётов исключено.
FINGERPRINT_CHARS = 16
# Больше стольких групп документов проверка переноса не делит: группы раскладываются по блокам,
# и каждый блок откладывается целиком. Иначе на сотне документов calibrate считал бы минуты.
TRANSFER_FOLDS = 10


class CalibrationError(ValueError):
    """Отчёт нельзя импортировать: например, другая разметка для уже импортированной версии текста."""


class CalibrationConflict(Exception):
    """Файл калибровки изменился, пока calibrate считал: его записал другой процесс."""


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:FINGERPRINT_CHARS]


def _normalized(text: str) -> str:
    """Текст без различий, которые не меняют документ: BOM, переводы строк, пробелы в концах строк, NFC."""
    text = unicodedata.normalize("NFC", text.removeprefix("\ufeff")).replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def doc_fingerprint(text: str) -> str:
    """Отпечаток версии документа: правка хоть одного слова даёт другой отпечаток."""
    return _digest(_normalized(text))


def calibration_fingerprint(calibration: Calibration) -> str:
    """Отпечаток калибровки для JSON-отчёта: по нему видно, с какой калибровкой посчитан отчёт."""
    return _digest(json.dumps(calibration.to_dict(), ensure_ascii=False, sort_keys=True))


@dataclass(frozen=True, slots=True)
class ReportInput:
    """Отчёт для импорта: документ и разметка из отчёта системы проверки."""

    # Имя для сообщений (путь к файлу документа); в калибровку не попадает.
    name: str
    text: str
    marked: list[str] | None = None
    share: float | None = None
    # Дата отчёта (ГГГГ-ММ-ДД) и версия модели системы, если пользователь их назвал.
    date: str | None = None
    system_version: str | None = None

    @property
    def kind(self) -> Literal["marked", "share"]:
        return "share" if self.share is not None else "marked"


def report_fingerprint(report: ReportInput) -> str:
    """Отпечаток разметки: куски без различий в пробелах или доля, вместе с версией системы."""
    if report.share is not None:
        markup = f"share:{float(report.share)!r}"
    else:
        pieces = [" ".join(_normalized(piece).split()) for piece in report.marked or []]
        markup = "marked:" + json.dumps(pieces, ensure_ascii=False)
    return _digest(f"{markup}\nversion:{report.system_version or ''}")


Status = Literal["added", "skipped", "replaced"]


@dataclass(frozen=True, slots=True)
class Outcome:
    """Что стало с отчётом при импорте."""

    name: str
    status: Status
    # Фрагментов документа; для разметки ещё и сколько из них подсвечено.
    fragments: int
    positives: int | None = None
    # Разметка была, но ни один кусок не нашёлся в документе.
    unmatched: bool = False


@dataclass(frozen=True, slots=True)
class ImportResult:
    calibration: Calibration
    outcomes: list[Outcome]
    # False, если все отчёты уже были в калибровке: файл переписывать не нужно.
    changed: bool


def import_reports(
    previous: Calibration | None, reports: list[ReportInput], *, today: str, replace: bool = False
) -> ImportResult:
    """Калибровка с новыми отчётами.

    Повтор того же отчёта пропускается. Другая разметка для версии текста, которая уже есть в
    калибровке, — CalibrationError, если не задан `replace`: тогда прежние образцы этой версии
    заменяются. Модель обучается заново от модели по умолчанию на всех образцах, поэтому итог не
    зависит от порядка импорта. Образцы без отпечатка (файлы прежних версий) остаются как есть.
    """
    samples: list[Sample] = list(previous.samples) if previous else []
    documents: list[DocSample] = list(previous.documents) if previous else []
    records: list[ReportRecord] = list(previous.reports) if previous else []
    outcomes: list[Outcome] = []
    changed = False
    for report in reports:
        doc = doc_fingerprint(report.text)
        markup = report_fingerprint(report)
        known = next((record for record in records if record.doc == doc), None)
        status: Status = "added"
        if known is not None:
            if known.report == markup:
                outcomes.append(Outcome(report.name, "skipped", known.fragments))
                continue
            if not replace:
                raise CalibrationError(
                    f"{report.name}: эта версия документа уже в калибровке с другой разметкой (импорт "
                    f"{known.imported}). Если это заново выписанная разметка того же отчёта или отчёт новой "
                    "версии системы, добавьте --replace. Если документ правили, импортируйте правленую версию: "
                    "она добавится рядом с прежней"
                )
            samples = [sample for sample in samples if sample.doc != doc]
            documents = [document for document in documents if document.doc != doc]
            records = [record for record in records if record.doc != doc]
            status = "replaced"
        if report.share is not None:
            document = document_sample(report.text, report.share).model_copy(update={"doc": doc})
            documents.append(document)
            outcome = Outcome(report.name, status, len(document.fragments))
        else:
            marked = report.marked or []
            labeled = [sample.model_copy(update={"doc": doc}) for sample in label_fragments(report.text, marked)]
            samples.extend(labeled)
            positives = sum(sample.y for sample in labeled)
            outcome = Outcome(report.name, status, len(labeled), positives, bool(marked) and positives == 0)
        records.append(
            ReportRecord(
                doc=doc,
                report=markup,
                kind=report.kind,
                fragments=outcome.fragments,
                imported=today,
                date=report.date,
                system_version=report.system_version,
            )
        )
        outcomes.append(outcome)
        changed = True
    if previous is not None and not changed:
        return ImportResult(previous, outcomes, changed=False)
    model = fit_share(fit(samples, DEFAULT_MODEL), documents)
    return ImportResult(
        Calibration(model=model, samples=samples, documents=documents, reports=records), outcomes, changed=True
    )


@dataclass(frozen=True, slots=True)
class Transfer:
    """Проверка переноса: каждый документ отложен целиком, его фрагменты предсказаны моделью без него.

    `accuracy` и `default_accuracy` — сбалансированная точность калибровки и модели по умолчанию на
    отложенных фрагментах; None, если проверить нельзя (меньше двух документов с разметкой или в
    отложенных фрагментах нет обоих классов).
    """

    documents: int
    accuracy: float | None = None
    default_accuracy: float | None = None


def _balanced(pairs: list[tuple[bool, int]]) -> float | None:
    positives = [hit for hit, y in pairs if y == 1]
    negatives = [hit for hit, y in pairs if y == 0]
    if not positives or not negatives:
        return None
    return (sum(positives) / len(positives) + sum(not hit for hit in negatives) / len(negatives)) / 2


def transfer_check(samples: list[Sample]) -> Transfer:
    """Перенос калибровки на документ, которого не было в обучении.

    Образцы группируются по документу (`Sample.doc`; образцы без отпечатка — одна группа). Каждый
    блок групп откладывается целиком, модель обучается на остальных и предсказывает отложенные
    фрагменты. Соседние абзацы того же документа в обучение не попадают.
    """
    groups: dict[str, list[Sample]] = {}
    for sample in samples:
        groups.setdefault(sample.doc or "", []).append(sample)
    keys = sorted(groups)
    if len(keys) < 2:
        return Transfer(documents=len(keys))
    folds = [keys[start::TRANSFER_FOLDS] for start in range(min(TRANSFER_FOLDS, len(keys)))]
    calibrated: list[tuple[bool, int]] = []
    default: list[tuple[bool, int]] = []
    for fold in folds:
        held = [sample for key in fold for sample in groups[key]]
        train = [sample for key in keys if key not in fold for sample in groups[key]]
        if len({sample.y for sample in train}) < 2:
            continue
        model = fit(train, DEFAULT_MODEL)
        calibrated += [(predict(model, sample.f) >= model.threshold, sample.y) for sample in held]
        default += [(predict(DEFAULT_MODEL, sample.f) >= DEFAULT_MODEL.threshold, sample.y) for sample in held]
    return Transfer(documents=len(keys), accuracy=_balanced(calibrated), default_accuracy=_balanced(default))


@dataclass(frozen=True, slots=True)
class Loaded:
    """Файл калибровки и его байты на момент чтения: по ним write_file замечает чужую запись."""

    data: CalibrationFile | None
    raw: bytes | None


def read_file(path: Path) -> Loaded:
    """Файл калибровки; `data` и `raw` — None, если файла нет. Ошибки формата поднимает pydantic."""
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return Loaded(None, None)
    return Loaded(CalibrationFile.model_validate_json(raw), raw)


def serialize(data: CalibrationFile) -> bytes:
    return (json.dumps(data.to_dict(), ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _new_file_mode() -> int:
    """Права нового файла, как у open(): 0o666 без битов umask."""
    umask = os.umask(0)
    os.umask(umask)
    return 0o666 & ~umask


def write_file(path: Path, data: CalibrationFile, expected: bytes | None) -> None:
    """Атомарная запись: временный файл в той же папке, fsync, затем замена целевого.

    `expected` — байты файла при чтении (None, если его не было). Если файл с тех пор изменился,
    поднимается CalibrationConflict и ничего не пишется: обновление другого процесса не теряется.
    При ошибке записи прежний файл остаётся нетронутым, а временный удаляется.
    """
    try:
        current = path.read_bytes()
    except FileNotFoundError:
        current = None
    if current != expected:
        raise CalibrationConflict(str(path))
    mode = stat.S_IMODE(path.stat().st_mode) if current is not None else _new_file_mode()
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(serialize(data))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise

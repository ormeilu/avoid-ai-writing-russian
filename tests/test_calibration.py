"""Надёжность калибровки: проверка модели, отпечатки и повторный импорт, перенос на новый документ,
атомарная запись и чтение файлов прежних версий."""

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from aiw_ru import DEFAULT_MODEL, Model, Sample, predict
from aiw_ru import calibration as cal
from aiw_ru.antiplagiat import Calibration, CalibrationFile, _sigmoid, empty_calibration_file
from aiw_ru.calibration import (
    CalibrationConflict,
    CalibrationError,
    ReportInput,
    doc_fingerprint,
    import_reports,
    read_file,
    report_fingerprint,
    serialize,
    transfer_check,
    write_file,
)

TODAY = "2026-11-15"
GOOD_MODEL: dict[str, Any] = {"bias": -2.0, "weights": [0.3, 1.9, 0.75, 0.16, 0.15, 1.05, 1.0], "threshold": 0.5}


@pytest.fixture
def doc(read_fixture: Callable[[str], str]) -> str:
    """Человеческий и ИИ-текст в жанре ВАК подряд: пять фрагментов, последние два — ИИ."""
    return f"{read_fixture('corpus/human/vak.md')}\n\n{read_fixture('corpus/ai/vak.md')}"


@pytest.fixture
def red(read_fixture: Callable[[str], str]) -> list[str]:
    """Подсвеченный кусок: второй абзац ИИ-текста."""
    return [read_fixture("corpus/ai/vak.md").split("\n\n")[1]]


def marked_report(text: str, marked: list[str], **extra: Any) -> ReportInput:
    return ReportInput(name="glava1.md", text=text, marked=marked, **extra)


# ─── модель калибровки ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "change",
    [
        pytest.param({"bias": math.inf}, id="bias бесконечный"),
        pytest.param({"bias": math.nan}, id="bias NaN"),
        pytest.param({"weights": [1.0] * 6}, id="весов меньше признаков"),
        pytest.param({"weights": [1.0] * 8}, id="весов больше признаков"),
        pytest.param({"weights": [1.0] * 6 + [math.nan]}, id="вес NaN"),
        pytest.param({"threshold": 0.0}, id="порог 0"),
        pytest.param({"threshold": 1.0}, id="порог 1"),
        pytest.param({"threshold": 1.5}, id="порог больше 1"),
        pytest.param({"threshold": -0.1}, id="порог меньше 0"),
    ],
)
def test_model_rejects_broken_numbers(change: dict[str, Any]):
    """модель с бесконечными числами, чужим числом весов или порогом вне (0, 1) не читается"""
    with pytest.raises(ValidationError):
        Model.model_validate({**GOOD_MODEL, **change})


def test_default_and_good_models_valid():
    """модель по умолчанию и обычная калибровка проходят проверку"""
    Model.model_validate(DEFAULT_MODEL.to_dict())
    Model.model_validate(GOOD_MODEL)


def test_sample_rejects_nan_features():
    """признак NaN в образце — ошибка формата, а не тихий NaN в обучении"""
    with pytest.raises(ValidationError):
        Sample.model_validate({"f": [0.1] * 6 + [math.nan], "y": 1})


@pytest.mark.parametrize("bias", [-1000.0, -745.5, 0.0, 745.5, 1000.0])
def test_predict_extreme_bias_without_overflow(bias: float):
    """крайний свободный член не роняет predict: вероятность остаётся в [0, 1]"""
    probability = predict(Model(bias=bias, weights=[0.0] * 7, threshold=0.5), [0.0] * 7)
    assert 0.0 <= probability <= 1.0
    assert probability == pytest.approx(_sigmoid(bias))


@pytest.mark.parametrize("z", [-50.0, -3.0, -0.5, 0.0, 0.5, 3.0, 50.0])
def test_sigmoid_symmetric(z: float):
    """устойчивая сигмоида та же функция: σ(−z) = 1 − σ(z), σ(0) = 0,5"""
    assert _sigmoid(-z) == pytest.approx(1 - _sigmoid(z))
    assert _sigmoid(0) == 0.5


# ─── отпечатки ──────────────────────────────────────────────────────────


def test_doc_fingerprint_ignores_line_endings_and_trailing_spaces():
    """отпечаток документа не зависит от переводов строк, пробелов в концах строк и BOM"""
    text = "Склад принял партию.\nКоробки пересчитали дважды."
    assert doc_fingerprint(text) == doc_fingerprint("\ufeff" + text.replace("\n", "  \r\n") + "\n")
    assert doc_fingerprint(text) != doc_fingerprint(text.replace("дважды", "трижды"))


def test_report_fingerprint_same_markup_any_spacing():
    """отпечаток разметки не зависит от пробелов внутри кусков, но меняется от другого куска и версии системы"""
    base = ReportInput(name="a", text="т", marked=["Коробки  пересчитали\nдважды."])
    assert report_fingerprint(base) == report_fingerprint(
        ReportInput(name="b", text="т", marked=["Коробки пересчитали дважды."])
    )
    assert report_fingerprint(base) != report_fingerprint(ReportInput(name="a", text="т", marked=["Другой кусок."]))
    assert report_fingerprint(base) != report_fingerprint(
        ReportInput(name="a", text="т", marked=base.marked, system_version="2")
    )
    share = ReportInput(name="a", text="т", share=10)
    assert report_fingerprint(share) == report_fingerprint(ReportInput(name="a", text="т", share=10.0))
    assert report_fingerprint(share) != report_fingerprint(ReportInput(name="a", text="т", share=12))


# ─── повторный импорт ───────────────────────────────────────────────────


def test_first_import_records_report(doc: str, red: list[str]):
    """первый импорт: образцы с отпечатком документа и запись об отчёте"""
    result = import_reports(None, [marked_report(doc, red, date="2026-11-05", system_version="5.0")], today=TODAY)
    assert result.changed
    assert [outcome.status for outcome in result.outcomes] == ["added"]
    calibration = result.calibration
    assert {sample.doc for sample in calibration.samples} == {doc_fingerprint(doc)}
    (record,) = calibration.reports
    assert (record.kind, record.imported, record.date, record.system_version) == ("marked", TODAY, "2026-11-05", "5.0")
    assert record.fragments == len(calibration.samples)


def test_same_report_twice_changes_nothing(doc: str, red: list[str]):
    """повтор того же отчёта пропускается: выборка и модель прежние"""
    first = import_reports(None, [marked_report(doc, red)], today=TODAY).calibration
    again = import_reports(first, [marked_report(doc.replace("\n", "\r\n"), red)], today="2026-12-01")
    assert not again.changed
    assert [outcome.status for outcome in again.outcomes] == ["skipped"]
    assert again.calibration == first


def test_same_report_twice_in_one_call(doc: str, red: list[str]):
    """тот же отчёт дважды в одном вызове: второй пропускается"""
    result = import_reports(None, [marked_report(doc, red), marked_report(doc, red)], today=TODAY)
    assert [outcome.status for outcome in result.outcomes] == ["added", "skipped"]
    assert len(result.calibration.reports) == 1


def test_edited_version_adds_samples(doc: str, red: list[str]):
    """отчёт по правленой версии документа добавляет образцы рядом с прежними"""
    first = import_reports(None, [marked_report(doc, red)], today=TODAY).calibration
    edited = doc + "\n\nПосле проверки автор дописал короткий вывод о складском учёте и сменил формулировку итога."
    second = import_reports(first, [marked_report(edited, red)], today=TODAY).calibration
    assert second.samples[: len(first.samples)] == first.samples
    assert len(second.samples) > len(first.samples)
    assert len(second.reports) == 2


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(
            {"marked": ["Заново выписанный кусок, которого не было в прежней разметке отчёта."]}, id="другие куски"
        ),
        pytest.param({"system_version": "6.0"}, id="новая версия системы"),
    ],
)
def test_other_markup_for_same_version_needs_replace(doc: str, red: list[str], change: dict[str, Any]):
    """другая разметка для той же версии текста: без replace — ошибка, с replace — замена"""
    first = import_reports(None, [marked_report(doc, red)], today=TODAY).calibration
    changed = marked_report(doc, change.get("marked", red), system_version=change.get("system_version"))
    with pytest.raises(CalibrationError, match="--replace"):
        import_reports(first, [changed], today=TODAY)
    replaced = import_reports(first, [changed], today=TODAY, replace=True)
    assert [outcome.status for outcome in replaced.outcomes] == ["replaced"]
    assert len(replaced.calibration.reports) == 1
    assert len(replaced.calibration.samples) == len(first.samples)


def test_share_report_repeat_and_conflict(doc: str):
    """отчёт с одной долей: повтор пропускается, другая доля для той же версии требует replace"""
    first = import_reports(None, [ReportInput(name="a", text=doc, share=30)], today=TODAY).calibration
    assert len(first.documents) == 1
    assert not import_reports(first, [ReportInput(name="a", text=doc, share=30.0)], today=TODAY).changed
    with pytest.raises(CalibrationError):
        import_reports(first, [ReportInput(name="a", text=doc, share=12)], today=TODAY)
    replaced = import_reports(first, [ReportInput(name="a", text=doc, share=12)], today=TODAY, replace=True)
    assert [document.share for document in replaced.calibration.documents] == [12]


def test_import_order_does_not_matter(doc: str, red: list[str]):
    """порядок импорта не меняет модель: обучение идёт заново от модели по умолчанию"""
    edited = doc + "\n\nПосле проверки автор дописал короткий вывод о складском учёте и сменил формулировку итога."
    a, b = marked_report(doc, red), marked_report(edited, [])
    forward = import_reports(import_reports(None, [a], today=TODAY).calibration, [b], today=TODAY).calibration
    backward = import_reports(import_reports(None, [b], today=TODAY).calibration, [a], today=TODAY).calibration
    assert forward.model.bias == pytest.approx(backward.model.bias, abs=1e-9)
    assert forward.model.weights == pytest.approx(backward.model.weights, abs=1e-9)
    assert forward.model.threshold == pytest.approx(backward.model.threshold)
    assert sorted(map(str, forward.samples)) == sorted(map(str, backward.samples))


def test_empty_import_keeps_previous(doc: str, red: list[str]):
    """вызов без отчётов ничего не меняет"""
    first = import_reports(None, [marked_report(doc, red)], today=TODAY).calibration
    result = import_reports(first, [], today=TODAY)
    assert not result.changed
    assert result.calibration == first


def test_old_samples_without_document_kept(doc: str, red: list[str]):
    """образцы файла прежней версии (без отпечатка) не теряются ни при импорте, ни при замене"""
    old = Calibration(model=DEFAULT_MODEL, samples=[Sample(f=[0.1] * 7, y=1), Sample(f=[0.0] * 7, y=0)])
    added = import_reports(old, [marked_report(doc, red)], today=TODAY).calibration
    assert added.samples[:2] == old.samples
    other = marked_report(doc, ["Заново выписанный кусок, которого не было в прежней разметке отчёта."])
    replaced = import_reports(added, [other], today=TODAY, replace=True).calibration
    assert replaced.samples[:2] == old.samples


# ─── перенос на новый документ ──────────────────────────────────────────


def separable(doc: str, count: int = 6) -> list[Sample]:
    """Образцы одного документа: у ИИ-фрагментов высокая плотность примет, у человеческих низкая."""
    return [
        Sample(f=[0.9 if index % 2 else 0.05, 0.5, 0.3, 0.0, 0.0, 0.0, 0.0], y=1 if index % 2 else 0, doc=doc)
        for index in range(count)
    ]


def test_transfer_needs_two_documents(doc: str, red: list[str]):
    """один документ — перенос не проверен"""
    samples = import_reports(None, [marked_report(doc, red)], today=TODAY).calibration.samples
    transfer = transfer_check(samples)
    assert (transfer.documents, transfer.accuracy) == (1, None)


def test_transfer_on_held_out_documents():
    """два документа и больше: каждый откладывается целиком, точность считается на отложенных"""
    transfer = transfer_check(separable("a") + separable("b") + separable("c"))
    assert transfer.documents == 3
    assert transfer.accuracy is not None and transfer.accuracy > 0.5
    assert transfer.default_accuracy is not None


def test_transfer_counts_old_samples_as_one_group():
    """образцы без отпечатка — отдельная группа документа"""
    old = [sample.model_copy(update={"doc": None}) for sample in separable("x")]
    assert transfer_check(old + separable("a")).documents == 2


def test_transfer_folds_capped(monkeypatch: pytest.MonkeyPatch):
    """много документов — не больше TRANSFER_FOLDS обучений"""
    calls: list[int] = []
    real_fit = cal.fit

    def counting_fit(samples: list[Sample], start: Model = DEFAULT_MODEL) -> Model:
        calls.append(len(samples))
        return real_fit(samples, start)

    monkeypatch.setattr(cal, "fit", counting_fit)
    samples = [sample for index in range(25) for sample in separable(f"doc{index:02}", 2)]
    assert transfer_check(samples).documents == 25
    assert len(calls) == cal.TRANSFER_FOLDS


def test_transfer_without_both_classes_held_out():
    """в отложенных фрагментах один класс — перенос не проверен"""
    samples = [Sample(f=[0.1] * 7, y=0, doc="a"), Sample(f=[0.9] * 7, y=1, doc="a"), Sample(f=[0.1] * 7, y=0, doc="b")]
    assert transfer_check(samples).accuracy is None


# ─── запись файла ───────────────────────────────────────────────────────


def calibration_with(samples: int) -> CalibrationFile:
    return empty_calibration_file().put(
        "antiplagiat", Calibration(model=DEFAULT_MODEL, samples=[Sample(f=[0.1] * 7, y=1)] * samples)
    )


def test_write_creates_and_replaces(tmp_path: Path):
    """запись создаёт файл и заменяет его целиком; временных файлов не остаётся"""
    path = tmp_path / "cal.json"
    write_file(path, calibration_with(1), None)
    first = path.read_bytes()
    assert first == serialize(calibration_with(1))
    write_file(path, calibration_with(2), first)
    assert len(json.loads(path.read_text(encoding="utf-8"))["samples"]) == 2
    assert [p.name for p in tmp_path.iterdir()] == ["cal.json"]


def test_write_detects_concurrent_change(tmp_path: Path):
    """файл изменился после чтения — CalibrationConflict, чужая запись не теряется"""
    path = tmp_path / "cal.json"
    path.write_bytes(serialize(calibration_with(3)))
    with pytest.raises(CalibrationConflict):
        write_file(path, calibration_with(1), serialize(calibration_with(2)))
    assert path.read_bytes() == serialize(calibration_with(3))
    with pytest.raises(CalibrationConflict):
        write_file(tmp_path / "new.json", calibration_with(1), b"{}")


def test_failed_replace_keeps_old_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """сбой при замене: прежний файл байт в байт, временный удалён"""
    path = tmp_path / "cal.json"
    old = serialize(calibration_with(2))
    path.write_bytes(old)

    def broken_replace(src: str, dst: Path) -> None:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(cal.os, "replace", broken_replace)
    with pytest.raises(PermissionError):
        write_file(path, calibration_with(5), old)
    assert path.read_bytes() == old
    assert [p.name for p in tmp_path.iterdir()] == ["cal.json"]


def test_read_old_file_without_reports(tmp_path: Path):
    """файл прежней версии без reports и отпечатков читается, образцы на месте"""
    path = tmp_path / "old.json"
    old = {"version": 1, "model": GOOD_MODEL, "samples": [{"f": [0.1] * 7, "y": 1}], "documents": []}
    path.write_text(json.dumps(old), encoding="utf-8")
    loaded = read_file(path)
    assert loaded.raw == path.read_bytes()
    assert loaded.data is not None
    calibration = loaded.data.get("antiplagiat")
    assert calibration is not None
    assert calibration.reports == []
    assert calibration.samples[0].doc is None


def test_read_missing_file(tmp_path: Path):
    """файла нет — пустой результат, а не ошибка"""
    assert read_file(tmp_path / "нет.json") == cal.Loaded(None, None)


def test_new_fields_ignored_by_old_reader(doc: str, red: list[str]):
    """старые версии aiw-ru читают новый файл: прежние поля на месте, новые не мешают модели без них"""
    calibration = import_reports(None, [marked_report(doc, red)], today=TODAY).calibration
    data = json.loads(serialize(empty_calibration_file().put("antiplagiat", calibration)))
    assert {"version", "model", "samples", "documents", "systems"} <= set(data)
    old_sample = {key: value for key, value in data["samples"][0].items() if key in ("f", "y")}
    assert Sample.model_validate(old_sample).doc is None

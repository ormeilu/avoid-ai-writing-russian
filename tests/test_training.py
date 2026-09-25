"""Тесты scripts/llmtrace.py, scripts/train.py и карточки модели (scripts/hub.py).

Нужны LightGBM и scikit-learn из группы train; без них тесты пропускаются.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("lightgbm")
pytest.importorskip("sklearn")
pytest.importorskip("huggingface_hub")

import hub
import llmtrace
import train
from aiw_ru import analyze
from aiw_ru.features import FEATURE_NAMES, describe, features

FIXTURES = llmtrace.ROOT / "tests" / "fixtures" / "corpus"


def test_roc_auc():
    assert llmtrace.roc_auc([3, 4], [1, 2]) == 1
    assert llmtrace.roc_auc([1, 2], [1, 2]) == 0.5
    assert llmtrace.roc_auc([1], [2]) == 0
    assert llmtrace.roc_auc([], [1]) == 0


def test_metrics_at_threshold():
    c = llmtrace.at_threshold([50, 10], [60, 5], 40)
    assert [c[k] for k in ("accuracy", "precision", "recall", "f1", "fpr")] == [0.5] * 5
    best = llmtrace.best_f1([90, 80], [10, 20])
    assert best["f1"] == 1 and best["threshold"] > 20


def test_spans():
    assert llmtrace.union_length([(0, 5), (3, 8), (10, 12)]) == 10
    assert llmtrace.overlap_length([(0, 5), (10, 12)], [(3, 11)]) == 3
    assert llmtrace.overlap_length([], [(0, 4)]) == 0


def test_features():
    text = "И это, и то. Ёж"
    x = features(text)
    assert len(x) == len(FEATURE_NAMES)
    named = dict(zip(FEATURE_NAMES, x, strict=True))
    assert named["word:и"] == 40  # два «и» на пять слов
    assert named["comma"] == 20
    assert features(text, analyze(text, "general")) == x


def test_features_rules():
    text = (FIXTURES / "ai" / "blog.md").read_text(encoding="utf-8")
    named = dict(zip(FEATURE_NAMES, features(text), strict=True))
    assert named["rule:tier1"] == 1


def row(label: str, name: str, **extra) -> str:
    text = (FIXTURES / label / f"{name}.md").read_text(encoding="utf-8")
    prompt = "create" if label == "ai" else None
    return json.dumps(
        {"lang": "ru", "label": label, "model": "m", "data_type": name, "prompt_type": prompt, "text": text, **extra},
        ensure_ascii=False,
    )


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    rows = [row(label, name) for label in ("human", "ai") for name in ("blog", "email", "vak")]
    for split in ("train", "valid", "test"):
        (tmp_path / f"ru-classification-{split}.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    ai = (FIXTURES / "ai" / "vak.md").read_text(encoding="utf-8")
    detection = [row("human", "vak", ai_char_intervals=[]), row("ai", "vak", ai_char_intervals=[[0, len(ai)]])]
    (tmp_path / "ru-detection-test.jsonl").write_text("\n".join(detection) + "\n", encoding="utf-8")
    return tmp_path


def test_eval_spans_and_cache(corpus: Path, capsys: pytest.CaptureFixture[str]):
    llmtrace.main(["eval", "--data", str(corpus), "--jobs", "2", "--rule", "tier1"])
    out = capsys.readouterr().out
    assert "ROC AUC" in out and "Правила" in out
    assert len(list(corpus.glob("detector-scan-general-classification-test-*.jsonl"))) == 1
    llmtrace.main(["spans", "--data", str(corpus), "--jobs", "1"])
    assert "По знакам" in capsys.readouterr().out


def test_missing_split(corpus: Path):
    with pytest.raises(SystemExit, match="llmtrace.py fetch"):
        llmtrace.main(["eval", "--data", str(corpus), "--split", "nope"])


def test_train_writes_model_card(corpus: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch):
    """Обучение на выборке пишет модель, метрики и карточку и ничего не выкладывает."""
    pushed = []
    monkeypatch.setattr(train, "push", lambda *a: pushed.append(a))
    train.main(["--data", str(corpus), "--jobs", "2", "--rounds", "5", "--limit", "6"])
    out = capsys.readouterr().out
    assert "LightGBM" in out and "Модель на test" in out and "не выкладывается" in out
    assert not pushed
    folder = corpus / "lightgbm"
    assert {p.name for p in folder.iterdir()} == {"model.txt", "features.json", "metrics.json", "README.md"}
    spec = json.loads((folder / "features.json").read_text(encoding="utf-8"))
    assert spec["names"] == list(FEATURE_NAMES)
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    card = (folder / "README.md").read_text(encoding="utf-8")
    assert card.startswith("---\n") and "model-index:" in card and hub.DATASET in card
    assert "library_name: lightgbm" in card and "## Ограничения" in card
    assert f"\\\\({metrics['test']['accuracy']:.3f}\\\\)" in card


def fake_metrics() -> dict:
    """metrics.json в том виде, в каком его пишет train.py, с игрушечными числами."""
    return {
        "created": "2026-01-01",
        "aiw_ru_version": "1.0.0",
        "git_commit": "abc",
        "features_version": 1,
        "environment": {"python": "3.14.0", "lightgbm": "4.7.0"},
        "dataset": {"repo": hub.DATASET, "revision": "abc123", "limit": 0, "train": 10, "valid": 5, "test": 5},
        "params": {
            "seed": 1,
            "deterministic": True,
            "force_row_wise": True,
            "bagging_freq": 1,
            "best_iteration": 3,
            "rounds": 5,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "lambda_l2": 1.0,
            "train_seconds": 1.0,
            "size_kb": 10.0,
            "cpu": "arm64",
            "features": 100,
        },
        "test": {
            "accuracy": 0.9,
            "roc_auc": 0.95,
            "roc_auc_created": 0.97,
            "macro": {"precision": 0.9, "recall": 0.9, "f1": 0.9},
            "classes": {k: {"precision": 0.9, "recall": 0.9, "f1": 0.9, "support": 5} for k in ("human", "ai")},
        },
        "valid": {"accuracy": 0.88, "roc_auc": 0.94},
        "detector": {
            "roc_auc": 0.6,
            "roc_auc_created": 0.59,
            "threshold": 40,
            "accuracy": 0.5,
            "macro": {"precision": 0.6, "recall": 0.5, "f1": 0.4},
            "classes": {k: {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 5} for k in ("human", "ai")},
        },
        "genres": [{"genre": "news", "human": 2, "ai": 3, "accuracy": 0.8, "roc_auc": None, "detector_roc_auc": 0.5}],
        "lengths": [{"bucket": "0–49", "n": 5, "accuracy": 0.8, "roc_auc": 0.9}],
        "prompt_types": [{"prompt_type": "create", "n": 3, "recall": 1.0}],
        "generators": [{"model": "gpt-4o", "n": 3, "recall": 1.0}],
        "features": [
            {"name": n, "description": describe(n), "gain_share": 0.01, "mean_human": 1.0, "mean_ai": 2.0}
            for n in FEATURE_NAMES
        ],
    }


def test_card_metadata():
    """model-index: итог на test, по классам, по жанрам и valid."""
    metrics = fake_metrics()
    results = hub.eval_results(metrics)
    names = {r.metric_name for r in results}
    assert {"Accuracy", "ROC AUC", "Macro F1", "Recall (AI)", "Accuracy (news)"} <= names
    assert "ROC AUC (news)" not in names  # в жанре один класс — AUC нет
    assert {r.dataset_split for r in results} == {"test", "validation"}
    text = str(hub.card(metrics, "someone/model"))
    assert "inference: false" in text and "pipeline_tag: text-classification" in text


def test_hub_math():
    """Hub рисует строчные формулы в \\\\(…\\\\); код не трогаем"""
    assert hub.num(0.9431) == "$0.943$" and hub.pct(0.165) == "$16.5\\%$"
    assert hub.count(52521) == "$52\\,521$" and hub.count(3000) == "$3000$"
    text = "ROC AUC $0.943$, `$x$`\n```bash\necho $HOME $PATH\n```"
    assert hub.hub_math(text) == "ROC AUC \\\\(0.943\\\\), `$x$`\n```bash\necho $HOME $PATH\n```"
    card = str(hub.card(fake_metrics(), "someone/model"))
    assert "\\\\(0.950\\\\)" in card and "$0.950$" not in card
    assert "0,95" not in card and "ai-generated-text-detection" in card


def test_card_describes_top_features():
    """карточка объясняет пятнадцать главных признаков, отчёт — все"""
    metrics = fake_metrics()
    card = str(hub.card(metrics, "someone/model"))
    assert describe(FEATURE_NAMES[0]) in card and describe(FEATURE_NAMES[20]) not in card
    assert "arxiv.org/abs/2509.21269" in card and hub.REPORT_PATH in card


def test_report_is_reproducible(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """отчёт в репозитории: ревизия набора, команды, версии, все признаки"""
    monkeypatch.setattr(hub, "ROOT", tmp_path)
    path = hub.write_report(fake_metrics(), "someone/model")
    text = path.read_text(encoding="utf-8")
    assert path == tmp_path / hub.REPORT_PATH
    assert "`abc123`" in text and "uv sync --group train --locked" in text
    assert "scripts/train.py --rounds 5 --leaves 31 --rate 0.05 --no-push" in text
    assert "--split train" in text and "--split valid" in text and "--split test" in text
    assert "| lightgbm | 4.7.0 |" in text
    assert all(f"`{n}`" in text for n in FEATURE_NAMES)
    assert "$0.950$" in text and "\\(" not in text  # на GitHub формулы в $…$
    assert "незакоммиченными правками" not in text
    saved = json.loads((tmp_path / hub.REPORT_METRICS).read_text(encoding="utf-8"))
    assert saved["dataset"]["revision"] == "abc123"

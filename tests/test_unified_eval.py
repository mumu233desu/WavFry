"""Tests for the unified evaluation utilities and metrics NaN defense."""

import numpy as np
import pandas as pd
import pytest
import torch

from wavfry.eval.unified import build_test_manifest, f1_scan_best_threshold, resolve_frozen_threshold
from wavfry.train.metrics import compute_metrics


# ── build_test_manifest ──────────────────────────────────────────────────

def _manifest():
    return pd.DataFrame({
        "dataset_id": ["cs_dialogue", "cs_dialogue", "cs_dialogue", "ucla_mazatec", "cs_dialogue"],
        "batch_split": ["test", "test", "train", "test", "test"],
        "phone_type": ["vowel", "sonorant", "vowel", "vowel", "vowel"],
        "label_creaky": [1.0, 0.0, 1.0, 1.0, float("nan")],
        "file_key": ["a", "b", "c", "d", "e"],
        "start_s": [0.0, 1.0, 2.0, 3.0, 4.0],
    })


def test_build_test_manifest_filters_dataset_split_phone_and_nan():
    out = build_test_manifest(_manifest(), task="creaky")
    # keeps: cs_dialogue + test + vowel + labeled → rows 0 only
    assert len(out) == 1
    assert out.iloc[0]["file_key"] == "a"


def test_build_test_manifest_legacy_split_column():
    df = _manifest().drop(columns=["batch_split"])
    df["split"] = ["test", "val", "train", "test", "test"]
    out = build_test_manifest(df, task="creaky")
    assert list(out["file_key"]) == ["a"]


def test_build_test_manifest_no_split_column_raises():
    df = _manifest().drop(columns=["batch_split"])
    with pytest.raises(KeyError):
        build_test_manifest(df)


# ── resolve_frozen_threshold ─────────────────────────────────────────────

def test_resolve_frozen_threshold_returns_value():
    assert resolve_frozen_threshold({"best_threshold": 0.65}) == 0.65


def test_resolve_frozen_threshold_fails_fast_on_missing():
    with pytest.raises(RuntimeError):
        resolve_frozen_threshold({}, "dummy.pt")


def test_resolve_frozen_threshold_fails_fast_on_none():
    with pytest.raises(RuntimeError):
        resolve_frozen_threshold({"best_threshold": None})


# ── f1_scan_best_threshold ───────────────────────────────────────────────

def test_f1_scan_ignores_nan_labels():
    logits = torch.tensor([3.0, 3.0, -3.0, -3.0, 2.0])
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0, float("nan")])
    best_t, best_f1 = f1_scan_best_threshold(logits, labels)
    # Perfectly separable non-NaN rows → F1 1.0 achievable
    assert best_f1 == pytest.approx(1.0)


def test_f1_scan_all_nan_raises():
    logits = torch.tensor([1.0, -1.0])
    labels = torch.tensor([float("nan"), float("nan")])
    with pytest.raises(ValueError):
        f1_scan_best_threshold(logits, labels)


# ── compute_metrics NaN defense ──────────────────────────────────────────

def test_compute_metrics_filters_nan_labels():
    logits = torch.tensor([5.0, 5.0, -5.0, -5.0, 1.0])
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0, float("nan")])
    m = compute_metrics(logits, labels)
    # Without filtering, the NaN row int-casts to -2147483648 and counts as a
    # negative: accuracy would be 4/5 either way here, but cm sums must equal
    # the number of VALID rows only.
    assert sum(sum(row) for row in m["cm"]) == 4


def test_compute_metrics_new_roc_auc_key():
    logits = torch.tensor([5.0, -5.0])
    labels = torch.tensor([1.0, 0.0])
    m = compute_metrics(logits, labels)
    assert m["roc_auc"] == pytest.approx(1.0)


def test_compute_metrics_all_nan_returns_zeroed():
    logits = torch.tensor([1.0, -1.0])
    labels = torch.tensor([float("nan"), float("nan")])
    m = compute_metrics(logits, labels)
    assert m["f1"] == 0.0
    assert np.isnan(m["auc_pr"])

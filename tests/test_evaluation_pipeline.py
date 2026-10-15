"""
Unit tests for unified evaluation pipeline, incremental CSV caching, and decoupled data schemas.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

from wavfry.eval.cache_manager import (
    build_interval_key,
    attach_interval_keys,
    IncrementalCSVCache,
    CANONICAL_TIERS,
)
from scripts.baseline_comparison.evaluate_qut_baselines import compute_binary_metrics


def test_build_interval_key_across_datasets():
    key_cs = build_interval_key("cs_dialogue", "conv_01", 1.05, 1.25)
    key_meli = build_interval_key("meli", "conv_01", 1.05, 1.25)
    key_jp = build_interval_key("ita_japanese", "conv_01", 1.05, 1.25)

    assert len({key_cs, key_meli, key_jp}) == 3
    assert key_cs.startswith("cs_dialogue::conv_01::")
    assert key_meli.startswith("meli::conv_01::")


def test_compute_binary_metrics():
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.1, 0.2, 0.85, 0.3, 0.7, 0.4])

    m = compute_binary_metrics(y_true, y_prob, threshold=0.5)
    assert m["Accuracy"] == 1.0
    assert m["F1-Score"] == 1.0
    assert m["PR-AUC"] == 1.0
    assert m["ROC-AUC"] == 1.0


def test_compute_binary_metrics_nan_handling():
    y_true = np.array([1, np.nan, 0, 1])
    y_prob = np.array([0.8, 0.5, np.nan, 0.9])

    m = compute_binary_metrics(y_true, y_prob, threshold=0.5)
    assert "Accuracy" in m
    assert "F1-Score" in m
    assert not np.isnan(m["F1-Score"])


def test_incremental_cache_append_and_retrieve(tmp_path: Path):
    cache_path = tmp_path / "test_incremental_cache.csv"
    cache = IncrementalCSVCache(cache_path)

    # Initial manifest
    m1 = pd.DataFrame({
        "dataset_id": ["cs_dialogue", "cs_dialogue"],
        "file_key": ["audio_1", "audio_2"],
        "start_s": [0.0, 1.0],
        "end_s": [0.5, 1.5],
    })
    cached, missing = cache.partition_manifest(m1)
    assert cached.empty
    assert len(missing) == 2

    # Append computed features
    computed = missing.copy()
    computed["metric_val"] = [0.85, 0.92]
    cache.append_records(computed)

    # Re-partition: should be 100% hits
    cached_after, missing_after = cache.partition_manifest(m1)
    assert len(cached_after) == 2
    assert missing_after.empty

    # Expand manifest with new dataset (e.g. meli)
    m2 = pd.DataFrame({
        "dataset_id": ["cs_dialogue", "meli"],
        "file_key": ["audio_1", "audio_new"],
        "start_s": [0.0, 0.2],
        "end_s": [0.5, 0.8],
    })
    cached_m2, missing_m2 = cache.partition_manifest(m2)
    assert len(cached_m2) == 1
    assert len(missing_m2) == 1
    assert missing_m2.iloc[0]["dataset_id"] == "meli"

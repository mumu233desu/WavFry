"""
Unit tests for IncrementalCSVCache and multi-dataset interval primary keys.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from wavfry.eval.cache_manager import (
    build_interval_key,
    attach_interval_keys,
    IncrementalCSVCache,
)


def test_build_interval_key_uniqueness():
    # Same file_key and timestamps in different datasets must have different interval_keys
    k1 = build_interval_key("cs_dialogue", "utt_001", 0.5, 1.2)
    k2 = build_interval_key("meli", "utt_001", 0.5, 1.2)
    k3 = build_interval_key("cs_dialogue", "utt_001", 0.5, 1.2001)

    assert k1 != k2
    assert k1 != k3
    assert k1 == "cs_dialogue::utt_001::0.5000_1.2000"


def test_attach_interval_keys():
    df = pd.DataFrame({
        "dataset_id": ["cs_dialogue", "meli"],
        "file_key": ["u1", "u1"],
        "start_s": [0.1, 0.2],
        "end_s": [0.3, 0.4],
    })
    attached = attach_interval_keys(df)
    assert "interval_key" in attached.columns
    assert attached["interval_key"].iloc[0] == "cs_dialogue::u1::0.1000_0.3000"
    assert attached["interval_key"].iloc[1] == "meli::u1::0.2000_0.4000"


def test_incremental_csv_cache_lifecycle(tmp_path: Path):
    cache_file = tmp_path / "test_cache.csv"
    cache = IncrementalCSVCache(cache_file)

    manifest_v1 = pd.DataFrame({
        "dataset_id": ["cs_dialogue", "cs_dialogue"],
        "file_key": ["audio1", "audio2"],
        "start_s": [0.0, 1.0],
        "end_s": [0.5, 1.5],
    })

    cached_v1, missing_v1 = cache.partition_manifest(manifest_v1)
    assert cached_v1.empty
    assert len(missing_v1) == 2

    # Append simulated computed records
    computed_v1 = missing_v1.copy()
    computed_v1["feature_x"] = [10.5, 20.3]
    cache.append_records(computed_v1)

    assert cache_file.exists()
    assert len(cache.df) == 2

    # Verify matching
    matched_v1 = cache.get_matched(manifest_v1)
    assert len(matched_v1) == 2
    assert list(matched_v1["feature_x"]) == [10.5, 20.3]

    # Test increment: manifest expands with 1 new sample (e.g. from meli)
    manifest_v2 = pd.DataFrame({
        "dataset_id": ["cs_dialogue", "cs_dialogue", "meli"],
        "file_key": ["audio1", "audio2", "audio1"],
        "start_s": [0.0, 1.0, 0.0],
        "end_s": [0.5, 1.5, 0.5],
    })

    cached_v2, missing_v2 = cache.partition_manifest(manifest_v2)
    assert len(cached_v2) == 2  # audio1 and audio2 hit cache
    assert len(missing_v2) == 1  # meli/audio1 is missing
    assert missing_v2.iloc[0]["dataset_id"] == "meli"

    # Append missing
    computed_v2 = missing_v2.copy()
    computed_v2["feature_x"] = [99.9]
    cache.append_records(computed_v2)

    assert len(cache.df) == 3
    matched_v2 = cache.get_matched(manifest_v2)
    assert len(matched_v2) == 3
    assert list(matched_v2["feature_x"]) == [10.5, 20.3, 99.9]

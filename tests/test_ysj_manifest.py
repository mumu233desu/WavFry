import json
import pytest
import pandas as pd
from pathlib import Path

from wavfry.data.manifest import (
    _speaker_id,
    _language_from_file_key,
    adapter_sharded_textgrid_cache,
    ADAPTERS,
)
from wavfry.data.splits import isolate_split_speakers


def test_wavfry_speaker_and_language_token_extraction():
    # YSJ file tokens
    assert _speaker_id("M_00_JP_JA_JA-JP_C0023_S0_12") == "C0023"
    assert _speaker_id("X_00_JP_JA_JA-JP_C0023_S0_12") == "C0023"
    assert _speaker_id("F_00_JP_JA_JA-JP_C0001_S0_1") == "C0001"
    # Legacy CS-Dialogue tokens
    assert _speaker_id("F_23_Hebei_ZH_ZH-CN_U1001_S0_1") == "U1001"

    # Language extraction
    assert _language_from_file_key("M_00_JP_JA_JA-JP_C0023_S0_12") == "ja"
    assert _language_from_file_key("F_00_JP_JPN_JA-JP_C0001_S0_1") == "ja"
    assert _language_from_file_key("F_23_Hebei_ZH_ZH-CN_U1001_S0_1") == "zh"
    assert _language_from_file_key("M_20_London_EN_EN-GB_U2001_S0_1") == "en"


def test_adapter_registration():
    assert "sharded_textgrid_cache" in ADAPTERS
    assert ADAPTERS["sharded_textgrid_cache"] is adapter_sharded_textgrid_cache


def test_adapter_sharded_textgrid_cache_with_manifest_and_filtering(tmp_path: Path):
    audio_dir = tmp_path / "audio"
    tg_dir = tmp_path / "textgrid"
    cache_dir = tg_dir / "_breathy_cache"
    batches_dir = cache_dir / "batches"
    audio_dir.mkdir(parents=True)
    tg_dir.mkdir(parents=True)
    batches_dir.mkdir(parents=True)

    # 1. Create dummy audio and textgrid files
    f1 = "M_00_JP_JA_JA-JP_C0023_S0_1"
    f2 = "X_00_JP_JA_JA-JP_C0023_S0_2"
    f3_noisy = "F_00_JP_JA_JA-JP_C0001_S0_1"

    for f_stem in [f1, f2, f3_noisy]:
        (audio_dir / f"{f_stem}.flac").write_bytes(b"dummy_audio")
        (tg_dir / f"{f_stem}.TextGrid").write_text("dummy_textgrid", encoding="utf-8")

    # 2. Create manifest_slices.csv
    manifest_csv = tmp_path / "manifest_slices.csv"
    manifest_csv.write_text(
        "slice_id,speaker_uid,gender,smr_db,mos_score\n"
        f"{f1},C0023,M,15.5,4.0\n"
        f"{f2},C0023,X,14.0,3.8\n"
        f"{f3_noisy},C0001,F,5.0,2.5\n",  # SMR 5.0 < 10.0 threshold
        encoding="utf-8",
    )

    # 3. Create sampling_state.json and batch annotations
    sampling_state = {
        "version": 2,
        "task": "breathy",
        "batches": [
            {"batch_id": 1, "name": "Batch 1", "split": "train"},
        ],
    }
    (cache_dir / "sampling_state.json").write_text(json.dumps(sampling_state), encoding="utf-8")

    batch_1_dir = batches_dir / "batch_1"
    batch_1_dir.mkdir(parents=True)
    batch_1_data = {
        "version": 1,
        "schema": "direct_breathy_annotation",
        "files": {
            f1: {
                "annotations": {
                    "0.1000|0.3000|a": "breathy",
                    "0.4000|0.6000|i": "modal",
                }
            },
            f2: {
                "annotations": {
                    "0.2000|0.4000|o": "breathy",
                }
            },
            f3_noisy: {
                "annotations": {
                    "0.1000|0.3000|e": "breathy",
                }
            },
        },
    }
    (batch_1_dir / "manual_creaky_annotations.json").write_text(json.dumps(batch_1_data), encoding="utf-8")

    # Test Case A: Without SMR or gender filtering
    cfg_a = {
        "dataset_id": "ysj_japanese",
        "domain": "spontaneous_podcast",
        "adapter_params": {
            "audio_dir": str(audio_dir),
            "textgrid_dir": str(tg_dir),
            "cache_dir": str(cache_dir),
            "manifest_csv": str(manifest_csv),
            "min_smr_db": None,
            "exclude_gender_x": False,
        },
        "label_schema": {"creaky": False, "breathy": True},
    }
    records_a = adapter_sharded_textgrid_cache(cfg_a)
    assert len(records_a) == 4  # 2 from f1 + 1 from f2 + 1 from f3_noisy
    assert {r["speaker_id"] for r in records_a} == {"C0023", "C0001"}
    assert {r["gender"] for r in records_a} == {"M", "X", "F"}
    assert all(r["batch_split"] == "train" for r in records_a)

    # Test Case B: With SMR >= 10.0 filtering (f3_noisy has 5.0 dB, must be excluded)
    cfg_b = {
        "dataset_id": "ysj_japanese",
        "domain": "spontaneous_podcast",
        "adapter_params": {
            "audio_dir": str(audio_dir),
            "textgrid_dir": str(tg_dir),
            "cache_dir": str(cache_dir),
            "manifest_csv": str(manifest_csv),
            "min_smr_db": 10.0,
            "exclude_gender_x": False,
        },
        "label_schema": {"creaky": False, "breathy": True},
    }
    records_b = adapter_sharded_textgrid_cache(cfg_b)
    assert len(records_b) == 3
    assert all(r["speaker_id"] == "C0023" for r in records_b)
    assert {r["file_key"] for r in records_b} == {f1, f2}

    # Test Case C: With exclude_gender_x=True (f2 has gender X, must be excluded)
    cfg_c = {
        "dataset_id": "ysj_japanese",
        "domain": "spontaneous_podcast",
        "adapter_params": {
            "audio_dir": str(audio_dir),
            "textgrid_dir": str(tg_dir),
            "cache_dir": str(cache_dir),
            "manifest_csv": str(manifest_csv),
            "min_smr_db": 10.0,
            "exclude_gender_x": True,
        },
        "label_schema": {"creaky": False, "breathy": True},
    }
    records_c = adapter_sharded_textgrid_cache(cfg_c)
    assert len(records_c) == 2
    assert all(r["file_key"] == f1 for r in records_c)
    assert all(r["gender"] == "M" for r in records_c)


def test_ysj_c_speaker_zero_leakage_isolation():
    train_df = pd.DataFrame([
        {"dataset_id": "ysj_japanese", "speaker_id": "C0023", "label": 1.0},
        {"dataset_id": "ysj_japanese", "speaker_id": "C0001", "label": 0.0},
    ])
    # Leaked C0023 in val_df
    val_df = pd.DataFrame([
        {"dataset_id": "ysj_japanese", "speaker_id": "C0023", "label": 1.0},  # Overlap!
        {"dataset_id": "ysj_japanese", "speaker_id": "C0086", "label": 0.0},
    ])
    test_df = pd.DataFrame([
        {"dataset_id": "ysj_japanese", "speaker_id": "C0099", "label": 1.0},
    ])

    tr_clean, val_clean, test_clean = isolate_split_speakers(train_df, val_df, test_df)

    # C0023 must be purged from val_clean
    assert set(tr_clean["speaker_id"]) == {"C0023", "C0001"}
    assert set(val_clean["speaker_id"]) == {"C0086"}
    assert set(test_clean["speaker_id"]) == {"C0099"}
    assert len(set(tr_clean["speaker_id"]) & set(val_clean["speaker_id"])) == 0

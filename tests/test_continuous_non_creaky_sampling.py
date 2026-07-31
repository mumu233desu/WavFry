import pandas as pd

from wavfry.data.continuous_non_creaky_sampling import (
    ContinuousNonCreakyWindowGenerator,
)
from wavfry.data.dataset import CreakyVowelDataset


def _row(start, end, phone_type="vowel", label=0):
    return {
        "file_key": "F_20_Test_EN_ZH-CN_U1001_S0_1",
        "audio_path": "test.wav",
        "tg_path": "test.TextGrid",
        "start_s": start,
        "end_s": end,
        "phone": "a" if phone_type != "silence" else "SP",
        "phone_type": phone_type,
        "speaker_id": "U1001",
        "language": "en",
        "label": label,
        "annotation_type": "full",
        "batch_id": None,
    }


def _config(**overrides):
    config = {
        "enabled": True,
        "extra_sample_ratio": 1.0,
        "min_duration_ms": 150,
        "target_duration_ms": 300,
        "max_duration_ms": 500,
        "max_intervals": 4,
        "max_sp_duration_ms": 120,
        "max_sp_ratio": 0.35,
        "allow_sp_only": False,
    }
    config.update(overrides)
    return config


def test_windows_are_deterministic_bounded_and_cross_interval_boundaries():
    df = pd.DataFrame([
        _row(0.00, 0.20),
        _row(0.20, 0.40),
        _row(0.40, 0.60),
        _row(0.60, 0.80),
    ])
    generator = ContinuousNonCreakyWindowGenerator(df, _config(), seed=42)

    first = generator.generate(epoch=3)
    second = generator.generate(epoch=3)
    assert first.to_dict("records") == second.to_dict("records")
    assert len(first) == len(df)
    assert first["start_s"].ge(0.0).all()
    assert first["end_s"].le(0.8).all()
    assert first["source_interval_count"].between(2, 4).all()
    assert first["context_padding_ms"].eq(0.0).all()
    assert generator.shuffle_seed_for_epoch(3) == generator.shuffle_seed_for_epoch(3)
    assert generator.shuffle_seed_for_epoch(3) != generator.shuffle_seed_for_epoch(4)


def test_runs_do_not_cross_creaky_or_time_gaps():
    df = pd.DataFrame([
        _row(0.00, 0.20),
        _row(0.20, 0.40, label=1),
        _row(0.40, 0.60),
        _row(0.70, 0.90),
    ])
    generator = ContinuousNonCreakyWindowGenerator(df, _config(), seed=42)
    assert generator.runs == []
    assert generator.generate(epoch=1).empty


def test_sp_limits_apply_to_final_window():
    df = pd.DataFrame([
        _row(0.00, 0.20),
        _row(0.20, 0.70, phone_type="silence"),
        _row(0.70, 0.90),
    ])
    generator = ContinuousNonCreakyWindowGenerator(
        df,
        _config(max_sp_duration_ms=20, max_sp_ratio=0.05),
        seed=42,
    )
    assert generator.generate(epoch=1).empty


def test_disabled_module_keeps_dataset_unchanged():
    df = pd.DataFrame([_row(0.00, 0.20), _row(0.20, 0.40)])
    dataset = CreakyVowelDataset(
        df,
        continuous_non_creaky_cfg={"enabled": False},
    )
    dataset.set_epoch(5)
    assert len(dataset) == len(df)
    assert dataset.df["start_s"].tolist() == df["start_s"].tolist()

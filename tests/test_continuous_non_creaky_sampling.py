import warnings

import pandas as pd

from wavfry.data.continuous_non_creaky_sampling import (
    ContinuousNonCreakyWindowGenerator,
)
from wavfry.data.dataset import CreakyVowelDataset
from wavfry.data.epoch_sampler import EpochShuffleSampler
from wavfry.data.grouped_sampler import GroupedBatchSampler


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
        "min_intervals": 2,
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


def test_unfillable_ratio_warns_and_returns_available_windows():
    df = pd.DataFrame([
        _row(0.00, 0.20),
        _row(0.20, 0.70, phone_type="silence"),
        _row(0.70, 0.90),
    ])
    generator = ContinuousNonCreakyWindowGenerator(
        df,
        _config(max_sp_duration_ms=0, max_sp_ratio=0.0),
        seed=42,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        generated = generator.generate(epoch=1)
    assert generated.empty
    assert len(caught) == 1
    message = str(caught[0].message)
    assert "requested=" in message
    assert "generated=" in message


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
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        generated = generator.generate(epoch=1)
    assert generated.empty
    assert len(caught) == 1
    assert generator.last_generation_stats["generated"] == 0


def test_single_long_interval_can_generate_partial_windows():
    df = pd.DataFrame([_row(0.00, 1.20)])
    generator = ContinuousNonCreakyWindowGenerator(
        df,
        _config(min_intervals=1, extra_sample_ratio=3.0),
        seed=42,
    )
    generated = generator.generate(epoch=1)
    assert len(generated) == 3
    assert generated["source_interval_count"].eq(1).all()
    assert generated["start_s"].ge(0.0).all()
    assert generated["end_s"].le(1.2).all()


def test_default_duration_cap_limits_length_shortcut():
    df = pd.DataFrame([_row(0.00, 1.20)])
    generator = ContinuousNonCreakyWindowGenerator(
        df,
        {"enabled": True, "extra_sample_ratio": 4.0},
        seed=42,
    )
    generated = generator.generate(epoch=1)
    durations_ms = (generated["end_s"] - generated["start_s"]) * 1000.0
    assert durations_ms.le(350.0 + 1e-6).all()


def test_legacy_empty_phone_is_counted_as_silence():
    legacy_silence = _row(0.00, 0.40, phone_type="vowel")
    legacy_silence["phone"] = ""
    df = pd.DataFrame([legacy_silence])
    generator = ContinuousNonCreakyWindowGenerator(
        df,
        _config(
            min_intervals=1,
            extra_sample_ratio=1.0,
            max_sp_duration_ms=0,
            max_sp_ratio=0.0,
        ),
        seed=42,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        generated = generator.generate(epoch=1)
    assert generated.empty
    assert len(caught) == 1


def test_grouped_sampler_degrades_when_anchor_has_only_one_label():
    df = pd.DataFrame([_row(0.0, 0.2), _row(0.2, 0.4)])
    sampler = GroupedBatchSampler(
        df=df,
        batch_size=2,
        anchor_speakers=["U1001"],
        anchor_speakers_per_batch=1,
        seed=42,
    )
    batch = next(iter(sampler))
    assert len(batch) == 2
    assert set(batch).issubset({0, 1})


def test_epoch_shuffle_sampler_is_resume_deterministic():
    dataset = list(range(10))
    first = EpochShuffleSampler(dataset, seed=42, fold_index=2)
    resumed = EpochShuffleSampler(dataset, seed=42, fold_index=2)
    first.set_epoch(5)
    resumed.set_epoch(5)
    assert list(first) == list(resumed)
    resumed.set_epoch(6)
    assert list(first) != list(resumed)


def test_generation_stats_report_exact_requested_count():
    df = pd.DataFrame([_row(0.00, 0.30), _row(0.30, 0.60)])
    generator = ContinuousNonCreakyWindowGenerator(df, _config(), seed=42)
    generated = generator.generate(epoch=2)
    assert generator.last_generation_stats["requested"] == len(df)
    assert generator.last_generation_stats["generated"] == len(generated)


def test_same_epoch_does_not_regenerate_dataset():
    df = pd.DataFrame([_row(0.00, 0.30), _row(0.30, 0.60)])
    dataset = CreakyVowelDataset(
        df,
        continuous_non_creaky_cfg=_config(extra_sample_ratio=1.0),
    )
    initial_df = dataset.df
    dataset.set_epoch(1)
    assert dataset.df is initial_df
    dataset.set_epoch(2)
    assert dataset.current_epoch == 2
    assert dataset.df is not initial_df


def test_epoch_one_dataset_includes_generated_rows_for_class_weighting():
    df = pd.DataFrame([
        _row(0.00, 0.30, label=0),
        _row(0.30, 0.60, label=0),
        _row(0.60, 0.90, label=1),
    ])
    dataset = CreakyVowelDataset(
        df,
        continuous_non_creaky_cfg=_config(extra_sample_ratio=1.0),
    )
    generated_count = int(dataset.df["sample_source"].eq("continuous_non_creaky").sum())
    assert generated_count == len(df)
    assert len(dataset.df) == len(df) + generated_count
    assert int(dataset.df["label"].sum()) == 1


def test_disabled_module_keeps_dataset_unchanged():
    df = pd.DataFrame([_row(0.00, 0.20), _row(0.20, 0.40)])
    dataset = CreakyVowelDataset(
        df,
        continuous_non_creaky_cfg={"enabled": False},
    )
    dataset.set_epoch(5)
    assert len(dataset) == len(df)
    assert dataset.df["start_s"].tolist() == df["start_s"].tolist()

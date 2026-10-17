"""Regression and hardening tests for architecture loading, caching, bounds, and optimizer grouping."""

import json
import math
from collections import OrderedDict
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import pandas as pd

from wavfry.model.wavlm_classifier import WavFryClassifier
from wavfry.data.paradigm_block import ContinuousAudioBlockDataset
from wavfry.train.trainer import Trainer
from wavfry.data.manifest import adapter_sharded_textgrid_cache


def _make_mock_wavlm():
    mock = nn.Module()
    mock.config = type("Config", (), {"hidden_size": 768, "num_hidden_layers": 12})()
    encoder = nn.Module()
    encoder.layers = nn.ModuleList([nn.Linear(768, 768) for _ in range(12)])
    mock.encoder = encoder
    mock.feature_extractor = nn.Module()
    mock._get_feat_extract_output_lengths = lambda lengths: torch.div(lengths, 320, rounding_mode="floor")
    hidden_states = tuple(torch.randn(1, 10, 768) for _ in range(13))
    mock_output = type("Output", (), {"hidden_states": hidden_states, "last_hidden_state": hidden_states[-1]})()
    mock.forward = lambda *args, **kwargs: mock_output
    return mock


def test_checkpoint_load_pooling_mean_and_no_scalar_mix():
    """Verify that loading legacy checkpoint keys does not raise AttributeError when pooling_mode='mean'

    and that scalar_mix_weights are safely popped when use_scalar_mix=False.
    """
    config = {
        "model": {
            "tasks": ["creaky", "breathy"],
            "backbone": "microsoft/wavlm-base-plus",
            "classifier_head": "linear",
            "pooling_mode": "mean",  # No attention heads instantiated
            "use_scalar_mix": False,  # No scalar mix instantiated
            "use_pos_encoding": False,
        }
    }
    model = WavFryClassifier(config, wavlm_instance=_make_mock_wavlm())

    # Legacy state dict with scalar mix and attention head bias
    legacy_state_dict = {
        "scalar_mix_weights.creaky": torch.ones(13),
        "scalar_mix_weights.breathy": torch.ones(13),
        "attention_heads.creaky.bias": torch.zeros(1),
        "attention_heads.breathy.bias": torch.zeros(1),
        "pos_scorer.weight": torch.zeros(1, 1),
    }

    metadata = {}
    missing_keys = []
    unexpected_keys = []
    error_msgs = []

    # Should not raise AttributeError: 'WavFryClassifier' object has no attribute 'attention_heads'
    model._load_from_state_dict(
        legacy_state_dict,
        prefix="",
        local_metadata=metadata,
        strict=False,
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
        error_msgs=error_msgs,
    )

    # All discarded keys should NOT appear in unexpected_keys
    assert not any("scalar_mix_weights" in k for k in unexpected_keys)
    assert not any("attention_heads" in k for k in unexpected_keys)
    assert not any("pos_scorer" in k for k in unexpected_keys)


def test_layer_mask_validation_and_extract_layer_parsing():
    """Verify layer_mask validations and string/int extraction layer parsing."""
    config = {
        "model": {
            "tasks": ["creaky"],
            "backbone": "microsoft/wavlm-base-plus",
            "use_scalar_mix": True,
            "pooling_mode": "mean",
        }
    }
    model = WavFryClassifier(config, wavlm_instance=_make_mock_wavlm())

    # 1. Invalid layer_mask shape
    with pytest.raises(ValueError, match="must equal num_layers \\+ 1"):
        invalid_mask = torch.ones(5, dtype=torch.bool)
        model._encode_sequence(torch.randn(1, 16000), layer_mask=invalid_mask)

    # 2. All-False layer_mask
    with pytest.raises(ValueError, match="cannot be all-False"):
        all_false_mask = torch.zeros(model.num_layers + 1, dtype=torch.bool)
        model._encode_sequence(torch.randn(1, 16000), layer_mask=all_false_mask)

    # 3. extract_layer parsing with use_scalar_mix=False
    config_no_sm = {
        "model": {
            "tasks": ["creaky"],
            "backbone": "microsoft/wavlm-base-plus",
            "use_scalar_mix": False,
            "extract_layer": "last",
            "pooling_mode": "mean",
        }
    }
    model_no_sm = WavFryClassifier(config_no_sm, wavlm_instance=_make_mock_wavlm())
    hidden_dict, out_lengths, padding_mask = model_no_sm._encode_sequence(torch.randn(1, 16000))
    assert "creaky" in hidden_dict
    assert hidden_dict["creaky"].shape == (1, 10, 768)


def test_continuous_audio_block_cloning_and_bound_clamping(monkeypatch, tmp_path):
    """Verify waveform slice cloning, bound clamping, and LRU eviction."""
    dummy_audio = tmp_path / "test.wav"
    sample_rate = 16000
    total_samples = 48000  # 3 seconds
    stored_waveform = torch.ones(total_samples, dtype=torch.float32) * 0.5

    # Mock torchaudio
    monkeypatch.setattr(
        "torchaudio.info",
        lambda p: type("AudioInfo", (), {"sample_rate": sample_rate, "num_frames": total_samples})(),
    )
    monkeypatch.setattr(
        "torchaudio.load",
        lambda p: (stored_waveform.unsqueeze(0).clone(), sample_rate),
    )

    df = pd.DataFrame([
        {
            "file_key": "test_audio",
            "audio_path": str(dummy_audio),
            "start_s": 0.5,
            "end_s": 1.5,
            "interval_id": "0.5|1.5|a",
            "label_creaky": 1.0,
            "speaker_id": "spk1",
        },
        {
            # Non-overlapping interval
            "file_key": "test_audio",
            "audio_path": str(dummy_audio),
            "start_s": 5.0,
            "end_s": 6.0,
            "interval_id": "5.0|6.0|i",
            "label_creaky": 0.0,
            "speaker_id": "spk1",
        },
        {
            # Inverted interval
            "file_key": "test_audio",
            "audio_path": str(dummy_audio),
            "start_s": 1.2,
            "end_s": 0.8,
            "interval_id": "1.2|0.8|u",
            "label_creaky": 0.0,
            "speaker_id": "spk1",
        },
    ])

    dataset = ContinuousAudioBlockDataset(
        df,
        duration_ms=2000.0,
        hop_ms=1000.0,
        cache_waveforms=True,
        max_cache_files=2,
    )

    assert len(dataset) > 0
    waveform, bounds, labels, speaker, block_id, ids = dataset[0]

    # Verify cloning: mutating returned waveform does not alter cached waveform
    waveform.fill_(999.0)
    cached_wave, _ = dataset.waveform_cache[str(dummy_audio)]
    assert not torch.allclose(cached_wave, torch.tensor(999.0))

    # Verify invalid intervals are filtered out
    for rel_b in bounds:
        assert rel_b[1] > rel_b[0]
        assert rel_b[0] >= 0

    # Verify LRU eviction
    dataset._get_audio_waveform("file1.wav")
    dataset._get_audio_waveform("file2.wav")
    assert len(dataset.waveform_cache) <= 2
    assert "file1.wav" in dataset.waveform_cache or "file2.wav" in dataset.waveform_cache


def test_optimizer_parameter_groups_norm_and_backbone():
    """Verify that Sequential LayerNorm parameters are placed in head_no_decay group

    and that scalar_mix parameters are not captured by backbone group.
    """
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.tasks = ["creaky"]
            self.backbone = nn.Sequential(nn.Linear(10, 10))
            self.scalar_mix_weights = nn.ParameterDict({"creaky": nn.Parameter(torch.ones(13))})
            self.head = nn.Sequential(
                nn.LayerNorm(10),
                nn.Linear(10, 1),
            )

    model = MockModel()
    config = {
        "training": {
            "optimizer": "adamw_fp32",
            "weight_decay": 0.01,
            "backbone_lr": 1e-5,
            "scalar_mix_lr": 1e-3,
            "head_lr": 1e-4,
            "epochs": 1,
            "paradigm": "segment",
        }
    }

    # Dummy trainer with minimal init for create_optimizer
    trainer = object.__new__(Trainer)
    trainer.model = model
    trainer.config = config

    optimizer = trainer.create_optimizer()
    param_groups = {g["name"]: g for g in optimizer.param_groups}

    # Verify groups exist
    assert "backbone" in param_groups
    assert "scalar_mix" in param_groups
    assert "head" in param_groups
    assert "head_no_decay" in param_groups

    # Verify backbone does NOT contain scalar mix weights
    backbone_params = set(param_groups["backbone"]["params"])
    sm_param = model.scalar_mix_weights["creaky"]
    assert sm_param not in backbone_params

    # Verify LayerNorm in Sequential is in head_no_decay (weight_decay == 0.0)
    norm_w = model.head[0].weight
    norm_b = model.head[0].bias
    head_no_decay_params = set(param_groups["head_no_decay"]["params"])
    assert norm_w in head_no_decay_params
    assert norm_b in head_no_decay_params
    assert param_groups["head_no_decay"]["weight_decay"] == 0.0


def test_manifest_mllm_breathy_split_gating(tmp_path):
    """Verify that mllm_breathy is only labeled 1.0 when b_split == 'train' and allow_mllm_pseudo_labels=True."""
    cache_dir = tmp_path / "_breathy_cache"
    cache_dir.mkdir()
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    # Create dummy audio files
    (audio_dir / "test_val.wav").write_bytes(b"RIFF....WAVEfmt ....data....")
    (audio_dir / "test_train.wav").write_bytes(b"RIFF....WAVEfmt ....data....")

    sampling_state = {
        "batches": [
            {
                "batch_id": 1,
                "name": "Batch 1",
                "split": "val",
            },
            {
                "batch_id": 2,
                "name": "Batch 2",
                "split": "train",
            },
        ]
    }
    (cache_dir / "sampling_state.json").write_text(json.dumps(sampling_state), encoding="utf-8")
    b1_dir = cache_dir / "batches" / "batch_1"
    b1_dir.mkdir(parents=True)
    b2_dir = cache_dir / "batches" / "batch_2"
    b2_dir.mkdir(parents=True)

    b1_shard = {
        "files": {
            "test_val": {
                "annotations": {
                    "0.1|0.5|a": "mllm_breathy",
                    "0.6|1.0|i": "breathy",
                }
            }
        }
    }
    (b1_dir / "manual_breathy_annotations.json").write_text(json.dumps(b1_shard), encoding="utf-8")

    b2_shard = {
        "files": {
            "test_train": {
                "annotations": {
                    "0.1|0.5|a": "mllm_breathy",
                }
            }
        }
    }
    (b2_dir / "manual_breathy_annotations.json").write_text(json.dumps(b2_shard), encoding="utf-8")

    ds_cfg = {
        "dataset_id": "test_ds",
        "adapter_params": {
            "audio_dir": str(audio_dir),
            "cache_dir": str(cache_dir),
            "allow_mllm_pseudo_labels": True,
            "lenient_missing": True,
        },
    }

    records = adapter_sharded_textgrid_cache(ds_cfg)
    rec_by_file_and_id = {(r["file_key"], f"{r['start_s']}|{r['end_s']}|{r['phone']}"): r for r in records}

    # In val split: mllm_breathy interval is filtered out (both labels NaN)
    assert ("test_val", "0.1|0.5|a") not in rec_by_file_and_id
    # In val split: human breathy annotation is kept with label 1.0
    assert ("test_val", "0.6|1.0|i") in rec_by_file_and_id
    assert rec_by_file_and_id[("test_val", "0.6|1.0|i")]["label_breathy"] == 1.0
    assert rec_by_file_and_id[("test_val", "0.6|1.0|i")]["annotation_type"] == "manual_annotation"

    # In train split: mllm_breathy interval is accepted with label 1.0 and annotation_type == "mllm_pseudo_label"
    assert ("test_train", "0.1|0.5|a") in rec_by_file_and_id
    assert rec_by_file_and_id[("test_train", "0.1|0.5|a")]["label_breathy"] == 1.0
    assert rec_by_file_and_id[("test_train", "0.1|0.5|a")]["annotation_type"] == "mllm_pseudo_label"

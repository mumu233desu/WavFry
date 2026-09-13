import math
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from wavfry.data.manifest import (
    MANIFEST_COLUMNS,
    adapter_grbas_csv,
    adapter_capev_csv,
    adapter_generic_csv,
    build_manifest_multi,
)
from wavfry.data.splits import isolate_split_speakers, make_batch_split, make_speaker_split
from wavfry.data.paradigm_block import ContinuousAudioBlockDataset, collate_block_fn
from wavfry.model.wavlm_classifier import WavFryClassifier
from wavfry.train.trainer import Trainer


# =====================================================================
# 1. Manifest & Adapters Tests
# =====================================================================

def test_manifest_columns_and_multi_adapter(tmp_path):
    # 1. Mock a GRBAS CSV dataset
    audio_dir = tmp_path / "grbas_audio"
    audio_dir.mkdir()
    (audio_dir / "audio_001.wav").write_bytes(b"RIFF" + b"\x00" * 100)
    (audio_dir / "audio_002.wav").write_bytes(b"RIFF" + b"\x00" * 100)

    csv_file = tmp_path / "grbas_annotations.csv"
    csv_df = pd.DataFrame([
        {"file_id": "audio_001", "B": 2, "speaker_id": "spk_g1"},
        {"file_id": "audio_002", "B": 0, "speaker_id": "spk_g2"},
    ])
    csv_df.to_csv(csv_file, index=False)

    ds_cfg = {
        "dataset_id": "femh_mock",
        "domain": "pathological_voice",
        "enabled": True,
        "annotation_adapter": "grbas_csv",
        "split_strategy": "all_train",
        "label_schema": {"creaky": False, "breathy": True},
        "adapter_params": {
            "annotation_granularity": "utterance",
            "audio_dir": str(audio_dir),
            "annotation_csv": str(csv_file),
            "csv_columns": {"file_id": "file_id", "grbas_b": "B", "speaker_id": "speaker_id"},
            "breathy_threshold": 1.0,
        }
    }

    datasets_cfg = {"datasets": [ds_cfg]}
    df = build_manifest_multi(datasets_cfg)

    # Verify column specification
    for col in MANIFEST_COLUMNS:
        assert col in df.columns, f"Missing required manifest column: {col}"

    assert len(df) == 2
    # Creaky is not labeled (NaN), Breathy is labeled (1.0 for B=2, 0.0 for B=0)
    assert np.isnan(df.iloc[0]["label_creaky"])
    assert df.iloc[0]["label_breathy"] == 1.0
    assert df.iloc[1]["label_breathy"] == 0.0
    assert df.iloc[0]["annotation_granularity"] == "utterance"
    assert df.iloc[0]["batch_split"] == "train"


def test_manifest_capev_and_generic_csv_adapters(tmp_path):
    # Mock CAPE-V CSV
    audio_dir = tmp_path / "capev_audio"
    audio_dir.mkdir()
    (audio_dir / "pv_001.wav").write_bytes(b"RIFF" + b"\x00" * 100)
    (audio_dir / "pv_002.wav").write_bytes(b"RIFF" + b"\x00" * 100)

    capev_csv = tmp_path / "capev.csv"
    pd.DataFrame([
        {"Sample": "pv_001", "Breathiness": 45.0, "Speaker": "spk_c1"},
        {"Sample": "pv_002", "Breathiness": 5.0, "Speaker": "spk_c2"},
    ]).to_csv(capev_csv, index=False)

    ds_cfg = {
        "dataset_id": "pvqd_mock",
        "enabled": True,
        "annotation_adapter": "capev_csv",
        "split_strategy": "all_train",
        "label_schema": {"creaky": False, "breathy": True},
        "adapter_params": {
            "audio_dir": str(audio_dir),
            "annotation_csv": str(capev_csv),
            "csv_columns": {"file_id": "Sample", "capev_b": "Breathiness", "speaker_id": "Speaker"},
            "breathy_threshold": 20.0,
        }
    }
    records = adapter_capev_csv(ds_cfg)
    assert len(records) == 2
    assert records[0]["label_breathy"] == 1.0
    assert records[1]["label_breathy"] == 0.0
    assert np.isnan(records[0]["label_creaky"])


# =====================================================================
# 2. Splits & Multi-Dataset Isolation Tests
# =====================================================================

def test_same_dataset_isolation_scope():
    # Two datasets have an overlapping speaker name "SPK_SHARED"
    train_df = pd.DataFrame([
        {"dataset_id": "cs_dialogue", "speaker_id": "SPK_SHARED", "label": 1},
        {"dataset_id": "cs_dialogue", "speaker_id": "SPK_CS_OTHER", "label": 0},
        {"dataset_id": "femh", "speaker_id": "SPK_FEMH_1", "label": 0},
    ])
    val_df = pd.DataFrame([
        # SPK_SHARED in cs_dialogue overlaps with cs_dialogue train
        {"dataset_id": "cs_dialogue", "speaker_id": "SPK_SHARED", "label": 1},
        {"dataset_id": "cs_dialogue", "speaker_id": "SPK_VAL_CLEAN", "label": 0},
        # SPK_SHARED in femh overlaps with cs_dialogue train, but NOT femh train
        {"dataset_id": "femh", "speaker_id": "SPK_SHARED", "label": 1},
    ])

    train_c, val_c, _ = isolate_split_speakers(
        train_df, val_df, enabled=True, isolation_scope="same_dataset"
    )

    # In cs_dialogue, SPK_SHARED was in train_df, so in cs_dialogue val it is pruned!
    cs_val_spks = val_c[val_c["dataset_id"] == "cs_dialogue"]["speaker_id"].tolist()
    assert "SPK_SHARED" not in cs_val_spks
    assert "SPK_VAL_CLEAN" in cs_val_spks

    # In femh, SPK_SHARED was NOT in femh's train_df! So it is preserved in femh val!
    femh_val_spks = val_c[val_c["dataset_id"] == "femh"]["speaker_id"].tolist()
    assert "SPK_SHARED" in femh_val_spks


# =====================================================================
# 3. Block Dataset Multi-Task Collate Tests
# =====================================================================

def test_block_collate_multitask():
    # Batch items where label is multi-task (2 tasks: creaky, breathy)
    bounds1 = torch.tensor([[0, 100], [100, 200]])
    labels1 = torch.tensor([[1.0, float("nan")], [0.0, float("nan")]])
    item1 = (torch.ones(300), bounds1, labels1, "spk1", "b1", ["i1", "i2"])

    bounds2 = torch.tensor([[50, 150]])
    labels2 = torch.tensor([[float("nan"), 1.0]])
    item2 = (torch.ones(200), bounds2, labels2, "spk2", "b2", ["i3"])

    batch = [item1, item2]
    waveforms, padded_bounds, padded_labels, valid, lengths, speakers, block_ids, interval_ids = collate_block_fn(batch)

    assert waveforms.shape == (2, 300)
    assert padded_bounds.shape == (2, 2, 2)
    assert padded_labels.shape == (2, 2, 2)  # [B=2, max_intervals=2, num_tasks=2]
    assert valid.tolist() == [[True, True], [True, False]]
    assert lengths.tolist() == [300, 200]

    # Slot (1, 1) is a padding interval: its labels must be NaN
    assert torch.isnan(padded_labels[1, 1, 0])
    assert torch.isnan(padded_labels[1, 1, 1])

    # Slot (0, 0) should match item 1
    assert padded_labels[0, 0, 0] == 1.0
    assert torch.isnan(padded_labels[0, 0, 1])

    # Slot (1, 0) should match item 2
    assert torch.isnan(padded_labels[1, 0, 0])
    assert padded_labels[1, 0, 1] == 1.0


# =====================================================================
# 4. Model Multi-Head & Padding Mask Defect Tests
# =====================================================================

class DummyWavLM(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.config = type("Config", (), {
            "hidden_size": hidden_dim,
            "num_hidden_layers": 12,
            "mask_time_prob": 0.0,
            "apply_spec_augment": False
        })()
        self.feature_extractor = nn.Identity()
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(12)])
        self.proj = nn.Linear(1, hidden_dim)

    def _get_feat_extract_output_lengths(self, lengths):
        return torch.clamp(lengths // 4, min=1)

    def forward(self, input_values, attention_mask=None, output_hidden_states=False):
        B, T = input_values.shape
        frame_len = max(1, T // 4)
        x = input_values[:, :frame_len * 4].view(B, frame_len, 4).mean(-1, keepdim=True)
        h = self.proj(x)
        hidden_states = tuple([h for _ in range(13)])
        return type("Output", (), {"hidden_states": hidden_states, "last_hidden_state": h})()


def test_model_multihead_forward_and_padding_mask():
    dummy_wavlm = DummyWavLM(hidden_dim=32)
    config = {
        "model": {
            "backbone": "dummy",
            "pooling": "attn",
            "tasks": ["creaky", "breathy"],
            "hidden_dim": 32,
            "num_classes": 1,
            "dropout": 0.0,
        },
        "training": {
            "paradigm": "block",
            "block": {
                "pooling": "attn",
                "boundary_trim_frames": 0,
            }
        }
    }

    model = WavFryClassifier(config, wavlm_instance=dummy_wavlm)
    assert set(model.tasks) == {"creaky", "breathy"}
    assert "creaky" in model.heads
    assert "breathy" in model.heads
    assert "creaky" in model.attention_heads
    assert "breathy" in model.attention_heads

    # Forward block mode
    B = 2
    T = 80
    waveforms = torch.randn(B, T)
    lengths = torch.tensor([80, 40])
    bounds = torch.tensor([
        [[0, 20], [20, 40]],
        [[0, 20], [0, 0]]
    ])
    valid = torch.tensor([[True, True], [True, False]])

    outputs = model(waveforms, lengths, interval_bounds=bounds, interval_valid=valid)
    assert "interval_logits" in outputs
    assert isinstance(outputs["interval_logits"], dict)
    assert "creaky" in outputs["interval_logits"]
    assert "breathy" in outputs["interval_logits"]
    assert outputs["interval_logits"]["creaky"].shape == (2, 2)
    assert outputs["interval_logits"]["breathy"].shape == (2, 2)

    # Forward segment mode
    seg_outputs = model(waveforms, lengths)
    assert isinstance(seg_outputs, dict)
    assert "creaky" in seg_outputs
    assert "breathy" in seg_outputs
    assert seg_outputs["creaky"].shape == (2,)
    assert seg_outputs["breathy"].shape == (2,)


def test_padding_mask_invariance():
    """Verify that padding frames do not alter the attention representation of real frames."""
    dummy_wavlm = DummyWavLM(hidden_dim=32)
    config = {
        "model": {
            "backbone": "dummy",
            "pooling": "attn",
            "tasks": ["creaky"],
            "hidden_dim": 32,
            "num_classes": 1,
            "dropout": 0.0,
        },
        "training": {"paradigm": "segment"}
    }
    model = WavFryClassifier(config, wavlm_instance=dummy_wavlm)
    model.eval()

    sample = torch.randn(1, 40)
    len_sample = torch.tensor([40])

    with torch.no_grad():
        out_unpadded = model(sample, len_sample)["creaky"]

        # Now place sample in a batch where the second sequence is 80 frames long,
        # forcing sample 1 to be zero-padded from 40 to 80.
        padded_batch = torch.zeros(2, 80)
        padded_batch[0, :40] = sample[0]
        padded_batch[1] = torch.randn(80)
        batch_lengths = torch.tensor([40, 80])

        out_padded = model(padded_batch, batch_lengths)["creaky"]

        # The prediction for sample 0 in the padded batch must match unpadded output!
        assert torch.allclose(out_unpadded[0], out_padded[0], atol=1e-5), (
            f"Padding mask failure: unpadded {out_unpadded[0].item():.5f} != padded {out_padded[0].item():.5f}"
        )


# =====================================================================
# 5. Trainer Multi-Task Loss Masking Tests
# =====================================================================

def test_trainer_loss_masking():
    config = {
        "model": {
            "backbone": "dummy",
            "pooling": "attn",
            "tasks": ["creaky", "breathy"],
            "hidden_dim": 32,
            "num_classes": 1,
            "dropout": 0.0,
        },
        "training": {
            "epochs": 1,
            "paradigm": "block",
            "train_heads": {"creaky": True, "breathy": False},
            "loss": "bce",
            "weight_decay": 0.01,
            "backbone_lr": 1e-4,
            "head_lr": 1e-3,
            "warmup_steps": 10,
            "gradient_accumulation_steps": 1,
            "grad_clip": 1.0,
            "block": {"pooling": "attn"}
        },
        "eval": {
            "primary_metric": "auc_pr",
            "primary_task": "creaky",
        },
        "checkpoint": {"dir": "tmp_ckpt"},
        "tensorboard": {"log_dir": "tmp_tb"}
    }

    dummy_wavlm = DummyWavLM(hidden_dim=32)
    model = WavFryClassifier(config, wavlm_instance=dummy_wavlm)

    trainer = Trainer.__new__(Trainer)
    trainer.model = model
    trainer.config = config
    trainer.device = torch.device("cpu")
    trainer.loss_fn = nn.BCEWithLogitsLoss()
    trainer.frame_cfg = {}
    trainer._current_frame_proxy = None

    slot_valid = torch.tensor([[True, True], [True, False]])
    outputs = {
        "interval_valid": slot_valid,
        "interval_logits": {
            "creaky": torch.tensor([[0.5, -0.5], [1.2, 0.0]]),
            "breathy": torch.tensor([[-0.2, 0.8], [-1.0, 0.0]]),
        }
    }

    # Case 1: Multi-task labels where breathy has values, but train_heads disables breathy during training
    labels = torch.tensor([
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0, float("nan")], [float("nan"), float("nan")]]
    ])

    loss_train, logits_dict, targets_dict = trainer._compute_block_loss(outputs, labels, slot_valid, training=True)
    assert not torch.isnan(loss_train)
    assert loss_train.item() > 0

    # In validation mode (training=False), breathy is evaluated
    loss_val, _, _ = trainer._compute_block_loss(outputs, labels, slot_valid, training=False)
    assert not torch.isnan(loss_val)
    assert loss_val.item() > loss_train.item()

    # Case 2: Creaky labels are present, breathy labels are entirely NaN
    labels_all_nan_breathy = torch.tensor([
        [[1.0, float("nan")], [0.0, float("nan")]],
        [[1.0, float("nan")], [float("nan"), float("nan")]]
    ])
    loss_nan_eval, _, _ = trainer._compute_block_loss(outputs, labels_all_nan_breathy, slot_valid, training=False)
    assert not torch.isnan(loss_nan_eval)


def test_single_task_backward_compatibility_target_extraction():
    """Verify single-task model extracts the correct task column even when dataset returns multi-task labels."""
    config = {
        "model": {"backbone": "dummy", "tasks": ["creaky"], "hidden_dim": 32, "num_classes": 1},
        "training": {"paradigm": "segment"}
    }
    dummy_wavlm = DummyWavLM(hidden_dim=32)
    model = WavFryClassifier(config, wavlm_instance=dummy_wavlm)

    trainer = Trainer.__new__(Trainer)
    trainer.model = model
    trainer.config = config

    # Dataset multi-label tensor: [B=2, tasks=2] where col 0 is creaky, col 1 is breathy
    multi_labels = torch.tensor([[1.0, float("nan")], [0.0, 1.0]])

    # When extracting 'creaky', it must return 1D tensor with col 0
    extracted_creaky = trainer._get_task_target(multi_labels, "creaky")
    assert extracted_creaky.shape == (2,)
    assert extracted_creaky[0] == 1.0
    assert extracted_creaky[1] == 0.0

    # When extracting 'breathy', it must return col 1
    extracted_breathy = trainer._get_task_target(multi_labels, "breathy")
    assert extracted_breathy.shape == (2,)
    assert torch.isnan(extracted_breathy[0])
    assert extracted_breathy[1] == 1.0


def test_segment_mean_attn_pooling():
    """Verify segment mode supports mean_attn without linear dimension mismatch."""
    dummy_wavlm = DummyWavLM(hidden_dim=32)
    config = {
        "model": {
            "backbone": "dummy",
            "tasks": ["creaky"],
            "hidden_dim": 32,
            "num_classes": 1,
            "dropout": 0.0,
        },
        "training": {
            "paradigm": "segment",
            "pooling_mode": "mean_attn"
        }
    }
    model = WavFryClassifier(config, wavlm_instance=dummy_wavlm)
    waveforms = torch.randn(2, 40)
    lengths = torch.tensor([40, 30])
    out = model(waveforms, lengths)
    assert "creaky" in out
    assert out["creaky"].shape == (2,)

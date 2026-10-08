import copy
import json

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch
from torch.utils.data import DataLoader
from transformers import WavLMConfig, WavLMModel

from wavfry.data.paradigm_block import ContinuousAudioBlockDataset, collate_block_fn
from wavfry.model.wavlm_classifier import WavFryClassifier
from wavfry.train.trainer import Trainer
from wavfry.train.weak_supervision import guard_weak_partitions, interleave_batches


def test_interleaving_preserves_both_sources():
    assert list(interleave_batches([1, 2, 3], [10, 11])) == [
        ("manual", 1), ("manual", 2), ("weak", 10), ("manual", 3), ("weak", 11)]
    assert list(interleave_batches([1, 2])) == [("manual", 1), ("manual", 2)]


def test_weak_speaker_and_file_exclusion():
    weak = pd.DataFrame([dict(supervision_id="a", dataset_id="d", file_key="f", speaker_id="s",
                             batch_split="train", import_status="eligible")])
    val = pd.DataFrame([dict(dataset_id="d", file_key="other", speaker_id="s")])
    with pytest.raises(ValueError, match="speaker"):
        guard_weak_partitions(weak, None, val)
    val.loc[0, "file_key"] = "f"
    val.loc[0, "speaker_id"] = "other"
    with pytest.raises(ValueError, match="file"):
        guard_weak_partitions(weak, None, val)


def setup(tmp_path, enabled=True):
    weak_audio = tmp_path / "weak.wav"
    sf.write(weak_audio, np.random.default_rng(42).normal(0, .05, 48000).astype("float32"), 16000)
    weak = pd.DataFrame([dict(supervision_id="w", dataset_id="d", file_key="weak", speaker_id="s_weak",
        audio_path=str(weak_audio), start_s=0., end_s=3., task="breathy", label=1.,
        import_status="eligible", batch_split="train", vowel_intervals=json.dumps([[.1, 2.9]]), fingerprint="test")])
    manifest = tmp_path / "weak.csv"
    from wavfry.data.weak_manifest import file_digest
    grid, state = tmp_path / 'grid.TextGrid', tmp_path / 'state.json'
    grid.write_text('test-grid')
    state.write_text('{}')
    stat = weak_audio.stat()
    weak['tg_path'] = str(grid)
    weak['source_stamp'] = json.dumps(dict(audio_size=stat.st_size, audio_mtime_ns=stat.st_mtime_ns,
        grid=file_digest(grid), cache='', sampling_state_path=str(state), sampling_state=file_digest(state)))
    weak.to_csv(manifest, index=False)
    config = {
        "model": {"backbone": "dummy", "fine_tune_layers": "none", "tasks": ["creaky", "breathy"],
                  "head_hidden_dim": 8, "dropout": 0.},
        "training": {"epochs": 2, "paradigm": "block", "quiet": True, "pooling_mode": "mean", "loss": "weighted_bce",
                     "backbone_lr": 0., "head_lr": .001, "warmup_steps": 0, "lr_schedule": "none",
                     "gradient_accumulation_steps": 3, "grad_clip": 1., "mixed_precision": "no",
                     "block": {"duration_ms": 2000, "hop_ms": 1000, "frame_head": {"enabled": True, "mil_mode": "lse"}},
                     "weak_supervision": {"enabled": enabled, "manifest": str(manifest), "start_epoch": 2,
                                          "max_windows": 4, "samples_per_epoch": 1, "loss_weight": .05}},
        "eval": {"primary_metric": "auc_pr", "primary_task": "breathy", "early_stopping_patience": 0},
        "tensorboard": {"log_dir": str(tmp_path / "log"), "log_audio_samples": False, "log_pr_curve": False},
        "checkpoint": {"dir": str(tmp_path / "ckpt"), "save_mode": "full", "keep_best_count": 1},
    }
    backbone = WavLMModel(WavLMConfig(hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=16, conv_dim=(4, 4, 4), conv_stride=(5, 4, 4), conv_kernel=(10, 8, 8),
        num_conv_pos_embeddings=8, num_conv_pos_embedding_groups=2))
    model = WavFryClassifier(config, wavlm_instance=backbone)
    loaders = []
    for split in ("train", "val"):
        audio = tmp_path / f"{split}.wav"
        sf.write(audio, np.zeros(16000, dtype="float32"), 16000)
        df = pd.DataFrame([dict(dataset_id="d", file_key=split, speaker_id=split, audio_path=str(audio),
            start_s=.1, end_s=.3, label_creaky=0., label_breathy=0.),
            dict(dataset_id="d", file_key=split, speaker_id=split, audio_path=str(audio),
                 start_s=.4, end_s=.6, label_creaky=1., label_breathy=1.)])
        loaders.append(DataLoader(ContinuousAudioBlockDataset(df), batch_size=1, collate_fn=collate_block_fn))
    trainer = Trainer(model, *loaders, config, torch.device("cpu"))
    return trainer, manifest


def test_mixed_training_and_checkpoint_resume(tmp_path):
    trainer, manifest = setup(tmp_path)
    frame_before = trainer.model.frame_heads["breathy"][1].weight.detach().clone()
    loss1, metrics1 = trainer.train_epoch(1)
    assert metrics1["weak_utterances"] == 0
    loss2, metrics2 = trainer.train_epoch(2)
    assert metrics2["weak_utterances"] == 1
    assert np.isfinite(loss1) and np.isfinite(loss2)
    assert not torch.equal(frame_before, trainer.model.frame_heads["breathy"][1].weight)
    assert trainer.scheduler.last_epoch == trainer.total_steps
    trainer._best_val_logits = torch.tensor([-1., 1.])
    trainer._best_val_labels = torch.tensor([0., 1.])
    trainer.save_checkpoint(2, {"f1": .5, "auc_pr": .6}, only_latest=True)
    checkpoint = torch.load(trainer.ckpt_dir / "latest.pt", weights_only=False)
    assert checkpoint["weak_supervision_state"]["fingerprint"] == trainer.weak_fingerprint
    assert checkpoint["training_provenance"]["complete"] is True
    assert ("d", "s_weak") in checkpoint["training_provenance"]["training_speakers"]
    assert trainer.load_checkpoint(trainer.ckpt_dir / "latest.pt") == 2
    trainer.weak_fingerprint = "changed"
    with pytest.raises(ValueError, match="manifest changed"):
        trainer.load_checkpoint(trainer.ckpt_dir / "latest.pt")
    trainer.writer.close()


def test_disabled_weak_path_keeps_original_data(tmp_path):
    trainer, _ = setup(tmp_path, enabled=False)
    assert trainer.weak_loader is None
    assert trainer.total_steps == 2
    assert len(trainer.train_loader.dataset.base_df) == 2
    trainer.writer.close()

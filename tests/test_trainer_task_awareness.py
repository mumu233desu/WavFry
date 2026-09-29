"""Tests for per-task loss functions, primary-task resolution, and the
checkpoint threshold chain (best_val_logits/labels persistence)."""

import math

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from wavfry.train.trainer import Trainer


# ── _resolve_primary_task ────────────────────────────────────────────────

def test_resolve_primary_task_explicit_config():
    config = {"eval": {"primary_task": "breathy"}, "training": {}}
    assert Trainer._resolve_primary_task(config, ["creaky", "breathy"]) == "breathy"


def test_resolve_primary_task_default_prefers_trainable_creaky():
    config = {"eval": {}, "training": {"train_heads": {"creaky": False}}}
    assert Trainer._resolve_primary_task(config, ["creaky", "breathy"]) == "breathy"


def test_resolve_primary_task_default_creaky_when_no_gating():
    config = {"training": {}}
    assert Trainer._resolve_primary_task(config, ["creaky", "breathy"]) == "creaky"


def test_resolve_primary_task_falls_back_to_first_task():
    config = {"training": {"train_heads": {"creaky": False, "breathy": False}}}
    assert Trainer._resolve_primary_task(config, ["creaky", "breathy"]) == "creaky"


def test_resolve_primary_task_rejects_inactive_task():
    config = {"eval": {"primary_task": "breathy"}, "training": {}}
    with pytest.raises(ValueError):
        Trainer._resolve_primary_task(config, ["creaky"])


# ── _task_pos_weight ─────────────────────────────────────────────────────

class _FakeBlockDataset:
    def __init__(self, blocks):
        self.blocks = blocks


class _FakeLoader:
    def __init__(self, dataset):
        self.dataset = dataset


def _block_loader(rows):
    return _FakeLoader(_FakeBlockDataset([{"intervals": rows}]))


def test_task_pos_weight_block_paradigm_per_task():
    trainer = object.__new__(Trainer)
    trainer.paradigm = "block"
    # creaky: 1 pos / 2 neg → 2.0; breathy: NaN ignored → 2 pos / 0 neg → 0.5
    rows = [
        {"label_creaky": 1.0, "label_breathy": float("nan")},
        {"label_creaky": 0.0, "label_breathy": 1.0},
        {"label_creaky": 0.0, "label_breathy": 1.0},
    ]
    loader = _block_loader(rows)
    creaky_w = trainer._task_pos_weight(loader, "creaky", torch.device("cpu"))
    breathy_w = trainer._task_pos_weight(loader, "breathy", torch.device("cpu"))
    assert math.isclose(creaky_w.item(), 2.0)
    assert math.isclose(breathy_w.item(), 0.5)


def test_task_pos_weight_block_paradigm_values():
    trainer = object.__new__(Trainer)
    trainer.paradigm = "block"
    rows = [
        {"label_creaky": 1.0, "label_breathy": 0.0},
        {"label_creaky": 1.0, "label_breathy": 0.0},
        {"label_creaky": 0.0, "label_breathy": 0.0},
        {"label_creaky": 0.0, "label_breathy": 0.0},
    ]
    loader = _block_loader(rows)
    creaky_w = trainer._task_pos_weight(loader, "creaky", torch.device("cpu"))
    breathy_w = trainer._task_pos_weight(loader, "breathy", torch.device("cpu"))
    assert math.isclose(creaky_w.item(), 1.0)   # 2 pos / 2 neg
    assert math.isclose(breathy_w.item(), 4.0)  # 0 pos → max(4,1)/max(0,1) = 4.0


def test_task_pos_weight_segment_paradigm_per_task():
    trainer = object.__new__(Trainer)
    trainer.paradigm = "segment"
    df = pd.DataFrame({
        "label_creaky": [1.0, 0.0, 0.0, float("nan")],
        "label_breathy": [float("nan"), 1.0, 1.0, 0.0],
    })
    loader = _FakeLoader(type("DS", (), {"df": df})())
    creaky_w = trainer._task_pos_weight(loader, "creaky", torch.device("cpu"))
    breathy_w = trainer._task_pos_weight(loader, "breathy", torch.device("cpu"))
    assert math.isclose(creaky_w.item(), 2.0)   # 1 pos / 2 neg (NaN dropped)
    assert math.isclose(breathy_w.item(), 0.5)  # 2 pos / 1 neg


# ── checkpoint threshold chain ───────────────────────────────────────────

def _bare_trainer(tmp_path):
    model = nn.Linear(4, 1)
    trainer = object.__new__(Trainer)
    trainer.config = {
        "checkpoint": {"save_mode": "full", "dir": str(tmp_path), "keep_best_count": 3},
        "training": {},
        "eval": {},
    }
    trainer.device = torch.device("cpu")
    trainer.model = model
    trainer.optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    trainer.scheduler = optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1.0)
    trainer.scaler = None
    trainer.ckpt_dir = tmp_path
    trainer.log_dir = tmp_path / "logs"
    trainer.log_dir.mkdir(exist_ok=True)
    trainer.history = []
    trainer.patience_counter = 2
    trainer.best_primary_metric_val = 0.9
    return trainer


def test_checkpoint_persists_best_val_outputs(tmp_path):
    import os
    trainer = _bare_trainer(tmp_path)
    logits = torch.tensor([0.1, 0.4, 0.9])
    labels = torch.tensor([0.0, 1.0, 1.0])
    trainer._best_val_logits = logits
    trainer._best_val_labels = labels
    trainer.save_checkpoint(3, {"f1": 0.8, "auc_pr": 0.9}, is_best=False, only_latest=True)

    ckpt = torch.load(tmp_path / "latest.pt", map_location="cpu", weights_only=False)
    assert torch.equal(ckpt["best_val_logits"], logits)
    assert torch.equal(ckpt["best_val_labels"], labels)

    # Restore into a fresh bare trainer that has lost the in-memory state
    trainer2 = _bare_trainer(tmp_path)
    trainer2._best_val_logits = None
    trainer2._best_val_labels = None
    trainer2.load_checkpoint(tmp_path / "latest.pt")
    assert trainer2._best_val_logits is not None
    assert torch.equal(trainer2._best_val_logits, logits)
    assert torch.equal(trainer2._best_val_labels, labels)
    assert trainer2.patience_counter == 2


def test_checkpoint_without_best_outputs_round_trips_none(tmp_path):
    trainer = _bare_trainer(tmp_path)
    trainer._best_val_logits = None
    trainer._best_val_labels = None
    trainer.save_checkpoint(1, {"f1": 0.5, "auc_pr": 0.5}, is_best=False, only_latest=True)

    trainer2 = _bare_trainer(tmp_path)
    trainer2.load_checkpoint(tmp_path / "latest.pt")
    assert trainer2._best_val_logits is None
    assert trainer2._best_val_labels is None


def test_threshold_injection_covers_latest_checkpoint(tmp_path):
    trainer = _bare_trainer(tmp_path)
    trainer._best_val_logits = None
    trainer._best_val_labels = None
    trainer.save_checkpoint(1, {"f1": 0.5, "auc_pr": 0.5}, is_best=False, only_latest=True)

    trainer._update_checkpoint_threshold(0.65, 0.77)
    # latest.pt (previously excluded from threshold injection) is now covered
    ckpt = torch.load(tmp_path / "latest.pt", map_location="cpu", weights_only=False)
    assert ckpt["best_threshold"] == 0.65
    assert ckpt["threshold_f1"] == 0.77
    assert ckpt["metrics"]["best_threshold"] == 0.65


def test_rebuild_writer_produces_usable_writer(tmp_path):
    from torch.utils.tensorboard import SummaryWriter

    trainer = object.__new__(Trainer)
    trainer.log_dir = tmp_path
    trainer.writer = SummaryWriter(log_dir=str(tmp_path))
    trainer.writer.close()
    # Watchdog recovery: after a failed run leaves the writer closed, rebuilding
    # must yield a writer that accepts new scalars instead of raising IOError.
    trainer.rebuild_writer()
    trainer.writer.add_scalar("recovery/check", 1.0, 0)
    trainer.writer.flush()
    trainer.writer.close()

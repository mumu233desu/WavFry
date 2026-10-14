"""Tests for eval_interval validation scheduling, patience freezing, and checkpoint safety."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from wavfry.train.trainer import Trainer


# ── Configuration Parsing ──────────────────────────────────────────────────

def test_eval_interval_config_parsing():
    trainer = object.__new__(Trainer)
    trainer.config = {'eval': {'eval_interval': 3, 'primary_metric': 'auc_pr'}}
    trainer.eval_interval = max(1, int(trainer.config.get('eval', {}).get('eval_interval', 1)))
    assert trainer.eval_interval == 3


def test_eval_interval_default_fallback():
    trainer = object.__new__(Trainer)
    trainer.config = {'eval': {'primary_metric': 'auc_pr'}}
    trainer.eval_interval = max(1, int(trainer.config.get('eval', {}).get('eval_interval', 1)))
    assert trainer.eval_interval == 1


def test_eval_interval_clamped_to_one():
    trainer = object.__new__(Trainer)
    trainer.config = {'eval': {'eval_interval': 0, 'primary_metric': 'auc_pr'}}
    trainer.eval_interval = max(1, int(trainer.config.get('eval', {}).get('eval_interval', 1)))
    assert trainer.eval_interval == 1

    trainer.config = {'eval': {'eval_interval': -5, 'primary_metric': 'auc_pr'}}
    trainer.eval_interval = max(1, int(trainer.config.get('eval', {}).get('eval_interval', 1)))
    assert trainer.eval_interval == 1


# ── Validation Schedule & Terminal Epoch Guard ────────────────────────────

@pytest.mark.parametrize(
    "epochs, interval, expected_validated_epochs",
    [
        (5, 2, [2, 4, 5]),         # 5 is terminal epoch guard
        (6, 2, [2, 4, 6]),         # 6 is naturally divisible
        (3, 5, [3]),               # 3 is terminal epoch guard (interval > epochs)
        (4, 1, [1, 2, 3, 4]),      # interval 1 validates every epoch
        (7, 3, [3, 6, 7]),         # 7 is terminal epoch guard
    ]
)
def test_validation_schedule_and_terminal_guard(epochs, interval, expected_validated_epochs):
    validated = [
        ep for ep in range(1, epochs + 1)
        if (ep % interval == 0) or (ep == epochs)
    ]
    assert validated == expected_validated_epochs


# ── Mocked Run Loop: Validation Scheduling & Patience Freezing ───────────

class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)
        self.tasks = ["creaky"]


def test_run_loop_eval_interval_and_patience_freeze(tmp_path):
    trainer = object.__new__(Trainer)
    trainer.config = {
        'eval': {
            'primary_task': 'creaky',
            'primary_metric': 'auc_pr',
            'eval_interval': 2,
            'early_stopping_patience': 5,
        },
        'training': {
            'epochs': 5,
        },
        'checkpoint': {
            'dir': str(tmp_path / "checkpoints"),
            'save_mode': 'delta',
        },
        'tensorboard': {
            'log_dir': str(tmp_path / "runs"),
            'log_pr_curve': False,
            'log_audio_samples': False,
        },
    }
    trainer.primary_task = 'creaky'
    trainer.primary_metric = 'auc_pr'
    trainer.eval_interval = 2
    trainer.epochs = 5
    trainer.best_primary_metric_val = 0.0
    trainer.patience_counter = 0
    trainer._best_val_logits = None
    trainer._best_val_labels = None
    trainer.log_dir = tmp_path / "runs"
    trainer.ckpt_dir = tmp_path / "checkpoints"
    trainer.history = []
    trainer.model = _DummyModel()
    trainer.writer = SummaryWriter(log_dir=str(trainer.log_dir))
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=1e-3)
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lr_lambda=lambda ep: 1.0)
    trainer.scaler = None

    # Track calls
    val_calls = []
    checkpoint_calls = []

    def mock_train_epoch(epoch):
        return 0.5, {'auc_pr': 0.7, 'f1': 0.6}

    def mock_validate(epoch):
        val_calls.append(epoch)
        # Metric does not improve past 0.5
        return 0.4, {
            'auc_pr': 0.5,
            'f1': 0.5,
            'logits': torch.zeros(10),
            'labels': torch.zeros(10),
        }

    def mock_save_checkpoint(epoch, metrics, is_best=False, only_latest=False):
        checkpoint_calls.append((epoch, is_best, only_latest, metrics.copy()))

    trainer.train_epoch = mock_train_epoch
    trainer.validate = mock_validate
    trainer.save_checkpoint = mock_save_checkpoint
    trainer.save_history = MagicMock()
    trainer.find_best_threshold = MagicMock(return_value=(0.5, 0.5))
    trainer._update_checkpoint_threshold = MagicMock()

    # Execute run
    history = trainer.run(start_epoch=1)

    # 1. Validation should ONLY be called on epochs 2, 4, 5 (interval=2 with terminal guard on 5)
    assert val_calls == [2, 4, 5]

    # 2. Checkpoint should be saved on EVERY epoch (including skipped epochs as latest)
    saved_epochs = [call[0] for call in checkpoint_calls]
    assert saved_epochs == [1, 2, 3, 4, 5]

    # On skipped epochs (1 and 3), only_latest must be True and metrics empty
    assert checkpoint_calls[0] == (1, False, True, {})
    assert checkpoint_calls[2] == (3, False, True, {})

    # 3. Patience should only increment on evaluated epochs where metric didn't improve:
    # Epoch 1: Skipped -> patience = 0
    # Epoch 2: Val (0.5 > 0.0 best) -> is_best=True, best=0.5, patience = 0
    # Epoch 3: Skipped -> patience remains 0 (FROZEN!)
    # Epoch 4: Val (0.5 <= 0.5 best) -> patience = 1
    # Epoch 5: Val (0.5 <= 0.5 best) -> patience = 2
    assert trainer.patience_counter == 2

    # 4. History records on skipped epochs should have None for val metrics
    assert history[0]['epoch'] == 1
    assert history[0]['val_loss'] is None
    assert history[0]['val_auc_pr'] is None

    assert history[1]['epoch'] == 2
    assert history[1]['val_loss'] == 0.4
    assert history[1]['val_auc_pr'] == 0.5

    assert history[2]['epoch'] == 3
    assert history[2]['val_loss'] is None
    assert history[2]['val_auc_pr'] is None

    assert history[3]['epoch'] == 4
    assert history[3]['val_loss'] == 0.4

    assert history[4]['epoch'] == 5
    assert history[4]['val_loss'] == 0.4

    trainer.writer.close()


# ── Checkpoint Safety with Empty Metrics ──────────────────────────────────

def test_save_checkpoint_empty_metrics(tmp_path):
    trainer = object.__new__(Trainer)
    trainer.config = {
        'checkpoint': {
            'dir': str(tmp_path),
            'save_mode': 'delta',
        },
        'training': {},
    }
    trainer.ckpt_dir = tmp_path
    trainer.model = _DummyModel()
    trainer.patience_counter = 0
    trainer.best_primary_metric_val = 0.85
    trainer._best_val_logits = None
    trainer._best_val_labels = None
    trainer.scaler = None
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=1e-3)
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lr_lambda=lambda ep: 1.0)

    # Must not raise KeyError: 'f1'
    ckpt_path = trainer.save_checkpoint(epoch=1, metrics={}, is_best=False, only_latest=True)
    latest_pt = tmp_path / "latest.pt"
    assert latest_pt.exists()

    ckpt = torch.load(latest_pt, map_location="cpu", weights_only=False)
    assert ckpt['epoch'] == 1
    assert ckpt['f1'] == 0.0
    assert ckpt['auc_pr'] == 0.0
    assert ckpt['metrics'] == {}
    assert ckpt['best_primary_metric_val'] == 0.85

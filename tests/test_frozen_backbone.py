from __future__ import annotations

import pytest
import torch
from wavfry.model.wavlm_classifier import WavFryClassifier


def test_frozen_backbone_parameter_gradients():
    """Verify that fine_tune_layers='none' completely freezes backbone while keeping head and scalar mix trainable."""
    config = {
        "model": {
            "backbone": "hf-internal-testing/tiny-random-wavlm",
            "fine_tune_layers": "none",
            "use_scalar_mix": True,
            "dropout": 0.1,
            "head_hidden_dim": 64,
            "tasks": ["creaky"],
        },
        "training": {
            "pooling_mode": "attn",
            "topk_ratio": 0.2,
            "topk_min_frames": 3,
            "ablation": {"use_positional_encoding": False},
            "block": {"boundary_trim_frames": 0},
            "frame_head": {"enabled": False},
        },
    }

    model = WavFryClassifier(config)

    # 1. All backbone parameters must have requires_grad == False
    is_backbone = lambda n: n.startswith("wavlm.") or n.startswith("backbone.")
    backbone_trainable = [p for n, p in model.named_parameters() if is_backbone(n) and p.requires_grad]
    assert len(backbone_trainable) == 0, f"Expected 0 trainable backbone params, got {len(backbone_trainable)}"

    # 2. Scalar mix and head parameters must have requires_grad == True
    head_trainable = [p for n, p in model.named_parameters() if not is_backbone(n) and p.requires_grad]
    assert len(head_trainable) > 0, "Classification head / scalar mix parameters should be trainable"

    # 3. Scalar mix weights exists and has correct length (num_layers + 1)
    assert "creaky" in model.scalar_mix_weights
    assert model.scalar_mix_weights["creaky"].shape[0] == model.num_layers + 1


def test_optimizer_empty_backbone_filter():
    """Verify that create_optimizer filters out empty parameter groups without crashing."""
    import yaml
    from pathlib import Path

    defaults_path = Path(__file__).parent.parent / "config" / "_defaults.yaml"
    with open(defaults_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["model"]["backbone"] = "hf-internal-testing/tiny-random-wavlm"
    config["model"]["fine_tune_layers"] = "none"
    config["model"]["head_hidden_dim"] = 64
    config["training"]["optimizer"] = "adamw_fp32"
    config["checkpoint"] = {"dir": "scratch/test_ckpt"}
    config["eval"] = {"primary_metric": "auc_pr"}

    model = WavFryClassifier(config)
    from wavfry.train.trainer import Trainer
    import pandas as pd

    dummy_ds = type("DummyDS", (), {"df": pd.DataFrame({"label_creaky": [0.0, 1.0], "label": [0.0, 1.0]})})()
    dummy_loader = type("DummyLoader", (), {"dataset": dummy_ds, "__len__": lambda s: 1, "__iter__": lambda s: iter([1])})()

    trainer = Trainer(
        model=model,
        train_loader=dummy_loader,
        val_loader=dummy_loader,
        config=config,
        device=torch.device("cpu"),
    )
    opt = trainer.optimizer

    # The optimizer should only have non-empty groups (no empty param group error)
    assert len(opt.param_groups) > 0
    for group in opt.param_groups:
        assert len(group["params"]) > 0

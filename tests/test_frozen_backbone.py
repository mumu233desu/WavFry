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
            "block": {"boundary_trim_frames": 0, "frame_head": {"enabled": False}},
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


def test_frozen_backbone_stays_in_eval_mode():
    """Verify that when backbone is frozen, model.train() does not put wavlm into train mode."""
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
            "ablation": {"use_positional_encoding": True, "pos_embed_dim": 32},
            "block": {"boundary_trim_frames": 0, "frame_head": {"enabled": False}},
        },
    }
    model = WavFryClassifier(config)
    assert model.is_backbone_frozen is True
    assert model.wavlm.training is False

    # Calling model.train() should keep wavlm in eval mode
    model.train()
    assert model.training is True
    assert model.wavlm.training is False

    # Calling model.eval() should also keep it in eval mode
    model.eval()
    assert model.training is False
    assert model.wavlm.training is False


def test_scalar_mix_parameter_dict_freezing():
    """Verify that requires_grad_(False) properly freezes parameters in ParameterDict."""
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
            "block": {"boundary_trim_frames": 0, "frame_head": {"enabled": False}},
        },
    }
    model = WavFryClassifier(config)
    assert any(p.requires_grad for p in model.scalar_mix_weights.parameters())

    # In-place freeze on ParameterDict
    model.scalar_mix_weights.requires_grad_(False)
    assert not any(p.requires_grad for p in model.scalar_mix_weights.parameters())


def test_vectorized_block_forward_equivalence():
    """Verify that vectorized block pooling produces correct shapes and outputs."""
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
            "ablation": {"use_positional_encoding": True, "pos_embed_dim": 32},
            "block": {"boundary_trim_frames": 0, "frame_head": {"enabled": False}},
        },
    }
    model = WavFryClassifier(config)
    model.eval()

    batch_size = 2
    samples = 16000
    x = torch.randn(batch_size, samples)
    lengths = torch.tensor([16000, 12000])
    interval_bounds = torch.tensor([
        [[0, 4000], [4000, 8000], [8000, 16000]],
        [[0, 3000], [3000, 6000], [0, 0]],
    ])
    interval_valid = torch.tensor([
        [True, True, True],
        [True, True, False],
    ])

    out = model(x, lengths=lengths, interval_bounds=interval_bounds, interval_valid=interval_valid)
    assert "interval_logits" in out
    assert "creaky" in out["interval_logits"]
    logits = out["interval_logits"]["creaky"]
    assert logits.shape == (2, 3)
    assert not torch.isnan(logits).any()


def test_backward_compatible_pool_batch():
    """Verify that _pool_batch works for segment feature pooling as expected by few-shot caching."""
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
            "block": {"boundary_trim_frames": 0, "frame_head": {"enabled": False}},
        },
    }
    model = WavFryClassifier(config)
    x = torch.randn(2, 16000)
    lengths = torch.tensor([16000, 12000])
    hidden_by_task, out_lengths, valid_mask = model._encode_sequence(x, lengths)
    assert isinstance(hidden_by_task, dict)
    seq = hidden_by_task["creaky"]
    feats = model._pool_batch(seq, valid_mask, out_lengths, task="creaky")
    assert feats.shape == (2, model.hidden_dim)
    logits = model.head(feats)
    assert logits.shape == (2, 1)


def test_forward_block_amp_dtype_safety():
    """Verify that _forward_block handles bfloat16 / float16 sequence under softmax without dtype mismatch."""
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
            "ablation": {"use_positional_encoding": True, "pos_embed_dim": 32},
            "block": {"boundary_trim_frames": 0, "frame_head": {"enabled": False}},
        },
    }
    model = WavFryClassifier(config)
    B, S, T, D = 2, 4, 20, model.hidden_dim
    # Simulate bfloat16 sequence from AMP
    seq_bf16 = torch.randn(B, T, D, dtype=torch.bfloat16)
    masks = torch.ones(B, S, T, dtype=torch.bool)
    seq_mask = torch.ones(B, T, dtype=torch.bool)
    bounds = torch.zeros(B, S, 2, dtype=torch.long)
    bounds[..., 1] = 1000
    valid = torch.ones(B, S, dtype=torch.bool)

    # In _forward_block under AMP, softmax internally defaults to float32 while sequence is bfloat16
    hidden_by_task = {"creaky": seq_bf16}
    lengths = torch.tensor([16000, 16000])
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = model._forward_block(hidden_by_task, seq_mask, lengths, bounds, valid)
    assert "interval_logits" in out
    logits = out["interval_logits"]["creaky"]
    assert logits.shape == (B, S)


def test_fast_save_history_csv_dictwriter():
    """Verify that save_history correctly dumps JSON and CSV via csv.DictWriter."""
    import yaml
    import shutil
    from pathlib import Path
    from wavfry.train.trainer import Trainer
    import pandas as pd

    test_dir = Path("scratch/test_save_history")
    if test_dir.exists():
        shutil.rmtree(test_dir, ignore_errors=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    defaults_path = Path(__file__).parent.parent / "config" / "_defaults.yaml"
    with open(defaults_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["model"]["backbone"] = "hf-internal-testing/tiny-random-wavlm"
    config["model"]["fine_tune_layers"] = "none"
    config["model"]["head_hidden_dim"] = 64
    config["checkpoint"]["dir"] = str(test_dir / "ckpt")
    config["tensorboard"]["log_dir"] = str(test_dir / "log")
    config["eval"]["primary_metric"] = "auc_pr"

    model = WavFryClassifier(config)
    dummy_ds = type("DummyDS", (), {"df": pd.DataFrame({"label_creaky": [0.0, 1.0], "label": [0.0, 1.0]})})()
    dummy_loader = type("DummyLoader", (), {"dataset": dummy_ds, "__len__": lambda s: 1, "__iter__": lambda s: iter([1])})()

    trainer = Trainer(
        model=model,
        train_loader=dummy_loader,
        val_loader=dummy_loader,
        config=config,
        device=torch.device("cpu"),
    )
    trainer.history = [
        {"epoch": 1, "train_loss": 0.5, "val_auc_pr": 0.8},
        {"epoch": 2, "train_loss": 0.3, "val_auc_pr": 0.9},
    ]
    trainer.save_history()

    # Check files exist in both ckpt and log dirs
    for d in [test_dir / "ckpt", test_dir / "log"]:
        assert (d / "epoch_history.json").exists()
        assert (d / "epoch_history.csv").exists()
        df = pd.read_csv(d / "epoch_history.csv")
        assert len(df) == 2
        assert list(df["val_auc_pr"]) == [0.8, 0.9]

    shutil.rmtree(test_dir, ignore_errors=True)



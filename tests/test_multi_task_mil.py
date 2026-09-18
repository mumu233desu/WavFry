import copy
import pytest
import torch
import torch.nn as nn
from wavfry.model.wavlm_classifier import WavFryClassifier
from wavfry.train.trainer import Trainer


def _make_dummy_config():
    return {
        "model": {
            "backbone": "hf-internal-testing/tiny-random-wavlm",
            "fine_tune_layers": "none",
            "use_scalar_mix": True,
            "dropout": 0.1,
            "head_hidden_dim": 64,
            "use_positional_encoding": False,
            "tasks": ["creaky", "breathy"],
        },
        "training": {
            "device": "cpu",
            "epochs": 1,
            "paradigm": "block",
            "gradient_accumulation_steps": 1,
            "grad_clip": 1.0,
            "block": {
                "batch_size": 2,
                "duration_ms": 2000,
                "hop_ms": 1000,
                "max_intervals": 4,
                "boundary_trim_frames": 0,
                "frame_head": {
                    "enabled": True,
                    "loss_weight": 0.1,
                    "mil_mode": "lse",
                    "temperature": 1.0,
                    "topk_ratio": 0.2,
                },
            },
        },
        "eval": {
            "primary_metric": "auc_pr",
            "early_stopping_patience": 3,
        },
        "tensorboard": {
            "log_pr_curve": False,
            "log_audio_samples": False,
        },
    }


def test_multi_task_frame_heads_forward_and_backward():
    config = _make_dummy_config()
    model = WavFryClassifier(config)

    assert hasattr(model, "frame_heads")
    assert "creaky" in model.frame_heads
    assert "breathy" in model.frame_heads
    assert model.frame_head is model.frame_heads["creaky"]

    batch_size = 2
    raw_audio = torch.randn(batch_size, 16000)
    lengths = torch.tensor([16000, 16000])
    interval_bounds = torch.tensor([
        [[0, 4000], [4000, 8000], [0, 0], [0, 0]],
        [[1000, 5000], [5000, 9000], [0, 0], [0, 0]],
    ])
    interval_valid = torch.tensor([
        [True, True, False, False],
        [True, True, False, False],
    ])

    outputs = model(raw_audio, lengths=lengths, interval_bounds=interval_bounds, interval_valid=interval_valid)

    assert "frame_logits" in outputs
    assert isinstance(outputs["frame_logits"], dict)
    assert "creaky" in outputs["frame_logits"]
    assert "breathy" in outputs["frame_logits"]

    frame_c = outputs["frame_logits"]["creaky"]
    frame_b = outputs["frame_logits"]["breathy"]
    assert frame_c.shape[0] == batch_size
    assert frame_b.shape[0] == batch_size

    # Test backprop through both frame heads
    loss = frame_c.sum() + frame_b.sum()
    loss.backward()

    for p in model.frame_heads["creaky"].parameters():
        assert p.grad is not None
    for p in model.frame_heads["breathy"].parameters():
        assert p.grad is not None


def test_legacy_checkpoint_remapping():
    config = _make_dummy_config()
    model = WavFryClassifier(config)

    sd = model.state_dict()
    # Create legacy state dict where frame_head is single head
    legacy_sd = copy.deepcopy(sd)
    keys_to_remove = [k for k in legacy_sd if k.startswith("frame_heads.creaky.")]
    for k in keys_to_remove:
        val = legacy_sd.pop(k)
        new_k = "frame_head." + k[len("frame_heads.creaky."):]
        legacy_sd[new_k] = val

    # Loading legacy state dict should remap frame_head -> frame_heads.creaky without missing keys
    missing, unexpected = model.load_state_dict(legacy_sd, strict=False)
    # Check that frame_heads.creaky.* are NOT in missing keys!
    creaky_missing = [k for k in missing if k.startswith("frame_heads.creaky.")]
    assert len(creaky_missing) == 0
    # Old keys must be popped, so unexpected should NOT contain frame_head.*
    legacy_unexpected = [k for k in unexpected if k.startswith("frame_head.")]
    assert len(legacy_unexpected) == 0

    # Also test strict=True on a single-task model
    single_cfg = copy.deepcopy(config)
    single_cfg["model"]["tasks"] = ["creaky"]
    single_model = WavFryClassifier(single_cfg)
    single_sd = single_model.state_dict()
    legacy_single_sd = copy.deepcopy(single_sd)
    for k in [k for k in legacy_single_sd if k.startswith("frame_heads.creaky.")]:
        val = legacy_single_sd.pop(k)
        legacy_single_sd["frame_head." + k[len("frame_heads.creaky."):]] = val
    # In strict=True, must load without raising unexpected keys error!
    single_model.load_state_dict(legacy_single_sd, strict=True)


def test_trainer_compute_block_loss_nan_handling():
    config = _make_dummy_config()
    model = WavFryClassifier(config)

    # Initialize a minimal mock Trainer instance
    trainer = object.__new__(Trainer)
    trainer.config = config
    trainer.model = model
    trainer.loss_fn = nn.BCEWithLogitsLoss()
    trainer.frame_head_enabled = True
    trainer.frame_cfg = config["training"]["block"]["frame_head"]
    trainer.device = torch.device("cpu")

    # Simulate batch outputs
    B, slots, T = 2, 4, 50
    interval_masks = torch.zeros(B, slots, T, dtype=torch.bool)
    interval_masks[0, 0, 5:15] = True
    interval_masks[0, 1, 20:30] = True
    interval_masks[1, 0, 10:20] = True
    interval_masks[1, 1, 25:35] = True

    slot_valid = torch.tensor([
        [True, True, False, False],
        [True, True, False, False],
    ])

    outputs = {
        "interval_logits": {
            "creaky": torch.randn(B, slots),
            "breathy": torch.randn(B, slots),
        },
        "frame_logits": {
            "creaky": torch.randn(B, T),
            "breathy": torch.randn(B, T),
        },
        "interval_masks": interval_masks,
        "interval_valid": slot_valid,
        "tasks": ["creaky", "breathy"],
    }

    # Case 1: CS-Dialogue batch (creaky labeled, breathy all NaN)
    # labels: [B, slots, 2] -> [label_creaky, label_breathy]
    labels_cs = torch.full((B, slots, 2), float("nan"))
    labels_cs[0, 0, 0] = 1.0  # creaky positive
    labels_cs[0, 1, 0] = 0.0  # creaky negative
    labels_cs[1, 0, 0] = 0.0
    labels_cs[1, 1, 0] = 1.0

    loss_cs, logits_by_task, targets_by_task = trainer._compute_block_loss(
        outputs, labels_cs, slot_valid, training=True
    )
    assert not torch.isnan(loss_cs)
    assert "creaky" in logits_by_task
    assert "breathy" in logits_by_task
    assert "creaky_frame" in trainer._current_losses
    assert "breathy_frame" not in trainer._current_losses

    # Case 2: UCLA batch (breathy labeled, creaky all NaN)
    labels_ucla = torch.full((B, slots, 2), float("nan"))
    labels_ucla[0, 0, 1] = 1.0  # breathy positive
    labels_ucla[0, 1, 1] = 0.0  # breathy negative
    labels_ucla[1, 0, 1] = 0.0
    labels_ucla[1, 1, 1] = 1.0

    loss_ucla, _, _ = trainer._compute_block_loss(
        outputs, labels_ucla, slot_valid, training=True
    )
    assert not torch.isnan(loss_ucla)
    assert "breathy_frame" in trainer._current_losses
    assert "creaky_frame" not in trainer._current_losses

    # Case 3: Mixed batch (both labeled on different slots)
    labels_mix = torch.full((B, slots, 2), float("nan"))
    labels_mix[0, 0, 0] = 1.0  # slot (0,0) is creaky
    labels_mix[0, 1, 0] = 0.0
    labels_mix[1, 0, 1] = 1.0  # slot (1,0) is breathy
    labels_mix[1, 1, 1] = 0.0

    loss_mix, _, _ = trainer._compute_block_loss(
        outputs, labels_mix, slot_valid, training=True
    )
    assert not torch.isnan(loss_mix)
    assert "creaky_frame" in trainer._current_losses
    assert "breathy_frame" in trainer._current_losses


def test_train_heads_gating_on_frame_loss():
    """Verify that when train_heads specifies a task as False, frame loss does NOT propagate to total_loss."""
    config = _make_dummy_config()
    config["training"]["train_heads"] = {"creaky": True, "breathy": False}
    model = WavFryClassifier(config)

    trainer = object.__new__(Trainer)
    trainer.config = config
    trainer.model = model
    trainer.loss_fn = nn.BCEWithLogitsLoss()
    trainer.frame_head_enabled = True
    trainer.frame_cfg = config["training"]["block"]["frame_head"]
    trainer.device = torch.device("cpu")

    B, slots, T = 2, 2, 20
    interval_masks = torch.zeros(B, slots, T, dtype=torch.bool)
    interval_masks[:, :, 5:15] = True
    slot_valid = torch.ones(B, slots, dtype=torch.bool)

    raw_audio = torch.randn(B, 16000)
    lengths = torch.tensor([16000, 16000])
    bounds = torch.tensor([[[0, 4000], [4000, 8000]], [[0, 4000], [4000, 8000]]])

    outputs = model(raw_audio, lengths=lengths, interval_bounds=bounds, interval_valid=slot_valid)

    # Batch with labeled breathy and labeled creaky
    labels = torch.zeros(B, slots, 2)
    labels[0, 0, 0] = 1.0  # creaky pos
    labels[0, 1, 0] = 0.0  # creaky neg
    labels[1, 0, 1] = 1.0  # breathy pos
    labels[1, 1, 1] = 0.0  # breathy neg

    total_loss, _, _ = trainer._compute_block_loss(outputs, labels, slot_valid, training=True)
    total_loss.backward()

    # Creaky heads must receive gradients
    assert any(p.grad is not None for p in model.heads["creaky"].parameters())
    assert any(p.grad is not None for p in model.frame_heads["creaky"].parameters())

    # Breathy heads must have ZERO grad because train_heads: breathy: False!
    for p in model.heads["breathy"].parameters():
        assert p.grad is None
    for p in model.frame_heads["breathy"].parameters():
        assert p.grad is None


def test_linear_softmax_pool_fp16_stability():
    """Verify linear_softmax_pool does not underflow to NaN under FP16 with extreme negative logits."""
    from wavfry.model.losses import linear_softmax_pool

    logits_fp16 = torch.full((4, 50), -60.0, dtype=torch.float16)
    mask = torch.ones((4, 50), dtype=torch.bool)
    mask[:, 25:] = False

    pooled = linear_softmax_pool(logits_fp16, mask)
    assert pooled.dtype == torch.float16
    assert not torch.isnan(pooled).any()
    assert not torch.isinf(pooled).any()

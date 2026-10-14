import pytest
import torch
import torch.nn as nn
from wavfry.model.wavlm_classifier import WavFryClassifier
from wavfry.model.losses import BayesianExpectedCrossEntropyLoss
from wavfry.train.trainer import Trainer

@pytest.fixture
def dummy_evidential_config():
    return {
        "model": {
            "backbone": "microsoft/wavlm-base-plus",
            "fine_tune_layers": "none",
            "use_scalar_mix": True,
            "dropout": 0.0,
            "head_hidden_dim": 256,
            "tasks": ["creaky", "breathy"],
            "head_type": "evidential",
        },
        "training": {
            "optimizer": "adamw_fp32",
            "backbone_lr": 0.0,
            "head_lr": 1e-4,
            "scalar_mix_lr": 1e-3,
            "weight_decay": 0.01,
            "loss": "bayes",
            "paradigm": "block",
            "train_heads": {"creaky": True, "breathy": True},
            "evidential": {
                "flow_loss_weight": 0.001,
                "evidence_floor": 10.0,
                "evidence_floor_epochs": 2,
            },
            "block": {
                "frame_head": {
                    "enabled": True,
                    "mil_mode": "linear_softmax",
                    "interval_from_frame": {
                        "enabled": True,
                        "mode": "linear_softmax",
                    }
                }
            }
        },
        "eval": {
            "primary_task": "creaky",
            "primary_metric": "auc_pr",
        }
    }

def test_evidential_model_block_forward_and_backward(dummy_evidential_config):
    # Mock WavLM encoder to avoid loading full weights in rapid unit tests
    model = WavFryClassifier(dummy_evidential_config)
    assert model.head_type == "evidential"
    assert "creaky" in model.frame_heads
    assert "breathy" in model.frame_heads

    B = 2
    T = 16000  # 1.0s at 16kHz
    num_slots = 4
    x = torch.randn(B, T)
    lengths = torch.tensor([16000, 16000])
    
    # 4 intervals per block: [start_s, end_s]
    bounds = torch.tensor([
        [[0.1, 0.3], [0.4, 0.6], [0.7, 0.9], [1.0, 1.2]],
        [[0.05, 0.25], [0.35, 0.55], [0.65, 0.85], [1.1, 1.3]],
    ])
    valid = torch.tensor([
        [True, True, True, False],
        [True, True, False, False],
    ])

    outputs = model(x, lengths, interval_bounds=bounds, interval_valid=valid)

    # Check contract outputs
    assert "interval_logits" in outputs
    assert "interval_degree" in outputs
    assert "interval_confidence" in outputs
    assert "interval_alpha" in outputs
    assert "interval_beta" in outputs
    assert "frame_evidential" in outputs

    for task in ["creaky", "breathy"]:
        deg = outputs["interval_degree"][task]
        conf = outputs["interval_confidence"][task]
        alpha = outputs["interval_alpha"][task]
        beta = outputs["interval_beta"][task]

        assert deg.shape == (B, num_slots)
        assert conf.shape == (B, num_slots)
        assert alpha.shape == (B, num_slots)
        assert beta.shape == (B, num_slots)

        # Bounds check on valid intervals
        assert (deg[valid] >= 0.0).all() and (deg[valid] <= 1.0).all()
        assert (conf[valid] >= 0.0).all() and (conf[valid] <= 1.0).all()
        assert (alpha[valid] >= 1.0).all()
        assert (beta[valid] >= 1.0).all()

    # Test Bayes Loss backward
    bayes_loss_fn = BayesianExpectedCrossEntropyLoss()
    targets = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    alpha_cr = outputs["interval_alpha"]["creaky"][valid]
    beta_cr = outputs["interval_beta"]["creaky"][valid]
    targ_cr = targets[valid]

    loss = bayes_loss_fn(alpha_cr, beta_cr, targ_cr)
    assert loss.requires_grad
    loss.backward()

    # Ensure evidential head parameters received gradients
    head = model.frame_heads["creaky"]
    assert head.stat_head[0].weight.grad is not None
    assert head.bottleneck.weight.grad is not None

    # Ensure WavLM backbone remains 100% frozen
    for p in model.wavlm.parameters():
        assert p.grad is None

def test_trainer_loss_computation_with_evidential(dummy_evidential_config):
    model = WavFryClassifier(dummy_evidential_config)
    
    # Minimal mock trainer object to test _compute_block_loss
    class MockTrainer:
        CANONICAL_TASKS = Trainer.CANONICAL_TASKS

        def __init__(self, model, config):
            self.model = model
            self.config = config
            self.primary_task = "creaky"
            self.frame_cfg = config["training"]["block"]["frame_head"]
            self.loss_fns = {
                "creaky": BayesianExpectedCrossEntropyLoss(),
                "breathy": BayesianExpectedCrossEntropyLoss(),
            }
            self.frame_loss_fns = {
                "creaky": nn.BCEWithLogitsLoss(),
                "breathy": nn.BCEWithLogitsLoss(),
            }
            self._current_losses = {}
            self._current_frame_proxy = None

        _get_task_target = Trainer._get_task_target
        _compute_block_loss = Trainer._compute_block_loss

    trainer = MockTrainer(model, dummy_evidential_config)

    B = 2
    T = 16000
    x = torch.randn(B, T)
    bounds = torch.tensor([
        [[0.1, 0.3], [0.4, 0.6], [0.0, 0.0]],
        [[0.05, 0.25], [0.35, 0.55], [0.0, 0.0]],
    ])
    valid = torch.tensor([
        [True, True, False],
        [True, True, False],
    ])
    labels = torch.tensor([
        [[1.0, 0.0], [0.0, 0.0], [float('nan'), float('nan')]],
        [[0.0, 1.0], [1.0, 0.0], [float('nan'), float('nan')]],
    ])

    outputs = model(x, interval_bounds=bounds, interval_valid=valid)
    clean_mask = torch.tensor([True, True])

    loss, logits_by_task, targets_by_task = trainer._compute_block_loss(
        outputs, labels, valid, training=True, clean_mask=clean_mask
    )

    assert loss.requires_grad
    assert "creaky" in logits_by_task
    assert "breathy" in logits_by_task
    assert "flow" in trainer._current_losses

    loss.backward()
    assert model.frame_heads["creaky"].stat_head[0].weight.grad is not None


def test_evidential_delta_checkpoint_roundtrip(dummy_evidential_config):
    """Verify that all NEPH parameters are serialized in delta mode and can be perfectly restored."""
    model = WavFryClassifier(dummy_evidential_config)

    # 1. Simulate saving delta state dict (only requires_grad=True parameters)
    delta_state_dict = {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    # Verify that flow, stat_head, and bottleneck are all included
    flow_keys = [k for k in delta_state_dict if ".flow." in k]
    assert len(flow_keys) > 0, "RealNVP flow layers must be saved in delta checkpoint"
    stat_keys = [k for k in delta_state_dict if ".stat_head." in k]
    assert len(stat_keys) > 0, "stat_head layers must be saved in delta checkpoint"
    bottle_keys = [k for k in delta_state_dict if ".bottleneck." in k]
    assert len(bottle_keys) > 0, "bottleneck layers must be saved in delta checkpoint"

    # Verify backbone wavlm.* is NOT in delta checkpoint (backbone frozen)
    backbone_keys = [k for k in delta_state_dict if k.startswith("wavlm.")]
    assert len(backbone_keys) == 0, "Frozen backbone parameters must not bloat delta checkpoint"

    # 2. Instantiate a fresh model and restore state_dict
    new_model = WavFryClassifier(dummy_evidential_config)
    missing, unexpected = new_model.load_state_dict(delta_state_dict, strict=False)

    assert len(unexpected) == 0, f"Unexpected keys during restore: {unexpected}"

    # Verify restored parameters exactly match original parameters
    for name, orig_param in delta_state_dict.items():
        restored_param = dict(new_model.named_parameters())[name]
        assert torch.allclose(orig_param, restored_param.cpu()), f"Mismatch in restored parameter: {name}"

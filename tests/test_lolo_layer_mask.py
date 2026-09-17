import pytest
import torch
import torch.nn as nn
from wavfry.model.wavlm_classifier import WavFryClassifier


def make_dummy_config():
    return {
        "model": {
            "backbone": "microsoft/wavlm-base-plus",
            "fine_tune_layers": "none",
            "use_scalar_mix": True,
            "dropout": 0.0,
            "head_hidden_dim": 64,
            "tasks": ["creaky"],
        },
        "training": {
            "paradigm": "block",
            "pooling_mode": "attn",
            "ablation": {
                "use_positional_encoding": False,
            },
            "block": {
                "duration_ms": 2000,
                "hop_ms": 1000,
                "max_intervals": 4,
                "frame_head": {
                    "enabled": False,
                },
            },
        },
    }


def test_layer_mask_mathematical_renormalization():
    """Verify that masking layer l sets its effective softmax weight to 0.0 and re-normalizes the rest to 1.0."""
    num_layers = 13
    logits = torch.randn(num_layers)

    for l in range(num_layers):
        mask = torch.ones(num_layers, dtype=torch.bool)
        mask[l] = False

        masked_logits = logits.clone()
        masked_logits[~mask] = -float("inf")
        weights = torch.softmax(masked_logits, dim=0)

        # Masked layer must be 0
        assert weights[l].item() == 0.0
        # Sum of all weights must be exactly 1.0
        assert abs(weights.sum().item() - 1.0) < 1e-6

        # Formula equivalence: w_k / sum_{j != l} w_j
        unmasked_indices = [k for k in range(num_layers) if k != l]
        manual_expected = torch.softmax(logits[unmasked_indices], dim=0)
        actual_unmasked = weights[unmasked_indices]
        assert torch.allclose(actual_unmasked, manual_expected, atol=1e-6)


def test_layer_mask_forward_pass_equivalence():
    """Test WavFryClassifier forward with layer_mask=None vs all-ones mask."""
    config = make_dummy_config()
    model = WavFryClassifier(config)
    model.eval()

    # Small dummy input: [B, T]
    x = torch.randn(2, 16000)
    lengths = torch.tensor([16000, 12000])

    with torch.no_grad():
        out_none = model(x, lengths=lengths, layer_mask=None)
        all_ones = torch.ones(13, dtype=torch.bool)
        out_ones = model(x, lengths=lengths, layer_mask=all_ones)

    assert torch.allclose(out_none["creaky"], out_ones["creaky"], atol=1e-5)

    # Now test masking layer 0 and layer 3
    for mask_idx in [0, 3, 12]:
        mask = torch.ones(13, dtype=torch.bool)
        mask[mask_idx] = False
        with torch.no_grad():
            out_masked = model(x, lengths=lengths, layer_mask=mask)
        assert out_masked["creaky"].shape == (2,)
        assert not torch.isnan(out_masked["creaky"]).any()


def test_layer_mask_forward_block_mode():
    """Test WavFryClassifier forward with layer_mask in continuous block mode."""
    config = make_dummy_config()
    model = WavFryClassifier(config)
    model.eval()

    x = torch.randn(2, 32000)
    lengths = torch.tensor([32000, 28000])
    interval_bounds = torch.tensor([
        [[1000, 5000], [6000, 12000], [0, 0], [0, 0]],
        [[2000, 8000], [10000, 20000], [22000, 26000], [0, 0]],
    ])
    interval_valid = torch.tensor([
        [True, True, False, False],
        [True, True, True, False],
    ])

    mask = torch.ones(13, dtype=torch.bool)
    mask[2] = False

    with torch.no_grad():
        out = model(x, lengths=lengths, interval_bounds=interval_bounds, interval_valid=interval_valid, layer_mask=mask)

    assert "interval_logits" in out
    logits = out["interval_logits"]["creaky"]
    assert logits.shape == (2, 4)
    assert not torch.isnan(logits[interval_valid]).any()

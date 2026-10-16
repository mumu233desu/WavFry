"""Unit tests for multi-backbone compatibility (WavLM, HuBERT, wav2vec 2.0).

Verifies:
1. Unified AutoModel loading and interface conformity across WavLM, HuBERT, and wav2vec 2.0.
2. Inviolable rule: backbone parameters remain 100% frozen when fine_tune_layers="none".
3. Forward pass equivalence and scalar mix dimension alignment across all three backbones.
4. YAML configuration integrity for ablation_backbone_*.yaml.
"""

from __future__ import annotations

import copy
from pathlib import Path
import pytest
import torch
import yaml
from transformers import HubertConfig, HubertModel, Wav2Vec2Config, Wav2Vec2Model, WavLMConfig, WavLMModel

from wavfry.model.wavlm_classifier import WavFryClassifier

BACKBONE_SPECS = [
    ("wavlm", WavLMConfig, WavLMModel),
    ("hubert", HubertConfig, HubertModel),
    ("wav2vec2", Wav2Vec2Config, Wav2Vec2Model),
]

BASE_CONFIG = {
    "model": {
        "tasks": ["creaky", "breathy"],
        "use_scalar_mix": True,
        "pooling_mode": "attn",
        "use_pos_encoding": True,
        "pos_dim": 32,
        "fine_tune_layers": "none",
        "dropout": 0.1,
        "head_hidden_dim": 128,
    },
    "training": {
        "pooling_mode": "attn",
        "block": {
            "frame_head": {
                "enabled": True,
                "interval_from_frame": {
                    "enabled": True,
                    "mode": "linear_softmax",
                    "temperature": 1.0,
                    "topk_ratio": 0.2,
                },
            }
        },
    },
}


@pytest.mark.parametrize("name,cfg_cls,model_cls", BACKBONE_SPECS)
def test_backbone_loading_and_properties(name, cfg_cls, model_cls):
    """Verify that all three backbones expose identical downsampling and layer properties."""
    cfg = cfg_cls()
    cfg.num_hidden_layers = 4
    cfg.hidden_size = 768
    raw_model = model_cls(cfg)

    model_config = copy.deepcopy(BASE_CONFIG)
    clf = WavFryClassifier(model_config, wavlm_instance=raw_model)

    # 1. Alias property check
    assert clf.backbone is clf.wavlm
    assert clf.hidden_dim == 768
    assert clf.num_layers == 4

    # 2. Inviolable frozen backbone rule
    clf.freeze_layers(model_config)
    assert clf.is_backbone_frozen is True
    for p in clf.backbone.parameters():
        assert not p.requires_grad

    # 3. Scalar mix weights: num_layers + 1
    for task in ["creaky", "breathy"]:
        assert task in clf.scalar_mix_weights
        assert clf.scalar_mix_weights[task].shape == (5,)  # 4 layers + 1 CNN output
        assert clf.scalar_mix_weights[task].requires_grad is True


@pytest.mark.parametrize("name,cfg_cls,model_cls", BACKBONE_SPECS)
def test_backbone_forward_parity(name, cfg_cls, model_cls):
    """Verify forward computation and output shape parity across backbones."""
    cfg = cfg_cls()
    cfg.num_hidden_layers = 2
    cfg.hidden_size = 768
    raw_model = model_cls(cfg)

    model_config = copy.deepcopy(BASE_CONFIG)
    clf = WavFryClassifier(model_config, wavlm_instance=raw_model)
    clf.eval()

    batch_size = 2
    audio_len = 16000  # 1.0s at 16kHz
    num_slots = 3

    x = torch.randn(batch_size, audio_len)
    lengths = torch.tensor([audio_len, audio_len])
    bounds = torch.tensor([
        [[500, 3000], [4000, 7000], [8000, 12000]],
        [[1000, 4000], [5000, 9000], [10000, 15000]],
    ])
    valid = torch.ones(batch_size, num_slots, dtype=torch.bool)

    with torch.no_grad():
        out = clf(x, lengths=lengths, interval_bounds=bounds, interval_valid=valid)

    assert "interval_logits" in out
    for task in ["creaky", "breathy"]:
        assert task in out["interval_logits"]
        logits = out["interval_logits"][task]
        assert logits.shape == (batch_size, num_slots)
        assert not torch.isnan(logits).any()


def test_backbone_configs_exist_and_valid():
    """Verify configuration files for HuBERT, wav2vec 2.0, and WavLM exist and have valid structure."""
    config_dir = Path(__file__).resolve().parents[1] / "config"
    configs = [
        ("ablation_backbone_hubert.yaml", "facebook/hubert-base-ls960"),
        ("ablation_backbone_wav2vec2.yaml", "facebook/wav2vec2-base-960h"),
        ("ablation_backbone_wavlm.yaml", "microsoft/wavlm-base-plus"),
    ]

    for fname, expected_backbone in configs:
        cpath = config_dir / fname
        assert cpath.exists(), f"Config file missing: {cpath}"

        with open(cpath, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        assert cfg["model"]["backbone"] == expected_backbone
        assert cfg["model"]["fine_tune_layers"] == "none"
        assert cfg["training"]["backbone_lr"] == 0.0
        assert cfg["training"]["paradigm"] == "block"
        assert "tasks" in cfg["model"]

import json
import tempfile
from pathlib import Path
import pandas as pd
import pytest
import torch
import torch.nn as nn

from wavfry.data.utterance_bag import (
    SpeakerCappedBagBatchSampler,
    collate_utterance_bags,
)
from wavfry.model.vsdc import LinearSoftmaxPooling
from wavfry.train.weak_supervision import compute_weak_quality_weights


class MockBagDataset:
    def __init__(self, spk_counts: dict[str, int], max_windows_per_bag: int = 4):
        self.spk_indices = {}
        rows = []
        idx = 0
        self.costs = []
        for spk, count in spk_counts.items():
            self.spk_indices[spk] = []
            for _ in range(count):
                self.spk_indices[spk].append(idx)
                rows.append({"speaker_id": spk, "duration": 2.5})
                self.costs.append(max_windows_per_bag)
                idx += 1
        self.df = pd.DataFrame(rows)
        self.speaker_pos_indices = self.spk_indices

    def __len__(self):
        return len(self.df)


def test_speaker_capped_sampler_limits():
    # Speaker A has 25 bags, Speaker B has 5 bags, Speaker C has 12 bags
    ds = MockBagDataset({"spkA": 25, "spkB": 5, "spkC": 12})
    sampler = SpeakerCappedBagBatchSampler(
        dataset=ds,
        max_windows=32,
        samples_per_epoch=100,
        max_per_speaker=8,
        seed=42,
    )

    sampler.set_epoch(1)
    batches = sampler.batches()
    all_selected = [idx for batch in batches for idx in batch]

    # Count how many items per speaker were drawn
    spk_counts = {}
    for idx in all_selected:
        spk = ds.df.iloc[idx]["speaker_id"]
        spk_counts[spk] = spk_counts.get(spk, 0) + 1

    assert spk_counts["spkA"] == 8, f"spkA had 25, should be capped at 8, got {spk_counts['spkA']}"
    assert spk_counts["spkB"] == 5, f"spkB had 5, should be 5, got {spk_counts['spkB']}"
    assert spk_counts["spkC"] == 8, f"spkC had 12, should be capped at 8, got {spk_counts['spkC']}"


def test_quality_weight_ladder(tmp_path):
    # Create mock JSON cache files
    high_cache = tmp_path / "high.json"
    high_cache.write_text(json.dumps({"parsed_prediction": {"overall_confidence": "high"}}), encoding="utf-8")

    med_cache = tmp_path / "med.json"
    med_cache.write_text(json.dumps({"parsed_prediction": {"overall_confidence": "medium"}}), encoding="utf-8")

    low_cache = tmp_path / "low.json"
    low_cache.write_text(json.dumps({"parsed_prediction": {"overall_confidence": "low"}}), encoding="utf-8")

    df = pd.DataFrame([
        {"cache_path": str(high_cache), "alignment_modes": "text_matched+acoustic"},
        {"cache_path": str(high_cache), "alignment_modes": "acoustic_only"},
        {"cache_path": str(med_cache), "alignment_modes": "acoustic_only"},
        {"cache_path": str(low_cache), "alignment_modes": "fallback"},
    ])

    cfg = {
        "quality_weighting": {
            "high_text": 1.0,
            "high_acoustic": 0.75,
            "medium": 0.50,
            "low": 0.25,
        }
    }

    weights = compute_weak_quality_weights(df, cfg)
    assert len(weights) == 4
    assert weights.iloc[0] == 1.0
    assert weights.iloc[1] == 0.75
    assert weights.iloc[2] == 0.50
    assert weights.iloc[3] == 0.25


def test_collate_utterance_bags_quality_weight():
    items = [
        {
            "waveforms": torch.randn(2, 32000),
            "lengths": torch.tensor([32000, 32000]),
            "starts": torch.tensor([0, 16000]),
            "vowels": [[0.1, 0.4]],
            "bag_label": 1.0,
            "quality_weight": 0.75,
            "metadata": {"speaker_id": "spk1"},
        },
        {
            "waveforms": torch.randn(1, 32000),
            "lengths": torch.tensor([32000]),
            "starts": torch.tensor([0]),
            "vowels": [[0.2, 0.5]],
            "bag_label": 1.0,
            "quality_weight": 1.0,
            "metadata": {"speaker_id": "spk2"},
        },
    ]

    collated = collate_utterance_bags(items)
    assert "quality_weights" in collated
    assert torch.allclose(collated["quality_weights"], torch.tensor([0.75, 1.0]))


def test_linear_softmax_mil_gradient_flow():
    # Simulate B=4 bags with K=6 vowels each
    B, K = 4, 6
    logits = torch.randn(B, K, requires_grad=True)
    mask = torch.ones(B, K, dtype=torch.bool)
    mask[0, 4:] = False  # bag 0 only has 4 valid vowels

    probs = torch.sigmoid(logits)
    pooling = LinearSoftmaxPooling()
    bag_probs = pooling(probs, mask)

    assert bag_probs.shape == (B,)
    assert (bag_probs >= 0.0).all() and (bag_probs <= 1.0).all()

    # Compute MIL loss
    loss = -torch.log(bag_probs.clamp(min=1e-7)).mean()
    loss.backward()

    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()
    # Masked positions should have 0 or near-zero gradient
    assert torch.allclose(logits.grad[0, 4:], torch.zeros(2), atol=1e-6)

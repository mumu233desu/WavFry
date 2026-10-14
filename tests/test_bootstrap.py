"""Unit tests for speaker-level cluster bootstrap and empirical p-value calculation."""

import numpy as np
import pytest

from scripts.statistical_bootstrap_audit.evaluate_paired_models_bootstrap import (
    run_speaker_cluster_bootstrap_paired,
)


def test_bootstrap_perfect_separation_no_zero_p_value():
    """Verify that when condition produces strictly worse results on all speakers,

    the empirical p-value never collapses to an illegal 0.0000,
    and is smoothed by the Davison & Hinkley continuity correction.
    """
    n_spk = 20
    preds_a_by_speaker = {}
    preds_b_by_speaker = {}
    labels_by_speaker = {}

    for i in range(n_spk):
        spk = f"SPK_{i:03d}"
        labels = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int32)
        # Model A has high logits for positive class
        base_logits = np.array([5.0, -5.0, 4.0, -4.0, 6.0, -6.0, 5.0, -5.0])
        # Model B has random / bad logits
        degraded_logits = np.array([-2.0, 2.0, -1.0, 1.0, -2.0, 2.0, -1.0, 1.0])

        preds_a_by_speaker[spk] = base_logits
        preds_b_by_speaker[spk] = degraded_logits
        labels_by_speaker[spk] = labels

    results = run_speaker_cluster_bootstrap_paired(
        preds_a_by_spk=preds_a_by_speaker,
        preds_b_by_spk=preds_b_by_speaker,
        labels_by_spk=labels_by_speaker,
        n_bootstrap=100,
        seed=42,
    )

    assert results["n_bootstrap"] == 100
    assert results["n_speakers"] == n_spk
    assert results["pr_a_mean"] > results["pr_b_mean"]

    # Even if degraded is strictly worse in 100/100 resamplings, p_value must be strictly positive (> 0.0)
    assert results["p_val_pr"] > 0.0
    # Must be bounded by 2.0 / (B + 1) ~ 0.0198
    assert results["p_val_pr"] <= 0.05
    assert results["delta_pr_mean"] > 0.0
    assert results["speaker_breakdown"]["wins_model_a"] == n_spk

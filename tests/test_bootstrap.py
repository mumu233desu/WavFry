"""Unit tests for speaker-level cluster bootstrap and empirical p-value calculation."""

import numpy as np
import pytest

from scripts.statistical_bootstrap_audit.evaluate_robustness_and_bootstrap import (
    run_speaker_cluster_bootstrap,
)


def test_bootstrap_perfect_separation_no_zero_p_value():
    """Verify that when condition produces strictly worse results on all speakers,

    the empirical p-value never collapses to an illegal 0.0000,
    and is smoothed by the Davison & Hinkley continuity correction.
    """
    n_spk = 20
    predictions_by_speaker = {}
    labels_by_speaker = {}

    for i in range(n_spk):
        spk = f"SPK_{i:03d}"
        labels = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int32)
        # Baseline has high logits for positive class
        base_logits = np.array([5.0, -5.0, 4.0, -4.0, 6.0, -6.0, 5.0, -5.0])
        # Degraded condition has random / bad logits
        degraded_logits = np.array([-2.0, 2.0, -1.0, 1.0, -2.0, 2.0, -1.0, 1.0])

        predictions_by_speaker[spk] = {
            "baseline": base_logits,
            "severe_noise": degraded_logits,
        }
        labels_by_speaker[spk] = labels

    results = run_speaker_cluster_bootstrap(
        predictions_by_speaker,
        labels_by_speaker,
        condition_names=["baseline", "severe_noise"],
        n_bootstrap=100,
        seed=42,
    )

    assert len(results) == 2
    base_res = results[0]
    deg_res = results[1]

    assert base_res["condition"] == "baseline"
    assert base_res["p_value"] == 1.0

    assert deg_res["condition"] == "severe_noise"
    # Even if degraded is strictly worse in 100/100 resamplings, p_value must be strictly positive (> 0.0)
    assert deg_res["p_value"] > 0.0
    # Must be bounded by 2.0 / (B + 1) ~ 0.0198
    assert deg_res["p_value"] <= 0.05
    assert deg_res["mean_paired_delta"] > 0.0

"""
Unit tests for confidence calibration (ECE, MCE, Brier Score, and Diagrams).
"""

from pathlib import Path
import numpy as np
import pytest
import torch

from scripts.representation_and_calibration_audit.calibration_core import (
    compute_ece_and_mce,
    compute_reliability_diagram_bins,
    plot_domain_shift_comparison,
    plot_reliability_diagram,
)


def test_perfect_calibration():
    """When empirical positive frequency equals confidence, ECE must be near 0."""
    # 1000 samples at confidence 0.2 with exactly 200 positives
    p1 = np.full(1000, 0.25)
    y1 = np.zeros(1000)
    y1[:250] = 1

    # 1000 samples at confidence 0.75 with exactly 750 positives
    p2 = np.full(1000, 0.75)
    y2 = np.zeros(1000)
    y2[:750] = 1

    probs = np.concatenate([p1, p2])
    labels = np.concatenate([y1, y2])

    res = compute_reliability_diagram_bins(probs, labels, num_bins=10)
    assert pytest.approx(0.0, abs=1e-5) == res["ece"]
    assert pytest.approx(0.0, abs=1e-5) == res["mce"]


def test_overconfident_prediction():
    """Predicting p=0.99 for all samples when true label is 0 must yield ECE ~ 0.99."""
    probs = np.full(500, 0.99)
    labels = np.zeros(500)

    ece, mce, brier = compute_ece_and_mce(probs, labels, num_bins=10)
    assert pytest.approx(0.99, abs=1e-4) == ece
    assert pytest.approx(0.99, abs=1e-4) == mce
    assert pytest.approx(0.99 ** 2, abs=1e-4) == brier


def test_underconfident_prediction():
    """Predicting p=0.02 for all samples when true label is 1 must yield ECE ~ 0.98."""
    probs = np.full(300, 0.02)
    labels = np.ones(300)

    ece, mce, brier = compute_ece_and_mce(probs, labels, num_bins=10)
    assert pytest.approx(0.98, abs=1e-4) == ece
    assert pytest.approx(0.98, abs=1e-4) == mce
    assert pytest.approx((1.0 - 0.02) ** 2, abs=1e-4) == brier


def test_brier_score_extremes():
    """Brier score should be 0.0 for perfect deterministic predictions and 1.0 for completely inverted."""
    probs_perfect = [1.0, 0.0, 1.0, 0.0]
    labels = [1.0, 0.0, 1.0, 0.0]
    _, _, brier_perf = compute_ece_and_mce(probs_perfect, labels)
    assert pytest.approx(0.0, abs=1e-6) == brier_perf

    probs_inverted = [0.0, 1.0, 0.0, 1.0]
    _, _, brier_inv = compute_ece_and_mce(probs_inverted, labels)
    assert pytest.approx(1.0, abs=1e-6) == brier_inv


def test_empty_bins_handling():
    """Calibration partition must handle sparse or empty bins safely without NaN."""
    probs = np.array([0.05, 0.95])
    labels = np.array([0, 1])

    res = compute_reliability_diagram_bins(probs, labels, num_bins=10)
    assert not np.isnan(res["ece"])
    assert not np.isnan(res["mce"])
    assert not np.isnan(res["brier_score"])
    assert np.all(~np.isnan(res["accuracies"]))
    assert np.all(~np.isnan(res["confidences"]))
    assert np.sum(res["counts"]) == 2


def test_tensor_input_compatibility():
    """Functions must seamlessly accept PyTorch tensors."""
    probs = torch.tensor([0.2, 0.8, 0.3, 0.9])
    labels = torch.tensor([0, 1, 0, 1])

    ece, mce, brier = compute_ece_and_mce(probs, labels)
    assert 0.0 <= ece <= 1.0
    assert 0.0 <= mce <= 1.0
    assert 0.0 <= brier <= 1.0


def test_plot_reliability_diagram(tmp_path: Path):
    """Reliability diagram rendering must produce PDF and PNG files."""
    np.random.seed(42)
    probs = np.random.uniform(0.0, 1.0, size=200)
    labels = (np.random.uniform(0.0, 1.0, size=200) < probs).astype(int)

    bin_data = compute_reliability_diagram_bins(probs, labels, num_bins=10)
    out_prefix = tmp_path / "test_diagram"

    fig = plot_reliability_diagram(
        bin_data,
        title="Test Calibration Diagram",
        task_name="Creaky",
        output_path=out_prefix,
    )
    assert fig is not None
    assert (tmp_path / "test_diagram.pdf").exists()
    assert (tmp_path / "test_diagram.png").exists()


def test_plot_domain_shift_comparison(tmp_path: Path):
    """Domain shift comparison plot must render multi-condition curves."""
    np.random.seed(42)
    domain_results = {}
    for dom in ["Clean", "Cafe (0dB)", "Car (0dB)"]:
        probs = np.random.uniform(0.0, 1.0, size=150)
        labels = (np.random.uniform(0.0, 1.0, size=150) < probs).astype(int)
        domain_results[dom] = compute_reliability_diagram_bins(probs, labels, num_bins=10)

    out_prefix = tmp_path / "test_domain_shift"
    fig = plot_domain_shift_comparison(
        domain_results=domain_results,
        task_name="Breathy",
        output_path=out_prefix,
    )
    assert fig is not None
    assert (tmp_path / "test_domain_shift.pdf").exists()
    assert (tmp_path / "test_domain_shift.png").exists()

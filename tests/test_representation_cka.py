"""
Unit tests for Centered Kernel Alignment (CKA) and HSIC calculations.
"""

import numpy as np
import pytest
import torch

from scripts.representation_and_calibration_audit.cka_core import (
    center_features,
    center_gram,
    compute_cross_condition_cka,
    compute_layer_cka_matrix,
    linear_cka,
    unbiased_hsic,
)


def test_linear_cka_identity():
    """Identical representation matrices must have CKA == 1.0."""
    np.random.seed(42)
    x = np.random.randn(100, 64)
    score_biased = linear_cka(x, x, unbiased=False)
    score_unbiased = linear_cka(x, x, unbiased=True)

    assert pytest.approx(1.0, abs=1e-5) == score_biased
    assert pytest.approx(1.0, abs=1e-5) == score_unbiased


def test_linear_cka_isotropic_scaling():
    """CKA must be invariant to isotropic scaling: CKA(X, alpha * X) == 1.0."""
    np.random.seed(42)
    x = np.random.randn(100, 32)
    alpha = 5.73
    score = linear_cka(x, alpha * x)
    assert pytest.approx(1.0, abs=1e-5) == score


def test_linear_cka_orthogonal_rotation():
    """CKA must be invariant to orthogonal transformation: Y = X @ R."""
    np.random.seed(42)
    x = np.random.randn(120, 32)
    # Generate random orthogonal matrix R via QR decomposition
    q, _ = np.linalg.qr(np.random.randn(32, 32))
    y = x @ q

    score_biased = linear_cka(x, y, unbiased=False)
    score_unbiased = linear_cka(x, y, unbiased=True)

    assert pytest.approx(1.0, abs=1e-4) == score_biased
    assert pytest.approx(1.0, abs=1e-4) == score_unbiased


def test_linear_cka_orthogonal_independent_representations():
    """Independent random representations should have near-zero CKA."""
    np.random.seed(123)
    x = np.random.randn(2000, 64)
    y = np.random.randn(2000, 64)

    score_biased = linear_cka(x, y, unbiased=False)
    assert score_biased < 0.05


def test_linear_cka_tensor_input():
    """linear_cka should seamlessly accept torch.Tensor on CPU or CUDA."""
    x_tensor = torch.randn(80, 48)
    y_tensor = x_tensor + 0.001 * torch.randn(80, 48)

    score = linear_cka(x_tensor, y_tensor)
    assert 0.99 <= score <= 1.0


def test_linear_cka_sample_mismatch_raises():
    """Different number of samples N must raise ValueError."""
    x = np.random.randn(50, 32)
    y = np.random.randn(60, 32)
    with pytest.raises(ValueError, match="Sample count mismatch"):
        linear_cka(x, y)


def test_unbiased_hsic_minimum_samples():
    """Unbiased HSIC requires n >= 4."""
    x = np.random.randn(3, 10)
    k = x @ x.T
    with pytest.raises(ValueError, match="n >= 4"):
        unbiased_hsic(k, k)


def test_compute_layer_cka_matrix():
    """compute_layer_cka_matrix returns symmetric matrix with 1.0 diagonal."""
    np.random.seed(42)
    layers = {
        0: np.random.randn(100, 32),
        1: np.random.randn(100, 32),
        2: np.random.randn(100, 32),
    }
    matrix, names = compute_layer_cka_matrix(layers)

    assert matrix.shape == (3, 3)
    assert names == ["CNN_L0", "Transformer_L1", "Transformer_L2"]
    # Check symmetry and diagonal
    for i in range(3):
        assert pytest.approx(1.0, abs=1e-5) == matrix[i, i]
        for j in range(i + 1, 3):
            assert matrix[i, j] == matrix[j, i]
            assert 0.0 <= matrix[i, j] <= 1.0


def test_compute_cross_condition_cka():
    """compute_cross_condition_cka computes layer-by-layer similarity correctly."""
    np.random.seed(42)
    clean = {0: np.random.randn(80, 32), 1: np.random.randn(80, 32)}
    noisy = {0: clean[0] + 0.1 * np.random.randn(80, 32), 1: np.random.randn(80, 32)}

    scores = compute_cross_condition_cka(clean, noisy)
    assert 0 in scores and 1 in scores
    assert scores[0] > 0.85
    assert scores[1] < 0.3

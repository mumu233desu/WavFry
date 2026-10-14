"""
Core mathematical functions for Centered Kernel Alignment (CKA) and
Hilbert-Schmidt Independence Criterion (HSIC).

References:
    - Kornblith et al., "Similarity of Neural Network Representations Revisited", ICML 2019.
    - Cortes et al., "Algorithms for Learning Kernels Based on Centered Alignment", JMLR 2012.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch


def _to_numpy(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Ensure input is a 2D float64 numpy array for high numerical precision."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().to(torch.float64).numpy()
    elif isinstance(x, np.ndarray):
        if x.dtype != np.float64:
            x = x.astype(np.float64)
    else:
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(x)}")
    if x.ndim != 2:
        raise ValueError(f"Expected 2D matrix (N, D), got shape {x.shape}")
    return x


def center_features(x: np.ndarray) -> np.ndarray:
    """Center feature matrix columns to have zero mean along samples."""
    return x - np.mean(x, axis=0, keepdims=True)


def center_gram(k: np.ndarray) -> np.ndarray:
    """Center a Gram matrix K: H * K * H where H = I - 1/n * 1 * 1^T."""
    n = k.shape[0]
    mean_row = np.mean(k, axis=0, keepdims=True)
    mean_col = np.mean(k, axis=1, keepdims=True)
    mean_all = np.mean(k)
    return k - mean_row - mean_col + mean_all


def unbiased_hsic(k: np.ndarray, l: np.ndarray) -> float:
    """
    Compute unbiased Hilbert-Schmidt Independence Criterion (HSIC).
    
    Given Gram matrices K = X X^T and L = Y Y^T of size (n, n), computes
    the finite-sample unbiased estimator from Song et al. (2012).
    """
    n = k.shape[0]
    if n < 4:
        raise ValueError(f"Unbiased HSIC requires n >= 4, got n={n}")
    
    # Set diagonals to 0
    k_tilde = k.copy()
    np.fill_diagonal(k_tilde, 0.0)
    l_tilde = l.copy()
    np.fill_diagonal(l_tilde, 0.0)
    
    trace_kl = float(np.sum(k_tilde * l_tilde.T))
    sum_k = float(np.sum(k_tilde))
    sum_l = float(np.sum(l_tilde))
    k_l_sum = float(np.sum(np.sum(k_tilde, axis=1) * np.sum(l_tilde, axis=1)))
    
    term1 = trace_kl
    term2 = (sum_k * sum_l) / ((n - 1) * (n - 2))
    term3 = (2.0 / (n - 2)) * k_l_sum
    
    hsic_val = (term1 + term2 - term3) / (n * (n - 3))
    return max(0.0, hsic_val)


def linear_cka(
    x: Union[torch.Tensor, np.ndarray],
    y: Union[torch.Tensor, np.ndarray],
    unbiased: bool = False,
) -> float:
    """
    Compute Linear Centered Kernel Alignment (CKA) between two representation matrices.

    Args:
        x: Feature matrix of shape (N, D1).
        y: Feature matrix of shape (N, D2).
        unbiased: If True, uses unbiased HSIC estimator on Gram matrices (O(N^2)).
                  If False, uses fast memory-efficient centered matrix Frobenius norm (O(N * D1 * D2)).

    Returns:
        Scalar CKA similarity in range [0.0, 1.0].
    """
    x_mat = _to_numpy(x)
    y_mat = _to_numpy(y)

    if x_mat.shape[0] != y_mat.shape[0]:
        raise ValueError(f"Sample count mismatch: X has {x_mat.shape[0]} samples, Y has {y_mat.shape[0]} samples")

    n = x_mat.shape[0]
    if n == 0:
        return 0.0

    if unbiased:
        # Gram matrix formulation
        k = x_mat @ x_mat.T
        l = y_mat @ y_mat.T
        hsic_xy = unbiased_hsic(k, l)
        hsic_xx = unbiased_hsic(k, k)
        hsic_yy = unbiased_hsic(l, l)
        denom = math.sqrt(hsic_xx * hsic_yy)
        if denom <= 1e-12:
            return 0.0
        return float(np.clip(hsic_xy / denom, 0.0, 1.0))
    else:
        # Fast centered Frobenius norm formulation
        x_centered = center_features(x_mat)
        y_centered = center_features(y_mat)

        # HSIC(X, Y) = ||Y^T X||_F^2
        yt_x = y_centered.T @ x_centered
        hsic_xy = float(np.sum(yt_x ** 2))

        xt_x = x_centered.T @ x_centered
        hsic_xx = float(np.sum(xt_x ** 2))

        yt_y = y_centered.T @ y_centered
        hsic_yy = float(np.sum(yt_y ** 2))

        denom = math.sqrt(hsic_xx * hsic_yy)
        if denom <= 1e-12:
            return 0.0
        return float(np.clip(hsic_xy / denom, 0.0, 1.0))


def compute_layer_cka_matrix(
    layer_representations: Dict[int, Union[torch.Tensor, np.ndarray]],
    unbiased: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute pairwise CKA matrix for all extracted hidden layers.

    Args:
        layer_representations: Mapping from layer index (0 to 12) to feature matrix (N, D).
        unbiased: Whether to use unbiased HSIC.

    Returns:
        matrix: 2D symmetric numpy array of shape (L, L) with values in [0, 1].
        layer_names: List of formatted layer names (e.g., ['CNN_L0', 'Transformer_L1', ...]).
    """
    sorted_layer_ids = sorted(layer_representations.keys())
    num_layers = len(sorted_layer_ids)
    cka_matrix = np.eye(num_layers, dtype=np.float64)

    layer_names = [
        "CNN_L0" if l == 0 else f"Transformer_L{l}"
        for l in sorted_layer_ids
    ]

    for i in range(num_layers):
        l_i = sorted_layer_ids[i]
        for j in range(i + 1, num_layers):
            l_j = sorted_layer_ids[j]
            score = linear_cka(
                layer_representations[l_i],
                layer_representations[l_j],
                unbiased=unbiased,
            )
            cka_matrix[i, j] = score
            cka_matrix[j, i] = score

    return cka_matrix, layer_names


def compute_cross_condition_cka(
    clean_representations: Dict[int, Union[torch.Tensor, np.ndarray]],
    noisy_representations: Dict[int, Union[torch.Tensor, np.ndarray]],
    unbiased: bool = False,
) -> Dict[int, float]:
    """
    Compute layer-by-layer CKA similarity between clean and degraded acoustic conditions.

    Args:
        clean_representations: Dict mapping layer index -> clean feature matrix (N, D).
        noisy_representations: Dict mapping layer index -> noisy/reverb feature matrix (N, D).
        unbiased: Whether to use unbiased HSIC.

    Returns:
        Dict mapping layer index -> CKA similarity score in [0, 1].
    """
    stability_scores: Dict[int, float] = {}
    common_layers = sorted(set(clean_representations.keys()) & set(noisy_representations.keys()))

    for layer_id in common_layers:
        score = linear_cka(
            clean_representations[layer_id],
            noisy_representations[layer_id],
            unbiased=unbiased,
        )
        stability_scores[layer_id] = score

    return stability_scores

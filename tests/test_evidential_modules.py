"""Unit tests for Natural Evidential Phonation Head (NEPH) and RealNVP Flow.

Verifies:
1. Invertibility of AffineCouplingLayer and LightweightRealNVP (reconstruction error < 1e-6).
2. Exactness of log-determinant of Jacobian.
3. Cold-start zero-initialization (identity mapping at t=0).
4. Relative density scaling and numerical boundedness of Confidence.
5. Invariance to float32 precision under AMP autocast.
"""

import math
import torch
import pytest

from wavfry.model.evidential_head import (
    AffineCouplingLayer,
    LightweightRealNVP,
    NaturalEvidentialPhonationHead,
)


def test_affine_coupling_invertibility():
    """Verify coupling layer is strictly invertible: z == f^-1(f(z))."""
    torch.manual_seed(42)
    layer = AffineCouplingLayer(dim=16, hidden_dim=32, parity=0)

    z = torch.randn(8, 20, 16, dtype=torch.float32)
    u, log_det_fwd = layer(z)
    z_recon, log_det_inv = layer.inverse(u)

    max_diff = torch.max(torch.abs(z - z_recon)).item()
    assert max_diff < 1e-6, f"Reconstruction error too large: {max_diff}"
    assert torch.allclose(log_det_fwd, -log_det_inv, atol=1e-6), "Log-det forward/inverse mismatch"


def test_coupling_zero_initialization():
    """Verify zero-initialization yields exact identity mapping at t=0."""
    layer = AffineCouplingLayer(dim=16, hidden_dim=32, parity=0)
    z = torch.randn(4, 16, dtype=torch.float32)

    u, log_det = layer(z)
    assert torch.equal(z, u), "Freshly initialized coupling layer must be exact identity"
    assert torch.equal(log_det, torch.zeros_like(log_det)), "Log det must be zero at initialization"


def test_realnvp_invertibility_and_density():
    """Verify 4-layer RealNVP invertibility and log_prob computation."""
    torch.manual_seed(42)
    flow = LightweightRealNVP(dim=16, num_layers=4, hidden_dim=32)

    z = torch.randn(4, 25, 16, dtype=torch.float32)
    u, log_det = flow(z)
    z_recon, _ = flow.inverse(u)

    max_diff = torch.max(torch.abs(z - z_recon)).item()
    assert max_diff < 1e-5, f"RealNVP reconstruction error: {max_diff}"

    log_prob, u_out = flow.log_prob(z)
    assert log_prob.shape == (4, 25)
    assert not torch.isnan(log_prob).any()
    assert not torch.isinf(log_prob).any()


def test_neph_forward_and_bounds():
    """Verify NEPH outputs Degree and Confidence bounded in [0, 1]."""
    torch.manual_seed(42)
    head = NaturalEvidentialPhonationHead(
        in_features=768,
        latent_dim=16,
        flow_layers=4,
        evidence_scale=100.0,
        temp_density=4.0,
    )

    h = torch.randn(2, 50, 768)
    out = head(h)

    assert "degree" in out
    assert "confidence" in out
    assert "alpha" in out
    assert "beta" in out

    degree = out["degree"]
    confidence = out["confidence"]
    evidence = out["evidence"]

    # Bounds check
    assert (degree >= 0.0).all() and (degree <= 1.0).all(), "Degree must be in [0, 1]"
    assert (confidence >= 0.0).all() and (confidence <= 1.0).all(), "Confidence must be in [0, 1]"
    assert (evidence >= 0.0).all(), "Evidence must be non-negative"

    # Beta relationship check: alpha = 1 + n*p, beta = 1 + n*(1-p)
    expected_alpha = 1.0 + evidence * degree
    expected_beta = 1.0 + evidence * (1.0 - degree)
    assert torch.allclose(out["alpha"], expected_alpha, atol=1e-5)
    assert torch.allclose(out["beta"], expected_beta, atol=1e-5)


def test_neph_noise_rejection_behavior():
    """Verify that features far from the manifold produce low confidence."""
    head = NaturalEvidentialPhonationHead(
        in_features=768,
        latent_dim=16,
        flow_layers=4,
        evidence_scale=100.0,
    )

    # Clean typical feature
    h_clean = torch.randn(1, 10, 768) * 0.5
    # Extreme outlier feature (simulating high-amplitude corruption)
    h_noisy = torch.randn(1, 10, 768) * 15.0

    out_clean = head(h_clean)
    out_noisy = head(h_noisy)

    mean_conf_clean = out_clean["confidence"].mean().item()
    mean_conf_noisy = out_noisy["confidence"].mean().item()

    assert mean_conf_clean > mean_conf_noisy, (
        f"Clean confidence ({mean_conf_clean}) must exceed noisy confidence ({mean_conf_noisy})"
    )
    assert mean_conf_noisy < 0.2, f"Noisy confidence must be small, got {mean_conf_noisy}"


def test_contract_methods():
    """Verify get_degree and get_confidence contract methods."""
    head = NaturalEvidentialPhonationHead(in_features=768, latent_dim=16)
    head.eval()  # Disable dropout for deterministic parity check
    h = torch.randn(3, 10, 768)

    deg = head.get_degree(h)
    conf = head.get_confidence(h)
    full_out = head(h)

    assert torch.allclose(deg, full_out["degree"], atol=1e-5)
    assert torch.allclose(conf, full_out["confidence"], atol=1e-5)


def test_bayesian_expected_ce_loss():
    """Verify BayesianExpectedCrossEntropyLoss computes digamma loss correctly."""
    from wavfry.model.losses import BayesianExpectedCrossEntropyLoss

    loss_fn = BayesianExpectedCrossEntropyLoss()

    # Confident correct positive prediction: alpha >> beta, y = 1 -> small loss
    alpha_high = torch.tensor([50.0, 100.0])
    beta_low = torch.tensor([1.0, 1.0])
    targets_pos = torch.tensor([1.0, 1.0])
    loss_small = loss_fn(alpha_high, beta_low, targets_pos)

    # Confident wrong prediction: alpha >> beta, y = 0 -> huge loss
    loss_huge = loss_fn(alpha_high, beta_low, 1.0 - targets_pos)

    assert loss_small < 0.1, f"Expected small loss for confident correct prediction, got {loss_small}"
    assert loss_huge > 3.0, f"Expected huge loss for confident wrong prediction, got {loss_huge}"

    # Vacuous / zero-evidence state: alpha = 1, beta = 1
    # Digamma difference psi(2) - psi(1) = 1.0
    alpha_vac = torch.tensor([1.0])
    beta_vac = torch.tensor([1.0])
    loss_vac = loss_fn(alpha_vac, beta_vac, torch.tensor([1.0]))
    assert torch.allclose(loss_vac, torch.tensor(1.0), atol=1e-4)


def test_pool_evidential_intervals_duration_invariance():
    """Verify duration-normalized pooling is invariant to raw phone duration."""
    from wavfry.model.losses import pool_evidential_intervals

    # Interval A: Short vowel (2 frames, 40ms)
    # Interval B: Long vowel (20 frames, 400ms)
    # Both have the identical acoustic features: degree = 0.8, evidence = 50.0
    N = 2
    T = 20
    frame_degree = torch.full((N, T), 0.8)
    frame_evidence = torch.full((N, T), 50.0)

    mask = torch.zeros(N, T, dtype=torch.bool)
    mask[0, :2] = True   # Short: 2 frames
    mask[1, :20] = True  # Long: 20 frames

    pooled = pool_evidential_intervals(frame_degree, frame_evidence, mask)

    deg_short, deg_long = pooled["degree"][0].item(), pooled["degree"][1].item()
    conf_short, conf_long = pooled["confidence"][0].item(), pooled["confidence"][1].item()

    # Both should have identical degree 0.8 and identical confidence 50 / (2 + 50) ~ 0.9615
    assert abs(deg_short - deg_long) < 1e-5, f"Degrees should match: {deg_short} vs {deg_long}"
    assert abs(conf_short - conf_long) < 1e-5, f"Confidences should match: {conf_short} vs {conf_long}"
    assert abs(deg_short - 0.8) < 0.05
    assert abs(conf_short - 50.0 / 52.0) < 0.01


def test_flow_amp_autocast_precision():
    """Verify flow computation remains in float32 without step artifacts under autocast."""
    flow = LightweightRealNVP(dim=16, num_layers=4, hidden_dim=32)
    z = torch.randn(4, 16)

    # Under CPU/CUDA autocast with bfloat16
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        log_p, u = flow.log_prob(z)

    assert log_p.dtype == torch.float32, "log_prob must remain float32 under autocast"
    assert not torch.isnan(log_p).any()


def test_neph_evidence_floor_training_only():
    """Verify evidence_floor is applied only when training=True, avoiding eval distortion."""
    head = NaturalEvidentialPhonationHead(in_features=768, latent_dim=16)
    head.evidence_floor = 10.0

    # Large out-of-distribution noise input
    noisy_h = torch.randn(2, 768) * 20.0

    # 1. Training mode: floor applies (evidence >= 10.0)
    head.train()
    out_train = head(noisy_h)
    assert (out_train["evidence"] >= 10.0).all()
    assert (out_train["confidence"] >= 10.0 / 12.0).all()

    # 2. Eval mode: floor does NOT apply (true confidence measured)
    head.eval()
    out_eval = head(noisy_h)
    assert (out_eval["evidence"] < 1.0).all()
    assert (out_eval["confidence"] < 0.35).all(), "Eval mode must not artificially inflate confidence"

"""Natural Evidential Phonation Head (NEPH) for WavFry.

Decouples continuous phonation degree (sufficient statistics) from model confidence
(latent density via Normalizing Flow) at 20ms frame resolution.
Based on Natural Posterior Networks (Charpentier et al., ICLR 2022) and Subjective
Logic evidence accumulation.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class AffineCouplingLayer(nn.Module):
    """Single invertible affine coupling transform for RealNVP.

    Splits input vector z in half. One half remains unchanged; the other half is
    scaled and shifted by functions of the first half:
        z1, z2 = split(z)
        s, t = net(z1)
        z2_prime = z2 * exp(s) + t
        z_prime = concat(z1, z2_prime)
    """

    def __init__(self, dim: int = 16, hidden_dim: int = 32, parity: int = 0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even for symmetric affine coupling, got {dim}")
        self.dim = dim
        self.split_dim = dim // 2
        self.parity = parity  # 0: first half transforms second half; 1: reversed

        in_dim = self.split_dim
        out_dim = dim - self.split_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim * 2),
        )

        # Defensive Zero-Initialization:
        # Ensures that at initialization, s=0 and t=0, so the coupling layer is an
        # exact identity mapping f(z) = z with log_det = 0.
        # This completely prevents cold-start gradient starvation!
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward mapping z -> u, returns transformed z and log determinant."""
        # Force float32 for invertible flow transformations (defending Pitfall 6)
        orig_dtype = z.dtype
        device_type = z.device.type if z.device.type in ("cuda", "cpu") else "cuda"
        with torch.autocast(device_type=device_type, enabled=False):
            z_f = z.float()
            if self.parity == 0:
                z1, z2 = z_f[..., :self.split_dim], z_f[..., self.split_dim:]
            else:
                z2, z1 = z_f[..., :self.split_dim], z_f[..., self.split_dim:]

            st = self.net(z1)
            s, t = st.chunk(2, dim=-1)
            # Bounded scaling to prevent exponential explosion
            s = 2.0 * torch.tanh(s / 2.0)

            z2_prime = z2 * torch.exp(s) + t
            log_det = s.sum(dim=-1)

            if self.parity == 0:
                out = torch.cat([z1, z2_prime], dim=-1)
            else:
                out = torch.cat([z2_prime, z1], dim=-1)

        return out.to(dtype=orig_dtype), log_det

    def inverse(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Inverse mapping u -> z, returns reconstructed z and log determinant."""
        orig_dtype = u.dtype
        device_type = u.device.type if u.device.type in ("cuda", "cpu") else "cuda"
        with torch.autocast(device_type=device_type, enabled=False):
            u_f = u.float()
            if self.parity == 0:
                u1, u2 = u_f[..., :self.split_dim], u_f[..., self.split_dim:]
            else:
                u2, u1 = u_f[..., :self.split_dim], u_f[..., self.split_dim:]

            st = self.net(u1)
            s, t = st.chunk(2, dim=-1)
            s = 2.0 * torch.tanh(s / 2.0)

            u2_orig = (u2 - t) * torch.exp(-s)
            log_det = -s.sum(dim=-1)

            if self.parity == 0:
                out = torch.cat([u1, u2_orig], dim=-1)
            else:
                out = torch.cat([u2_orig, u1], dim=-1)

        return out.to(dtype=orig_dtype), log_det


class LightweightRealNVP(nn.Module):
    """Multi-layer RealNVP Normalizing Flow for compact latent density estimation."""

    def __init__(self, dim: int = 16, num_layers: int = 4, hidden_dim: int = 32):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([
            AffineCouplingLayer(dim=dim, hidden_dim=hidden_dim, parity=i % 2)
            for i in range(num_layers)
        ])
        # Standard Normal Prior constant: log((2*pi)^(-d/2))
        self.register_buffer(
            "log_norm_const",
            torch.tensor(-0.5 * dim * math.log(2.0 * math.pi), dtype=torch.float32),
            persistent=False,
        )

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform z -> u in standard normal space."""
        total_log_det = torch.zeros(z.shape[:-1], device=z.device, dtype=torch.float32)
        cur = z.float()
        for layer in self.layers:
            cur, log_det = layer(cur)
            total_log_det = total_log_det + log_det.float()
        return cur, total_log_det

    def inverse(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform u -> z from standard normal space back to latent space."""
        total_log_det = torch.zeros(u.shape[:-1], device=u.device, dtype=torch.float32)
        cur = u.float()
        for layer in reversed(self.layers):
            cur, log_det = layer.inverse(cur)
            total_log_det = total_log_det + log_det.float()
        return cur, total_log_det

    def log_prob(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute exact log probability density log p_omega(z).

        Returns:
            log_prob: Tensor with shape z.shape[:-1] in float32
            u: Transformed standard normal latent tensor
        """
        device_type = z.device.type if z.device.type in ("cuda", "cpu") else "cuda"
        with torch.autocast(device_type=device_type, enabled=False):
            u, log_det = self.forward(z)
            # Log-density under standard Gaussian N(0, I)
            log_prior = self.log_norm_const - 0.5 * torch.sum(u.float() ** 2, dim=-1)
            log_prob = log_prior + log_det.float()
        return log_prob, u


class NaturalEvidentialPhonationHead(nn.Module):
    """Natural Evidential Phonation Head (NEPH).

    Jointly estimates phonation degree (severity on prototype continuum) and
    representation confidence (density on true speech manifold).
    """

    def __init__(
        self,
        in_features: int = 768,
        latent_dim: int = 16,
        flow_layers: int = 4,
        flow_hidden_dim: int = 32,
        stat_hidden_dim: int = 64,
        evidence_scale: float = 100.0,
        temp_density: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.latent_dim = latent_dim
        self.evidence_scale = float(evidence_scale)
        self.temp_density = float(temp_density)
        self.evidence_floor = 0.0

        # 1. Bottleneck projection: 768 -> 16
        # Note: Do NOT use LayerNorm here. LayerNorm divides by instance variance,
        # which erases radial distance and destroys OOD/noise amplitude rejection!
        self.bottleneck = nn.Linear(in_features, latent_dim)

        # 2. Normalizing Flow density estimator on 16D latent manifold
        self.flow = LightweightRealNVP(
            dim=latent_dim,
            num_layers=flow_layers,
            hidden_dim=flow_hidden_dim,
        )

        # 3. Sufficient statistics head: maps 16D latent feature to Degree logit
        self.stat_head = nn.Sequential(
            nn.Linear(latent_dim, stat_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(stat_hidden_dim, 1),
        )

        # Mode of standard normal prior for max-normalization: log p_0(0)
        self.register_buffer(
            "log_p0_max",
            torch.tensor(-0.5 * latent_dim * math.log(2.0 * math.pi), dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        h: torch.Tensor,
        evidence_floor: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """Forward pass over frame sequence features h: [B, T, D] or [N, D].

        Args:
            h: Acoustic representations from backbone/pooling.
            evidence_floor: Warmup evidence lower bound (defending Pitfall 3).

        Returns:
            Dict containing:
                degree: Phonation severity probability in [0, 1]
                confidence: Manifold representation confidence in [0, 1]
                raw_logits: Logit before sigmoid
                evidence: Scalar evidence count n
                alpha, beta: Beta distribution parameters
                log_prob: Flow log density
                latent_z: Compressed 16D features
        """
        # 1. Project to 16D compact latent space
        z = self.bottleneck(h)  # [..., latent_dim]

        # 2. Compute log density via RealNVP flow (in float32)
        log_prob, u = self.flow.log_prob(z)  # [...]

        # 3. Compute Degree logit via sufficient statistics head
        raw_logits = self.stat_head(z).squeeze(-1)  # [...]
        degree = torch.sigmoid(raw_logits)

        # 4. Max-Normalized Relative Density scaling (defending Pitfall 1 & Pitfall 2)
        # Detach log_prob so that Bayes loss updates stat_head/bottleneck,
        # while RealNVP flow density is exclusively learned on clean speech via L_flow!
        delta_log_p = (log_prob.detach().float() - self.log_p0_max) / self.temp_density
        # Clamp to avoid underflow/overflow: exp(-12) ~ 6e-6, exp(2) ~ 7.4
        relative_density = torch.exp(torch.clamp(delta_log_p, min=-12.0, max=2.0))
        evidence = self.evidence_scale * relative_density

        # Apply cold-start evidence floor during warmup strictly during training (defending Pitfall 3)
        floor = getattr(self, "evidence_floor", 0.0) if evidence_floor == 0.0 else evidence_floor
        if floor > 0.0 and self.training:
            evidence = torch.clamp_min(evidence, float(floor))

        evidence = evidence.to(dtype=h.dtype)
        degree = degree.to(dtype=h.dtype)
        raw_logits = raw_logits.to(dtype=h.dtype)

        # 5. Beta distribution parameters (Second-order uncertainty)
        # alpha = 1 + n * p,  beta = 1 + n * (1 - p)
        alpha = 1.0 + evidence * degree
        beta = 1.0 + evidence * (1.0 - degree)

        # 6. Confidence metric: Confidence = n / (2 + n) in [0, 1]
        confidence = evidence / (2.0 + evidence)

        return {
            "degree": degree,
            "confidence": confidence,
            "raw_logits": raw_logits,
            "evidence": evidence,
            "alpha": alpha,
            "beta": beta,
            "log_prob": log_prob,
            "log_p": log_prob,
            "latent_z": z,
            "latent_u": u,
        }

    def get_degree(self, h: torch.Tensor) -> torch.Tensor:
        """Unified contract method for extracting Degree probability."""
        z = self.bottleneck(h)
        return torch.sigmoid(self.stat_head(z).squeeze(-1))

    def get_confidence(self, h: torch.Tensor) -> torch.Tensor:
        """Unified contract method for extracting Confidence."""
        z = self.bottleneck(h)
        log_prob, _ = self.flow.log_prob(z)
        delta = (log_prob.float() - self.log_p0_max) / self.temp_density
        rel_dens = torch.exp(torch.clamp(delta, min=-12.0, max=2.0))
        n = self.evidence_scale * rel_dens
        return (n / (2.0 + n)).to(dtype=h.dtype)

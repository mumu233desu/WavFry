import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean', pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        if pos_weight is not None:
            if not isinstance(pos_weight, torch.Tensor):
                pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(self, inputs, targets):
        # inputs are logits
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=self.pos_weight, reduction="none"
        )
        p = torch.sigmoid(inputs)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = bce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

def logmeanexp_pool(logits, mask, temperature=1.0):
    """Pool valid frame logits with length-normalized LogMeanExp in float32 for AMP stability."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    orig_dtype = logits.dtype
    logits_fp32 = logits.float()
    masked = logits_fp32.masked_fill(~mask, -1e4)
    count = mask.sum(dim=-1).clamp_min(1).float()
    pooled = (
        torch.logsumexp(masked * temperature, dim=-1) - count.log()
    ) / temperature
    return pooled.to(dtype=orig_dtype)


def linear_softmax_pool(logits, mask):
    """Pool frame logits using probability-weighted averaging (numerically stable)."""
    orig_dtype = logits.dtype
    logits_fp32 = logits.float()
    probs = torch.sigmoid(logits_fp32).masked_fill(~mask, 0.0)
    sum_probs = probs.sum(dim=-1, keepdim=True)
    count = mask.sum(dim=-1, keepdim=True).clamp_min(1.0).float()
    # When all frames have vanishing probability (e.g. extremely negative logits),
    # fallback to uniform average across valid frames to prevent weights from vanishing
    # and collapsing the pooled logit to 0.0 (prob 0.50).
    is_vanishing = sum_probs < 1e-7
    uniform_weights = mask.float() / count
    norm_weights = probs / sum_probs.clamp_min(1e-7)
    weights = torch.where(is_vanishing, uniform_weights, norm_weights)
    pooled = (logits_fp32 * weights).sum(dim=-1)
    return pooled.to(dtype=orig_dtype)


def pool_frame_logits(logits, mask, mode="lse", topk_ratio=0.2, temperature=1.0):
    """Return one MIL logit per interval from ``[N, T]`` frame logits."""
    if not torch.jit.is_tracing():
        if logits.ndim != 2 or mask.shape != logits.shape:
            raise ValueError("logits and mask must both have shape [N, T]")
        if not mask.any(dim=-1).all():
            raise ValueError("each interval must contain at least one valid frame")
    if mode == "lse":
        return logmeanexp_pool(logits, mask, temperature)
    if mode == "max":
        return logits.masked_fill(~mask, -1e4).max(dim=-1).values
    if mode == "linear_softmax":
        return linear_softmax_pool(logits, mask)
    if mode == "top_k_mean":
        count = mask.sum(dim=-1)
        k = (count.float() * float(topk_ratio)).long().clamp_min(1)
        values = logits.masked_fill(~mask, -1e4).sort(dim=-1, descending=True).values
        return torch.stack([values[i, : int(k[i].item())].mean() for i in range(values.size(0))])
    raise ValueError(f"Unknown frame MIL pooling mode: {mode}")


class BayesianExpectedCrossEntropyLoss(nn.Module):
    """Bayesian Expected Cross-Entropy Loss for Beta/Dirichlet evidential distributions.

    Computes the exact analytic expectation of cross-entropy under the second-order
    Beta(alpha, beta) distribution:
        L = y * (psi(alpha + beta) - psi(alpha)) + (1 - y) * (psi(alpha + beta) - psi(beta))
    where psi is the digamma function.
    """

    def __init__(self, pos_weight=None, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        if pos_weight is not None:
            if not isinstance(pos_weight, torch.Tensor):
                pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Bayes cross-entropy loss.

        Args:
            alpha: Positive evidence parameter [..., ] (>= 1.0)
            beta: Negative evidence parameter [..., ] (>= 1.0)
            targets: Binary ground truth in {0, 1} [..., ]
        """
        # Ensure float32 for digamma numerical precision
        alpha_f = alpha.float().clamp_min(1.0)
        beta_f = beta.float().clamp_min(1.0)
        targets_f = targets.float()
        s_f = alpha_f + beta_f

        psi_s = torch.digamma(s_f)
        term_pos = psi_s - torch.digamma(alpha_f)  # Expected -log(p)
        term_neg = psi_s - torch.digamma(beta_f)  # Expected -log(1-p)

        if self.pos_weight is not None:
            term_pos = term_pos * self.pos_weight

        loss = targets_f * term_pos + (1.0 - targets_f) * term_neg

        if self.reduction == "mean":
            return loss.mean().to(dtype=alpha.dtype)
        elif self.reduction == "sum":
            return loss.sum().to(dtype=alpha.dtype)
        return loss.to(dtype=alpha.dtype)


def pool_evidential_intervals(
    frame_degree: torch.Tensor,
    frame_evidence: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Pool frame-level evidential predictions over interval masks using Subjective Logic.

    Applies duration-normalized cumulative fusion (defending Pitfall 4) to ensure
    evidence accumulation is invariant to raw phone duration.

    Args:
        frame_degree: [N, T] frame phonation degree in [0, 1]
        frame_evidence: [N, T] frame evidence count (>= 0)
        mask: [N, T] boolean mask selecting frames in each interval
        weights: Optional [N, T] frame importance weights

    Returns:
        Dict with interval-level:
            degree: [N] pooled degree in [0, 1]
            confidence: [N] pooled confidence in [0, 1]
            alpha: [N] pooled Beta alpha parameter
            beta: [N] pooled Beta beta parameter
            evidence: [N] duration-normalized total evidence
            logits: [N] equivalent logit for backward compatibility
    """
    orig_dtype = frame_degree.dtype
    deg_f = frame_degree.float()
    ev_f = frame_evidence.float()
    mask_f = mask.float()

    if weights is not None:
        w = weights.float() * mask_f
    else:
        w = mask_f

    # Duration normalization weights: sum(w_norm) == 1 per interval
    w_sum = w.sum(dim=-1, keepdim=True).clamp_min(1e-7)
    w_norm = w / w_sum

    # Subjective Logic cumulative evidence integration
    e1 = (w_norm * ev_f * deg_f).sum(dim=-1)
    e0 = (w_norm * ev_f * (1.0 - deg_f)).sum(dim=-1)

    alpha = 1.0 + e1
    beta = 1.0 + e0
    s = alpha + beta

    degree = alpha / s
    evidence = e1 + e0
    confidence = evidence / (2.0 + evidence)

    # Invert to logit for seamless drop-in compatibility with existing evaluators
    clamped_deg = degree.clamp(min=1e-7, max=1.0 - 1e-7)
    logits = torch.log(clamped_deg / (1.0 - clamped_deg))

    return {
        "degree": degree.to(dtype=orig_dtype),
        "confidence": confidence.to(dtype=orig_dtype),
        "alpha": alpha.to(dtype=orig_dtype),
        "beta": beta.to(dtype=orig_dtype),
        "evidence": evidence.to(dtype=orig_dtype),
        "logits": logits.to(dtype=orig_dtype),
    }


def get_loss_fn(config, pos_weight=None):
    loss_type = config['training'].get('loss', 'focal')
    if loss_type == 'bayes':
        return BayesianExpectedCrossEntropyLoss(pos_weight=pos_weight)
    elif loss_type == 'focal':
        return FocalLoss(
            alpha=config['training'].get('focal_alpha', 0.75),
            gamma=config['training'].get('focal_gamma', 2.0),
            pos_weight=pos_weight,
        )
    elif loss_type == 'weighted_bce':
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        return nn.BCEWithLogitsLoss()


from .vsdc import LinearSoftmaxPooling, SpeakerAnchoredInfoNCELoss



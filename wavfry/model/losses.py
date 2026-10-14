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


def get_loss_fn(config, pos_weight=None):
    loss_type = config['training'].get('loss', 'focal')
    if loss_type == 'focal':
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



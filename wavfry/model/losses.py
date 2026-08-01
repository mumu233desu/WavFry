import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs are logits
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

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
    """Pool valid frame logits with length-normalized LogMeanExp."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    masked = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    count = mask.sum(dim=-1).clamp_min(1).to(logits.dtype)
    return (
        torch.logsumexp(masked * temperature, dim=-1) - count.log()
    ) / temperature


def linear_softmax_pool(logits, mask):
    """Pool frame logits using probability-weighted averaging."""
    probs = torch.sigmoid(logits).masked_fill(~mask, 0.0)
    weights = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return (logits * weights).sum(dim=-1)


def pool_frame_logits(logits, mask, mode="lse", topk_ratio=0.2, temperature=1.0):
    """Return one MIL logit per interval from ``[N, T]`` frame logits."""
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
            gamma=config['training'].get('focal_gamma', 2.0)
        )
    elif loss_type == 'weighted_bce':
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        return nn.BCEWithLogitsLoss()

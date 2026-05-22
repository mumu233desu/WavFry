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

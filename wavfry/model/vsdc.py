"""VSDC-Net: Vowel-Synchronous Decoupled & Contrastive Mean-Teacher Network.

Implements cross-disciplinary Weakly-Supervised Temporal Action Localization (WTAL)
and Sound Event Detection (SED) mechanisms adapted for fine-grained paralinguistic
breathy voice detection:
1. LinearSoftmaxPooling (TALNet SED closed-form gradient cutoff pooling)
2. VowelBreathySuppressor (BaS-Net dual-branch foreground/background decoupling + L1 sparsity)
3. SpeakerAnchoredInfoNCELoss (Same-speaker acoustic timbre cancellation)
4. MeanTeacherEngine (EMA teacher with 1D median filter & temperature-sharpened consistency)
"""
from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearSoftmaxPooling(nn.Module):
    """
    TALNet-style Linear Softmax Pooling on probabilities:
        y = sum(p^2) / (sum(p) + eps)
    Gradient w.r.t p_i is positive only when p_i > y / 2, naturally cutting off
    positive gradients to low-confidence background frames / vowels.
    Always computed in float32 and clamped to [1e-5, 1.0 - 1e-5] to eliminate
    BCE gradient explosions and underflow deadlocks under AMP (bf16/fp16).
    """
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = float(eps)

    def forward(self, p: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        p_fp32 = p.float()
        if mask is not None:
            p_fp32 = p_fp32.masked_fill(~mask.bool(), 0.0)
        sum_p_sq = torch.sum(p_fp32 ** 2, dim=-1)
        sum_p = torch.sum(p_fp32, dim=-1) + self.eps
        y_bag = sum_p_sq / sum_p
        # Maintain in float32 and clamp to open interval to prevent extreme BCE derivatives
        return torch.clamp(y_bag, min=1e-5, max=1.0 - 1e-5)


class VowelBreathySuppressor(nn.Module):
    """
    BaS-Net inspired dual-branch background suppression network operating on vowel sequences:
    - temporal_enc: Conv1D + step-wise LayerNorm over vowel instances to capture local phonetic context.
    - supp_filter: Temporal foreground mask generator w in [0, 1].
    - base branch: Predicts breathy probability without suppression.
    - supp branch: Foreground probability modulated strictly in probability space: w * p_base.
    - bg branch: Background probability modulated strictly in probability space: (1 - w) * p_base.
    - proj_head: Projection head producing L2-normalized embeddings for InfoNCE.
    """
    def __init__(
        self,
        in_features: int = 768,
        hidden_dim: int = 256,
        proj_dim: int = 128,
        pooling_mode: str = "linear_softmax",
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.hidden_dim = int(hidden_dim)
        self.proj_dim = int(proj_dim)

        # Temporal encoder over vowel sequence.
        # Conv1d + token-wise LayerNorm avoids sequence-length and padding contamination.
        self.conv = nn.Conv1d(self.in_features, self.hidden_dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.2)

        # Foreground suppression filter: outputs w_k in [0, 1]
        self.supp_filter = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Base classifier producing base logits
        self.classifier = nn.Linear(self.hidden_dim, 1)

        # Metric projection head for contrastive learning
        self.proj_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.proj_dim),
            nn.LayerNorm(self.proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.proj_dim, self.proj_dim),
        )

        if pooling_mode == "linear_softmax":
            self.pooling = LinearSoftmaxPooling()
        else:
            self.pooling = LinearSoftmaxPooling()

    def forward(
        self,
        h_vowels: torch.Tensor,
        vowel_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            h_vowels: [B, K, in_features] vowel instance representations.
            vowel_mask: [B, K] boolean mask of valid vowel slots.
        Returns:
            Dict containing probs_base, probs_supp, probs_bg, weights,
            bag_prob_base, bag_prob_supp, bag_prob_bg, z_proj.
        """
        B, K, D = h_vowels.shape
        # Conv1d expects [B, D, K]
        x = h_vowels.transpose(1, 2)
        h_conv = self.conv(x).transpose(1, 2)  # [B, K, hidden_dim]
        feat_t = self.dropout(self.act(self.norm(h_conv)))  # [B, K, hidden_dim]
        feat = feat_t.transpose(1, 2)  # [B, hidden_dim, K]

        # Foreground attention weights: [B, K]
        weights = self.supp_filter(feat).squeeze(1)
        if vowel_mask is not None:
            weights = weights.masked_fill(~vowel_mask.bool(), 0.0)

        # 1. Base Branch (unfiltered)
        logits_base = self.classifier(feat_t).squeeze(-1)
        probs_base = torch.sigmoid(logits_base)
        if vowel_mask is not None:
            probs_base = probs_base.masked_fill(~vowel_mask.bool(), 0.0)
        bag_prob_base = self.pooling(probs_base, vowel_mask)

        # 2. Suppressed Branch (foreground modulated in probability space: w * p_base)
        # Guarantees that w -> 0 forces bag_prob_supp -> 0, triggering severe BCE loss penalty.
        probs_supp = weights * probs_base
        if vowel_mask is not None:
            probs_supp = probs_supp.masked_fill(~vowel_mask.bool(), 0.0)
        bag_prob_supp = self.pooling(probs_supp, vowel_mask)

        # 3. Background Branch (background modulated in probability space: (1 - w) * p_base)
        bg_weights = 1.0 - weights
        if vowel_mask is not None:
            bg_weights = bg_weights.masked_fill(~vowel_mask.bool(), 0.0)
        probs_bg = bg_weights * probs_base
        if vowel_mask is not None:
            probs_bg = probs_bg.masked_fill(~vowel_mask.bool(), 0.0)
        bag_prob_bg = self.pooling(probs_bg, vowel_mask)

        # 4. Projected L2-normalized embeddings for contrastive metric learning
        z_proj = F.normalize(self.proj_head(feat_t), p=2, dim=-1)

        return {
            "probs_base": probs_base,
            "probs_supp": probs_supp,
            "probs_bg": probs_bg,
            "weights": weights,
            "bag_prob_base": bag_prob_base,
            "bag_prob_supp": bag_prob_supp,
            "bag_prob_bg": bag_prob_bg,
            "z_proj": z_proj,
        }


class SpeakerAnchoredInfoNCELoss(nn.Module):
    """
    Speaker-Anchored InfoNCE contrastive loss:
    Uses pure modal negative bag from the same speaker as hard negatives to cancel out speaker voice timbre.
    Gracefully handles missing same-speaker negatives by returning zero loss without breaking autograd.
    """
    def __init__(self, temperature: float = 0.07, topk: int = 2):
        super().__init__()
        self.temperature = float(temperature)
        self.topk = int(topk)

    def forward(
        self,
        pos_z: torch.Tensor,
        pos_weights: torch.Tensor,
        neg_z: torch.Tensor | None,
        neg_mask: torch.Tensor | None = None,
        pos_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if neg_z is None or neg_z.numel() == 0:
            return torch.tensor(0.0, device=pos_z.device, requires_grad=True)
        valid_negatives = neg_z[neg_mask.bool()] if neg_mask is not None else neg_z
        if valid_negatives.shape[0] == 0:
            return torch.tensor(0.0, device=pos_z.device, requires_grad=True)

        if pos_mask is not None:
            # Mask out padding dummy tokens so they cannot be selected as top-k anchors
            masked_pos_weights = pos_weights.masked_fill(~pos_mask.bool(), -1e9)
            valid_pos_count = int(pos_mask.sum().item())
            num_candidates = min(self.topk, valid_pos_count)
        else:
            masked_pos_weights = pos_weights
            num_candidates = min(self.topk, pos_z.shape[0])

        if num_candidates <= 0:
            return torch.tensor(0.0, device=pos_z.device, requires_grad=True)

        _, topk_idx = torch.topk(masked_pos_weights, k=num_candidates)
        anchors = pos_z[topk_idx]

        # Ensure unit L2 normalization for strict cosine similarity
        anchors = F.normalize(anchors, p=2, dim=-1)
        valid_negatives = F.normalize(valid_negatives, p=2, dim=-1)

        pos_prototype = F.normalize(anchors.mean(dim=0, keepdim=True), p=2, dim=-1)
        sim_pos = torch.mm(anchors, pos_prototype.T) / self.temperature
        sim_neg = torch.mm(anchors, valid_negatives.T) / self.temperature
        logits = torch.cat([sim_pos, sim_neg], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=pos_z.device)
        return F.cross_entropy(logits, labels)


class MeanTeacherEngine:
    """
    EMA Mean-Teacher distillation engine with temporal 1D median filtering & temperature sharpening.
    Maintains a shadow copy of VowelBreathySuppressor updated via EMA.
    """
    def __init__(
        self,
        student_module: nn.Module,
        ema_decay: float = 0.98,
        median_window: int = 3,
        temp_sharp: float = 0.5,
        conf_margin: float = 0.35,
    ):
        self.student = student_module
        self.teacher = copy.deepcopy(student_module)
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.ema_decay = float(ema_decay)
        self.median_window = int(median_window)
        self.temp_sharp = float(temp_sharp)
        self.conf_margin = float(conf_margin)

    @torch.no_grad()
    def update_teacher(self) -> None:
        """Update teacher parameters via Exponential Moving Average (EMA)."""
        for s_param, t_param in zip(self.student.parameters(), self.teacher.parameters()):
            t_param.data.mul_(self.ema_decay).add_(s_param.data, alpha=1.0 - self.ema_decay)

    @torch.no_grad()
    def generate_refined_pseudolabels(
        self,
        h_vowels: torch.Tensor,
        vowel_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Run teacher forward pass, apply 1D median filtering across valid vowels
        (isolated from padding zeros to protect boundary/clause-final breathy vowels),
        and sharpen via temperature scaling.
        """
        if hasattr(self.teacher, "eval"):
            self.teacher.eval()
        t_out = self.teacher(h_vowels, vowel_mask)
        raw_probs = t_out["probs_supp"]  # [B, K]

        # 1D median filter of window size 3 per sample within valid vowel bounds
        pad_size = self.median_window // 2
        median_probs = raw_probs.clone()
        if vowel_mask is not None:
            for b in range(raw_probs.shape[0]):
                k_val = int(vowel_mask[b].sum().item())
                if k_val <= 0:
                    continue
                row = raw_probs[b, :k_val].unsqueeze(0).unsqueeze(0)  # [1, 1, k_val]
                if k_val < self.median_window:
                    median_probs[b, :k_val] = row.squeeze()
                else:
                    row_pad = F.pad(row, (pad_size, pad_size), mode="replicate")
                    row_unfold = row_pad.unfold(dimension=-1, size=self.median_window, step=1)
                    median_probs[b, :k_val] = row_unfold.median(dim=-1).values.squeeze()
                if k_val < raw_probs.shape[1]:
                    median_probs[b, k_val:] = 0.0
        else:
            padded = F.pad(raw_probs.unsqueeze(1), (pad_size, pad_size), mode="replicate")
            unfolded = padded.unfold(dimension=-1, size=self.median_window, step=1)
            median_probs = unfolded.median(dim=-1).values.squeeze(1)

        # Temperature sharpening: T=0.5 -> power = 1/0.5 = 2.0
        power = 1.0 / max(self.temp_sharp, 1e-4)
        p_pow = median_probs ** power
        q_pseudo = p_pow / (p_pow + (1.0 - median_probs) ** power + 1e-8)

        if vowel_mask is not None:
            q_pseudo = q_pseudo.masked_fill(~vowel_mask.bool(), 0.0)
        return q_pseudo

    def consistency_loss(
        self,
        student_probs: torch.Tensor,
        pseudo_labels: torch.Tensor,
        vowel_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Consistency loss gated by confidence margin:
        Only penalize positions where teacher confidence satisfies |q_pseudo - 0.5| > conf_margin.
        """
        conf_gate = (pseudo_labels - 0.5).abs() > self.conf_margin
        if vowel_mask is not None:
            conf_gate = conf_gate & vowel_mask.bool()
        if not conf_gate.any():
            return torch.tensor(0.0, device=student_probs.device, requires_grad=True)
        diff_sq = (student_probs - pseudo_labels) ** 2
        return diff_sq[conf_gate].mean()

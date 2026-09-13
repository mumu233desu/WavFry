from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import WavLMModel


class WavFryClassifier(nn.Module):
    """WavLM classifier supporting the original segment and block paradigms."""

    def __init__(self, config: dict[str, Any], wavlm_instance=None):
        super().__init__()
        self.backbone_name = config["model"]["backbone"]
        self.wavlm = wavlm_instance or WavLMModel.from_pretrained(self.backbone_name)
        self.wavlm.config.apply_spec_augment = False
        self.wavlm.config.mask_time_prob = 0.0
        self.wavlm.config.mask_feature_prob = 0.0

        self.hidden_dim = self.wavlm.config.hidden_size
        self.num_layers = self.wavlm.config.num_hidden_layers
        self.use_scalar_mix = config["model"].get("use_scalar_mix", True)
        if self.use_scalar_mix:
            self.scalar_mix_weights = nn.Parameter(torch.zeros(self.num_layers + 1))

        training_cfg = config["training"]
        self.pooling_mode = training_cfg.get("pooling_mode", "mean")
        self.supported_pooling_modes = {
            "mean": 1,
            "mean_topk": 2,
            "mean_attn": 2,
            "attn": 1,
        }
        if self.pooling_mode not in self.supported_pooling_modes:
            raise ValueError(
                f"Unsupported pooling_mode: '{self.pooling_mode}'. "
                f"Supported modes: {list(self.supported_pooling_modes.keys())}"
            )

        self.topk_ratio = float(training_cfg.get("topk_ratio", 0.2))
        self.topk_min_frames = int(training_cfg.get("topk_min_frames", 3))
        self.use_pos_encoding = training_cfg.get("ablation", {}).get("use_positional_encoding", False)
        self.pos_dim = int(training_cfg.get("ablation", {}).get("pos_embed_dim", 32))
        if "attn" in self.pooling_mode:
            self.attention_head = nn.Linear(self.hidden_dim, 1)
            if self.use_pos_encoding:
                self.pos_scorer = nn.Linear(self.pos_dim, 1, bias=False)
                nn.init.zeros_(self.pos_scorer.weight)

        head_hidden = config["model"].get("head_hidden_dim", 256)
        dropout = config["model"].get("dropout", 0.1)
        in_features = self.hidden_dim * self.supported_pooling_modes[self.pooling_mode]
        self.head = nn.Sequential(
            nn.Linear(in_features, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

        self.boundary_trim_frames = max(
            0, int(training_cfg.get("block", {}).get("boundary_trim_frames", 0))
        )
        frame_cfg = training_cfg.get("frame_head", {})
        self.frame_head_enabled = bool(frame_cfg.get("enabled", False))
        if self.frame_head_enabled:
            self.frame_head = nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, 1),
            )
            if frame_cfg.get("mil_mode", "lse") == "lse_mean":
                self.frame_mil_fusion = nn.Linear(2, 1)

        self.freeze_layers(config)
        if training_cfg.get("gradient_checkpointing", False):
            self.wavlm.gradient_checkpointing_enable()

    def freeze_layers(self, config: dict[str, Any]) -> None:
        for param in self.wavlm.parameters():
            param.requires_grad = False
        # fine_tune_layers specifies 0-indexed Transformer encoder layers (0..11).
        # e.g., "4-8" unfreezes layers 4, 5, 6, 7, 8 (5 layers), keeping 0-3 (4 layers) and 9-11 (3 layers) frozen.
        fine_tune_layers = config["model"].get("fine_tune_layers", "4-8")
        if fine_tune_layers == "all":
            trainable_indices = list(range(self.num_layers))
        elif fine_tune_layers == "none":
            trainable_indices = []
        elif isinstance(fine_tune_layers, str):
            start, end = map(int, fine_tune_layers.split("-"))
            trainable_indices = list(range(start, end + 1))
        else:
            trainable_indices = list(fine_tune_layers)
        for i, layer in enumerate(self.wavlm.encoder.layers):
            if i in trainable_indices:
                for param in layer.parameters():
                    param.requires_grad = True

    def _encode_sequence(self, x: torch.Tensor, lengths: torch.Tensor):
        outputs = self.wavlm(x, output_hidden_states=True)
        if self.use_scalar_mix:
            hidden = torch.stack(outputs.hidden_states, dim=0)
            weights = torch.softmax(self.scalar_mix_weights, dim=0)
            sequence = (hidden * weights.view(-1, 1, 1, 1)).sum(dim=0)
        else:
            sequence = outputs.last_hidden_state
        out_lengths = self.wavlm._get_feat_extract_output_lengths(lengths).to(sequence.device)
        out_lengths = out_lengths.clamp(min=1, max=sequence.size(1))
        valid_mask = torch.arange(sequence.size(1), device=sequence.device).unsqueeze(0) < out_lengths.unsqueeze(1)
        return sequence, out_lengths, valid_mask

    @staticmethod
    def _relative_positions(mask: torch.Tensor) -> torch.Tensor:
        """Return positions normalized independently inside each mask span."""
        positions = torch.arange(mask.size(-1), device=mask.device, dtype=torch.float32)
        first = torch.argmax(mask.to(torch.int64), dim=-1, keepdim=True)
        counts = mask.sum(dim=-1, keepdim=True).clamp(min=1)
        rel = (positions - first.float()) / (counts.float() - 1).clamp(min=1)
        return rel.clamp(0.0, 1.0).masked_fill(~mask, 0.0)

    def _pool_one(self, x: torch.Tensor, mask: torch.Tensor, relative_positions: torch.Tensor | None = None) -> torch.Tensor:
        """Pool one sequence using the configured segment-level representation."""
        float_mask = mask.unsqueeze(-1).to(dtype=x.dtype)
        x_mean = (x * float_mask).sum(dim=0) / float_mask.sum(dim=0).clamp_min(1e-9)
        if self.pooling_mode == "mean":
            return x_mean
        elif self.pooling_mode == "mean_attn":
            scores = self.attention_head(x).squeeze(-1)
            if self.use_pos_encoding:
                rel = relative_positions if relative_positions is not None else self._relative_positions(mask.unsqueeze(0))[0]
                pos_emb = self._sinusoidal_pos_encoding(rel.unsqueeze(0), self.pos_dim)[0]
                scores = scores + self.pos_scorer(pos_emb).squeeze(-1)
            scores = scores.masked_fill(~mask, -1e4)
            weights = torch.softmax(scores, dim=-1)
            x_attn = (x * weights.unsqueeze(-1)).sum(dim=0)
            return torch.cat([x_mean, x_attn], dim=-1)
        elif self.pooling_mode == "attn":
            scores = self.attention_head(x).squeeze(-1)
            if self.use_pos_encoding:
                rel = relative_positions if relative_positions is not None else self._relative_positions(mask.unsqueeze(0))[0]
                pos_emb = self._sinusoidal_pos_encoding(rel.unsqueeze(0), self.pos_dim)[0]
                scores = scores + self.pos_scorer(pos_emb).squeeze(-1)
            scores = scores.masked_fill(~mask, -1e4)
            weights = torch.softmax(scores, dim=-1)
            return (x * weights.unsqueeze(-1)).sum(dim=0)
        elif self.pooling_mode == "mean_topk":
            scores = torch.norm(x, p=2, dim=-1).masked_fill(~mask, -1e4)
            order = torch.argsort(scores, descending=True)
            sorted_x = x[order]
            mask_sum = int(mask.sum().item())
            k = min(max(int(mask_sum * self.topk_ratio), self.topk_min_frames), mask_sum)
            return torch.cat([x_mean, sorted_x[:max(k, 1)].mean(dim=0)], dim=-1)
        else:
            raise ValueError(f"Unknown pooling mode: '{self.pooling_mode}'")

    def _pool_batch(self, x: torch.Tensor, mask: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return torch.stack([
            self._pool_one(x[i], mask[i]) for i in range(x.size(0))
        ])

    def _interval_frame_masks(
        self,
        interval_bounds: torch.Tensor,
        slot_valid: torch.Tensor,
        lengths: torch.Tensor,
        frame_count: int,
    ) -> torch.Tensor:
        """Vectorize sample-space interval bounds into WavLM frame masks."""
        conv_stride = 320
        audio_lengths = lengths.unsqueeze(1)
        start_samples = interval_bounds[..., 0].clamp_min(0)
        start_samples = torch.minimum(start_samples, audio_lengths)
        end_samples = interval_bounds[..., 1].clamp_min(0)
        end_samples = torch.minimum(end_samples, audio_lengths)

        is_positive = end_samples > start_samples
        start_frames = (start_samples // conv_stride).clamp(min=0, max=frame_count)
        end_frames = ((end_samples + conv_stride - 1) // conv_stride).clamp(min=0, max=frame_count)
        end_frames = torch.where(
            is_positive & (end_frames <= start_frames),
            (start_frames + 1).clamp(max=frame_count),
            end_frames
        )

        frame_positions = torch.arange(frame_count, device=interval_bounds.device).view(1, 1, -1)
        return (
            slot_valid.unsqueeze(-1)
            & is_positive.unsqueeze(-1)
            & (frame_positions >= start_frames.unsqueeze(-1))
            & (frame_positions < end_frames.unsqueeze(-1))
        )

    def _forward_block(
        self,
        sequence: torch.Tensor,
        seq_mask: torch.Tensor,
        lengths: torch.Tensor,
        interval_bounds: torch.Tensor,
        slot_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        interval_masks = self._interval_frame_masks(
            interval_bounds, slot_valid, lengths, sequence.size(1)
        )
        trim = 0
        # Trimming is applied in frame space after the convolutional stride is known.
        # Invalid or very short spans remain unchanged instead of becoming empty.
        if hasattr(self, "boundary_trim_frames"):
            trim = self.boundary_trim_frames
        if trim > 0:
            for b in range(interval_masks.size(0)):
                for s in range(interval_masks.size(1)):
                    indices = interval_masks[b, s].nonzero(as_tuple=False).flatten()
                    if indices.numel() > 2 * trim:
                        interval_masks[b, s, indices[:trim]] = False
                        interval_masks[b, s, indices[-trim:]] = False
        interval_masks = interval_masks & seq_mask.unsqueeze(1)
        # A trimmed interval with no usable frames is excluded from all losses.
        slot_valid = slot_valid & interval_masks.any(dim=-1)
        batch, slots, _ = interval_masks.shape
        if self.pooling_mode == "mean":
            mask_expanded = interval_masks.unsqueeze(-1).to(sequence.dtype)
            counts = mask_expanded.sum(dim=2).clamp_min(1.0)
            pooled_tensor = (sequence.unsqueeze(1) * mask_expanded).sum(dim=2) / counts
        else:
            pooled = []
            for b in range(batch):
                rows = []
                for s in range(slots):
                    mask = interval_masks[b, s]
                    rel = self._relative_positions(mask.unsqueeze(0))[0] if self.use_pos_encoding else None
                    rows.append(self._pool_one(sequence[b], mask, rel))
                pooled.append(torch.stack(rows))
            pooled_tensor = torch.stack(pooled)
        interval_logits = self.head(pooled_tensor).squeeze(-1)
        result = {
            "interval_logits": interval_logits,
            "interval_masks": interval_masks,
            "interval_valid": slot_valid,
        }
        if self.frame_head_enabled:
            result["frame_logits"] = self.frame_head(sequence).squeeze(-1)
        return result

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        interval_bounds: torch.Tensor | None = None,
        interval_valid: torch.Tensor | None = None,
    ):
        sequence, out_lengths, valid_mask = self._encode_sequence(x, lengths)
        if interval_bounds is not None:
            if interval_valid is None:
                interval_valid = torch.ones(interval_bounds.shape[:2], dtype=torch.bool, device=x.device)
            return self._forward_block(sequence, valid_mask, lengths, interval_bounds, interval_valid)
        x_final = self._pool_batch(sequence, valid_mask, out_lengths)
        return self.head(x_final).squeeze(-1)

    @staticmethod
    def _sinusoidal_pos_encoding(pos: torch.Tensor, d_model: int) -> torch.Tensor:
        i = torch.arange(d_model // 2, device=pos.device, dtype=pos.dtype)
        freq = 1.0 / (10000.0 ** (2.0 * i / d_model))
        angles = pos.unsqueeze(-1) * freq
        return torch.cat([angles.sin(), angles.cos()], dim=-1)

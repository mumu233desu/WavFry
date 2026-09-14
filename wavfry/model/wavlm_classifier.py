from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModel, WavLMModel


class WavFryClassifier(nn.Module):
    """WavLM multi-task classifier supporting both segment and block paradigms.

    Features:
    - Multi-head multi-task architecture (e.g. 'creaky', 'breathy').
    - Task-specific learnable scalar mixing across all 13 WavLM layers.
    - Task-specific temporal attention pooling with lengths-based padding masking.
    - Full Transformer self-attention padding mask protection in WavLM backbone.
    """

    def __init__(self, config: dict[str, Any], wavlm_instance=None):
        super().__init__()
        self.config = config
        self.backbone_name = config["model"]["backbone"]
        if wavlm_instance is not None:
            self.wavlm = wavlm_instance
        else:
            try:
                self.wavlm = AutoModel.from_pretrained(self.backbone_name, local_files_only=True)
            except Exception:
                self.wavlm = AutoModel.from_pretrained(self.backbone_name)
        for attr in ["apply_spec_augment", "mask_time_prob", "mask_feature_prob"]:
            if hasattr(self.wavlm.config, attr):
                setattr(self.wavlm.config, attr, 0.0 if "prob" in attr else False)

        self.hidden_dim = self.wavlm.config.hidden_size
        self.num_layers = self.wavlm.config.num_hidden_layers

        # Tasks initialization: preserve canonical order ("creaky", "breathy")
        tasks = config["model"].get("tasks", ["creaky"])
        if isinstance(tasks, str):
            tasks = [tasks]
        canonical_order = ["creaky", "breathy"]
        self.tasks = [t for t in canonical_order if t in tasks] + [t for t in tasks if t not in canonical_order]

        self.use_scalar_mix = config["model"].get("use_scalar_mix", True)
        if self.use_scalar_mix:
            # Independent scalar mix weights per task
            self.scalar_mix_weights = nn.ParameterDict({
                task: nn.Parameter(torch.zeros(self.num_layers + 1))
                for task in self.tasks
            })

        training_cfg = config.get("training", {})
        self.pooling_mode = training_cfg.get("pooling_mode", "attn")
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

        # Task-specific attention heads
        if "attn" in self.pooling_mode:
            self.attention_heads = nn.ModuleDict({
                task: nn.Linear(self.hidden_dim, 1)
                for task in self.tasks
            })
            if self.use_pos_encoding:
                self.pos_scorer = nn.Linear(self.pos_dim, 1, bias=False)
                nn.init.zeros_(self.pos_scorer.weight)

        head_hidden = config["model"].get("head_hidden_dim", 256)
        dropout = config["model"].get("dropout", 0.1)
        in_features = self.hidden_dim * self.supported_pooling_modes[self.pooling_mode]

        # Task-specific classification heads
        self.heads = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(in_features, head_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, 1),
            )
            for task in self.tasks
        })

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

    @property
    def head(self) -> nn.Module:
        """Backward compatibility alias for primary head."""
        if "creaky" in self.heads:
            return self.heads["creaky"]
        return next(iter(self.heads.values()))

    @property
    def attention_head(self) -> nn.Module:
        """Backward compatibility alias for primary attention head."""
        if hasattr(self, "attention_heads"):
            if "creaky" in self.attention_heads:
                return self.attention_heads["creaky"]
            return next(iter(self.attention_heads.values()))
        raise AttributeError("attention_heads not configured for current pooling mode")

    def freeze_layers(self, config: dict[str, Any]) -> None:
        for param in self.wavlm.parameters():
            param.requires_grad = False
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

    def _encode_sequence(self, x: torch.Tensor, lengths: torch.Tensor | None = None):
        """Encode waveform batch through WavLM with sample-level padding attention mask."""
        batch_size, max_len = x.shape
        if lengths is not None:
            lengths_clamped = lengths.to(x.device).clamp(min=1, max=max_len)
            attention_mask = (torch.arange(max_len, device=x.device).unsqueeze(0) < lengths_clamped.unsqueeze(1)).long()
        else:
            attention_mask = None

        outputs = self.wavlm(x, attention_mask=attention_mask, output_hidden_states=True)

        hidden_by_task: dict[str, torch.Tensor] = {}
        if self.use_scalar_mix:
            hidden_stack = torch.stack(outputs.hidden_states, dim=0)  # [13, B, T, D]
            for task in self.tasks:
                weights = torch.softmax(self.scalar_mix_weights[task], dim=0)
                hidden_by_task[task] = (hidden_stack * weights.view(-1, 1, 1, 1)).sum(dim=0)
        else:
            for task in self.tasks:
                hidden_by_task[task] = outputs.last_hidden_state

        first_seq = next(iter(hidden_by_task.values()))
        sample_lengths = lengths if lengths is not None else torch.full((batch_size,), max_len, device=x.device)
        out_lengths = self.wavlm._get_feat_extract_output_lengths(sample_lengths).to(x.device)
        out_lengths = out_lengths.clamp(min=1, max=first_seq.size(1))
        valid_mask = torch.arange(first_seq.size(1), device=x.device).unsqueeze(0) < out_lengths.unsqueeze(1)
        return hidden_by_task, out_lengths, valid_mask

    @staticmethod
    def _relative_positions(mask: torch.Tensor) -> torch.Tensor:
        """Return positions normalized independently inside each mask span."""
        positions = torch.arange(mask.size(-1), device=mask.device, dtype=torch.float32)
        first = torch.argmax(mask.to(torch.int64), dim=-1, keepdim=True)
        counts = mask.sum(dim=-1, keepdim=True).clamp(min=1)
        rel = (positions - first.float()) / (counts.float() - 1).clamp(min=1)
        return rel.clamp(0.0, 1.0).masked_fill(~mask, 0.0)

    def _pool_one(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        task: str,
        relative_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pool one sequence using the configured segment-level representation for a given task."""
        float_mask = mask.unsqueeze(-1).to(dtype=x.dtype)
        x_mean = (x * float_mask).sum(dim=0) / float_mask.sum(dim=0).clamp_min(1e-9)

        if self.pooling_mode == "mean":
            return x_mean
        elif self.pooling_mode == "attn":
            scores = self.attention_heads[task](x).squeeze(-1)
            if self.use_pos_encoding:
                rel = relative_positions if relative_positions is not None else self._relative_positions(mask.unsqueeze(0))[0]
                pos_emb = self._sinusoidal_pos_encoding(rel.unsqueeze(0), self.pos_dim)[0]
                scores = scores + self.pos_scorer(pos_emb).squeeze(-1)
            scores = scores.masked_fill(~mask, -1e4)
            weights = torch.softmax(scores, dim=-1)
            return (x * weights.unsqueeze(-1)).sum(dim=0)
        elif self.pooling_mode == "mean_attn":
            scores = self.attention_heads[task](x).squeeze(-1)
            if self.use_pos_encoding:
                rel = relative_positions if relative_positions is not None else self._relative_positions(mask.unsqueeze(0))[0]
                pos_emb = self._sinusoidal_pos_encoding(rel.unsqueeze(0), self.pos_dim)[0]
                scores = scores + self.pos_scorer(pos_emb).squeeze(-1)
            scores = scores.masked_fill(~mask, -1e4)
            weights = torch.softmax(scores, dim=-1)
            x_attn = (x * weights.unsqueeze(-1)).sum(dim=0)
            return torch.cat([x_mean, x_attn], dim=-1)
        elif self.pooling_mode == "mean_topk":
            scores = torch.norm(x, p=2, dim=-1).masked_fill(~mask, -1e4)
            order = torch.argsort(scores, descending=True)
            sorted_x = x[order]
            mask_sum = int(mask.sum().item())
            k = min(max(int(mask_sum * self.topk_ratio), self.topk_min_frames), mask_sum)
            return torch.cat([x_mean, sorted_x[:max(k, 1)].mean(dim=0)], dim=-1)
        else:
            raise ValueError(f"Unknown pooling mode: '{self.pooling_mode}'")

    def _attn_pool(self, hidden: torch.Tensor, valid_mask: torch.Tensor | None, task: str) -> torch.Tensor:
        """Attention pooling for segment/utterance mode with exact padding mask from feature extractor."""
        logits = self.attention_heads[task](hidden).squeeze(-1)  # [B, T_frames]
        if valid_mask is not None:
            logits = logits.masked_fill(~valid_mask, -1e4)

        weights = torch.softmax(logits, dim=-1)
        return (weights.unsqueeze(-1) * hidden).sum(dim=1)  # [B, D]

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
            end_frames,
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
        hidden_by_task: dict[str, torch.Tensor],
        seq_mask: torch.Tensor,
        lengths: torch.Tensor,
        interval_bounds: torch.Tensor,
        slot_valid: torch.Tensor,
    ) -> dict[str, Any]:
        first_seq = next(iter(hidden_by_task.values()))
        frame_count = first_seq.size(1)
        interval_masks = self._interval_frame_masks(
            interval_bounds, slot_valid, lengths, frame_count
        )
        trim = getattr(self, "boundary_trim_frames", 0)
        if trim > 0:
            for b in range(interval_masks.size(0)):
                for s in range(interval_masks.size(1)):
                    indices = interval_masks[b, s].nonzero(as_tuple=False).flatten()
                    if indices.numel() > 2 * trim:
                        interval_masks[b, s, indices[:trim]] = False
                        interval_masks[b, s, indices[-trim:]] = False
        interval_masks = interval_masks & seq_mask.unsqueeze(1)
        slot_valid = slot_valid & interval_masks.any(dim=-1)
        batch, slots, _ = interval_masks.shape

        interval_logits_by_task: dict[str, torch.Tensor] = {}
        for task in self.tasks:
            sequence = hidden_by_task[task]
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
                        rows.append(self._pool_one(sequence[b], mask, task, rel))
                    pooled.append(torch.stack(rows))
                pooled_tensor = torch.stack(pooled)
            interval_logits_by_task[task] = self.heads[task](pooled_tensor).squeeze(-1)

        result: dict[str, Any] = {
            "interval_logits": interval_logits_by_task,
            "interval_masks": interval_masks,
            "interval_valid": slot_valid,
            "tasks": self.tasks,
        }
        if self.frame_head_enabled:
            result["frame_logits"] = self.frame_head(first_seq).squeeze(-1)
        return result

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
        interval_bounds: torch.Tensor | None = None,
        interval_valid: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Forward pass supporting multi-task block and segment inference.

        Returns:
            In block mode:
                Dict with "interval_logits" -> {task: Tensor[B, slots]}, "interval_valid" -> Tensor[B, slots]
            In segment mode:
                Dict with {task: Tensor[B]}
        """
        hidden_by_task, out_lengths, valid_mask = self._encode_sequence(x, lengths)
        if interval_bounds is not None:
            if interval_valid is None:
                interval_valid = torch.ones(interval_bounds.shape[:2], dtype=torch.bool, device=x.device)
            return self._forward_block(hidden_by_task, valid_mask, lengths, interval_bounds, interval_valid)

        # Segment / Utterance mode
        outputs: dict[str, torch.Tensor] = {}
        for task in self.tasks:
            sequence = hidden_by_task[task]
            mask_f = valid_mask.unsqueeze(-1).to(sequence.dtype) if valid_mask is not None else None
            if self.pooling_mode == "mean":
                counts = mask_f.sum(dim=1).clamp_min(1.0) if mask_f is not None else sequence.size(1)
                pooled = (sequence * mask_f).sum(dim=1) / counts if mask_f is not None else sequence.mean(dim=1)
            elif self.pooling_mode in ("mean_attn", "mean_topk"):
                counts = mask_f.sum(dim=1).clamp_min(1.0) if mask_f is not None else sequence.size(1)
                mean_p = (sequence * mask_f).sum(dim=1) / counts if mask_f is not None else sequence.mean(dim=1)
                attn_p = self._attn_pool(sequence, valid_mask, task)
                pooled = torch.cat([mean_p, attn_p], dim=-1)
            else:
                pooled = self._attn_pool(sequence, valid_mask, task)
            outputs[task] = self.heads[task](pooled).squeeze(-1)
        return outputs

    @staticmethod
    def _sinusoidal_pos_encoding(pos: torch.Tensor, d_model: int) -> torch.Tensor:
        i = torch.arange(d_model // 2, device=pos.device, dtype=pos.dtype)
        freq = 1.0 / (10000.0 ** (2.0 * i / d_model))
        angles = pos.unsqueeze(-1) * freq
        return torch.cat([angles.sin(), angles.cos()], dim=-1)

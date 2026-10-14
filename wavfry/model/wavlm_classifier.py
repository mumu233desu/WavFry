import logging
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModel, WavLMModel

logger = logging.getLogger(__name__)


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
        self.backbone_name = config["model"].get("backbone", config["model"].get("wavlm_model", "microsoft/wavlm-base-plus"))
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
        self.extract_layer = config["model"].get("extract_layer", None)
        if self.use_scalar_mix and self.extract_layer is not None:
            logger.warning(
                f"model.extract_layer={self.extract_layer} is configured, but use_scalar_mix=True. "
                f"Scalar mix will take precedence. Set use_scalar_mix=False to use single-layer extraction."
            )
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
        model_cfg = config.get("model", {})
        ablation_cfg = training_cfg.get("ablation", {})
        self.use_pos_encoding = bool(
            model_cfg.get("use_positional_encoding", False)
            or ablation_cfg.get("use_positional_encoding", False)
        )
        self.pos_dim = int(
            model_cfg.get("pos_embed_dim", ablation_cfg.get("pos_embed_dim", 32))
        )

        # Task-specific attention heads
        if "attn" in self.pooling_mode:
            self.attention_heads = nn.ModuleDict({
                task: nn.Linear(self.hidden_dim, 1, bias=False)
                for task in self.tasks
            })
            if self.use_pos_encoding:
                self.pos_scorer = nn.Linear(self.pos_dim, 1, bias=False)
                nn.init.zeros_(self.pos_scorer.weight)
                inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.pos_dim // 2, dtype=torch.float32) * 2.0 / self.pos_dim))
                self.register_buffer("inv_freq", inv_freq, persistent=False)

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
        frame_cfg = training_cfg.get("block", {}).get("frame_head", {})
        self.frame_head_enabled = bool(frame_cfg.get("enabled", False))
        interval_from_frame_cfg = frame_cfg.get("interval_from_frame", {})
        self.interval_from_frame_enabled = bool(interval_from_frame_cfg.get("enabled", False))
        self.interval_from_frame_cfg = {
            "mode": interval_from_frame_cfg.get("mode", "lse"),
            "temperature": float(interval_from_frame_cfg.get("temperature", 1.0)),
            "topk_ratio": float(interval_from_frame_cfg.get("topk_ratio", 0.2)),
        }
        if self.interval_from_frame_enabled:
            if not self.frame_head_enabled:
                raise ValueError(
                    "training.block.frame_head.interval_from_frame.enabled requires "
                    "training.block.frame_head.enabled=true"
                )
            mode = self.interval_from_frame_cfg["mode"]
            if mode not in {"lse", "max", "linear_softmax", "top_k_mean", "lse_mean"}:
                raise ValueError(f"Unsupported interval_from_frame mode: {mode}")
            if self.interval_from_frame_cfg["temperature"] <= 0:
                raise ValueError("interval_from_frame.temperature must be positive")
            if not 0 < self.interval_from_frame_cfg["topk_ratio"] <= 1:
                raise ValueError("interval_from_frame.topk_ratio must be in (0, 1]")
        if self.frame_head_enabled:
            self.frame_heads = nn.ModuleDict({
                task: nn.Sequential(
                    nn.LayerNorm(self.hidden_dim),
                    nn.Linear(self.hidden_dim, 1),
                )
                for task in self.tasks
            })
            if (frame_cfg.get("mil_mode", "lse") == "lse_mean"
                    or (self.interval_from_frame_enabled and self.interval_from_frame_cfg["mode"] == "lse_mean")):
                self.frame_mil_fusions = nn.ModuleDict({
                    task: nn.Linear(2, 1) for task in self.tasks
                })

        vsdc_cfg = training_cfg.get("vsdc", {})
        self.vsdc_enabled = bool(vsdc_cfg.get("enabled", False))
        if self.vsdc_enabled and "breathy" in self.tasks:
            from .vsdc import VowelBreathySuppressor
            bas_cfg = vsdc_cfg.get("bas_net", {})
            supp_hidden = int(bas_cfg.get("hidden_dim", 256))
            proj_dim = int(vsdc_cfg.get("speaker_contrast", {}).get("proj_dim", 128))
            self.breathy_suppressor = VowelBreathySuppressor(
                in_features=self.hidden_dim,
                hidden_dim=supp_hidden,
                proj_dim=proj_dim,
                pooling_mode=vsdc_cfg.get("mil_pooling", "linear_softmax"),
            )
        else:
            self.breathy_suppressor = None

        self.freeze_layers(config)
        if self.is_backbone_frozen:
            self.wavlm.eval()
            if training_cfg.get("mixed_precision") == "bf16":
                self.wavlm.to(torch.bfloat16)
        if training_cfg.get("gradient_checkpointing", False):
            self.wavlm.gradient_checkpointing_enable()

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "is_backbone_frozen", False):
            self.wavlm.eval()
        return self

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

    @property
    def frame_head(self) -> nn.Module | None:
        """Backward compatibility alias for primary frame head."""
        if not getattr(self, "frame_head_enabled", False):
            return None
        if hasattr(self, "frame_heads"):
            if "creaky" in self.frame_heads:
                return self.frame_heads["creaky"]
            return next(iter(self.frame_heads.values()))
        return None

    @property
    def frame_mil_fusion(self) -> nn.Module | None:
        """Backward compatibility alias for primary frame MIL fusion."""
        if hasattr(self, "frame_mil_fusions"):
            if "creaky" in self.frame_mil_fusions:
                return self.frame_mil_fusions["creaky"]
            return next(iter(self.frame_mil_fusions.values()))
        return None

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        """Remap legacy single-task keys to multi-task keys (defaulting to 'creaky')."""
        # 1. scalar_mix_weights -> scalar_mix_weights.creaky
        legacy_smw = prefix + "scalar_mix_weights"
        if legacy_smw in state_dict:
            new_k = prefix + "scalar_mix_weights.creaky"
            if new_k not in state_dict:
                state_dict[new_k] = state_dict[legacy_smw]
            state_dict.pop(legacy_smw, None)

        # 1b. If current model disables scalar mix, discard all scalar_mix_weights keys
        if not self.use_scalar_mix:
            for k in list(state_dict.keys()):
                if k.startswith(prefix + "scalar_mix_weights"):
                    state_dict.pop(k, None)

        # 2. attention_head.* -> attention_heads.creaky.*
        legacy_attn_prefix = prefix + "attention_head."
        for k in list(state_dict.keys()):
            if k.startswith(legacy_attn_prefix):
                new_k = prefix + "attention_heads.creaky." + k[len(legacy_attn_prefix):]
                if new_k not in state_dict:
                    state_dict[new_k] = state_dict[k]
                state_dict.pop(k, None)

        # 3. head.* -> heads.creaky.*
        legacy_head_prefix = prefix + "head."
        for k in list(state_dict.keys()):
            if k.startswith(legacy_head_prefix):
                new_k = prefix + "heads.creaky." + k[len(legacy_head_prefix):]
                if new_k not in state_dict:
                    state_dict[new_k] = state_dict[k]
                state_dict.pop(k, None)

        # 4. frame_head.* -> frame_heads.creaky.*
        legacy_frame_head_prefix = prefix + "frame_head."
        for k in list(state_dict.keys()):
            if k.startswith(legacy_frame_head_prefix):
                new_k = prefix + "frame_heads.creaky." + k[len(legacy_frame_head_prefix):]
                if new_k not in state_dict:
                    state_dict[new_k] = state_dict[k]
                state_dict.pop(k, None)

        # 5. frame_mil_fusion.* -> frame_mil_fusions.creaky.*
        legacy_fusion_prefix = prefix + "frame_mil_fusion."
        for k in list(state_dict.keys()):
            if k.startswith(legacy_fusion_prefix):
                new_k = prefix + "frame_mil_fusions.creaky." + k[len(legacy_fusion_prefix):]
                if new_k not in state_dict:
                    state_dict[new_k] = state_dict[k]
                state_dict.pop(k, None)

        # 6. Discard legacy attention_heads.*.bias if current model has no attention heads or bias=False
        has_attn = hasattr(self, "attention_heads")
        for task in self.tasks:
            bias_k = prefix + f"attention_heads.{task}.bias"
            if bias_k in state_dict:
                if not has_attn or (task in self.attention_heads and getattr(self.attention_heads[task], "bias", None) is None):
                    state_dict.pop(bias_k, None)

        # 7. Discard legacy pos_scorer if positional encoding is disabled
        if not self.use_pos_encoding:
            for k in list(state_dict.keys()):
                if k.startswith(prefix + "pos_scorer."):
                    state_dict.pop(k, None)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

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

    @property
    def is_backbone_frozen(self) -> bool:
        """Derived from live requires_grad state so runtime re-freezing (e.g.
        TTA adaptation) can never desynchronize the flag from reality."""
        return not any(p.requires_grad for p in self.wavlm.parameters())

    def _encode_sequence(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
        layer_mask: torch.Tensor | None = None,
    ):
        """Encode waveform batch through WavLM with sample-level padding attention mask and optional layer mask."""
        batch_size, max_len = x.shape
        if lengths is not None:
            lengths_clamped = lengths.to(x.device).clamp(min=1, max=max_len)
            attention_mask = (torch.arange(max_len, device=x.device).unsqueeze(0) < lengths_clamped.unsqueeze(1)).long()
        else:
            attention_mask = None

        if getattr(self, "is_backbone_frozen", False):
            with torch.no_grad():
                wavlm_dtype = next(self.wavlm.parameters()).dtype
                x_in = x.to(wavlm_dtype) if x.dtype != wavlm_dtype else x
                outputs = self.wavlm(x_in, attention_mask=attention_mask, output_hidden_states=True)
        else:
            outputs = self.wavlm(x, attention_mask=attention_mask, output_hidden_states=True)

        hidden_by_task: dict[str, torch.Tensor] = {}
        if self.use_scalar_mix:
            hidden_states = (
                [h.detach() for h in outputs.hidden_states]
                if getattr(self, "is_backbone_frozen", False)
                else outputs.hidden_states
            )
            task_weight_list = []
            for task in self.tasks:
                if layer_mask is not None:
                    if layer_mask.shape[0] != self.num_layers + 1:
                        raise ValueError(
                            f"layer_mask length ({layer_mask.shape[0]}) must equal num_layers + 1 ({self.num_layers + 1})"
                        )
                    if not layer_mask.any():
                        raise ValueError("layer_mask cannot be all-False; at least one active layer is required.")
                    # Dynamic Softmax re-normalization for Leave-One-Layer-Out (LOLO) ablation
                    masked_logits = self.scalar_mix_weights[task].clone()
                    masked_logits[~layer_mask.to(masked_logits.device)] = -float("inf")
                    task_weight_list.append(torch.softmax(masked_logits, dim=0))
                else:
                    task_weight_list.append(torch.softmax(self.scalar_mix_weights[task], dim=0))
            task_weights = torch.stack(task_weight_list, dim=0)  # [num_tasks, num_layers]

            # Vectorized layer scalar mix via GEMM across all tasks
            H = torch.stack(hidden_states, dim=0)
            num_layers, b_sz, t_sz, d_sz = H.shape
            flat_H = H.view(num_layers, -1)
            mixed = (task_weights.to(flat_H.dtype) @ flat_H).view(len(self.tasks), b_sz, t_sz, d_sz)
            for idx, task in enumerate(self.tasks):
                hidden_by_task[task] = mixed[idx]
        else:
            if layer_mask is not None:
                logger.warning("layer_mask is provided but use_scalar_mix=False; layer_mask will be ignored.")

            if self.extract_layer is not None and hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                total_h = len(outputs.hidden_states)
                if isinstance(self.extract_layer, str) and self.extract_layer.lower() in ("last", "-1"):
                    target_idx = -1
                elif isinstance(self.extract_layer, str) and self.extract_layer.lower() in ("cnn", "feat", "0"):
                    target_idx = 0
                else:
                    try:
                        target_idx = int(self.extract_layer)
                    except (ValueError, TypeError):
                        target_idx = -1

                if target_idx < 0:
                    target_idx = max(0, total_h + target_idx)
                else:
                    target_idx = min(target_idx, total_h - 1)

                selected_hidden = outputs.hidden_states[target_idx]
            else:
                selected_hidden = outputs.last_hidden_state

            selected_hidden = (
                selected_hidden.detach()
                if getattr(self, "is_backbone_frozen", False)
                else selected_hidden
            )
            for task in self.tasks:
                hidden_by_task[task] = selected_hidden

        if not torch.is_autocast_enabled():
            head_dtype = next(self.heads.parameters()).dtype
            for task in self.tasks:
                if hidden_by_task[task].dtype != head_dtype:
                    hidden_by_task[task] = hidden_by_task[task].to(head_dtype)

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

    def _pool_batch(
        self,
        hidden: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
        task: str = "creaky",
    ) -> torch.Tensor:
        """Backward-compatibility pooling method for segment representation."""
        mask_f = valid_mask.unsqueeze(-1).to(hidden.dtype) if valid_mask is not None else None
        if self.pooling_mode == "mean":
            counts = mask_f.sum(dim=1).clamp_min(1.0) if mask_f is not None else hidden.size(1)
            return (hidden * mask_f).sum(dim=1) / counts if mask_f is not None else hidden.mean(dim=1)
        elif self.pooling_mode in ("mean_attn", "mean_topk"):
            counts = mask_f.sum(dim=1).clamp_min(1.0) if mask_f is not None else hidden.size(1)
            mean_p = (hidden * mask_f).sum(dim=1) / counts if mask_f is not None else hidden.mean(dim=1)
            attn_p = self._attn_pool(hidden, valid_mask, task)
            return torch.cat([mean_p, attn_p], dim=-1)
        else:
            return self._attn_pool(hidden, valid_mask, task)

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
        has_valid = interval_masks.any(dim=-1, keepdim=True)

        # Precompute positional bias once for all tasks if enabled
        pos_bias = None
        if self.use_pos_encoding and self.pooling_mode in ("attn", "mean_attn"):
            rel = self._relative_positions(interval_masks)
            pos_emb = self._sinusoidal_pos_encoding(rel, self.pos_dim)
            pos_bias = self.pos_scorer(pos_emb).squeeze(-1)

        interval_logits_by_task: dict[str, torch.Tensor] = {}
        frame_logits_by_task: dict[str, torch.Tensor] | None = None
        if self.frame_head_enabled:
            frame_logits_by_task = {
                task: self.frame_heads[task](hidden_by_task[task]).squeeze(-1)
                for task in self.tasks
            }

        if self.interval_from_frame_enabled:
            from .losses import pool_frame_logits

            flat_mask = interval_masks.reshape(batch * slots, -1)
            valid_flat = slot_valid.reshape(-1)
            for task in self.tasks:
                frame_logits = frame_logits_by_task[task]
                expanded = frame_logits.unsqueeze(1).expand(-1, slots, -1).reshape(batch * slots, -1)
                pooled_flat = torch.zeros(batch * slots, dtype=frame_logits.dtype, device=frame_logits.device)
                if valid_flat.any():
                    valid_logits = expanded[valid_flat]
                    valid_masks = flat_mask[valid_flat]
                    mode = self.interval_from_frame_cfg["mode"]
                    if mode == "lse_mean":
                        lse = pool_frame_logits(
                            valid_logits,
                            valid_masks,
                            mode="lse",
                            temperature=self.interval_from_frame_cfg["temperature"],
                        )
                        mean = (
                            valid_logits.masked_fill(~valid_masks, 0.0).sum(dim=-1)
                            / valid_masks.sum(dim=-1).clamp_min(1)
                        )
                        pooled = self.frame_mil_fusions[task](torch.stack([lse, mean], dim=-1)).squeeze(-1)
                    else:
                        pooled = pool_frame_logits(
                            valid_logits,
                            valid_masks,
                            mode=mode,
                            topk_ratio=self.interval_from_frame_cfg["topk_ratio"],
                            temperature=self.interval_from_frame_cfg["temperature"],
                        )
                    pooled_flat[valid_flat] = pooled
                interval_logits_by_task[task] = pooled_flat.view(batch, slots)
        else:
            for task in self.tasks:
                sequence = hidden_by_task[task]
                if self.pooling_mode == "mean":
                    counts = interval_masks.sum(dim=-1, keepdim=True).clamp_min(1.0).to(sequence.dtype)
                    norm_mask = interval_masks.to(sequence.dtype) / counts
                    pooled_tensor = torch.bmm(norm_mask, sequence)
                    pooled_tensor = pooled_tensor.masked_fill(~has_valid, 0.0)
                elif self.pooling_mode in ("attn", "mean_attn"):
                    scores = self.attention_heads[task](sequence).squeeze(-1).unsqueeze(1).expand(-1, slots, -1)
                    if pos_bias is not None:
                        scores = scores + pos_bias
                    scores = scores.masked_fill(~interval_masks, -1e4)
                    # Under AMP autocast, softmax returns float32; cast to sequence.dtype to ensure safe bmm
                    weights = torch.softmax(scores, dim=-1).to(sequence.dtype)
                    pooled_attn = torch.bmm(weights, sequence)
                    pooled_attn = pooled_attn.masked_fill(~has_valid, 0.0)

                    if self.pooling_mode == "mean_attn":
                        counts = interval_masks.sum(dim=-1, keepdim=True).clamp_min(1.0).to(sequence.dtype)
                        norm_mask = interval_masks.to(sequence.dtype) / counts
                        pooled_mean = torch.bmm(norm_mask, sequence)
                        pooled_mean = pooled_mean.masked_fill(~has_valid, 0.0)
                        pooled_tensor = torch.cat([pooled_mean, pooled_attn], dim=-1)
                    else:
                        pooled_tensor = pooled_attn
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
            result["frame_logits"] = frame_logits_by_task
        return result

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
        interval_bounds: torch.Tensor | None = None,
        interval_valid: torch.Tensor | None = None,
        layer_mask: torch.Tensor | None = None,
        return_frames: bool = False,
    ) -> dict[str, Any]:
        """Forward pass supporting multi-task block and segment inference.

        Returns:
            In block mode:
                Dict with "interval_logits" -> {task: Tensor[B, slots]}, "interval_valid" -> Tensor[B, slots]
            In segment mode:
                Dict with {task: Tensor[B]}
        """
        hidden_by_task, out_lengths, valid_mask = self._encode_sequence(x, lengths, layer_mask=layer_mask)
        frames = None
        if return_frames:
            if not self.frame_head_enabled:
                raise ValueError("Frame output requires training.block.frame_head.enabled=true")
            stride, receptive = 1, 1
            for kernel, step in zip(self.wavlm.config.conv_kernel, self.wavlm.config.conv_stride):
                receptive += (int(kernel) - 1) * stride
                stride *= int(step)
            frames = {
                "frame_logits": {task: self.frame_heads[task](hidden_by_task[task]).squeeze(-1)
                                 for task in self.tasks},
                "frame_hidden": {task: hidden_by_task[task] for task in self.tasks},
                "frame_valid": valid_mask,
                "frame_stride_samples": stride,
                "frame_receptive_samples": receptive,
            }
        if interval_bounds is not None:
            if interval_valid is None:
                interval_valid = torch.ones(interval_bounds.shape[:2], dtype=torch.bool, device=x.device)
            result = self._forward_block(hidden_by_task, valid_mask, lengths, interval_bounds, interval_valid)
            if frames is not None:
                result.update(frames)
            return result
        if frames is not None:
            return frames

        # Segment / Utterance mode
        if getattr(self, "interval_from_frame_enabled", False):
            from .losses import pool_frame_logits

            outputs: dict[str, torch.Tensor] = {}
            mode = self.interval_from_frame_cfg.get("mode", "linear_softmax")
            for task in self.tasks:
                frame_logits = self.frame_heads[task](hidden_by_task[task]).squeeze(-1)
                if mode == "lse_mean":
                    lse = pool_frame_logits(
                        frame_logits,
                        valid_mask,
                        mode="lse",
                        temperature=self.interval_from_frame_cfg.get("temperature", 1.0),
                    )
                    mean = (
                        frame_logits.masked_fill(~valid_mask, 0.0).sum(dim=-1)
                        / valid_mask.sum(dim=-1).clamp_min(1)
                    )
                    pooled = self.frame_mil_fusions[task](torch.stack([lse, mean], dim=-1)).squeeze(-1)
                else:
                    pooled = pool_frame_logits(
                        frame_logits,
                        valid_mask,
                        mode=mode,
                        topk_ratio=self.interval_from_frame_cfg.get("topk_ratio", 0.2),
                        temperature=self.interval_from_frame_cfg.get("temperature", 1.0),
                    )
                outputs[task] = pooled
            return outputs

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

    def _sinusoidal_pos_encoding(self, pos: torch.Tensor, d_model: int) -> torch.Tensor:
        if hasattr(self, "inv_freq") and self.inv_freq is not None and self.inv_freq.numel() == d_model // 2:
            freq = self.inv_freq.to(dtype=pos.dtype, device=pos.device)
        else:
            i = torch.arange(d_model // 2, device=pos.device, dtype=pos.dtype)
            freq = 1.0 / (10000.0 ** (2.0 * i / d_model))
        angles = pos.unsqueeze(-1) * freq
        return torch.cat([angles.sin(), angles.cos()], dim=-1)

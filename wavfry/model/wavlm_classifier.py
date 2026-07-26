import torch
import torch.nn as nn
from transformers import WavLMModel

class WavFryClassifier(nn.Module):
    def __init__(self, config, wavlm_instance=None):
        super().__init__()
        self.backbone_name = config['model']['backbone']
        
        if wavlm_instance is not None:
            self.wavlm = wavlm_instance
        else:
            # Set output_hidden_states=True in config before loading or use it in forward
            self.wavlm = WavLMModel.from_pretrained(self.backbone_name)
        
        # Disable internal masking/SpecAugment which causes issues with very short segments
        self.wavlm.config.apply_spec_augment = False
        self.wavlm.config.mask_time_prob = 0.0
        self.wavlm.config.mask_feature_prob = 0.0
        
        # Hidden dimension
        self.hidden_dim = self.wavlm.config.hidden_size
        self.num_layers = self.wavlm.config.num_hidden_layers
        
        # Scalar Mix
        self.use_scalar_mix = config['model'].get('use_scalar_mix', True)
        if self.use_scalar_mix:
            # num_layers + 1 because hidden_states includes the input embedding (Layer 0)
            self.scalar_mix_weights = nn.Parameter(torch.zeros(self.num_layers + 1))
            
        # Pooling parameters
        self.pooling_mode = config['training'].get('pooling_mode', 'mean')
        
        if 'attn' in self.pooling_mode:
            self.attention_head = nn.Linear(self.hidden_dim, 1)
            self.use_pos_encoding = config['training'].get('ablation', {}).get('use_positional_encoding', False)
            if self.use_pos_encoding:
                self.pos_dim = config['training'].get('ablation', {}).get('pos_embed_dim', 32)
                self.pos_scorer = nn.Linear(self.pos_dim, 1, bias=False)
                nn.init.zeros_(self.pos_scorer.weight)
        
        # Classification Head
        head_hidden = config['model'].get('head_hidden_dim', 256)
        dropout = config['model'].get('dropout', 0.1)
        
        in_features = self.hidden_dim * 2 if 'mean' in self.pooling_mode and ('topk' in self.pooling_mode or 'attn' in self.pooling_mode) else self.hidden_dim
        
        self.head = nn.Sequential(
            nn.Linear(in_features, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1)
        )
        
        # Freeze layers
        self.freeze_layers(config)
        
        # Gradient Checkpointing
        if config['training'].get('gradient_checkpointing', False):
            self.wavlm.gradient_checkpointing_enable()
        
    def freeze_layers(self, config):
        # 1. First, freeze the entire WavLM model (including encoder, feature_extractor, projections, etc.)
        for param in self.wavlm.parameters():
            param.requires_grad = False
            
        fine_tune_layers = config['model'].get('fine_tune_layers', '4-8')
        
        if fine_tune_layers == 'all':
            trainable_indices = list(range(self.num_layers))
        elif fine_tune_layers == 'none':
            trainable_indices = []
        else:
            if isinstance(fine_tune_layers, str):
                start, end = map(int, fine_tune_layers.split('-'))
                trainable_indices = list(range(start, end + 1))
            else:
                trainable_indices = fine_tune_layers
        
        # 2. Unfreeze only the selected transformer layers
        for i, layer in enumerate(self.wavlm.encoder.layers):
            if i in trainable_indices:
                for param in layer.parameters():
                    param.requires_grad = True
                
    def forward(self, x, lengths):
        # x: [B, T]
        outputs = self.wavlm(x, output_hidden_states=True)
        
        if self.use_scalar_mix:
            # hidden_states: tuple of (B, T, D) for each layer (including input embedding)
            h = torch.stack(outputs.hidden_states, dim=0) # [L+1, B, T, D]
            norm_weights = torch.softmax(self.scalar_mix_weights, dim=0)
            # Weighted sum over layers
            x = (h * norm_weights.view(-1, 1, 1, 1)).sum(dim=0) # [B, T, D]
        else:
            x = outputs.last_hidden_state # [B, T, D]
            
        # Accurate output lengths after CNN feature extractor
        out_lengths = self.wavlm._get_feat_extract_output_lengths(lengths)
        
        # 1. Mean Pooling
        # mask shape: [B, T]
        bool_mask = torch.arange(x.size(1), device=x.device).expand(x.size(0), x.size(1)) < out_lengths.unsqueeze(1)
        float_mask = bool_mask.unsqueeze(-1).float() # [B, T, 1]
        
        # Avoid division by zero
        x_mean = (x * float_mask).sum(dim=1) / (float_mask.sum(dim=1) + 1e-9) # [B, D]
        
        if self.pooling_mode == 'mean_attn':
            # 2. Attention-weighted Pooling
            # Compute scores: [B, T, 1] -> [B, T]
            attn_scores = self.attention_head(x).squeeze(-1)
            
            if hasattr(self, 'use_pos_encoding') and self.use_pos_encoding:
                max_T = x.size(1)
                # Create relative positions 0.0 to 1.0 (shape: [B, T])
                positions = torch.arange(max_T, device=x.device, dtype=x.dtype).unsqueeze(0).expand(x.size(0), -1)
                # Avoid division by zero when out_lengths is 1: out_lengths - 1 clamped to min=1
                rel_pos = (positions / (out_lengths.unsqueeze(1).float() - 1).clamp(min=1)).clamp(0.0, 1.0)
                rel_pos = rel_pos.masked_fill(~bool_mask, 0.0)  # padding 帧归零
                
                pos_emb = self._sinusoidal_pos_encoding(rel_pos, self.pos_dim)  # [B, T, D_pos]
                pos_bias = self.pos_scorer(pos_emb).squeeze(-1)                  # [B, T]
                attn_scores = attn_scores + pos_bias  # 叠加，不破坏声学特征
                
            # Mask padding frames with -1e9 for softmax
            # Use a value compatible with FP16/Half
            attn_scores = attn_scores.masked_fill(~bool_mask, -1e4)
            # Compute weights: [B, T]
            attn_weights = torch.softmax(attn_scores, dim=-1)
            # Weighted sum: [B, T, D] * [B, T, 1] -> [B, D]
            x_attn = (x * attn_weights.unsqueeze(-1)).sum(dim=1)
            
            # 3. Fusion
            x_final = torch.cat([x_mean, x_attn], dim=-1) # [B, 2*D]
        elif self.pooling_mode == 'mean_topk':
            # 2. Top-K Mean Pooling (Legacy / Alternative)
            frame_scores = torch.norm(x, p=2, dim=-1) 
            frame_scores = frame_scores.masked_fill(~bool_mask, -1e9)
            _, indices = frame_scores.sort(dim=1, descending=True)
            indices_exp = indices.unsqueeze(-1).expand(-1, -1, x.size(-1))
            sorted_x = x.gather(dim=1, index=indices_exp)
            
            # We still need topk params if we want to support this mode
            topk_ratio = getattr(self, 'topk_ratio', 0.2)
            topk_min_frames = getattr(self, 'topk_min_frames', 3)
            k_tensor = torch.clamp((out_lengths.float() * topk_ratio).long(), min=topk_min_frames)
            k_tensor = torch.min(k_tensor, out_lengths)
            
            topk_mask = torch.arange(sorted_x.size(1), device=x.device).expand(x.size(0), x.size(1)) < k_tensor.unsqueeze(1)
            topk_mask_float = topk_mask.unsqueeze(-1).float()
            x_topk = (sorted_x * topk_mask_float).sum(dim=1) / (topk_mask_float.sum(dim=1) + 1e-9)
            
            x_final = torch.cat([x_mean, x_topk], dim=-1)
        else:
            x_final = x_mean
            
        logits = self.head(x_final).squeeze(-1) # [B]
        return logits

    @staticmethod
    def _sinusoidal_pos_encoding(pos: torch.Tensor, d_model: int) -> torch.Tensor:
        """pos: [B, T] in [0,1]; returns [B, T, d_model]. No learnable parameters."""
        i = torch.arange(d_model // 2, device=pos.device, dtype=pos.dtype)
        freq = 1.0 / (10000.0 ** (2.0 * i / d_model))
        angles = pos.unsqueeze(-1) * freq          # [B, T, d_model//2]
        return torch.cat([angles.sin(), angles.cos()], dim=-1)  # [B, T, d_model]

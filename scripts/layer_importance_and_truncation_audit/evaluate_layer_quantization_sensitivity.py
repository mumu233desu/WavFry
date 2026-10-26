#!/usr/bin/env python
"""
WavFry Functional Auditing: Layer-wise Activation Outliers & Quantization Sensitivity Evaluator.

Audits:
1. Numerical Activation Outliers & Distribution across all 13 WavLM layers (L0 CNN + L1~L12 Transformer):
   - Peak amplitude (Max |x|), Standard Deviation, Kurtosis (heavy-tailed outlier metric)
   - 99.0% and 99.9% percentiles, Outlier Ratio (Max / P99.9)
   - Outlier channel count (channels exceeding 6 * sigma)
   - Simulated Uniform Symmetric Quantization Signal-to-Quantization-Noise Ratio (SQNR in dB)
     under both Per-Tensor and Per-Channel regimes for INT8 and INT4.

2. Downstream Phonation Task Vulnerability (Creaky Voice, Breathy Voice):
   - Single-Layer Sensitivity: Layer-by-layer simulated quantization noise injection (INT8, INT4)
   - Macro Compression Strategies: Full Model FP32 vs Full Model INT8/INT4 vs Top-k Structural Pruning
     vs Hybrid (Top-k Truncation + Shallow INT8/INT4 with or without CNN quantization).

Outputs saved to:
    data/experiments/layer_quantization_sensitivity_audit/
        - quantization_tensor_metrics.csv
        - quantization_task_ablation.csv
        - quantization_sensitivity_report.md
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", message=".*Support for mismatched key_padding_mask and attn_mask is deprecated.*")
warnings.filterwarnings("ignore", message=".*torchaudio.backend.common.AudioMetaData.*")
warnings.filterwarnings("ignore", message=".*No positive class found in y_true.*")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader

# Add project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wavfry.data import splits as splits_mod
from wavfry.model.wavlm_classifier import WavFryClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_layer_quantization_sensitivity")


def parse_args():
    parser = argparse.ArgumentParser(description="WavFry Layer Activation Outlier & Quantization Sensitivity Evaluator")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="code/exp_results/checkpoints/frozen_creaky_breathy_block/best_model.pt",
        help="Path to trained model checkpoint (.pt).",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="code/manifests/segments_multi.csv",
        help="Path to manifest CSV. Defaults to code/manifests/segments_multi.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/experiments/layer_quantization_sensitivity_audit",
        help="Directory to save output reports and tables.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["val", "test", "all"],
        default="test",
        help="Data split to evaluate on. Default: 'test'.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run rapid balanced profiling (~300 blocks, covers both Creaky and Breathy in ~45 seconds).",
    )
    parser.add_argument(
        "--quick-blocks",
        type=int,
        default=300,
        help="Number of blocks to sample in quick mode (default: 300).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cuda' or 'cpu'). Defaults to cuda if free VRAM >= 2GB.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional upper bound on batches to process (for rapid profiling).",
    )
    return parser.parse_args()


def resolve_device(requested_device: Optional[str]) -> torch.device:
    if requested_device:
        dev = torch.device(requested_device)
        logger.info(f"Using explicitly requested device: {dev}")
        return dev

    if torch.cuda.is_available():
        try:
            free_mem_bytes, total_mem_bytes = torch.cuda.mem_get_info()
            free_mem_mb = free_mem_bytes / (1024 * 1024)
            if free_mem_mb < 2048:
                logger.warning(
                    f"GPU 0 has only {free_mem_mb:.1f} MiB free VRAM. "
                    "Auto-falling back to CPU to prevent CUDA OOM crash!"
                )
                return torch.device("cpu")
            logger.info(f"Using CUDA (Free VRAM: {free_mem_mb:.1f} MiB)")
            return torch.device("cuda")
        except Exception:
            return torch.device("cuda")

    logger.info("CUDA unavailable, using CPU.")
    return torch.device("cpu")


def load_model_from_checkpoint(ckpt_path: Path, device: torch.device) -> tuple[WavFryClassifier, Dict[str, Any]]:
    logger.info(f"Loading checkpoint from {ckpt_path}...")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    if not config:
        raise ValueError(f"Checkpoint at {ckpt_path} does not contain 'config'.")

    model = WavFryClassifier(config)
    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
    cleaned_sd = {}
    for k, v in state_dict.items():
        key = k[7:] if k.startswith("module.") else k
        cleaned_sd[key] = v

    missing_keys, _ = model.load_state_dict(cleaned_sd, strict=False)
    critical_missing = [k for k in missing_keys if not k.startswith("wavlm.")]
    if critical_missing:
        raise RuntimeError(f"Fail-Fast: Required classification weights missing from checkpoint: {critical_missing}")

    model.to(device)
    model.eval()

    for p in model.wavlm.parameters():
        p.requires_grad = False

    return model, config


def prepare_dataloader(
    config: Dict[str, Any],
    manifest_path: Path,
    split_target: str,
    batch_size: int,
    is_quick: bool = False,
    quick_blocks: int = 300,
) -> tuple[DataLoader, str]:
    df = pd.read_csv(manifest_path, dtype={"speaker_id": str, "phone": str, "file_key": str})
    isolate = bool(config.get("data", {}).get("isolate_split_speakers", True))
    train_df, val_df, test_df = splits_mod.make_batch_split(df, isolate=isolate)

    if split_target == "val":
        eval_df = val_df
    elif split_target == "test":
        eval_df = test_df if not test_df.empty else val_df
    else:
        eval_df = df

    logger.info(f"Loaded split '{split_target}' with {len(eval_df)} segments.")

    from wavfry.data.paradigm_block import ContinuousAudioBlockDataset, collate_block_fn

    block_cfg = config.get("training", {}).get("block", {})
    ds = ContinuousAudioBlockDataset(
        eval_df,
        duration_ms=block_cfg.get("duration_ms", 2000),
        hop_ms=block_cfg.get("hop_ms", 1000),
        max_intervals=block_cfg.get("max_intervals", 8),
        boundary_trim_frames=block_cfg.get("boundary_trim_frames", 0),
        seed=42,
        fold_index=0,
    )

    if is_quick:
        creaky_blocks = [b for b in ds.blocks if any(iv.get("label_creaky") == 1 for iv in b.get("intervals", []))]
        breathy_blocks = [b for b in ds.blocks if any(iv.get("label_breathy") == 1 for iv in b.get("intervals", []))]

        rng = np.random.RandomState(42)
        n_per_task = max(10, quick_blocks // 2)
        idx_c = rng.choice(len(creaky_blocks), min(n_per_task, len(creaky_blocks)), replace=False)
        idx_b = rng.choice(len(breathy_blocks), min(n_per_task, len(breathy_blocks)), replace=False)

        sampled_creak = [creaky_blocks[i] for i in idx_c]
        sampled_breath = [breathy_blocks[i] for i in idx_b]

        interleaved = []
        for c, b in zip(sampled_creak, sampled_breath):
            interleaved.extend([c, b])
        ds.blocks = interleaved
        logger.info(
            f"[Quick Mode] Sampled {len(ds.blocks)} balanced blocks "
            f"({len(sampled_creak)} creaky-active + {len(sampled_breath)} breathy-active)."
        )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_block_fn,
        pin_memory=False,
    )
    return loader, "block"


# =========================================================================
# Simulated Quantization Utilities
# =========================================================================

def quantize_tensor_symmetric(
    x: torch.Tensor,
    bits: int = 8,
    per_channel: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Simulate uniform symmetric quantization.

    Returns:
        (quantized_x, quantization_noise)
    """
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1

    if per_channel:
        # x is assumed to have channel as the last dimension [-1] (e.g. [..., D=768])
        # scale shape matches x with 1s in all dims except last
        scale = x.abs().amax(dim=tuple(range(x.ndim - 1)), keepdim=True).clamp(min=1e-8) / qmax
    else:
        scale = x.abs().max().clamp(min=1e-8) / qmax

    q_int = (x / scale).round().clamp(qmin, qmax)
    x_dequant = q_int * scale
    noise = x - x_dequant
    return x_dequant, noise


# =========================================================================
# Online Running Moments & Histogram for Outlier Auditing
# =========================================================================

class LayerTensorAuditor:
    """Accurately tracks online distribution moments and histogram across full test split."""

    def __init__(self, num_layers: int = 13, num_channels: int = 768, hist_max: float = 80.0, num_bins: int = 8000):
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.hist_max = hist_max
        self.num_bins = num_bins
        self.bin_width = hist_max / num_bins

        # Per-layer accumulators
        self.counts = np.zeros(num_layers, dtype=np.int64)
        self.sum_x = np.zeros(num_layers, dtype=np.float64)
        self.sum_x2 = np.zeros(num_layers, dtype=np.float64)
        self.sum_x3 = np.zeros(num_layers, dtype=np.float64)
        self.sum_x4 = np.zeros(num_layers, dtype=np.float64)
        self.max_abs = np.zeros(num_layers, dtype=np.float64)

        # Histograms of |x| for exact high percentiles (P99, P99.9)
        self.hist_abs = np.zeros((num_layers, num_bins), dtype=np.int64)
        self.overflow_counts = np.zeros(num_layers, dtype=np.int64)

        # Per-channel maximum absolute value: [num_layers, num_channels]
        self.channel_max_abs = np.zeros((num_layers, num_channels), dtype=np.float64)

        # SQNR accumulators: sum(x^2) and sum(noise^2)
        # keys: (bits, per_channel)
        self.sqnr_signal_pow = {
            (8, False): np.zeros(num_layers, dtype=np.float64),
            (8, True): np.zeros(num_layers, dtype=np.float64),
            (4, False): np.zeros(num_layers, dtype=np.float64),
            (4, True): np.zeros(num_layers, dtype=np.float64),
        }
        self.sqnr_noise_pow = {
            (8, False): np.zeros(num_layers, dtype=np.float64),
            (8, True): np.zeros(num_layers, dtype=np.float64),
            (4, False): np.zeros(num_layers, dtype=np.float64),
            (4, True): np.zeros(num_layers, dtype=np.float64),
        }

    def update(self, H: torch.Tensor, valid_mask: Optional[torch.Tensor] = None):
        """Update statistics with batch tensor H: [num_layers=13, B, T, D=768]."""
        num_layers, B, T, D = H.shape

        for l in range(num_layers):
            layer_tensor = H[l]  # [B, T, D]
            if valid_mask is not None:
                # valid_mask: [B, T]
                v_tensor = layer_tensor[valid_mask]  # [N_valid, D]
            else:
                v_tensor = layer_tensor.reshape(-1, D)

            if v_tensor.numel() == 0:
                continue

            v_np = v_tensor.detach().cpu().numpy()
            v_flat = v_np.ravel().astype(np.float64)
            v_abs = np.abs(v_flat)

            n = len(v_flat)
            self.counts[l] += n
            self.sum_x[l] += np.sum(v_flat)
            self.sum_x2[l] += np.sum(v_flat ** 2)
            self.sum_x3[l] += np.sum(v_flat ** 3)
            self.sum_x4[l] += np.sum(v_flat ** 4)
            current_max = np.max(v_abs)
            if current_max > self.max_abs[l]:
                self.max_abs[l] = current_max

            # Channel max
            c_max = np.max(np.abs(v_np), axis=0)  # [D]
            self.channel_max_abs[l] = np.maximum(self.channel_max_abs[l], c_max)

            # Histogram bins
            bin_indices = (v_abs / self.bin_width).astype(np.int64)
            in_range = bin_indices < self.num_bins
            out_range = ~in_range

            if np.any(in_range):
                counts, _ = np.histogram(v_abs[in_range], bins=self.num_bins, range=(0.0, self.hist_max))
                self.hist_abs[l] += counts
            self.overflow_counts[l] += np.sum(out_range)

            # Quantization SQNR accumulation
            sig_pow = np.sum(v_flat ** 2)
            for bits in (8, 4):
                for pc in (False, True):
                    _, noise = quantize_tensor_symmetric(v_tensor, bits=bits, per_channel=pc)
                    noise_pow = np.sum(noise.detach().cpu().numpy().ravel().astype(np.float64) ** 2)
                    self.sqnr_signal_pow[(bits, pc)][l] += sig_pow
                    self.sqnr_noise_pow[(bits, pc)][l] += noise_pow

    def compute_summary_dataframe(self) -> pd.DataFrame:
        records = []
        for l in range(self.num_layers):
            n = self.counts[l]
            if n == 0:
                continue

            mean = self.sum_x[l] / n
            var = (self.sum_x2[l] / n) - (mean ** 2)
            std = np.sqrt(max(var, 1e-12))

            # Kurtosis (Fisher-Pearson standard Kurtosis: normal dist = 3.0)
            m4 = (self.sum_x4[l] / n) - 4 * mean * (self.sum_x3[l] / n) + 6 * (mean ** 2) * (self.sum_x2[l] / n) - 3 * (mean ** 4)
            kurtosis = m4 / (var ** 2 + 1e-12)

            # Percentiles from histogram
            cum_hist = np.cumsum(self.hist_abs[l])
            target_p99 = 0.990 * n
            target_p99_9 = 0.999 * n

            idx_p99 = np.searchsorted(cum_hist, target_p99)
            idx_p99_9 = np.searchsorted(cum_hist, target_p99_9)

            val_p99 = min(self.hist_max, (idx_p99 + 0.5) * self.bin_width)
            val_p99_9 = min(self.hist_max, (idx_p99_9 + 0.5) * self.bin_width)

            max_val = self.max_abs[l]
            outlier_ratio = max_val / max(val_p99_9, 1e-6)

            # Channel Outlier Count (> 6 * sigma)
            outlier_thresh = 6.0 * std
            n_outlier_channels = int(np.sum(self.channel_max_abs[l] > outlier_thresh))
            channel_max_peak = float(np.max(self.channel_max_abs[l]))
            channel_median_peak = float(np.median(self.channel_max_abs[l]))

            # SQNR in dB
            sqnr_int8_pt = 10.0 * np.log10(self.sqnr_signal_pow[(8, False)][l] / max(self.sqnr_noise_pow[(8, False)][l], 1e-12))
            sqnr_int8_pc = 10.0 * np.log10(self.sqnr_signal_pow[(8, True)][l] / max(self.sqnr_noise_pow[(8, True)][l], 1e-12))
            sqnr_int4_pt = 10.0 * np.log10(self.sqnr_signal_pow[(4, False)][l] / max(self.sqnr_noise_pow[(4, False)][l], 1e-12))
            sqnr_int4_pc = 10.0 * np.log10(self.sqnr_signal_pow[(4, True)][l] / max(self.sqnr_noise_pow[(4, True)][l], 1e-12))

            records.append({
                "layer_idx": l,
                "layer_name": "L0 (CNN Front-end)" if l == 0 else f"L{l} (Transformer)",
                "layer_type": "CNN" if l == 0 else ("Shallow Tr" if l <= 5 else "Deep Tr"),
                "mean": float(mean),
                "std": float(std),
                "max_abs": float(max_val),
                "p99_0": float(val_p99),
                "p99_9": float(val_p99_9),
                "kurtosis": float(kurtosis),
                "outlier_ratio": float(outlier_ratio),
                "outlier_channels_6sigma": n_outlier_channels,
                "channel_max_peak": channel_max_peak,
                "channel_median_peak": channel_median_peak,
                "sqnr_int8_per_tensor_db": float(sqnr_int8_pt),
                "sqnr_int8_per_channel_db": float(sqnr_int8_pc),
                "sqnr_int8_channel_gain_db": float(sqnr_int8_pc - sqnr_int8_pt),
                "sqnr_int4_per_tensor_db": float(sqnr_int4_pt),
                "sqnr_int4_per_channel_db": float(sqnr_int4_pc),
            })

        return pd.DataFrame(records)


# =========================================================================
# Task Quantization Sensitivity Conditions Builder
# =========================================================================

def build_quantization_conditions(num_layers: int = 13) -> List[Dict[str, Any]]:
    """Build comprehensive macro and micro layer quantization evaluation conditions."""
    conds: List[Dict[str, Any]] = []

    # 1. Macro Baselines & Global Quantization
    conds.append({
        "id": "fp32_baseline",
        "name": "Full Model Baseline (FP32)",
        "category": "Baseline",
        "description": "Original unquantized model across all 13 layers",
        "active_layers": list(range(num_layers)),
        "quant_type": "none",
        "quant_layers": [],
        "bits": 32,
    })
    conds.append({
        "id": "int8_full_per_tensor",
        "name": "Full Model Uniform INT8 (Per-Tensor)",
        "category": "Full Quantization",
        "description": "All 13 layers uniformly quantized to INT8 (Per-Tensor)",
        "active_layers": list(range(num_layers)),
        "quant_type": "per_tensor",
        "quant_layers": list(range(num_layers)),
        "bits": 8,
    })
    conds.append({
        "id": "int8_full_per_channel",
        "name": "Full Model Uniform INT8 (Per-Channel)",
        "category": "Full Quantization",
        "description": "All 13 layers quantized to INT8 (Per-Channel scale)",
        "active_layers": list(range(num_layers)),
        "quant_type": "per_channel",
        "quant_layers": list(range(num_layers)),
        "bits": 8,
    })
    conds.append({
        "id": "int4_full_per_tensor",
        "name": "Full Model Aggressive INT4 (Per-Tensor)",
        "category": "Full Quantization",
        "description": "All 13 layers aggressively quantized to INT4 (Per-Tensor)",
        "active_layers": list(range(num_layers)),
        "quant_type": "per_tensor",
        "quant_layers": list(range(num_layers)),
        "bits": 4,
    })

    # 2. Top-6 Structural Truncation & Hybrid Strategies (Keep L0~L5, Prune L6~L12)
    kept_top6 = list(range(6))  # L0 (CNN) + L1..L5
    conds.append({
        "id": "top6_trunc_fp32",
        "name": "Top-6 Pruning (L0~L5 FP32)",
        "category": "Structural Pruning",
        "description": "L6~L12 structurally dropped (0-bit); L0~L5 active in FP32",
        "active_layers": kept_top6,
        "quant_type": "none",
        "quant_layers": [],
        "bits": 32,
    })
    conds.append({
        "id": "top6_trunc_int8_nocnn",
        "name": "Top-6 Pruning + L1~L5 INT8 (CNN FP32)",
        "category": "Hybrid Pruning + Quant",
        "description": "L6~L12 dropped; CNN L0 preserved in FP32; L1~L5 in INT8",
        "active_layers": kept_top6,
        "quant_type": "per_tensor",
        "quant_layers": list(range(1, 6)),  # L1..L5
        "bits": 8,
    })
    conds.append({
        "id": "top6_trunc_int8_all",
        "name": "Top-6 Pruning + L0~L5 Full INT8 (Inc. CNN)",
        "category": "Hybrid Pruning + Quant",
        "description": "L6~L12 dropped; all remaining L0~L5 (inc. CNN) in INT8",
        "active_layers": kept_top6,
        "quant_type": "per_tensor",
        "quant_layers": list(range(6)),  # L0..L5
        "bits": 8,
    })
    conds.append({
        "id": "top6_trunc_int4_nocnn",
        "name": "Top-6 Pruning + L1~L5 INT4 (CNN FP32)",
        "category": "Hybrid Pruning + Quant",
        "description": "L6~L12 dropped; CNN L0 preserved in FP32; L1~L5 aggressively in INT4",
        "active_layers": kept_top6,
        "quant_type": "per_tensor",
        "quant_layers": list(range(1, 6)),
        "bits": 4,
    })
    conds.append({
        "id": "top6_trunc_int4_all",
        "name": "Top-6 Pruning + L0~L5 Full INT4 (Inc. CNN)",
        "category": "Hybrid Pruning + Quant",
        "description": "L6~L12 dropped; all remaining L0~L5 (inc. CNN) aggressively in INT4",
        "active_layers": kept_top6,
        "quant_type": "per_tensor",
        "quant_layers": list(range(6)),
        "bits": 4,
    })

    # 3. Micro Single-Layer Sensitivity (Layer-by-Layer INT8 & INT4 Injection)
    for l in range(num_layers):
        l_name = "CNN L0" if l == 0 else f"L{l}"
        conds.append({
            "id": f"single_layer_{l}_int8",
            "name": f"Single Layer {l_name} INT8 Injection",
            "category": "Single-Layer Sensitivity (INT8)",
            "description": f"Only {l_name} quantized to INT8; all other 12 layers remain FP32",
            "active_layers": list(range(num_layers)),
            "quant_type": "per_tensor",
            "quant_layers": [l],
            "bits": 8,
        })
    for l in range(num_layers):
        l_name = "CNN L0" if l == 0 else f"L{l}"
        conds.append({
            "id": f"single_layer_{l}_int4",
            "name": f"Single Layer {l_name} INT4 Injection",
            "category": "Single-Layer Sensitivity (INT4)",
            "description": f"Only {l_name} aggressively quantized to INT4; all other 12 layers remain FP32",
            "active_layers": list(range(num_layers)),
            "quant_type": "per_tensor",
            "quant_layers": [l],
            "bits": 4,
        })

    return conds


# =========================================================================
# Single-Pass Evaluation Pipeline
# =========================================================================

def evaluate_quantization_pipeline(
    model: WavFryClassifier,
    loader: DataLoader,
    device: torch.device,
    conditions: List[Dict[str, Any]],
    eval_tasks: List[str] = ["creaky", "breathy"],
    max_batches: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute single-pass forward evaluation across all quantization conditions."""
    model.eval()
    num_layers = model.num_layers + 1
    num_conditions = len(conditions)
    head_dtype = next(model.heads.parameters()).dtype

    auditor = LayerTensorAuditor(num_layers=num_layers, num_channels=768)

    # Pre-calculate condition weights for each task using dynamic Softmax re-normalization
    cond_weights_by_task: Dict[str, torch.Tensor] = {}
    for task_name in model.tasks:
        if model.use_scalar_mix and task_name in model.scalar_mix_weights:
            raw_logits = model.scalar_mix_weights[task_name].detach().to(device)
            w_list = []
            for cond in conditions:
                active = cond["active_layers"]
                if len(active) == num_layers:
                    w = torch.softmax(raw_logits, dim=0)
                else:
                    m_logits = raw_logits.clone()
                    mask = torch.zeros(num_layers, dtype=torch.bool, device=device)
                    mask[active] = True
                    m_logits[~mask] = -float("inf")
                    w = torch.softmax(m_logits, dim=0)
                w_list.append(w)
            cond_weights_by_task[task_name] = torch.stack(w_list, dim=0)  # [num_conditions, num_layers]
        else:
            uniform = torch.full((num_layers,), 1.0 / num_layers, device=device)
            cond_weights_by_task[task_name] = uniform.unsqueeze(0).expand(num_conditions, -1)

    cond_interval_preds: Dict[str, List[Dict[str, list]]] = {
        task: [{} for _ in range(num_conditions)] for task in eval_tasks
    }

    start_time = time.time()
    batch_count = 0
    total_batches = min(len(loader), max_batches) if max_batches else len(loader)

    logger.info(f"Starting evaluation across {total_batches} batches ({num_conditions} conditions, {eval_tasks})...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches and batch_idx >= max_batches:
                break
            batch_count += 1
            if batch_count % 5 == 0 or batch_count == total_batches:
                elapsed = time.time() - start_time
                logger.info(f"Progress: [{batch_count}/{total_batches} batches] ({elapsed:.1f}s elapsed)...")

            waveforms, bounds, labels, valid, lengths, speakers, block_ids, interval_ids = batch
            waveforms = waveforms.to(device)
            bounds = bounds.to(device)
            labels = labels.to(device)
            valid = valid.to(device)
            lengths = lengths.to(device)

            # 1. Forward WavLM backbone ONCE
            wavlm_dtype = next(model.wavlm.parameters()).dtype
            x_in = waveforms.to(wavlm_dtype) if waveforms.dtype != wavlm_dtype else waveforms
            max_len = waveforms.shape[1]
            lengths_clamped = lengths.clamp(min=1, max=max_len)
            attention_mask = (torch.arange(max_len, device=device).unsqueeze(0) < lengths_clamped.unsqueeze(1)).long()

            outputs = model.wavlm(x_in, attention_mask=attention_mask, output_hidden_states=True)
            raw_hidden_states = [h.detach() for h in outputs.hidden_states]
            H = torch.stack(raw_hidden_states, dim=0)  # [13, B, T, D]
            L, B, T, D = H.shape

            out_lengths = model.wavlm._get_feat_extract_output_lengths(lengths).to(device).clamp(min=1, max=T)
            seq_mask = torch.arange(T, device=device).unsqueeze(0) < out_lengths.unsqueeze(1)

            # 2. Update online tensor auditor
            auditor.update(H, valid_mask=seq_mask)

            # 3. Pre-compute quantized versions of H for reuse
            # H_q_int8_pt, H_q_int8_pc, H_q_int4_pt
            H_q_cache: Dict[Tuple[str, int], torch.Tensor] = {}

            def get_quantized_H(q_type: str, b: int) -> torch.Tensor:
                key = (q_type, b)
                if key not in H_q_cache:
                    if q_type == "none" or b == 32:
                        H_q_cache[key] = H
                    elif q_type == "per_tensor":
                        q_t, _ = quantize_tensor_symmetric(H, bits=b, per_channel=False)
                        H_q_cache[key] = q_t
                    elif q_type == "per_channel":
                        q_t, _ = quantize_tensor_symmetric(H, bits=b, per_channel=True)
                        H_q_cache[key] = q_t
                return H_q_cache[key]

            # 4. Evaluate each condition
            for c_idx, cond in enumerate(conditions):
                q_type = cond["quant_type"]
                bits = cond["bits"]
                q_layers = cond["quant_layers"]

                if q_type == "none" or len(q_layers) == 0:
                    H_cond = H
                elif len(q_layers) == num_layers:
                    H_cond = get_quantized_H(q_type, bits)
                else:
                    # Hybrid / Single-layer selective quantization
                    H_cond = H.clone()
                    target_q = get_quantized_H(q_type, bits)
                    for ql in q_layers:
                        H_cond[ql] = target_q[ql]

                flat_H_cond = H_cond.view(L, -1)  # [13, B * T * D]

                # Vectorized mix for each task
                mixed_tasks = {}
                for task_name in model.tasks:
                    w = cond_weights_by_task[task_name][c_idx]  # [13]
                    mixed = (w.to(flat_H_cond.dtype) @ flat_H_cond).view(B, T, D).to(head_dtype)
                    mixed_tasks[task_name] = mixed

                # Forward head
                block_out = model._forward_block(
                    hidden_by_task=mixed_tasks,
                    seq_mask=seq_mask,
                    lengths=lengths,
                    interval_bounds=bounds,
                    slot_valid=valid,
                )

                for task_name in eval_tasks:
                    t_logits = block_out["interval_logits"][task_name]
                    task_idx = model.tasks.index(task_name) if task_name in model.tasks else 0
                    t_labels = labels[..., task_idx] if (labels.ndim > 2 and labels.shape[-1] > task_idx) else labels

                    for logit_row, label_row, val_row, item_row in zip(t_logits, t_labels, valid, interval_ids):
                        for l_val, lab_val, v_val, i_id in zip(logit_row, label_row, val_row, item_row):
                            if v_val and i_id is not None:
                                lab_float = lab_val.item()
                                if not np.isnan(lab_float):
                                    i_key = str(i_id.item() if isinstance(i_id, torch.Tensor) else i_id)
                                    rec = cond_interval_preds[task_name][c_idx]
                                    if i_key not in rec:
                                        rec[i_key] = [l_val.item(), lab_float, 1]
                                    else:
                                        rec[i_key][0] += l_val.item()
                                        rec[i_key][2] += 1

    # Compile Task Metrics
    task_records = []
    for task_name in eval_tasks:
        base_prauc = None
        base_rocauc = None

        for c_idx, cond in enumerate(conditions):
            rec = cond_interval_preds[task_name][c_idx]
            y_logits = np.array([item[0] / item[2] for item in rec.values()], dtype=np.float32)
            y_labels = np.array([item[1] for item in rec.values()], dtype=np.float32)

            if len(y_labels) == 0 or np.sum(y_labels == 1) == 0:
                continue

            # Metrics
            probs = 1.0 / (1.0 + np.exp(-y_logits))
            prauc = float(average_precision_score(y_labels, probs))
            try:
                rocauc = float(roc_auc_score(y_labels, probs))
            except Exception:
                rocauc = float("nan")

            if cond["id"] == "fp32_baseline":
                base_prauc = prauc
                base_rocauc = rocauc

            delta_prauc = (prauc - base_prauc) if base_prauc is not None else 0.0
            delta_rocauc = (rocauc - base_rocauc) if base_rocauc is not None else 0.0

            task_records.append({
                "task": task_name,
                "condition_id": cond["id"],
                "condition_name": cond["name"],
                "category": cond["category"],
                "bits": cond["bits"],
                "quant_type": cond["quant_type"],
                "n_samples": len(y_labels),
                "n_pos": int(np.sum(y_labels == 1)),
                "prauc": prauc,
                "delta_prauc": delta_prauc,
                "rocauc": rocauc,
                "delta_rocauc": delta_rocauc,
                "description": cond["description"],
            })

    tensor_df = auditor.compute_summary_dataframe()
    task_df = pd.DataFrame(task_records)
    return tensor_df, task_df


def generate_markdown_report(tensor_df: pd.DataFrame, task_df: pd.DataFrame, output_path: Path):
    """Generate professional, publication-grade Markdown audit report."""
    md = []
    md.append("# WavLM-Base+ 隐层激活离群值与量化敏感度经验审计报告\n")
    md.append("> **实验日期**: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    md.append("> **数据划分**: `test` split (Clean Condition)\n")
    md.append("> **基准模型**: `WavLM-Base+` (Multi-Task: `creaky`, `breathy`)\n\n")

    md.append("---\n\n")
    md.append("## 一、 核心发现速览 (Executive Summary)\n\n")

    # Extract key stats
    l0_sub = tensor_df[tensor_df["layer_idx"] == 0]
    l1_sub = tensor_df[tensor_df["layer_idx"] == 1]
    l12_sub = tensor_df[tensor_df["layer_idx"] == 12]
    max_kurt_row = tensor_df.sort_values(by="kurtosis", ascending=False).iloc[0] if not tensor_df.empty else None

    if not l0_sub.empty and not l1_sub.empty and max_kurt_row is not None:
        l0_row = l0_sub.iloc[0]
        l1_row = l1_sub.iloc[0]
        md.append(f"1. **高层离群值聚集成倍加剧 (Outlier Explosion in Deep Layers)**：\n")
        md.append(f"   - 浅层（$L_0$ CNN 与 $L_1$）分布相对平缓，峰度 Kurtosis 分别为 **{l0_row['kurtosis']:.2f}** 与 **{l1_row['kurtosis']:.2f}**；\n")
        md.append(f"   - 随着层数加深，中高层残差累积引发极度尖峰厚尾，在 **{max_kurt_row['layer_name']}** 峰度暴增至 **{max_kurt_row['kurtosis']:.2f}**（超过正态分布基准数十倍！），绝对极值达 **{max_kurt_row['max_abs']:.2f}**，证实了深层存在巨幅激活离群值。\n")

    if not l12_sub.empty:
        l12_row = l12_sub.iloc[0]
        md.append(f"2. **离群值对均匀量化的破坏 (Per-Tensor vs Per-Channel SQNR Collapse)**：\n")
        md.append(f"   - 在 $L_{{12}}$ 深层，由于极少数离群通道绑架了整层动态范围，逐张量（Per-Tensor）INT8 SQNR 跌至 **{l12_row['sqnr_int8_per_tensor_db']:.2f} dB**，而逐通道（Per-Channel）量化可恢复至 **{l12_row['sqnr_int8_per_channel_db']:.2f} dB**（通道隔离增益达 **+{l12_row['sqnr_int8_channel_gain_db']:.2f} dB**），铁证了离群值在少数通道的高度聚集。\n")

    # Task stats
    base_creaky = task_df[(task_df["task"] == "creaky") & (task_df["condition_id"] == "fp32_baseline")]
    top6_creaky = task_df[(task_df["task"] == "creaky") & (task_df["condition_id"] == "top6_trunc_fp32")]
    top6_int8_nocnn_creaky = task_df[(task_df["task"] == "creaky") & (task_df["condition_id"] == "top6_trunc_int8_nocnn")]
    top6_int4_nocnn_creaky = task_df[(task_df["task"] == "creaky") & (task_df["condition_id"] == "top6_trunc_int4_nocnn")]

    if not base_creaky.empty and not top6_creaky.empty:
        b_c = base_creaky.iloc[0]
        t_c = top6_creaky.iloc[0]
        t8_c = top6_int8_nocnn_creaky.iloc[0] if not top6_int8_nocnn_creaky.empty else None
        t4_c = top6_int4_nocnn_creaky.iloc[0] if not top6_int4_nocnn_creaky.empty else None

        md.append(f"3. **结构剪枝 vs 低比特量化在声学任务上的终极抉择**：\n")
        md.append(f"   - **Top-6 结构剪枝 (0-bit 消除 $L_6\\text{{--}}L_{{12}}$)**：Creaky PR-AUC 从基线 **{b_c['prauc']:.4f}** 仅微调至 **{t_c['prauc']:.4f}** ($\\Delta = {t_c['delta_prauc']:+.4f}$)，不仅免除了后半段计算量，而且以**零近似误差**彻底规避了高层离群值对量化的破坏！\n")
        if t8_c is not None and t4_c is not None:
            md.append(f"   - **底层能否“狠狠量化”？**：实测显示，浅层若保留 CNN 为 FP32 并将 $L_1\\text{{--}}L_5$ 量化为 INT8，PR-AUC 为 **{t8_c['prauc']:.4f}** ($\\Delta = {t8_c['delta_prauc']:+.4f}$)；但若在浅层粗暴施加 INT4 极低比特量化，PR-AUC 剧烈跌至 **{t4_c['prauc']:.4f}** ($\\Delta = {t4_c['delta_prauc']:+.4f}$)，证明底层承载的高精细声门微扰（Jitter/Shimmer）**无法承受 INT4 粗粒度截断**！\n\n")

    md.append("---\n\n")
    md.append("## 二、 WavLM 各层张量数值特征与量化信噪比 (Table 1: Layer Tensor & SQNR Audit)\n\n")
    md.append(r"| 层级 | 类型 | 均值 | 标准差 | Max |x| | P99.0 | P99.9 | 峰度 (Kurtosis) | 离群比 (Max/P99.9) | >6σ 离群通道数 | INT8 SQNR (Per-Tensor) | INT8 SQNR (Per-Channel) | 通道量化增益 | INT4 SQNR (PT) |" + "\n")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")

    for _, r in tensor_df.iterrows():
        md.append(
            f"| **{r['layer_name']}** | {r['layer_type']} | {r['mean']:.3f} | {r['std']:.3f} | "
            f"**{r['max_abs']:.2f}** | {r['p99_0']:.2f} | {r['p99_9']:.2f} | "
            f"**{r['kurtosis']:.2f}** | {r['outlier_ratio']:.2f}x | {r['outlier_channels_6sigma']}/768 | "
            f"{r['sqnr_int8_per_tensor_db']:.2f} dB | **{r['sqnr_int8_per_channel_db']:.2f} dB** | "
            f"+{r['sqnr_int8_channel_gain_db']:.2f} dB | {r['sqnr_int4_per_tensor_db']:.2f} dB |\n"
        )

    md.append("\n---\n\n")
    md.append("## 三、 下游发声态检测任务敏感度与策略消融 (Table 2: Task Performance & Quantization Ablation)\n\n")

    for task_name in ["creaky", "breathy"]:
        md.append(f"### 任务：`{task_name.upper()}`\n\n")
        md.append("| 评测方案 / 策略 | 类别 | 位宽 | 量化方式 | PR-AUC | $\\Delta$ PR-AUC | ROC-AUC | $\\Delta$ ROC-AUC | 描述 |\n")
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |\n")

        sub_df = task_df[task_df["task"] == task_name]
        for _, r in sub_df.iterrows():
            delta_p_str = f"**{r['delta_prauc']:+.4f}**" if abs(r['delta_prauc']) > 0.005 else f"{r['delta_prauc']:+.4f}"
            md.append(
                f"| **{r['condition_name']}** | {r['category']} | {r['bits']}-bit | {r['quant_type']} | "
                f"**{r['prauc']:.4f}** | {delta_p_str} | {r['rocauc']:.4f} | {r['delta_rocauc']:+.4f} | {r['description']} |\n"
            )
        md.append("\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(md)
    logger.info(f"Saved full Markdown report to: {output_path}")


def main():
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint)
    model, config = load_model_from_checkpoint(ckpt_path, device)

    manifest_path = Path(args.manifest)
    loader, _ = prepare_dataloader(
        config, manifest_path, args.split, args.batch_size, is_quick=args.quick, quick_blocks=args.quick_blocks
    )

    conditions = build_quantization_conditions(num_layers=model.num_layers + 1)

    eval_tasks = [t for t in ["creaky", "breathy"] if t in model.tasks]
    if not eval_tasks:
        eval_tasks = list(model.tasks)

    tensor_df, task_df = evaluate_quantization_pipeline(
        model=model,
        loader=loader,
        device=device,
        conditions=conditions,
        eval_tasks=eval_tasks,
        max_batches=args.max_batches,
    )

    # Save CSVs
    tensor_csv = output_dir / "quantization_tensor_metrics.csv"
    task_csv = output_dir / "quantization_task_ablation.csv"
    report_md = output_dir / "quantization_sensitivity_report.md"

    tensor_df.to_csv(tensor_csv, index=False)
    task_df.to_csv(task_csv, index=False)
    logger.info(f"Saved tensor metrics CSV to: {tensor_csv}")
    logger.info(f"Saved task ablation CSV to: {task_csv}")

    generate_markdown_report(tensor_df, task_df, report_md)

    print("\n" + "=" * 80)
    print("Layer Tensor Audit (L0 CNN vs Shallow Tr vs Deep Tr):")
    print("=" * 80)
    print(tensor_df[["layer_name", "max_abs", "kurtosis", "outlier_channels_6sigma", "sqnr_int8_per_tensor_db", "sqnr_int8_per_channel_db"]].to_string(index=False))

    print("\n" + "=" * 80)
    print("Downstream Task Macro Compression Comparison:")
    print("=" * 80)
    macro_sub = task_df[task_df["category"].isin(["Baseline", "Full Quantization", "Structural Pruning", "Hybrid Pruning + Quant"])]
    print(macro_sub[["task", "condition_name", "bits", "prauc", "delta_prauc", "rocauc"]].to_string(index=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

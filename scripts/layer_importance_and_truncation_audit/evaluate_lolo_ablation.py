#!/usr/bin/env python
"""
WavFry Functional Auditing: Leave-One-Layer-Out (LOLO) Layer Ablation Evaluator.
Evaluates the internal necessity and decision sensitivity of each Transformer layer
in frozen SSL backbones (WavLM, HuBERT, wav2vec 2.0) across multi-task phonation heads
(Creaky Voice, Breathy Voice) using dynamic Softmax re-normalization.

Mathematical Formulation:
    h_{-l} = sum_{k != l} exp(w_k) * h_k / sum_{k != l} exp(w_k)
Implemented via logit masking (masked_logits[l] = -inf) prior to Softmax,
strictly preserving feature scale and preventing norm-shrinkage artifacts.

Computational Optimization:
    Single-Pass Forward Caching (WavLM executes ONCE per audio block, while all
    13 LOLO conditions evaluate simultaneously via GEMM in memory).

Outputs:
    - CSV table (per-task or combined) with PR-AUC, F1, ROC-AUC deltas.
    - Markdown formatted summary table with layer indispensability rankings.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

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

from wavfry.data import dataset as dataset_mod
from wavfry.data import splits as splits_mod
from wavfry.model.wavlm_classifier import WavFryClassifier
from wavfry.eval.acoustic_degradation import get_acoustic_degradation_engine


def parse_args():
    parser = argparse.ArgumentParser(description="WavFry LOLO Multi-Task Layer Ablation Evaluator")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained model checkpoint (.pt). If None, automatically searches candidate directories.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to manifest CSV. Defaults to manifests/segments_multi.csv or segments_all.csv.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["all", "creaky", "breathy"],
        help="Target task to evaluate ('creaky', 'breathy', or 'all'). Default: 'all'.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/experiments/lolo_ablation_results.csv",
        help="Path to save output CSV table.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["val", "test", "all"],
        default="val",
        help="Data split to evaluate on.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cuda' or 'cpu'). Defaults to cuda if >= 2GB VRAM free, else cpu.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--group-tiers",
        action="store_true",
        help="Also evaluate tier groups (Bottom 0-3, Middle 4-7, Top 8-11, Last-only 12).",
    )
    parser.add_argument(
        "--smoketest",
        action="store_true",
        help="Run fast smoke test with a small sample slice (32 pos + 32 neg).",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional upper bound on batches to process (useful for rapid profiling).",
    )
    parser.add_argument(
        "--noise-condition",
        type=str,
        default=None,
        help="Acoustic degradation condition for robust LOLO layer audit (e.g., 'cafe_5db', 'car_0db', 'reverb', 'street_5db'). Defaults to clean.",
    )
    return parser.parse_args()


def resolve_checkpoint(ckpt_path: Optional[str]) -> Path:
    if ckpt_path:
        p = Path(ckpt_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Specified checkpoint not found: {ckpt_path}")

    # Auto search candidates
    candidates = [
        Path("R:/checkpoints/ablation_weak_on_vsdc_real_noise/best_model.pt"),
        Path("R:/checkpoints/ablation_weak_on_vsdc_full/best_model.pt"),
        Path("R:/checkpoints/csl_frozen_wavlm_base_plus/best_model.pt"),
        Path("R:/checkpoints/frozen_wavlm_ablation/single_attn/best_model.pt"),
        PROJECT_ROOT / "exp_results/checkpoints/ablation_weak_on_vsdc_full/best_model.pt",
        PROJECT_ROOT / "exp_results/checkpoints/csl_frozen_wavlm_base_plus/best_model.pt",
        PROJECT_ROOT / "exp_results/checkpoints/frozen_wavlm_base_plus/best_model.pt",
    ]
    for c in candidates:
        if c.exists():
            print(f"[Checkpoint] Auto-resolved to existing checkpoint: {c}")
            return c

    raise FileNotFoundError(
        "No default checkpoint found. Please provide --checkpoint path to a trained model."
    )


def resolve_device(requested_device: Optional[str]) -> torch.device:
    if requested_device:
        dev = torch.device(requested_device)
        print(f"[Device] Using explicitly requested device: {dev}")
        return dev

    if torch.cuda.is_available():
        try:
            free_mem_bytes, total_mem_bytes = torch.cuda.mem_get_info()
            free_mem_mb = free_mem_bytes / (1024 * 1024)
            if free_mem_mb < 2048:
                print(
                    f"[Device Warning] GPU 0 has only {free_mem_mb:.1f} MiB free VRAM "
                    f"(likely training running concurrently). "
                    f"Auto-falling back to CPU to prevent CUDA OOM crash!"
                )
                return torch.device("cpu")
            print(f"[Device] Using CUDA (Free VRAM: {free_mem_mb:.1f} MiB)")
            return torch.device("cuda")
        except Exception:
            return torch.device("cuda")

    print("[Device] CUDA unavailable, using CPU.")
    return torch.device("cpu")


def load_model_from_checkpoint(ckpt_path: Path, device: torch.device) -> tuple[WavFryClassifier, Dict[str, Any]]:
    print(f"[Loading] Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    if not config:
        raise ValueError(f"Checkpoint at {ckpt_path} does not contain 'config'.")

    # Instantiate model
    model = WavFryClassifier(config)
    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
    # Handle possible DDP module prefix
    cleaned_sd = {}
    for k, v in state_dict.items():
        key = k[7:] if k.startswith("module.") else k
        cleaned_sd[key] = v

    model.load_state_dict(cleaned_sd, strict=False)
    model.to(device)
    model.eval()

    # Backbone is frozen in evaluation
    for p in model.wavlm.parameters():
        p.requires_grad = False
    return model, config


def prepare_dataloader(
    config: Dict[str, Any],
    manifest_path: Path,
    split_target: str,
    batch_size: int,
    is_smoketest: bool = False,
    task_filter: str = "all",
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

    if is_smoketest:
        # Pick relevant label column for sampling
        target_col = "label"
        if task_filter == "breathy" and "label_breathy" in eval_df.columns:
            target_col = "label_breathy"
        elif task_filter == "creaky" and "label_creaky" in eval_df.columns:
            target_col = "label_creaky"
        elif "label_creaky" in eval_df.columns:
            target_col = "label_creaky"

        sub_labeled = eval_df[eval_df[target_col].notna()].copy()
        pos = sub_labeled[sub_labeled[target_col] == 1]
        neg = sub_labeled[sub_labeled[target_col] == 0]
        n_pos = min(len(pos), 32)
        n_neg = min(len(neg), 32)
        eval_df = pd.concat([pos.sample(n=n_pos, random_state=42), neg.sample(n=n_neg, random_state=42)]).sample(frac=1.0, random_state=42).reset_index(drop=True)
        print(f"[Smoketest] Sliced evaluation dataset to {len(eval_df)} segments using label column '{target_col}'.")

    paradigm = config.get("training", {}).get("paradigm", "block")
    if paradigm == "block":
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
        collate = collate_block_fn
    else:
        ds = dataset_mod.CreakyVowelDataset(
            eval_df,
            context_padding_ms=config.get("training", {}).get("segment", {}).get("context_padding_ms", 30),
        )
        collate = dataset_mod.collate_fn

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
        pin_memory=False,
    )
    return loader, paradigm


def build_lolo_conditions(num_layers: int, group_tiers: bool = False) -> List[Dict[str, Any]]:
    """Build condition masks and metadata for Leave-One-Layer-Out."""
    conditions: List[Dict[str, Any]] = []

    # 1. Full Model Baseline
    conditions.append({
        "id": "full_model",
        "name": "Full Model Baseline",
        "ablated_layer": "None",
        "relative_depth": "1.0",
        "mask": None,
        "is_tier": False,
    })

    # 2. Leave-One-Layer-Out (L0 to L12)
    for l in range(num_layers):
        mask = torch.ones(num_layers, dtype=torch.bool)
        mask[l] = False
        layer_name = "CNN_Feat" if l == 0 else f"Transformer_L{l}"
        rel_depth = f"{l / (num_layers - 1):.2f}"
        conditions.append({
            "id": f"leave_out_layer_{l}",
            "name": f"Ablate {layer_name}",
            "ablated_layer": layer_name,
            "relative_depth": rel_depth,
            "layer_index": l,
            "mask": mask,
            "is_tier": False,
        })

    # 3. Optional Group Tiers
    if group_tiers:
        tiers = {
            "tier_bottom_0_3": ("Bottom Layers 0-3 Only", [0, 1, 2, 3]),
            "tier_middle_4_7": ("Middle Layers 4-7 Only", [4, 5, 6, 7]),
            "tier_top_8_11": ("Top Layers 8-11 Only", [8, 9, 10, 11]),
            "tier_last_only_12": ("Last Layer 12 Only", [12]),
        }
        for tier_id, (tier_name, active_indices) in tiers.items():
            mask = torch.zeros(num_layers, dtype=torch.bool)
            for idx in active_indices:
                mask[idx] = True
            conditions.append({
                "id": tier_id,
                "name": tier_name,
                "ablated_layer": tier_name,
                "relative_depth": "Group",
                "active_indices": active_indices,
                "mask": mask,
                "is_tier": True,
            })

    return conditions


def compute_binary_metrics(y_logits: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    valid_mask = ~np.isnan(y_true)
    y_logits = y_logits[valid_mask]
    y_true = y_true[valid_mask].astype(int)

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {
            "auc_pr": 0.0,
            "f1": 0.0,
            "auc_roc": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "n_samples": int(len(y_true)),
            "n_pos": int(y_true.sum()) if len(y_true) > 0 else 0,
        }

    probs = 1.0 / (1.0 + np.exp(-y_logits))
    auc_pr = float(average_precision_score(y_true, probs))
    try:
        auc_roc = float(roc_auc_score(y_true, probs))
    except Exception:
        auc_roc = 0.0

    # Grid search for optimal F1
    thresholds = np.linspace(0.1, 0.9, 41)
    best_f1, best_acc, best_p, best_r = 0.0, 0.0, 0.0, 0.0
    for tau in thresholds:
        preds = (probs >= tau).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
        if f > best_f1:
            best_f1 = float(f)
            best_p = float(p)
            best_r = float(r)
            best_acc = float((preds == y_true).mean())

    return {
        "auc_pr": auc_pr,
        "f1": best_f1,
        "auc_roc": auc_roc,
        "accuracy": best_acc,
        "precision": best_p,
        "recall": best_r,
        "n_samples": int(len(y_true)),
        "n_pos": int(y_true.sum()),
    }


def evaluate_lolo_single_pass(
    model: WavFryClassifier,
    loader: DataLoader,
    device: torch.device,
    paradigm: str,
    conditions: List[Dict[str, Any]],
    eval_tasks: List[str],
    max_batches: Optional[int] = None,
    noise_condition: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Execute single-pass forward caching LOLO evaluation across conditions and tasks."""
    model.eval()
    num_layers = model.num_layers + 1
    num_conditions = len(conditions)
    head_dtype = next(model.heads.parameters()).dtype

    # Extract learned scalar mix weights for each task
    learned_weights: Dict[str, List[float]] = {}
    stacked_cond_weights: Dict[str, torch.Tensor] = {}

    for task_name in model.tasks:
        if model.use_scalar_mix and task_name in model.scalar_mix_weights:
            raw_logits = model.scalar_mix_weights[task_name].detach().to(device)
            base_w = torch.softmax(raw_logits, dim=0).cpu().numpy().tolist()
            learned_weights[task_name] = base_w

            # Precompute normalized weights for each condition
            cond_w_list = []
            for cond in conditions:
                if cond["mask"] is None:
                    w = torch.softmax(raw_logits, dim=0)
                else:
                    m_logits = raw_logits.clone()
                    m_logits[~cond["mask"].to(device)] = -float("inf")
                    w = torch.softmax(m_logits, dim=0)
                cond_w_list.append(w)
            stacked_cond_weights[task_name] = torch.stack(cond_w_list, dim=0)  # [num_conditions, 13]
        else:
            uniform = torch.full((num_layers,), 1.0 / num_layers, device=device)
            learned_weights[task_name] = [1.0 / num_layers] * num_layers
            stacked_cond_weights[task_name] = uniform.unsqueeze(0).expand(num_conditions, -1)

    # Accumulators: {task: [condition_idx -> dict/list of preds]}
    cond_interval_preds: Dict[str, List[Dict[str, list]]] = {
        task: [{} for _ in range(num_conditions)] for task in eval_tasks
    }
    cond_segment_preds: Dict[str, List[list]] = {
        task: [[] for _ in range(num_conditions)] for task in eval_tasks
    }
    cond_segment_labels: Dict[str, List[list]] = {
        task: [[] for _ in range(num_conditions)] for task in eval_tasks
    }

    start_time = time.time()
    batch_count = 0
    total_batches = min(len(loader), max_batches) if max_batches else len(loader)

    print(f"\n[Evaluating] Processing {total_batches} batches across {num_conditions} conditions for tasks {eval_tasks} (degradation: {noise_condition or 'clean'})...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches and batch_idx >= max_batches:
                break
            batch_count += 1
            if batch_count % 25 == 0 or batch_count == total_batches:
                elapsed = time.time() - start_time
                print(f"  Processed {batch_count}/{total_batches} batches ({elapsed:.1f}s)...", flush=True)

            if paradigm == "block":
                waveforms, bounds, labels, valid, lengths, speakers, block_ids, interval_ids = batch
                waveforms = waveforms.to(device)
                if noise_condition:
                    waveforms = get_acoustic_degradation_engine().apply_tensor(waveforms, noise_condition, seed=42 + batch_idx * 1009)
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
                flat_H = H.view(L, -1)  # [13, B * T * D]

                out_lengths = model.wavlm._get_feat_extract_output_lengths(lengths).to(device).clamp(min=1, max=T)
                seq_mask = torch.arange(T, device=device).unsqueeze(0) < out_lengths.unsqueeze(1)

                # 2. Vectorized mixture for all tasks
                mixed_by_task: Dict[str, torch.Tensor] = {}
                for task_name in model.tasks:
                    mixed_by_task[task_name] = (
                        stacked_cond_weights[task_name].to(flat_H.dtype) @ flat_H
                    ).view(num_conditions, B, T, D).to(head_dtype)

                # 3. Head forward pass for each condition
                for c_idx in range(num_conditions):
                    hidden_cond = {task_name: mixed_by_task[task_name][c_idx] for task_name in model.tasks}
                    block_out = model._forward_block(
                        hidden_by_task=hidden_cond,
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
            else:
                # Segment Mode
                waveforms, labels, lengths, speakers, meta = batch
                waveforms = waveforms.to(device)
                if noise_condition:
                    waveforms = get_acoustic_degradation_engine().apply_tensor(waveforms, noise_condition, seed=42 + batch_idx * 1009)
                labels = labels.to(device)
                lengths = lengths.to(device)

                wavlm_dtype = next(model.wavlm.parameters()).dtype
                x_in = waveforms.to(wavlm_dtype) if waveforms.dtype != wavlm_dtype else waveforms
                outputs = model.wavlm(x_in, output_hidden_states=True)
                raw_hidden_states = [h.detach() for h in outputs.hidden_states]
                H = torch.stack(raw_hidden_states, dim=0)
                L, B, T, D = H.shape
                flat_H = H.view(L, -1)

                out_lengths = model.wavlm._get_feat_extract_output_lengths(lengths).to(device).clamp(min=1, max=T)
                seq_mask = torch.arange(T, device=device).unsqueeze(0) < out_lengths.unsqueeze(1)

                mixed_by_task = {}
                for task_name in model.tasks:
                    mixed_by_task[task_name] = (
                        stacked_cond_weights[task_name].to(flat_H.dtype) @ flat_H
                    ).view(num_conditions, B, T, D).to(head_dtype)

                for c_idx in range(num_conditions):
                    for task_name in eval_tasks:
                        task_idx = model.tasks.index(task_name) if task_name in model.tasks else 0
                        t_labels = labels[..., task_idx] if (labels.ndim > 1 and labels.shape[-1] > task_idx) else labels
                        pooled = model._pool_batch(
                            mixed_by_task[task_name][c_idx],
                            valid_mask=seq_mask,
                            lengths=out_lengths,
                            task=task_name,
                        )
                        c_logits = model.heads[task_name](pooled).squeeze(-1)
                        cond_segment_preds[task_name][c_idx].extend(c_logits.cpu().numpy().tolist())
                        cond_segment_labels[task_name][c_idx].extend(t_labels.cpu().numpy().tolist())

    # Aggregate and compute metrics
    task_results: Dict[str, List[Dict[str, Any]]] = {}

    for task_name in eval_tasks:
        results: List[Dict[str, Any]] = []
        base_auc_pr = 0.0
        base_f1 = 0.0

        for c_idx, cond in enumerate(conditions):
            if paradigm == "block":
                rec = cond_interval_preds[task_name][c_idx]
                y_logits = np.array([item[0] / item[2] for item in rec.values()], dtype=np.float32)
                y_labels = np.array([item[1] for item in rec.values()], dtype=np.float32)
            else:
                y_logits = np.array(cond_segment_preds[task_name][c_idx], dtype=np.float32)
                y_labels = np.array(cond_segment_labels[task_name][c_idx], dtype=np.float32)

            metrics = compute_binary_metrics(y_logits, y_labels)
            auc_pr = metrics["auc_pr"]
            f1 = metrics["f1"]

            if cond["id"] == "full_model":
                base_auc_pr = auc_pr
                base_f1 = f1
                delta_pr = 0.0
                delta_f1 = 0.0
                drop_pct = 0.0
                mix_w = 1.0
            else:
                delta_pr = base_auc_pr - auc_pr
                delta_f1 = base_f1 - f1
                drop_pct = (delta_pr / base_auc_pr * 100.0) if base_auc_pr > 0 else 0.0
                if cond.get("is_tier", False):
                    mix_w = sum(learned_weights[task_name][i] for i in cond["active_indices"])
                else:
                    l_idx = cond.get("layer_index", 0)
                    mix_w = learned_weights[task_name][l_idx] if l_idx < len(learned_weights[task_name]) else 0.0

            results.append({
                "task": task_name,
                "condition": cond["id"],
                "ablated_layer": cond["ablated_layer"],
                "relative_depth": cond["relative_depth"],
                "scalar_mix_weight": round(mix_w, 5),
                "auc_pr": round(auc_pr, 4),
                "f1": round(f1, 4),
                "auc_roc": round(metrics["auc_roc"], 4),
                "delta_auc_pr": round(delta_pr, 4),
                "delta_f1": round(delta_f1, 4),
                "drop_pct": round(drop_pct, 2),
                "n_samples": metrics["n_samples"],
                "n_pos": metrics["n_pos"],
            })

        task_results[task_name] = results

    return task_results


def generate_markdown_summary(
    md_path: Path,
    ckpt_name: str,
    split_name: str,
    paradigm_name: str,
    task_results: Dict[str, List[Dict[str, Any]]],
):
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# WavFry Leave-One-Layer-Out (LOLO) Multi-Task Functional Audit\n\n")
        f.write(f"- **Checkpoint**: `{ckpt_name}`\n")
        f.write(f"- **Split**: `{split_name}` | **Paradigm**: `{paradigm_name}`\n")
        f.write(f"- **Evaluation Time**: `{time.strftime('%Y-%m-%d %H:%M:%S')}`\n\n")

        for task_name, rows in task_results.items():
            base_row = next(r for r in rows if r["condition"] == "full_model")
            lolo_rows = [r for r in rows if r["condition"].startswith("leave_out_layer_")]
            tier_rows = [r for r in rows if r["condition"].startswith("tier_")]
            lolo_sorted = sorted(lolo_rows, key=lambda x: x["delta_auc_pr"], reverse=True)

            f.write(f"## Task: `{task_name.upper()}`\n\n")
            f.write(f"- **Full Model Baseline PR-AUC**: `{base_row['auc_pr']:.4f}` | **F1**: `{base_row['f1']:.4f}` | **ROC-AUC**: `{base_row['auc_roc']:.4f}`\n")
            f.write(f"- **Evaluated Labeled Samples**: `{base_row['n_samples']}` (Positives: `{base_row['n_pos']}`)\n\n")

            f.write("### Layer Indispensability Ranking (Sorted by Decision Sensitivity ΔPR-AUC)\n\n")
            f.write("| Rank | Layer Name | Rel Depth | Mix Weight | Ablated PR-AUC | ΔPR-AUC | Drop (%) | Necessity Level |\n")
            f.write("|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|\n")

            for rank, row in enumerate(lolo_sorted, 1):
                if row["delta_auc_pr"] > 0.005:
                    nec = "🔥🔥 Critical"
                elif row["delta_auc_pr"] > 0.001:
                    nec = "⚠️ Moderate"
                elif row["delta_auc_pr"] >= 0.0:
                    nec = "🌱 Marginal"
                else:
                    nec = "🛡️ Redundant / Suppressive"

                delta_sign = "+" if row["delta_auc_pr"] >= 0 else ""
                f.write(
                    f"| {rank} | **{row['ablated_layer']}** | {row['relative_depth']} | "
                    f"{row['scalar_mix_weight']*100:.2f}% | {row['auc_pr']:.4f} | "
                    f"**{delta_sign}{row['delta_auc_pr']:.4f}** | -{row['drop_pct']:.1f}% | {nec} |\n"
                )

            if tier_rows:
                f.write("\n### Tier Group Ablations\n\n")
                f.write("| Tier Name | Retained Mix Weight | Ablated PR-AUC | ΔPR-AUC | Drop (%) |\n")
                f.write("|:---|:---:|:---:|:---:|:---:|\n")
                for row in tier_rows:
                    delta_sign = "+" if row["delta_auc_pr"] >= 0 else ""
                    f.write(
                        f"| **{row['ablated_layer']}** | {row['scalar_mix_weight']*100:.2f}% | "
                        f"{row['auc_pr']:.4f} | **{delta_sign}{row['delta_auc_pr']:.4f}** | -{row['drop_pct']:.1f}% |\n"
                    )

            f.write("\n---\n\n")

        f.write("*Note: Positive ΔPR-AUC indicates that the model decision degrades upon removal of this layer (functional necessity). Negative ΔPR-AUC indicates slight performance improvement or redundancy upon ablation.*\n")


def main():
    args = parse_args()
    device = resolve_device(args.device)

    ckpt_path = resolve_checkpoint(args.checkpoint)
    model, config = load_model_from_checkpoint(ckpt_path, device)

    # Determine tasks
    model_tasks = getattr(model, "tasks", ["creaky"])
    if args.task == "all":
        eval_tasks = [t for t in ["creaky", "breathy"] if t in model_tasks]
    else:
        if args.task not in model_tasks:
            raise ValueError(f"Requested task '{args.task}' not supported by model tasks: {model_tasks}")
        eval_tasks = [args.task]

    # Manifest resolution
    manifest_path = Path(args.manifest) if args.manifest else (PROJECT_ROOT / "manifests" / "segments_multi.csv")
    if not manifest_path.exists():
        manifest_path = PROJECT_ROOT / "manifests" / "segments_all.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    loader, paradigm = prepare_dataloader(
        config=config,
        manifest_path=manifest_path,
        split_target=args.split,
        batch_size=args.batch_size,
        is_smoketest=args.smoketest,
        task_filter=args.task,
    )

    num_layers = model.num_layers + 1
    conditions = build_lolo_conditions(num_layers, group_tiers=args.group_tiers)

    print("\n" + "=" * 78)
    print("      WavFry LOLO (Leave-One-Layer-Out) Multi-Task Evaluator")
    print(f"  Checkpoint: {ckpt_path.name}")
    print(f"  Tasks: {eval_tasks} | Split: {args.split.upper()} | Paradigm: {paradigm.upper()}")
    print(f"  Conditions: {len(conditions)} | Device: {device} | Single-Pass Caching: ENABLED")
    print("=" * 78)

    task_results = evaluate_lolo_single_pass(
        model=model,
        loader=loader,
        device=device,
        paradigm=paradigm,
        conditions=conditions,
        eval_tasks=eval_tasks,
        max_batches=args.max_batches,
        noise_condition=args.noise_condition,
    )

    # Flatten results for CSV
    all_rows: List[Dict[str, Any]] = []
    for t_name, rows in task_results.items():
        all_rows.extend(rows)

    out_path = Path(args.output) if Path(args.output).is_absolute() else (PROJECT_ROOT / args.output)
    if args.noise_condition and args.output == "data/experiments/lolo_ablation_results.csv":
        out_path = out_path.with_name(f"lolo_ablation_results_{args.noise_condition}.csv")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    res_df = pd.DataFrame(all_rows)
    res_df.to_csv(out_path, index=False)
    print(f"\n[Saved] Multi-task LOLO Ablation CSV saved to: {out_path}")

    md_path = out_path.with_suffix(".md")
    generate_markdown_summary(
        md_path=md_path,
        ckpt_name=ckpt_path.name,
        split_name=args.split,
        paradigm_name=paradigm,
        task_results=task_results,
    )
    print(f"[Saved] LOLO Multi-Task Markdown summary saved to: {md_path}")

    # Print summary console table
    for t_name, rows in task_results.items():
        base_r = next(r for r in rows if r["condition"] == "full_model")
        print(f"\n>>> Summary for {t_name.upper()} (Baseline PR-AUC: {base_r['auc_pr']:.4f}, F1: {base_r['f1']:.4f}):")
        lolo_sorted = sorted([r for r in rows if r["condition"].startswith("leave_out_layer_")], key=lambda x: x["delta_auc_pr"], reverse=True)
        for rank, r in enumerate(lolo_sorted[:5], 1):
            print(f"    Rank {rank}: {r['ablated_layer']:<16} ΔPR-AUC: +{r['delta_auc_pr']:.4f} (-{r['drop_pct']:.1f}%) | Weight: {r['scalar_mix_weight']*100:.2f}%")


if __name__ == "__main__":
    main()

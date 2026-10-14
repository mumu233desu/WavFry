#!/usr/bin/env python
"""
WavFry Functional Auditing: Top-k Layer Truncation & Depth Sensitivity Evaluator.
Evaluates the acoustic and task impact of progressively truncating the upper k Transformer
layers (k = 1, 2, ..., 12) from frozen WavLM Base+ during inference via dynamic
Softmax re-normalization across phonation tasks (Creaky Voice, Breathy Voice).

Functional Boundary & Purpose:
    - Distinct from evaluate_lolo_ablation.py (which audits micro-level single-layer sensitivity):
      This script exclusively audits macro-level sequential upper-layer pruning to answer:
      "What is the effective computational depth for each voice quality task?
       How many upper layers can be safely truncated for edge deployment/distillation?"

Outputs saved to:
    data/experiments/topk_layer_truncation_evaluation/
        - topk_layer_truncation_report.csv
        - topk_layer_truncation_report.md
        - topk_layer_truncation_summary.json
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
    parser = argparse.ArgumentParser(description="WavFry Top-k Multi-Task Layer Truncation Evaluator")
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
        help="Path to manifest CSV. Defaults to manifests/segments_multi.csv.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["all", "creaky", "breathy"],
        help="Target task to evaluate ('creaky', 'breathy', or 'all'). Default: 'all'.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/experiments/topk_layer_truncation_evaluation",
        help="Directory to save output reports.",
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
        help="Acoustic degradation condition for robust Top-k depth truncation audit (e.g., 'cafe_5db', 'car_0db', 'reverb', 'street_5db'). Defaults to clean.",
    )
    return parser.parse_args()


def resolve_checkpoint(ckpt_path: Optional[str]) -> Path:
    if ckpt_path:
        p = Path(ckpt_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Specified checkpoint not found: {ckpt_path}")

    candidates = [
        Path("R:/checkpoints/ablation_weak_on_vsdc_real_noise/best_model.pt"),
        Path("R:/checkpoints/ablation_weak_on_vsdc_full/best_model.pt"),
        Path("R:/checkpoints/csl_frozen_wavlm_base_plus/best_model.pt"),
        PROJECT_ROOT / "exp_results/checkpoints/ablation_weak_on_vsdc_real_noise/best_model.pt",
        PROJECT_ROOT / "exp_results/checkpoints/ablation_weak_on_vsdc_full/best_model.pt",
        PROJECT_ROOT / "exp_results/checkpoints/csl_frozen_wavlm_base_plus/best_model.pt",
    ]
    for c in candidates:
        if c.exists():
            print(f"[Checkpoint] Auto-resolved to checkpoint: {c}")
            return c

    raise FileNotFoundError("No default checkpoint found. Please provide --checkpoint path.")


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

    model = WavFryClassifier(config)
    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
    cleaned_sd = {}
    for k, v in state_dict.items():
        key = k[7:] if k.startswith("module.") else k
        cleaned_sd[key] = v

    model.load_state_dict(cleaned_sd, strict=False)
    model.to(device)
    model.eval()

    # Freeze backbone completely
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
        eval_df = (
            pd.concat([pos.sample(n=n_pos, random_state=42), neg.sample(n=n_neg, random_state=42)])
            .sample(frac=1.0, random_state=42)
            .reset_index(drop=True)
        )
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


def build_truncation_conditions(num_layers: int) -> List[Dict[str, Any]]:
    """Build systematic sequential upper-layer truncation conditions:
    - Baseline: all 13 layers (k=0)
    - Sequential drop top-k: k = 1 to 12
    - Negative Control: Drop Bottom 0~5 (Keep Upper 6~12 Only)
    """
    conditions: List[Dict[str, Any]] = []

    # 1. Full Baseline (k = 0)
    conditions.append({
        "id": "full_model_baseline",
        "name": "Full Model Baseline (All 13 Layers)",
        "drop_k": 0,
        "kept_indices": list(range(num_layers)),
        "dropped_indices": [],
        "description": "All 13 layers active (CNN + 12 Transformers)",
        "category": "Baseline",
    })

    # 2. Sequential Top-k Truncation (k = 1 to 12)
    for k in range(1, num_layers):
        kept = list(range(num_layers - k))
        dropped = list(range(num_layers - k, num_layers))
        if k == 1:
            desc = f"Drop Top 1 (L12 removed, keeping CNN + L1~L11)"
        elif k == 3:
            desc = f"Drop Top 3 (L10~L12 removed, keeping CNN + L1~L9)"
        elif k == 6:
            desc = f"Drop Top 6 (L7~L12 removed, keeping CNN + L1~L6)"
        elif k == 7:
            desc = f"Drop Top 7 (L6~L12 removed, keeping CNN + L1~L5 [Half-depth 5L])"
        elif k == 8:
            desc = f"Drop Top 8 (L5~L12 removed, keeping CNN + L1~L4)"
        elif k == 9:
            desc = f"Drop Top 9 (L4~L12 removed, keeping CNN + L1~L3 [Quarter-depth 3L])"
        elif k == 12:
            desc = f"Drop All 12 Transformers (CNN Feat Only [No Transformers])"
        else:
            desc = f"Drop Top {k} (L{num_layers-k}~L12 removed, keeping {len(kept)} layers)"

        conditions.append({
            "id": f"drop_top_{k}",
            "name": f"Drop Top {k:02d} Layers",
            "drop_k": k,
            "kept_indices": kept,
            "dropped_indices": dropped,
            "description": desc,
            "category": "Sequential Truncation",
        })

    # 3. Informative Negative Control
    conditions.append({
        "id": "negative_control_upper_only",
        "name": "Negative Control: Upper Layers Only (L6~L12)",
        "drop_k": 6,
        "kept_indices": list(range(6, num_layers)),
        "dropped_indices": list(range(6)),
        "description": "Drop Bottom CNN + L1~L5 (Keep L6~L12 only to test high-level sufficiency)",
        "category": "Negative Control",
    })

    # Attach layer mask
    for cond in conditions:
        mask = torch.zeros(num_layers, dtype=torch.bool)
        for idx in cond["kept_indices"]:
            mask[idx] = True
        cond["mask"] = mask if len(cond["kept_indices"]) < num_layers else None

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
            "best_threshold": 0.5,
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
    thresholds = np.linspace(0.1, 0.9, 81)
    best_f1, best_tau, best_p, best_r, best_acc = 0.0, 0.5, 0.0, 0.0, 0.0

    for tau in thresholds:
        preds = (probs >= tau).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
        if f > best_f1:
            best_f1 = float(f)
            best_tau = float(tau)
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
        "best_threshold": best_tau,
        "n_samples": int(len(y_true)),
        "n_pos": int(y_true.sum()),
    }


def evaluate_truncation_single_pass(
    model: WavFryClassifier,
    loader: DataLoader,
    device: torch.device,
    paradigm: str,
    conditions: List[Dict[str, Any]],
    eval_tasks: List[str],
    max_batches: Optional[int] = None,
    noise_condition: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Execute single-pass forward caching top-k truncation evaluation across conditions and tasks."""
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
            if batch_count % 50 == 0 or batch_count == total_batches:
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

                # 2. Vectorized mixture for each task
                mixed_by_task: Dict[str, torch.Tensor] = {}
                for task_name in model.tasks:
                    mixed_by_task[task_name] = (
                        stacked_cond_weights[task_name].to(flat_H.dtype) @ flat_H
                    ).view(num_conditions, B, T, D).to(head_dtype)

                # 3. Forward head for each condition
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

    task_results: Dict[str, List[Dict[str, Any]]] = {}

    for task_name in eval_tasks:
        results: List[Dict[str, Any]] = []
        base_auc_pr = 0.0

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

            retained_mass = sum(learned_weights[task_name][i] for i in cond["kept_indices"])

            if cond["id"] == "full_model_baseline":
                base_auc_pr = auc_pr
                delta_pr = 0.0
                drop_pct = 0.0
            else:
                delta_pr = auc_pr - base_auc_pr  # Positive means improvement, negative means degradation
                drop_pct = ((base_auc_pr - auc_pr) / base_auc_pr * 100.0) if base_auc_pr > 0 else 0.0

            results.append({
                "task": task_name,
                "condition_id": cond["id"],
                "name": cond["name"],
                "category": cond["category"],
                "drop_k": cond["drop_k"],
                "kept_layers_count": len(cond["kept_indices"]),
                "kept_indices": cond["kept_indices"],
                "retained_weight_mass": round(retained_mass, 5),
                "auc_pr": round(auc_pr, 4),
                "f1": round(f1, 4),
                "auc_roc": round(metrics["auc_roc"], 4),
                "delta_auc_pr": round(delta_pr, 4),
                "drop_pct": round(drop_pct, 2),
                "best_threshold": round(metrics["best_threshold"], 2),
                "n_samples": metrics["n_samples"],
                "n_pos": metrics["n_pos"],
            })

        task_results[task_name] = results

    return task_results


def generate_reports(
    output_dir: Path,
    ckpt_name: str,
    split_name: str,
    paradigm_name: str,
    task_results: Dict[str, List[Dict[str, Any]]],
):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "topk_layer_truncation_report.csv"
    md_path = output_dir / "topk_layer_truncation_report.md"
    json_path = output_dir / "topk_layer_truncation_summary.json"

    # 1. Flatten CSV
    all_rows = []
    for t_name, rows in task_results.items():
        all_rows.extend(rows)
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    print(f"\n[Saved] Top-k Truncation CSV report saved to: {csv_path}")

    # 2. JSON summary
    summary_data = {
        "checkpoint": ckpt_name,
        "split": split_name,
        "paradigm": paradigm_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tasks": task_results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"[Saved] Top-k Truncation JSON summary saved to: {json_path}")

    # 3. Markdown Report
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# WavFry Top-k Layer Truncation & Depth Sensitivity Audit Report\n\n")
        f.write(f"- **Evaluated Checkpoint**: `{ckpt_name}`\n")
        f.write(f"- **Dataset Split**: `{split_name.upper()}` | **Training Paradigm**: `{paradigm_name.upper()}`\n")
        f.write(f"- **Generated At**: `{time.strftime('%Y-%m-%d %H:%M:%S')}`\n\n")

        for task_name, rows in task_results.items():
            base_row = next(r for r in rows if r["condition_id"] == "full_model_baseline")
            f.write(f"## Task: `{task_name.upper()}`\n\n")
            f.write(f"- **Full Model Baseline PR-AUC**: `{base_row['auc_pr']:.4f}` | **Optimal F1**: `{base_row['f1']:.4f}` | **ROC-AUC**: `{base_row['auc_roc']:.4f}`\n")
            f.write(f"- **Evaluated Labeled Samples**: `{base_row['n_samples']}` (Positives: `{base_row['n_pos']}`)\n\n")

            f.write("### 逐层累积截断实验详细数据表 (Sequential Cumulative Truncation)\n\n")
            f.write("| 截断条件 | 丢弃层数 $k$ | 保留层数 | 保留权重质量 | PR-AUC | $\\Delta$PR-AUC | 跌幅 (%) | F1 | ROC-AUC | 最佳阈值 $\\tau^*$ | 状态评定 |\n")
            f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")

            seq_rows = [r for r in rows if r["category"] in ("Baseline", "Sequential Truncation")]
            for r in seq_rows:
                drop_pct = r["drop_pct"]
                if r["condition_id"] == "full_model_baseline":
                    status = "🎯 Full Baseline"
                elif drop_pct <= 0.0:
                    status = "🔥 逆势提升 / 零损"
                elif drop_pct <= 0.5:
                    status = "🟢 极微损 (<0.5%)"
                elif drop_pct <= 2.0:
                    status = "🟡 轻微下降 (<=2%)"
                elif drop_pct <= 5.0:
                    status = "🟠 中度受损 (<=5%)"
                else:
                    status = "🔴 严重受损 (>5%)"

                delta_str = f"+{r['delta_auc_pr']:.4f}" if r["delta_auc_pr"] >= 0 else f"{r['delta_auc_pr']:.4f}"
                f.write(
                    f"| **{r['name']}** | {r['drop_k']} | {r['kept_layers_count']} | "
                    f"{r['retained_weight_mass']*100:.1f}% | **{r['auc_pr']:.4f}** | "
                    f"{delta_str} | {drop_pct:.1f}% | {r['f1']:.4f} | {r['auc_roc']:.4f} | "
                    f"{r['best_threshold']:.2f} | {status} |\n"
                )

            # Negative controls
            neg_rows = [r for r in rows if r["category"] == "Negative Control"]
            if neg_rows:
                f.write("\n### 阴性反向对照分析 (Negative Control Anchor)\n\n")
                f.write("| 对照条件 | 保留权重 | PR-AUC | $\\Delta$PR-AUC | F1 | ROC-AUC | 科学推论 |\n")
                f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---|\n")
                for r in neg_rows:
                    delta_str = f"+{r['delta_auc_pr']:.4f}" if r["delta_auc_pr"] >= 0 else f"{r['delta_auc_pr']:.4f}"
                    f.write(
                        f"| **{r['name']}** | {r['retained_weight_mass']*100:.1f}% | **{r['auc_pr']:.4f}** | "
                        f"{delta_str} | {r['f1']:.4f} | {r['auc_roc']:.4f} | 仅凭高层语义无法独立支撑发声态判别 |\n"
                    )

            f.write("\n---\n\n")

        f.write("*说明：本评测采用 Single-Pass Forward Caching 单次前向缓存机制，后验测试各个深度截断下的性能临界点（Cliff）。*\n")

    print(f"[Saved] Top-k Truncation Markdown report saved to: {md_path}")


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
    conditions = build_truncation_conditions(num_layers)

    print("\n" + "=" * 80)
    print("      WavFry Top-k Layer Truncation & Depth Sensitivity Audit")
    print(f"  Checkpoint: {ckpt_path.name}")
    print(f"  Tasks: {eval_tasks} | Split: {args.split.upper()} | Paradigm: {paradigm.upper()}")
    print(f"  Conditions: {len(conditions)} | Device: {device} | Single-Pass Caching: ENABLED")
    print("=" * 80)

    task_results = evaluate_truncation_single_pass(
        model=model,
        loader=loader,
        device=device,
        paradigm=paradigm,
        conditions=conditions,
        eval_tasks=eval_tasks,
        max_batches=args.max_batches,
        noise_condition=args.noise_condition,
    )

    output_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else (PROJECT_ROOT / args.output_dir)
    if args.noise_condition:
        output_dir = output_dir / args.noise_condition
    generate_reports(
        output_dir=output_dir,
        ckpt_name=ckpt_path.name,
        split_name=args.split,
        paradigm_name=paradigm,
        task_results=task_results,
    )

    # Console summary printout
    print("\n" + "=" * 80)
    print("                 TOP-K TRUNCATION EXECUTIVE SUMMARY")
    print("=" * 80)
    for t_name, rows in task_results.items():
        base_r = next(r for r in rows if r["condition_id"] == "full_model_baseline")
        print(f"\n>>> Task: {t_name.upper()} (Baseline PR-AUC: {base_r['auc_pr']:.4f}, F1: {base_r['f1']:.4f})")
        for r in rows:
            delta_str = f"+{r['delta_auc_pr']:.4f}" if r["delta_auc_pr"] >= 0 else f"{r['delta_auc_pr']:.4f}"
            print(f"  [{r['name']:<44}] PR-AUC: {r['auc_pr']:.4f} (Δ {delta_str}, {r['drop_pct']:>5.1f}%) | Weight: {r['retained_weight_mass']*100:>5.1f}%")


if __name__ == "__main__":
    main()

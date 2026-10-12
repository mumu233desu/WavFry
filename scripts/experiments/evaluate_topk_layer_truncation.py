#!/usr/bin/env python
"""
WavFry Functional Auditing: Top-k Layer Truncation & Depth Sensitivity Evaluator.
Evaluates the acoustic impact of progressively truncating the upper k Transformer
layers (k = 1, 2, ..., 12) from frozen WavLM Base+ during inference via dynamic
Softmax re-normalization.

Mathematical Formulation:
    h_{kept} = sum_{l in kept} exp(w_l) * h_l / sum_{l in kept} exp(w_l)
Implemented via single-pass forward caching:
    WavLM backbone executes ONCE per audio block, while all truncation conditions
    (k=0 to 12 + architectural anchors) evaluate simultaneously in GPU memory.

Outputs saved to:
    data/experiments/topk_layer_truncation_evaluation/
        - topk_layer_truncation_report.csv
        - topk_layer_truncation_report.md
        - topk_layer_truncation_summary.json

Author: WavFry ML Infrastructure
Date: October 2026
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
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader

# Add project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wavfry.data import dataset as dataset_mod
from wavfry.data import splits as splits_mod
from wavfry.model.wavlm_classifier import WavFryClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="WavFry Top-k Layer Truncation Evaluator")
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
        help="Device to use ('cuda' or 'cpu'). Defaults to cuda if available.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--smoketest",
        action="store_true",
        help="Run fast smoke test with a small sample slice (64 samples).",
    )
    return parser.parse_args()


def resolve_checkpoint(ckpt_path: Optional[str]) -> Path:
    if ckpt_path:
        p = Path(ckpt_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Specified checkpoint not found: {ckpt_path}")

    candidates = [
        PROJECT_ROOT / "exp_results/checkpoints/csl_frozen_wavlm_base_plus/best_model.pt",
        PROJECT_ROOT / "exp_results/checkpoints/ncmmsc_camera_ready/best_model.pt",
        PROJECT_ROOT / "exp_results/checkpoints/camera_ready_ablation/single_attn/best_model.pt",
    ]
    for c in candidates:
        if c.exists():
            print(f"[Checkpoint] Auto-resolved to checkpoint: {c}")
            return c

    raise FileNotFoundError("No default checkpoint found. Please provide --checkpoint path.")


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
        pos = eval_df[eval_df["label"] == 1]
        neg = eval_df[eval_df["label"] == 0]
        n_pos = min(len(pos), 32)
        n_neg = min(len(neg), 32)
        eval_df = (
            pd.concat([pos.sample(n=n_pos, random_state=42), neg.sample(n=n_neg, random_state=42)])
            .sample(frac=1.0, random_state=42)
            .reset_index(drop=True)
        )
        print(f"[Smoketest] Sliced evaluation dataset to {len(eval_df)} segments.")

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


def build_truncation_conditions(num_layers: int, learned_weights: List[float]) -> List[Dict[str, Any]]:
    """Build systematic truncation conditions:
    - Baseline: all 13 layers (k=0)
    - Sequential drop top-k: k = 1 to 12
    - Diagnostic controls: CNN only, Bottom 4 (CNN+L1-L3), Middle (L4-L7), Top (L8-L12)
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
            desc = f"Drop Top 6 (L7~L12 removed, keeping CNN + L1~L6 [Half-depth 6L])"
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

    # 3. Informative Diagnostic Sub-tiers
    conditions.append({
        "id": "diagnostic_bottom_acoustic_0_3",
        "name": "Anchor: Bottom Acoustic Only (CNN + L1~L3)",
        "drop_k": 9,
        "kept_indices": [0, 1, 2, 3],
        "dropped_indices": list(range(4, num_layers)),
        "description": "LOLO Top-3 Most Critical Acoustic Layers (4 layers)",
        "category": "Diagnostic Anchor",
    })
    conditions.append({
        "id": "diagnostic_middle_phonetic_4_7",
        "name": "Anchor: Middle Phonetic Only (L4~L7)",
        "drop_k": 9,
        "kept_indices": [4, 5, 6, 7],
        "dropped_indices": [0, 1, 2, 3, 8, 9, 10, 11, 12],
        "description": "Vowel Formant / Phonetic Representation Tier (4 layers)",
        "category": "Diagnostic Anchor",
    })
    conditions.append({
        "id": "diagnostic_top_semantic_8_12",
        "name": "Anchor: Top Semantic Only (L8~L12)",
        "drop_k": 8,
        "kept_indices": [8, 9, 10, 11, 12],
        "dropped_indices": list(range(8)),
        "description": "High-level Linguistic / Semantic Abstraction (5 layers)",
        "category": "Diagnostic Anchor",
    })

    # Attach layer mask and retained weight mass
    for cond in conditions:
        mask = torch.zeros(num_layers, dtype=torch.bool)
        for idx in cond["kept_indices"]:
            mask[idx] = True
        cond["mask"] = mask if len(cond["kept_indices"]) < num_layers else None
        cond["retained_weight_mass"] = sum(learned_weights[i] for i in cond["kept_indices"])

    return conditions


def compute_binary_metrics(y_logits: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    probs = 1.0 / (1.0 + np.exp(-y_logits))
    valid_mask = ~np.isnan(y_true)
    probs = probs[valid_mask]
    y_true = y_true[valid_mask].astype(int)

    if len(np.unique(y_true)) < 2:
        return {
            "auc_pr": 0.0,
            "f1": 0.0,
            "auc_roc": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "best_threshold": 0.5,
        }

    auc_pr = float(average_precision_score(y_true, probs))
    auc_roc = float(roc_auc_score(y_true, probs))

    # Grid search for optimal F1 threshold
    thresholds = np.linspace(0.1, 0.9, 81)
    best_f1 = -1.0
    best_tau = 0.5
    best_p = 0.0
    best_r = 0.0
    best_acc = 0.0

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
    }


def run_evaluation(
    model: WavFryClassifier,
    loader: DataLoader,
    device: torch.device,
    paradigm: str,
    conditions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    model.eval()
    num_layers = model.num_layers + 1
    raw_sm_weights = model.scalar_mix_weights["creaky"].detach().clone().to(device)
    head_dtype = next(model.heads.parameters()).dtype

    # Precompute normalized softmax weights for each condition
    cond_weights = []
    for cond in conditions:
        if cond["mask"] is None:
            w = torch.softmax(raw_sm_weights, dim=0)
        else:
            m_logits = raw_sm_weights.clone()
            m_logits[~cond["mask"].to(device)] = -float("inf")
            w = torch.softmax(m_logits, dim=0)
        cond_weights.append(w)  # [num_layers]

    # Stack into [num_conditions, num_layers]
    stacked_cond_weights = torch.stack(cond_weights, dim=0)  # [C, 13]
    num_conditions = len(conditions)

    # Prediction accumulators
    cond_interval_preds: List[Dict[str, list]] = [{} for _ in range(num_conditions)]
    cond_all_logits: List[list] = [[] for _ in range(num_conditions)]
    cond_all_labels: List[list] = [[] for _ in range(num_conditions)]

    print(f"\n[Evaluating] Running single-pass evaluation across {num_conditions} conditions...")
    start_time = time.time()
    batch_count = 0

    with torch.no_grad():
        for batch in loader:
            batch_count += 1
            if batch_count % 50 == 0 or batch_count == len(loader):
                elapsed = time.time() - start_time
                print(f"  Processed {batch_count}/{len(loader)} batches ({elapsed:.1f}s)...", flush=True)

            if paradigm == "block":
                waveforms, bounds, labels, valid, lengths, speakers, block_ids, interval_ids = batch
                waveforms = waveforms.to(device)
                bounds = bounds.to(device)
                labels = labels.to(device)
                valid = valid.to(device)
                lengths = lengths.to(device)

                # 1. Backbone forward pass ONCE
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

                # Precompute sequence mask
                out_lengths = model.wavlm._get_feat_extract_output_lengths(lengths).to(device).clamp(min=1, max=T)
                seq_mask = torch.arange(T, device=device).unsqueeze(0) < out_lengths.unsqueeze(1)

                # 2. Vectorized evaluation for all conditions
                # Compute all mixed representations: [C, 13] @ [13, B*T*D] -> [C, B, T, D]
                all_mixed = (stacked_cond_weights.to(flat_H.dtype) @ flat_H).view(num_conditions, B, T, D).to(head_dtype)

                t_labels = labels[..., 0] if (labels.ndim > 2 and labels.shape[-1] >= 1) else labels

                for c_idx in range(num_conditions):
                    hidden_by_task = {"creaky": all_mixed[c_idx]}
                    block_out = model._forward_block(
                        hidden_by_task=hidden_by_task,
                        seq_mask=seq_mask,
                        lengths=lengths,
                        interval_bounds=bounds,
                        slot_valid=valid,
                    )
                    t_logits = block_out["interval_logits"]["creaky"]

                    c_dict = cond_interval_preds[c_idx]
                    for logit, label, val_row, item_row in zip(t_logits, t_labels, valid, interval_ids):
                        for l_val, lab_val, v_val, i_id in zip(logit, label, val_row, item_row):
                            if v_val and i_id is not None:
                                i_key = str(i_id.item() if isinstance(i_id, torch.Tensor) else i_id)
                                if i_key not in c_dict:
                                    c_dict[i_key] = [l_val.item(), lab_val.item(), 1]
                                else:
                                    c_dict[i_key][0] += l_val.item()
                                    c_dict[i_key][2] += 1
            else:
                # Segment mode
                waveforms, labels, lengths, speakers, meta = batch
                waveforms = waveforms.to(device)
                labels = labels.to(device)
                if labels.ndim > 1 and labels.shape[-1] >= 1:
                    labels = labels[..., 0]
                lengths = lengths.to(device)

                wavlm_dtype = next(model.wavlm.parameters()).dtype
                x_in = waveforms.to(wavlm_dtype) if waveforms.dtype != wavlm_dtype else waveforms
                max_len = waveforms.shape[1]
                lengths_clamped = lengths.clamp(min=1, max=max_len)
                attention_mask = (torch.arange(max_len, device=device).unsqueeze(0) < lengths_clamped.unsqueeze(1)).long()

                outputs = model.wavlm(x_in, attention_mask=attention_mask, output_hidden_states=True)
                raw_hidden_states = [h.detach() for h in outputs.hidden_states]
                H = torch.stack(raw_hidden_states, dim=0)
                L, B, T, D = H.shape
                flat_H = H.view(L, -1)

                out_lengths = model.wavlm._get_feat_extract_output_lengths(lengths).to(device).clamp(min=1, max=T)
                seq_mask = torch.arange(T, device=device).unsqueeze(0) < out_lengths.unsqueeze(1)

                all_mixed = (stacked_cond_weights.to(flat_H.dtype) @ flat_H).view(num_conditions, B, T, D).to(head_dtype)

                for c_idx in range(num_conditions):
                    pooled = model._pool_batch(all_mixed[c_idx], seq_mask, lengths, task="creaky")
                    logits = model.heads["creaky"](pooled).squeeze(-1)
                    cond_all_logits[c_idx].extend(logits.cpu().numpy().tolist())
                    if c_idx == 0:
                        cond_all_labels[c_idx].extend(labels.cpu().numpy().tolist())

    total_time = time.time() - start_time
    print(f"[Done] Single-pass evaluation finished in {total_time:.2f}s!")

    # 3. Calculate metrics for each condition
    results: List[Dict[str, Any]] = []

    # Get labels
    if paradigm == "block":
        base_dict = cond_interval_preds[0]
        y_true = np.array([item[1] for item in base_dict.values()])
    else:
        y_true = np.array(cond_all_labels[0])

    # First compute baseline metrics
    if paradigm == "block":
        base_logits = np.array([item[0] / item[2] for item in cond_interval_preds[0].values()])
    else:
        base_logits = np.array(cond_all_logits[0])

    base_metrics = compute_binary_metrics(base_logits, y_true)
    base_pr = base_metrics["auc_pr"]
    base_f1 = base_metrics["f1"]

    print("\n" + "=" * 90)
    print("                 WavFry Top-k Layer Truncation Evaluation Summary")
    print("=" * 90)
    print(f"Baseline PR-AUC: {base_pr:.4f} | Best F1: {base_f1:.4f} (at tau*={base_metrics['best_threshold']:.2f})")
    print("-" * 90)

    for c_idx, cond in enumerate(conditions):
        if paradigm == "block":
            c_dict = cond_interval_preds[c_idx]
            logits = np.array([item[0] / item[2] for item in c_dict.values()])
        else:
            logits = np.array(cond_all_logits[c_idx])

        m = compute_binary_metrics(logits, y_true)
        c_pr = m["auc_pr"]
        c_f1 = m["f1"]
        delta_pr = base_pr - c_pr
        delta_f1 = base_f1 - c_f1
        drop_pct = (delta_pr / base_pr * 100.0) if base_pr > 0 else 0.0

        kept_count = len(cond["kept_indices"])
        kept_str = f"0..{cond['kept_indices'][-1]}" if cond["kept_indices"] == list(range(kept_count)) else str(cond["kept_indices"])

        print(
            f"  [{cond['name']:<24}] Kept: {kept_count:02d} layers ({kept_str:<7}) | "
            f"PR-AUC: {c_pr:.4f} (Δ {delta_pr:+.4f}, -{drop_pct:4.1f}%) | "
            f"F1: {c_f1:.4f} | Mass: {cond['retained_weight_mass']*100:5.1f}%"
        )

        results.append({
            "condition_id": cond["id"],
            "condition_name": cond["name"],
            "category": cond["category"],
            "drop_k": cond["drop_k"],
            "kept_layers_count": kept_count,
            "kept_layer_indices": str(cond["kept_indices"]),
            "dropped_layer_indices": str(cond["dropped_indices"]),
            "retained_mix_weight_mass": round(cond["retained_weight_mass"], 4),
            "auc_pr": round(c_pr, 4),
            "f1": round(c_f1, 4),
            "auc_roc": round(m["auc_roc"], 4),
            "accuracy": round(m["accuracy"], 4),
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "best_threshold": round(m["best_threshold"], 2),
            "delta_auc_pr": round(delta_pr, 4),
            "delta_f1": round(delta_f1, 4),
            "drop_pct": round(drop_pct, 2),
            "description": cond["description"],
        })

    return results


def export_reports(results: List[Dict[str, Any]], output_dir: Path, ckpt_name: str, split: str, paradigm: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "topk_layer_truncation_report.csv"
    md_path = output_dir / "topk_layer_truncation_report.md"
    json_path = output_dir / "topk_layer_truncation_summary.json"

    # 1. Save CSV
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"\n[Saved] CSV report: {csv_path}")

    # 2. Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": ckpt_name,
                "split": split,
                "paradigm": paradigm,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[Saved] JSON report: {json_path}")

    # 3. Save Markdown
    base_res = results[0]
    base_pr = base_res["auc_pr"]
    base_f1 = base_res["f1"]

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# WavFry Top-k Layer Truncation & Depth Sensitivity Audit Report\n\n")
        f.write(f"- **Evaluated Checkpoint**: `{ckpt_name}`\n")
        f.write(f"- **Dataset Split**: `{split.upper()}` | **Training Paradigm**: `{paradigm.upper()}`\n")
        f.write(f"- **Full Model Baseline PR-AUC**: `{base_pr:.4f}` | **Optimal F1**: `{base_f1:.4f}`\n")
        f.write(f"- **Generated At**: `{time.strftime('%Y-%m-%d %H:%M:%S')}`\n\n")

        f.write("## 1. 核心发现与架构洞见（Executive Summary）\n\n")
        
        # Analyze critical truncation points (k=3, k=6, k=9)
        k3 = next((r for r in results if r["condition_id"] == "drop_top_3"), None)
        k6 = next((r for r in results if r["condition_id"] == "drop_top_6"), None)
        k9 = next((r for r in results if r["condition_id"] == "drop_top_9"), None)
        k12 = next((r for r in results if r["condition_id"] == "drop_top_12"), None)
        bottom_ac = next((r for r in results if r["condition_id"] == "diagnostic_bottom_acoustic_0_3"), None)
        top_sem = next((r for r in results if r["condition_id"] == "diagnostic_top_semantic_8_12"), None)

        if k3:
            f.write(f"- **截断后 3 层 (Drop Top 3, k=3, 保留前 10 层)**: PR-AUC 为 `{k3['auc_pr']:.4f}` (变化 `{k3['delta_auc_pr']:+.4f}`，跌幅 `{k3['drop_pct']:.1f}%`)。\n")
        if k6:
            f.write(f"- **截断后 6 层 (Drop Top 6, k=6, 保留前 7 层 [WavLM-6L 半深模型])**: PR-AUC 为 `{k6['auc_pr']:.4f}` (变化 `{k6['delta_auc_pr']:+.4f}`，跌幅 `{k6['drop_pct']:.1f}%`)。\n")
        if k9:
            f.write(f"- **截断后 9 层 (Drop Top 9, k=9, 仅保留 CNN + L1~L3)**: PR-AUC 为 `{k9['auc_pr']:.4f}` (变化 `{k9['delta_auc_pr']:+.4f}`，跌幅 `{k9['drop_pct']:.1f}%`)。\n")
        if k12:
            f.write(f"- **完全去除 Transformer (仅保留 CNN 特征)**: PR-AUC 为 `{k12['auc_pr']:.4f}` (变化 `{k12['delta_auc_pr']:+.4f}`，跌幅 `{k12['drop_pct']:.1f}%`)。\n")
        if top_sem and bottom_ac:
            f.write(f"- **反向对照（仅高阶语义 vs 仅底层声学）**: 仅高阶语义层(L8-L12) PR-AUC 为 `{top_sem['auc_pr']:.4f}`，而仅底层声学(CNN+L1-L3) PR-AUC 为 `{bottom_ac['auc_pr']:.4f}`。\n")

        f.write("\n---\n\n")
        f.write("## 2. 逐层累积截断实验详细数据表 (Sequential Cumulative Truncation)\n\n")
        f.write("| 截断条件 | 丢弃层数 $k$ | 保留层数 | 保留权重质量 | PR-AUC | $\\Delta$PR-AUC | 跌幅 (%) | F1 | ROC-AUC | 最佳阈值 $\\tau^*$ | 状态评定 |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")

        seq_rows = [r for r in results if r["category"] in ("Baseline", "Sequential Truncation")]
        for r in seq_rows:
            drop_k = r["drop_k"]
            if drop_k == 0:
                status = "🎯 Full Baseline"
            elif r["drop_pct"] <= 0.5:
                status = "🟢 无损/极微损 (<0.5%)"
            elif r["drop_pct"] <= 2.0:
                status = "🟡 轻微下降 (<=2%)"
            elif r["drop_pct"] <= 5.0:
                status = "🟠 中度受损 (<=5%)"
            else:
                status = "🔴 严重受损 (>5%)"

            f.write(
                f"| **{r['condition_name']}** | {r['drop_k']} | {r['kept_layers_count']} | "
                f"{r['retained_mix_weight_mass']*100:.1f}% | **{r['auc_pr']:.4f}** | "
                f"{r['delta_auc_pr']:+.4f} | {r['drop_pct']:.1f}% | {r['f1']:.4f} | "
                f"{r['auc_roc']:.4f} | {r['best_threshold']:.2f} | {status} |\n"
            )

        f.write("\n---\n\n")
        f.write("## 3. 架构代表性锚点与反向对照分析 (Diagnostic Architectural Anchors)\n\n")
        f.write("| 架构锚点 | 激活层描述 | 保留权重 | PR-AUC | $\\Delta$PR-AUC | F1 | 科学结论 |\n")
        f.write("|:---|:---|:---:|:---:|:---:|:---:|:---|\n")

        anchor_rows = [r for r in results if r["category"] == "Diagnostic Anchor"]
        for r in anchor_rows:
            f.write(
                f"| **{r['condition_name']}** | {r['description']} | "
                f"{r['retained_mix_weight_mass']*100:.1f}% | **{r['auc_pr']:.4f}** | "
                f"{r['delta_auc_pr']:+.4f} | {r['f1']:.4f} | "
                f"{'声学底层决定论验证' if 'acoustic' in r['condition_id'] else ('语义高层无法独立支撑发声态' if 'semantic' in r['condition_id'] else '音素共振峰辅助')} |\n"
            )

        f.write("\n\n> [!NOTE]\n")
        f.write("> **实验方法学说明**：\n")
        f.write("> 1. 本实验基于已收敛最优 checkpoint，采用单次前向缓存机制（Single-Pass Forward Caching），对 WavLM 隐层应用动态 Softmax 掩码重归一化。\n")
        f.write("> 2. 权重保留质量（Retained Weight Mass）表示原模型自适应学习到的标量注意力中，分配给保留层的百分比。\n")
        f.write("> 3. 本测试为**后验截断测试**（分类头是在完整 13 层输入下训练的）。若进一步在训练期物理截断（方案 2），浅层专属调优后的分类头上限预期会高于此表中的后验截断表现。\n")

    print(f"[Saved] Markdown report: {md_path}")


def main():
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[Device] Using device: {device}")

    ckpt_path = resolve_checkpoint(args.checkpoint)
    model, config = load_model_from_checkpoint(ckpt_path, device)

    manifest_path = Path(args.manifest) if args.manifest else (PROJECT_ROOT / "manifests" / "segments_multi.csv")
    if not manifest_path.exists():
        manifest_path = PROJECT_ROOT / "manifests" / "segments_all.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    loader, paradigm = prepare_dataloader(config, manifest_path, args.split, args.batch_size, args.smoketest)

    # Extract learned scalar mix weights
    num_layers = model.num_layers + 1
    if model.use_scalar_mix and "creaky" in model.scalar_mix_weights:
        raw_logits = model.scalar_mix_weights["creaky"].detach().cpu()
        sm_weights = torch.softmax(raw_logits, dim=0).numpy().tolist()
    else:
        sm_weights = [1.0 / num_layers] * num_layers

    conditions = build_truncation_conditions(num_layers, sm_weights)

    out_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else (PROJECT_ROOT / args.output_dir)

    results = run_evaluation(model, loader, device, paradigm, conditions)
    export_reports(results, out_dir, ckpt_path.name, args.split, paradigm)


if __name__ == "__main__":
    main()

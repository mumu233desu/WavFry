#!/usr/bin/env python
"""
WavFry Functional Auditing: Leave-One-Layer-Out (LOLO) Layer Ablation Evaluator.
Evaluates the internal necessity and decision sensitivity of each Transformer layer
in frozen SSL backbones (WavLM, HuBERT, wav2vec 2.0) using dynamic Softmax re-normalization.

Mathematical Formulation:
    h_{-l} = sum_{k != l} exp(w_k) * h_k / sum_{k != l} exp(w_k)
Implemented via logit masking (masked_logits[l] = -inf) prior to Softmax,
strictly preserving feature scale and preventing norm-shrinkage artifacts.

Outputs:
    - data/experiments/lolo_ablation_results.csv
    - Markdown formatted summary table with layer indispensability rankings.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

# Add project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wavfry.data import dataset as dataset_mod
from wavfry.data import splits as splits_mod
from wavfry.model.wavlm_classifier import WavFryClassifier
from wavfry.train import metrics as metrics_mod


def parse_args():
    parser = argparse.ArgumentParser(description="WavFry LOLO Layer Ablation Evaluator")
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
        help="Device to use ('cuda' or 'cpu'). Defaults to cuda if available.",
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
        help="Run fast smoke test with a small sample slice.",
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
        Path("R:/checkpoints/csl_frozen_wavlm_base_plus/best_model.pt"),
        Path("R:/checkpoints/frozen_wavlm_ablation/single_attn/best_model.pt"),
        PROJECT_ROOT / "checkpoints/csl_frozen_wavlm_base_plus/best_model.pt",
        PROJECT_ROOT / "checkpoints/frozen_wavlm_base_plus/best_model.pt",
        PROJECT_ROOT / "exp_results/wavfry_exp_20260803_frame-level/checkpoints/full_ablation/block_linear_softmax/best_model.pt",
    ]
    for c in candidates:
        if c.exists():
            print(f"[Checkpoint] Auto-resolved to existing checkpoint: {c}")
            return c

    raise FileNotFoundError(
        "No default checkpoint found. Please provide --checkpoint path to a trained model."
    )


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
    model.is_backbone_frozen = True
    return model, config


def prepare_dataloader(
    config: Dict[str, Any],
    manifest_path: Path,
    split_target: str,
    batch_size: int,
    is_smoketest: bool = False,
) -> tuple[DataLoader, str]:
    df = pd.read_csv(manifest_path, dtype={"speaker_id": str, "phone": str, "file_key": str})

    val_speakers = config.get("data", {}).get("val_speakers", ["sampled"])
    isolate = bool(config.get("data", {}).get("isolate_split_speakers", True))

    if val_speakers in ("batch", "batch_split", ["batch"], ["batch_split"]) or (
        isinstance(val_speakers, list) and "batch2" in val_speakers
    ):
        train_df, val_df, test_df = splits_mod.make_batch_split(df, isolate=isolate)
    else:
        train_df, val_df = splits_mod.make_speaker_split(df, val_speakers, isolate=isolate)
        test_df = df.iloc[0:0].copy()

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
        eval_df = pd.concat([pos.sample(n=n_pos, random_state=42), neg.sample(n=n_neg, random_state=42)]).sample(frac=1.0, random_state=42).reset_index(drop=True)
        print(f"[Smoketest] Sliced evaluation dataset to {len(eval_df)} segments.")

    paradigm = config.get("training", {}).get("paradigm", "block")
    lowpass_cfg = config.get("preprocessing", {}).get("lowpass", {})

    if paradigm == "block":
        from wavfry.data.paradigm_block import ContinuousAudioBlockDataset, collate_block_fn
        block_cfg = config.get("training", {}).get("block", {})
        ds = ContinuousAudioBlockDataset(
            eval_df,
            duration_ms=block_cfg.get("duration_ms", 2000),
            hop_ms=block_cfg.get("hop_ms", 1000),
            max_intervals=block_cfg.get("max_intervals", 8),
            boundary_trim_frames=block_cfg.get("boundary_trim_frames", 0),
            lowpass_cfg=lowpass_cfg,
            seed=42,
            fold_index=0,
        )
        collate = collate_block_fn
    else:
        ds = dataset_mod.CreakyVowelDataset(
            eval_df,
            context_padding_ms=config.get("data", {}).get("context_padding_ms", 30),
            lowpass_cfg=lowpass_cfg,
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


def evaluate_with_mask(
    model: WavFryClassifier,
    loader: DataLoader,
    device: torch.device,
    paradigm: str,
    layer_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Run forward evaluation over the dataloader with a given layer_mask."""
    model.eval()
    all_logits = []
    all_labels = []

    interval_predictions: Dict[str, list] = {}

    with torch.no_grad():
        for batch in loader:
            if paradigm == "block":
                waveforms, bounds, labels, valid, lengths, speakers, block_ids, interval_ids = batch
                waveforms = waveforms.to(device)
                bounds = bounds.to(device)
                labels = labels.to(device)
                valid = valid.to(device)
                lengths = lengths.to(device)

                out = model(
                    waveforms,
                    lengths=lengths,
                    interval_bounds=bounds,
                    interval_valid=valid,
                    layer_mask=layer_mask,
                )
                t_logits = out["interval_logits"]["creaky"]
                t_labels = labels

                for logit, label, val_row, item_row in zip(t_logits, t_labels, valid, interval_ids):
                    for l_val, lab_val, v_val, i_id in zip(logit, label, val_row, item_row):
                        if v_val and i_id is not None:
                            i_key = str(i_id.item() if isinstance(i_id, torch.Tensor) else i_id)
                            if i_key not in interval_predictions:
                                interval_predictions[i_key] = [l_val.item(), lab_val.item(), 1]
                            else:
                                interval_predictions[i_key][0] += l_val.item()
                                interval_predictions[i_key][2] += 1
            else:
                waveforms, labels, lengths, speakers, meta = batch
                waveforms = waveforms.to(device)
                labels = labels.to(device)
                lengths = lengths.to(device)
                out = model(waveforms, lengths=lengths, layer_mask=layer_mask)
                logits = out["creaky"] if isinstance(out, dict) else out
                all_logits.extend(logits.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

    if paradigm == "block":
        pred_logits = [item[0] / item[2] for item in interval_predictions.values()]
        pred_labels = [item[1] for item in interval_predictions.values()]
    else:
        pred_logits = all_logits
        pred_labels = all_labels

    t_logits = torch.tensor(pred_logits, dtype=torch.float32)
    t_labels = torch.tensor(pred_labels, dtype=torch.float32)
    valid_m = ~torch.isnan(t_labels)

    metrics = metrics_mod.compute_metrics(t_logits[valid_m], t_labels[valid_m])
    return {
        "auc_pr": float(metrics.get("auc_pr", 0.0)),
        "f1": float(metrics.get("f1", 0.0)),
        "auc_roc": float(metrics.get("auc_roc", 0.0)),
        "accuracy": float(metrics.get("accuracy", 0.0)),
        "precision": float(metrics.get("precision", 0.0)),
        "recall": float(metrics.get("recall", 0.0)),
    }


def main():
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[Device] Using device: {device}")

    ckpt_path = resolve_checkpoint(args.checkpoint)
    model, config = load_model_from_checkpoint(ckpt_path, device)

    # Manifest resolution
    manifest_path = Path(args.manifest) if args.manifest else (PROJECT_ROOT / "manifests" / "segments_multi.csv")
    if not manifest_path.exists():
        manifest_path = PROJECT_ROOT / "manifests" / "segments_all.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    loader, paradigm = prepare_dataloader(config, manifest_path, args.split, args.batch_size, args.smoketest)

    # Extract learned scalar mix weights for reference
    num_layers = 13
    learned_weights = []
    if model.use_scalar_mix and "creaky" in model.scalar_mix_weights:
        raw_logits = model.scalar_mix_weights["creaky"].detach().cpu()
        sm_weights = torch.softmax(raw_logits, dim=0).numpy()
        learned_weights = sm_weights.tolist()
    else:
        learned_weights = [1.0 / num_layers] * num_layers

    print("\n" + "=" * 78)
    print("      WavFry LOLO (Leave-One-Layer-Out) Ablation Evaluator")
    print(f"  Checkpoint: {ckpt_path.name}")
    print(f"  Split: {args.split.upper()} | Paradigm: {paradigm.upper()} | Batches: {len(loader)}")
    print(f"  Dynamic Softmax Re-normalization: ENABLED")
    print("=" * 78)

    results: List[Dict[str, Any]] = []

    # 1. Full Model Baseline (No ablation)
    print("\n[Evaluating] Baseline: Full Model (All 13 Layers Active)...")
    base_metrics = evaluate_with_mask(model, loader, device, paradigm, layer_mask=None)
    base_auc_pr = base_metrics["auc_pr"]
    base_f1 = base_metrics["f1"]
    print(f"  >> Full Baseline | PR-AUC: {base_auc_pr:.4f} | F1: {base_f1:.4f} | Acc: {base_metrics['accuracy']:.4f}")

    results.append({
        "condition": "full_model",
        "ablated_layer": "None",
        "relative_depth": "1.0",
        "scalar_mix_weight": 1.0,
        "auc_pr": base_auc_pr,
        "f1": base_f1,
        "auc_roc": base_metrics["auc_roc"],
        "delta_auc_pr": 0.0,
        "delta_f1": 0.0,
        "drop_pct": 0.0,
    })

    # 2. Leave-One-Layer-Out for Layers 0 to 12
    print("\n[Evaluating] Leave-One-Layer-Out (LOLO) across 13 layers:")
    for l in range(num_layers):
        mask = torch.ones(num_layers, dtype=torch.bool)
        mask[l] = False

        m_metrics = evaluate_with_mask(model, loader, device, paradigm, layer_mask=mask)
        m_auc_pr = m_metrics["auc_pr"]
        m_f1 = m_metrics["f1"]
        delta_pr = base_auc_pr - m_auc_pr
        delta_f1 = base_f1 - m_f1
        drop_pct = (delta_pr / base_auc_pr * 100.0) if base_auc_pr > 0 else 0.0

        layer_name = "CNN_Feat" if l == 0 else f"Transformer_L{l}"
        rel_depth = f"{l / (num_layers - 1):.2f}"
        weight = learned_weights[l] if l < len(learned_weights) else 0.0

        print(
            f"  [LOLO Layer {l:02d} ({layer_name:>14})] "
            f"PR-AUC: {m_auc_pr:.4f} (Δ {delta_pr:+.4f}, -{drop_pct:.1f}%) | "
            f"F1: {m_f1:.4f} | MixWeight: {weight * 100:.2f}%"
        )

        results.append({
            "condition": f"leave_out_layer_{l}",
            "ablated_layer": layer_name,
            "relative_depth": rel_depth,
            "scalar_mix_weight": round(weight, 5),
            "auc_pr": round(m_auc_pr, 4),
            "f1": round(m_f1, 4),
            "auc_roc": round(m_metrics["auc_roc"], 4),
            "delta_auc_pr": round(delta_pr, 4),
            "delta_f1": round(delta_f1, 4),
            "drop_pct": round(drop_pct, 2),
        })

    # 3. Optional Grouped Tiers
    if args.group_tiers:
        print("\n[Evaluating] Grouped Tier Ablations:")
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

            t_metrics = evaluate_with_mask(model, loader, device, paradigm, layer_mask=mask)
            t_auc_pr = t_metrics["auc_pr"]
            t_f1 = t_metrics["f1"]
            delta_pr = base_auc_pr - t_auc_pr
            drop_pct = (delta_pr / base_auc_pr * 100.0) if base_auc_pr > 0 else 0.0

            print(f"  [{tier_name:>22}] PR-AUC: {t_auc_pr:.4f} (Δ {delta_pr:+.4f}) | F1: {t_f1:.4f}")

            results.append({
                "condition": tier_id,
                "ablated_layer": tier_name,
                "relative_depth": "Group",
                "scalar_mix_weight": round(sum(learned_weights[i] for i in active_indices), 5),
                "auc_pr": round(t_auc_pr, 4),
                "f1": round(t_f1, 4),
                "auc_roc": round(t_metrics["auc_roc"], 4),
                "delta_auc_pr": round(delta_pr, 4),
                "delta_f1": round(delta_f1, 4),
                "drop_pct": round(drop_pct, 2),
            })

    # Save to CSV
    out_path = Path(args.output) if Path(args.output).is_absolute() else (PROJECT_ROOT / args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res_df = pd.DataFrame(results)
    res_df.to_csv(out_path, index=False)
    print(f"\n[Saved] LOLO Ablation table saved to: {out_path}")

    # Generate Markdown Summary
    md_path = out_path.with_suffix(".md")
    lolo_rows = [r for r in results if r["condition"].startswith("leave_out_layer_")]
    lolo_sorted = sorted(lolo_rows, key=lambda x: x["delta_auc_pr"], reverse=True)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# WavFry Leave-One-Layer-Out (LOLO) Functional Audit\n\n")
        f.write(f"- **Checkpoint**: `{ckpt_path.name}`\n")
        f.write(f"- **Split**: `{args.split}` | **Paradigm**: `{paradigm}`\n")
        f.write(f"- **Full Model Baseline PR-AUC**: `{base_auc_pr:.4f}` | **F1**: `{base_f1:.4f}`\n\n")
        f.write("### Layer Indispensability Ranking (Sorted by Decision Sensitivity ΔPR-AUC)\n\n")
        f.write("| Rank | Layer Name | Rel Depth | Mix Weight | Ablated PR-AUC | ΔPR-AUC | Drop (%) | Necessity Level |\n")
        f.write("|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|\n")

        for rank, row in enumerate(lolo_sorted, 1):
            nec = "🔥🔥 Critical" if rank <= 3 else ("⚠️ Moderate" if rank <= 7 else "🌱 Marginal")
            f.write(
                f"| {rank} | **{row['ablated_layer']}** | {row['relative_depth']} | "
                f"{row['scalar_mix_weight']*100:.2f}% | {row['auc_pr']:.4f} | "
                f"**+{row['delta_auc_pr']:.4f}** | -{row['drop_pct']:.1f}% | {nec} |\n"
            )
        f.write("\n\n*Note: High ΔPR-AUC indicates that the model decision is highly sensitive to the removal of this layer (functional necessity).*\n")

    print(f"[Saved] LOLO Markdown summary saved to: {md_path}")


if __name__ == "__main__":
    main()

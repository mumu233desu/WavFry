#!/usr/bin/env python
"""Canonical runner for Multi-Backbone Comparison Ablation (WavLM vs HuBERT vs wav2vec 2.0).

This script serves as the SINGLE CANONICAL ENTRY POINT for comparing the three major
self-supervised speech backbones under frozen parameter settings:
- WavLM-Base+ (microsoft/wavlm-base-plus)
- HuBERT-Base (facebook/hubert-base-ls960)
- wav2vec 2.0-Base (facebook/wav2vec2-base-960h)

Key Principles (strictly adhering to AGENTS.md):
1. Inviolable Rule: Backbones are 100% frozen (backbone_lr: 0.0, fine_tune_layers: "none").
   Only scalar mix weights (13 layers) and task heads are tuned.
2. Flushed I/O: All progress messages and evaluation tables are explicitly flushed.
3. Observability: Emits transparent progress milestones across training and evaluation.
4. Academic Rigor: Evaluates on both Clean and Degraded acoustic tiers, outputting
   direct LaTeX/Markdown tables for journal publication (Table 5).
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wavfry.data.manifest import build_manifest, build_manifest_multi
from wavfry.model.wavlm_classifier import WavFryClassifier

logger = logging.getLogger("backbone_comparison")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BACKBONE_REGISTRY = {
    "wavlm": {
        "display_name": "WavLM-Base (960h)",
        "hf_model_id": "microsoft/wavlm-base",
        "config_file": ROOT / "config" / "ablation_backbone_wavlm_base.yaml",
        "default_ckpt": ROOT / "exp_results" / "checkpoints" / "ablation_backbone_wavlm_base" / "best_model.pt",
    },
    "hubert": {
        "display_name": "HuBERT-Base (960h)",
        "hf_model_id": "facebook/hubert-base-ls960",
        "config_file": ROOT / "config" / "ablation_backbone_hubert.yaml",
        "default_ckpt": ROOT / "exp_results" / "checkpoints" / "ablation_backbone_hubert" / "best_model.pt",
    },
    "wav2vec2": {
        "display_name": "wav2vec 2.0-Base (960h)",
        "hf_model_id": "facebook/wav2vec2-base",
        "config_file": ROOT / "config" / "ablation_backbone_wav2vec2.yaml",
        "default_ckpt": ROOT / "exp_results" / "checkpoints" / "ablation_backbone_wav2vec2" / "best_model.pt",
    },
    "wavlm_plus": {
        "display_name": "WavLM-Base+ (60k h)",
        "hf_model_id": "microsoft/wavlm-base-plus",
        "config_file": ROOT / "config" / "ablation_backbone_wavlm_baseplus.yaml",
        "default_ckpt": ROOT / "exp_results" / "checkpoints" / "ablation_backbone_wavlm_baseplus" / "best_model.pt",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-backbone comparison ablation study for WavLM, HuBERT, and wav2vec 2.0."
    )
    parser.add_argument(
        "--backbones",
        nargs="+",
        choices=["wavlm", "hubert", "wav2vec2", "wavlm_plus"],
        default=["wavlm", "hubert", "wav2vec2"],
        help="List of backbones to evaluate (default: the three 960h baselines).",
    )
    parser.add_argument(
        "--mode",
        choices=["train_and_eval", "eval_only", "train_only"],
        default="train_and_eval",
        help="Execution mode (default: train_and_eval).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training epochs (default: inherit from config, usually 20).",
    )
    parser.add_argument(
        "--smoketest",
        action="store_true",
        help="Run a fast end-to-end smoke test (1 short epoch on a representative subset of audio files).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (default: cuda if available).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "exp_results" / "backbone_comparison",
        help="Target directory for comparative results and summary tables.",
    )
    parser.add_argument(
        "--eval-conditions",
        nargs="+",
        default=None,
        help="Acoustic evaluation conditions (default: all 18 canonical conditions; pass 'all' or specific conditions).",
    )
    return parser.parse_args()


def load_config_for_backbone(
    backbone_key: str,
    epochs_override: int | None = None,
    smoketest: bool = False,
) -> dict[str, Any]:
    meta = BACKBONE_REGISTRY[backbone_key]
    cfg_path = meta["config_file"]
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if epochs_override is not None:
        cfg["training"]["epochs"] = epochs_override
    elif smoketest:
        cfg["training"]["epochs"] = 1

    if smoketest:
        cfg["training"]["warmup_steps"] = 2
        cfg["eval"]["eval_interval"] = 1
        cfg["training"]["block"]["batch_size"] = 16
        cfg["training"]["block"]["val_batch_size"] = 16
        cfg["checkpoint"]["dir"] = f"exp_results/checkpoints/smoketest_backbone_{backbone_key}"
        cfg["tensorboard"]["log_dir"] = f"exp_results/runs/smoketest_backbone_{backbone_key}"

    return cfg


def create_smoketest_manifest_subset(df: Any) -> Any:
    """Extract a small, balanced subset of audio files for fast smoke testing (<5s execution).

    Guarantees:
    - Speaker isolation between train/val/test
    - Both positive and negative labels for each task
    - ~1-2 blocks per split for instantaneous pipeline verification (<5s per backbone)
    """
    import pandas as pd

    subsets = []
    for split in ["train", "val", "test"]:
        sdf = df[df["batch_split"] == split]
        chosen: list[str] = []
        # CS-Dialogue files with positive and negative creaky labels
        cs_df = sdf[sdf["dataset_id"] == "cs_dialogue"]
        for p, g in cs_df.groupby("audio_path"):
            if (g["label_creaky"] > 0.5).any() and (g["label_creaky"] <= 0.5).any():
                chosen.append(p)
                break
        if not chosen and len(cs_df) > 0:
            chosen.append(cs_df["audio_path"].iloc[0])

        # Breathy dataset files with positive and negative breathy labels
        br_df = sdf[sdf["dataset_id"].isin(["breathy_tts", "ysj_japanese", "ita_japanese"])]
        for p, g in br_df.groupby("audio_path"):
            if (g["label_breathy"] > 0.5).any() and (g["label_breathy"] <= 0.5).any():
                chosen.append(p)
                break
        if len(chosen) < 2 and len(br_df) > 0:
            chosen.append(br_df["audio_path"].iloc[0])

        split_sub = sdf[sdf["audio_path"].isin(chosen)]
        # Cap to at most 10 intervals per split so ContinuousAudioBlockDataset creates ~1 block (1 batch)
        if len(split_sub) > 10:
            pos = split_sub[(split_sub["label_creaky"] > 0.5) | (split_sub["label_breathy"] > 0.5)]
            neg = split_sub[(split_sub["label_creaky"] <= 0.5) & (split_sub["label_breathy"] <= 0.5)]
            split_sub = pd.concat([pos.iloc[:5], neg.iloc[:5]], ignore_index=True)
            if len(split_sub) == 0:
                split_sub = sdf.iloc[:10]

        subsets.append(split_sub)

    subset_df = pd.concat(subsets, ignore_index=True)
    print(
        f"[Smoketest] Subset manifest created: {len(subset_df)} total intervals across "
        f"{subset_df['audio_path'].nunique()} audio files ({subset_df['speaker_id'].nunique()} speakers).",
        flush=True,
    )
    return subset_df


def train_single_backbone(
    backbone_key: str,
    cfg: dict[str, Any],
    df: Any,
    device: str,
) -> Path:
    meta = BACKBONE_REGISTRY[backbone_key]
    display = meta["display_name"]
    print(f"\n{'='*70}", flush=True)
    print(f"[Backbone Comparison] Starting training for: {display} ({meta['hf_model_id']})", flush=True)
    print(f"[Backbone Comparison] Epochs: {cfg['training']['epochs']} | Device: {device}", flush=True)
    print(f"{'='*70}\n", flush=True)

    # Import train runner dynamically from scripts/train.py
    from scripts import train as train_module

    # Pre-flight check: ensure checkpoint directory exists
    ckpt_dir = Path(cfg["checkpoint"]["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Dynamically import required modules
    from wavfry.data import dataset as dataset_mod
    from wavfry.data import splits as splits_mod
    from wavfry.model import wavlm_classifier as model_mod
    from wavfry.train import trainer as trainer_mod

    start_time = time.time()
    train_module.run_training(
        config=cfg,
        df=df,
        device=device,
        dataset_mod=dataset_mod,
        splits_mod=splits_mod,
        model_mod=model_mod,
        trainer_mod=trainer_mod,
        fold_idx=0,
    )
    elapsed = time.time() - start_time
    print(f"[Backbone Comparison] Training finished for {display} in {elapsed / 60:.2f} mins.", flush=True)

    best_ckpt = ckpt_dir / "best_model.pt"
    if not best_ckpt.exists():
        latest_ckpt = ckpt_dir / "latest.pt"
        if latest_ckpt.exists():
            best_ckpt = latest_ckpt
    return best_ckpt


CANONICAL_TASKS = ("creaky", "breathy")


def _extract_task_target(labels: torch.Tensor, task: str, tasks: list[str]) -> torch.Tensor:
    """Extract 1D/2D targets for specific task from collated labels tensor."""
    if labels.ndim > 1 and labels.shape[-1] == len(CANONICAL_TASKS):
        if task in CANONICAL_TASKS:
            return labels[..., CANONICAL_TASKS.index(task)]
    if labels.ndim > 1 and labels.shape[-1] == len(tasks):
        if task in tasks:
            return labels[..., tasks.index(task)]
    if labels.ndim > 1 and labels.shape[-1] == 1:
        return labels.squeeze(-1)
    return labels


def _merge_interval_prediction(
    store: dict[str, list],
    interval_id: str,
    logit: torch.Tensor,
    label: torch.Tensor,
    speaker: str | None = None,
) -> None:
    """Accumulate block predictions into the per-interval store."""
    logit_cpu = logit if logit.device.type == "cpu" else logit.detach().cpu()
    label_cpu = label if label.device.type == "cpu" else label.detach().cpu()
    entry = store.get(interval_id)
    if entry is None:
        store[interval_id] = [logit_cpu, label_cpu, 1, speaker]
        return
    entry[0] += logit_cpu
    entry[2] += 1
    # If first seen label was NaN, replace with valid label if observed
    if torch.isnan(entry[1]) and not torch.isnan(label_cpu):
        entry[1] = label_cpu


def evaluate_backbone(
    backbone_key: str,
    cfg: dict[str, Any],
    ckpt_path: Path,
    test_df: Any,
    device: str,
    conditions: list[str] | None = None,
) -> dict[str, Any]:
    meta = BACKBONE_REGISTRY[backbone_key]
    display = meta["display_name"]
    eval_conditions = list(conditions or ["clean", "cafe_15db", "reverb"])
    if "clean" not in eval_conditions:
        eval_conditions.insert(0, "clean")

    print(f"\n[Backbone Comparison] Evaluating: {display} using checkpoint {ckpt_path}...", flush=True)
    print(f"  Conditions: {', '.join(eval_conditions)}", flush=True)

    # Instantiate model with exact backbone
    model = WavFryClassifier(cfg).to(device)
    calibrated_threshold = 0.5
    if ckpt_path.exists():
        print(f"[Backbone Comparison] Loading checkpoint weights from {ckpt_path}...", flush=True)
        ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)
        calibrated_threshold = float(ckpt_data.get("best_threshold", 0.5))
        state_dict = ckpt_data.get("model_state_dict", ckpt_data)
        model.load_state_dict(state_dict, strict=False)
        print(f"[Backbone Comparison] Loaded checkpoint (validation-calibrated threshold: {calibrated_threshold:.3f})", flush=True)
    else:
        print(f"[Warning] Checkpoint not found at {ckpt_path}; evaluating raw initialized heads.", flush=True)

    model.eval()
    print(f"[Data] Test set loaded: {len(test_df)} intervals across {test_df['speaker_id'].nunique()} speakers.", flush=True)

    from wavfry.data.paradigm_block import ContinuousAudioBlockDataset, collate_block_fn
    from torch.utils.data import DataLoader
    from wavfry.train.metrics import compute_metrics
    from wavfry.eval.acoustic_degradation import get_acoustic_degradation_engine

    degradation_engine = get_acoustic_degradation_engine()

    block_cfg = cfg["training"]["block"]
    common_block = dict(
        duration_ms=block_cfg.get("duration_ms", 2000),
        hop_ms=block_cfg.get("hop_ms", 1000),
        max_intervals=block_cfg.get("max_intervals", 8),
        boundary_trim_frames=block_cfg.get("boundary_trim_frames", 0),
        seed=cfg["training"].get("seed", 42),
    )
    test_ds = ContinuousAudioBlockDataset(
        df=test_df,
        reverb_augmentor=None,
        **common_block,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=block_cfg.get("val_batch_size", 128),
        shuffle=False,
        collate_fn=collate_block_fn,
    )

    tasks = cfg.get("model", {}).get("tasks", ["creaky"])
    use_amp = bool(cfg.get("training", {}).get("use_amp", False)) and ("cuda" in str(device).lower())
    amp_dtype = torch.bfloat16 if cfg.get("training", {}).get("amp_dtype") == "bfloat16" else torch.float16

    results_by_condition: dict[str, dict[str, float]] = {}
    ckpt_dir = Path(cfg["checkpoint"]["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    total_batches = len(test_loader)

    for cond in eval_conditions:
        start_cond = time.time()
        print(f"  [Condition: {cond}] Running vectorized inference ({total_batches} batches)...", flush=True)
        interval_predictions: dict[str, dict[str, list]] = {task: {} for task in tasks}

        with torch.inference_mode():
            for batch_idx, batch in enumerate(test_loader):
                waveforms, bounds, labels, valid, lengths, speakers, block_ids, interval_ids = batch
                if waveforms.numel() == 0:
                    continue

                # Apply standardized in-memory acoustic degradation
                if cond != "clean":
                    waveforms = degradation_engine.apply_tensor(waveforms, cond, seed=42 + batch_idx * 1009)

                waveforms = waveforms.to(device, non_blocking=True)
                bounds = bounds.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                valid = valid.to(device, non_blocking=True)
                lengths = lengths.to(device, non_blocking=True)

                device_type = "cuda" if "cuda" in str(device).lower() else "cpu"
                with torch.amp.autocast(device_type=device_type, enabled=use_amp, dtype=amp_dtype):
                    outputs = model(waveforms, lengths, interval_bounds=bounds, interval_valid=valid)

                effective_valid = outputs.get("interval_valid", valid)
                raw_interval_logits = outputs.get("interval_logits", {})
                if not isinstance(raw_interval_logits, dict):
                    raw_interval_logits = {tasks[0]: raw_interval_logits}

                valid_cpu = effective_valid.detach().cpu().numpy()
                flat_ids: list[str] = []
                flat_speakers: list[str] = []
                for b in range(valid_cpu.shape[0]):
                    for s in range(valid_cpu.shape[1]):
                        if valid_cpu[b, s]:
                            flat_ids.append(interval_ids[b][s])
                            flat_speakers.append(speakers[b])

                for task in tasks:
                    if task not in raw_interval_logits:
                        continue
                    task_logits = raw_interval_logits[task][effective_valid].detach().cpu()
                    task_targets = _extract_task_target(labels, task, tasks)[effective_valid].detach().cpu()
                    for logit, label, interval_id, spk in zip(task_logits, task_targets, flat_ids, flat_speakers):
                        _merge_interval_prediction(interval_predictions[task], interval_id, logit, label, spk)

                if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == total_batches:
                    elapsed_cond = time.time() - start_cond
                    pct = (batch_idx + 1) / total_batches * 100
                    print(f"    [{cond}] Processed {batch_idx + 1}/{total_batches} batches ({pct:.1f}%) in {elapsed_cond:.1f}s", flush=True)

        # Aggregate interval predictions and compute metrics
        cond_metrics: dict[str, float] = {}
        cond_predictions: dict[str, Any] = {}

        for task in tasks:
            t_dict = interval_predictions[task]
            if not t_dict:
                continue
            sorted_iids = sorted(t_dict.keys())
            task_logits_all = torch.stack([(t_dict[iid][0] / t_dict[iid][2]).reshape(()) for iid in sorted_iids])
            task_labels_all = torch.stack([t_dict[iid][1].reshape(()) for iid in sorted_iids])
            sorted_speakers = [t_dict[iid][3] for iid in sorted_iids]

            valid_m = ~torch.isnan(task_labels_all)
            if valid_m.sum() > 0:
                m_cal = compute_metrics(task_logits_all[valid_m], task_labels_all[valid_m], threshold=calibrated_threshold)
                m_def = compute_metrics(task_logits_all[valid_m], task_labels_all[valid_m], threshold=0.5)
                cond_metrics[f"auc_pr_{task}"] = float(m_cal.get("auc_pr", 0.0))
                cond_metrics[f"roc_auc_{task}"] = float(m_cal.get("roc_auc", 0.0))
                cond_metrics[f"f1_{task}_calibrated"] = float(m_cal.get("f1", 0.0))
                cond_metrics[f"f1_{task}_default"] = float(m_def.get("f1", 0.0))
                cond_metrics[f"precision_{task}"] = float(m_cal.get("precision", 0.0))
                cond_metrics[f"recall_{task}"] = float(m_cal.get("recall", 0.0))

                cond_predictions[task] = {
                    "interval_ids": [iid for iid, v in zip(sorted_iids, valid_m) if v],
                    "speakers": [spk for spk, v in zip(sorted_speakers, valid_m) if v],
                    "logits": task_logits_all[valid_m],
                    "labels": task_labels_all[valid_m],
                }
            else:
                cond_metrics[f"auc_pr_{task}"] = 0.0
                cond_metrics[f"f1_{task}_calibrated"] = 0.0
                cond_metrics[f"f1_{task}_default"] = 0.0

        results_by_condition[cond] = cond_metrics

        # Persist prediction tensors for statistical bootstrap audit (AGENTS.md Rule 3)
        pred_dump_path = ckpt_dir / f"test_predictions_{backbone_key}_{cond}.pt"
        torch.save({
            "backbone": backbone_key,
            "display_name": display,
            "condition": cond,
            "calibrated_threshold": calibrated_threshold,
            "metrics": cond_metrics,
            "predictions": cond_predictions,
            "logits": {task: cond_predictions[task]["logits"] for task in cond_predictions},
            "labels": {task: cond_predictions[task]["labels"] for task in cond_predictions},
        }, pred_dump_path)

        if cond == "clean":
            default_clean_dump = ckpt_dir / f"test_predictions_{backbone_key}.pt"
            torch.save({
                "backbone": backbone_key,
                "display_name": display,
                "checkpoint": str(ckpt_path),
                "calibrated_threshold": calibrated_threshold,
                "metrics": cond_metrics,
                "predictions": cond_predictions,
                "logits": {task: cond_predictions[task]["logits"] for task in cond_predictions},
                "labels": {task: cond_predictions[task]["labels"] for task in cond_predictions},
            }, default_clean_dump)

        c_auc = cond_metrics.get("auc_pr_creaky", 0.0)
        c_f1 = cond_metrics.get("f1_creaky_calibrated", 0.0)
        b_auc = cond_metrics.get("auc_pr_breathy", 0.0)
        b_f1 = cond_metrics.get("f1_breathy_calibrated", 0.0)
        print(f"  [{cond}] Creaky PR: {c_auc:.4f} | Creaky F1: {c_f1:.4f} | Breathy PR: {b_auc:.4f} | Breathy F1: {b_f1:.4f}", flush=True)

    clean_metrics = copy.deepcopy(results_by_condition.get("clean", {}))
    clean_metrics["calibrated_threshold"] = calibrated_threshold

    return {
        "backbone": backbone_key,
        "display_name": display,
        "checkpoint": str(ckpt_path),
        "conditions": results_by_condition,
        "metrics": clean_metrics,
    }


def _format_cond_header(cond: str) -> str:
    """Format condition slug into academic column title, e.g. cafe_15db -> Cafe 15dB."""
    parts = cond.split("_")
    items = []
    for p in parts:
        if p.endswith("db"):
            items.append(p[:-2].upper() + "dB")
        else:
            items.append(p.capitalize())
    return " ".join(items)


def format_academic_summary_table(
    results: list[dict[str, Any]],
    conditions: list[str] | None = None,
) -> str:
    if not results:
        return "No results available."

    if conditions is None:
        first_conds = results[0].get("conditions", {})
        conditions = list(first_conds.keys()) if first_conds else ["clean"]

    degraded_conds = [c for c in conditions if c != "clean"]

    # Table headers
    headers = [
        "Self-Supervised Backbone",
        "Parameters",
        "Pretraining Task",
        "Clean Creaky PR",
    ]
    aligns = [":---", ":---:", ":---", ":---:"]

    for dc in degraded_conds:
        headers.append(f"{_format_cond_header(dc)} PR (Δ)")
        aligns.append(":---:")

    headers.append("Creaky F1 (Cal.)")
    aligns.append(":---:")

    has_breathy = any(
        res.get("conditions", {}).get("clean", {}).get("auc_pr_breathy", 0.0) > 0.0
        or res.get("metrics", {}).get("auc_pr_breathy", 0.0) > 0.0
        for res in results
    )

    if has_breathy:
        headers.extend(["Breathy PR", "Breathy F1 (Cal.)"])
        aligns.extend([":---:", ":---:"])

    lines = [
        "\n### Table 5: Multi-Backbone Comparison Ablation (Frozen SSL Backbones)",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(aligns) + " |",
    ]

    backbone_meta = {
        "wavlm": ("94.7M", "Masked Speech Denoising & Prediction (960h LibriSpeech)"),
        "hubert": ("94.7M", "Masked Discrete Cluster Prediction (960h LibriSpeech)"),
        "wav2vec2": ("94.4M", "Contrastive Learning & Quantization (960h LibriSpeech)"),
        "wavlm_plus": ("94.7M", "Masked Speech Denoising & Prediction (60k hr Multi-Corpus)"),
    }

    # Identify best score per column for bolding
    best_clean_pr = max((res.get("conditions", {}).get("clean", {}).get("auc_pr_creaky", 0.0) for res in results), default=0.0)
    best_deg_prs = {
        dc: max((res.get("conditions", {}).get(dc, {}).get("auc_pr_creaky", 0.0) for res in results), default=0.0)
        for dc in degraded_conds
    }
    best_clean_f1 = max((res.get("conditions", {}).get("clean", {}).get("f1_creaky_calibrated", 0.0) for res in results), default=0.0)
    best_b_pr = max((res.get("conditions", {}).get("clean", {}).get("auc_pr_breathy", 0.0) for res in results), default=0.0)
    best_b_f1 = max((res.get("conditions", {}).get("clean", {}).get("f1_breathy_calibrated", 0.0) for res in results), default=0.0)

    for res in results:
        b_key = res["backbone"]
        disp = res["display_name"]
        conds_res = res.get("conditions", {})
        clean_m = conds_res.get("clean", res.get("metrics", {}))
        params, pretrain = backbone_meta.get(b_key, ("94M", "Self-Supervised"))

        row = [f"**{disp}**", params, pretrain]

        # Clean Creaky PR
        c_auc = clean_m.get("auc_pr_creaky", 0.0)
        c_auc_str = f"**{c_auc:.4f}**" if abs(c_auc - best_clean_pr) < 1e-5 and best_clean_pr > 0 else f"{c_auc:.4f}"
        row.append(c_auc_str)

        # Degraded conditions with Delta
        for dc in degraded_conds:
            deg_m = conds_res.get(dc, {})
            deg_auc = deg_m.get("auc_pr_creaky", 0.0)
            delta = deg_auc - c_auc
            sign = "+" if delta > 0 else ""
            delta_str = f"({sign}{delta:.4f})"
            val_str = f"{deg_auc:.4f} {delta_str}"
            if abs(deg_auc - best_deg_prs[dc]) < 1e-5 and best_deg_prs[dc] > 0:
                val_str = f"**{deg_auc:.4f}** {delta_str}"
            row.append(val_str)

        # Creaky F1
        c_f1 = clean_m.get("f1_creaky_calibrated", 0.0)
        c_f1_str = f"**{c_f1:.4f}**" if abs(c_f1 - best_clean_f1) < 1e-5 and best_clean_f1 > 0 else f"{c_f1:.4f}"
        row.append(c_f1_str)

        if has_breathy:
            b_auc = clean_m.get("auc_pr_breathy", 0.0)
            b_auc_str = f"**{b_auc:.4f}**" if abs(b_auc - best_b_pr) < 1e-5 and best_b_pr > 0 else f"{b_auc:.4f}"
            b_f1 = clean_m.get("f1_breathy_calibrated", 0.0)
            b_f1_str = f"**{b_f1:.4f}**" if abs(b_f1 - best_b_f1) < 1e-5 and best_b_f1 > 0 else f"{b_f1:.4f}"
            row.extend([b_auc_str, b_f1_str])

        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("*(Note: All backbones frozen; Δ denotes AUC-PR shift relative to Clean. F1 scores evaluated under validation-calibrated decision thresholds.)*")
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    if args.eval_conditions is None:
        if args.smoketest:
            args.eval_conditions = ["clean", "cafe_15db", "reverb_compound"]
        else:
            from wavfry.eval.acoustic_degradation import CANONICAL_EVAL_CONDITIONS
            args.eval_conditions = list(CANONICAL_EVAL_CONDITIONS)
    elif any(c.lower() == "all" for c in args.eval_conditions):
        from wavfry.eval.acoustic_degradation import CANONICAL_EVAL_CONDITIONS
        args.eval_conditions = list(CANONICAL_EVAL_CONDITIONS)

    if args.smoketest and args.output_dir == ROOT / "exp_results" / "backbone_comparison":
        args.output_dir = ROOT / "exp_results" / "backbone_comparison" / "smoketest"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*75}", flush=True)
    print(f"  WavFry Backbone Comparison Suite: WavLM vs HuBERT vs wav2vec 2.0", flush=True)
    print(f"  Selected Backbones: {', '.join(args.backbones)}", flush=True)
    print(f"  Execution Mode:     {args.mode}", flush=True)
    print(f"  Smoke Test Mode:    {args.smoketest}", flush=True)
    print(f"  Compute Device:     {args.device}", flush=True)
    print(f"  Output Directory:   {args.output_dir}", flush=True)
    print(f"  Eval Conditions ({len(args.eval_conditions)}): {', '.join(args.eval_conditions)}", flush=True)
    print(f"{'='*75}\n", flush=True)

    # Build or load multi-dataset manifest once (cached to parquet for instant reload)
    datasets_cfg_path = ROOT / "config" / "datasets.yaml"
    with open(datasets_cfg_path, "r", encoding="utf-8") as f:
        datasets_cfg = yaml.safe_load(f)

    canonical_manifest = ROOT / "manifests" / "segments_multi.csv"
    if canonical_manifest.exists():
        print(f"[Manifest] Loading canonical multi-dataset manifest from {canonical_manifest}...", flush=True)
        import pandas as pd
        df = pd.read_csv(canonical_manifest)
    else:
        dummy_cfg = load_config_for_backbone("wavlm", smoketest=args.smoketest)
        df = build_manifest_multi(datasets_cfg, global_cfg=dummy_cfg)

    if args.smoketest:
        df = create_smoketest_manifest_subset(df)

    all_results = []

    for b_key in args.backbones:
        cfg = load_config_for_backbone(b_key, epochs_override=args.epochs, smoketest=args.smoketest)
        ckpt_path = Path(cfg["checkpoint"]["dir"]) / "best_model.pt"

        if args.mode in ("train_and_eval", "train_only"):
            ckpt_path = train_single_backbone(b_key, cfg, df=df, device=args.device)

        if args.mode in ("train_and_eval", "eval_only"):
            test_df = df[df["batch_split"] == "test"].copy()
            eval_res = evaluate_backbone(
                backbone_key=b_key,
                cfg=cfg,
                ckpt_path=ckpt_path,
                test_df=test_df,
                device=args.device,
                conditions=args.eval_conditions,
            )
            all_results.append(eval_res)

    if all_results:
        import pandas as pd
        from wavfry.eval.acoustic_degradation import get_condition_metadata

        # 1. Primary Machine-Readable: Tidy CSV with TitleCase LaTeX-aligned headers
        detail_rows = []
        for res in all_results:
            b_name = res.get("display_name", res.get("backbone", "Unknown"))
            conds_dict = res.get("conditions", {})
            for cond, metrics in conds_dict.items():
                meta = get_condition_metadata(cond)
                for task in ["creaky", "breathy"]:
                    if f"auc_pr_{task}" in metrics:
                        clean_pr = conds_dict.get("clean", {}).get(f"auc_pr_{task}", 0.0)
                        pr_auc = float(metrics.get(f"auc_pr_{task}", 0.0))
                        delta_pr = pr_auc - clean_pr if cond != "clean" else 0.0
                        detail_rows.append({
                            "Model": b_name,
                            "Task": task,
                            "Condition": cond,
                            "Scenario": meta["Scenario"],
                            "SNR_dB": meta["SNR_dB"],
                            "Tier": meta["Tier"],
                            "Has_RIR": meta["Has_RIR"],
                            "PR-AUC": round(pr_auc, 4),
                            "Delta-PR-AUC": round(delta_pr, 4),
                            "ROC-AUC": round(float(metrics.get(f"roc_auc_{task}", 0.0)), 4),
                            "F1-Score": round(float(metrics.get(f"f1_{task}_default", 0.0)), 4),
                            "F1-Calibrated": round(float(metrics.get(f"f1_{task}_calibrated", 0.0)), 4),
                            "Precision": round(float(metrics.get(f"precision_{task}", 0.0)), 4),
                            "Recall": round(float(metrics.get(f"recall_{task}", 0.0)), 4),
                        })

        if detail_rows:
            csv_path = args.output_dir / "table5_conditions_detail.csv"
            pd.DataFrame(detail_rows).to_csv(csv_path, index=False, encoding="utf-8")
            print(f"\n[Saved] Tidy CSV saved to {csv_path}", flush=True)

        # 2. Hierarchical Machine-Readable JSON
        json_path = args.output_dir / "backbone_comparison_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"[Saved] Hierarchical JSON saved to {json_path}", flush=True)

        # 3. Human-Friendly Markdown Table Byproduct
        table_md = format_academic_summary_table(all_results, conditions=args.eval_conditions)
        print("\n" + table_md, flush=True)

        table_path = args.output_dir / "table5_backbone_comparison.md"
        table_path.write_text(table_md, encoding="utf-8")
        print(f"[Saved] Markdown Table byproduct saved to {table_path}", flush=True)


if __name__ == "__main__":
    main()

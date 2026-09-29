#!/usr/bin/env python
"""
Automated Ablation Study Runner for Frozen WavLM Acoustic Feature Evaluation.
Fail-Safe Unattended Overnight Execution Edition with Full Epoch-by-Epoch Metric Tracking.

Key Features:
  - 100% Config-Driven: Reads all hyperparameters (epochs, batch_size, paradigm, lr, patience)
    from base YAML config, avoiding hardcoded overrides.
  - Zero LOOCV: Trains and validates on the configured dataset split.
  - Full Epoch-by-Epoch Tracking: Collects and exports full learning trajectories
    (loss, PR-AUC, F1, acc, prec, rec, lr) for every single epoch into JSON, CSV, and Markdown.
  - Tailored for Frozen Models: Supports pooling suite, scalar mix suite, and augmentation suite.
  - Robust Fail-Safe: Windows sleep prevention, sub-process isolation, atomic file writes.

Author: WavFry ML Infrastructure
Date: September 2026
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import gc
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
import warnings
from torch.utils.data import DataLoader

# Suppress upstream PyTorch 2.6+ deprecation warning regarding WavLM's boolean key_padding_mask
warnings.filterwarnings("ignore", message=".*Support for mismatched key_padding_mask and attn_mask is deprecated.*")
warnings.filterwarnings("ignore", message=".*torchaudio.backend.common.AudioMetaData.*")
warnings.filterwarnings("ignore", message=".*No positive class found in y_true.*")

# Add project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe stdout reconfigure
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Force offline mode for HuggingFace / Transformers globally
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# ── Windows Power Management (Sleep Prevention) ──────────────────────────────
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_AWAYMODE_REQUIRED = 0x00000040


def prevent_windows_sleep():
    """Prevent Windows from sleeping or suspending while this script runs."""
    if sys.platform == "win32":
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
            )
            print("[Power] Windows Sleep Prevention ENABLED (Continuous System & Execution lock).")
        except Exception as e:
            print(f"[Power] Warning: Failed to enable sleep prevention: {e}")


def restore_windows_sleep():
    """Restore default Windows sleep behavior."""
    if sys.platform == "win32":
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
            print("[Power] Windows Sleep Prevention RESTORED to normal.")
        except Exception:
            pass


def set_console_title(title: str):
    """Set Windows command prompt window title for progress monitoring."""
    if sys.platform == "win32":
        try:
            ctypes.windll.kernel32.SetConsoleTitleW(title)
        except Exception:
            pass


# ── Ablation Suites Definition ───────────────────────────────────────────────
ABLATION_SUITES: Dict[str, List[Dict[str, Any]]] = {
    "pooling": [
        {
            "id": "single_attn",
            "name": "Single-Branch (Attention Pooling, Baseline)",
            "overrides": [
                "training.pooling_mode=attn",
            ],
        },
        {
            "id": "single_mean",
            "name": "Single-Branch (Mean Pooling)",
            "overrides": [
                "training.pooling_mode=mean",
            ],
        },
        {
            "id": "dual_mean_attn",
            "name": "Dual-Branch (Mean + Attn, Proposed)",
            "overrides": [
                "training.pooling_mode=mean_attn",
            ],
        },
        {
            "id": "dual_mean_topk",
            "name": "Dual-Branch (Mean + Top-K)",
            "overrides": [
                "training.pooling_mode=mean_topk",
                "training.topk_ratio=0.2",
                "training.topk_min_frames=3",
            ],
        },
    ],
    "scalar_mix": [
        {
            "id": "scalar_mix_all",
            "name": "Learnable 13-Layer Scalar Mix (Full Mix)",
            "overrides": [
                "model.use_scalar_mix=true",
            ],
        },
        {
            "id": "top_layer_only",
            "name": "Top-Layer Feature Only (Layer 12)",
            "overrides": [
                "model.use_scalar_mix=false",
            ],
        },
    ],
    "augmentation": [
        {
            "id": "with_aug",
            "name": "With Acoustic Augmentation (Noise + Denoise)",
            "overrides": [
                "augmentation.enabled=true",
            ],
        },
        {
            "id": "no_aug",
            "name": "Clean Audio Only (No Augmentation)",
            "overrides": [
                "augmentation.enabled=false",
            ],
        },
    ],
    "loss": [
        {
            "id": "focal_loss",
            "name": "Focal Loss (gamma=2.0, alpha=0.75)",
            "overrides": [
                "training.loss=focal",
            ],
        },
        {
            "id": "weighted_bce",
            "name": "Class-Weighted BCE Loss",
            "overrides": [
                "training.loss=weighted_bce",
            ],
        },
    ],
}


def get_ablation_configs(suite_name: str, requested_ids: List[str] | None = None) -> List[Dict[str, Any]]:
    """Resolve target configurations based on suite name or specific IDs."""
    if suite_name == "all":
        all_cfgs: List[Dict[str, Any]] = []
        seen_ids = set()
        for suite_list in ABLATION_SUITES.values():
            for c in suite_list:
                if c["id"] not in seen_ids:
                    all_cfgs.append(c)
                    seen_ids.add(c["id"])
        selected = all_cfgs
    elif suite_name in ABLATION_SUITES:
        selected = list(ABLATION_SUITES[suite_name])
    else:
        raise ValueError(f"Unknown ablation suite: '{suite_name}'. Valid suites: {list(ABLATION_SUITES.keys()) + ['all']}")

    if requested_ids:
        req_set = set(requested_ids)
        candidates: List[Dict[str, Any]] = []
        for suite_list in ABLATION_SUITES.values():
            for c in suite_list:
                if c["id"] in req_set and not any(cand["id"] == c["id"] for cand in candidates):
                    candidates.append(c)
        return candidates

    return selected


def robust_load_checkpoint(checkpoint_path: Path, map_location: str = "cpu", max_retries: int = 3) -> Dict[str, Any]:
    """Load PyTorch checkpoint with retries to protect against transient Windows file lock races."""
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        except (PermissionError, OSError) as e:
            last_err = e
            print(f"[Warning] Transient file lock reading {checkpoint_path.name} (attempt {attempt}/{max_retries}): {e}")
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"Failed to load checkpoint after {max_retries} attempts: {checkpoint_path}") from last_err


def run_training_subprocess(
    config_path: Path,
    overrides: List[str],
    cwd: Path,
    log_file: Path,
    resume: bool = False,
) -> bool:
    """Run train.py in an isolated subprocess with UTF-8 handling and progress streaming."""
    if shutil.which("uv"):
        cmd = ["uv", "run", "python", "scripts/train.py"]
    else:
        cmd = [sys.executable, "scripts/train.py"]

    cmd.extend(["--config", str(config_path)])
    if resume:
        cmd.append("--resume")
    if overrides:
        cmd.extend(["--override"] + overrides)

    print(f"\n[Ablation Runner] Launching training subprocess (log: {log_file.as_posix()}):")
    print(f"  Command: {' '.join(cmd)}")
    print("-" * 75)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    log_file.parent.mkdir(parents=True, exist_ok=True)
    tail_buffer = deque(maxlen=40)
    in_traceback = False

    with open(log_file, "w", encoding="utf-8", errors="replace") as f_out:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
        )
        for line in iter(proc.stdout.readline, ""):
            f_out.write(line)
            f_out.flush()
            tail_buffer.append(line)
            stripped = line.strip()

            if "Traceback (most recent call last):" in stripped:
                in_traceback = True

            should_print = in_traceback or any(
                stripped.startswith(prefix)
                for prefix in (
                    "[Epoch ",
                    "[Split]",
                    "Train:",
                    "Best threshold",
                    "Synchronized optimal",
                    "Early stopping",
                    "[Golden]",
                    "Resuming",
                )
            ) or any(k in stripped.lower() for k in ("error", "exception", "failed", "fatal"))

            if should_print:
                print(f"  {stripped}", flush=True)

            if in_traceback and len(stripped) > 0 and not stripped.startswith(" ") and not stripped.startswith("File"):
                in_traceback = False

        proc.wait()

    if proc.returncode != 0:
        print("\n" + "!" * 75)
        print(f"[CRITICAL ERROR] Subprocess exited with code {proc.returncode}!")
        print("Tail of training log:")
        print("-" * 75)
        for t_line in tail_buffer:
            print("  " + t_line.rstrip(), flush=True)
        print("!" * 75 + "\n")
        return False

    return True


def evaluate_checkpoint_on_val(
    checkpoint_path: Path,
    manifest_df: pd.DataFrame,
    device: torch.device,
    is_smoketest: bool = False,
) -> Dict[str, Any]:
    """Load model from checkpoint, build exact validation split from config, and evaluate at optimal threshold."""
    from wavfry.data import splits as splits_mod
    from wavfry.data import dataset as dataset_mod
    from wavfry.model import wavlm_classifier as model_mod
    from wavfry.train import metrics as metrics_mod

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[Eval] Loading checkpoint from {checkpoint_path}...")
    ckpt = robust_load_checkpoint(checkpoint_path, map_location="cpu")
    config = ckpt["config"]
    paradigm = config["training"].get("paradigm", "block")
    isolate = bool(config["data"].get("isolate_split_speakers", True))

    eval_df = manifest_df.copy()
    # Exact matching of train.py dataset split logic
    train_df, val_df, test_df = splits_mod.make_batch_split(eval_df, isolate=isolate)

    if is_smoketest:
        seed = config["training"].get("seed", 42)
        pos = val_df[val_df["label"] == 1]
        neg = val_df[val_df["label"] == 0]
        n_pos = min(len(pos), 32)
        n_neg = min(len(neg), 32)
        val_df = pd.concat([pos.sample(n=n_pos, random_state=seed), neg.sample(n=n_neg, random_state=seed)]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        print(f"[Smoketest Eval] Sliced validation set to {len(val_df)} segments for rapid sanity check.")

    print(f"[Eval] Evaluating on Validation split: {len(val_df)} segments (Split Mode: batch_split, Paradigm: {paradigm}).")

    if paradigm == "block":
        from wavfry.data.paradigm_block import ContinuousAudioBlockDataset, collate_block_fn
        block_cfg = config["training"].get("block", {})
        common_block = dict(
            duration_ms=block_cfg.get("duration_ms", 2000),
            hop_ms=block_cfg.get("hop_ms", 1000),
            max_intervals=block_cfg.get("max_intervals", 8),
            boundary_trim_frames=block_cfg.get("boundary_trim_frames", 0),
            seed=42,
            fold_index=0,
        )
        val_ds = ContinuousAudioBlockDataset(val_df, **common_block)
        collate = collate_block_fn
        # In eval mode (no_grad), double batch size safely for 2x faster throughput
        batch_size = int(block_cfg.get("batch_size") or config["training"].get("batch_size", 16)) * 2
    else:
        val_ds = dataset_mod.CreakyVowelDataset(
            val_df,
            context_padding_ms=config["training"]["segment"].get("context_padding_ms", 30),
        )
        collate = dataset_mod.collate_fn
        batch_size = int(config["training"]["segment"].get("batch_size", 32))

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Safe on Windows
        collate_fn=collate,
        pin_memory=False,
    )

    model = None
    all_logits = []
    all_labels = []

    try:
        model = model_mod.WavFryClassifier(config)
        save_mode = config["checkpoint"].get("save_mode", "delta")
        if save_mode == "delta":
            load_result = model.load_state_dict(ckpt["model_state_dict"], strict=False)
            if load_result.unexpected_keys:
                print(f"[Eval Warning] Unexpected keys in delta checkpoint: {load_result.unexpected_keys}")
        else:
            model.load_state_dict(ckpt["model_state_dict"])

        try:
            model.to(device)
        except torch.cuda.OutOfMemoryError:
            print("[Eval Warning] CUDA OOM while allocating evaluation model. Falling back to CPU.")
            device = torch.device("cpu")
            model.to(device)

        model.eval()

        print(f"[Eval] Running forward passes on {device}...")
        with torch.no_grad():
            if paradigm == "block":
                interval_predictions = {}
                for batch in val_loader:
                    waveforms, bounds, labels, valid, lengths, speakers, block_ids, interval_ids = batch
                    waveforms = waveforms.to(device)
                    bounds = bounds.to(device)
                    valid = valid.to(device)
                    lengths = lengths.to(device)

                    outputs = model(waveforms, lengths, interval_bounds=bounds, interval_valid=valid)
                    slot_valid = outputs.get("interval_valid", valid)
                    raw_logits = outputs["interval_logits"]
                    if isinstance(raw_logits, dict):
                        task_logits = raw_logits.get("creaky", next(iter(raw_logits.values())))
                    else:
                        task_logits = raw_logits

                    logits = task_logits[slot_valid].detach().cpu()
                    lbls = labels.to(device)
                    if lbls.ndim > 2 and lbls.shape[-1] >= 1:
                        lbls = lbls[..., 0]
                    lbls = lbls[slot_valid].detach().cpu()

                    valid_cpu = slot_valid.detach().cpu().numpy()
                    flat_ids = []
                    for b_i in range(valid_cpu.shape[0]):
                        for s_i in range(valid_cpu.shape[1]):
                            if valid_cpu[b_i, s_i]:
                                flat_ids.append(interval_ids[b_i][s_i])

                    for logit_val, lbl_val, iid in zip(logits, lbls, flat_ids):
                        l_item = logit_val.item()
                        lbl_item = lbl_val.item()
                        if iid not in interval_predictions:
                            interval_predictions[iid] = [l_item, lbl_item, 1]
                        else:
                            interval_predictions[iid][0] += l_item
                            interval_predictions[iid][2] += 1

                for item in interval_predictions.values():
                    all_logits.append(item[0] / item[2])
                    all_labels.append(item[1])
                logits_tensor = torch.tensor(all_logits, dtype=torch.float32)
                labels_tensor = torch.tensor(all_labels, dtype=torch.float32)
            else:
                for waveforms, labels, lengths, speaker_ids, item_ids in val_loader:
                    waveforms = waveforms.to(device)
                    lengths = lengths.to(device)
                    outputs = model(waveforms, lengths)
                    if isinstance(outputs, dict):
                        logits = outputs.get("creaky", next(iter(outputs.values())))
                    else:
                        logits = outputs
                    lbls = labels
                    if lbls.ndim > 1 and lbls.shape[-1] >= 1:
                        lbls = lbls[..., 0]
                    all_logits.append(logits.detach().cpu())
                    all_labels.append(lbls.detach().cpu())
                logits_tensor = torch.cat(all_logits).flatten()
                labels_tensor = torch.cat(all_labels).flatten()

    finally:
        if model is not None:
            model.to("cpu")
            del model
        del val_loader, val_ds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Optimal threshold search (consistent step: 0.005 from 0.01 to 0.99).
    # NaN-safe: rows without this task's label are excluded instead of being
    # int-cast into garbage negatives.
    from wavfry.eval.unified import f1_scan_best_threshold
    best_t, best_f1 = f1_scan_best_threshold(logits_tensor, labels_tensor)

    # Compute metrics at best_t
    metrics_at_best_t = metrics_mod.compute_metrics(logits_tensor, labels_tensor, threshold=best_t)
    metrics_at_best_t["best_threshold"] = float(best_t)

    print(
        f"[Eval Result] PR-AUC: {metrics_at_best_t['auc_pr']:.4f} | "
        f"Optimal Threshold: {best_t:.3f} | F1: {metrics_at_best_t['f1']:.4f} | "
        f"Accuracy: {metrics_at_best_t['accuracy']:.4f} | "
        f"Precision: {metrics_at_best_t['precision']:.4f} | "
        f"Recall: {metrics_at_best_t['recall']:.4f}"
    )
    return metrics_at_best_t


def format_latex_table(results: List[Dict[str, Any]]) -> str:
    """Format comparison table with bold best and underlined second-best."""
    metrics_keys = ["auc_pr", "f1", "accuracy", "precision", "recall"]
    valid_results = [r for r in results if r.get("status") != "failed" and "metrics" in r]

    ranks: Dict[str, Tuple[float, float]] = {}
    for k in metrics_keys:
        vals = [r["metrics"][k] for r in valid_results if k in r["metrics"] and not math.isnan(r["metrics"][k])]
        unique_rounded = sorted(list(set(round(v, 4) for v in vals)), reverse=True)
        best = unique_rounded[0] if len(unique_rounded) > 0 else -1.0
        second = unique_rounded[1] if len(unique_rounded) > 1 else -1.0
        ranks[k] = (best, second)

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Ablation study on validation set. "
        r"PR-AUC denotes Precision-Recall Area Under Curve. F1, Accuracy, Precision, "
        r"and Recall are evaluated at the optimal decision threshold ($\tau^*$). "
        r"Best results are highlighted in bold, and second-best results are underlined.}"
    )
    lines.append(r"\label{tab:ablation_frozen}")
    lines.append(r"\resizebox{\linewidth}{!}{")
    lines.append(r"\begin{tabular}{@{}lccccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Configuration} & \textbf{PR-AUC} & \textbf{F1} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} \\")
    lines.append(r"\midrule")

    for r in results:
        cfg_name = r["name"]
        if r.get("status") == "failed":
            lines.append(f"{cfg_name:<35} & \\multicolumn{{5}}{{c}}{{\\textit{{Failed / Incomplete}}}} \\\\")
            continue

        m = r.get("metrics", {})
        row_str = f"{cfg_name:<35}"
        for k in metrics_keys:
            val = m.get(k, float("nan"))
            if math.isnan(val):
                row_str += " & --"
                continue

            val_str = f"{val:.4f}"
            val_round = round(val, 4)
            best, second = ranks[k]
            if abs(val_round - best) < 1e-6:
                formatted = f"\\textbf{{{val_str}}}"
            elif abs(val_round - second) < 1e-6:
                formatted = f"\\underline{{{val_str}}}"
            else:
                formatted = val_str
            row_str += f" & {formatted}"
        row_str += r" \\"
        lines.append(row_str)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def save_intermediate_summaries(summary_results: List[Dict[str, Any]], base_out_dir: Path, base_cfg: Dict[str, Any]):
    """Save raw JSON, combined epoch CSV, Markdown report, and LaTeX table atomically after each configuration."""
    base_out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Full JSON (including epoch histories)
    summary_json_path = base_out_dir / "ablation_summary.json"
    tmp_json = summary_json_path.with_suffix(".json.tmp")
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)
    os.replace(tmp_json, summary_json_path)

    # 2. Combined Epoch History CSV (all configurations in a single tidy table)
    epoch_rows = []
    for r in summary_results:
        cfg_id = r["id"]
        cfg_name = r["name"]
        history = r.get("epoch_history", [])
        for h in history:
            row = {"config_id": cfg_id, "config_name": cfg_name}
            row.update(h)
            epoch_rows.append(row)

    if epoch_rows:
        epoch_csv_path = base_out_dir / "ablation_epoch_history.csv"
        tmp_csv = epoch_csv_path.with_suffix(".csv.tmp")
        pd.DataFrame(epoch_rows).to_csv(tmp_csv, index=False)
        os.replace(tmp_csv, epoch_csv_path)

    # 3. Markdown Report with Epoch Trajectories
    metrics_keys = ["auc_pr", "f1", "accuracy", "precision", "recall"]
    valid_results = [r for r in summary_results if r.get("status") != "failed" and "metrics" in r]
    ranks = {}
    for k in metrics_keys:
        vals = [r["metrics"][k] for r in valid_results if k in r["metrics"] and not math.isnan(r["metrics"][k])]
        unique_vals = sorted(list(set(round(v, 4) for v in vals)), reverse=True)
        ranks[k] = (unique_vals[0] if len(unique_vals) > 0 else -1.0, unique_vals[1] if len(unique_vals) > 1 else -1.0)

    paradigm = base_cfg.get("training", {}).get("paradigm", "block")
    val_spk = "batch_split"

    summary_md_path = base_out_dir / "ablation_summary.md"
    tmp_md = summary_md_path.with_suffix(".md.tmp")
    md_lines = [
        "# WavFry Frozen Backbone Ablation Study Results",
        "",
        f"- **Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Backbone**: `{base_cfg.get('model', {}).get('backbone', 'wavlm')}` (Frozen: `{base_cfg.get('model', {}).get('fine_tune_layers', 'none')}`)",
        f"- **Paradigm**: `{paradigm}`",
        f"- **Validation Split**: `{val_spk}`",
        "",
        "## 1. Summary of Best Validation Metrics",
        "",
        "| Configuration | PR-AUC | F1 (at $\\tau^*$) | Accuracy | Precision | Recall | Optimal $\\tau^*$ | Best Epoch | Total Epochs |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in summary_results:
        if r.get("status") == "failed":
            md_lines.append(f"| {r['name']} | *Failed* | *Failed* | *Failed* | *Failed* | *Failed* | -- | -- | -- |")
            continue
        m = r.get("metrics", {})
        cols = []
        for k in metrics_keys:
            val = m.get(k, float("nan"))
            if math.isnan(val):
                cols.append("--")
            else:
                best, second = ranks.get(k, (-1.0, -1.0))
                s = f"{val:.4f}"
                if abs(round(val, 4) - best) < 1e-6:
                    cols.append(f"**{s}**")
                elif abs(round(val, 4) - second) < 1e-6:
                    cols.append(f"*{s}*")
                else:
                    cols.append(s)
        best_t_str = f"{m.get('best_threshold', 0.5):.3f}"
        best_ep = str(r.get("best_epoch", m.get("epoch", "--")))
        total_ep = str(len(r.get("epoch_history", [])))
        md_lines.append(f"| {r['name']} | {cols[0]} | {cols[1]} | {cols[2]} | {cols[3]} | {cols[4]} | {best_t_str} | {best_ep} | {total_ep} |")

    # 4. Epoch-by-Epoch Progress Trajectory Comparison Table
    if epoch_rows:
        md_lines.extend([
            "",
            "## 2. Epoch-by-Epoch Learning Trajectories",
            "",
            "Evolution of Validation PR-AUC (and Loss) across epochs (*marks the best epoch per configuration):",
            "",
        ])
        max_ep = max(h.get("epoch", 1) for r in summary_results for h in r.get("epoch_history", []))
        header = "| Epoch | " + " | ".join(f"{r['id']} (PR-AUC / Val Loss)" for r in summary_results) + " |"
        sep = "| :---: | " + " | ".join(":---:" for _ in summary_results) + " |"
        md_lines.append(header)
        md_lines.append(sep)

        for ep in range(1, max_ep + 1):
            row_vals = [f"**{ep}**"]
            for r in summary_results:
                h_match = next((h for h in r.get("epoch_history", []) if h.get("epoch") == ep), None)
                if h_match:
                    prauc = h_match.get("val_auc_pr", 0.0)
                    vloss = h_match.get("val_loss", 0.0)
                    is_b = h_match.get("is_best", False)
                    mark = "**" if is_b else ""
                    row_vals.append(f"{mark}{prauc:.4f} ({vloss:.3f}){mark}")
                else:
                    row_vals.append("--")
            md_lines.append("| " + " | ".join(row_vals) + " |")

    tmp_md.write_text("\n".join(md_lines), encoding="utf-8")
    os.replace(tmp_md, summary_md_path)

    # 5. LaTeX Snippet
    latex_code = format_latex_table(summary_results)
    latex_file = base_out_dir / "table1_ablation.tex"
    tmp_tex = latex_file.with_suffix(".tex.tmp")
    tmp_tex.write_text(latex_code, encoding="utf-8")
    os.replace(tmp_tex, latex_file)


def resolve_best_checkpoint(run_ckpt_dir: Path) -> Path | None:
    """Find the highest-performing checkpoint in run_ckpt_dir robustly."""
    best_fixed = run_ckpt_dir / "best_model.pt"
    if best_fixed.exists():
        return best_fixed

    candidates = list(run_ckpt_dir.glob("best_ep*.pt"))
    if candidates:
        def extract_prauc(p: Path) -> float:
            try:
                part = p.name.split("_prauc")[1].split("_")[0]
                return float(part)
            except Exception:
                return 0.0
        candidates.sort(key=extract_prauc, reverse=True)
        return candidates[0]

    latest_single = run_ckpt_dir / "latest.pt"
    if latest_single.exists():
        return latest_single

    latest_candidates = list(run_ckpt_dir.glob("latest_*.pt"))
    if latest_candidates:
        return latest_candidates[0]

    return None


def main():
    parser = argparse.ArgumentParser(description="Automated Ablation Study for Frozen WavLM Experiments")
    parser.add_argument("--config", type=str, default="config/experiments/frozen_wavlm_base_plus.yaml", help="Path to base config")
    parser.add_argument("--suite", type=str, default="pooling", choices=list(ABLATION_SUITES.keys()) + ["all"], help="Ablation suite to execute")
    parser.add_argument("--configs", type=str, nargs="+", default=None, help="Subset of configuration IDs to run")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs (default: read from config)")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size (default: read from config)")
    parser.add_argument("--paradigm", type=str, default=None, choices=["segment", "block"], help="Override paradigm (default: read from config)")
    parser.add_argument("--patience", type=int, default=None, help="Override early stopping patience (default: read from config)")
    parser.add_argument("--grad-accum", type=int, default=None, help="Override gradient accumulation steps (default: read from config)")
    parser.add_argument("--output-dir", type=str, default=None, help="TensorBoard/run output dir (default: read from config)")
    parser.add_argument("--ckpt-dir", type=str, default=None, help="Checkpoint root dir (default: read from config)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing evaluation results")
    parser.add_argument("--smoketest", action="store_true", help="Quick dry-run: 1 epoch, small data test")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Eval device")
    args = parser.parse_args()

    # Prevent Windows from sleeping during long ablation runs
    prevent_windows_sleep()

    device = torch.device(args.device)
    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Read all base hyperparameters dynamically from config
    with open(config_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    paradigm = args.paradigm if args.paradigm is not None else base_cfg.get("training", {}).get("paradigm", "block")
    epochs = args.epochs if args.epochs is not None else base_cfg.get("training", {}).get("epochs", 25)
    if args.batch_size is not None:
        batch_size = args.batch_size
    elif paradigm == "block":
        batch_size = base_cfg.get("training", {}).get("block", {}).get("batch_size") or base_cfg.get("training", {}).get("batch_size", 16)
    else:
        batch_size = base_cfg.get("training", {}).get("batch_size", 16)

    grad_accum = args.grad_accum if args.grad_accum is not None else base_cfg.get("training", {}).get("gradient_accumulation_steps", 1)
    patience = args.patience if args.patience is not None else base_cfg.get("eval", {}).get("early_stopping_patience", 3)

    raw_out_dir = args.output_dir or base_cfg.get("tensorboard", {}).get("log_dir", "R:/runs/frozen_wavlm_ablation")
    base_out_dir = Path(raw_out_dir) if Path(raw_out_dir).is_absolute() else (PROJECT_ROOT / raw_out_dir)

    raw_ckpt_dir = args.ckpt_dir or base_cfg.get("checkpoint", {}).get("dir", "R:/checkpoints/frozen_wavlm_ablation")
    base_ckpt_dir = Path(raw_ckpt_dir) if Path(raw_ckpt_dir).is_absolute() else (PROJECT_ROOT / raw_ckpt_dir)

    manifest_path = PROJECT_ROOT / "manifests" / "segments_multi.csv"
    if not manifest_path.exists():
        manifest_path = PROJECT_ROOT / "manifests" / "segments_all.csv"
    if not manifest_path.exists():
        print(f"[Error] Manifest {manifest_path} not found. Rebuilding...")
        builder_cmd = ["uv", "run", "python", "scripts/build_manifest.py"] if shutil.which("uv") else [sys.executable, "scripts/build_manifest.py"]
        subprocess.run(builder_cmd, cwd=str(PROJECT_ROOT), check=True)

    manifest_df = pd.read_csv(manifest_path, dtype={"speaker_id": str, "phone": str, "file_key": str})
    base_out_dir.mkdir(parents=True, exist_ok=True)
    base_ckpt_dir.mkdir(parents=True, exist_ok=True)

    target_configs = get_ablation_configs(args.suite, args.configs)

    effective_epochs = 1 if args.smoketest else epochs
    effective_patience = 0 if args.smoketest else patience

    print("=" * 80)
    print("      WavFry Frozen Backbone Ablation Study Suite")
    print(f"  Base Config: {config_path.as_posix()}")
    print(f"  Suite: {args.suite.upper()} | Target Configs ({len(target_configs)}): {[c['id'] for c in target_configs]}")
    print(f"  Paradigm: {paradigm.upper()} | Epochs: {effective_epochs} | Batch Size: {batch_size} | Grad Accum: {grad_accum}")
    print(f"  Patience: {effective_patience} | Device: {device}")
    print(f"  Output Dir: {base_out_dir}")
    print(f"  Checkpoint Dir: {base_ckpt_dir}")
    print("=" * 80)

    summary_results: List[Dict[str, Any]] = []
    any_failed = False

    try:
        for idx, cfg in enumerate(target_configs, 1):
            cfg_id = cfg["id"]
            cfg_name = cfg["name"]
            set_console_title(f"[{idx}/{len(target_configs)}] WavFry Ablation - {cfg_id}")
            print(f"\n[{idx}/{len(target_configs)}] Processing Configuration: {cfg_name} ({cfg_id})")

            run_log_dir = base_out_dir / cfg_id
            run_ckpt_dir = base_ckpt_dir / cfg_id
            eval_metrics_file = run_ckpt_dir / "eval_val_metrics.json"
            epoch_hist_file = run_ckpt_dir / "epoch_history.json"

            done_marker = run_ckpt_dir / "_DONE"

            # 1. Check resume: complete result exists (marker + metrics both required)
            if args.resume and eval_metrics_file.exists() and done_marker.exists():
                try:
                    with open(eval_metrics_file, "r", encoding="utf-8") as f:
                        metrics = json.load(f)
                    epoch_hist = []
                    if epoch_hist_file.exists():
                        with open(epoch_hist_file, "r", encoding="utf-8") as f_h:
                            epoch_hist = json.load(f_h)
                    print(f"[Resume] Found existing evaluation metrics at {eval_metrics_file}. Loaded successfully.")
                    summary_results.append({
                        "id": cfg_id,
                        "name": cfg_name,
                        "status": "success",
                        "best_epoch": metrics.get("epoch", 1),
                        "metrics": metrics,
                        "epoch_history": epoch_hist,
                    })
                    save_intermediate_summaries(summary_results, base_out_dir, base_cfg)
                    continue
                except Exception as e:
                    print(f"[Resume Warning] Corrupt eval metrics file {eval_metrics_file}: {e}. Rerunning.")

            # 2. Check resume: training already complete but eval pending.
            # Only a _DONE-marked run may skip the training pass — a crashed
            # run also leaves checkpoints, and skipping would evaluate an
            # unterminated model and mark it success.
            best_ckpt = resolve_best_checkpoint(run_ckpt_dir)
            skip_training = False
            if args.resume and best_ckpt is not None and done_marker.exists():
                print(f"[Resume] Existing checkpoint detected: {best_ckpt}. Checking if evaluation can proceed directly...")
                try:
                    ckpt_test = robust_load_checkpoint(best_ckpt, map_location="cpu")
                    if "model_state_dict" in ckpt_test:
                        print(f"[Resume] Checkpoint {best_ckpt.name} verified intact. Skipping training pass.")
                        skip_training = True
                    del ckpt_test
                except Exception as e:
                    print(f"[Resume Warning] Checkpoint verification failed: {e}. Will resume/retrain via subprocess.")

            if not skip_training:
                # Prepare overrides without overwriting un-ablated hyperparameters
                overrides = [
                    "training.quiet=true",
                    f"tensorboard.log_dir={run_log_dir.as_posix()}",
                    f"checkpoint.dir={run_ckpt_dir.as_posix()}",
                ]

                # If smoketest or CLI overrides explicitly given, pass them
                if args.smoketest:
                    overrides.extend([
                        "smoketest=true",
                        "training.epochs=1",
                        "eval.early_stopping_patience=0",
                    ])
                else:
                    if args.epochs is not None:
                        overrides.append(f"training.epochs={epochs}")
                    if args.patience is not None:
                        overrides.append(f"eval.early_stopping_patience={patience}")
                    if args.grad_accum is not None:
                        overrides.append(f"training.gradient_accumulation_steps={grad_accum}")
                    if args.batch_size is not None:
                        overrides.append(f"training.{paradigm}.batch_size={batch_size}")
                    if args.paradigm is not None:
                        overrides.append(f"training.paradigm={paradigm}")

                # Append specific ablation overrides
                overrides.extend(cfg["overrides"])

                start_time = time.time()
                log_file = run_ckpt_dir / "train.log"

                success = run_training_subprocess(
                    config_path=config_path,
                    overrides=overrides,
                    cwd=PROJECT_ROOT,
                    log_file=log_file,
                    resume=args.resume,
                )
                elapsed = time.time() - start_time
                print(f"[Training] Finished in {elapsed:.1f}s. Subprocess Success: {success}")

                if not success:
                    print(f"[ERROR] Training failed for configuration {cfg_id}! Continuing with next configuration...")
                    any_failed = True
                    summary_results.append({
                        "id": cfg_id,
                        "name": cfg_name,
                        "status": "failed",
                        "error": "Training subprocess returned non-zero exit code",
                    })
                    save_intermediate_summaries(summary_results, base_out_dir, base_cfg)
                    continue

                best_ckpt = resolve_best_checkpoint(run_ckpt_dir)

            if best_ckpt is None or not best_ckpt.exists():
                print(f"[ERROR] No valid checkpoint found in {run_ckpt_dir} after training!")
                any_failed = True
                summary_results.append({
                    "id": cfg_id,
                    "name": cfg_name,
                    "status": "failed",
                    "error": f"No checkpoint found in {run_ckpt_dir}",
                })
                save_intermediate_summaries(summary_results, base_out_dir, base_cfg)
                continue

            # Load epoch history generated by Trainer
            epoch_history = []
            for h_candidate in (run_ckpt_dir / "epoch_history.json", run_log_dir / "epoch_history.json"):
                if h_candidate.exists():
                    try:
                        with open(h_candidate, "r", encoding="utf-8") as f_h:
                            epoch_history = json.load(f_h)
                        break
                    except Exception as e:
                        print(f"[Warning] Could not read {h_candidate}: {e}")

            # 3. Evaluate checkpoint on validation set
            try:
                metrics = evaluate_checkpoint_on_val(
                    checkpoint_path=best_ckpt,
                    manifest_df=manifest_df,
                    device=device,
                    is_smoketest=args.smoketest,
                )

                best_ep = 1
                for h in epoch_history:
                    if h.get("is_best"):
                        best_ep = h.get("epoch", 1)
                metrics["epoch"] = best_ep

                with open(eval_metrics_file, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2)
                # Completion marker: a metrics json alone cannot distinguish a
                # finished run from a crashed one (D-08); --resume must never
                # aggregate an unterminated run as success.
                (run_ckpt_dir / "_DONE").write_text(
                    json.dumps({
                        "id": cfg_id,
                        "best_epoch": best_ep,
                        "best_threshold": metrics.get("best_threshold"),
                    }),
                    encoding="utf-8",
                )

                summary_results.append({
                    "id": cfg_id,
                    "name": cfg_name,
                    "status": "success",
                    "best_epoch": best_ep,
                    "metrics": metrics,
                    "epoch_history": epoch_history,
                })
            except Exception as eval_err:
                print(f"[ERROR] Evaluation failed for {cfg_id}: {eval_err}")
                traceback.print_exc()
                any_failed = True
                summary_results.append({
                    "id": cfg_id,
                    "name": cfg_name,
                    "status": "failed",
                    "error": str(eval_err),
                    "epoch_history": epoch_history,
                })
            finally:
                save_intermediate_summaries(summary_results, base_out_dir, base_cfg)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    finally:
        save_intermediate_summaries(summary_results, base_out_dir, base_cfg)
        restore_windows_sleep()
        set_console_title("WavFry Ablation Suite - Completed")

    summary_json_path = base_out_dir / "ablation_summary.json"
    summary_md_path = base_out_dir / "ablation_summary.md"
    epoch_csv_path = base_out_dir / "ablation_epoch_history.csv"
    latex_file = base_out_dir / "table1_ablation.tex"
    latex_code = latex_file.read_text(encoding="utf-8") if latex_file.exists() else ""

    print("\n" + "=" * 80)
    print("                     ABLATION STUDY COMPLETE")
    print("=" * 80)
    print(f"\nMarkdown Report:    {summary_md_path}")
    print(f"Full JSON Summary:  {summary_json_path}")
    print(f"Epoch History CSV:  {epoch_csv_path}")
    print(f"LaTeX Snippet:      {latex_file}")
    print("\n" + "-" * 30 + " Generated LaTeX Table " + "-" * 30)
    print(latex_code)
    print("-" * 80)

    if any_failed:
        print("[WARNING] One or more configurations failed or were incomplete. Check logs in checkpoint directories.")
        sys.exit(1)


if __name__ == "__main__":
    main()

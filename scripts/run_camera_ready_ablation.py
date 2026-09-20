#!/usr/bin/env python
"""
Automated Ablation Study Runner for NCMMSC 2026 Camera-Ready Table 1.
Fail-Safe Unattended Overnight Execution Edition.

Evaluates four pooling configurations on the clean validation set (Batch 1, zero speaker leakage):
  1. Dual-Branch (Mean + Attn, Proposed)
  2. Dual-Branch (Mean + Top-K)
  3. Single-Branch (Mean Only)
  4. Single-Branch (Attn Only)

Author: WavFry ML Infrastructure Auditor
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
from torch.utils.data import DataLoader

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

# ── 4 Pooling Configurations for Table 1 ─────────────────────────────────────
ABLATION_CONFIGS = [
    {
        "id": "dual_mean_attn",
        "name": "Dual-Branch (Mean + Attn, Proposed)",
        "pooling_mode": "mean_attn",
        "overrides": [
            "training.pooling_mode=mean_attn",
        ],
    },
    {
        "id": "dual_mean_topk",
        "name": "Dual-Branch (Mean + Top-K)",
        "pooling_mode": "mean_topk",
        "overrides": [
            "training.pooling_mode=mean_topk",
            "training.topk_ratio=0.2",
            "training.topk_min_frames=3",
        ],
    },
    {
        "id": "single_mean",
        "name": "Single-Branch (Mean Only)",
        "pooling_mode": "mean",
        "overrides": [
            "training.pooling_mode=mean",
        ],
    },
    {
        "id": "single_attn",
        "name": "Single-Branch (Attn Only)",
        "pooling_mode": "attn",
        "overrides": [
            "training.pooling_mode=attn",
        ],
    },
]


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
    """Run train.py in an isolated subprocess with strict UTF-8 handling and progress streaming."""
    # Determine python execution command
    if shutil.which("uv"):
        cmd = ["uv", "run", "python", "scripts/train.py"]
    else:
        cmd = [sys.executable, "scripts/train.py"]

    cmd.extend(["--config", str(config_path)])
    if resume:
        cmd.append("--resume")
    cmd.extend(["--override"] + overrides)

    print(f"\n[Ablation Runner] Launching training subprocess (full log: {log_file.as_posix()}):")
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

            # Track full tracebacks
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
                # Traceback header/frames ended on error message line
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
    paradigm: str = "segment",
    batch_size: int = 32,
) -> Dict[str, Any]:
    """Load model from checkpoint, run validation pass, and compute metrics at optimal threshold."""
    from wavfry.data import splits as splits_mod
    from wavfry.data import dataset as dataset_mod
    from wavfry.model import wavlm_classifier as model_mod
    from wavfry.train import metrics as metrics_mod

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[Eval] Loading checkpoint from {checkpoint_path}...")
    ckpt = robust_load_checkpoint(checkpoint_path, map_location="cpu")
    config = ckpt["config"]

    # Exclude golden test speakers if marked in manifest (consistency with train.py)
    eval_df = manifest_df.copy()
    if "is_golden" in eval_df.columns:
        is_golden = eval_df["is_golden"].astype(str).str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        ).fillna(False)
        golden_spks = set(eval_df[is_golden]["speaker_id"].dropna().unique())
        if golden_spks:
            eval_df = eval_df[~eval_df["speaker_id"].isin(golden_spks)].copy()

    # Build clean split
    isolate = bool(config["data"].get("isolate_split_speakers", True))
    train_df, val_df, test_df = splits_mod.make_batch_split(eval_df, isolate=isolate)

    print(f"[Eval] Evaluating on Validation split: {len(val_df)} segments across {val_df['speaker_id'].nunique()} unseen speakers.")

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
    else:
        val_ds = dataset_mod.CreakyVowelDataset(
            val_df,
            context_padding_ms=config.get("training", {}).get("segment", {}).get("context_padding_ms", 30),
        )
        collate = dataset_mod.collate_fn

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # safe on Windows
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
                    logits = outputs["interval_logits"][slot_valid].detach().cpu()
                    lbls = labels.to(device)[slot_valid].detach().cpu()

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
                    logits = model(waveforms, lengths)
                    all_logits.append(logits.detach().cpu())
                    all_labels.append(labels.detach().cpu())
                logits_tensor = torch.cat(all_logits).flatten()
                labels_tensor = torch.cat(all_labels).flatten()

    finally:
        # Strict memory cleanup: guarantee ZERO GPU memory leak before next training run
        if model is not None:
            model.to("cpu")
            del model
        del val_loader, val_ds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()

    # Optimal threshold search (consistent step: 0.005 from 0.01 to 0.99)
    probs = torch.sigmoid(logits_tensor).numpy()
    labels_np = labels_tensor.numpy().astype(np.int32)

    thresholds = [round(t, 3) for t in np.arange(0.01, 1.00, 0.005).tolist()]
    from sklearn.metrics import f1_score
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        preds = (probs >= t).astype(np.int32)
        score = float(f1_score(labels_np, preds, average="binary", zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_t = t

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
    """Format Table 1 with bold best and underlined second-best."""
    metrics_keys = ["auc_pr", "f1", "accuracy", "precision", "recall"]

    valid_results = [r for r in results if r.get("status") != "failed" and "metrics" in r]

    # Find highest and second highest unique values for each metric
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
        r"\caption{Ablation study of pooling strategies on the validation set. "
        r"PR-AUC denotes Precision-Recall Area Under Curve. F1, Accuracy, Precision, "
        r"and Recall are evaluated at the optimal decision threshold ($\tau^*$). "
        r"Best results are highlighted in bold, and second-best results are underlined.}"
    )
    lines.append(r"\label{tab:ablation}")
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


def save_intermediate_summaries(summary_results: List[Dict[str, Any]], base_out_dir: Path, paradigm: str):
    """Save raw JSON, Markdown report, and LaTeX table atomically after each configuration."""
    base_out_dir.mkdir(parents=True, exist_ok=True)

    # 1. JSON
    summary_json_path = base_out_dir / "ablation_summary.json"
    tmp_json = summary_json_path.with_suffix(".json.tmp")
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)
    os.replace(tmp_json, summary_json_path)

    # 2. Markdown
    metrics_keys = ["auc_pr", "f1", "accuracy", "precision", "recall"]
    valid_results = [r for r in summary_results if r.get("status") != "failed" and "metrics" in r]
    ranks = {}
    for k in metrics_keys:
        vals = [r["metrics"][k] for r in valid_results if k in r["metrics"] and not math.isnan(r["metrics"][k])]
        unique_vals = sorted(list(set(round(v, 4) for v in vals)), reverse=True)
        ranks[k] = (unique_vals[0] if len(unique_vals) > 0 else -1.0, unique_vals[1] if len(unique_vals) > 1 else -1.0)

    summary_md_path = base_out_dir / "ablation_summary.md"
    tmp_md = summary_md_path.with_suffix(".md.tmp")
    md_lines = [
        "# WavFry Table 1 Pooling Strategy Ablation Results",
        "",
        f"- **Updated**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Paradigm**: `{paradigm}`",
        f"- **Evaluation Split**: Validation Set (Batch 1, $N=3,191$, Unseen Speakers=47)",
        "",
        "| Configuration | PR-AUC | F1 (at $\\tau^*$) | Accuracy | Precision | Recall | Optimal $\\tau^*$ |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in summary_results:
        if r.get("status") == "failed":
            md_lines.append(f"| {r['name']} | *Failed* | *Failed* | *Failed* | *Failed* | *Failed* | -- |")
            continue
        m = r.get("metrics", {})
        cols = []
        for k in ["auc_pr", "f1", "accuracy", "precision", "recall"]:
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
        md_lines.append(f"| {r['name']} | {cols[0]} | {cols[1]} | {cols[2]} | {cols[3]} | {cols[4]} | {best_t_str} |")

    tmp_md.write_text("\n".join(md_lines), encoding="utf-8")
    os.replace(tmp_md, summary_md_path)

    # 3. LaTeX
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

    # Scan for best_ep*_prauc*.pt
    candidates = list(run_ckpt_dir.glob("best_ep*.pt"))
    if candidates:
        def extract_prauc(p: Path) -> float:
            try:
                # Format: best_ep{epoch}_prauc{auc_pr:.4f}_...
                part = p.name.split("_prauc")[1].split("_")[0]
                return float(part)
            except Exception:
                return 0.0
        candidates.sort(key=extract_prauc, reverse=True)
        return candidates[0]

    # Fallback: check for latest_*.pt
    latest_candidates = list(run_ckpt_dir.glob("latest_*.pt"))
    if latest_candidates:
        return latest_candidates[0]

    return None


def main():
    parser = argparse.ArgumentParser(description="Automated Ablation Study for WavFry (Table 1)")
    parser.add_argument("--config", type=str, default="config/_defaults.yaml", help="Path to base config")
    parser.add_argument("--paradigm", type=str, default="segment", choices=["segment", "block"], help="Training paradigm")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs per ablation run")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (0 = disable)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--output-dir", type=str, default="exp_results/runs/camera_ready_ablation", help="TensorBoard/run output dir")
    parser.add_argument("--ckpt-dir", type=str, default="R:/checkpoints/camera_ready_ablation", help="Checkpoint root dir (defaults to R: RAMDisk)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing evaluation results")
    parser.add_argument("--smoketest", action="store_true", help="Quick dry-run: 1 epoch, small data test")
    parser.add_argument("--configs", type=str, nargs="+", default=None, help="Subset of config IDs to run")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Eval device")
    args = parser.parse_args()

    # Engage sleep prevention for unattended overnight execution
    prevent_windows_sleep()

    device = torch.device(args.device)
    config_path = PROJECT_ROOT / args.config
    base_out_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else (PROJECT_ROOT / args.output_dir)
    base_ckpt_dir = Path(args.ckpt_dir) if Path(args.ckpt_dir).is_absolute() else (PROJECT_ROOT / args.ckpt_dir)
    manifest_path = PROJECT_ROOT / "manifests" / "segments_all.csv"

    if not manifest_path.exists():
        print(f"[Error] Manifest {manifest_path} not found. Rebuilding...")
        builder_cmd = ["uv", "run", "python", "scripts/build_manifest.py"] if shutil.which("uv") else [sys.executable, "scripts/build_manifest.py"]
        subprocess.run(builder_cmd, cwd=str(PROJECT_ROOT), check=True)

    manifest_df = pd.read_csv(manifest_path, dtype={"speaker_id": str, "phone": str, "file_key": str})
    base_out_dir.mkdir(parents=True, exist_ok=True)
    base_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Select configs
    target_configs = ABLATION_CONFIGS
    if args.configs:
        target_configs = [c for c in ABLATION_CONFIGS if c["id"] in args.configs]

    print("=" * 80)
    print("      WavFry Camera-Ready Table 1 Ablation Study Suite")
    print(f"  Paradigm: {args.paradigm.upper()} | Epochs: {args.epochs if not args.smoketest else 1} | Device: {device}")
    print(f"  Patience: {args.patience if not args.smoketest else 0} | Grad Accum: {args.grad_accum}")
    print(f"  Target Configurations ({len(target_configs)}): {[c['id'] for c in target_configs]}")
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

            # 1. Check resume: complete result exists
            if args.resume and eval_metrics_file.exists():
                try:
                    with open(eval_metrics_file, "r", encoding="utf-8") as f:
                        metrics = json.load(f)
                    print(f"[Resume] Found existing evaluation metrics at {eval_metrics_file}. Loaded successfully.")
                    summary_results.append({
                        "id": cfg_id,
                        "name": cfg_name,
                        "status": "success",
                        "metrics": metrics,
                    })
                    save_intermediate_summaries(summary_results, base_out_dir, args.paradigm)
                    continue
                except Exception as e:
                    print(f"[Resume Warning] Corrupt eval metrics file {eval_metrics_file}: {e}. Rerunning.")

            # 2. Check resume: training already complete but eval pending
            best_ckpt = resolve_best_checkpoint(run_ckpt_dir)
            skip_training = False
            if args.resume and best_ckpt is not None:
                print(f"[Resume] Existing checkpoint detected: {best_ckpt}. Checking if evaluation can proceed directly...")
                try:
                    ckpt_test = robust_load_checkpoint(best_ckpt, map_location="cpu")
                    # If checkpoint has reached target epochs or early stopped with metrics
                    if "model_state_dict" in ckpt_test:
                        print(f"[Resume] Checkpoint {best_ckpt.name} verified intact. Skipping training pass.")
                        skip_training = True
                    del ckpt_test
                except Exception as e:
                    print(f"[Resume Warning] Checkpoint verification failed: {e}. Will resume/retrain via subprocess.")

            if not skip_training:
                # Prepare overrides
                epochs = 1 if args.smoketest else args.epochs
                batch_size = 32 if args.smoketest else args.batch_size
                patience = 0 if args.smoketest else args.patience

                overrides = [
                    "data.isolate_split_speakers=true",
                    "training.quiet=true",
                    f"training.paradigm={args.paradigm}",
                    f"training.epochs={epochs}",
                    f"training.{args.paradigm}.batch_size={batch_size}",
                    f"training.gradient_accumulation_steps={args.grad_accum}",
                    f"eval.early_stopping_patience={patience}",
                    f"tensorboard.log_dir={run_log_dir.as_posix()}",
                    f"checkpoint.dir={run_ckpt_dir.as_posix()}",
                ] + cfg["overrides"]

                start_time = time.time()
                log_file = run_ckpt_dir / "train.log"

                # Run isolated training pass
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
                    save_intermediate_summaries(summary_results, base_out_dir, args.paradigm)
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
                save_intermediate_summaries(summary_results, base_out_dir, args.paradigm)
                continue

            # 3. Evaluate checkpoint on validation set
            try:
                metrics = evaluate_checkpoint_on_val(
                    checkpoint_path=best_ckpt,
                    manifest_df=manifest_df,
                    device=device,
                    paradigm=args.paradigm,
                    batch_size=args.batch_size,
                )

                with open(eval_metrics_file, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2)

                # Mirror winning checkpoint and metrics to persistent disk as safety backup
                try:
                    local_backup_dir = PROJECT_ROOT / "exp_results" / "checkpoints" / "camera_ready_ablation" / cfg_id
                    local_backup_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(best_ckpt, local_backup_dir / "best_model.pt")
                    shutil.copy2(eval_metrics_file, local_backup_dir / "eval_val_metrics.json")
                    print(f"[Backup] Successfully mirrored best_model.pt to local disk: {local_backup_dir.as_posix()}")
                except Exception as b_err:
                    print(f"[Backup Warning] Could not mirror checkpoint to local disk: {b_err}")

                summary_results.append({
                    "id": cfg_id,
                    "name": cfg_name,
                    "status": "success",
                    "metrics": metrics,
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
                })
            finally:
                # Flush intermediate results to disk after EVERY configuration
                save_intermediate_summaries(summary_results, base_out_dir, args.paradigm)

            # Re-collect memory between configurations
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    finally:
        # Guarantee summary files are saved and sleep prevention is released
        save_intermediate_summaries(summary_results, base_out_dir, args.paradigm)
        restore_windows_sleep()
        set_console_title("WavFry Ablation Suite - Completed")

    summary_json_path = base_out_dir / "ablation_summary.json"
    summary_md_path = base_out_dir / "ablation_summary.md"
    latex_file = base_out_dir / "table1_ablation.tex"
    latex_code = latex_file.read_text(encoding="utf-8") if latex_file.exists() else ""

    print("\n" + "=" * 80)
    print("                     ABLATION STUDY COMPLETE")
    print("=" * 80)
    print(f"\nMarkdown Report: {summary_md_path}")
    print(f"JSON Raw Data:   {summary_json_path}")
    print(f"LaTeX Snippet:   {latex_file}")
    print("\n" + "-" * 30 + " Generated LaTeX Table " + "-" * 30)
    print(latex_code)
    print("-" * 80)

    if any_failed:
        print("[WARNING] One or more configurations failed or were incomplete. Check logs in checkpoint directories.")
        sys.exit(1)


if __name__ == "__main__":
    main()

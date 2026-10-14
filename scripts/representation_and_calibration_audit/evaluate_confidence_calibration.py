#!/usr/bin/env python
"""
WavFry Model Confidence Calibration & Reliability Diagram Evaluator.

Computes:
1. Expected Calibration Error (ECE), Maximum Calibration Error (MCE), and Brier Score.
2. Publication-grade two-panel Reliability Diagrams (Confidence vs. Empirical Accuracy + Histogram).
3. Domain-Shift Calibration Drift Analysis (Clean vs. Noise & Reverb).

Strict Checkpoint Policy:
- --checkpoint is mandatory and strictly validated. No silent fallback or auto-detection.
"""

from __future__ import annotations

import argparse
import csv
import json
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
from torch.utils.data import DataLoader

# Add project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wavfry.data import splits as splits_mod
from wavfry.model.wavlm_classifier import WavFryClassifier
from wavfry.eval.acoustic_degradation import (
    AcousticDegradationEngine,
    PARADIGM_A_CONDITIONS,
    get_acoustic_degradation_engine,
)
from scripts.representation_and_calibration_audit.calibration_core import (
    compute_reliability_diagram_bins,
    format_paradigm_a_summary_table,
    plot_domain_shift_comparison,
    plot_reliability_diagram,
    plot_snr_decay_curves,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="WavFry Confidence Calibration & Reliability Diagram Evaluator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pt). Mandatory; no silent fallback allowed.",
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
        help="Target task to evaluate ('creaky', 'breathy', or 'all').",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["val", "test", "train"],
        default="val",
        help="Data split to evaluate on.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Number of confidence bins for ECE and Reliability Diagrams.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/experiments/confidence_calibration_audit",
        help="Directory to save output reliability diagrams, CSV tables, and audit report.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cuda' or 'cpu'). Defaults to CUDA if available.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit total evaluated target intervals (for debugging or fast evaluation).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Shortcut for --max-samples 300 for rapid smoke testing.",
    )
    parser.add_argument(
        "--eval-domains",
        nargs="+",
        default=["clean"],
        help="List of acoustic domains to evaluate: 'clean', 'cafe_5db', 'reverb', etc.",
    )
    parser.add_argument(
        "--paradigm-a",
        action="store_true",
        help="Execute complete Paradigm A Factorial Matrix (3 Noise Scenarios [Cafe, Car, Street] x 3 SNR levels [15, 5, 0 dB] + Clean & Reverb controls).",
    )
    return parser.parse_args()


def resolve_device(requested_device: Optional[str]) -> torch.device:
    if requested_device:
        dev = torch.device(requested_device)
        print(f"[Device] Using explicitly requested device: {dev}")
        return dev

    if torch.cuda.is_available():
        try:
            free_mem_bytes, total_mem_bytes = torch.cuda.mem_get_info()
            free_mem_mb = free_mem_bytes / (1024 * 1024)
            if free_mem_mb < 1500:
                print(f"[Device Warning] Low GPU VRAM ({free_mem_mb:.1f} MiB free). Falling back to CPU.")
                return torch.device("cpu")
            return torch.device("cuda")
        except Exception:
            return torch.device("cuda")
    return torch.device("cpu")


def load_model(ckpt_path: Path, device: torch.device) -> Tuple[WavFryClassifier, Dict[str, Any]]:
    print(f"[Loading] Validating checkpoint at {ckpt_path}...")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"[FATAL ERROR] Checkpoint file does not exist: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    if not config:
        raise ValueError(f"Checkpoint at {ckpt_path} does not contain 'config'.")

    model = WavFryClassifier(config)
    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
    cleaned_sd = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}

    model.load_state_dict(cleaned_sd, strict=False)
    model.to(device)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    print(f"[Model] Successfully loaded WavFryClassifier on {device}. Active tasks: {model.tasks}")
    return model, config


def prepare_dataloader(
    config: Dict[str, Any],
    manifest_path: Path,
    split_target: str,
    batch_size: int,
) -> Tuple[DataLoader, str]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"[FATAL ERROR] Manifest file does not exist: {manifest_path}")

    df = pd.read_csv(manifest_path, dtype={"speaker_id": str, "phone": str, "file_key": str})
    isolate = bool(config.get("data", {}).get("isolate_split_speakers", True))
    train_df, val_df, test_df = splits_mod.make_batch_split(df, isolate=isolate)

    if split_target == "val":
        eval_df = val_df
    elif split_target == "test":
        eval_df = test_df if not test_df.empty else val_df
    else:
        eval_df = df

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
        from wavfry.data import dataset as dataset_mod
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
    print(f"[Data] Loaded split '{split_target}' ({len(eval_df)} segments, {len(ds)} blocks) using paradigm '{paradigm}'.")
    return loader, paradigm


def evaluate_domain_predictions(
    model: WavFryClassifier,
    loader: DataLoader,
    device: torch.device,
    domain: str,
    eval_tasks: List[str],
    max_samples: Optional[int] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Run forward evaluation under a given acoustic domain, aggregating interval predictions.
    Returns: Dict[task -> (probabilities_array, ground_truth_labels_array)]
    """
    model.eval()
    start_time = time.time()
    task_preds: Dict[str, Dict[str, List[float]]] = {t: {} for t in eval_tasks}

    print(f"[Evaluation] Evaluating domain '{domain}' across tasks {eval_tasks}...")
    engine = get_acoustic_degradation_engine()

    total_evaluated = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            waveforms, bounds, labels, valid, lengths, speakers, block_ids, interval_ids = batch
            waveforms = waveforms.to(device)
            bounds = bounds.to(device)
            labels = labels.to(device)
            valid = valid.to(device)
            lengths = lengths.to(device)

            # Apply domain-specific perturbation via unified AcousticDegradationEngine
            batch_seed = 42 + batch_idx * 1009
            wf_pert = engine.apply_tensor(waveforms, domain, seed=batch_seed)

            # Forward block
            block_out = model(
                wf_pert,
                lengths=lengths,
                interval_bounds=bounds,
                interval_valid=valid,
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
                                store = task_preds[task_name]
                                if i_key not in store:
                                    store[i_key] = [l_val.item(), lab_float, 1]
                                else:
                                    store[i_key][0] += l_val.item()
                                    store[i_key][2] += 1
                                total_evaluated += 1

            if max_samples and (total_evaluated // len(eval_tasks)) >= max_samples:
                print(f"[Evaluation] Reached max sample budget ({max_samples}). Stopping domain '{domain}'.")
                break

    elapsed = time.time() - start_time
    print(f"[Evaluation] Domain '{domain}' completed in {elapsed:.1f}s.")

    # Average logits and compute sigmoid probabilities
    results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for task_name in eval_tasks:
        store = task_preds[task_name]
        probs_list = []
        labels_list = []
        for i_key, (sum_logit, lab, count) in store.items():
            avg_logit = sum_logit / max(1, count)
            prob = 1.0 / (1.0 + np.exp(-avg_logit))
            probs_list.append(prob)
            labels_list.append(lab)
        results[task_name] = (np.array(probs_list, dtype=np.float64), np.array(labels_list, dtype=np.float64))

    return results


def main():
    args = parse_args()

    # 1. Enforce strict checkpoint existence
    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.is_file():
        print(f"[FATAL ERROR] Specified checkpoint does not exist: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = (
        Path(args.manifest).resolve()
        if args.manifest
        else PROJECT_ROOT / "manifests" / "segments_multi.csv"
    )

    max_samples = 300 if args.quick else args.max_samples
    device = resolve_device(args.device)

    # 2. Load model & data
    model, config = load_model(ckpt_path, device)
    loader, paradigm = prepare_dataloader(config, manifest_path, args.split, args.batch_size)

    eval_tasks = ["creaky", "breathy"] if args.task == "all" else [args.task]
    if args.paradigm_a:
        domains = PARADIGM_A_CONDITIONS
        print(f"[Paradigm A] Activated complete 3x3 Factorial Matrix + Controls ({len(domains)} conditions).")
    else:
        domains = args.eval_domains

    all_metrics_rows: List[Dict[str, Any]] = []
    task_domain_bins: Dict[str, Dict[str, Any]] = {t: {} for t in eval_tasks}

    report_md_lines = [
        "# WavFry Model Confidence Calibration & Reliability Audit Report",
        f"- **Checkpoint**: `{ckpt_path.name}`",
        f"- **Split**: `{args.split}` | **Evaluation Paradigm**: `{paradigm}`",
        f"- **Confidence Bins**: `{args.bins}`",
        f"- **Evaluated Domains**: `{', '.join(domains)}`",
        f"- **Protocol**: `{'Paradigm A (3x3 Factorial Matrix + Controls)' if args.paradigm_a else 'Custom Domain Selection'}`",
        f"- **Timestamp**: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        "---",
        "",
    ]

    for domain in domains:
        domain_preds = evaluate_domain_predictions(
            model=model,
            loader=loader,
            device=device,
            domain=domain,
            eval_tasks=eval_tasks,
            max_samples=max_samples,
        )

        for task_name in eval_tasks:
            probs, labels = domain_preds[task_name]
            bin_data = compute_reliability_diagram_bins(probs, labels, num_bins=args.bins)
            task_domain_bins[task_name][domain] = bin_data

            ece = bin_data["ece"]
            mce = bin_data["mce"]
            brier = bin_data["brier_score"]
            n_samples = bin_data["total_samples"]

            all_metrics_rows.append({
                "task": task_name,
                "domain": domain,
                "n_samples": n_samples,
                "ece": round(ece, 5),
                "ece_pct": round(ece * 100, 2),
                "mce": round(mce, 5),
                "mce_pct": round(mce * 100, 2),
                "brier_score": round(brier, 5),
            })

            # Plot single-domain Reliability Diagram for clean condition
            if domain == "clean":
                plot_path = output_dir / f"reliability_diagram_{task_name}_clean"
                plot_reliability_diagram(
                    bin_data=bin_data,
                    title="Model Confidence Calibration",
                    task_name=task_name,
                    output_path=plot_path,
                )

    # 3. Process Paradigm A outputs and Domain Shift comparisons
    for task_name in eval_tasks:
        # Check if any Paradigm A conditions exist
        has_paradigm_a = any(c in task_domain_bins[task_name] for c in ("cafe_15db", "cafe_5db", "car_5db", "street_5db"))
        if has_paradigm_a:
            p_a_table_md, p_a_rows = format_paradigm_a_summary_table(task_domain_bins[task_name], task_name)
            report_md_lines.extend([
                p_a_table_md,
                f"- **SNR-ECE Degradation Curves**: `snr_ece_decay_{task_name}.pdf` / `.png`",
                "",
                "---",
                "",
            ])

            # Export Paradigm A CSV
            pa_csv_path = output_dir / f"paradigm_a_factorial_matrix_{task_name}.csv"
            with open(pa_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["task", "condition", "scenario", "snr_db", "ece_pct", "mce_pct", "brier_score", "n_samples"])
                writer.writeheader()
                writer.writerows(p_a_rows)
            print(f"[Done] Exported Paradigm A Factorial CSV to {pa_csv_path}")

            # Plot publication-grade SNR decay curves
            decay_plot_path = output_dir / f"snr_ece_decay_{task_name}"
            plot_snr_decay_curves(
                domain_results=task_domain_bins[task_name],
                task_name=task_name,
                output_path=decay_plot_path,
            )

        if len(domains) > 1 and not has_paradigm_a:
            comparison_path = output_dir / f"reliability_diagram_domain_shift_{task_name}"
            plot_domain_shift_comparison(
                domain_results=task_domain_bins[task_name],
                task_name=task_name,
                output_path=comparison_path,
            )

        # Markdown report table
        report_md_lines.extend([
            f"## Task: `{task_name.upper()}` Detailed Domain Calibration Metrics",
            "| Domain | Evaluated Samples | ECE (%) | MCE (%) | Brier Score | Reliability Assessment |",
            "| :--- | :---: | :---: | :---: | :---: | :--- |",
        ])
        for row in all_metrics_rows:
            if row["task"] == task_name:
                ece_val = row["ece_pct"]
                grade = "🟢 Well-Calibrated (< 5%)" if ece_val < 5.0 else ("🟡 Moderate (5-10%)" if ece_val < 10.0 else "🔴 Miscalibrated (> 10%)")
                report_md_lines.append(
                    f"| **{row['domain'].capitalize()}** | {row['n_samples']:,} | **{ece_val:.2f}%** | {row['mce_pct']:.2f}% | {row['brier_score']:.4f} | {grade} |"
                )
        report_md_lines.extend([
            "",
            f"- **Clean Reliability Diagram**: `reliability_diagram_{task_name}_clean.pdf` / `.png`",
            f"- **Domain Shift Comparison**: `reliability_diagram_domain_shift_{task_name}.pdf` / `.png`" if (len(domains) > 1 and not has_paradigm_a) else "",
            "",
            "---",
            "",
        ])

    # 4. Save CSV table
    csv_path = output_dir / "ece_calibration_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["task", "domain", "n_samples", "ece", "ece_pct", "mce", "mce_pct", "brier_score"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_metrics_rows)
    print(f"[Done] Exported ECE metrics CSV to {csv_path}")

    # 5. Save Markdown report
    md_path = output_dir / "calibration_audit_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_md_lines))
    print(f"[Done] Exported Calibration Markdown report to {md_path}")


if __name__ == "__main__":
    main()

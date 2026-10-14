#!/usr/bin/env python
"""
WavFry Centered Kernel Alignment (CKA) Representation Similarity Evaluator.

Computes:
1. Intra-model Layer-by-Layer CKA Similarity Matrix (13x13 Heatmap):
   Reveals acoustic representation modularity across CNN frontend (L0) and
   12 Transformer layers (L1-L12).
2. Cross-Condition Acoustic Degradation CKA Stability:
   Computes representation fidelity under Clean vs. Real Noise / Reverb conditions.

Strict Checkpoint Policy:
- --checkpoint is mandatory and strictly validated. No silent fallback or auto-detection.
"""

from __future__ import annotations

import argparse
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader

# Add project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wavfry.data import splits as splits_mod
from wavfry.model.wavlm_classifier import WavFryClassifier
from scripts.representation_and_calibration_audit.cka_core import (
    compute_cross_condition_cka,
    compute_layer_cka_matrix,
    linear_cka,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="WavFry Representation CKA Heatmap and Stability Evaluator",
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
        "--output-dir",
        type=str,
        default="data/experiments/representation_cka_audit",
        help="Directory to save output heatmaps, plots, JSON matrices, and report.",
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
        help="Batch size for feature extraction.",
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
        "--compare-noise",
        action="store_true",
        help="Also extract representations under additive noise to evaluate cross-condition degradation stability.",
    )
    parser.add_argument(
        "--noise-snr-db",
        type=float,
        default=10.0,
        help="SNR level in dB for cross-condition noise evaluation when --compare-noise is set.",
    )
    parser.add_argument(
        "--unbiased",
        action="store_true",
        help="Use unbiased HSIC on Gram matrices (O(N^2)). Default uses fast centered Frobenius norm (O(N*D^2)).",
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

    # Backbone is frozen in evaluation
    for p in model.wavlm.parameters():
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


def add_gaussian_noise(waveforms: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Add scaled Gaussian white noise at specified SNR (dB)."""
    signal_pwr = torch.mean(waveforms ** 2, dim=-1, keepdim=True).clamp_min(1e-9)
    noise_pwr_target = signal_pwr / (10.0 ** (snr_db / 10.0))
    noise = torch.randn_like(waveforms) * torch.sqrt(noise_pwr_target)
    return waveforms + noise


def extract_layer_representations(
    model: WavFryClassifier,
    loader: DataLoader,
    device: torch.device,
    target_task: str,
    max_samples: Optional[int] = None,
    compare_noise: bool = False,
    noise_snr_db: float = 10.0,
) -> Tuple[Dict[int, np.ndarray], Optional[Dict[int, np.ndarray]], int]:
    """
    Extract pooled hidden state vectors for each layer (0 to 12) across target intervals.
    """
    model.eval()
    num_layers = 13  # CNN L0 + 12 Transformer layers

    clean_features: Dict[int, List[np.ndarray]] = {l: [] for l in range(num_layers)}
    noisy_features: Dict[int, List[np.ndarray]] = {l: [] for l in range(num_layers)} if compare_noise else None

    total_samples = 0
    start_time = time.time()
    task_idx = model.tasks.index(target_task) if target_task in model.tasks else 0

    print(f"[Extraction] Extracting representations for task '{target_task}'...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            waveforms, bounds, labels, valid, lengths, speakers, block_ids, interval_ids = batch
            waveforms = waveforms.to(device)
            bounds = bounds.to(device)
            labels = labels.to(device)
            valid = valid.to(device)
            lengths = lengths.to(device)

            wavlm_dtype = next(model.wavlm.parameters()).dtype
            x_in = waveforms.to(wavlm_dtype) if waveforms.dtype != wavlm_dtype else waveforms
            max_len = waveforms.shape[1]
            lengths_clamped = lengths.clamp(min=1, max=max_len)
            attention_mask = (torch.arange(max_len, device=device).unsqueeze(0) < lengths_clamped.unsqueeze(1)).long()

            # Clean forward pass
            outputs_clean = model.wavlm(x_in, attention_mask=attention_mask, output_hidden_states=True)
            clean_states = [h.detach() for h in outputs_clean.hidden_states]  # 13 x [B, T, D]

            # Noisy forward pass (if requested)
            noisy_states = None
            if compare_noise:
                noisy_waveforms = add_gaussian_noise(waveforms, snr_db=noise_snr_db)
                x_noisy = noisy_waveforms.to(wavlm_dtype) if noisy_waveforms.dtype != wavlm_dtype else noisy_waveforms
                outputs_noisy = model.wavlm(x_noisy, attention_mask=attention_mask, output_hidden_states=True)
                noisy_states = [h.detach() for h in outputs_noisy.hidden_states]

            # Extract interval representations
            B = waveforms.shape[0]
            for b in range(B):
                slot_valid_b = valid[b]
                bounds_b = bounds[b]
                labels_b = labels[b]

                for slot_idx in range(len(slot_valid_b)):
                    if not slot_valid_b[slot_idx]:
                        continue

                    # Check if sample is valid for target task
                    if labels_b.ndim > 1 and labels_b.shape[-1] > task_idx:
                        lab_val = labels_b[slot_idx, task_idx].item()
                    else:
                        lab_val = labels_b[slot_idx].item()

                    if np.isnan(lab_val):
                        continue

                    # Frame range
                    start_f = int(bounds_b[slot_idx, 0].item())
                    end_f = int(bounds_b[slot_idx, 1].item())
                    if end_f <= start_f:
                        end_f = start_f + 1

                    # Pool across frames for each layer
                    for l in range(num_layers):
                        h_l_clean = clean_states[l][b, start_f:end_f, :].mean(dim=0).cpu().numpy()
                        clean_features[l].append(h_l_clean)
                        if compare_noise and noisy_states is not None:
                            h_l_noisy = noisy_states[l][b, start_f:end_f, :].mean(dim=0).cpu().numpy()
                            noisy_features[l].append(h_l_noisy)

                    total_samples += 1
                    if max_samples and total_samples >= max_samples:
                        break
                if max_samples and total_samples >= max_samples:
                    break

            if max_samples and total_samples >= max_samples:
                print(f"[Extraction] Reached sample budget cap ({max_samples} samples).")
                break

    elapsed = time.time() - start_time
    print(f"[Extraction] Extracted {total_samples} interval vectors in {elapsed:.1f}s.")

    clean_arrays = {l: np.array(vecs, dtype=np.float64) for l, vecs in clean_features.items()}
    noisy_arrays = (
        {l: np.array(vecs, dtype=np.float64) for l, vecs in noisy_features.items()}
        if compare_noise else None
    )
    return clean_arrays, noisy_arrays, total_samples


def plot_cka_heatmap(
    cka_matrix: np.ndarray,
    layer_names: List[str],
    task_name: str,
    output_path: Path,
) -> None:
    """Render publication-grade CKA 13x13 similarity heatmap."""
    plt.style.use("seaborn-v0_8-white" if "seaborn-v0_8-white" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8.0, 7.2))

    # Clean short tick labels
    short_labels = [name.replace("Transformer_", "") for name in layer_names]

    sns.heatmap(
        cka_matrix,
        xticklabels=short_labels,
        yticklabels=short_labels,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8.0, "fontweight": "normal"},
        cbar_kws={"label": "Linear CKA Similarity"},
        square=True,
        linewidths=0.5,
        linecolor="#e0e0e0",
        ax=ax,
    )

    ax.set_title(
        f"WavFry Representation Similarity Matrix (Linear CKA)\nTask: {task_name.capitalize()} (WavLM 13 Layers)",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Layer Index", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Layer Index", fontsize=11, fontweight="bold", labelpad=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_path.with_suffix(".pdf")
    png_path = output_path.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight", format="pdf")
    fig.savefig(png_path, bbox_inches="tight", dpi=300, format="png")
    plt.close(fig)
    print(f"[Plot] Saved CKA Heatmap to {pdf_path} and {png_path}")


def plot_noise_stability_curve(
    stability_dict: Dict[int, float],
    layer_names: List[str],
    task_name: str,
    snr_db: float,
    output_path: Path,
) -> None:
    """Render layer-wise CKA representation fidelity under acoustic degradation."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    layers = sorted(stability_dict.keys())
    scores = [stability_dict[l] for l in layers]
    short_labels = [layer_names[l].replace("Transformer_", "") for l in layers]

    ax.plot(
        layers,
        scores,
        marker="o",
        color="#2b5c8f",
        linewidth=2.2,
        markersize=7,
        label=f"Clean vs. Noisy (SNR = {snr_db:.0f} dB)",
    )
    ax.axhline(0.8, color="#27ae60", linestyle=":", alpha=0.8, label="High Fidelity Threshold (CKA > 0.8)")

    ax.set_xticks(layers)
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=9.5)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("SSL Layer Index", fontsize=11, fontweight="bold")
    ax.set_ylabel("Representation CKA Stability", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Acoustic Degradation Representation Stability\nTask: {task_name.capitalize()} (Clean vs. {snr_db:.0f}dB Noise)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax.legend(loc="lower left", frameon=True, fontsize=10)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_path.with_suffix(".pdf")
    png_path = output_path.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight", format="pdf")
    fig.savefig(png_path, bbox_inches="tight", dpi=300, format="png")
    plt.close(fig)
    print(f"[Plot] Saved Noise Stability Plot to {pdf_path} and {png_path}")


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
    summary_report: Dict[str, Any] = {
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "max_samples": max_samples,
        "tasks": {},
    }

    report_md_lines = [
        "# WavFry Representation CKA (Centered Kernel Alignment) Audit Report",
        f"- **Checkpoint**: `{ckpt_path.name}`",
        f"- **Split**: `{args.split}` | **Evaluation Paradigm**: `{paradigm}`",
        f"- **Timestamp**: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        "---",
        "",
    ]

    for task_name in eval_tasks:
        print(f"\n==================== Running CKA Audit: Task {task_name.upper()} ====================")
        clean_reps, noisy_reps, n_samples = extract_layer_representations(
            model=model,
            loader=loader,
            device=device,
            target_task=task_name,
            max_samples=max_samples,
            compare_noise=args.compare_noise,
            noise_snr_db=args.noise_snr_db,
        )

        # 3. Compute 13x13 CKA Matrix
        cka_matrix, layer_names = compute_layer_cka_matrix(clean_reps, unbiased=args.unbiased)
        heatmap_path = output_dir / f"cka_layers_heatmap_{task_name}"
        plot_cka_heatmap(cka_matrix, layer_names, task_name, heatmap_path)

        task_summary: Dict[str, Any] = {
            "n_samples": n_samples,
            "layer_names": layer_names,
            "cka_matrix": cka_matrix.tolist(),
        }

        # 4. Optional Noise Stability
        if args.compare_noise and noisy_reps is not None:
            stability_dict = compute_cross_condition_cka(clean_reps, noisy_reps, unbiased=args.unbiased)
            stability_path = output_dir / f"cka_cross_noise_stability_{task_name}"
            plot_noise_stability_curve(stability_dict, layer_names, task_name, args.noise_snr_db, stability_path)
            task_summary["noise_stability"] = {str(k): v for k, v in stability_dict.items()}

        summary_report["tasks"][task_name] = task_summary

        # Build Markdown report section
        report_md_lines.extend([
            f"## Task: `{task_name.upper()}` (Evaluated Samples: {n_samples})",
            "",
            f"- **Heatmap File**: `cka_layers_heatmap_{task_name}.pdf` / `.png`",
            "",
            "### Layer CKA Inter-Correlation (Sample Key Strata)",
            "| Layer Pair | Linear CKA | Acoustic Interpretation |",
            "| :--- | :---: | :--- |",
            f"| **CNN (L0) vs. Transformer L1** | {cka_matrix[0, 1]:.4f} | Local Waveform vs. First Contextual Acoustic State |",
            f"| **Transformer L1 vs. L3** | {cka_matrix[1, 3]:.4f} | Glottal Pulse vs. Harmonic Resonant Structure |",
            f"| **Transformer L1 vs. L12** | {cka_matrix[1, 12]:.4f} | Low-level Acoustic vs. High-level Semantic State |",
            f"| **Transformer L6 vs. L12** | {cka_matrix[6, 12]:.4f} | Intermediate vs. Deep Abstract Representation |",
            "",
        ])

        if args.compare_noise and noisy_reps is not None:
            stab = task_summary.get("noise_stability", {})
            report_md_lines.extend([
                f"### Representation Stability under {args.noise_snr_db:.0f}dB Noise",
                "| Layer | CKA Stability (Clean vs. Noisy) | Fidelity Status |",
                "| :--- | :---: | :--- |",
            ])
            for l in sorted(clean_reps.keys()):
                score = stab.get(str(l), 0.0)
                status = "🟢 High Resilience (CKA > 0.85)" if score > 0.85 else ("🟡 Moderate (0.65-0.85)" if score > 0.65 else "🔴 Severe Degradation (< 0.65)")
                report_md_lines.append(f"| **{layer_names[l]}** | {score:.4f} | {status} |")
            report_md_lines.append("")

        report_md_lines.append("---")
        report_md_lines.append("")

    # Save summary JSON
    json_path = output_dir / "cka_matrices.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)
    print(f"[Done] Exported CKA JSON metrics to {json_path}")

    # Save Markdown report
    md_path = output_dir / "cka_audit_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_md_lines))
    print(f"[Done] Exported CKA Markdown report to {md_path}")


if __name__ == "__main__":
    main()

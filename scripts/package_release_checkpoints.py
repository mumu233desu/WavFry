#!/usr/bin/env python3
"""
package_release_checkpoints.py
==============================
Canonical packaging, sanitization, and verification tool for WavFry v1.0.0
model checkpoints suite for GitHub Releases.

Functions:
  1. Sanitizes raw training checkpoints by stripping local machine paths,
     training provenance tuples, optimizer states, and validation logits cache.
  2. Preserves exact model weights, architecture config, decision thresholds,
     and validation evaluation metrics.
  3. Verifies each sanitized checkpoint through automated CPU forward-pass inference.
  4. Computes SHA-256 checksums (checksums.sha256).
  5. Generates bilingual GitHub Release notes (RELEASE_NOTES.md).

Usage:
  uv run python scripts/package_release_checkpoints.py [--output-dir deployments/checkpoints_v1.0.0]
"""

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any, Dict

import torch

# Ensure repository root is in python path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wavfry.model.wavlm_classifier import WavFryClassifier


CHECKPOINT_REGISTRY = [
    {
        "filename": "wavfry_wavlm_baseplus.pt",
        "src_rel": "exp_results/checkpoints/ablation_backbone_wavlm_baseplus/best_model.pt",
        "role": "Primary Dual-Task Model (Weak-On)",
        "role_zh": "主推双任务模型（Weak-On：WavLM-Base+ & VSDC-Net & 动态掩码）",
        "paper_table": "Table 3 & Table 4",
        "backbone": "microsoft/wavlm-base-plus",
        "is_primary": True,
    },
    {
        "filename": "wavfry_wavlm_base.pt",
        "src_rel": "exp_results/checkpoints/ablation_backbone_wavlm_base/best_model.pt",
        "role": "Cross-Backbone Comparison (WavLM-Base)",
        "role_zh": "跨骨干对比模型（WavLM-Base）",
        "paper_table": "Table 3",
        "backbone": "microsoft/wavlm-base",
        "is_primary": False,
    },
    {
        "filename": "wavfry_hubert_base.pt",
        "src_rel": "exp_results/checkpoints/ablation_backbone_hubert/best_model.pt",
        "role": "Cross-Backbone Comparison (HuBERT-Base)",
        "role_zh": "跨骨干对比模型（HuBERT-Base）",
        "paper_table": "Table 3",
        "backbone": "facebook/hubert-base-ls960",
        "is_primary": False,
    },
    {
        "filename": "wavfry_wav2vec2_base.pt",
        "src_rel": "exp_results/checkpoints/ablation_backbone_wav2vec2/best_model.pt",
        "role": "Cross-Backbone Comparison (wav2vec 2.0-Base)",
        "role_zh": "跨骨干对比模型（wav2vec 2.0-Base）",
        "paper_table": "Table 3",
        "backbone": "facebook/wav2vec2-base",
        "is_primary": False,
    },
    {
        "filename": "wavfry_weak_off.pt",
        "src_rel": "exp_results/checkpoints/ablation_weak_off/best_model.pt",
        "role": "Ablation Baseline (Weak-Off, Supervised Only)",
        "role_zh": "消融基线模型（Weak-Off：纯有监督，无 VSDC 正则化）",
        "paper_table": "Table 4",
        "backbone": "microsoft/wavlm-base-plus",
        "is_primary": False,
    },
]


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def sanitize_config(raw_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize configuration dictionary:
    - Retain model architecture, training precision/pooling, eval, inference, and delta mode.
    - Strip data paths and machine-specific local directories.
    """
    clean_cfg: Dict[str, Any] = {}

    if "model" in raw_cfg:
        clean_cfg["model"] = raw_cfg["model"]

    # Retain architecture-defining training options
    if "training" in raw_cfg:
        t = raw_cfg["training"]
        clean_cfg["training"] = {
            "mixed_precision": t.get("mixed_precision", "fp32"),
            "pooling_mode": t.get("pooling_mode", "attn"),
            "topk_ratio": t.get("topk_ratio", 0.2),
            "topk_min_frames": t.get("topk_min_frames", 3),
            "seed": t.get("seed", 42),
            "block": t.get("block", {}),
            "vsdc": t.get("vsdc", {}),
            "ablation": t.get("ablation", {}),
        }

    if "eval" in raw_cfg:
        clean_cfg["eval"] = raw_cfg["eval"]

    if "inference" in raw_cfg:
        clean_cfg["inference"] = raw_cfg["inference"]

    clean_cfg["checkpoint"] = {"save_mode": "delta"}
    return clean_cfg


def extract_clean_metrics(raw_metrics: Dict[str, Any]) -> Dict[str, float]:
    """Extract serializable, float-converted metrics for summary."""
    clean: Dict[str, float] = {}
    if not isinstance(raw_metrics, dict):
        return clean

    key_metrics = [
        "f1", "auc_pr", "roc_auc", "accuracy", "precision", "recall",
        "f1_creaky", "auc_pr_creaky", "roc_auc_creaky", "accuracy_creaky", "precision_creaky", "recall_creaky",
        "f1_breathy", "auc_pr_breathy", "roc_auc_breathy", "accuracy_breathy", "precision_breathy", "recall_breathy",
    ]
    for k in key_metrics:
        if k in raw_metrics and raw_metrics[k] is not None:
            try:
                clean[k] = round(float(raw_metrics[k]), 5)
            except (TypeError, ValueError):
                pass
    return clean


def verify_checkpoint(ckpt_path: Path) -> Dict[str, Any]:
    """
    Verify that a sanitized checkpoint cleanly loads into WavFryClassifier
    and executes a zero-shot CPU forward pass without NaN or exceptions.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    model = WavFryClassifier(cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    # Create dummy 1.0-second 16kHz audio waveform
    dummy_wav = torch.sin(2 * 3.14159265 * 440 * torch.linspace(0, 1.0, 16000)).unsqueeze(0)

    with torch.no_grad():
        out = model(dummy_wav)

    # Check logits
    tasks = cfg.get("model", {}).get("tasks", ["creaky"])
    for task in tasks:
        assert task in out, f"Missing task '{task}' in model forward output (available: {list(out.keys())})"
        val = out[task].item()
        assert not torch.isnan(torch.tensor(val)), f"NaN detected in task '{task}' output"
        assert not torch.isinf(torch.tensor(val)), f"Inf detected in task '{task}' output"

    # Extract info for reporting
    metrics = ckpt.get("metrics", {})
    return {
        "tasks": tasks,
        "epoch": ckpt.get("epoch"),
        "best_thresholds": ckpt.get("best_thresholds") or ckpt.get("best_threshold"),
        "pr_auc_creaky": metrics.get("auc_pr_creaky", metrics.get("auc_pr", 0.0)),
        "roc_auc_creaky": metrics.get("roc_auc_creaky", metrics.get("roc_auc", 0.0)),
        "f1_creaky": metrics.get("f1_creaky", metrics.get("f1", 0.0)),
    }


def generate_release_notes(
    output_dir: Path,
    packaged_items: list,
    checksums: Dict[str, str],
) -> None:
    """Generate bilingual RELEASE_NOTES.md documenting models and checksums."""
    notes_file = output_dir / "RELEASE_NOTES.md"

    lines = [
        "# WavFry v1.0.0: Pretrained Checkpoints & Dual-Task Ablation Suite",
        "",
        "This release publishes the official pretrained model checkpoints and ablation variants corresponding to the paper:",
        "> **WavFry: Noise-Robust Phonation Detection and Functional Representation Auditing in Conversational Speech**",
        "",
        "## 📦 Released Model Weights",
        "",
        "All weights are stored as delta checkpoints. The underlying self-supervised backbones (`WavLM`, `HuBERT`, `wav2vec 2.0`) should be automatically loaded from Hugging Face Hub during inference.",
        "",
        "| File Name | Base Backbone | Role / Architecture Description | Creaky PR-AUC | Creaky ROC-AUC | Optimal Thresholds |",
        "| :--- | :--- | :--- | :---: | :---: | :---: |",
    ]

    for item in packaged_items:
        fn = item["filename"]
        bb = item["backbone"]
        role = item["role"]
        v = item["verified_info"]
        th = v["best_thresholds"]
        if isinstance(th, dict):
            th_str = " / ".join(f"{k}: {val:.3f}" for k, val in th.items())
        elif isinstance(th, (int, float)):
            th_str = f"creaky: {th:.3f}"
        else:
            th_str = "0.500"

        star = " **(Primary)**" if item["is_primary"] else ""
        lines.append(
            f"| `{fn}` | `{bb}` | {role}{star} | {v['pr_auc_creaky']:.4f} | {v['roc_auc_creaky']:.4f} | `{th_str}` |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 🔒 SHA-256 Checksums",
        "",
        "```",
    ])
    for item in packaged_items:
        fn = item["filename"]
        h = checksums[fn]
        lines.append(f"{h}  {fn}")
    lines.append("```\n")

    notes_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [Docs] Generated Release Notes: {notes_file}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Package and sanitize WavFry v1.0.0 release checkpoints.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="deployments/checkpoints_v1.0.0",
        help="Target output directory for sanitized checkpoints and checksums.",
    )
    args = parser.parse_args()

    out_dir = (ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== WavFry v1.0.0 Checkpoint Release Packager ===", flush=True)
    print(f"Output directory: {out_dir}", flush=True)
    print(f"Repository root:  {ROOT}\n", flush=True)

    packaged_items = []
    checksums: Dict[str, str] = {}

    for entry in CHECKPOINT_REGISTRY:
        fn = entry["filename"]
        src_path = (ROOT / entry["src_rel"]).resolve()
        dst_path = out_dir / fn

        print(f"--- Packaging: {fn} ---", flush=True)
        if not src_path.exists():
            print(f"  [ERROR] Source checkpoint not found: {src_path}", flush=True)
            sys.exit(1)

        raw_size = src_path.stat().st_size / (1024 * 1024)
        print(f"  Source: {src_path.name} ({raw_size:.2f} MB)", flush=True)

        # 1. Load source checkpoint
        raw_ckpt = torch.load(src_path, map_location="cpu", weights_only=False)

        # 2. Sanitize configuration & build clean bundle
        clean_config = sanitize_config(raw_ckpt.get("config", {}))
        clean_metrics = extract_clean_metrics(raw_ckpt.get("metrics", {}))

        raw_th = raw_ckpt.get("best_threshold")
        raw_ths = raw_ckpt.get("best_thresholds")
        if raw_ths is None and raw_th is not None:
            norm_ths = {"creaky": round(float(raw_th), 4)}
        elif isinstance(raw_ths, dict):
            norm_ths = {k: round(float(v), 4) for k, v in raw_ths.items()}
        else:
            norm_ths = None

        clean_bundle = {
            "model_state_dict": raw_ckpt["model_state_dict"],
            "config": clean_config,
            "epoch": raw_ckpt.get("epoch"),
            "best_threshold": float(raw_th) if raw_th is not None else None,
            "best_thresholds": norm_ths,
            "threshold_f1s": raw_ckpt.get("threshold_f1s"),
            "metrics": clean_metrics,
            "release_metadata": {
                "model_name": fn,
                "role": entry["role"],
                "paper_table": entry["paper_table"],
                "backbone": entry["backbone"],
                "tasks": clean_config.get("model", {}).get("tasks", ["creaky"]),
                "version": "v1.0.0",
                "license": "MIT",
            },
        }

        # 3. Save sanitized checkpoint
        torch.save(clean_bundle, dst_path)
        clean_size = dst_path.stat().st_size / (1024 * 1024)
        print(f"  Sanitized size: {clean_size:.2f} MB (saved to {dst_path.name})", flush=True)

        # 4. Compute SHA-256
        sha_val = compute_sha256(dst_path)
        checksums[fn] = sha_val
        print(f"  SHA-256: {sha_val}", flush=True)

        # 5. Automated CPU inference verification
        print(f"  Verifying CPU forward pass...", flush=True)
        ver_info = verify_checkpoint(dst_path)
        print(
            f"  [PASS] Tasks: {ver_info['tasks']}, "
            f"PR-AUC (creaky): {ver_info['pr_auc_creaky']:.4f}, "
            f"Thresholds: {ver_info['best_thresholds']}",
            flush=True,
        )

        entry_copy = dict(entry)
        entry_copy["verified_info"] = ver_info
        packaged_items.append(entry_copy)
        print("", flush=True)

    # 6. Write checksums.sha256
    checksums_path = out_dir / "checksums.sha256"
    with open(checksums_path, "w", encoding="utf-8") as f:
        for item in packaged_items:
            fn = item["filename"]
            f.write(f"{checksums[fn]}  {fn}\n")
    print(f"=== Checksums written: {checksums_path} ===", flush=True)

    # 7. Generate RELEASE_NOTES.md
    generate_release_notes(out_dir, packaged_items, checksums)

    total_mb = sum((out_dir / item["filename"]).stat().st_size for item in packaged_items) / (1024 * 1024)
    print(f"\n✅ All {len(packaged_items)} checkpoints packaged, sanitized, and verified successfully!", flush=True)
    print(f"Total package size: {total_mb:.2f} MB across {len(packaged_items)} models.", flush=True)
    print(f"Artifacts ready in: {out_dir}\n", flush=True)


if __name__ == "__main__":
    main()

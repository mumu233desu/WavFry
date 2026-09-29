#!/usr/bin/env python
"""
Evaluate WavFry Single-Attn Checkpoint on the Unified Test Set (Batch 2 + Batch 3).
Evaluates Clean + 6 Degraded Environments (SNR 20/10/0/-10/-20 dB, Reverb RT60=0.5s)
and computes overall + demographic subgroup metrics (Female/Male, Beijing/Guangdong/Sichuan).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, f1_score, precision_recall_curve, precision_score, recall_score, accuracy_score
from torch.utils.data import DataLoader

# Project setup
CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from wavfry.data import dataset as dataset_mod
from wavfry.model import wavlm_classifier as model_mod

WAVE_ROOT = Path(r"E:\Corpus\CS-Dialogue\data\short_wav\WAVE")
BASELINES_DIR = Path(r"E:\Coding\WavFry\data\experiments\baselines")
DEFAULT_CKPT = Path(r"R:\checkpoints\ncmmsc_camera_ready\best_model.pt")
FALLBACK_CKPT = Path(r"E:\Corpus\CS-Dialogue\data\short_wav\WAVE\06_wavfry\exp_results\checkpoints\camera_ready_ablation\single_attn\best_model.pt")
CHECKPOINT_PATH = DEFAULT_CKPT if DEFAULT_CKPT.exists() else FALLBACK_CKPT

ENVIRONMENTS = [
    {"name": "Clean", "dir": WAVE_ROOT / "01_en_dereverb"},
    {"name": "20dB", "dir": WAVE_ROOT / "01_en_dereverb_SNR20"},
    {"name": "10dB", "dir": WAVE_ROOT / "01_en_dereverb_SNR10"},
    {"name": "0dB", "dir": WAVE_ROOT / "01_en_dereverb_SNR0"},
    {"name": "-10dB", "dir": WAVE_ROOT / "01_en_dereverb_SNR-10"},
    {"name": "-20dB", "dir": WAVE_ROOT / "01_en_dereverb_SNR-20"},
    {"name": "Rev(0.5s)", "dir": WAVE_ROOT / "01_en_dereverb_Reverb"},
]

def parse_gender(file_key: str) -> str:
    parts = str(file_key).split("_")
    return parts[0] if len(parts) >= 1 else ""

def parse_region(file_key: str) -> str:
    parts = str(file_key).split("_")
    return parts[2] if len(parts) >= 3 else ""

def compute_metrics_at_t(probs: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = (probs >= threshold).astype(np.int32)
    acc = float(accuracy_score(labels, preds))
    p = float(precision_score(labels, preds, zero_division=0))
    r = float(recall_score(labels, preds, zero_division=0))
    f1 = float(f1_score(labels, preds, zero_division=0))
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

def compute_prauc(probs: np.ndarray, labels: np.ndarray) -> float:
    p_curve, r_curve, _ = precision_recall_curve(labels, probs)
    return float(auc(r_curve, p_curve))

def compute_subgroup_metrics(df: pd.DataFrame, probs: np.ndarray, threshold: float) -> Dict[str, Any]:
    """Compute per-speaker macro-averaged metrics across demographic subgroups."""
    work = df.copy()
    work["prob"] = probs
    work["pred"] = (probs >= threshold).astype(np.int32)
    work["gold"] = work["label"].astype(np.int32)

    spk_metrics = []
    for spk_id, gdf in work.groupby("speaker_id"):
        y_true = gdf["gold"].to_numpy()
        y_pred = gdf["pred"].to_numpy()
        spk_metrics.append({
            "speaker_id": spk_id,
            "gender": gdf["gender"].iloc[0],
            "region": gdf["region"].iloc[0],
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        })

    spk_df = pd.DataFrame(spk_metrics)

    results = {
        "overall": {
            "accuracy": float(spk_df["accuracy"].mean()),
            "precision": float(spk_df["precision"].mean()),
            "recall": float(spk_df["recall"].mean()),
            "f1": float(spk_df["f1"].mean()),
        }
    }

    for gender in ["F", "M"]:
        sub = spk_df[spk_df["gender"] == gender]
        key = "Female" if gender == "F" else "Male"
        results[key] = {
            "accuracy": float(sub["accuracy"].mean()),
            "precision": float(sub["precision"].mean()),
            "recall": float(sub["recall"].mean()),
            "f1": float(sub["f1"].mean()),
        }

    for region in ["Beijing", "Guangdong", "Sichuan"]:
        sub = spk_df[spk_df["region"] == region]
        results[region] = {
            "accuracy": float(sub["accuracy"].mean()),
            "precision": float(sub["precision"].mean()),
            "recall": float(sub["recall"].mean()),
            "f1": float(sub["f1"].mean()),
        }

    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate WavFry on Unified Test Set")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH, help="Path to best_model.pt checkpoint")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    print("=" * 70)
    print(f"  WavFry Camera-Ready Unified Test Set Evaluation")
    print(f"  Checkpoint: {checkpoint_path}")
    print("=" * 70)

    # 1. Load manifest and isolate test set
    manifest_path = CODE_ROOT / "manifests" / "segments_all.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    full_df = pd.read_csv(manifest_path, low_memory=False)
    is_cs = (full_df["dataset_id"] == "cs_dialogue") if "dataset_id" in full_df.columns else full_df["label_creaky"].notna()
    if "batch_split" in full_df.columns:
        test_df = full_df[(full_df["batch_split"] == "test") & is_cs].copy().reset_index(drop=True)
    elif "split" in full_df.columns:
        test_df = full_df[(full_df["split"] == "test") & is_cs].copy().reset_index(drop=True)
    else:
        b0_spks = set(full_df[full_df["batch_id"] == 0]["speaker_id"].unique())
        test_df = full_df[full_df["batch_id"].isin([2, 3]) & (~full_df["speaker_id"].isin(b0_spks)) & is_cs].copy().reset_index(drop=True)

    if "phone_type" in test_df.columns:
        test_df = test_df[test_df["phone_type"] == "vowel"].copy().reset_index(drop=True)

    test_df["gender"] = test_df["file_key"].apply(parse_gender)
    test_df["region"] = test_df["file_key"].apply(parse_region)

    print(f"Unified Test Set (Vowel Only): {len(test_df)} segments across {test_df['speaker_id'].nunique()} unseen speakers, {test_df['file_key'].nunique()} audio files.")

    # 2. Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from {checkpoint_path} on {device}...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    if "best_threshold" not in ckpt:
        raise RuntimeError(
            f"Checkpoint '{checkpoint_path}' has no frozen 'best_threshold'. "
            "This field is injected by the trainer after validation-set threshold grid search. "
            "Refusing to silently fall back to an arbitrary constant — evaluate with a "
            "checkpoint produced by the current training pipeline."
        )
    best_threshold = float(ckpt["best_threshold"])
    print(f"Loaded config: fine_tune={config['model'].get('fine_tune_layers')}, pooling={config['training'].get('pooling_mode')}")
    print(f"Frozen decision threshold from validation set: tau* = {best_threshold:.3f}")

    model = model_mod.WavFryClassifier(config)
    save_mode = config["checkpoint"].get("save_mode", "delta")
    if save_mode == "delta":
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    results: Dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "test_segments": len(test_df),
        "test_speakers": int(test_df["speaker_id"].nunique()),
        "test_files": int(test_df["file_key"].nunique()),
        "frozen_threshold": best_threshold,
        "environments": {},
    }

    labels_all = test_df["label"].to_numpy().astype(np.int32)

    for env in ENVIRONMENTS:
        env_name = env["name"]
        audio_dir = env["dir"]
        print(f"\n--- Evaluating Condition: {env_name} (audio_dir: {audio_dir.name}) ---")

        if not audio_dir.exists():
            print(f"[Warning] Audio dir {audio_dir} does not exist! Skipping.")
            continue

        # Map audio path
        env_df = test_df.copy()
        def resolve_path(k: str) -> str:
            w = audio_dir / f"{k}.wav"
            if w.exists(): return str(w)
            f = audio_dir / f"{k}.flac"
            if f.exists(): return str(f)
            return ""
        env_df["audio_path"] = env_df["file_key"].apply(resolve_path)
        missing = (env_df["audio_path"] == "").sum()
        if missing > 0:
            print(f"[Warning] Missing {missing} audio files in {audio_dir}!")

        # Create dataset & loader
        ds = dataset_mod.CreakyVowelDataset(
            env_df,
            context_padding_ms=config.get("training", {}).get("segment", {}).get("context_padding_ms", 30),
        )
        loader = DataLoader(
            ds,
            batch_size=64,
            shuffle=False,
            num_workers=0,
            collate_fn=dataset_mod.collate_fn,
            pin_memory=True if device.type == "cuda" else False,
        )

        all_logits = []
        with torch.no_grad():
            for waveforms, _, lengths, _, _ in loader:
                waveforms = waveforms.to(device)
                lengths = lengths.to(device)
                outputs = model(waveforms, lengths)
                logits = outputs["creaky"] if isinstance(outputs, dict) else outputs
                all_logits.append(logits.detach().cpu())

        logits_tensor = torch.cat(all_logits).flatten()
        probs = torch.sigmoid(logits_tensor).numpy()

        prauc = compute_prauc(probs, labels_all)
        metrics_best_t = compute_metrics_at_t(probs, labels_all, best_threshold)
        metrics_default_t = compute_metrics_at_t(probs, labels_all, 0.5)

        print(f"[{env_name}] PR-AUC: {prauc:.4f} | At tau*={best_threshold:.3f}: F1={metrics_best_t['f1']:.4f}, Acc={metrics_best_t['accuracy']:.4f}, P={metrics_best_t['precision']:.4f}, R={metrics_best_t['recall']:.4f}")
        print(f"[{env_name}] At tau =0.500: F1={metrics_default_t['f1']:.4f}, Acc={metrics_default_t['accuracy']:.4f}, P={metrics_default_t['precision']:.4f}, R={metrics_default_t['recall']:.4f}")

        env_result = {
            "pr_auc": prauc,
            "metrics_best_threshold": metrics_best_t,
            "metrics_default_threshold": metrics_default_t,
        }

        # If Clean, also compute subgroup metrics
        if env_name == "Clean":
            subgroups = compute_subgroup_metrics(test_df, probs, best_threshold)
            env_result["subgroups"] = subgroups
            print(f"Subgroups (Macro Avg F1): Overall={subgroups['overall']['f1']:.4f}, Female={subgroups['Female']['f1']:.4f}, Male={subgroups['Male']['f1']:.4f}, Beijing={subgroups['Beijing']['f1']:.4f}, Guangdong={subgroups['Guangdong']['f1']:.4f}, Sichuan={subgroups['Sichuan']['f1']:.4f}")

        results["environments"][env_name] = env_result

    # Save to json
    out_json = BASELINES_DIR / "wavfry_single_attn_eval_results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Done] All WavFry evaluation results saved to: {out_json}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
WavFry Macro-Prosodic Control Baseline System.
Constructs a tiered prosodic baseline using CREPE-estimated pitch (F0),
periodicity, RMS energy contour, duration, and phonetic vowel identity.

Answers the fundamental phonetics critique:
    "Does the SSL representation provide orthogonal voice-quality information
     beyond macro-prosodic cues (e.g. low pitch, sentence-final pitch fall, duration)?"

Classifiers:
    1. Logistic Regression (Strict linear capacity alignment with SSL linear probing)
    2. Gradient Boosting (GBDT, bounding non-linear prosodic interactions)

Data Storage:
    - Features: data/features/prosodic_features.csv (Human-readable CSV format)
    - Results:  data/experiments/prosodic_baseline_results.csv
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*Support for mismatched key_padding_mask and attn_mask is deprecated.*")
warnings.filterwarnings("ignore", message=".*torchaudio.backend.common.AudioMetaData.*")
warnings.filterwarnings("ignore", message=".*No positive class found in y_true.*")
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import yaml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc, f1_score, roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Add project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wavfry.data import splits as splits_mod


def parse_args():
    parser = argparse.ArgumentParser(description="WavFry Macro-Prosodic Baseline System")
    parser.add_argument(
        "--config",
        type=str,
        default="config/experiments/frozen_wavlm_base_plus.yaml",
        help="Path to YAML config to inherit data split rules from.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to manifest CSV. Defaults to manifests/segments_multi.csv.",
    )
    parser.add_argument(
        "--features-cache",
        type=str,
        default="data/features/prosodic_features.csv",
        help="Path to save or load extracted prosodic feature CSV table.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/experiments/prosodic_baseline_results.csv",
        help="Path to save baseline evaluation metrics CSV.",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Force re-extraction of acoustic features even if cache CSV exists.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for CREPE inference ('cuda' or 'cpu'). Defaults to auto.",
    )
    parser.add_argument(
        "--smoketest",
        action="store_true",
        help="Run quick smoke test on first 32 samples.",
    )
    return parser.parse_args()


# ── Feature Extraction Functions ─────────────────────────────────────────────

def extract_single_prosodic_features(
    audio_path: str,
    start_s: float,
    end_s: float,
    context_padding_ms: float = 30.0,
    device: str = "cpu",
) -> Dict[str, float]:
    """Extract CREPE pitch contour, periodicity, RMS energy, and duration from audio segment."""
    import torchcrepe

    pad_s = context_padding_ms / 1000.0
    seg_dur_ms = max((end_s - start_s) * 1000.0, 1.0)

    try:
        # Read with context padding to prevent boundary truncation
        info = sf.info(audio_path)
        sr = info.samplerate
        total_dur = info.duration

        read_start = max(0.0, start_s - pad_s)
        read_end = min(total_dur, end_s + pad_s)
        start_frame = int(read_start * sr)
        num_frames = max(1, int((read_end - read_start) * sr))

        wav, _ = sf.read(audio_path, start=start_frame, stop=start_frame + num_frames, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
    except Exception as e:
        # Fallback empty metrics on read failure
        return {
            "duration_ms": seg_dur_ms,
            "f0_mean": 0.0, "f0_std": 0.0, "f0_min": 0.0, "f0_max": 0.0, "f0_slope": 0.0,
            "periodicity_mean": 0.0, "periodicity_min": 0.0, "periodicity_low_ratio": 1.0,
            "energy_rms_mean": 0.0, "energy_rms_std": 0.0, "energy_rms_slope": 0.0,
        }

    # 1. Resample to 16kHz if needed for CREPE
    t_wav = torch.from_numpy(wav).unsqueeze(0)
    if sr != 16000:
        import torchaudio.transforms as T
        resampler = T.Resample(orig_freq=sr, new_freq=16000)
        t_wav = resampler(t_wav)
        sr = 16000

    # 2. RMS Energy calculation (25ms window, 10ms hop)
    hop_len = int(0.010 * sr)
    win_len = int(0.025 * sr)
    if t_wav.size(-1) >= win_len:
        unfolded = t_wav.unfold(-1, win_len, hop_len)
        rms = torch.sqrt((unfolded ** 2).mean(dim=-1) + 1e-9).squeeze(0).numpy()
    else:
        rms = np.array([float(torch.sqrt((t_wav ** 2).mean() + 1e-9))])

    energy_mean = float(np.mean(rms))
    energy_std = float(np.std(rms))
    energy_min = float(np.min(rms))
    energy_max = float(np.max(rms))
    energy_slope = float(rms[-1] - rms[0]) if len(rms) > 1 else 0.0

    # 3. CREPE Pitch & Periodicity Estimation
    # Hop length: 160 samples (10ms at 16kHz)
    # Range: 50 Hz to 550 Hz (human vocal range)
    try:
        pitch, periodicity = torchcrepe.predict(
            t_wav,
            sr,
            hop_length=160,
            fmin=50,
            fmax=550,
            model="tiny",
            decoder="viterbi",
            return_periodicity=True,
            device=device,
            batch_size=2048,
        )
        pitch_np = pitch.squeeze(0).cpu().numpy()
        per_np = periodicity.squeeze(0).cpu().numpy()
    except Exception:
        pitch_np = np.zeros(1)
        per_np = np.zeros(1)

    # Filter voiced frames using periodicity threshold >= 0.2
    voiced_mask = per_np >= 0.2
    if np.any(voiced_mask):
        voiced_pitch = pitch_np[voiced_mask]
        f0_mean = float(np.mean(voiced_pitch))
        f0_std = float(np.std(voiced_pitch))
        f0_min = float(np.min(voiced_pitch))
        f0_max = float(np.max(voiced_pitch))
        f0_slope = float(voiced_pitch[-1] - voiced_pitch[0]) if len(voiced_pitch) > 1 else 0.0
    else:
        f0_mean = 0.0
        f0_std = 0.0
        f0_min = 0.0
        f0_max = 0.0
        f0_slope = 0.0

    per_mean = float(np.mean(per_np))
    per_min = float(np.min(per_np))
    per_low_ratio = float(np.mean(per_np < 0.25))  # Fraction of low periodicity frames (fry indicator)

    return {
        "duration_ms": round(seg_dur_ms, 2),
        "f0_mean": round(f0_mean, 2),
        "f0_std": round(f0_std, 2),
        "f0_min": round(f0_min, 2),
        "f0_max": round(f0_max, 2),
        "f0_slope": round(f0_slope, 2),
        "periodicity_mean": round(per_mean, 4),
        "periodicity_min": round(per_min, 4),
        "periodicity_low_ratio": round(per_low_ratio, 4),
        "energy_rms_mean": round(energy_mean, 5),
        "energy_rms_std": round(energy_std, 5),
        "energy_rms_slope": round(energy_slope, 5),
    }


def build_or_load_prosodic_features(
    manifest_df: pd.DataFrame,
    cache_path: Path,
    device: str = "cpu",
    force_extract: bool = False,
    is_smoketest: bool = False,
) -> pd.DataFrame:
    """Build or load the prosodic feature table."""
    if cache_path.exists() and not force_extract and not is_smoketest:
        print(f"[Features] Loading cached prosodic feature table from: {cache_path}")
        feat_df = pd.read_csv(cache_path)
        # Identity check: cached feature rows must correspond 1:1, in order, to
        # the current manifest via (file_key, start_s). Positional merges have
        # silently mismatched features and labels whenever the manifest was
        # rebuilt or re-split; refuse instead.
        key_cols = [c for c in ("file_key", "start_s") if c in feat_df.columns and c in manifest_df.columns]
        if len(key_cols) < 2:
            raise RuntimeError(
                "Prosodic feature cache lacks identity columns (file_key, start_s); "
                "regenerate it with --force-extract."
            )
        manifest_keys = manifest_df[key_cols].astype(str).agg("|".join, axis=1).reset_index(drop=True)
        feat_keys = feat_df[key_cols].astype(str).agg("|".join, axis=1).reset_index(drop=True)
        if len(manifest_keys) != len(feat_keys) or not manifest_keys.equals(feat_keys):
            raise RuntimeError(
                f"Cached prosodic features ({len(feat_keys)} rows) do not match the "
                f"current manifest ({len(manifest_keys)} rows) by file_key+start_s. "
                "Re-run with --force-extract to rebuild the cache against the "
                "current manifest — stale caches otherwise pair old features with "
                "new labels/splits."
            )
        # Identity verified: refresh ALL manifest metadata (labels, splits,
        # speakers) from the current manifest so stale cached splits cannot leak.
        for col in manifest_df.columns:
            feat_df[col] = manifest_df[col].values
        return feat_df

    if is_smoketest:
        # Pick 16 from train speakers and 16 from val speakers (e.g. annotation_type == 'sampled')
        train_slice = manifest_df[manifest_df["speaker_id"].isin(["U1001", "U0046"])].head(16)
        val_slice = manifest_df[manifest_df["annotation_type"] == "sampled"].head(16)
        if len(val_slice) == 0:
            val_slice = manifest_df.tail(16)
        target_df = pd.concat([train_slice, val_slice]).reset_index(drop=True)
    else:
        target_df = manifest_df
    total = len(target_df)
    print(f"\n[Extraction] Extracting prosodic features for {total} segments (Device: {device})...")

    rows = []
    t0 = time.time()

    for idx, row in target_df.iterrows():
        f_dict = extract_single_prosodic_features(
            audio_path=row["audio_path"],
            start_s=float(row["start_s"]),
            end_s=float(row["end_s"]),
            context_padding_ms=30.0,
            device=device,
        )
        for col in manifest_df.columns:
            if col not in f_dict:
                f_dict[col] = row[col]
        rows.append(f_dict)

        if (len(rows) % 200 == 0) or (len(rows) == total):
            elapsed = time.time() - t0
            rate = len(rows) / max(elapsed, 0.1)
            eta = (total - len(rows)) / max(rate, 0.1)
            print(f"  Processed {len(rows)}/{total} ({len(rows)/total*100:.1f}%) - {rate:.1f} seg/s - ETA: {eta:.0f}s", flush=True)

    out_df = pd.DataFrame(rows)
    if not is_smoketest:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(cache_path, index=False)
        print(f"[Saved] Persisted prosodic features to: {cache_path}")

    return out_df


# ── Model Training & Evaluation ──────────────────────────────────────────────

def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float | None = None,
) -> Dict[str, float]:
    """Calculate PR-AUC, ROC-AUC, F1, Precision, and Recall.

    Protocol: pass ``threshold`` (frozen on the validation split) for test-set
    reporting. Only when ``threshold`` is None (i.e. called on validation data
    itself) is the F1-optimal threshold scanned on the provided labels.
    Scanning the threshold on test data is test-set tuning and inflates F1.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = 0.5

    if threshold is None:
        f1_scores = 2 * (precision * recall) / np.clip(precision + recall, 1e-9, None)
        best_idx = np.nanargmax(f1_scores)
        best_f1 = float(f1_scores[best_idx])
        best_thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    else:
        best_thresh = float(threshold)

    y_pred = (y_prob >= best_thresh).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1_at_t = f1_score(y_true, y_pred, zero_division=0)

    return {
        "pr_auc": round(float(pr_auc), 4),
        "f1": round(float(f1_at_t), 4),
        "roc_auc": round(float(roc_auc), 4),
        "accuracy": round(float(acc), 4),
        "best_threshold": round(best_thresh, 4),
    }


def main():
    args = parse_args()
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    config_path = PROJECT_ROOT / args.config
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    manifest_path = Path(args.manifest) if args.manifest else (PROJECT_ROOT / "manifests" / "segments_multi.csv")
    if not manifest_path.exists():
        manifest_path = PROJECT_ROOT / "manifests" / "segments_all.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest_df = pd.read_csv(manifest_path, dtype={"speaker_id": str, "phone": str, "file_key": str})
    cache_path = Path(args.features_cache) if Path(args.features_cache).is_absolute() else (PROJECT_ROOT / args.features_cache)

    feat_df = build_or_load_prosodic_features(
        manifest_df=manifest_df,
        cache_path=cache_path,
        device=device,
        force_extract=args.force_extract,
        is_smoketest=args.smoketest,
    )

    # Exclude rows without this task's label BEFORE splitting: NaN labels
    # int-cast into garbage classes would poison training and every metric.
    label_col = "label_creaky" if "label_creaky" in feat_df.columns else "label"
    labels_num = pd.to_numeric(feat_df[label_col], errors="coerce")
    n_nan = int(labels_num.isna().sum())
    if n_nan:
        print(f"[Labels] Excluding {n_nan} rows with NaN '{label_col}' labels (unlabeled for this task).")
        feat_df = feat_df[labels_num.notna()].reset_index(drop=True)

    # Partition dataset dynamically according to configuration
    isolate = bool(cfg.get("data", {}).get("isolate_split_speakers", True))
    train_df, val_df, test_df = splits_mod.make_batch_split(feat_df, isolate=isolate)

    print(f"\n[Split] Dataset Partition: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Feature sets
    feature_cols = [
        "duration_ms",
        "f0_mean", "f0_std", "f0_min", "f0_max", "f0_slope",
        "periodicity_mean", "periodicity_min", "periodicity_low_ratio",
        "energy_rms_mean", "energy_rms_std", "energy_rms_slope",
    ]

    # Add vowel one-hot indicators if phone column exists
    top_vowels = ["aa", "ae", "ah", "ao", "eh", "er", "ey", "ih", "iy", "ow", "uw"]
    for v in top_vowels:
        col_name = f"vowel_{v}"
        train_df[col_name] = (train_df["phone"].astype(str).str.lower().str.startswith(v)).astype(float)
        test_df[col_name] = (test_df["phone"].astype(str).str.lower().str.startswith(v)).astype(float)
        feature_cols.append(col_name)

    X_train = train_df[feature_cols].fillna(0.0).values
    y_train = train_df[label_col].values.astype(int)
    X_val = val_df[feature_cols].fillna(0.0).values
    y_val = val_df[label_col].values.astype(int)
    X_test = test_df[feature_cols].fillna(0.0).values
    y_test = test_df[label_col].values.astype(int)

    if len(np.unique(y_val)) < 2:
        raise RuntimeError(
            "Validation split is single-class; cannot freeze an F1 threshold on it. "
            "Check the manifest batch_split distribution."
        )

    def freeze_val_threshold(predict_fn):
        """Scan the F1-optimal threshold on the VALIDATION split only."""
        val_prob = predict_fn(X_val)
        precision, recall, thresholds = precision_recall_curve(y_val, val_prob)
        f1_scores = 2 * (precision * recall) / np.clip(precision + recall, 1e-9, None)
        best_idx = np.nanargmax(f1_scores)
        return float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

    print(f"[Features] Matrix shapes: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape} ({len(feature_cols)} features)")

    results: List[Dict[str, Any]] = []

    # 1. Logistic Regression (Strict Linear Probing Capacity)
    print("\n[Model 1/2] Training Logistic Regression Baseline (StandardScaler + L2 Balanced)...")
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42, C=1.0)),
    ])
    lr_pipe.fit(X_train, y_train)
    lr_tau = freeze_val_threshold(lr_pipe.predict_proba)
    lr_prob = lr_pipe.predict_proba(X_test)[:, 1]
    lr_metrics = evaluate_predictions(y_test, lr_prob, threshold=lr_tau)
    print(f"  >> Logistic Regression | PR-AUC: {lr_metrics['pr_auc']:.4f} | F1: {lr_metrics['f1']:.4f} | ROC-AUC: {lr_metrics['roc_auc']:.4f}")

    results.append({
        "model": "Logistic Regression (Linear Probe)",
        "feature_set": "Macro-Prosody (F0, Energy, Duration, Vowel)",
        "num_features": len(feature_cols),
        "pr_auc": lr_metrics["pr_auc"],
        "f1": lr_metrics["f1"],
        "roc_auc": lr_metrics["roc_auc"],
        "accuracy": lr_metrics["accuracy"],
        "best_threshold": lr_metrics["best_threshold"],
    })

    # 2. Gradient Boosted Decision Trees (Strong Non-Linear Interactions)
    print("\n[Model 2/2] Training Gradient Boosting Baseline (GBDT n=100, depth=4)...")
    gbdt = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    gbdt.fit(X_train, y_train)
    gbdt_tau = freeze_val_threshold(gbdt.predict_proba)
    gbdt_prob = gbdt.predict_proba(X_test)[:, 1]
    gbdt_metrics = evaluate_predictions(y_test, gbdt_prob, threshold=gbdt_tau)
    print(f"  >> Gradient Boosting   | PR-AUC: {gbdt_metrics['pr_auc']:.4f} | F1: {gbdt_metrics['f1']:.4f} | ROC-AUC: {gbdt_metrics['roc_auc']:.4f}")

    results.append({
        "model": "Gradient Boosting (GBDT)",
        "feature_set": "Macro-Prosody (F0, Energy, Duration, Vowel)",
        "num_features": len(feature_cols),
        "pr_auc": gbdt_metrics["pr_auc"],
        "f1": gbdt_metrics["f1"],
        "roc_auc": gbdt_metrics["roc_auc"],
        "accuracy": gbdt_metrics["accuracy"],
        "best_threshold": gbdt_metrics["best_threshold"],
    })

    # Save to CSV
    out_path = Path(args.output) if Path(args.output).is_absolute() else (PROJECT_ROOT / args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res_df = pd.DataFrame(results)
    res_df.to_csv(out_path, index=False)
    print(f"\n[Saved] Prosodic baseline results saved to: {out_path}")

    # Generate Markdown Summary
    md_path = out_path.with_suffix(".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# WavFry Macro-Prosodic Control Baseline Results\n\n")
        f.write(f"- **Dataset Manifest**: `{manifest_path.name}` (Train: {len(train_df)}, Test: {len(test_df)})\n")
        f.write(f"- **Features Included**: CREPE F0 statistics, Periodicity, RMS Energy contour, Duration, Vowel one-hot ({len(feature_cols)} dims)\n\n")
        f.write("### Benchmark Comparison\n\n")
        f.write("| Model Type | Model Capacity | PR-AUC | F1 | ROC-AUC | Accuracy | Best Threshold |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for r in results:
            cap = "Linear" if "Linear" in r["model"] else "Non-Linear Tree"
            f.write(f"| **{r['model']}** | {cap} | **{r['pr_auc']:.4f}** | **{r['f1']:.4f}** | {r['roc_auc']:.4f} | {r['accuracy']:.4f} | {r['best_threshold']:.4f} |\n")
        f.write("\n\n*Threshold protocol: F1-optimal tau frozen on the validation split, applied unchanged to the test split. Compare against WavFry numbers computed on the same manifest generation and split definition.*\n")

    print(f"[Saved] Markdown summary saved to: {md_path}")


if __name__ == "__main__":
    main()

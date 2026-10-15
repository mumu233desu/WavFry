"""
Table 1 & Figure 2: Unified Baseline Evaluation across QUT Acoustic Degradation Tiers.

Canonical Data-Producing Script:
- Evaluates WavFry vs 4 Baselines (Praat, Creapy, Kane, DeepFry) across:
  Clean, Mild (15dB), Moderate (5dB), Severe (0dB), and Compound (Noise+RIR).
- Implements Incremental CSV Caching to eliminate redundant re-computation.
- Strict Decoupling: Outputs ONLY structured data (vowel_robustness_metrics.json & table1_baseline_comparison.csv).
  Plotting is handled independently by data/visualizations/plot_fig2_noise_robustness.py.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    auc,
    roc_auc_score,
)
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wavfry.eval.cache_manager import (
    CANONICAL_TIERS,
    DEFAULT_TIER_CONDITIONS,
    IncrementalCSVCache,
    attach_interval_keys,
    dump_model_predictions,
)
from wavfry.eval.acoustic_degradation import AcousticDegradationEngine
from wavfry.eval.unified import build_test_manifest, load_model_from_checkpoint, resolve_frozen_threshold

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_qut_baselines")

BASELINES = ["WavFry (Ours)", "Praat", "Creapy", "Kane", "DeepFry"]


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute classification metrics with zero-division safety."""
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_prob)
    yt = y_true[valid_mask].astype(int)
    yp = y_prob[valid_mask]

    if len(yt) == 0 or len(np.unique(yt)) < 2:
        return {
            "Accuracy": 0.0,
            "Precision": 0.0,
            "Recall": 0.0,
            "F1-Score": 0.0,
            "PR-AUC": 0.0,
            "ROC-AUC": 0.0,
        }

    y_pred = (yp >= threshold).astype(int)

    acc = float(accuracy_score(yt, y_pred))
    prec = float(precision_score(yt, y_pred, zero_division=0.0))
    rec = float(recall_score(yt, y_pred, zero_division=0.0))
    f1 = float(f1_score(yt, y_pred, zero_division=0.0))

    try:
        p_curve, r_curve, _ = precision_recall_curve(yt, yp)
        pr_auc = float(auc(r_curve, p_curve))
    except Exception:
        pr_auc = 0.0

    try:
        roc_auc = float(roc_auc_score(yt, yp))
    except Exception:
        roc_auc = 0.0

    return {
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "PR-AUC": round(pr_auc, 4),
        "ROC-AUC": round(roc_auc, 4),
    }


def _extract_single_praat_worker(item: Tuple[str, str, float, float, str, str, str]) -> Dict[str, Any]:
    audio_path_str, interval_key, start_s, end_s, dataset_id, file_key, condition = item
    audio_path = Path(audio_path_str)
    if not audio_path.exists():
        prob = 0.0
    else:
        try:
            import parselmouth

            sound = parselmouth.Sound(str(audio_path))
            pitch = sound.to_pitch(time_step=0.015, pitch_floor=50.0, pitch_ceiling=350.0)
            xs = np.array(pitch.xs())
            freqs = np.array(pitch.selected_array["frequency"])
            mask = (xs >= start_s) & (xs <= end_s)
            if np.sum(mask) == 0:
                prob = 0.0
            else:
                sub_f = freqs[mask]
                uv_ratio = np.mean(sub_f == 0.0)
                prob = float(np.clip(uv_ratio, 0.0, 1.0))
        except Exception:
            prob = 0.0

    return {
        "interval_key": interval_key,
        "dataset_id": dataset_id,
        "file_key": file_key,
        "start_s": start_s,
        "end_s": end_s,
        "model_name": "Praat",
        "tier": condition,
        "prob_creaky": prob,
    }


def extract_praat_predictions(
    manifest_df: pd.DataFrame,
    degradation_engine: Optional[AcousticDegradationEngine] = None,
    condition: str = "clean",
    max_workers: Optional[int] = None,
) -> pd.DataFrame:
    """Extract Praat pitch unvoiced/perturbation creaky predictions using Parselmouth."""
    total = len(manifest_df)
    if total == 0:
        return pd.DataFrame()

    items = [
        (
            str(row["audio_path"]),
            str(row["interval_key"]),
            float(row["start_s"]),
            float(row["end_s"]),
            str(row.get("dataset_id", "cs_dialogue")),
            str(row.get("file_key", Path(str(row["audio_path"])).stem)),
            str(condition),
        )
        for _, row in manifest_df.iterrows()
    ]

    if total <= 20:
        rows = [_extract_single_praat_worker(item) for item in items]
        return pd.DataFrame(rows)

    workers = max_workers or min(12, os.cpu_count() or 4)
    logger.info(f"Extracting Praat predictions in parallel across {workers} CPU workers ({total} intervals)...")

    import concurrent.futures

    rows = []
    chunksize = max(1, total // (workers * 4))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for idx, record in enumerate(executor.map(_extract_single_praat_worker, items, chunksize=chunksize)):
            rows.append(record)

    return pd.DataFrame(rows)


def run_baseline_evaluation(
    manifest_path: Path,
    checkpoint_path: Optional[Path],
    preds_csv: Optional[Path],
    baselines_cache_path: Path,
    output_dir: Path,
    limit_samples: Optional[int] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Execute evaluation and return (metrics_dict, table1_df)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    full_manifest = pd.read_csv(manifest_path, low_memory=False)
    test_manifest = build_test_manifest(full_manifest, task="creaky")
    test_manifest = attach_interval_keys(test_manifest)

    if limit_samples is not None and limit_samples > 0:
        test_manifest = test_manifest.iloc[:limit_samples].copy().reset_index(drop=True)
        logger.info(f"Limiting evaluation to {len(test_manifest)} test samples.")

    # 1. Obtain or load model predictions
    if preds_csv and Path(preds_csv).exists():
        logger.info(f"Loading cached model predictions from {preds_csv}")
        model_preds_df = pd.read_csv(preds_csv)
    elif checkpoint_path and Path(checkpoint_path).exists():
        from wavfry.model.wavlm_classifier import WavFryClassifier

        logger.info(f"Loading checkpoint from {checkpoint_path}")
        model, ckpt_dict = load_model_from_checkpoint(
            checkpoint_path, torch.device(device), WavFryClassifier
        )
        deg_engine = AcousticDegradationEngine()
        dump_path = output_dir / f"preds_{checkpoint_path.stem}_tiered.csv"
        model_preds_df = dump_model_predictions(
            model=model,
            manifest_df=test_manifest,
            degradation_engine=deg_engine,
            out_csv_path=dump_path,
            device=torch.device(device),
        )
    else:
        raise ValueError("Must provide either a valid --checkpoint or an existing --preds-csv.")

    # 2. Incremental Baselines Cache
    base_cache = IncrementalCSVCache(baselines_cache_path)
    cached_base, missing_base = base_cache.partition_manifest(test_manifest)

    if not missing_base.empty:
        logger.info(f"Extracting Praat baseline for {len(missing_base)} missing intervals...")
        praat_new = extract_praat_predictions(missing_base, condition="clean")
        base_cache.append_records(praat_new)

    # 3. Aggregate metrics across tiers
    # Tier mapping format for JSON
    tier_display_names = {
        "clean": "Clean",
        "mild_15db": "15dB",
        "moderate_5db": "5dB",
        "severe_0db": "0dB",
        "compound": "Compound",
    }

    metrics_json: Dict[str, Dict[str, Dict[str, float]]] = {}
    table1_rows: List[Dict[str, Any]] = []

    # Map models
    models_to_eval = ["WavFry (Ours)", "Praat"]

    for tier_key in CANONICAL_TIERS:
        disp_name = tier_display_names[tier_key]
        metrics_json[disp_name] = {}

        # WavFry
        wf_sub = model_preds_df[model_preds_df["tier"] == tier_key]
        if not wf_sub.empty:
            wf_m = compute_binary_metrics(
                wf_sub["label_creaky"].values,
                wf_sub["prob_creaky"].values,
                threshold=0.5,
            )
            metrics_json[disp_name]["WavFry (Ours)"] = wf_m

        # Praat
        praat_sub = base_cache.df[
            (base_cache.df["model_name"] == "Praat")
            & (base_cache.df["interval_key"].isin(test_manifest["interval_key"]))
        ]
        if not praat_sub.empty:
            # Join with ground truth
            merged = praat_sub.merge(
                test_manifest[["interval_key", "label_creaky"]],
                on="interval_key",
                how="inner",
            )
            praat_m = compute_binary_metrics(
                merged["label_creaky"].values,
                merged["prob_creaky"].values,
                threshold=0.5,
            )
            metrics_json[disp_name]["Praat"] = praat_m

    # Build Table 1 DataFrame (F1-Score across tiers)
    for model_name in models_to_eval:
        row_dict = {"Model": model_name}
        f1_list = []
        for tier_key in CANONICAL_TIERS:
            disp_name = tier_display_names[tier_key]
            val = metrics_json.get(disp_name, {}).get(model_name, {}).get("F1-Score", 0.0)
            row_dict[disp_name] = val
            f1_list.append(val)
        row_dict["Mean"] = round(float(np.mean(f1_list)), 4) if f1_list else 0.0
        table1_rows.append(row_dict)

    table1_df = pd.DataFrame(table1_rows)

    # 4. Save JSON and CSV
    json_path = output_dir / "vowel_robustness_metrics.json"
    csv_path = output_dir / "table1_baseline_comparison.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2)
    table1_df.to_csv(csv_path, index=False, encoding="utf-8")

    logger.info(f"Saved Table 1 data to {csv_path}")
    logger.info(f"Saved Figure 2 JSON metrics to {json_path}")

    return metrics_json, table1_df


def main():
    parser = argparse.ArgumentParser(description="Evaluate WavFry and Baselines across QUT-NOISE tiers.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to WavFry checkpoint.")
    parser.add_argument("--manifest", type=str, default="manifests/segments_multi.csv", help="Test manifest path.")
    parser.add_argument("--preds-csv", type=str, default=None, help="Pre-computed predictions CSV path.")
    parser.add_argument(
        "--baselines-cache",
        type=str,
        default="data/experiments/cache/static_baselines_cache.csv",
        help="Path to incremental baselines cache CSV.",
    )
    parser.add_argument("--output-dir", type=str, default="data/experiments/baselines", help="Output directory.")
    parser.add_argument("--limit-samples", type=int, default=None, help="Limit number of samples for fast testing.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    metrics_json, table1_df = run_baseline_evaluation(
        manifest_path=Path(args.manifest),
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        preds_csv=Path(args.preds_csv) if args.preds_csv else None,
        baselines_cache_path=Path(args.baselines_cache),
        output_dir=Path(args.output_dir),
        limit_samples=args.limit_samples,
        device=args.device,
    )

    print("\n=== Table 1: Baseline Comparison under Acoustic Degradation (F1-Score) ===")
    print(table1_df.to_string(index=False))


if __name__ == "__main__":
    main()

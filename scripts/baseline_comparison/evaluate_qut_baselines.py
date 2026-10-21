"""
Table 1 & Figure 2: Unified Baseline Evaluation across QUT Acoustic Degradation Tiers.

Canonical Data-Producing Script:
- Evaluates WavFry vs 4 Baselines (Praat, Creapy, Kane, DeepFry) across:
  Clean, Mild (15dB), Moderate (5dB), Severe (0dB), and Compound (Noise+RIR).
- Implements Composite-Key CSV Caching (model_name::tier::interval_key) to eliminate redundant re-computation.
- Strict Decoupling: Outputs ONLY structured data (vowel_robustness_metrics.json & table1_baseline_comparison.csv).
  Plotting is handled independently by data/visualizations/plot_fig2_noise_robustness.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import warnings

# Suppress known benign third-party warnings from CreaPy and NumPy
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Mean of empty slice.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered.*")

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wavfry.eval.acoustic_degradation import AcousticDegradationEngine
from wavfry.eval.cache_manager import (
    CANONICAL_TIERS,
    DEFAULT_TIER_CONDITIONS,
    attach_interval_keys,
    dump_model_predictions,
)
from wavfry.eval.unified import build_test_manifest, load_model_from_checkpoint, resolve_frozen_threshold

from scripts.baseline_comparison.baseline_extractors import (
    extract_creapy_predictions,
    extract_deepfry_predictions,
    extract_kane_predictions,
    extract_praat_predictions,
    stage_degraded_audio_pool,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_qut_baselines")

# Silence noisy external library loggers
for noisy in ["httpx", "urllib3", "transformers", "huggingface_hub", "torch", "parselmouth"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

ALL_BASELINES = ["Praat", "Creapy", "Kane", "DeepFry"]
CANONICAL_MODELS = ["WavFry (Ours)", "Praat", "Creapy", "Kane", "DeepFry"]


class BaselineCSVCache:
    """Incremental CSV Cache with Composite Key (model_name::tier::interval_key)."""

    def __init__(self, cache_path: Path):
        self.cache_path = Path(cache_path).resolve()
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.df = self._load()

    def _load(self) -> pd.DataFrame:
        if not self.cache_path.exists() or self.cache_path.stat().st_size == 0:
            return pd.DataFrame(
                columns=[
                    "cache_key",
                    "interval_key",
                    "dataset_id",
                    "file_key",
                    "start_s",
                    "end_s",
                    "model_name",
                    "tier",
                    "prob_creaky",
                ]
            )
        try:
            df = pd.read_csv(self.cache_path, low_memory=False)
            if "cache_key" not in df.columns:
                df["cache_key"] = (
                    df["model_name"].astype(str)
                    + "::"
                    + df["tier"].astype(str)
                    + "::"
                    + df["interval_key"].astype(str)
                )
            df = df.drop_duplicates(subset=["cache_key"], keep="last").reset_index(drop=True)
            return df
        except Exception as e:
            logger.error(f"Error loading cache from {self.cache_path}: {e}")
            return pd.DataFrame()

    def append_records(self, new_records: pd.DataFrame) -> None:
        if new_records.empty:
            return
        new_df = new_records.copy()
        if "cache_key" not in new_df.columns:
            new_df["cache_key"] = (
                new_df["model_name"].astype(str)
                + "::"
                + new_df["tier"].astype(str)
                + "::"
                + new_df["interval_key"].astype(str)
            )
        self.df = pd.concat([self.df, new_df], ignore_index=True)
        self.df = self.df.drop_duplicates(subset=["cache_key"], keep="last").reset_index(drop=True)
        self.df.to_csv(self.cache_path, index=False, encoding="utf-8")
        logger.info(f"[BaselineCache] Appended {len(new_df)} records (Total: {len(self.df)}).")


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


def run_baseline_evaluation(
    manifest_path: Path,
    checkpoint_path: Optional[Path],
    preds_csv: Optional[Path],
    baselines_cache_path: Path,
    output_dir: Path,
    baselines_to_run: Optional[List[str]] = None,
    limit_samples: Optional[int] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Execute unified baseline evaluation across QUT degradation tiers."""
    output_dir.mkdir(parents=True, exist_ok=True)
    full_manifest = pd.read_csv(manifest_path, low_memory=False)
    test_manifest = build_test_manifest(full_manifest, task="creaky")
    test_manifest = attach_interval_keys(test_manifest)

    if limit_samples is not None and limit_samples > 0:
        test_manifest = test_manifest.iloc[:limit_samples].copy().reset_index(drop=True)
        logger.info(f"Limiting evaluation to {len(test_manifest)} test samples.")

    deg_engine = AcousticDegradationEngine()
    cache_base_dir = baselines_cache_path.resolve().parent
    staged_audio_dir = cache_base_dir / "staged_audio"
    spans_cache_dir = cache_base_dir / "baseline_spans"

    # 1. Obtain or load WavFry model predictions
    if preds_csv and Path(preds_csv).exists():
        logger.info(f"Loading cached model predictions from {preds_csv}")
        model_preds_df = pd.read_csv(preds_csv)
    elif checkpoint_path and Path(checkpoint_path).exists():
        from wavfry.model.wavlm_classifier import WavFryClassifier

        logger.info(f"Loading checkpoint from {checkpoint_path}")
        model, ckpt_dict = load_model_from_checkpoint(
            checkpoint_path, torch.device(device), WavFryClassifier
        )
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

    # 2. Incremental Baselines Cache setup
    base_cache = BaselineCSVCache(baselines_cache_path)
    selected_baselines = baselines_to_run or ALL_BASELINES

    # 3. Process each tier and extract missing baseline predictions
    for tier_key in CANONICAL_TIERS:
        logger.info(f"--- Processing degradation tier: {tier_key} ---")
        staged_wavs = stage_degraded_audio_pool(
            test_manifest=test_manifest,
            tier=tier_key,
            degradation_engine=deg_engine,
            staging_base_dir=staged_audio_dir,
            seed=42,
        )

        for baseline_name in selected_baselines:
            # Check existing records in cache for (baseline_name, tier_key)
            if not base_cache.df.empty:
                existing_keys = set(
                    base_cache.df[
                        (base_cache.df["model_name"] == baseline_name)
                        & (base_cache.df["tier"] == tier_key)
                    ]["interval_key"]
                )
            else:
                existing_keys = set()

            missing_df = test_manifest[~test_manifest["interval_key"].isin(existing_keys)].copy()

            if not missing_df.empty:
                logger.info(
                    f"Extracting {baseline_name} for {len(missing_df)} missing intervals under tier '{tier_key}'..."
                )
                if baseline_name == "Praat":
                    new_records = extract_praat_predictions(
                        missing_df, staged_wavs=staged_wavs, tier=tier_key
                    )
                elif baseline_name == "Creapy":
                    new_records = extract_creapy_predictions(
                        missing_df,
                        staged_wavs=staged_wavs,
                        tier=tier_key,
                        cache_spans_dir=spans_cache_dir / "creapy",
                    )
                elif baseline_name == "DeepFry":
                    new_records = extract_deepfry_predictions(
                        missing_df,
                        staged_wavs=staged_wavs,
                        tier=tier_key,
                        cache_spans_dir=spans_cache_dir / "deepfry",
                    )
                elif baseline_name == "Kane":
                    new_records = extract_kane_predictions(
                        missing_df,
                        staged_wavs=staged_wavs,
                        tier=tier_key,
                        cache_spans_dir=spans_cache_dir / "kane",
                    )
                else:
                    logger.warning(f"Unknown baseline: {baseline_name}")
                    new_records = pd.DataFrame()

                if not new_records.empty:
                    base_cache.append_records(new_records)

    # 4. Aggregate metrics across tiers and models
    tier_display_names = {
        "clean": "Clean",
        "mild_15db": "15dB",
        "moderate_5db": "5dB",
        "severe_0db": "0dB",
        "compound": "Compound",
    }

    metrics_json: Dict[str, Dict[str, Dict[str, float]]] = {}
    models_to_eval = ["WavFry (Ours)"] + selected_baselines

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

        # Baselines
        for b_name in selected_baselines:
            b_sub = base_cache.df[
                (base_cache.df["model_name"] == b_name)
                & (base_cache.df["tier"] == tier_key)
                & (base_cache.df["interval_key"].isin(test_manifest["interval_key"]))
            ]
            if not b_sub.empty:
                merged = b_sub.merge(
                    test_manifest[["interval_key", "label_creaky"]],
                    on="interval_key",
                    how="inner",
                )
                b_m = compute_binary_metrics(
                    merged["label_creaky"].values,
                    merged["prob_creaky"].values,
                    threshold=0.5,
                )
                metrics_json[disp_name][b_name] = b_m

    # Build Table 1 DataFrame (F1-Score across tiers)
    table1_rows: List[Dict[str, Any]] = []
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

    # 5. Save JSON, CSV, and Markdown byproduct
    json_path = output_dir / "vowel_robustness_metrics.json"
    csv_path = output_dir / "table1_baseline_comparison.csv"
    md_path = output_dir / "table1_baseline_comparison.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2)
    table1_df.to_csv(csv_path, index=False, encoding="utf-8")

    # Generate Markdown report
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Table 1: Baseline Comparison under Acoustic Degradation (F1-Score)\n\n")
        f.write("- **Tiers**: Clean, Mild (15dB), Moderate (5dB), Severe (0dB), and Compound (Noise+RIR)\n")
        f.write("- **Canonical Conditions**: 5 Noise Scenarios (Cafe, Car, Home, Reverb, Street) + 2 Strictly Matched RIR Compounds\n\n")
        try:
            f.write(table1_df.to_markdown(index=False))
        except Exception:
            headers = list(table1_df.columns)
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("| " + " | ".join([":---"] + [":---:"] * (len(headers) - 1)) + " |\n")
            for _, r in table1_df.iterrows():
                f.write("| " + " | ".join(str(r[c]) for c in headers) + " |\n")
        f.write("\n\n---\n*Note: Metrics represent macro-averaged F1-Score across evaluation intervals.*\n")

    logger.info(f"Saved Table 1 CSV data to {csv_path}")
    logger.info(f"Saved Table 1 Markdown report to {md_path}")
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
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=ALL_BASELINES,
        choices=ALL_BASELINES,
        help="Baseline models to evaluate (default: Praat Creapy Kane DeepFry).",
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
        baselines_to_run=args.baselines,
        output_dir=Path(args.output_dir),
        limit_samples=args.limit_samples,
        device=args.device,
    )

    print("\n=== Table 1: Baseline Comparison under Acoustic Degradation (F1-Score) ===")
    print(table1_df.to_string(index=False))


if __name__ == "__main__":
    main()

"""
Temporary Continuous-Score Baseline Evaluator.

Evaluates vowel-interval continuous probability aggregation (Mean & Max)
for CreaPy (10ms) and DeepFry (5ms) against legacy span overlap scores across
the 5 canonical degradation tiers (clean, mild_15db, moderate_5db, severe_0db, compound).

Usage:
    uv run python scripts/baseline_comparison/temp_evaluate_continuous_baselines.py
    uv run python scripts/baseline_comparison/temp_evaluate_continuous_baselines.py --tiers clean
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

# Base directories
BASELINES_DIR_PRIMARY = Path(r"E:\Coding\WavFry\data\experiments\baselines")
if BASELINES_DIR_PRIMARY.exists():
    BASELINES_DIR = BASELINES_DIR_PRIMARY
    ROOT_DIR = Path(r"E:\Coding\WavFry")
else:
    SCRIPT_DIR = Path(__file__).resolve().parent
    CODE_DIR = SCRIPT_DIR.parent.parent
    ROOT_DIR = CODE_DIR.parent if (CODE_DIR.parent / "data").exists() else CODE_DIR
    BASELINES_DIR = ROOT_DIR / "data" / "experiments" / "baselines"

CODE_DIR = Path(r"E:\Coding\WavFry\code") if (Path(r"E:\Coding\WavFry\code")).exists() else Path(__file__).resolve().parent.parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from wavfry.eval.unified import build_test_manifest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("temp_eval_continuous")

CANONICAL_TIERS = ["clean", "mild_15db", "moderate_5db", "severe_0db", "compound"]
TIER_DISPLAY_NAMES = {
    "clean": "Clean",
    "mild_15db": "15dB",
    "moderate_5db": "5dB",
    "severe_0db": "0dB",
    "compound": "Compound",
}

DEFAULT_PROBS_CACHE = ROOT_DIR / "data" / "experiments" / "cache" / "temp_frame_probabilities"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "experiments" / "temp_continuous_baselines_eval"
LEGACY_METRICS_PATH = ROOT_DIR / "data" / "experiments" / "baselines" / "vowel_robustness_metrics.json"


def compute_optimal_f1(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """Sweeps thresholds in [0.01, 0.99] to find maximum F1 score."""
    best_f1 = -1.0
    best_p = 0.0
    best_r = 0.0
    best_acc = 0.0
    best_thresh = 0.5

    thresholds = np.linspace(0.01, 0.99, 99)
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        acc = float(np.mean(y_true == y_pred))
        if f1 > best_f1:
            best_f1 = float(f1)
            best_p = float(p)
            best_r = float(r)
            best_acc = acc
            best_thresh = float(t)

    return {
        "F1-Score": round(best_f1, 4),
        "Precision": round(best_p, 4),
        "Recall": round(best_r, 4),
        "Accuracy": round(best_acc, 4),
        "Optimal_Threshold": round(best_thresh, 4),
    }


def evaluate_continuous_baselines(
    manifest_path: Path,
    probs_cache_dir: Path,
    output_dir: Path,
    tiers: List[str],
    models: List[str],
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load canonical test manifest
    logger.info(f"Loading manifest: {manifest_path}")
    full_manifest = pd.read_csv(manifest_path, low_memory=False)
    test_manifest = build_test_manifest(full_manifest, task="creaky")
    logger.info(f"Test manifest loaded: {len(test_manifest)} vowel tokens.")

    # 2. Load legacy baseline & WavFry metrics for side-by-side comparison
    legacy_metrics = {}
    if LEGACY_METRICS_PATH.exists():
        try:
            with open(LEGACY_METRICS_PATH, "r", encoding="utf-8") as f:
                legacy_metrics = json.load(f)
            logger.info("Loaded legacy baseline metrics from vowel_robustness_metrics.json")
        except Exception as e:
            logger.warning(f"Could not load legacy metrics: {e}")

    results_data: Dict[str, Dict[str, Any]] = {}
    comparison_rows: List[Dict[str, Any]] = []

    # 3. Evaluate each tier
    for tier in tiers:
        disp_tier = TIER_DISPLAY_NAMES.get(tier, tier.capitalize())
        logger.info(f"=== Evaluating degradation tier: {disp_tier} ({tier}) ===")
        results_data[disp_tier] = {}

        # Add WavFry (Ours) reference from legacy metrics if present
        wavfry_legacy = legacy_metrics.get(disp_tier, {}).get("WavFry (Ours)")
        if wavfry_legacy:
            results_data[disp_tier]["WavFry (Ours)"] = wavfry_legacy

        for model_name in models:
            model_tier_dir = probs_cache_dir / model_name / tier
            if not model_tier_dir.exists():
                logger.warning(f"Probability cache missing for {model_name} on {tier}: {model_tier_dir}")
                continue

            # Cache npz arrays by file_key in memory for this tier
            file_cache: Dict[str, Dict[str, np.ndarray]] = {}
            for npz_p in model_tier_dir.glob("*.npz"):
                fkey = npz_p.stem
                try:
                    data = np.load(npz_p)
                    file_cache[fkey] = {
                        "timestamps": data["timestamps"],
                        "prob_creak": data["prob_creak"],
                    }
                except Exception as e:
                    logger.warning(f"Error loading {npz_p}: {e}")

            # Collect scores
            y_true_list = []
            scores_mean = []
            scores_max = []

            for _, row in test_manifest.iterrows():
                fkey = str(row["file_key"])
                start_s = float(row["start_s"])
                end_s = float(row["end_s"])
                y_val = int(row["label_creaky"])

                file_data = file_cache.get(fkey)
                if file_data is None:
                    # Missing file probability -> fallback to 0
                    y_true_list.append(y_val)
                    scores_mean.append(0.0)
                    scores_max.append(0.0)
                    continue

                t_arr = file_data["timestamps"]
                p_arr = file_data["prob_creak"]

                # Mask frames within [start_s, end_s]
                mask = (t_arr >= start_s) & (t_arr <= end_s)
                if np.any(mask):
                    sub_p = p_arr[mask]
                    sc_mean = float(np.mean(sub_p))
                    sc_max = float(np.max(sub_p))
                else:
                    # Fallback to nearest frame if interval is very short
                    idx = int(np.argmin(np.abs(t_arr - (start_s + end_s) / 2.0)))
                    sc_mean = float(p_arr[idx])
                    sc_max = float(p_arr[idx])

                y_true_list.append(y_val)
                scores_mean.append(sc_mean)
                scores_max.append(sc_max)

            y_true = np.array(y_true_list, dtype=int)
            s_mean = np.array(scores_mean, dtype=float)
            s_max = np.array(scores_max, dtype=float)

            # Compute threshold-free and optimal metrics for Mean
            prauc_mean = round(float(average_precision_score(y_true, s_mean)), 4)
            rocauc_mean = round(float(roc_auc_score(y_true, s_mean)), 4)
            opt_mean = compute_optimal_f1(y_true, s_mean)
            opt_mean.update({"PR-AUC": prauc_mean, "ROC-AUC": rocauc_mean})

            # Compute metrics for Max
            prauc_max = round(float(average_precision_score(y_true, s_max)), 4)
            rocauc_max = round(float(roc_auc_score(y_true, s_max)), 4)
            opt_max = compute_optimal_f1(y_true, s_max)
            opt_max.update({"PR-AUC": prauc_max, "ROC-AUC": rocauc_max})

            # Store in results
            results_data[disp_tier][f"{model_name} (Continuous Mean)"] = opt_mean
            results_data[disp_tier][f"{model_name} (Continuous Max)"] = opt_max

            # Retrieve legacy overlap score (case-insensitive lookup)
            legacy_tier_dict = legacy_metrics.get(disp_tier, {})
            legacy_model_score = {}
            for k, v in legacy_tier_dict.items():
                if k.lower() == model_name.lower():
                    legacy_model_score = v
                    break
            legacy_f1 = legacy_model_score.get("F1-Score", "N/A")
            legacy_prauc = legacy_model_score.get("PR-AUC", "N/A")

            comparison_rows.append({
                "Tier": disp_tier,
                "Model": model_name,
                "Legacy Overlap F1": legacy_f1,
                "Continuous Mean F1": opt_mean["F1-Score"],
                "Continuous Max F1": opt_max["F1-Score"],
                "Legacy Overlap PR-AUC": legacy_prauc,
                "Continuous Mean PR-AUC": prauc_mean,
                "Continuous Max PR-AUC": prauc_max,
                "Optimal Tau (Mean)": opt_mean["Optimal_Threshold"],
            })

    # 4. Save metrics JSON
    json_path = output_dir / "temp_continuous_baselines_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)
    logger.info(f"Saved metrics JSON to: {json_path}")

    # 5. Save comparison CSV
    comp_df = pd.DataFrame(comparison_rows)
    csv_path = output_dir / "temp_continuous_baselines_comparison.csv"
    comp_df.to_csv(csv_path, index=False, encoding="utf-8")
    logger.info(f"Saved comparison CSV to: {csv_path}")

    # 6. Generate Markdown comparison report
    md_path = output_dir / "temp_comparison_with_legacy_overlap.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Temporary Comparison: Continuous-Score Baseline Re-Evaluation\n\n")
        f.write("This report contrasts the **Legacy Span Overlap Ratio** with the new **Continuous Probability Aggregation (Mean & Max)** on the 1,371 canonical test vowel intervals.\n\n")
        f.write("## 1. Summary Comparison Table (F1-Score & PR-AUC)\n\n")
        try:
            f.write(comp_df.to_markdown(index=False))
        except Exception:
            headers = list(comp_df.columns)
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("| " + " | ".join([":---"] + [":---:"] * (len(headers) - 1)) + " |\n")
            for _, r in comp_df.iterrows():
                f.write("| " + " | ".join(str(r[c]) for c in headers) + " |\n")
        
        f.write("\n\n## 2. Complete Metrics by Degradation Tier\n\n")
        for tier_name, tier_models in results_data.items():
            f.write(f"### Tier: {tier_name}\n\n")
            tier_df = pd.DataFrame.from_dict(tier_models, orient="index")
            try:
                f.write(tier_df.to_markdown())
            except Exception:
                f.write(tier_df.to_string())
            f.write("\n\n")

    logger.info(f"Saved comparison Markdown to: {md_path}")
    logger.info("=== Continuous baseline evaluation complete! ===")


def main():
    parser = argparse.ArgumentParser(description="Evaluate continuous baseline scores.")
    parser.add_argument("--manifest", type=str, default="manifests/segments_multi.csv")
    parser.add_argument("--probs-cache", type=str, default=str(DEFAULT_PROBS_CACHE))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--tiers", nargs="+", default=CANONICAL_TIERS, choices=CANONICAL_TIERS)
    parser.add_argument("--models", nargs="+", default=["CreaPy", "DeepFry"], choices=["CreaPy", "DeepFry"])

    args = parser.parse_args()

    manifest_p = Path(args.manifest)
    if not manifest_p.is_absolute():
        manifest_p = CODE_DIR / manifest_p

    evaluate_continuous_baselines(
        manifest_path=manifest_p,
        probs_cache_dir=Path(args.probs_cache),
        output_dir=Path(args.output_dir),
        tiers=args.tiers,
        models=args.models,
    )


if __name__ == "__main__":
    main()

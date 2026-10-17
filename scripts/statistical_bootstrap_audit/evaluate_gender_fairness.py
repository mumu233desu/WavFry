"""
Table 4: Speaker Gender Fairness and Demographic Disparity Audit.

Canonical Data-Producing Script:
- Evaluates model performance and probability distributions disaggregated by Gender (Female vs Male).
- Quantifies performance parity: Delta = |Metric(Female) - Metric(Male)|.
- Strict Decoupling: Outputs ONLY structured data (gender_metrics.json & gender_fairness_distribution.csv).
  Plotting is handled by data/visualizations/plot_gender_fairness_distribution.py.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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
import sys
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wavfry.eval.cache_manager import attach_interval_keys, dump_model_predictions
from wavfry.eval.acoustic_degradation import AcousticDegradationEngine
from wavfry.eval.unified import build_test_manifest, load_model_from_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_gender_fairness")


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_prob)
    yt = y_true[valid_mask].astype(int)
    yp = y_prob[valid_mask]

    if len(yt) == 0 or len(np.unique(yt)) < 2:
        return {"N": len(yt), "Accuracy": 0.0, "Precision": 0.0, "Recall": 0.0, "F1-Score": 0.0, "PR-AUC": 0.0, "ROC-AUC": 0.0}

    y_pred = (yp >= threshold).astype(int)
    acc = float(accuracy_score(yt, y_pred))
    prec = float(precision_score(yt, y_pred, zero_division=0.0))
    rec = float(recall_score(yt, y_pred, zero_division=0.0))
    f1 = float(f1_score(yt, y_pred, zero_division=0.0))

    try:
        p_c, r_c, _ = precision_recall_curve(yt, yp)
        pr_auc = float(auc(r_c, p_c))
    except Exception:
        pr_auc = 0.0

    try:
        roc_auc = float(roc_auc_score(yt, yp))
    except Exception:
        roc_auc = 0.0

    return {
        "N": int(len(yt)),
        "Positive_Rate": round(float(np.mean(yt)), 4),
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "PR-AUC": round(pr_auc, 4),
        "ROC-AUC": round(roc_auc, 4),
    }


def run_gender_fairness_audit(
    manifest_path: Path,
    checkpoint_path: Optional[Path],
    preds_csv: Optional[Path],
    output_dir: Path,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    full_manifest = pd.read_csv(manifest_path, low_memory=False)
    test_manifest = build_test_manifest(full_manifest, task="creaky")
    test_manifest = attach_interval_keys(test_manifest)

    # 1. Obtain Clean test predictions
    if preds_csv and Path(preds_csv).exists():
        logger.info(f"Loading predictions from {preds_csv}")
        preds_df = pd.read_csv(preds_csv)
        # Filter clean tier if tiered
        if "tier" in preds_df.columns:
            clean_preds = preds_df[preds_df["tier"] == "clean"].copy()
        else:
            clean_preds = preds_df.copy()
    elif checkpoint_path and Path(checkpoint_path).exists():
        from wavfry.model.wavlm_classifier import WavFryClassifier

        logger.info(f"Running clean evaluation on checkpoint {checkpoint_path}")
        model, ckpt_dict = load_model_from_checkpoint(
            checkpoint_path, torch.device(device), WavFryClassifier
        )
        deg_engine = AcousticDegradationEngine()
        clean_preds = dump_model_predictions(
            model=model,
            manifest_df=test_manifest,
            degradation_engine=deg_engine,
            out_csv_path=output_dir / "clean_preds_temp.csv",
            device=torch.device(device),
            tier_conditions={"clean": ["clean"]},
        )
    else:
        raise ValueError("Must provide either --checkpoint or --preds-csv.")

    # 2. Harmonize gender
    if "gender" not in clean_preds.columns or clean_preds["gender"].isna().all():
        # Fallback extract from file_key: F_... or M_...
        clean_preds["gender"] = clean_preds["file_key"].apply(
            lambda k: "F" if str(k).startswith("F") else ("M" if str(k).startswith("M") else "Unknown")
        )

    clean_preds["Gender"] = clean_preds["gender"].map({"F": "Female", "M": "Male"}).fillna("Unknown")

    # Filter known genders
    valid_gender_df = clean_preds[clean_preds["Gender"].isin(["Female", "Male"])].copy()

    # 3. Disaggregated Metrics
    female_sub = valid_gender_df[valid_gender_df["Gender"] == "Female"]
    male_sub = valid_gender_df[valid_gender_df["Gender"] == "Male"]

    metrics_female = compute_metrics(female_sub["label_creaky"].values, female_sub["prob_creaky"].values)
    metrics_male = compute_metrics(male_sub["label_creaky"].values, male_sub["prob_creaky"].values)
    metrics_overall = compute_metrics(valid_gender_df["label_creaky"].values, valid_gender_df["prob_creaky"].values)

    # Compute Disparity Delta
    disparity = {
        "Delta_Accuracy": round(abs(metrics_female["Accuracy"] - metrics_male["Accuracy"]), 4),
        "Delta_F1": round(abs(metrics_female["F1-Score"] - metrics_male["F1-Score"]), 4),
        "Delta_PR_AUC": round(abs(metrics_female["PR-AUC"] - metrics_male["PR-AUC"]), 4),
        "Delta_ROC_AUC": round(abs(metrics_female["ROC-AUC"] - metrics_male["ROC-AUC"]), 4),
    }

    metrics_summary = {
        "Overall": metrics_overall,
        "Female": metrics_female,
        "Male": metrics_male,
        "Disparity": disparity,
    }

    # 4. Save JSON, Tidy Detail CSV, and Table 4 Presentation CSV
    json_path = output_dir / "gender_metrics.json"
    dist_csv_path = output_dir / "gender_fairness_distribution.csv"
    detail_csv_path = output_dir / "gender_fairness_detail.csv"
    table4_csv_path = output_dir / "table4_gender_fairness.csv"
    md_path = output_dir / "table4_gender_fairness.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    col_mapping = {
        "interval_key": "Interval-Key",
        "dataset_id": "Dataset-ID",
        "file_key": "File-Key",
        "speaker_id": "Speaker-ID",
        "Gender": "Gender",
        "label_creaky": "Label-Creaky",
        "prob_creaky": "Prob-Creaky",
        "label_breathy": "Label-Breathy",
        "prob_breathy": "Prob-Breathy",
    }
    avail_cols = [c for c in col_mapping.keys() if c in valid_gender_df.columns]
    detail_df = valid_gender_df[avail_cols].rename(columns=col_mapping)
    detail_df.to_csv(detail_csv_path, index=False, encoding="utf-8")
    valid_gender_df[avail_cols].to_csv(dist_csv_path, index=False, encoding="utf-8")

    # Build Table 4 presentation DataFrame
    table4_rows = [
        {
            "Group": "Overall",
            "N": metrics_overall["N"],
            "Positive-Rate": metrics_overall["Positive_Rate"],
            "Accuracy": metrics_overall["Accuracy"],
            "Precision": metrics_overall["Precision"],
            "Recall": metrics_overall["Recall"],
            "F1-Score": metrics_overall["F1-Score"],
            "PR-AUC": metrics_overall["PR-AUC"],
            "ROC-AUC": metrics_overall["ROC-AUC"],
        },
        {
            "Group": "Female",
            "N": metrics_female["N"],
            "Positive-Rate": metrics_female["Positive_Rate"],
            "Accuracy": metrics_female["Accuracy"],
            "Precision": metrics_female["Precision"],
            "Recall": metrics_female["Recall"],
            "F1-Score": metrics_female["F1-Score"],
            "PR-AUC": metrics_female["PR-AUC"],
            "ROC-AUC": metrics_female["ROC-AUC"],
        },
        {
            "Group": "Male",
            "N": metrics_male["N"],
            "Positive-Rate": metrics_male["Positive_Rate"],
            "Accuracy": metrics_male["Accuracy"],
            "Precision": metrics_male["Precision"],
            "Recall": metrics_male["Recall"],
            "F1-Score": metrics_male["F1-Score"],
            "PR-AUC": metrics_male["PR-AUC"],
            "ROC-AUC": metrics_male["ROC-AUC"],
        },
    ]
    table4_df = pd.DataFrame(table4_rows)
    table4_df.to_csv(table4_csv_path, index=False, encoding="utf-8")

    # 5. Markdown Report Byproduct
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Table 4: Speaker Gender Fairness & Demographic Disparity Audit\n\n")
        f.write("- **Condition**: `Clean` (Strict clean-only to prevent noise spectral masking demographic confounds)\n")
        f.write(f"- **Evaluated Clean Intervals**: Female = {metrics_female['N']}, Male = {metrics_male['N']} (Total = {metrics_overall['N']})\n\n")
        f.write("| Group | N | Positive Rate | Accuracy | Precision | Recall | F1-Score | PR-AUC | ROC-AUC |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in table4_rows:
            f.write(
                f"| **{r['Group']}** | {r['N']} | {r['Positive-Rate']:.4f} | {r['Accuracy']:.4f} | "
                f"{r['Precision']:.4f} | {r['Recall']:.4f} | **{r['F1-Score']:.4f}** | **{r['PR-AUC']:.4f}** | {r['ROC-AUC']:.4f} |\n"
            )
        f.write("\n### Demographic Disparity ($|\\text{Female} - \\text{Male}|$)\n\n")
        f.write(f"- **$\\Delta$ PR-AUC**: `{disparity['Delta_PR_AUC']:.4f}`\n")
        f.write(f"- **$\\Delta$ F1-Score**: `{disparity['Delta_F1']:.4f}`\n")
        f.write(f"- **$\\Delta$ Accuracy**: `{disparity['Delta_Accuracy']:.4f}`\n")
        f.write(f"- **$\\Delta$ ROC-AUC**: `{disparity['Delta_ROC_AUC']:.4f}`\n")

    logger.info(f"Saved Table 4 metrics to {json_path}")
    logger.info(f"Saved Table 4 CSV to {table4_csv_path}")
    logger.info(f"Saved Table 4 Markdown report to {md_path}")

    return metrics_summary, table4_df


def main():
    parser = argparse.ArgumentParser(description="Evaluate Speaker Gender Fairness (Table 4).")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint.")
    parser.add_argument("--manifest", type=str, default="manifests/segments_multi.csv", help="Test manifest path.")
    parser.add_argument("--preds-csv", type=str, default=None, help="Pre-computed predictions CSV path.")
    parser.add_argument("--output-dir", type=str, default="data/experiments/fairness", help="Output directory.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    metrics_summary, table4_df = run_gender_fairness_audit(
        manifest_path=Path(args.manifest),
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        preds_csv=Path(args.preds_csv) if args.preds_csv else None,
        output_dir=Path(args.output_dir),
        device=args.device,
    )

    print("\n=== Table 4: Speaker Gender Fairness Audit ===")
    print(table4_df.to_string(index=False))
    print("\n=== Disparity (|Female - Male|): ===")
    for k, v in metrics_summary["Disparity"].items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()

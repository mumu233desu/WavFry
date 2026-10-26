"""
Figure 8 & Table 8b: Acoustic-Physical Interpretability & Correlation Validation.

Canonical Data-Producing Script:
- Measures statistical correlations between deep representation predicted probabilities
  (Creaky Voice and Breathy Voice) and traditional gold-standard acoustic-physical parameters:
  * Creaky: Subharmonic-to-Harmonic Ratio (SHR), Jitter (local), Shimmer (local), Formant-corrected H1*-H2*
  * Breathy: CPPS (Cepstral Peak Prominence Smoothed), H1*-H2*, HNR (Harmonics-to-Noise Ratio)
- Eliminates cherry-picking by evaluating across the full test set.
- Incremental Caching: Pre-computes and caches expensive Parselmouth features into a CSV cache.
- Strict Decoupling: Outputs ONLY structured data (acoustic_correlations_metrics.json & acoustic_correlations.csv).
  Plotting is handled by data/visualizations/plot_fig8_acoustic_correlations.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Ensure 04_creaky_tools is importable for acoustic features
TOOLS_ROOT = Path(r"E:\Corpus\CS-Dialogue\data\short_wav\WAVE\04_creaky_tools")
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

try:
    from src.acoustic_features import compute_cpps, compute_h1_h2_star
except ImportError:
    compute_cpps = None
    compute_h1_h2_star = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_acoustic_correlations")


def _extract_single_audio_file(
    file_item: Tuple[str, List[Tuple[str, float, float, str, str]]]
) -> List[Dict[str, Any]]:
    """Extract physical acoustic metrics for all intervals in an audio file using continuous full-sentence Praat objects.

    Eliminates short-snippet edge tapering and pitch floor truncation artifacts by computing Praat analysis objects
    (Pitch, PointProcess, Harmonicity, PowerCepstrogram, Formant) continuously on the entire utterance, then querying
    local metrics over [start_s, end_s].
    """
    import parselmouth
    from parselmouth.praat import call

    audio_path_str, intervals = file_item
    audio_path = Path(audio_path_str)

    empty_results = []
    for interval_key, start_s, end_s, dataset_id, file_key in intervals:
        empty_results.append({
            "interval_key": interval_key,
            "dataset_id": dataset_id,
            "file_key": file_key,
            "start_s": start_s,
            "end_s": end_s,
            "jitter_local": np.nan,
            "shimmer_local": np.nan,
            "hnr": np.nan,
            "cpps": np.nan,
            "h1h2_star": np.nan,
            "shr": np.nan,
        })

    if not audio_path.exists():
        return empty_results

    try:
        sound = parselmouth.Sound(str(audio_path))
        dur = float(sound.duration)
        if dur < 0.05:
            return empty_results

        # 1. Continuous Pitch tracking & PointProcess
        pitch = sound.to_pitch(time_step=0.01, pitch_floor=50.0, pitch_ceiling=400.0)
        point_proc = call(sound, "To PointProcess (periodic, cc)", 50.0, 400.0)

        # 2. Continuous Harmonicity (HNR)
        harmonicity = sound.to_harmonicity(time_step=0.01, minimum_pitch=50.0)

        # 3. Continuous CPPS
        t_cpps, cpps_arr = np.array([]), np.array([])
        if compute_cpps is not None:
            try:
                t_cpps, cpps_arr = compute_cpps(sound, time_step=0.01)
            except Exception:
                pass

        # 4. Continuous H1*-H2*
        t_h1h2, h1h2_arr = np.array([]), np.array([])
        if compute_h1_h2_star is not None:
            try:
                t_h1h2, h1h2_arr = compute_h1_h2_star(sound, time_step=0.01)
            except Exception:
                pass

        p_times = pitch.xs()
        p_freqs_all = pitch.selected_array["frequency"]

        file_results = []
        for interval_key, start_s, end_s, dataset_id, file_key in intervals:
            res = {
                "interval_key": interval_key,
                "dataset_id": dataset_id,
                "file_key": file_key,
                "start_s": start_s,
                "end_s": end_s,
                "jitter_local": np.nan,
                "shimmer_local": np.nan,
                "hnr": np.nan,
                "cpps": np.nan,
                "h1h2_star": np.nan,
                "shr": np.nan,
            }

            s = max(0.0, min(dur, float(start_s)))
            e = max(0.0, min(dur, float(end_s)))
            if e - s < 0.02:
                file_results.append(res)
                continue

            # Silence floor check
            try:
                rms = call(sound, "Get root-mean-square", s, e)
                if rms < 1e-4:
                    file_results.append(res)
                    continue
            except Exception:
                pass

            # Jitter (local)
            try:
                jitter = call(point_proc, "Get jitter (local)", s, e, 0.0001, 0.02, 1.3)
                if not np.isnan(jitter) and jitter > 0:
                    res["jitter_local"] = float(jitter)
            except Exception:
                pass

            # Shimmer (local)
            try:
                shimmer = call([sound, point_proc], "Get shimmer (local)", s, e, 0.0001, 0.02, 1.3, 1.6)
                if not np.isnan(shimmer) and shimmer > 0:
                    res["shimmer_local"] = float(shimmer)
            except Exception:
                pass

            # HNR (mean)
            try:
                hnr_val = call(harmonicity, "Get mean", s, e)
                if not np.isnan(hnr_val):
                    res["hnr"] = float(hnr_val)
            except Exception:
                pass

            # CPPS
            if len(t_cpps) > 0 and len(cpps_arr) > 0:
                mask_c = (t_cpps >= s) & (t_cpps <= e)
                if np.any(mask_c):
                    v = cpps_arr[mask_c]
                    valid_v = v[~np.isnan(v)]
                    if len(valid_v) > 0:
                        res["cpps"] = float(np.mean(valid_v))

            # H1*-H2*
            if len(t_h1h2) > 0 and len(h1h2_arr) > 0:
                mask_h = (t_h1h2 >= s) & (t_h1h2 <= e)
                if np.any(mask_h):
                    v = h1h2_arr[mask_h]
                    valid_v = v[~np.isnan(v)]
                    if len(valid_v) > 0:
                        res["h1h2_star"] = float(np.mean(valid_v))

            # SHR
            if len(p_times) > 0:
                mask_p = (p_times >= s) & (p_times <= e)
                if np.any(mask_p):
                    sub_f = p_freqs_all[mask_p]
                    v_freqs = sub_f[sub_f > 0]
                    if len(v_freqs) >= 2:
                        f_diffs = np.abs(np.diff(v_freqs))
                        res["shr"] = float(np.mean(f_diffs > 25.0))

            file_results.append(res)

        return file_results

    except Exception as e:
        logger.warning(f"Error extracting features for {audio_path}: {e}")
        return empty_results


def extract_batch_physical_features(
    manifest_df: pd.DataFrame, max_workers: Optional[int] = None
) -> pd.DataFrame:
    """Extract physical acoustic metrics for intervals grouped by audio file using continuous sentence-level Praat extraction."""
    total = len(manifest_df)
    if total == 0:
        return pd.DataFrame()

    file_map: Dict[str, List[Tuple[str, float, float, str, str]]] = {}
    for _, row in manifest_df.iterrows():
        a_path = str(row["audio_path"])
        file_map.setdefault(a_path, []).append((
            str(row["interval_key"]),
            float(row["start_s"]),
            float(row["end_s"]),
            str(row.get("dataset_id", "cs_dialogue")),
            str(row.get("file_key", Path(str(row["audio_path"])).stem)),
        ))

    file_items = list(file_map.items())
    num_files = len(file_items)
    workers = max_workers or min(6, os.cpu_count() or 4)
    logger.info(
        f"Extracting acoustic features across {workers} CPU workers "
        f"({num_files} unique audio files, {total} total intervals)..."
    )

    import concurrent.futures

    all_rows = []
    chunksize = max(1, num_files // (workers * 4))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for idx, file_records in enumerate(executor.map(_extract_single_audio_file, file_items, chunksize=chunksize)):
            all_rows.extend(file_records)
            if (idx + 1) % 20 == 0 or idx == num_files - 1:
                logger.info(f"Acoustic features progress: [{idx + 1}/{num_files} files] ({(idx + 1) / num_files * 100:.1f}%)")

    return pd.DataFrame(all_rows)


def run_acoustic_correlation_audit(
    manifest_path: Path,
    checkpoint_path: Optional[Path],
    preds_csv: Optional[Path],
    acoustic_cache_path: Path,
    output_dir: Path,
    limit_samples: Optional[int] = None,
    device: Optional[str] = None,
    force_refresh: bool = False,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    import torch
    from wavfry.eval.cache_manager import IncrementalCSVCache, attach_interval_keys, dump_model_predictions
    from wavfry.eval.acoustic_degradation import AcousticDegradationEngine
    from wavfry.eval.unified import build_test_manifest, load_model_from_checkpoint

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir.mkdir(parents=True, exist_ok=True)
    full_manifest = pd.read_csv(manifest_path, low_memory=False)
    test_manifest = build_test_manifest(full_manifest, require_label=False)
    test_manifest = attach_interval_keys(test_manifest)

    if limit_samples is not None and limit_samples > 0:
        test_manifest = test_manifest.iloc[:limit_samples].copy().reset_index(drop=True)
        logger.info(f"Limiting evaluation to {len(test_manifest)} test samples.")

    # 1. Obtain Clean test predictions
    if preds_csv and Path(preds_csv).exists():
        logger.info(f"Loading predictions from {preds_csv}")
        preds_df = pd.read_csv(preds_csv)
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
            checkpoint_path=checkpoint_path,
        )
    else:
        raise ValueError("Must provide either --checkpoint or --preds-csv.")

    clean_preds = attach_interval_keys(clean_preds)

    # 2. Incremental Acoustic Feature Cache
    ac_cache = IncrementalCSVCache(acoustic_cache_path)
    if force_refresh:
        logger.info(
            f"Force-refresh requested: extracting acoustic features for all {len(test_manifest)} intervals "
            "using continuous sentence-level extraction..."
        )
        all_feats_df = extract_batch_physical_features(test_manifest)
        all_feats_df.to_csv(acoustic_cache_path, index=False)
        matched_feats = all_feats_df
    else:
        cached_feats, missing_feats = ac_cache.partition_manifest(test_manifest)
        if not missing_feats.empty:
            logger.info(f"Extracting acoustic features for {len(missing_feats)} missing intervals...")
            new_feats_df = extract_batch_physical_features(missing_feats)
            ac_cache.append_records(new_feats_df)
        matched_feats = ac_cache.get_matched(test_manifest)

    # 3. Merge predictions with acoustic features
    merged_df = clean_preds.merge(
        matched_feats[
            ["interval_key", "jitter_local", "shimmer_local", "hnr", "cpps", "h1h2_star", "shr"]
        ],
        on="interval_key",
        how="inner",
    )

    # 4. Statistical Correlation Matrix (Table 8b)
    metrics_to_correlate = {
        "creaky": [
            ("SHR", "shr", "positive"),
            ("Jitter", "jitter_local", "positive"),
            ("Shimmer", "shimmer_local", "positive"),
            ("H1*-H2*", "h1h2_star", "negative"),
        ],
        "breathy": [
            ("CPPS", "cpps", "negative"),
            ("HNR", "hnr", "negative"),
            ("H1*-H2*", "h1h2_star", "positive"),
        ],
    }

    table8b_data: Dict[str, Any] = {}

    for task_name, feat_specs in metrics_to_correlate.items():
        prob_col = f"prob_{task_name}"
        if prob_col not in merged_df.columns:
            continue

        label_col = f"label_{task_name}"
        if label_col in merged_df.columns and merged_df[label_col].notna().any():
            task_df = merged_df[merged_df[label_col].notna()]
        else:
            task_df = merged_df

        probs = task_df[prob_col].values
        task_results = []

        for feat_label, col_name, expected_trend in feat_specs:
            if col_name not in task_df.columns:
                continue
            vals = task_df[col_name].values
            valid = ~np.isnan(probs) & ~np.isnan(vals)

            if np.sum(valid) < 10:
                task_results.append({
                    "Feature": feat_label,
                    "Expected": expected_trend,
                    "Spearman_rho": 0.0,
                    "Spearman_p": 1.0,
                    "Pearson_r": 0.0,
                    "Pearson_p": 1.0,
                    "N": int(np.sum(valid)),
                })
                continue

            rho, p_rho = spearmanr(probs[valid], vals[valid])
            r_val, p_r = pearsonr(probs[valid], vals[valid])

            task_results.append({
                "Feature": feat_label,
                "Expected": expected_trend,
                "Spearman_rho": round(float(rho), 4),
                "Spearman_p": float(p_rho),
                "Pearson_r": round(float(r_val), 4),
                "Pearson_p": float(p_r),
                "N": int(np.sum(valid)),
            })

        table8b_data[task_name] = task_results

    # Add physiological acoustic meaning descriptions
    meaning_map = {
        ("creaky", "SHR"): "Subharmonic-to-Harmonic Ratio (Glottal period doubling / creak)",
        ("creaky", "Jitter"): "Cycle-to-cycle F0 perturbation / vocal fry irregularity",
        ("creaky", "Shimmer"): "Cycle-to-cycle amplitude perturbation",
        ("creaky", "H1*-H2*"): "Negative spectral tilt (steep glottal closure / medial compression)",
        ("breathy", "CPPS"): "Cepstral Peak Prominence Smoothed (Harmonic periodicity vs aspiration)",
        ("breathy", "HNR"): "Harmonics-to-Noise Ratio (Turbulence aspiration noise)",
        ("breathy", "H1*-H2*"): "Positive spectral tilt (incomplete glottal closure / glottal chink)",
    }

    tidy_table8b_rows = []
    for t_name, rows in table8b_data.items():
        for r in rows:
            meaning = meaning_map.get((t_name, r["Feature"]), "Acoustic-physical parameter")
            tidy_table8b_rows.append({
                "Task": t_name,
                "Feature": r["Feature"],
                "Expected-Trend": r["Expected"],
                "Spearman-Rho": r["Spearman_rho"],
                "Spearman-p": r["Spearman_p"],
                "Pearson-R": r["Pearson_r"],
                "Pearson-p": r["Pearson_p"],
                "N-Samples": r["N"],
                "Physiological-Acoustic-Meaning": meaning,
            })

    # 5. Add Quantile Stratification Bins for Figure 8 Plotting
    bins = [0.0, 0.25, 0.50, 0.75, 1.0]
    bin_labels = ["[0.0, 0.25)", "[0.25, 0.50)", "[0.50, 0.75)", "[0.75, 1.0]"]

    merged_df["Creaky_Bin"] = pd.cut(
        merged_df["prob_creaky"], bins=bins, labels=bin_labels, include_lowest=True
    )
    if "prob_breathy" in merged_df.columns:
        merged_df["Breathy_Bin"] = pd.cut(
            merged_df["prob_breathy"], bins=bins, labels=bin_labels, include_lowest=True
        )

    # 6. Save JSON & CSV
    json_path = output_dir / "acoustic_correlations_metrics.json"
    summary_json_path = output_dir / "acoustic_correlations_summary.json"
    legacy_csv_path = output_dir / "acoustic_correlations.csv"
    table8b_csv_path = output_dir / "table8b_acoustic_correlations.csv"
    md_path = output_dir / "table8b_acoustic_correlations.md"

    # Export machine-readable JSON
    summary_json_payload = {
        "condition": "clean",
        "description": "Physical acoustic interpretability evaluated strictly under clean conditions to prevent Praat pitch extraction breakdown.",
        "tasks": table8b_data,
        "tidy_records": tidy_table8b_rows,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(table8b_data, f, indent=2)
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json_payload, f, indent=2)

    # Export machine-readable CSVs
    pd.DataFrame(tidy_table8b_rows).to_csv(table8b_csv_path, index=False, encoding="utf-8")
    merged_df.to_csv(legacy_csv_path, index=False, encoding="utf-8")

    # Export Markdown Report Byproduct
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Table 8b: Acoustic-Physical Interpretability & Gold-Standard Parameter Correlations\n\n")
        f.write("- **Condition**: `Clean` (Strict clean-only to prevent Praat pitch/harmonic tracking breakdown under noise)\n")
        f.write("- **Gold-Standard Tools**: Parselmouth / Praat physical acoustic extractors\n\n")

        for t_name, rows in table8b_data.items():
            f.write(f"## Task: `{t_name.upper()}`\n\n")
            f.write("| Acoustic Feature | Expected Trend | Spearman $\\rho$ | Spearman $p$ | Pearson $r$ | Pearson $p$ | N | Physiological Mechanism |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |\n")
            for r in rows:
                meaning = meaning_map.get((t_name, r["Feature"]), "")
                p_rho_str = f"{r['Spearman_p']:.2e}" if r['Spearman_p'] < 0.001 else f"{r['Spearman_p']:.4f}"
                p_r_str = f"{r['Pearson_p']:.2e}" if r['Pearson_p'] < 0.001 else f"{r['Pearson_p']:.4f}"
                f.write(
                    f"| **{r['Feature']}** | `{r['Expected']}` | **{r['Spearman_rho']:+.4f}** | {p_rho_str} | "
                    f"{r['Pearson_r']:+.4f} | {p_r_str} | {r['N']} | {meaning} |\n"
                )
            f.write("\n---\n\n")

    logger.info(f"Saved Table 8b correlation metrics to {json_path} and {summary_json_path}")
    logger.info(f"Saved Table 8b CSV to {table8b_csv_path}")
    logger.info(f"Saved Figure 8 distribution CSV to {legacy_csv_path}")
    logger.info(f"Saved Table 8b Markdown report to {md_path}")

    return table8b_data, merged_df


def main():
    parser = argparse.ArgumentParser(
        description="Acoustic-Physical Interpretability & Correlation Audit (Table 8b & Figure 8)."
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint.")
    parser.add_argument("--manifest", type=str, default="manifests/segments_multi.csv", help="Test manifest path.")
    parser.add_argument("--preds-csv", type=str, default=None, help="Pre-computed predictions CSV path.")
    parser.add_argument(
        "--acoustic-cache",
        type=str,
        default="data/experiments/cache/static_acoustic_features_cache.csv",
        help="Incremental acoustic features CSV cache.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/experiments/interpretability", help="Output directory."
    )
    parser.add_argument("--limit-samples", type=int, default=None, help="Limit samples for fast test.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force recomputing all acoustic features using continuous full-sentence Praat extraction.",
    )

    args = parser.parse_args()

    table8b_data, _ = run_acoustic_correlation_audit(
        manifest_path=Path(args.manifest),
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        preds_csv=Path(args.preds_csv) if args.preds_csv else None,
        acoustic_cache_path=Path(args.acoustic_cache),
        output_dir=Path(args.output_dir),
        limit_samples=args.limit_samples,
        device=args.device,
        force_refresh=args.force_refresh,
    )

    print("\n=== Table 8b: Acoustic-Physical Correlation Validation ===")
    for task_name, rows in table8b_data.items():
        print(f"\n--- Task: {task_name.upper()} ---")
        tdf = pd.DataFrame(rows)
        print(tdf.to_string(index=False))


if __name__ == "__main__":
    main()

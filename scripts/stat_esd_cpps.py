#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stratified sampling and CPPS acoustic distribution analysis on Emotion Speech Dataset (ESD).

Analyzes CPPS distribution across:
- 5 Emotions: Angry, Happy, Neutral, Sad, Surprise
- 2 Languages: Mandarin (0001-0010), English (0011-0020)
- Gender: Female vs Male
- Both Voiced-Segment CPPS (acoustic vocal fold periodicity) and Whole-Utterance CPPS.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EMOTIONS = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]

# Known ESD Speaker Genders (from F0 measurements & ESD paper)
# Mandarin: 0001-0005 (F: 0001, 0002, 0003, 0007, 0009; M: 0004, 0005, 0006, 0008, 0010)
# English:  0011-0020 (F: 0015, 0016, 0017, 0018; M: 0011, 0012, 0013, 0014, 0019, 0020)
SPEAKER_GENDER = {
    "0001": "F", "0002": "F", "0003": "F", "0004": "M", "0005": "M",
    "0006": "M", "0007": "F", "0008": "M", "0009": "F", "0010": "M",
    "0011": "M", "0012": "M", "0013": "M", "0014": "M", "0015": "F",
    "0016": "F", "0017": "F", "0018": "F", "0019": "M", "0020": "M",
}

SPEAKER_LANG = {
    spk: ("zh" if int(spk) <= 10 else "en")
    for spk in [f"{i:04d}" for i in range(1, 21)]
}


def analyze_file_cpps(args_tuple: Tuple[str, str, str, str, str]) -> Dict[str, Any]:
    """Compute Praat whole-utterance CPPS and voiced-frame CPPS for one audio file."""
    wav_path_str, spk, emotion, lang, gender = args_tuple
    try:
        snd = parselmouth.Sound(wav_path_str)
        dur = float(snd.duration)

        # 1. Whole sentence PowerCepstrogram
        cep = call(snd, "To PowerCepstrogram", 60.0, 0.005, 5000.0, 50.0)
        cpps_utterance = call(
            cep,
            "Get CPPS",
            True,
            0.01,
            0.001,
            60.0,
            600.0,
            0.05,
            "Parabolic",
            0.001,
            0.0,
            "Straight",
            "Robust",
        )

        # 2. Extract Pitch to identify voiced frames
        pitch = snd.to_pitch(time_step=0.010, pitch_floor=60.0, pitch_ceiling=600.0)
        n_frames = call(pitch, "Get number of frames")

        voiced_cpps_list = []
        f0_list = []
        for i in range(1, n_frames + 1):
            f0 = call(pitch, "Get value in frame", i, "Hertz")
            if f0 and not np.isnan(f0) and f0 > 0:
                f0_list.append(f0)
                t = call(pitch, "Get time from frame number", i)
                try:
                    sl = call(cep, "To PowerCepstrum (slice)", float(t))
                    fc = call(
                        sl,
                        "Get peak prominence",
                        60.0,
                        600.0,
                        "Parabolic",
                        0.001,
                        0.0,
                        "Straight",
                        "Robust",
                    )
                    voiced_cpps_list.append(fc)
                except Exception:
                    pass

        voiced_ratio = len(f0_list) / max(1, n_frames)
        median_f0 = float(np.median(f0_list)) if f0_list else float("nan")

        if voiced_cpps_list:
            v_mean = float(np.mean(voiced_cpps_list))
            v_median = float(np.median(voiced_cpps_list))
            v_std = float(np.std(voiced_cpps_list))
            v_min = float(np.min(voiced_cpps_list))
            v_max = float(np.max(voiced_cpps_list))
            v_p10 = float(np.percentile(voiced_cpps_list, 10))
            v_p25 = float(np.percentile(voiced_cpps_list, 25))
            v_p75 = float(np.percentile(voiced_cpps_list, 75))
            v_p90 = float(np.percentile(voiced_cpps_list, 90))
        else:
            v_mean = v_median = v_std = v_min = v_max = v_p10 = v_p25 = v_p75 = v_p90 = float("nan")

        return {
            "file": Path(wav_path_str).stem,
            "speaker": spk,
            "emotion": emotion,
            "language": lang,
            "gender": gender,
            "duration": dur,
            "voiced_ratio": voiced_ratio,
            "median_f0": median_f0,
            "cpps_utterance": float(cpps_utterance),
            "cpps_voiced_mean": v_mean,
            "cpps_voiced_median": v_median,
            "cpps_voiced_std": v_std,
            "cpps_voiced_min": v_min,
            "cpps_voiced_max": v_max,
            "cpps_voiced_p10": v_p10,
            "cpps_voiced_p25": v_p25,
            "cpps_voiced_p75": v_p75,
            "cpps_voiced_p90": v_p90,
            "voiced_frames_count": len(voiced_cpps_list),
        }
    except Exception as e:
        return {
            "file": Path(wav_path_str).stem,
            "speaker": spk,
            "emotion": emotion,
            "language": lang,
            "gender": gender,
            "error": str(e),
        }


def stratified_sampling(
    base_dir: Path,
    samples_per_cell: int = 10,
    seed: int = 42,
) -> List[Tuple[str, str, str, str, str]]:
    """Draw a balanced stratified sample across 20 speakers x 5 emotions."""
    rng = random.Random(seed)
    tasks = []

    spk_dirs = sorted([
        d for d in base_dir.iterdir()
        if d.is_dir() and d.name.isdigit()
    ], key=lambda d: d.name)

    for spk_d in spk_dirs:
        spk = spk_d.name
        lang = SPEAKER_LANG.get(spk, "und")
        gender = SPEAKER_GENDER.get(spk, "U")

        for emo in EMOTIONS:
            emo_dir = spk_d / emo
            if not emo_dir.exists():
                continue
            wavs = sorted(emo_dir.glob("*.wav"))
            if not wavs:
                continue

            chosen = rng.sample(wavs, min(len(wavs), samples_per_cell))
            for w in chosen:
                tasks.append((str(w), spk, emo, lang, gender))

    return tasks


def main():
    parser = argparse.ArgumentParser(
        description="Stratified CPPS distribution analysis on Emotion Speech Dataset (ESD)."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(r"E:\Corpus\Breathy\Emotion Speech Dataset"),
        help="Path to Emotion Speech Dataset root directory.",
    )
    parser.add_argument(
        "--samples-per-cell",
        type=int,
        default=10,
        help="Samples per (speaker x emotion) cell (default: 10 -> 20 spk x 5 emo x 10 = 1000 files).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(os.cpu_count() or 4, 8),
        help="Parallel worker count (default: 8).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible stratified sampling.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(r"E:\Coding\WavFry\results\esd_cpps_analysis.csv"),
        help="Path to save detailed per-file measurements.",
    )

    args = parser.parse_args()

    base_dir = args.dataset_dir
    if not base_dir.exists():
        sys.exit(f"Error: dataset-dir does not exist: {base_dir}")

    print("=" * 70)
    print("Stratified CPPS Acoustic Analysis on Emotion Speech Dataset (ESD)")
    print(f"Dataset Dir:       {base_dir}")
    print(f"Samples per Cell:  {args.samples_per_cell} (target total: {args.samples_per_cell * 20 * 5})")
    print(f"Workers:           {args.workers}")
    print(f"Random Seed:       {args.seed}")
    print("=" * 70)

    # 1. Stratified Sampling
    sampled_tasks = stratified_sampling(base_dir, samples_per_cell=args.samples_per_cell, seed=args.seed)
    print(f"Drawn {len(sampled_tasks)} stratified audio samples across 20 speakers x 5 emotions.")

    # 2. Parallel Extraction
    t0 = time.time()
    results = []
    print(f"\nExtracting CPPS on {len(sampled_tasks)} files using {args.workers} processes...")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for r in tqdm(executor.map(analyze_file_cpps, sampled_tasks, chunksize=16), total=len(sampled_tasks), mininterval=5.0, ncols=80):
            results.append(r)

    elapsed = time.time() - t0
    print(f"\nExtraction completed in {elapsed:.1f}s ({len(sampled_tasks)/max(0.001, elapsed):.1f} files/s).")

    # 3. Analyze Data
    df = pd.DataFrame(results)
    if "error" in df.columns:
        valid_df = df[df["error"].isna()].copy()
    else:
        valid_df = df.copy()

    # Save detailed CSV
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    valid_df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"Detailed measurements saved to: {args.output_csv}")

    # 4. Generate Aggregated Statistics by Emotion
    print("\n" + "=" * 70)
    print("1. CPPS Distribution Across EMOTIONS (Voiced Segments)")
    print("=" * 70)

    # Metrics on voiced CPPS mean
    emo_stats = []
    for emo in EMOTIONS:
        sub = valid_df[valid_df["emotion"] == emo]
        v_mean = sub["cpps_voiced_mean"]
        u_mean = sub["cpps_utterance"]
        
        # Check percentage >= 15 dB
        pct_ge_15 = (v_mean >= 15.0).mean() * 100.0
        pct_ge_18 = (v_mean >= 18.0).mean() * 100.0
        pct_lt_12 = (v_mean < 12.0).mean() * 100.0

        emo_stats.append({
            "Emotion": emo,
            "Count": len(sub),
            "Voiced CPPS Mean": f"{v_mean.mean():.2f} dB",
            "Median (P50)": f"{v_mean.median():.2f} dB",
            "Std": f"{v_mean.std():.2f}",
            "P10": f"{v_mean.quantile(0.10):.2f} dB",
            "P90": f"{v_mean.quantile(0.90):.2f} dB",
            "% >= 15dB": f"{pct_ge_15:.1f}%",
            "% < 12dB (Breathy-like)": f"{pct_lt_12:.1f}%",
            "Utterance CPPS": f"{u_mean.mean():.2f} dB",
        })

    stat_df = pd.DataFrame(emo_stats)
    print(stat_df.to_string(index=False))

    # 5. Language & Gender Breakdown
    print("\n" + "=" * 70)
    print("2. Language & Gender Breakdown (Voiced CPPS Mean)")
    print("=" * 70)
    
    subgroup_stats = []
    for grp_col in ["language", "gender"]:
        for grp_val, grp_sub in valid_df.groupby(grp_col):
            v_mean = grp_sub["cpps_voiced_mean"]
            subgroup_stats.append({
                "Group": f"{grp_col}={grp_val}",
                "Count": len(grp_sub),
                "Mean": f"{v_mean.mean():.2f} dB",
                "Median": f"{v_mean.median():.2f} dB",
                "P10": f"{v_mean.quantile(0.10):.2f} dB",
                "P90": f"{v_mean.quantile(0.90):.2f} dB",
                "% >= 15dB": f"{(v_mean >= 15.0).mean() * 100.0:.1f}%",
            })
    print(pd.DataFrame(subgroup_stats).to_string(index=False))

    # 6. Emotion x Gender Cross Table
    print("\n" + "=" * 70)
    print("3. Emotion x Gender Cross-Table: Mean Voiced CPPS (dB)")
    print("=" * 70)
    pivot = valid_df.pivot_table(
        index="emotion",
        columns="gender",
        values="cpps_voiced_mean",
        aggfunc=["mean", "median"],
    )
    print(pivot.round(2))

    # 7. Low-CPPS analysis (potential breathy candidates)
    low_cpps = valid_df[valid_df["cpps_voiced_mean"] < 12.0]
    print("\n" + "=" * 70)
    print(f"4. Potential Breathy Voice Analysis (Voiced CPPS < 12 dB): {len(low_cpps)} files")
    print("=" * 70)
    if not low_cpps.empty:
        print("Emotion distribution of low-CPPS utterances:")
        print(low_cpps["emotion"].value_counts(normalize=True).mul(100).round(1).astype(str) + "%")
        print("\nSpeaker distribution of low-CPPS utterances:")
        print(low_cpps["speaker"].value_counts().head(5))


if __name__ == "__main__":
    main()

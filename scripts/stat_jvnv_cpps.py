#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Acoustic CPPS distribution analysis on JVNV (Japanese Versatile Nonverbal Voice) Dataset.

Analyzes CPPS distribution across:
- 6 Emotions: anger, disgust, fear, happy, sad, surprise
- 4 Speakers: F1 (female), F2 (female), M1 (male), M2 (male)
- 2 Sessions: regular vs free
- Verbal speech segments vs Nonverbal Vocalization (NV) segments (laughter, cries, gasps, etc.)
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
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

EMOTIONS = ["anger", "disgust", "fear", "happy", "sad", "surprise"]
SPEAKERS = ["F1", "F2", "M1", "M2"]
GENDERS = {"F1": "F", "F2": "F", "M1": "M", "M2": "M"}


def load_nv_intervals(nv_label_path: Path) -> List[Tuple[float, float]]:
    """Parse non-verbal start and end timestamps from nv_label file."""
    intervals = []
    if not nv_label_path.exists():
        return intervals
    try:
        with open(nv_label_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        t0 = float(parts[0])
                        t1 = float(parts[1])
                        intervals.append((min(t0, t1), max(t0, t1)))
                    except ValueError:
                        pass
    except Exception:
        pass
    return intervals


def is_in_nv(t: float, intervals: List[Tuple[float, float]]) -> bool:
    for t0, t1 in intervals:
        if t0 <= t <= t1:
            return True
    return False


def analyze_jvnv_file(args_tuple: Tuple[str, str, str, str, str, str]) -> Dict[str, Any]:
    """Compute Praat CPPS metrics for one JVNV audio file."""
    wav_path_str, spk, gender, emo, session, nv_txt_str = args_tuple
    try:
        snd = parselmouth.Sound(wav_path_str)
        dur = float(snd.duration)

        nv_intervals = load_nv_intervals(Path(nv_txt_str)) if nv_txt_str else []
        total_nv_dur = sum(t1 - t0 for t0, t1 in nv_intervals)

        # 1. Whole utterance PowerCepstrogram
        cep = call(snd, "To PowerCepstrogram", 60.0, 0.005, 5000.0, 50.0)
        cpps_utterance = call(
            cep,
            "Get CPPS",
            True,
            0.01,
            0.001,
            60.0,
            700.0,
            0.05,
            "Parabolic",
            0.001,
            0.0,
            "Straight",
            "Robust",
        )

        # 2. Extract Pitch to identify voiced frames
        pitch = snd.to_pitch(time_step=0.010, pitch_floor=60.0, pitch_ceiling=700.0)
        n_frames = call(pitch, "Get number of frames")

        voiced_cpps_list = []
        nv_voiced_cpps_list = []
        verbal_voiced_cpps_list = []
        f0_list = []

        for i in range(1, n_frames + 1):
            f0 = call(pitch, "Get value in frame", i, "Hertz")
            if f0 and not np.isnan(f0) and f0 > 0:
                f0_list.append(f0)
                t = float(call(pitch, "Get time from frame number", i))
                try:
                    sl = call(cep, "To PowerCepstrum (slice)", t)
                    fc = float(
                        call(
                            sl,
                            "Get peak prominence",
                            60.0,
                            700.0,
                            "Parabolic",
                            0.001,
                            0.0,
                            "Straight",
                            "Robust",
                        )
                    )
                    voiced_cpps_list.append(fc)
                    if is_in_nv(t, nv_intervals):
                        nv_voiced_cpps_list.append(fc)
                    else:
                        verbal_voiced_cpps_list.append(fc)
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

        nv_mean = float(np.mean(nv_voiced_cpps_list)) if nv_voiced_cpps_list else float("nan")
        verbal_mean = float(np.mean(verbal_voiced_cpps_list)) if verbal_voiced_cpps_list else float("nan")

        return {
            "file": Path(wav_path_str).stem,
            "speaker": spk,
            "gender": gender,
            "emotion": emo,
            "session": session,
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
            "cpps_nv_mean": nv_mean,
            "cpps_verbal_mean": verbal_mean,
            "nv_duration": total_nv_dur,
            "nv_voiced_frames": len(nv_voiced_cpps_list),
            "verbal_voiced_frames": len(verbal_voiced_cpps_list),
            "voiced_frames_count": len(voiced_cpps_list),
        }
    except Exception as e:
        return {
            "file": Path(wav_path_str).stem,
            "speaker": spk,
            "gender": gender,
            "emotion": emo,
            "session": session,
            "error": str(e),
        }


def collect_jvnv_tasks(base_dir: Path) -> List[Tuple[str, str, str, str, str, str]]:
    """Scan all WAV files and match them to nv_label files."""
    tasks = []
    nv_dir = base_dir / "nv_label"

    for spk in SPEAKERS:
        spk_dir = base_dir / spk
        if not spk_dir.exists():
            continue
        gender = GENDERS.get(spk, "U")

        for emo in EMOTIONS:
            emo_dir = spk_dir / emo
            if not emo_dir.exists():
                continue

            for session in ["regular", "free"]:
                sess_dir = emo_dir / session
                if not sess_dir.exists():
                    continue

                for wav_path in sorted(sess_dir.glob("*.wav")):
                    stem = wav_path.stem
                    nv_file = nv_dir / spk / f"{stem}.txt"
                    nv_str = str(nv_file) if nv_file.exists() else ""
                    tasks.append((str(wav_path), spk, gender, emo, session, nv_str))

    return tasks


def main():
    parser = argparse.ArgumentParser(
        description="Acoustic CPPS distribution analysis on JVNV Dataset."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(r"E:\Corpus\Breathy\jvnv_v1"),
        help="Path to JVNV dataset root directory.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(os.cpu_count() or 4, 8),
        help="Parallel worker count (default: 8).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(r"E:\Coding\WavFry\results\jvnv_cpps_analysis.csv"),
        help="Path to save detailed per-file measurements.",
    )

    args = parser.parse_args()

    base_dir = args.dataset_dir
    if not base_dir.exists():
        sys.exit(f"Error: dataset-dir does not exist: {base_dir}")

    print("=" * 70)
    print("Acoustic CPPS Distribution Analysis on JVNV Dataset")
    print(f"Dataset Dir: {base_dir}")
    print(f"Workers:     {args.workers}")
    print("=" * 70)

    # 1. Collect Tasks
    tasks = collect_jvnv_tasks(base_dir)
    print(f"Discovered {len(tasks)} audio files across 4 speakers and 6 emotions.")

    # 2. Parallel Extraction
    t0 = time.time()
    results = []
    print(f"\nExtracting CPPS on {len(tasks)} files using {args.workers} processes...")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for r in tqdm(executor.map(analyze_jvnv_file, tasks, chunksize=16), total=len(tasks), mininterval=5.0, ncols=80):
            results.append(r)

    elapsed = time.time() - t0
    print(f"\nExtraction completed in {elapsed:.1f}s ({len(tasks)/max(0.001, elapsed):.1f} files/s).")

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

    # 4. Overall Emotion Statistics
    print("\n" + "=" * 70)
    print("1. CPPS Distribution Across EMOTIONS (Voiced Segments)")
    print("=" * 70)

    emo_stats = []
    for emo in EMOTIONS:
        sub = valid_df[valid_df["emotion"] == emo]
        v_mean = sub["cpps_voiced_mean"]
        u_mean = sub["cpps_utterance"]
        nv_mean = sub["cpps_nv_mean"].dropna()
        verbal_mean = sub["cpps_verbal_mean"].dropna()

        pct_ge_15 = (v_mean >= 15.0).mean() * 100.0
        pct_lt_12 = (v_mean < 12.0).mean() * 100.0

        emo_stats.append({
            "Emotion": emo,
            "Count": len(sub),
            "Voiced CPPS Mean": f"{v_mean.mean():.2f} dB",
            "Median (P50)": f"{v_mean.median():.2f} dB",
            "Std": f"{v_mean.std():.2f}",
            "P10": f"{v_mean.quantile(0.10):.2f} dB",
            "P90": f"{v_mean.quantile(0.90):.2f} dB",
            "NV CPPS Mean": f"{nv_mean.mean():.2f} dB" if len(nv_mean) else "N/A",
            "Verbal CPPS": f"{verbal_mean.mean():.2f} dB" if len(verbal_mean) else "N/A",
            "% >= 15dB": f"{pct_ge_15:.1f}%",
        })

    print(pd.DataFrame(emo_stats).to_string(index=False))

    # 5. Non-Verbal (NV) vs Verbal Speech Comparison
    print("\n" + "=" * 70)
    print("2. Non-Verbal (NV) vs Verbal Speech Acoustic Contrast")
    print("=" * 70)
    nv_comp = []
    for emo in EMOTIONS:
        sub = valid_df[valid_df["emotion"] == emo]
        valid_pairs = sub.dropna(subset=["cpps_nv_mean", "cpps_verbal_mean"])
        nv_m = valid_pairs["cpps_nv_mean"].mean()
        vb_m = valid_pairs["cpps_verbal_mean"].mean()
        diff = nv_m - vb_m
        nv_comp.append({
            "Emotion": emo,
            "Pairs": len(valid_pairs),
            "NV Segment CPPS": f"{nv_m:.2f} dB",
            "Verbal Segment CPPS": f"{vb_m:.2f} dB",
            "Delta (NV - Verbal)": f"{diff:+.2f} dB",
        })
    print(pd.DataFrame(nv_comp).to_string(index=False))

    # 6. Speaker Breakdown
    print("\n" + "=" * 70)
    print("3. Speaker & Gender Breakdown (Voiced CPPS Mean)")
    print("=" * 70)
    spk_stats = []
    for spk in SPEAKERS:
        sub = valid_df[valid_df["speaker"] == spk]
        v_mean = sub["cpps_voiced_mean"]
        spk_stats.append({
            "Speaker": spk,
            "Gender": GENDERS[spk],
            "Count": len(sub),
            "Mean": f"{v_mean.mean():.2f} dB",
            "Median": f"{v_mean.median():.2f} dB",
            "Std": f"{v_mean.std():.2f}",
            "P10": f"{v_mean.quantile(0.10):.2f} dB",
            "P90": f"{v_mean.quantile(0.90):.2f} dB",
            "Min": f"{v_mean.min():.2f} dB",
        })
    print(pd.DataFrame(spk_stats).to_string(index=False))

    # 7. Emotion x Speaker Cross Table
    print("\n" + "=" * 70)
    print("4. Emotion x Speaker Cross-Table: Mean Voiced CPPS (dB)")
    print("=" * 70)
    pivot = valid_df.pivot_table(
        index="emotion",
        columns="speaker",
        values="cpps_voiced_mean",
        aggfunc="mean",
    )
    print(pivot.round(2))

    # 8. Session Breakdown (regular vs free)
    print("\n" + "=" * 70)
    print("5. Session Breakdown (Regular vs Phrase-Free)")
    print("=" * 70)
    sess_stats = []
    for sess, sub in valid_df.groupby("session"):
        v_mean = sub["cpps_voiced_mean"]
        sess_stats.append({
            "Session": sess,
            "Count": len(sub),
            "Mean": f"{v_mean.mean():.2f} dB",
            "Median": f"{v_mean.median():.2f} dB",
            "Std": f"{v_mean.std():.2f}",
        })
    print(pd.DataFrame(sess_stats).to_string(index=False))


if __name__ == "__main__":
    main()

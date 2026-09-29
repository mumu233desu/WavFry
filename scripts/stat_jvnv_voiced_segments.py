#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fine-grained Continuous Voiced Segment (Scheme 1) CPPS Analysis on JVNV Dataset.

Scheme 1:
- Extracts contiguous voiced segments (Continuous Voiced Islands / Syllabic Nuclei, dur >= 60ms) using Praat F0 tracking.
- Applies Core Trimming (drops boundary onset/offset frames, 10-20ms) to isolate steady-state vowel phonation.
- Distinguishes between Nonverbal Vocalization (NV) segments vs Verbal reading segments.
- Analyzes across 6 Emotions, 4 Speakers, and NV vs Verbal segments.
"""

from __future__ import annotations

import argparse
import os
import random
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


def is_segment_in_nv(t_start: float, t_end: float, nv_intervals: List[Tuple[float, float]]) -> bool:
    """Check if the segment center or significant part is in the NV interval."""
    t_mid = 0.5 * (t_start + t_end)
    for n0, n1 in nv_intervals:
        if n0 <= t_mid <= n1:
            return True
        # Or overlap >= 50%
        overlap = max(0.0, min(t_end, n1) - max(t_start, n0))
        if overlap / max(0.001, t_end - t_start) >= 0.5:
            return True
    return False


def extract_voiced_segments_for_file(args_tuple: Tuple[str, str, str, str, str, str]) -> List[Dict[str, Any]]:
    """Extract all continuous voiced segments for a single JVNV audio file."""
    wav_path_str, spk, gender, emo, session, nv_txt_str = args_tuple
    file_stem = Path(wav_path_str).stem
    nv_intervals = load_nv_intervals(Path(nv_txt_str)) if nv_txt_str else []

    segments_out = []
    try:
        snd = parselmouth.Sound(wav_path_str)
        # PowerCepstrogram
        cep = call(snd, "To PowerCepstrogram", 60.0, 0.005, 5000.0, 50.0)
        # Pitch tracking
        pitch = snd.to_pitch(time_step=0.010, pitch_floor=60.0, pitch_ceiling=700.0)
        n_frames = call(pitch, "Get number of frames")

        f0s = [call(pitch, "Get value in frame", i, "Hertz") for i in range(1, n_frames + 1)]
        times = [float(call(pitch, "Get time from frame number", i)) for i in range(1, n_frames + 1)]
        voiced = [f is not None and not np.isnan(f) and f > 0 for f in f0s]

        # Contiguous segment identification
        in_seg = False
        start_idx = 0
        seg_counter = 0

        for i, v in enumerate(voiced):
            if v and not in_seg:
                in_seg = True
                start_idx = i
            elif not v and in_seg:
                in_seg = False
                n_seg_frames = i - start_idx
                # Keep segments >= 60ms (6 frames of 10ms)
                if n_seg_frames >= 6:
                    t_start = times[start_idx]
                    t_end = times[i - 1]
                    dur_ms = (t_end - t_start + 0.010) * 1000.0

                    seg_cpps = []
                    seg_f0 = []
                    for k in range(start_idx, i):
                        t = times[k]
                        seg_f0.append(f0s[k])
                        try:
                            sl = call(cep, "To PowerCepstrum (slice)", t)
                            c = float(
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
                            seg_cpps.append(c)
                        except Exception:
                            pass

                    if seg_cpps:
                        # Core trimming: drop first and last frame to remove boundary transition
                        core_cpps = seg_cpps[1:-1] if len(seg_cpps) >= 4 else seg_cpps
                        in_nv = is_segment_in_nv(t_start, t_end, nv_intervals)

                        segments_out.append({
                            "file": file_stem,
                            "speaker": spk,
                            "gender": gender,
                            "emotion": emo,
                            "session": session,
                            "seg_idx": seg_counter,
                            "start_sec": round(t_start, 3),
                            "end_sec": round(t_end, 3),
                            "dur_ms": round(dur_ms, 1),
                            "frames": len(seg_cpps),
                            "is_nv": in_nv,
                            "f0_mean": round(float(np.mean(seg_f0)), 1),
                            "raw_mean": round(float(np.mean(seg_cpps)), 2),
                            "raw_min": round(float(np.min(seg_cpps)), 2),
                            "core_mean": round(float(np.mean(core_cpps)), 2),
                            "core_median": round(float(np.median(core_cpps)), 2),
                            "core_std": round(float(np.std(core_cpps)), 2),
                            "core_min": round(float(np.min(core_cpps)), 2),
                            "core_max": round(float(np.max(core_cpps)), 2),
                            "core_p10": round(float(np.percentile(core_cpps, 10)), 2),
                            "core_p90": round(float(np.percentile(core_cpps, 90)), 2),
                        })
                        seg_counter += 1

        # Check last segment if audio ended while voiced
        if in_seg and (len(voiced) - start_idx) >= 6:
            i = len(voiced)
            t_start = times[start_idx]
            t_end = times[i - 1]
            dur_ms = (t_end - t_start + 0.010) * 1000.0

            seg_cpps = []
            seg_f0 = []
            for k in range(start_idx, i):
                t = times[k]
                seg_f0.append(f0s[k])
                try:
                    sl = call(cep, "To PowerCepstrum (slice)", t)
                    c = float(
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
                    seg_cpps.append(c)
                except Exception:
                    pass

            if seg_cpps:
                core_cpps = seg_cpps[1:-1] if len(seg_cpps) >= 4 else seg_cpps
                in_nv = is_segment_in_nv(t_start, t_end, nv_intervals)
                segments_out.append({
                    "file": file_stem,
                    "speaker": spk,
                    "gender": gender,
                    "emotion": emo,
                    "session": session,
                    "seg_idx": seg_counter,
                    "start_sec": round(t_start, 3),
                    "end_sec": round(t_end, 3),
                    "dur_ms": round(dur_ms, 1),
                    "frames": len(seg_cpps),
                    "is_nv": in_nv,
                    "f0_mean": round(float(np.mean(seg_f0)), 1),
                    "raw_mean": round(float(np.mean(seg_cpps)), 2),
                    "raw_min": round(float(np.min(seg_cpps)), 2),
                    "core_mean": round(float(np.mean(core_cpps)), 2),
                    "core_median": round(float(np.median(core_cpps)), 2),
                    "core_std": round(float(np.std(core_cpps)), 2),
                    "core_min": round(float(np.min(core_cpps)), 2),
                    "core_max": round(float(np.max(core_cpps)), 2),
                    "core_p10": round(float(np.percentile(core_cpps, 10)), 2),
                    "core_p90": round(float(np.percentile(core_cpps, 90)), 2),
                })
    except Exception as e:
        pass

    return segments_out


def collect_jvnv_tasks(
    base_dir: Path,
    samples_per_cell: Optional[int] = 15,
    seed: int = 42,
) -> List[Tuple[str, str, str, str, str, str]]:
    """Collect balanced tasks across 4 speakers x 6 emotions."""
    tasks = []
    nv_dir = base_dir / "nv_label"
    rng = random.Random(seed)

    for spk in SPEAKERS:
        spk_dir = base_dir / spk
        if not spk_dir.exists():
            continue
        gender = GENDERS.get(spk, "U")

        for emo in EMOTIONS:
            emo_dir = spk_dir / emo
            if not emo_dir.exists():
                continue

            all_wavs_in_cell = []
            for session in ["regular", "free"]:
                sess_dir = emo_dir / session
                if not sess_dir.exists():
                    continue
                for w in sess_dir.glob("*.wav"):
                    all_wavs_in_cell.append((w, session))

            if not all_wavs_in_cell:
                continue

            if samples_per_cell is not None and samples_per_cell > 0:
                chosen = rng.sample(all_wavs_in_cell, min(len(all_wavs_in_cell), samples_per_cell))
            else:
                chosen = all_wavs_in_cell

            for wav_path, session in sorted(chosen, key=lambda x: str(x[0])):
                stem = wav_path.stem
                nv_file = nv_dir / spk / f"{stem}.txt"
                nv_str = str(nv_file) if nv_file.exists() else ""
                tasks.append((str(wav_path), spk, gender, emo, session, nv_str))

    return tasks


def main():
    parser = argparse.ArgumentParser(
        description="Scheme 1: Continuous Voiced Segment CPPS Analysis on JVNV Dataset."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(r"E:\Corpus\Breathy\jvnv_v1"),
        help="Path to JVNV dataset root directory.",
    )
    parser.add_argument(
        "--samples-per-cell",
        type=int,
        default=15,
        help="Samples per (speaker x emotion) cell (default: 15 -> 4 spk x 6 emo x 15 = 360 files). Set to 0 for all files.",
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
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(r"E:\Coding\WavFry\results\jvnv_voiced_segments_analysis.csv"),
        help="Path to save detailed per-segment measurements.",
    )

    args = parser.parse_args()

    base_dir = args.dataset_dir
    if not base_dir.exists():
        sys.exit(f"Error: dataset-dir does not exist: {base_dir}")

    samples_per_cell = args.samples_per_cell if args.samples_per_cell > 0 else None

    print("=" * 70)
    print("Scheme 1: Continuous Voiced Segment CPPS Analysis on JVNV Dataset")
    print(f"Dataset Dir:       {base_dir}")
    print(f"Samples per Cell:  {samples_per_cell if samples_per_cell else 'ALL (1615 files)'}")
    print(f"Workers:           {args.workers}")
    print("=" * 70)

    # 1. Collect Tasks
    tasks = collect_jvnv_tasks(base_dir, samples_per_cell=samples_per_cell, seed=args.seed)
    print(f"Sampled {len(tasks)} audio files across 4 speakers x 6 emotions.")

    # 2. Parallel Extraction
    t0 = time.time()
    all_segments = []
    print(f"\nExtracting voiced segments on {len(tasks)} files using {args.workers} processes...")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for seg_list in tqdm(executor.map(extract_voiced_segments_for_file, tasks, chunksize=16), total=len(tasks), mininterval=5.0, ncols=80):
            all_segments.extend(seg_list)

    elapsed = time.time() - t0
    print(f"\nExtraction completed in {elapsed:.1f}s ({len(tasks)/max(0.001, elapsed):.1f} files/s).")
    print(f"Total Voiced Segments (Vowel Nuclei) Extracted: {len(all_segments)}")

    # 3. Analyze Data
    df = pd.DataFrame(all_segments)
    if df.empty:
        sys.exit("Error: No voiced segments were extracted.")

    # Save detailed CSV
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"Detailed per-segment measurements saved to: {args.output_csv}")

    # 4. Voiced Segment CPPS Distribution Across EMOTIONS
    print("\n" + "=" * 70)
    print("1. Voiced Segment Core CPPS Distribution Across EMOTIONS")
    print("=" * 70)

    emo_stats = []
    for emo in EMOTIONS:
        sub = df[df["emotion"] == emo]
        c_mean = sub["core_mean"]
        c_min = sub["core_min"]

        pct_min_ge_15 = (c_min >= 15.0).mean() * 100.0
        pct_mean_ge_15 = (c_mean >= 15.0).mean() * 100.0
        pct_lt_14 = (c_mean < 14.0).mean() * 100.0
        pct_lt_12 = (c_mean < 12.0).mean() * 100.0

        emo_stats.append({
            "Emotion": emo,
            "Seg Count": len(sub),
            "Core Mean": f"{c_mean.mean():.2f} dB",
            "Median (P50)": f"{c_mean.median():.2f} dB",
            "Std": f"{c_mean.std():.2f}",
            "P10": f"{c_mean.quantile(0.10):.2f} dB",
            "P90": f"{c_mean.quantile(0.90):.2f} dB",
            "CoreMin >= 15dB": f"{pct_min_ge_15:.1f}%",
            "CoreMean >= 15dB": f"{pct_mean_ge_15:.1f}%",
            "CoreMean < 14dB": f"{pct_lt_14:.1f}%",
            "CoreMean < 12dB": f"{pct_lt_12:.1f}%",
        })

    print(pd.DataFrame(emo_stats).to_string(index=False))

    # 5. Nonverbal (NV) Segments vs Verbal Segments (Fine-Grained Contrast)
    print("\n" + "=" * 70)
    print("2. Nonverbal (NV) vs Verbal Speech: Voiced Segment Contrast")
    print("=" * 70)

    nv_sub = df[df["is_nv"] == True]
    vb_sub = df[df["is_nv"] == False]

    comp_rows = []
    for name, sub in [("NV Segments (抽泣/叹气/笑声)", nv_sub), ("Verbal Segments (正文朗读)", vb_sub)]:
        c_mean = sub["core_mean"]
        c_min = sub["core_min"]
        comp_rows.append({
            "Segment Type": name,
            "Count": len(sub),
            "Core Mean": f"{c_mean.mean():.2f} dB",
            "Median": f"{c_mean.median():.2f} dB",
            "Std": f"{c_mean.std():.2f}",
            "P10": f"{c_mean.quantile(0.10):.2f} dB",
            "P90": f"{c_mean.quantile(0.90):.2f} dB",
            "CoreMin >= 15dB": f"{(c_min >= 15.0).mean() * 100.0:.1f}%",
            "CoreMean < 14dB": f"{(c_mean < 14.0).mean() * 100.0:.1f}%",
            "CoreMean < 12dB": f"{(c_mean < 12.0).mean() * 100.0:.1f}%",
        })
    print(pd.DataFrame(comp_rows).to_string(index=False))

    # 6. NV Segment CPPS Breakdown by Emotion
    print("\n" + "=" * 70)
    print("3. NV Segments Breakdown by Emotion (Nonverbal Vocalizations Only)")
    print("=" * 70)
    nv_emo_rows = []
    for emo in EMOTIONS:
        sub = nv_sub[nv_sub["emotion"] == emo]
        if sub.empty:
            continue
        c_mean = sub["core_mean"]
        c_min = sub["core_min"]
        nv_emo_rows.append({
            "Emotion": emo,
            "NV Segs": len(sub),
            "Core Mean": f"{c_mean.mean():.2f} dB",
            "Median": f"{c_mean.median():.2f} dB",
            "P10": f"{c_mean.quantile(0.10):.2f} dB",
            "CoreMin >= 15dB": f"{(c_min >= 15.0).mean() * 100.0:.1f}%",
            "CoreMean < 14dB": f"{(c_mean < 14.0).mean() * 100.0:.1f}%",
            "CoreMean < 12dB": f"{(c_mean < 12.0).mean() * 100.0:.1f}%",
        })
    print(pd.DataFrame(nv_emo_rows).to_string(index=False))

    # 7. Speaker & Gender Breakdown
    print("\n" + "=" * 70)
    print("4. Speaker & Gender Breakdown (Voiced Segments)")
    print("=" * 70)
    spk_rows = []
    for spk in SPEAKERS:
        sub = df[df["speaker"] == spk]
        c_mean = sub["core_mean"]
        c_min = sub["core_min"]
        spk_rows.append({
            "Speaker": spk,
            "Gender": GENDERS[spk],
            "Segments": len(sub),
            "Core Mean": f"{c_mean.mean():.2f} dB",
            "Median": f"{c_mean.median():.2f} dB",
            "Std": f"{c_mean.std():.2f}",
            "P10": f"{c_mean.quantile(0.10):.2f} dB",
            "P90": f"{c_mean.quantile(0.90):.2f} dB",
            "CoreMin >= 15dB": f"{(c_min >= 15.0).mean() * 100.0:.1f}%",
        })
    print(pd.DataFrame(spk_rows).to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Syllable Nuclei & Vowel Boundary Extraction using Praat Acoustic Analysis (Parselmouth).
Based on the classic de Jong & Wempe (2009) syllable nucleus detection algorithm,
adapted to output Praat .TextGrid files with vowel intervals for 04_creaky_tools GUI.
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np
import parselmouth
from textgrid import TextGrid, IntervalTier, PointTier


def detect_syllable_nuclei_and_boundaries(
    wav_path: str | Path,
    silence_threshold_db: float = -25.0,  # relative to maximum intensity
    min_dip_db: float = 2.5,              # dip between peaks to distinguish syllables
    min_pitch: float = 65.0,              # Hz, floor for pitch tracking
    max_pitch: float = 500.0,             # Hz, ceiling for pitch tracking
    time_step: float = 0.01,              # 10ms frame shift
    min_vowel_dur: float = 0.04,          # 40ms minimum vowel duration
    vowel_drop_db: float = 6.0,           # dB drop from peak to determine vowel boundary (6.0 dB for natural vowel span)
) -> tuple[list[tuple[float, float, str]], list[tuple[float, str]], float]:
    """
    Detect syllable nuclei and estimate vowel boundaries using intensity & pitch contours.
    
    Returns:
        vowel_intervals: list of (start_s, end_s, label)
        nuclei_points: list of (time_s, label)
        total_duration: float
    """
    sound = parselmouth.Sound(str(wav_path))
    duration = sound.get_total_duration()

    # 1. Compute Intensity Contour
    intensity = sound.to_intensity(minimum_pitch=min_pitch, time_step=time_step)
    int_times = intensity.xs()
    int_values = intensity.values[0, :]  # shape: (n_frames,)

    # Replace -inf or invalid values with minimum valid dB
    valid_mask = np.isfinite(int_values) & (int_values > 0)
    if not np.any(valid_mask):
        return [], [], duration
    
    min_db = np.min(int_values[valid_mask])
    int_values = np.where(valid_mask, int_values, min_db)

    max_intensity = np.max(int_values)
    abs_silence_threshold = max_intensity + silence_threshold_db

    # 2. Compute Pitch (Voicing) Contour
    pitch = sound.to_pitch_ac(
        time_step=time_step,
        pitch_floor=min_pitch,
        pitch_ceiling=max_pitch
    )

    # 3. Find candidate intensity peaks (above silence threshold)
    candidate_peaks = []
    for i in range(1, len(int_values) - 1):
        # Local peak condition
        if int_values[i] > int_values[i - 1] and int_values[i] >= int_values[i + 1]:
            if int_values[i] >= abs_silence_threshold:
                t = int_times[i]
                # Check voicing at this peak
                f0 = pitch.get_value_at_time(t)
                if f0 is not None and not np.isnan(f0) and f0 > 0:
                    candidate_peaks.append((i, t, int_values[i]))

    if not candidate_peaks:
        return [], [], duration

    # 4. Filter peaks by dip threshold (merge peaks that do not have sufficient dip)
    filtered_peaks = []
    for p_idx, (idx, t, val) in enumerate(candidate_peaks):
        if not filtered_peaks:
            filtered_peaks.append((idx, t, val))
            continue
        
        prev_idx, prev_t, prev_val = filtered_peaks[-1]
        # Check the minimum intensity between prev_idx and current idx
        valley_val = np.min(int_values[prev_idx : idx + 1])
        dip_from_prev = prev_val - valley_val
        dip_from_curr = val - valley_val

        # If both sides show sufficient dip, accept current as a new peak
        if dip_from_prev >= min_dip_db and dip_from_curr >= min_dip_db:
            filtered_peaks.append((idx, t, val))
        else:
            # If dip is not enough, keep the higher peak
            if val > prev_val:
                filtered_peaks[-1] = (idx, t, val)

    # 5. Determine Vowel Boundaries for each filtered peak
    vowel_intervals = []
    nuclei_points = []

    n_peaks = len(filtered_peaks)
    for p_i in range(n_peaks):
        idx, t_peak, val_peak = filtered_peaks[p_i]
        nuclei_points.append((t_peak, f"N_{p_i + 1}"))

        # Target threshold for boundary: drop by vowel_drop_db from peak
        target_db = val_peak - vowel_drop_db

        # Left boundary search
        left_limit = 0 if p_i == 0 else filtered_peaks[p_i - 1][0]
        # Find valley index between prev peak and current peak
        if p_i > 0:
            valley_left_idx = left_limit + np.argmin(int_values[left_limit : idx + 1])
        else:
            valley_left_idx = 0

        curr = idx
        while curr > valley_left_idx:
            t_curr = int_times[curr]
            f0_curr = pitch.get_value_at_time(t_curr)
            is_unvoiced = (f0_curr is None or np.isnan(f0_curr) or f0_curr <= 0)
            is_below_target = (int_values[curr] <= target_db)
            if is_unvoiced or is_below_target:
                break
            curr -= 1
        t_start = int_times[curr]

        # Right boundary search
        right_limit = len(int_values) - 1 if p_i == n_peaks - 1 else filtered_peaks[p_i + 1][0]
        if p_i < n_peaks - 1:
            valley_right_idx = idx + np.argmin(int_values[idx : right_limit + 1])
        else:
            valley_right_idx = len(int_values) - 1

        curr = idx
        while curr < valley_right_idx:
            t_curr = int_times[curr]
            f0_curr = pitch.get_value_at_time(t_curr)
            is_unvoiced = (f0_curr is None or np.isnan(f0_curr) or f0_curr <= 0)
            is_below_target = (int_values[curr] <= target_db)
            if is_unvoiced or is_below_target:
                break
            curr += 1
        t_end = int_times[curr]

        # Validate minimum duration
        if t_end - t_start >= min_vowel_dur:
            vowel_intervals.append((t_start, t_end, f"V_{p_i + 1}"))

    return vowel_intervals, nuclei_points, duration


def export_textgrid(
    vowel_intervals: list[tuple[float, float, str]],
    nuclei_points: list[tuple[float, str]],
    duration: float,
    output_path: str | Path,
):
    """
    Export TextGrid with 'phones' tier (vowel intervals with empty non-vowel gaps)
    and 'nuclei' point tier for 04_creaky_tools GUI inspection.
    """
    tg = TextGrid(maxTime=duration)
    
    # 1. Phones tier (IntervalTier)
    tier_phones = IntervalTier(name="phones", maxTime=duration)
    tier_words = IntervalTier(name="words", maxTime=duration)
    
    last_end = 0.0
    for s, e, mark in vowel_intervals:
        s_clamped = max(0.0, s)
        e_clamped = min(duration, e)
        if s_clamped > last_end + 0.001:
            tier_phones.add(minTime=last_end, maxTime=s_clamped, mark="")
            tier_words.add(minTime=last_end, maxTime=s_clamped, mark="")
        
        if e_clamped > s_clamped:
            tier_phones.add(minTime=s_clamped, maxTime=e_clamped, mark=mark)
            tier_words.add(minTime=s_clamped, maxTime=e_clamped, mark=mark)
            last_end = e_clamped

    if last_end < duration:
        tier_phones.add(minTime=last_end, maxTime=duration, mark="")
        tier_words.add(minTime=last_end, maxTime=duration, mark="")

    tg.append(tier_phones)
    tg.append(tier_words)

    # 2. Nuclei tier (PointTier)
    tier_nuclei = PointTier(name="nuclei", maxTime=duration)
    for t, mark in nuclei_points:
        if 0.0 <= t <= duration:
            tier_nuclei.add(time=t, mark=mark)
    tg.append(tier_nuclei)

    # Write TextGrid
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tg.write(str(output_path))
    print(f"  -> Generated: {output_path} ({len(vowel_intervals)} vowels detected)")


def main():
    parser = argparse.ArgumentParser(description="Praat Syllable Nuclei & Vowel Extractor")
    parser.add_argument(
        "--wav_dir",
        type=str,
        default=r"C:\Users\mumu\Downloads\wu-test",
        help="Input directory containing .wav files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"C:\Users\mumu\Downloads\wu-test\praat_syllable_nuclei",
        help="Output directory for .TextGrid files",
    )
    parser.add_argument(
        "--silence_threshold_db",
        type=float,
        default=-25.0,
        help="Silence threshold in dB relative to maximum intensity",
    )
    parser.add_argument(
        "--min_dip_db",
        type=float,
        default=2.5,
        help="Minimum intensity dip between adjacent peaks (dB)",
    )
    parser.add_argument(
        "--vowel_drop_db",
        type=float,
        default=6.0,
        help="Intensity drop (dB) from peak to define vowel boundary",
    )
    args = parser.parse_args()

    wav_dir = Path(args.wav_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(list(wav_dir.glob("*.wav")))
    print(f"Found {len(wav_files)} WAV files in {wav_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    for wav_path in wav_files:
        print(f"\nProcessing: {wav_path.name}")
        vowels, nuclei, dur = detect_syllable_nuclei_and_boundaries(
            wav_path=wav_path,
            silence_threshold_db=args.silence_threshold_db,
            min_dip_db=args.min_dip_db,
            vowel_drop_db=args.vowel_drop_db,
        )

        tg_path = output_dir / f"{wav_path.stem}.TextGrid"
        export_textgrid(
            vowel_intervals=vowels,
            nuclei_points=nuclei,
            duration=dur,
            output_path=tg_path,
        )

    print("\n" + "=" * 60)
    print("Done! All TextGrids exported successfully.")


if __name__ == "__main__":
    main()

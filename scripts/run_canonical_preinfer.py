#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical full-dataset GPU pre-inference runner for ita_rion_processed.

Uses 04_creaky_tools's native PhoneInferenceEngine, WavFryPredictor, and WavFryInferCache
to generate exact, canonical wavfry_infer_cache.json files under _creaky_cache and _breathy_cache.

Features:
- Very low log frequency: only outputs clean milestones every 500 files (~10 lines total).
- Atomic disk flushing every 500 files and on completion.
- Resumable: automatically skips all files already present in the cache.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add 04_creaky_tools to path
CREAKY_TOOLS_ROOT = Path(r"E:\Corpus\CS-Dialogue\data\short_wav\WAVE\04_creaky_tools")
sys.path.insert(0, str(CREAKY_TOOLS_ROOT))
sys.path.insert(0, str(CREAKY_TOOLS_ROOT / "src"))

from src.phone_inventory import (
    IntervalType,
    TypedInterval,
    classify_phone_label,
    get_phone_inventory,
    make_interval_id,
)
from src.creaky_vowel_analysis_core import (
    choose_tier_name,
    extract_target_intervals,
    get_wavfry_breathy_cache,
    get_wavfry_creaky_cache,
    get_wavfry_predictor,
    parse_textgrid_long,
)
from src.phone_inference_engine import PhoneInferenceEngine


def main():
    audio_dir = Path(r"E:\Corpus\Breathy\ita_rion_processed\audio")
    tg_dir = Path(r"E:\Corpus\Breathy\ita_rion_processed\textgrid_flat")
    model_path = Path(
        r"E:\Corpus\CS-Dialogue\data\short_wav\WAVE\06_wavfry\exp_results\checkpoints\frozen_creaky_breathy_block_silence_neg\best_ep10_prauc0.9516_f10.8431.pt"
    )

    batch_size = 64
    os.environ["WAVFRY_SUB_BATCH_SIZE"] = str(batch_size)

    print("=" * 70)
    print("Canonical Dataset Pre-Inference (WavFry Multi-Task on GPU)")
    print(f"Audio Dir:   {audio_dir}")
    print(f"TextGrid:    {tg_dir}")
    print(f"Model Path:  {model_path.name}")
    print("=" * 70)

    # 1. Initialize predictor and dual caches
    predictor = get_wavfry_predictor(model_path=model_path)
    if predictor is None:
        sys.exit(f"Error: Could not load WavFryPredictor from {model_path}")

    creaky_cache = get_wavfry_creaky_cache(textgrid_dir=str(tg_dir), model_path=model_path)
    breathy_cache = get_wavfry_breathy_cache(textgrid_dir=str(tg_dir), model_path=model_path)

    engine = PhoneInferenceEngine(
        predictor=predictor,
        creaky_cache=creaky_cache,
        breathy_cache=breathy_cache,
    )

    inventory = get_phone_inventory("ja")

    # 2. Gather audio & TextGrid files
    audio_files = {p.stem: p for p in audio_dir.glob("*.flac")}
    tg_files = {p.stem: p for p in tg_dir.glob("*.TextGrid")}
    common_keys = sorted(set(audio_files.keys()) & set(tg_files.keys()))

    # Check cache status
    already_cached = 0
    to_process = []
    for k in common_keys:
        c_hit = creaky_cache is not None and creaky_cache.is_file_cached(k)
        b_hit = breathy_cache is not None and breathy_cache.is_file_cached(k)
        if c_hit and b_hit:
            already_cached += 1
        else:
            to_process.append(k)

    total_target = len(to_process)
    print(f"Status: {already_cached} files already cached ({already_cached/len(common_keys)*100:.1f}%).")
    print(f"Remaining to infer: {total_target} files.\n")

    if not to_process:
        print("All files are already in cache. Nothing to do!")
        return

    # 3. Stream through remaining files with very low log frequency (every 500 files)
    t0 = time.time()
    num_processed = 0
    total_intervals_inferred = 0
    LOG_INTERVAL = 500

    for file_key in to_process:
        wav_path = audio_files[file_key]
        tg_path = tg_files[file_key]

        try:
            tiers = parse_textgrid_long(tg_path)
            tier_name = choose_tier_name(tiers, "phones")
            raw_intervals = tiers.get(tier_name, [])

            typed_intervals = []
            for iv in raw_intervals:
                ptype = classify_phone_label(iv.label, inventory)
                typed_intervals.append(
                    TypedInterval(
                        start=float(iv.start),
                        end=float(iv.end),
                        raw_label=str(iv.label),
                        interval_id=make_interval_id(iv.start, iv.end, iv.label),
                        phone_type=ptype,
                    )
                )

            target_intervals = extract_target_intervals(typed_intervals, mode="all_speech")

            waveform, sr = sf.read(str(wav_path), dtype="float32")
            if len(waveform.shape) > 1 and waveform.shape[1] > 1:
                waveform = np.mean(waveform, axis=1)

            if sr != 16000:
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000, res_type="soxr_hq")
                sr = 16000

            with contextlib.redirect_stdout(io.StringIO()):
                res = engine.infer_file(
                    file_key=file_key,
                    waveform=waveform,
                    sr=sr,
                    target_intervals=target_intervals,
                    wavfry_cache_flush=False,
                    compute_frame_probs=False,
                )

            total_intervals_inferred += len(target_intervals)
            num_processed += 1

            # Log milestone and flush every LOG_INTERVAL files
            if num_processed % LOG_INTERVAL == 0:
                if creaky_cache:
                    creaky_cache.flush_to_disk()
                if breathy_cache:
                    breathy_cache.flush_to_disk()

                elapsed_so_far = time.time() - t0
                speed = num_processed / max(0.001, elapsed_so_far)
                rem_files = total_target - num_processed
                eta_s = rem_files / max(0.001, speed)
                eta_min = int(eta_s // 60)
                eta_sec = int(eta_s % 60)
                pct = num_processed / total_target * 100.0
                total_done = already_cached + num_processed
                print(
                    f"[{time.strftime('%H:%M:%S')}] Progress: {total_done}/{len(common_keys)} "
                    f"({total_done/len(common_keys)*100:.1f}%) | "
                    f"Speed: {speed:.1f} files/s | ETA: {eta_min}m {eta_sec:02d}s | Flushed to disk"
                )

        except Exception as e:
            print(f"[Warning] Error on {file_key}: {e}")

    # Final flush to disk
    if creaky_cache:
        creaky_cache.flush_to_disk()
    if breathy_cache:
        breathy_cache.flush_to_disk()

    elapsed = time.time() - t0
    rate = num_processed / max(0.001, elapsed)
    print("\n" + "=" * 70)
    print(f"GPU Pre-Inference Fully Completed!")
    print(f"Total files in cache: {already_cached + num_processed} / {len(common_keys)}")
    print(f"New inferred:         {num_processed} files in {elapsed:.1f}s ({rate:.1f} files/s)")
    print(f"Creaky Cache:         {creaky_cache._cache_path}")
    print(f"Breathy Cache:        {breathy_cache._cache_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

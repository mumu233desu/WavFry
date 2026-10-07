#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rule-based pseudo-negative (non-breathy) annotation script for ITA-Corpus-Rion
combining Carlos T. Ishi (2010, 2011) NBP, Praat CPPS dual thresholds, and WavFry creaky probabilities.

Criteria for modal non-breathy negative sample:
1. min_CPPS >= 15.0 dB (no frame-level unvoiced, glitch, or dropout in core region)
2. mean_CPPS >= 18.0 dB (robust continuous harmonic periodicity across core)
3. max_NBP <= -35.0 dB (no mid-frequency 1.5~2.5kHz aspiration/turbulent noise)
4. creaky_prob <= 0.30 (no glottal fry / creakiness from WavFry infer cache)

Target label: strictly "non_breathy"
Storage format: sharded_json_v2 (synced with App 04_creaky_tools and WavFry manifests)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import parselmouth
from parselmouth.praat import call
from tqdm import tqdm

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add 04_creaky_tools/src to sys.path for breathiness_ishii
TOOLS_SRC_DIR = Path(r"E:\Corpus\CS-Dialogue\data\short_wav\WAVE\04_creaky_tools\src")
if TOOLS_SRC_DIR.exists() and str(TOOLS_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_SRC_DIR))

try:
    from breathiness_ishii import compute_ishii_nbp
except ImportError:
    # Fallback to local import if available
    from src.breathiness_ishii import compute_ishii_nbp

ANNOTATION_FILENAME_BREATHY = "manual_breathy_annotations.json"
ANNOTATION_FILENAME_CREAKY = "manual_creaky_annotations.json"
ANNOTATION_STORAGE_MODE = "sharded_json_v2"
ANNOTATION_SHARD_DIR_SUFFIX = ".d"

JAPANESE_VOWELS = {
    "a", "aː", "e", "eː", "i", "iː", "o", "oː", "ɯ", "ɯː", "ɨ", "ɨː",
}


def parse_vowels_from_textgrid(tg_path: Path) -> List[Tuple[float, float, str]]:
    """Extract vowel intervals from the phones tier of a Praat TextGrid."""
    vowels = []
    text = None
    for enc in ["utf-8", "utf-16", "cp932", "latin1"]:
        try:
            text = tg_path.read_text(encoding=enc)
            break
        except Exception:
            pass
    if text is None:
        return vowels

    items = text.split("item [")
    for it in items[1:]:
        if 'name = "phones"' not in it:
            continue
        int_blocks = re.findall(
            r'xmin = ([\d\.]+)\s+xmax = ([\d\.]+)\s+text = "(.*?)"',
            it,
            re.DOTALL,
        )
        for x1, x2, label in int_blocks:
            clean_lbl = label.strip()
            base_v = clean_lbl.replace("ɨ", "ɯ")
            if base_v in JAPANESE_VOWELS:
                vowels.append((float(x1), float(x2), clean_lbl))
        break
    return vowels


def process_single_file(args_tuple: Tuple) -> Dict[str, Any]:
    """Worker function to process one audio file with Ishi NBP + CPPS + Creaky rule."""
    (
        wav_path_str,
        tg_path_str,
        file_key,
        min_cpps_thresh,
        mean_cpps_thresh,
        max_nbp_thresh,
        creaky_thresh,
        edge_margin_s,
        min_dur_s,
        creaky_file_preds,
    ) = args_tuple

    wav_path = Path(wav_path_str)
    tg_path = Path(tg_path_str)

    if not wav_path.exists() or not tg_path.exists():
        return {"file_key": file_key, "annotations": {}, "stats": []}

    try:
        vowels = parse_vowels_from_textgrid(tg_path)
        if not vowels:
            return {"file_key": file_key, "annotations": {}, "stats": []}

        # 1. Compute Continuous Ishi NBP (time_step = 10ms)
        times_nbp, nbp_vals, _ = compute_ishii_nbp(str(wav_path), time_step=0.01)

        # 2. Compute PowerCepstrogram on the entire continuous sentence
        sound = parselmouth.Sound(str(wav_path))
        cep_whole = call(sound, "To PowerCepstrogram", 60.0, 0.005, 5000.0, 50.0)

        file_annotations = {}
        interval_stats = []

        for t1, t2, phone in vowels:
            dur = t2 - t1
            if dur < min_dur_s:
                continue

            # Core region: exclude edge margin to avoid transition overlap
            if dur >= (edge_margin_s * 2 + 0.010):
                c_start = t1 + edge_margin_s
                c_end = t2 - edge_margin_s
            else:
                margin = dur * 0.20
                c_start = t1 + margin
                c_end = t2 - margin

            # Sample CPPS frames every 10ms
            t_samples = np.arange(c_start, c_end, 0.010)
            if len(t_samples) == 0:
                t_samples = np.array([(t1 + t2) / 2.0])

            cpps_vals = []
            for t in t_samples:
                try:
                    slice_cep = call(cep_whole, "To PowerCepstrum (slice)", float(t))
                    fc = call(
                        slice_cep,
                        "Get peak prominence",
                        60.0,
                        600.0,
                        "Parabolic",
                        0.001,
                        0.0,
                        "Straight",
                        "Robust",
                    )
                    cpps_vals.append(fc)
                except Exception:
                    pass

            if not cpps_vals:
                continue

            # Sample Ishi NBP frames in core region
            mask_nbp = (times_nbp >= c_start) & (times_nbp <= c_end)
            v_nbp = nbp_vals[mask_nbp]
            valid_nbp = v_nbp[~np.isnan(v_nbp)]

            if len(valid_nbp) == 0:
                continue

            min_cpps = float(np.min(cpps_vals))
            mean_cpps = float(np.mean(cpps_vals))
            max_nbp = float(np.max(valid_nbp))
            mean_nbp = float(np.mean(valid_nbp))

            interval_id = f"{t1:.4f}|{t2:.4f}|{phone}"

            creak_p = None
            if creaky_file_preds:
                creak_p = creaky_file_preds.get(interval_id)

            # Combined Rule Check:
            # 1. min_cpps >= 15.0 dB
            # 2. mean_cpps >= 18.0 dB
            # 3. max_nbp <= -35.0 dB
            # 4. creaky_prob <= 0.30
            cond_cpps_min = (min_cpps >= min_cpps_thresh)
            cond_cpps_mean = (mean_cpps >= mean_cpps_thresh)
            cond_nbp = (max_nbp <= max_nbp_thresh)
            cond_creak = (creak_p is None or creak_p <= creaky_thresh)

            is_non_breathy = cond_cpps_min and cond_cpps_mean and cond_nbp and cond_creak

            stat_entry = {
                "phone": phone,
                "dur": dur,
                "min_cpps": min_cpps,
                "mean_cpps": mean_cpps,
                "max_nbp": max_nbp,
                "mean_nbp": mean_nbp,
                "creaky_prob": creak_p,
                "is_non_breathy": is_non_breathy,
            }
            interval_stats.append(stat_entry)

            if is_non_breathy:
                file_annotations[interval_id] = "non_breathy"

        return {
            "file_key": file_key,
            "annotations": file_annotations,
            "stats": interval_stats,
        }
    except Exception as e:
        return {"file_key": file_key, "annotations": {}, "stats": [], "error": str(e)}


def save_sharded_annotations(
    output_dir: Path,
    file_annotations_map: Dict[str, Dict[str, str]],
    speaker_map: Dict[str, str],
    now_unix: float,
) -> None:
    """Save annotations in sharded JSON format."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for manifest_name in (ANNOTATION_FILENAME_BREATHY, ANNOTATION_FILENAME_CREAKY):
        manifest_path = output_dir / manifest_name
        shard_dir = output_dir / (manifest_name + ANNOTATION_SHARD_DIR_SUFFIX)
        shard_dir.mkdir(parents=True, exist_ok=True)

        manifest_payload = {
            "version": 1,
            "schema": "direct_breathy_annotation",
            "storage_mode": ANNOTATION_STORAGE_MODE,
            "updated_unix_time": now_unix,
        }
        manifest_path.write_text(
            json.dumps(manifest_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        for file_key, annotations in file_annotations_map.items():
            spk_id = speaker_map.get(file_key, "UNK")
            file_entry = {
                "version": 1,
                "schema": "direct_breathy_annotation",
                "file_key": file_key,
                "speaker_id": spk_id,
                "annotations": annotations,
                "reviewed_at": now_unix,
            }
            digest = hashlib.sha256(str(file_key).encode("utf-8")).hexdigest()
            shard_path = shard_dir / f"{digest}.json"
            shard_path.write_text(
                json.dumps(file_entry, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )


def main():
    parser = argparse.ArgumentParser(
        description="Acoustic rule-based non-breathy pseudo-labeling using Ishi NBP + CPPS + Creaky."
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path(r"E:\Corpus\Breathy\ita_rion_processed\audio"),
        help="Directory containing audio FLAC/WAV files.",
    )
    parser.add_argument(
        "--textgrid-dir",
        type=Path,
        default=Path(r"E:\Corpus\Breathy\ita_rion_processed\textgrid_flat"),
        help="Directory containing aligned TextGrid files.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Target _breathy_cache directory. Defaults to <textgrid-dir>/_breathy_cache.",
    )
    parser.add_argument(
        "--creaky-cache-file",
        type=Path,
        default=None,
        help="Path to wavfry_infer_cache.json for creaky probabilities.",
    )
    parser.add_argument(
        "--min-cpps-thresh",
        type=float,
        default=15.0,
        help="CPPS minimum core frame threshold in dB (default: 15.0 dB).",
    )
    parser.add_argument(
        "--mean-cpps-thresh",
        type=float,
        default=18.0,
        help="CPPS mean core frame threshold in dB (default: 18.0 dB).",
    )
    parser.add_argument(
        "--max-nbp-thresh",
        type=float,
        default=-35.0,
        help="Ishi NBP maximum core frame threshold in dB (default: -35.0 dB).",
    )
    parser.add_argument(
        "--creaky-thresh",
        type=float,
        default=0.30,
        help="WavFry creaky probability ceiling (default: 0.30).",
    )
    parser.add_argument(
        "--edge-margin-ms",
        type=float,
        default=10.0,
        help="Boundary margin in ms (default: 10.0 ms).",
    )
    parser.add_argument(
        "--min-dur-ms",
        type=float,
        default=40.0,
        help="Minimum vowel duration in ms (default: 40.0 ms).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(os.cpu_count() or 4, 8),
        help="Number of parallel processes.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit number of files for quick testing.",
    )

    args = parser.parse_args()

    audio_dir = args.audio_dir
    tg_dir = args.textgrid_dir
    cache_dir = args.cache_dir or (tg_dir / "_breathy_cache")
    creaky_cache_file = args.creaky_cache_file or (tg_dir / "_creaky_cache" / "wavfry_infer_cache.json")

    if not audio_dir.exists():
        sys.exit(f"Error: audio-dir does not exist: {audio_dir}")
    if not tg_dir.exists():
        sys.exit(f"Error: textgrid-dir does not exist: {tg_dir}")

    print("=" * 70)
    print("Ishi NBP + Dual-CPPS + Creaky Non-Breathy Pseudo-Annotation")
    print(f"Audio Dir:        {audio_dir}")
    print(f"TextGrid Dir:     {tg_dir}")
    print(f"Cache Dir:        {cache_dir}")
    print(f"Creaky Cache:     {creaky_cache_file}")
    print(f"Rule Parameters:")
    print(f"  • min_CPPS >=   {args.min_cpps_thresh:.1f} dB")
    print(f"  • mean_CPPS >=  {args.mean_cpps_thresh:.1f} dB")
    print(f"  • max_NBP <=    {args.max_nbp_thresh:.1f} dB (Ishi 2011 classic)")
    print(f"  • creaky_prob <= {args.creaky_thresh:.2f}")
    print(f"  • edge_margin:  {args.edge_margin_ms:.1f} ms")
    print(f"Target Label:     non_breathy")
    print(f"Workers:          {args.num_workers}")
    print("=" * 70)

    # 1. Discover audio and TextGrid pairs
    audio_map = {p.stem: p for p in audio_dir.glob("*.flac")}
    if not audio_map:
        audio_map = {p.stem: p for p in audio_dir.glob("*.wav")}
    tg_map = {p.stem: p for p in tg_dir.glob("*.TextGrid")}

    common_keys = sorted(set(audio_map.keys()) & set(tg_map.keys()))
    print(f"Discovered {len(common_keys)} matched audio-TextGrid pairs.")

    # 2. Check sampling_state.json batches
    sampling_state_path = cache_dir / "sampling_state.json"
    sampling_state = None
    batch_files_map = defaultdict(set)
    if sampling_state_path.exists():
        try:
            sampling_state = json.loads(sampling_state_path.read_text(encoding="utf-8"))
            for b in sampling_state.get("batches", []):
                b_id = b["batch_id"]
                for f_info in b.get("files", []):
                    batch_files_map[b_id].add(f_info["file_key"])
            print(f"Found existing sampling_state with {len(batch_files_map)} batches.")
        except Exception as e:
            print(f"[Warning] Failed to read sampling_state: {e}")

    # Build speaker map
    speaker_map = {}
    for k in common_keys:
        m = re.search(r"_(U\d{4})_", k)
        speaker_map[k] = m.group(1) if m else "spk"

    target_keys = common_keys
    if args.max_files:
        target_keys = target_keys[: args.max_files]
        print(f"Restricted to first {len(target_keys)} files for testing.")

    # 3. Load creaky predictions from cache
    creaky_predictions_map = {}
    if creaky_cache_file.exists():
        try:
            creaky_data = json.loads(creaky_cache_file.read_text(encoding="utf-8"))
            files_dict = creaky_data.get("files", {})
            for fk, fv in files_dict.items():
                if isinstance(fv, dict) and "predictions" in fv:
                    creaky_predictions_map[fk] = fv["predictions"]
            print(f"Loaded creaky probabilities for {len(creaky_predictions_map)} files from {creaky_cache_file.name}.")
        except Exception as e:
            print(f"[Warning] Failed to load creaky cache: {e}")
    else:
        print(f"[Notice] No creaky cache found at {creaky_cache_file}; skipping creaky check.")

    # 4. Prepare worker tasks
    tasks = []
    edge_margin_s = args.edge_margin_ms / 1000.0
    min_dur_s = args.min_dur_ms / 1000.0

    for k in target_keys:
        k_creak = creaky_predictions_map.get(k)
        tasks.append((
            str(audio_map[k]),
            str(tg_map[k]),
            k,
            args.min_cpps_thresh,
            args.mean_cpps_thresh,
            args.max_nbp_thresh,
            args.creaky_thresh,
            edge_margin_s,
            min_dur_s,
            k_creak,
        ))

    print(f"\nProcessing {len(tasks)} files across {args.num_workers} processes...")
    t0 = time.time()

    results = []
    if args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            for res in tqdm(executor.map(process_single_file, tasks, chunksize=16), total=len(tasks), mininterval=5.0, ncols=80):
                results.append(res)
    else:
        for t in tqdm(tasks, mininterval=5.0, ncols=80):
            results.append(process_single_file(t))

    elapsed = time.time() - t0
    print(f"\nAcoustic evaluation completed in {elapsed:.2f}s ({len(tasks)/max(0.001, elapsed):.1f} files/s).")

    # 5. Aggregate results and statistics
    all_file_annotations = {}
    total_evaluated_vowels = 0
    total_non_breathy_vowels = 0
    phone_counter = Counter()
    phone_passed = Counter()

    for r in results:
        f_key = r["file_key"]
        annos = r["annotations"]
        all_file_annotations[f_key] = annos

        for st in r["stats"]:
            total_evaluated_vowels += 1
            ph = st["phone"].replace("ː", "").replace("ɨ", "ɯ")
            phone_counter[ph] += 1
            if st["is_non_breathy"]:
                total_non_breathy_vowels += 1
                phone_passed[ph] += 1

    now_unix = time.time()

    # 6. Save per-batch annotations if batches exist
    if batch_files_map:
        print("\nUpdating per-batch annotation stores...")
        for b_id, b_keys in sorted(batch_files_map.items()):
            batch_annos = {k: all_file_annotations[k] for k in b_keys if k in all_file_annotations}
            batch_dir = cache_dir / "batches" / f"batch_{b_id}"
            save_sharded_annotations(
                output_dir=batch_dir,
                file_annotations_map=batch_annos,
                speaker_map=speaker_map,
                now_unix=now_unix,
            )
            print(f"  • Batch [{b_id}]: {len(batch_annos)} files saved to {batch_dir}")

        # Update annotated flag in sampling_state.json
        if sampling_state:
            for b in sampling_state.get("batches", []):
                for f_info in b.get("files", []):
                    if f_info["file_key"] in all_file_annotations:
                        f_info["annotated"] = True
            sampling_state_path.write_text(
                json.dumps(sampling_state, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print("  • Updated sampling_state.json (marked annotated=True).")

    # 7. Save root aggregated annotations
    print(f"\nSaving root aggregated annotations to {cache_dir}...")
    save_sharded_annotations(
        output_dir=cache_dir,
        file_annotations_map=all_file_annotations,
        speaker_map=speaker_map,
        now_unix=now_unix,
    )

    # 8. Mirror to audio_dir / _breathy_cache
    audio_cache_dir = audio_dir / "_breathy_cache"
    if audio_cache_dir != cache_dir:
        print(f"Mirroring cache to audio directory: {audio_cache_dir}...")
        save_sharded_annotations(
            output_dir=audio_cache_dir,
            file_annotations_map=all_file_annotations,
            speaker_map=speaker_map,
            now_unix=now_unix,
        )
        if batch_files_map:
            for b_id, b_keys in sorted(batch_files_map.items()):
                batch_annos = {k: all_file_annotations[k] for k in b_keys if k in all_file_annotations}
                b_dir = audio_cache_dir / "batches" / f"batch_{b_id}"
                save_sharded_annotations(
                    output_dir=b_dir,
                    file_annotations_map=batch_annos,
                    speaker_map=speaker_map,
                    now_unix=now_unix,
                )
            if sampling_state:
                (audio_cache_dir / "sampling_state.json").write_text(
                    json.dumps(sampling_state, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

    # 9. Print final audit summary
    print("\n" + "=" * 70)
    print("Final Re-Annotation Summary Report")
    print(f"Total audio files:           {len(all_file_annotations)}")
    print(f"Total evaluated vowels:      {total_evaluated_vowels}")
    pct = (total_non_breathy_vowels / max(1, total_evaluated_vowels)) * 100.0
    print(f"Annotated non_breathy:       {total_non_breathy_vowels} ({pct:.1f}%)")
    print(f"Filtered out (not modal):    {total_evaluated_vowels - total_non_breathy_vowels} ({100.0 - pct:.1f}%)")
    print("-" * 70)
    print("Vowel breakdown:")
    for ph, total in sorted(phone_counter.items()):
        pass_cnt = phone_passed[ph]
        v_pct = pass_cnt / max(1, total) * 100.0
        print(f"  /{ph}/: {pass_cnt:5d} / {total:5d} ({v_pct:5.1f}%) | Filtered: {total - pass_cnt:5d} ({100.0 - v_pct:5.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rule-based pseudo-negative (non-breathy) annotation script for ITA-Corpus-Rion.

Applies acoustic prior rules (whole-sentence continuous CPPS >= 15dB + creaky prob < 0.4)
to automatically annotate modal/non-breathy vowels in compliance with:
1. Annotation APP (04_creaky_tools) per-batch and aggregated storage format
2. WavFry training manifest adapters (adapter_breathy_cache and adapter_cs_dialogue_textgrid)

Notice:
Target label is strictly "non_breathy", NOT "non_creaky"!
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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

ANNOTATION_FILENAME_BREATHY = "manual_breathy_annotations.json"
ANNOTATION_FILENAME_CREAKY = "manual_creaky_annotations.json"  # for 04_creaky_tools backward compatibility
ANNOTATION_STORAGE_MODE = "sharded_json_v2"
ANNOTATION_SHARD_DIR_SUFFIX = ".d"

# Approved Japanese vowel phonemes from MFA
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

    # Parse items in TextGrid
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
            # Normalize vowel label (remove tones/markers if any)
            base_v = clean_lbl.replace("ɨ", "ɯ")
            if base_v in JAPANESE_VOWELS:
                vowels.append((float(x1), float(x2), clean_lbl))
        break
    return vowels


import re


def process_single_file(
    args_tuple: Tuple[str, str, str, float, float, float, float, Optional[Dict[str, float]]]
) -> Dict[str, Any]:
    """Worker function to process one audio file and its TextGrid."""
    wav_path_str, tg_path_str, file_key, cpps_min_thresh, edge_margin_s, min_dur_s, creaky_thresh, creaky_file_preds = args_tuple
    
    wav_path = Path(wav_path_str)
    tg_path = Path(tg_path_str)

    if not wav_path.exists() or not tg_path.exists():
        return {"file_key": file_key, "annotations": {}, "stats": []}

    try:
        vowels = parse_vowels_from_textgrid(tg_path)
        if not vowels:
            return {"file_key": file_key, "annotations": {}, "stats": []}

        sound = parselmouth.Sound(str(wav_path))
        # 1. Compute PowerCepstrogram on the ENTIRE continuous sentence
        # Window length: 3.2 / pitch_floor (60Hz -> ~53ms)
        cep_whole = call(sound, "To PowerCepstrogram", 60.0, 0.005, 5000.0, 50.0)

        file_annotations = {}
        interval_stats = []

        for t1, t2, phone in vowels:
            dur = t2 - t1
            if dur < min_dur_s:
                continue

            # Core region: exclude edge margin to avoid window overlap with adjacent unvoiced phones
            if dur >= (edge_margin_s * 2 + 0.010):
                c_start = t1 + edge_margin_s
                c_end = t2 - edge_margin_s
            else:
                margin = dur * 0.20
                c_start = t1 + margin
                c_end = t2 - margin

            # Sample frames every 10ms
            t_samples = np.arange(c_start, c_end, 0.010)
            if len(t_samples) == 0:
                t_samples = np.array([(t1 + t2) / 2.0])

            frame_vals = []
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
                    frame_vals.append(fc)
                except Exception:
                    pass

            if not frame_vals:
                continue

            min_cpps = min(frame_vals)
            mean_cpps = float(np.mean(frame_vals))
            interval_id = f"{t1:.4f}|{t2:.4f}|{phone}"

            creak_p = None
            if creaky_file_preds:
                creak_p = creaky_file_preds.get(interval_id)

            # Rule check:
            # 1. All core frames >= cpps_min_thresh (e.g. 15.0 dB)
            # 2. Creaky prob < creaky_thresh (0.40)
            cond_cpps = (min_cpps >= cpps_min_thresh)
            cond_creak = (creak_p is None or creak_p < creaky_thresh)
            is_non_breathy = cond_cpps and cond_creak

            stat_entry = {
                "phone": phone,
                "dur": dur,
                "mean_cpps": mean_cpps,
                "min_cpps": min_cpps,
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
    """Save annotations in sharded JSON format compatible with both APP and manifest adapter."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # We write both manual_breathy_annotations.json and manual_creaky_annotations.json
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
        description="Acoustic rule-based non-breathy pseudo-labeling for ITA-Corpus-Rion."
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
        "--cpps-thresh",
        type=float,
        default=15.0,
        help="CPPS minimum core frame threshold in dB (default: 15.0 dB).",
    )
    parser.add_argument(
        "--edge-margin-ms",
        type=float,
        default=10.0,
        help="Boundary margin in ms to exclude window overlap (default: 10.0 ms).",
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
        "--batch-only",
        action="store_true",
        help="Only process files listed in sampling_state.json batches (if present).",
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

    if not audio_dir.exists():
        sys.exit(f"Error: audio-dir does not exist: {audio_dir}")
    if not tg_dir.exists():
        sys.exit(f"Error: textgrid-dir does not exist: {tg_dir}")

    print("=" * 70)
    print("Rule-Based Non-Breathy (Modal) Pseudo-Annotation")
    print(f"Audio Dir:        {audio_dir}")
    print(f"TextGrid Dir:     {tg_dir}")
    print(f"Cache Dir:        {cache_dir}")
    print(f"CPPS Core Thresh: {args.cpps_thresh:.1f} dB")
    print(f"Edge Margin:      {args.edge_margin_ms:.1f} ms")
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

    # Check sampling_state.json
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

    # Build speaker map (from filename or speaker_mapping.tsv)
    speaker_map = {}
    for k in common_keys:
        # e.g. F_00_JP_JA_JA-JP_U0001_S0_1 -> U0001
        m = re.search(r"_(U\d{4})_", k)
        speaker_map[k] = m.group(1) if m else "spk"

    target_keys = common_keys
    if args.batch_only and batch_files_map:
        all_batch_keys = set().union(*batch_files_map.values())
        target_keys = [k for k in target_keys if k in all_batch_keys]
        print(f"Filtered to {len(target_keys)} files defined in batches.")

    if args.max_files:
        target_keys = target_keys[: args.max_files]
        print(f"Restricted to first {len(target_keys)} files for testing.")

    # 2. Load creaky inference cache if available
    creaky_cache_file = (tg_dir / "_creaky_cache" / "wavfry_infer_cache.json")
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
        print("Notice: No _creaky_cache/wavfry_infer_cache.json found; skipping creaky threshold check.")

    # 3. Prepare task arguments for multiprocessing
    tasks = []
    edge_margin_s = args.edge_margin_ms / 1000.0
    min_dur_s = args.min_dur_ms / 1000.0
    creaky_thresh = 0.40

    for k in target_keys:
        k_creak = creaky_predictions_map.get(k)
        tasks.append((
            str(audio_map[k]),
            str(tg_map[k]),
            k,
            args.cpps_thresh,
            edge_margin_s,
            min_dur_s,
            creaky_thresh,
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
    print(f"Acoustic evaluation completed in {elapsed:.2f}s ({len(tasks)/max(0.001, elapsed):.1f} files/s).")

    # 3. Aggregate results and statistics
    all_file_annotations = {}
    total_evaluated_vowels = 0
    total_non_breathy_vowels = 0
    phone_counter = Counter()
    phone_passed = Counter()
    gender_counter = Counter()
    gender_passed = Counter()

    for r in results:
        f_key = r["file_key"]
        annos = r["annotations"]
        all_file_annotations[f_key] = annos

        gender = "F" if f_key.startswith("F_") else ("M" if f_key.startswith("M_") else "Other")

        for st in r["stats"]:
            total_evaluated_vowels += 1
            ph = st["phone"].replace("ː", "").replace("ɨ", "ɯ")
            phone_counter[ph] += 1
            gender_counter[gender] += 1
            if st["is_non_breathy"]:
                total_non_breathy_vowels += 1
                phone_passed[ph] += 1
                gender_passed[gender] += 1

    now_unix = time.time()

    # 4. Save per-batch annotations if batches exist
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

    # 5. Save root aggregated annotations
    print(f"\nSaving root aggregated annotations to {cache_dir}...")
    save_sharded_annotations(
        output_dir=cache_dir,
        file_annotations_map=all_file_annotations,
        speaker_map=speaker_map,
        now_unix=now_unix,
    )

    # 6. Also mirror to audio_dir / _breathy_cache for 100% interoperability
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

    # 7. Print final audit summary
    print("\n" + "=" * 70)
    print("Annotation Summary Report")
    print(f"Total audio files:           {len(all_file_annotations)}")
    print(f"Total evaluated vowels:      {total_evaluated_vowels}")
    pct = (total_non_breathy_vowels / max(1, total_evaluated_vowels)) * 100.0
    print(f"Annotated non_breathy:       {total_non_breathy_vowels} ({pct:.1f}%)")
    print("-" * 70)
    print("Vowel breakdown:")
    for ph, total in sorted(phone_counter.items()):
        pass_cnt = phone_passed[ph]
        v_pct = pass_cnt / max(1, total) * 100.0
        print(f"  /{ph}/: {pass_cnt:5d} / {total:5d} ({v_pct:.1f}%)")
    print("-" * 70)
    print("Gender breakdown:")
    for g, total in sorted(gender_counter.items()):
        pass_cnt = gender_passed[g]
        g_pct = pass_cnt / max(1, total) * 100.0
        print(f"  {g}: {pass_cnt:5d} / {total:5d} ({g_pct:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Acoustic rule-based modal non-breathy pseudo-annotation for BreathyTTS.

Applies 4-AND acoustic criteria on whole-audio continuous features:
1. min_CPPS >= 17.0 dB (no frame-level unvoiced, glitch, or dropout in core region)
2. mean_CPPS >= 21.0 dB (robust continuous harmonic periodicity across core)
3. max_Ishi_NBP <= -28.0 dB (no mid-frequency 1.5~2.5kHz aspiration/turbulent noise)
4. creaky_prob <= 0.20 (no glottal fry / creakiness from WavFry infer cache)

Engineering Guardrails:
- Whole-audio continuous feature extraction (Praat CPPS 2D-smoothing + Ishi 2011 continuous NBP)
- Slices precisely to TextGrid vowels with core boundary margin
- Strictly protects Batch 1 (100% human-annotated) and any reviewed files
- Writes sharded JSON format compliant with 04_creaky_tools and WavFry manifests
- Mirrors output between textgrid/_breathy_cache and _breathy_cache
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import parselmouth
from tqdm import tqdm

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add 04_creaky_tools/src to sys.path
TOOLS_SRC_DIR = Path(r"E:\Corpus\CS-Dialogue\data\short_wav\WAVE\04_creaky_tools\src")
if TOOLS_SRC_DIR.exists() and str(TOOLS_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_SRC_DIR))

try:
    from acoustic_features import compute_cpps
    from breathiness_ishii import compute_ishii_nbp
except ImportError:
    from src.acoustic_features import compute_cpps
    from src.breathiness_ishii import compute_ishii_nbp

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("annotate_breathy_tts_modal")

ANNOTATION_FILENAME_BREATHY = "manual_breathy_annotations.json"
ANNOTATION_SHARD_DIR_SUFFIX = ".d"

JAPANESE_VOWELS: Set[str] = {
    "a", "i", "u", "e", "o",
    "a:", "i:", "u:", "e:", "o:",
    "um",
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
            r'xmin = ([\d\.]+)\s+xmax = ([\d\.]+)\s+text = "([^"]*)"',
            it,
            re.DOTALL,
        )
        for x1, x2, label in int_blocks:
            clean_lbl = label.strip()
            if clean_lbl in JAPANESE_VOWELS:
                vowels.append((float(x1), float(x2), clean_lbl))
        break
    return vowels


def process_single_file(args_tuple: Tuple) -> Dict[str, Any]:
    """Worker function to process one audio file with whole-audio continuous CPPS + Ishi NBP."""
    (
        wav_path_str,
        tg_path_str,
        file_key,
        min_cpps_thresh,
        mean_cpps_thresh,
        max_nbp_thresh,
        creaky_thresh,
        edge_margin_pct,
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

        # 1. Whole-audio continuous CPPS (time_step = 10ms, smooth=True)
        times_cpps, cpps_arr = compute_cpps(wav_path, time_step=0.01, smooth=True)

        # 2. Whole-audio continuous Ishi NBP (time_step = 10ms)
        times_nbp, nbp_arr, _ = compute_ishii_nbp(wav_path, time_step=0.01)

        file_annotations = {}
        interval_stats = []

        for t1, t2, phone in vowels:
            dur = t2 - t1
            if dur < min_dur_s:
                continue

            # Core region: exclude edge margin to avoid transition overlap
            margin = dur * edge_margin_pct
            c_start = t1 + margin
            c_end = t2 - margin

            # Slice CPPS frames in core region
            m_cpps = (times_cpps >= c_start) & (times_cpps <= c_end)
            c_vals = cpps_arr[m_cpps]
            c_valid = c_vals[~np.isnan(c_vals)]
            if len(c_valid) == 0:
                continue

            # Slice Ishi NBP frames in core region
            m_nbp = (times_nbp >= c_start) & (times_nbp <= c_end)
            n_vals = nbp_arr[m_nbp]
            n_valid = n_vals[~np.isnan(n_vals)]
            if len(n_valid) == 0:
                continue

            min_cpps = float(np.min(c_valid))
            mean_cpps = float(np.mean(c_valid))
            max_nbp = float(np.max(n_valid))
            mean_nbp = float(np.mean(n_valid))

            interval_id = f"{t1:.4f}|{t2:.4f}|{phone}"

            creak_p = None
            if creaky_file_preds:
                creak_p = creaky_file_preds.get(interval_id)

            # 4-AND Acoustic Rule:
            # 1. min_cpps >= 17.0 dB
            # 2. mean_cpps >= 21.0 dB
            # 3. max_nbp <= -28.0 dB
            # 4. creaky_prob <= 0.20
            cond_cpps_min = (min_cpps >= min_cpps_thresh)
            cond_cpps_mean = (mean_cpps >= mean_cpps_thresh)
            cond_nbp = (max_nbp <= max_nbp_thresh)
            cond_creak = (creak_p is not None and creak_p <= creaky_thresh)

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


def save_merged_shards(
    batch_dir: Path,
    file_annotations_map: Dict[str, Dict[str, str]],
    speaker_map: Dict[str, str],
    now_unix: float,
) -> int:
    """Save or non-destructively merge shard annotations for a batch."""
    shard_dir = batch_dir / f"{ANNOTATION_FILENAME_BREATHY}{ANNOTATION_SHARD_DIR_SUFFIX}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = batch_dir / ANNOTATION_FILENAME_BREATHY
    if not manifest_path.exists():
        manifest_payload = {
            "version": 1,
            "schema": "direct_breathy_annotation",
            "storage_mode": "sharded_json_v2",
            "updated_unix_time": now_unix,
        }
        manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    updated_count = 0
    for file_key, new_annos in file_annotations_map.items():
        if not new_annos:
            continue

        digest = hashlib.sha256(str(file_key).encode("utf-8")).hexdigest()
        shard_path = shard_dir / f"{digest}.json"

        existing_entry: Dict[str, Any] = {}
        if shard_path.exists():
            try:
                existing_entry = json.loads(shard_path.read_text(encoding="utf-8"))
            except Exception:
                existing_entry = {}

        spk_id = speaker_map.get(file_key, existing_entry.get("speaker_id", "UNK"))

        merged_annos = dict(existing_entry.get("annotations", {}))
        # Non-destructively merge: do not overwrite existing labels
        for k_int, label in new_annos.items():
            if k_int not in merged_annos:
                merged_annos[k_int] = label

        entry_to_write = {
            "version": 1,
            "schema": "direct_breathy_annotation",
            "file_key": file_key,
            "speaker_id": spk_id,
            "annotations": merged_annos,
            "reviewed_at": existing_entry.get("reviewed_at", now_unix),
        }

        with tempfile.NamedTemporaryFile("w", dir=shard_dir, delete=False, encoding="utf-8") as tf:
            json.dump(entry_to_write, tf, indent=2, ensure_ascii=False)
            tmp_name = tf.name
        os.replace(tmp_name, shard_path)
        updated_count += 1

    return updated_count


def main():
    parser = argparse.ArgumentParser(
        description="Acoustic rule-based modal non-breathy pseudo-annotation for BreathyTTS."
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path(r"E:\Corpus\Breathy\BreathyTTS\audio"),
        help="Directory containing audio FLAC files.",
    )
    parser.add_argument(
        "--textgrid-dir",
        type=Path,
        default=Path(r"E:\Corpus\Breathy\BreathyTTS\textgrid"),
        help="Directory containing aligned TextGrid files.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(r"E:\Corpus\Breathy\BreathyTTS\textgrid\_breathy_cache"),
        help="Primary _breathy_cache directory containing batches.",
    )
    parser.add_argument(
        "--creaky-cache-file",
        type=Path,
        default=Path(r"E:\Corpus\Breathy\BreathyTTS\textgrid\_creaky_cache\wavfry_infer_cache.json"),
        help="Path to wavfry_infer_cache.json for creaky probabilities.",
    )
    parser.add_argument(
        "--min-cpps-thresh",
        type=float,
        default=17.0,
        help="Minimum core frame CPPS threshold in dB (default: 17.0 dB).",
    )
    parser.add_argument(
        "--mean-cpps-thresh",
        type=float,
        default=21.0,
        help="Mean core frame CPPS threshold in dB (default: 21.0 dB).",
    )
    parser.add_argument(
        "--max-nbp-thresh",
        type=float,
        default=-28.0,
        help="Maximum core frame Ishi NBP threshold in dB (default: -28.0 dB).",
    )
    parser.add_argument(
        "--creaky-thresh",
        type=float,
        default=0.20,
        help="WavFry creaky probability ceiling (default: 0.20).",
    )
    parser.add_argument(
        "--edge-margin-pct",
        type=float,
        default=0.15,
        help="Boundary margin fraction on each side (default: 0.15 = 15%).",
    )
    parser.add_argument(
        "--min-dur-ms",
        type=float,
        default=40.0,
        help="Minimum vowel duration in ms (default: 40.0 ms).",
    )
    parser.add_argument(
        "--sections",
        type=str,
        default="modal",
        help="Comma-separated sections to annotate ('modal' or 'modal,breathy'). Default: 'modal'.",
    )
    parser.add_argument(
        "--skip-batches",
        type=str,
        default="1",
        help="Comma-separated batch IDs to strictly skip from annotation (default: '1' for Batch 1).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(os.cpu_count() or 4, 8),
        help="Number of parallel processes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform acoustic analysis and report statistics without writing shards.",
    )

    args = parser.parse_args()

    audio_dir: Path = args.audio_dir
    tg_dir: Path = args.textgrid_dir
    cache_dir: Path = args.cache_dir
    creaky_cache_file: Path = args.creaky_cache_file

    if not audio_dir.exists():
        sys.exit(f"Error: audio-dir does not exist: {audio_dir}")
    if not tg_dir.exists():
        sys.exit(f"Error: textgrid-dir does not exist: {tg_dir}")
    if not creaky_cache_file.exists():
        sys.exit(f"Error: creaky-cache-file does not exist: {creaky_cache_file}")

    target_sections = {s.strip().lower() for s in args.sections.split(",") if s.strip()}
    skip_batch_ids = {int(b.strip()) for b in args.skip_batches.split(",") if b.strip().isdigit()}

    print("=" * 70)
    print("BreathyTTS Modal Non-Breathy Acoustic Rule Pseudo-Annotation")
    print(f"Audio Dir:          {audio_dir}")
    print(f"TextGrid Dir:       {tg_dir}")
    print(f"Cache Dir:          {cache_dir}")
    print(f"Creaky Cache:       {creaky_cache_file.name}")
    print(f"Target Sections:    {target_sections}")
    print(f"Protected Batches:  {sorted(skip_batch_ids)} (Strictly skipped)")
    print(f"Rule Parameters:")
    print(f"  • mean_CPPS >=    {args.mean_cpps_thresh:.1f} dB")
    print(f"  • min_CPPS >=     {args.min_cpps_thresh:.1f} dB")
    print(f"  • max_Ishi_NBP <= {args.max_nbp_thresh:.1f} dB")
    print(f"  • creaky_prob <=  {args.creaky_thresh:.2f}")
    print(f"  • edge_margin:    {args.edge_margin_pct*100:.0f}%")
    print(f"  • min_dur:        {args.min_dur_ms:.1f} ms")
    print(f"Target Label:       non_breathy")
    print(f"Workers:            {args.num_workers}")
    print(f"Dry Run:            {args.dry_run}")
    print("=" * 70)

    # 1. Load sampling_state.json to understand batch assignments
    sampling_state_path = cache_dir / "sampling_state.json"
    if not sampling_state_path.exists():
        sys.exit(f"Error: sampling_state.json not found in {cache_dir}")

    sampling_state = json.loads(sampling_state_path.read_text(encoding="utf-8"))
    file_to_batch: Dict[str, int] = {}
    speaker_map: Dict[str, str] = {}
    for b in sampling_state.get("batches", []):
        b_id = b["batch_id"]
        for f_info in b.get("files", []):
            fk = f_info["file_key"]
            file_to_batch[fk] = b_id
            speaker_map[fk] = f_info.get("speaker_id", "spk")

    print(f"Loaded sampling state with {len(file_to_batch)} files across {len(sampling_state.get('batches', []))} batches.")

    # 1.1 Load manifest_slices.csv if available
    file_to_section: Dict[str, str] = {}
    manifest_slices_csv = audio_dir.parent / "manifest_slices.csv"
    if manifest_slices_csv.exists():
        import csv
        with open(manifest_slices_csv, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fk = row.get("file_key") or row.get("slice_id")
                sec = row.get("section")
                if fk and sec:
                    file_to_section[fk.strip()] = sec.strip().lower()
        print(f"Loaded section mapping for {len(file_to_section)} files from {manifest_slices_csv.name}.")

    # 2. Filter target files according to section and protected batches
    audio_map = {p.stem: p for p in audio_dir.glob("*.flac")}
    if not audio_map:
        audio_map = {p.stem: p for p in audio_dir.glob("*.wav")}
    tg_map = {p.stem: p for p in tg_dir.glob("*.TextGrid")}

    common_keys = sorted(set(audio_map.keys()) & set(tg_map.keys()))

    target_keys = []
    skipped_batch_files = 0
    skipped_section_files = 0

    for k in common_keys:
        b_id = file_to_batch.get(k)
        if b_id in skip_batch_ids:
            skipped_batch_files += 1
            continue

        # 2.1 Resolve section from manifest_slices.csv or sub-index parity
        sec = file_to_section.get(k)
        if not sec:
            m_idx = re.search(r"_S0_(\d+)$", k)
            if m_idx:
                sec = "modal" if int(m_idx.group(1)) % 2 == 1 else "breathy"
            else:
                sec = "modal"
        if sec not in target_sections:
            skipped_section_files += 1
            continue

        target_keys.append(k)

    print(f"Target files selected for acoustic evaluation: {len(target_keys)}")
    print(f"  • Protected batch files skipped: {skipped_batch_files}")
    print(f"  • Non-target section files skipped: {skipped_section_files}")

    if not target_keys:
        print("No target files to process. Exiting.")
        return

    # 3. Load creaky probabilities from cache
    print(f"\nLoading creaky probabilities from {creaky_cache_file}...")
    creaky_predictions_map = {}
    try:
        creaky_data = json.loads(creaky_cache_file.read_text(encoding="utf-8"))
        files_dict = creaky_data.get("files", {})
        for fk, fv in files_dict.items():
            if isinstance(fv, dict) and "predictions" in fv:
                creaky_predictions_map[fk] = fv["predictions"]
        print(f"Loaded creaky probabilities for {len(creaky_predictions_map)} files.")
    except Exception as e:
        sys.exit(f"Error loading creaky cache: {e}")

    # 4. Prepare worker tasks
    tasks = []
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
            args.edge_margin_pct,
            min_dur_s,
            k_creak,
        ))

    print(f"\nProcessing {len(tasks)} files across {args.num_workers} processes...")
    t0 = time.time()

    results = []
    if args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            for res in tqdm(executor.map(process_single_file, tasks, chunksize=16), total=len(tasks), mininterval=2.0, ncols=80):
                results.append(res)
    else:
        for t in tqdm(tasks, mininterval=2.0, ncols=80):
            results.append(process_single_file(t))

    elapsed = time.time() - t0
    print(f"\nAcoustic evaluation completed in {elapsed:.2f}s ({len(tasks)/max(0.001, elapsed):.1f} files/s).")

    # 5. Aggregate results and statistics
    all_file_annotations = {}
    total_evaluated_vowels = 0
    total_non_breathy_vowels = 0
    phone_counter = Counter()
    phone_passed = Counter()
    failed_reasons = Counter()

    for r in results:
        f_key = r["file_key"]
        annos = r["annotations"]
        all_file_annotations[f_key] = annos

        for st in r["stats"]:
            total_evaluated_vowels += 1
            ph = st["phone"]
            phone_counter[ph] += 1
            if st["is_non_breathy"]:
                total_non_breathy_vowels += 1
                phone_passed[ph] += 1
            else:
                # Track why it failed
                if st["mean_cpps"] < args.mean_cpps_thresh:
                    failed_reasons["mean_cpps_low"] += 1
                if st["min_cpps"] < args.min_cpps_thresh:
                    failed_reasons["min_cpps_low"] += 1
                if st["max_nbp"] > args.max_nbp_thresh:
                    failed_reasons["max_nbp_high"] += 1
                if st["creaky_prob"] is None or st["creaky_prob"] > args.creaky_thresh:
                    failed_reasons["creaky_high"] += 1

    # 6. Save per-batch shards
    now_unix = time.time()
    files_with_mined_vowels = sum(1 for annos in all_file_annotations.values() if annos)

    if not args.dry_run:
        print("\nUpdating per-batch shard stores...")
        batch_files_map = defaultdict(dict)
        for fk, annos in all_file_annotations.items():
            b_id = file_to_batch[fk]
            batch_files_map[b_id][fk] = annos

        total_shards_written = 0
        for b_id, b_annos in sorted(batch_files_map.items()):
            b_dir = cache_dir / "batches" / f"batch_{b_id}"
            n_w = save_merged_shards(b_dir, b_annos, speaker_map, now_unix)
            total_shards_written += n_w

        print(f"Successfully wrote {total_shards_written} shards across Batches 2~20 to {cache_dir}.")

    # 7. Final Summary Audit Report
    print("\n" + "=" * 70)
    print("BreathyTTS Modal Acoustic Annotation Summary")
    print(f"Total target files evaluated:      {len(target_keys)}")
    print(f"Files yielding non_breathy labels: {files_with_mined_vowels} ({files_with_mined_vowels/len(target_keys)*100:.1f}%)")
    print(f"Total evaluated vowels:            {total_evaluated_vowels}")
    pct = (total_non_breathy_vowels / max(1, total_evaluated_vowels)) * 100.0
    print(f"Annotated non_breathy vowels:      {total_non_breathy_vowels} ({pct:.1f}%)")
    print(f"Filtered out (not modal enough):   {total_evaluated_vowels - total_non_breathy_vowels} ({100.0 - pct:.1f}%)")
    print("-" * 70)
    print("Vowel breakdown:")
    for ph, total in sorted(phone_counter.items()):
        pass_cnt = phone_passed[ph]
        v_pct = pass_cnt / max(1, total) * 100.0
        print(f"  /{ph:3s}/: {pass_cnt:5d} / {total:5d} ({v_pct:5.1f}%) | Filtered: {total - pass_cnt:5d} ({100.0 - v_pct:5.1f}%)")
    print("-" * 70)
    print("Filter rejection analysis (multiple reasons possible per vowel):")
    for reason, count in failed_reasons.most_common():
        print(f"  • {reason:18s}: {count:5d} ({count/max(1, total_evaluated_vowels)*100:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TextGrid labeler and ground-truth annotation projector for BreathyTTS.

Extracts vowels and sonorants from sliced breathy TextGrids, applies direct
'breathy = 1.0' labels, and exports standard YSJ-compatible shard JSON files.
Modal slices are deliberately excluded from shards (0% false-negative protection).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("textgrid_labeler")

JAPANESE_VOWELS: Set[str] = {
    "a", "i", "u", "e", "o",
    "a:", "i:", "u:", "e:", "o:",
    "um",
}

JAPANESE_SONORANTS: Set[str] = {
    "m", "my", "n", "ny", "N",
    "r", "ry", "w", "y",
}


def parse_praat_textgrid_intervals(tg_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Lightweight pure-python parser for Praat TextGrid interval tiers."""
    with open(tg_path, "r", encoding="utf-8", errors="replace") as f:
        lines = [line.strip() for line in f]

    tiers: Dict[str, List[Dict[str, Any]]] = {}
    current_tier_name = None
    cur_xmin = 0.0
    cur_xmax = 0.0

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('name = "'):
            current_tier_name = line.split('"')[1]
            tiers[current_tier_name] = []
        elif line.startswith("intervals [") or line.startswith("item ["):
            pass
        elif line.startswith("xmin ="):
            cur_xmin = float(line.split("=")[1].strip())
        elif line.startswith("xmax ="):
            cur_xmax = float(line.split("=")[1].strip())
        elif line.startswith('text = "'):
            parts = line.split('text = "')
            cur_text = parts[1].rstrip('"') if len(parts) > 1 else ""
            if current_tier_name is not None and cur_text != "":
                tiers[current_tier_name].append({
                    "xmin": cur_xmin,
                    "xmax": cur_xmax,
                    "text": cur_text,
                    "duration": round(cur_xmax - cur_xmin, 4),
                })
        i += 1

    return tiers


def extract_breathy_slice_intervals(
    breathy_tg_path: Path,
    file_key: str,
    speaker_id: str,
    context_id: str = "",
    min_duration_sec: float = 0.03,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Extract vowels and sonorants from Part 2 (Breathy) remapped TextGrid.

    All extracted valid vowels and sonorants receive breathy = 1.0.

    Returns:
        (manifest_interval_records, shard_dict)
    """
    tiers = parse_praat_textgrid_intervals(breathy_tg_path)
    phones_tier = tiers.get("phones") or tiers.get("phone") or tiers.get("tokens") or []

    if not phones_tier:
        logger.warning(f"No phones tier found in {breathy_tg_path.name}")
        return [], {}

    manifest_records: List[Dict[str, Any]] = []
    shard_annotations: Dict[str, str] = {}

    for p in phones_tier:
        raw_phone = p["text"].strip().lower()
        norm_phone = re.sub(r"[0-9_#^]", "", raw_phone)

        # Check vowel or sonorant
        is_vowel = norm_phone in JAPANESE_VOWELS or norm_phone.startswith(("a", "i", "u", "e", "o"))
        is_sonorant = norm_phone in JAPANESE_SONORANTS

        if not (is_vowel or is_sonorant):
            continue

        start_s = round(p["xmin"], 4)
        end_s = round(p["xmax"], 4)
        duration_s = round(end_s - start_s, 4)

        if duration_s < min_duration_sec:
            continue

        # Format exact interval_id matching YSJ convention: "{start:.4f}|{end:.4f}|{phone}"
        interval_id = f"{start_s:.4f}|{end_s:.4f}|{norm_phone}"
        shard_annotations[interval_id] = "mllm_breathy"

        record = {
            "dataset_id": "breathy_tts",
            "file_key": file_key,
            "speaker_id": speaker_id,
            "segment_id": f"{file_key}_{start_s:.3f}_{end_s:.3f}",
            "start_s": start_s,
            "end_s": end_s,
            "duration_s": duration_s,
            "phone": norm_phone,
            "phone_type": "vowel" if is_vowel else "sonorant",
            "label_breathy": 1.0,
            "label_creaky": float("nan"),
            "context_id": context_id,
            "section": "breathy",
        }
        manifest_records.append(record)

    shard_data = {
        "version": 1,
        "schema": "direct_breathy_annotation",
        "file_key": file_key,
        "speaker_id": speaker_id,
        "annotations": shard_annotations,
        "reviewed_at": time.time(),
    }

    logger.info(f"Extracted {len(manifest_records)} breathy intervals for {file_key}.")
    return manifest_records, shard_data


def extract_modal_slice_intervals(
    modal_tg_path: Path,
    file_key: str,
    speaker_id: str,
    context_id: str = "",
    min_duration_sec: float = 0.03,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Extract vowels and sonorants from Modal slice TextGrid.

    All extracted valid vowels receive label_breathy = 0.0 and 'non_breathy'.

    Returns:
        (manifest_interval_records, shard_dict)
    """
    tiers = parse_praat_textgrid_intervals(modal_tg_path)
    phones_tier = tiers.get("phones") or tiers.get("phone") or tiers.get("tokens") or []

    if not phones_tier:
        logger.warning(f"No phones tier found in {modal_tg_path.name}")
        return [], {}

    manifest_records: List[Dict[str, Any]] = []
    shard_annotations: Dict[str, str] = {}

    for p in phones_tier:
        raw_phone = p["text"].strip().lower()
        norm_phone = re.sub(r"[0-9_#^]", "", raw_phone)

        # Check vowel or sonorant
        is_vowel = norm_phone in JAPANESE_VOWELS or norm_phone.startswith(("a", "i", "u", "e", "o"))
        is_sonorant = norm_phone in JAPANESE_SONORANTS

        if not (is_vowel or is_sonorant):
            continue

        start_s = round(p["xmin"], 4)
        end_s = round(p["xmax"], 4)
        duration_s = round(end_s - start_s, 4)

        if duration_s < min_duration_sec:
            continue

        # Format exact interval_id matching YSJ convention: "{start:.4f}|{end:.4f}|{phone}"
        interval_id = f"{start_s:.4f}|{end_s:.4f}|{norm_phone}"
        shard_annotations[interval_id] = "non_breathy"

        record = {
            "dataset_id": "modal_tts",
            "file_key": file_key,
            "speaker_id": speaker_id,
            "segment_id": f"{file_key}_{start_s:.3f}_{end_s:.3f}",
            "start_s": start_s,
            "end_s": end_s,
            "duration_s": duration_s,
            "phone": norm_phone,
            "phone_type": "vowel" if is_vowel else "sonorant",
            "label_breathy": 0.0,
            "label_creaky": float("nan"),
            "context_id": context_id,
            "section": "modal",
        }
        manifest_records.append(record)

    shard_data = {
        "version": 1,
        "schema": "direct_breathy_annotation",
        "file_key": file_key,
        "speaker_id": speaker_id,
        "annotations": shard_annotations,
        "reviewed_at": time.time(),
    }

    logger.info(f"Extracted {len(manifest_records)} modal intervals for {file_key}.")
    return manifest_records, shard_data


def write_breathy_shard(
    dataset_root: Path,
    file_key: str,
    shard_data: Dict[str, Any],
    batch_name: str = "batch_1",
) -> Path:
    """Atomically write shard JSON to _breathy_cache/batches/{batch_name}/manual_breathy_annotations.json.d/."""
    batch_dir = dataset_root / "textgrid" / "_breathy_cache" / "batches" / batch_name
    shard_dir = batch_dir / "manual_breathy_annotations.json.d"
    shard_dir.mkdir(parents=True, exist_ok=True)

    key_hash = hashlib.sha256(file_key.encode("utf-8")).hexdigest()
    shard_path = shard_dir / f"{key_hash}.json"

    # Atomic write via tempfile
    with tempfile.NamedTemporaryFile("w", dir=shard_dir, delete=False, encoding="utf-8") as tf:
        json.dump(shard_data, tf, ensure_ascii=False, indent=2)
        temp_name = tf.name
    os.replace(temp_name, shard_path)

    # Maintain index file
    index_file = batch_dir / "manual_breathy_annotations.json"
    if not index_file.exists():
        index_file.write_text(
            json.dumps({"version": 1, "batch": batch_name, "files": {}}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return shard_path


if __name__ == "__main__":
    # Test extraction on the remapped breathy TextGrid from temp_test
    test_tg = Path(r"exp_results/tts_smoke_test/temp_test/test_slice_breathy.TextGrid")
    if test_tg.exists():
        records, shard = extract_breathy_slice_intervals(
            breathy_tg_path=test_tg,
            file_key="F_00_JP_JA_JA-JP_U0001_S0_2",
            speaker_id="U0001",
            context_id="prominence",
        )
        print(f"Extracted {len(records)} records.")
        print("Sample records:", records[:3])
        print("Sample shard annotations:", list(shard["annotations"].items())[:5])
    else:
        print("Test file not found.")

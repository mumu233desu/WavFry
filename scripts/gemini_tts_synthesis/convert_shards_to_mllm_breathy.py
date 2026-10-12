#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""In-place conversion of BreathyTTS shard annotations from 'breathy' to 'mllm_breathy'.

Also updates manifest_slices.csv to set is_annotated=False for breathy slices,
reflecting that these are coarse candidate pre-fills awaiting human ground-truth verification.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("convert_shards_to_mllm_breathy")


def convert_cache_directory(cache_dir: Path) -> tuple[int, int]:
    """Convert all shards under cache_dir from 'breathy' to 'mllm_breathy'."""
    shards_converted = 0
    intervals_converted = 0

    if not cache_dir.exists():
        logger.warning(f"Cache dir {cache_dir} does not exist.")
        return 0, 0

    shard_files = list(cache_dir.glob("batches/*/manual_breathy_annotations.json.d/*.json"))
    logger.info(f"Scanning {len(shard_files)} shard files in {cache_dir}...")

    for sf in shard_files:
        try:
            data = json.loads(sf.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"Error reading {sf}: {e}")
            continue

        annos = data.get("annotations", {})
        changed = False
        new_annos = {}
        for interval_id, state in annos.items():
            if state == "breathy":
                new_annos[interval_id] = "mllm_breathy"
                intervals_converted += 1
                changed = True
            else:
                new_annos[interval_id] = state

        if changed:
            data["annotations"] = new_annos
            # Atomic rewrite via temp file in the same directory
            with tempfile.NamedTemporaryFile("w", dir=sf.parent, delete=False, encoding="utf-8") as tf:
                json.dump(data, tf, ensure_ascii=False, indent=2)
                tmp_name = tf.name
            os.replace(tmp_name, sf)
            shards_converted += 1

    return shards_converted, intervals_converted


def update_manifest_slices(dataset_root: Path) -> int:
    """Set is_annotated=False for all breathy sections in manifest_slices.csv."""
    slices_csv = dataset_root / "manifest_slices.csv"
    if not slices_csv.exists():
        logger.warning(f"manifest_slices.csv not found at {slices_csv}")
        return 0

    with open(slices_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    updated_count = 0
    for r in rows:
        if r.get("section") == "breathy" and r.get("is_annotated") in ("True", True, "true"):
            r["is_annotated"] = "False"
            updated_count += 1

    # Atomic rewrite
    with tempfile.NamedTemporaryFile("w", dir=dataset_root, delete=False, encoding="utf-8", newline="") as tf:
        writer = csv.DictWriter(tf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        tmp_name = tf.name
    os.replace(tmp_name, slices_csv)

    return updated_count


def run_conversion() -> None:
    dataset_root = Path(r"E:\Corpus\Breathy\BreathyTTS")
    tg_cache = dataset_root / "textgrid" / "_breathy_cache"
    root_cache = dataset_root / "_breathy_cache"

    logger.info("Starting in-place conversion of breathy shards to mllm_breathy...")

    # 1. Convert textgrid/_breathy_cache
    s_cnt, i_cnt = convert_cache_directory(tg_cache)
    logger.info(f"Converted textgrid cache: {s_cnt} shards, {i_cnt} intervals changed to 'mllm_breathy'.")

    # 2. Convert root _breathy_cache if exists
    if root_cache.exists():
        r_s_cnt, r_i_cnt = convert_cache_directory(root_cache)
        logger.info(f"Converted root cache: {r_s_cnt} shards, {r_i_cnt} intervals changed to 'mllm_breathy'.")

    # 3. Update manifest_slices.csv
    slices_updated = update_manifest_slices(dataset_root)
    logger.info(f"Updated manifest_slices.csv: {slices_updated} breathy slice rows reset to is_annotated=False.")

    print("\n" + "=" * 65)
    print("IN-PLACE CONVERSION COMPLETED")
    print("=" * 65)
    print(f"Shards updated to 'mllm_breathy': {s_cnt} / 540")
    print(f"Total intervals converted:        {i_cnt}")
    print(f"manifest_slices.csv updated rows: {slices_updated} (breathy is_annotated -> False)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_conversion()

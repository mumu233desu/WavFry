#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Distribute all 1,080 BreathyTTS slices into 20 balanced batches (12 Train, 4 Val, 4 Test).

Architecture Specification:
- Total Batches: 20
  * Batches 1 ~ 12:  Train (12 batches * 54 files = 648 files / 324 pairs, 18 speakers: 9F + 9M)
  * Batches 13 ~ 16: Val   (4 batches * 54 files  = 216 files / 108 pairs, 6 speakers:  3F + 3M)
  * Batches 17 ~ 20: Test  (4 batches * 54 files  = 216 files / 108 pairs, 6 speakers:  3F + 3M)
- Total Files Enrolled: 1,080 files (540 dialogues, 100% full dataset utilization)
- Files per Batch: exactly 54 files (27 dialogue pairs: 27 modal + 27 breathy)
- Speakers per Batch: exactly 6 speakers (3 Female + 3 Male, 50/50 gender balanced)
- Dialogues per Speaker per Batch: strictly 4 or 5 pairs (8 or 10 files), preventing review fatigue
- Zero Cross-Split Leakage: Train, Val, and Test speaker sets are strictly 100% disjoint
- Natural Paired Layout: Part 1 modal (_S0_1) is directly followed by Part 2 breathy (_S0_2)
- Shard Preservation: Breathy slices maintain mllm_breathy candidate labels
- Full Qt Annotation Tool & Downstream wavfry/data/manifest.py compatibility
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("distribute_breathy_batches_20")

# Speaker allocation (30 speakers: 15 Female, 15 Male)
F_TRAIN = ["U0001", "U0002", "U0003", "U0004", "U0005", "U0011", "U0012", "U0013", "U0014"]
M_TRAIN = ["U0006", "U0007", "U0008", "U0009", "U0010", "U0021", "U0022", "U0023", "U0024"]

F_VAL = ["U0015", "U0016", "U0017"]
M_VAL = ["U0025", "U0026", "U0027"]

F_TEST = ["U0018", "U0019", "U0020"]
M_TEST = ["U0028", "U0029", "U0030"]

CHUNK_DEFS = [(1, 5), (6, 10), (11, 14), (15, 18)]  # Sizes: [5, 5, 4, 4]


def natural_slice_sort_key(slice_item: Dict[str, str]) -> tuple:
    spk = slice_item.get("speaker_id", "")
    fk = slice_item.get("file_key", "")
    m = re.search(r"_S0_(\d+)$", fk)
    sub_idx = int(m.group(1)) if m else 0
    return (spk, sub_idx)


def get_speaker_dialogue_chunk(
    speaker_dialogues: Dict[int, Dict[str, Dict[str, str]]],
    chunk_idx: int,
) -> List[Dict[str, str]]:
    """Return slices for a specific dialogue chunk (1-indexed inclusive)."""
    start_d, end_d = CHUNK_DEFS[chunk_idx]
    slices = []
    for d in range(start_d, end_d + 1):
        pair = speaker_dialogues.get(d, {})
        if "modal" in pair:
            slices.append(pair["modal"])
        if "breathy" in pair:
            slices.append(pair["breathy"])
    return slices


def build_schedule_assignments() -> List[Dict[str, Any]]:
    """Generate exact speaker chunk assignments for all 20 batches."""
    batch_plans = []

    # 1. Train Batches (1 to 12)
    f_train_batches = [[] for _ in range(12)]
    m_train_batches = [[] for _ in range(12)]

    for i in range(9):
        # Female assignments
        f_train_batches[i % 6].append((F_TRAIN[i], 0))
        f_train_batches[(i + 3) % 6].append((F_TRAIN[i], 1))
        f_train_batches[6 + (i % 6)].append((F_TRAIN[i], 2))
        f_train_batches[6 + ((i + 3) % 6)].append((F_TRAIN[i], 3))

        # Male assignments (complementary sizes)
        m_train_batches[6 + (i % 6)].append((M_TRAIN[i], 0))
        m_train_batches[6 + ((i + 3) % 6)].append((M_TRAIN[i], 1))
        m_train_batches[i % 6].append((M_TRAIN[i], 2))
        m_train_batches[(i + 3) % 6].append((M_TRAIN[i], 3))

    for b_idx in range(12):
        batch_plans.append({
            "batch_id": b_idx + 1,
            "name": f"Batch {b_idx + 1}",
            "split": "train",
            "f_assignments": f_train_batches[b_idx],
            "m_assignments": m_train_batches[b_idx],
        })

    # 2. Val Batches (13 to 16)
    f_val_batches = [
        [(F_VAL[0], 0), (F_VAL[1], 0), (F_VAL[2], 2)],
        [(F_VAL[1], 1), (F_VAL[2], 1), (F_VAL[0], 2)],
        [(F_VAL[2], 0), (F_VAL[0], 3), (F_VAL[1], 2)],
        [(F_VAL[0], 1), (F_VAL[1], 3), (F_VAL[2], 3)],
    ]
    m_val_batches = [
        [(M_VAL[2], 0), (M_VAL[0], 3), (M_VAL[1], 2)],
        [(M_VAL[0], 1), (M_VAL[1], 3), (M_VAL[2], 3)],
        [(M_VAL[0], 0), (M_VAL[1], 0), (M_VAL[2], 2)],
        [(M_VAL[1], 1), (M_VAL[2], 1), (M_VAL[0], 2)],
    ]
    for b_idx in range(4):
        batch_plans.append({
            "batch_id": 13 + b_idx,
            "name": f"Batch {13 + b_idx}",
            "split": "val",
            "f_assignments": f_val_batches[b_idx],
            "m_assignments": m_val_batches[b_idx],
        })

    # 3. Test Batches (17 to 20)
    f_test_batches = [
        [(F_TEST[0], 0), (F_TEST[1], 0), (F_TEST[2], 2)],
        [(F_TEST[1], 1), (F_TEST[2], 1), (F_TEST[0], 2)],
        [(F_TEST[2], 0), (F_TEST[0], 3), (F_TEST[1], 2)],
        [(F_TEST[0], 1), (F_TEST[1], 3), (F_TEST[2], 3)],
    ]
    m_test_batches = [
        [(M_TEST[2], 0), (M_TEST[0], 3), (M_TEST[1], 2)],
        [(M_TEST[0], 1), (M_TEST[1], 3), (M_TEST[2], 3)],
        [(M_TEST[0], 0), (M_TEST[1], 0), (M_TEST[2], 2)],
        [(M_TEST[1], 1), (M_TEST[2], 1), (M_TEST[0], 2)],
    ]
    for b_idx in range(4):
        batch_plans.append({
            "batch_id": 17 + b_idx,
            "name": f"Batch {17 + b_idx}",
            "split": "test",
            "f_assignments": f_test_batches[b_idx],
            "m_assignments": m_test_batches[b_idx],
        })

    return batch_plans


def distribute_batches(
    dataset_root: Path = Path(r"E:\Corpus\Breathy\BreathyTTS"),
    target_cache_dir: Path = Path(r"E:\Corpus\Breathy\BreathyTTS\textgrid\_breathy_cache"),
) -> None:
    manifest_csv = dataset_root / "manifest_slices.csv"
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Missing manifest_slices.csv at {manifest_csv}")

    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_slices = list(reader)

    logger.info(f"Loaded total {len(all_slices)} slices from manifest_slices.csv.")
    modal_count = sum(1 for s in all_slices if s.get("section") == "modal")
    breathy_count = sum(1 for s in all_slices if s.get("section") == "breathy")
    logger.info(f"Modal slices: {modal_count}, Breathy slices: {breathy_count}")

    # Index dialogues by speaker and dialogue number (1..18)
    spk_dialogues: Dict[str, Dict[int, Dict[str, Dict[str, str]]]] = {}
    for s in all_slices:
        spk = s["speaker_id"]
        fk = s["file_key"]
        sec = s["section"]
        m = int(fk.split("_S0_")[-1])
        d_idx = (m + 1) // 2
        spk_dialogues.setdefault(spk, {}).setdefault(d_idx, {})[sec] = s

    # Collect existing breathy shard payloads across all batches or backups into memory
    source_shard_payloads: Dict[str, Dict[str, Any]] = {}
    batches_root = target_cache_dir / "batches"
    backup_root = target_cache_dir / "batches_pre_partition_backup"

    search_dirs = [d for d in [batches_root, backup_root] if d.exists()]
    for s_dir in search_dirs:
        for shard_file in s_dir.glob("*/manual_breathy_annotations.json.d/*.json"):
            try:
                data = json.loads(shard_file.read_text(encoding="utf-8"))
                fk = data.get("file_key")
                if fk and fk not in source_shard_payloads:
                    source_shard_payloads[fk] = data
            except Exception as e:
                logger.debug(f"Error reading shard {shard_file}: {e}")

    logger.info(f"Discovered {len(source_shard_payloads)} existing shard JSON payloads across cache batches.")

    # Safety backup
    if batches_root.exists() and not backup_root.exists():
        logger.info(f"Creating safety backup at {backup_root}...")
        shutil.copytree(batches_root, backup_root)

    # Clean existing batches directory to ensure fresh 20-batch layout
    if batches_root.exists():
        logger.info(f"Resetting {batches_root} for clean 20-batch layout...")
        shutil.rmtree(batches_root)
    batches_root.mkdir(parents=True, exist_ok=True)

    # Build schedules
    batch_plans = build_schedule_assignments()
    created_at = datetime.datetime.now().isoformat()
    state_batches = []

    for plan in batch_plans:
        b_id = plan["batch_id"]
        b_name = plan["name"]
        b_split = plan["split"]
        b_dir = batches_root / f"batch_{b_id}"
        shard_dir = b_dir / "manual_breathy_annotations.json.d"
        shard_dir.mkdir(parents=True, exist_ok=True)

        batch_slices: List[Dict[str, str]] = []
        for spk, chunk_idx in plan["f_assignments"]:
            batch_slices.extend(get_speaker_dialogue_chunk(spk_dialogues[spk], chunk_idx))
        for spk, chunk_idx in plan["m_assignments"]:
            batch_slices.extend(get_speaker_dialogue_chunk(spk_dialogues[spk], chunk_idx))

        batch_slices.sort(key=natural_slice_sort_key)
        assert len(batch_slices) == 54, f"Batch {b_id} slice count = {len(batch_slices)} != 54"

        batch_files_state = []
        for s in batch_slices:
            fk = s["file_key"]
            spk = s["speaker_id"]
            gender = s["gender"]
            section = s.get("section", "modal")
            key_hash = hashlib.sha256(fk.encode("utf-8")).hexdigest()
            target_shard = shard_dir / f"{key_hash}.json"

            # Breathy slices retain/ensure mllm_breathy state
            if section == "breathy":
                if fk in source_shard_payloads:
                    s_data = dict(source_shard_payloads[fk])
                    annos = dict(s_data.get("annotations", {}))
                    for k_int, st in list(annos.items()):
                        if st == "breathy":
                            annos[k_int] = "mllm_breathy"
                    s_data["annotations"] = annos
                    target_shard.write_text(json.dumps(s_data, indent=2, ensure_ascii=False), encoding="utf-8")
                else:
                    fallback_shard = {
                        "version": 1,
                        "schema": "direct_breathy_annotation",
                        "file_key": fk,
                        "speaker_id": spk,
                        "annotations": {},
                        "reviewed_at": datetime.datetime.now().timestamp(),
                    }
                    target_shard.write_text(json.dumps(fallback_shard, indent=2, ensure_ascii=False), encoding="utf-8")

            batch_files_state.append({
                "file_key": fk,
                "speaker_id": spk,
                "gender": gender,
                "region": "JP",
                "annotated": False,
            })

        # Write manifest file manual_breathy_annotations.json
        manifest_file = b_dir / "manual_breathy_annotations.json"
        manifest_payload = {
            "version": 1,
            "batch": f"batch_{b_id}",
            "files": {},
        }
        manifest_file.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8")

        state_batches.append({
            "batch_id": b_id,
            "name": b_name,
            "split": b_split,
            "exclude_train_speakers": (b_split != "train"),
            "exclude_val_speakers": (b_split != "val"),
            "exclude_test_speakers": (b_split != "test"),
            "created_at": created_at,
            "files": batch_files_state,
            "target_count": len(batch_files_state),
            "actual_count": len(batch_files_state),
        })

    # Save sampling_state.json
    sampling_state = {
        "version": 2,
        "task": "breathy",
        "batches": state_batches,
        "fully_annotated_speakers": [],
        "history": [],
    }
    state_file = target_cache_dir / "sampling_state.json"
    state_file.write_text(json.dumps(sampling_state, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Successfully saved sampling_state.json to {state_file}.")

    # Verification report
    print("\n" + "=" * 68)
    print("BREATHYTTS 20-BATCH PARTITION SUMMARY (12 Train, 4 Val, 4 Test)")
    print("=" * 68)
    total_files = sum(b["actual_count"] for b in state_batches)
    print(f"Total Batches Created: {len(state_batches)}")
    print(f"Total Files Enrolled:  {total_files} / 1080 (100% Complete Utilization)")
    print("-" * 68)

    train_cnt = sum(b["actual_count"] for b in state_batches if b["split"] == "train")
    val_cnt = sum(b["actual_count"] for b in state_batches if b["split"] == "val")
    test_cnt = sum(b["actual_count"] for b in state_batches if b["split"] == "test")
    print(f"Train Batches (1~12):  {train_cnt} files (324 pairs, 18 disjoint speakers: 9F+9M)")
    print(f"Val Batches (13~16):   {val_cnt} files (108 pairs, 6 disjoint speakers: 3F+3M)")
    print(f"Test Batches (17~20):  {test_cnt} files (108 pairs, 6 disjoint speakers: 3F+3M)")
    print("-" * 68)

    for b in state_batches:
        spks = sorted(list({f["speaker_id"] for f in b["files"]}))
        f_cnt = sum(1 for f in b["files"] if f["gender"] == "F")
        m_cnt = sum(1 for f in b["files"] if f["gender"] == "M")
        spk_dialogues_count = {}
        for spk in spks:
            spk_dialogues_count[spk] = sum(1 for f in b["files"] if f["speaker_id"] == spk) // 2
        d_summary = ", ".join(f"{s}:{c}p" for s, c in spk_dialogues_count.items())
        print(f"  - [{b['batch_id']:2d}] {b['name']:8s} [{b['split'].upper():5s}]: {len(b['files'])} files (F:{f_cnt}, M:{m_cnt}) | Pairs: {d_summary}")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    distribute_batches()

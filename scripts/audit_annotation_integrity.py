#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import hashlib
import json
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

cache_dir = Path(r"E:\Corpus\Breathy\ita_rion_processed\textgrid_flat\_breathy_cache")
tg_dir = Path(r"E:\Corpus\Breathy\ita_rion_processed\textgrid_flat")
audio_dir = Path(r"E:\Corpus\Breathy\ita_rion_processed\audio")

print("=" * 70)
print("AUDIT REPORT: Non-Breathy Annotation Integrity Check")
print("=" * 70)

# 1. File counts
audio_files = list(audio_dir.glob("*.flac"))
tg_files = list(tg_dir.glob("*.TextGrid"))
print(f"1. Audio FLAC files:    {len(audio_files)}")
print(f"   TextGrid files:      {len(tg_files)}")

root_manifest_b = cache_dir / "manual_breathy_annotations.json"
root_shard_dir_b = cache_dir / "manual_breathy_annotations.json.d"
root_manifest_c = cache_dir / "manual_creaky_annotations.json"
root_shard_dir_c = cache_dir / "manual_creaky_annotations.json.d"

shards_b = list(root_shard_dir_b.glob("*.json"))
shards_c = list(root_shard_dir_c.glob("*.json"))
print(f"   Root Breathy Shards: {len(shards_b)}")
print(f"   Root Creaky Shards:  {len(shards_c)}")

# 2. Batches
print("\n2. Batch Structure & Storage:")
batches_dir = cache_dir / "batches"
batch_counts = {}
for b in sorted(batches_dir.glob("batch_*")):
    b_shards_b = list((b / "manual_breathy_annotations.json.d").glob("*.json"))
    b_shards_c = list((b / "manual_creaky_annotations.json.d").glob("*.json"))
    batch_counts[b.name] = (len(b_shards_b), len(b_shards_c))
    print(f"   • {b.name}: breathy={len(b_shards_b)}, creaky_compat={len(b_shards_c)}")

# Check sampling_state.json
sampling_state_path = cache_dir / "sampling_state.json"
if sampling_state_path.exists():
    state = json.loads(sampling_state_path.read_text(encoding="utf-8"))
    print(f"\n3. Sampling State (sampling_state.json):")
    print(f"   Task: {state.get('task')}")
    print(f"   Batches in state: {len(state.get('batches', []))}")
    for b in state.get("batches", []):
        all_annotated = all(f.get("annotated") for f in b.get("files", []))
        print(f"   • Batch [{b['batch_id']}] '{b['name']}': split={b['split']}, files={len(b['files'])}, all_annotated={all_annotated}")

# 3. Shard JSON validation
print("\n4. Shard Content & Label Validation:")
bad_shards = 0
unexpected_labels = set()
total_annotations = 0
empty_shards = 0
non_empty_shards = 0
sample_entries = []

for sf in shards_b:
    try:
        data = json.loads(sf.read_text(encoding="utf-8"))
        fkey = data.get("file_key")
        if not fkey:
            bad_shards += 1
            continue
        # verify digest
        digest = hashlib.sha256(str(fkey).encode("utf-8")).hexdigest()
        if sf.name != f"{digest}.json":
            print(f"   [Error] Hash mismatch for {fkey}: {sf.name} vs {digest}")
            bad_shards += 1

        annos = data.get("annotations", {})
        if not annos:
            empty_shards += 1
        else:
            non_empty_shards += 1

        total_annotations += len(annos)
        for iid, label in annos.items():
            if label != "non_breathy":
                unexpected_labels.add(label)
            parts = iid.split("|")
            if len(parts) != 3:
                unexpected_labels.add(f"bad_iid_{iid}")
            try:
                s, e = float(parts[0]), float(parts[1])
                if e <= s:
                    unexpected_labels.add(f"negative_dur_{iid}")
            except Exception:
                unexpected_labels.add(f"nan_time_{iid}")

        if len(sample_entries) < 3 and annos:
            sample_entries.append((fkey, list(annos.items())[:2]))
    except Exception as e:
        bad_shards += 1

print(f"   Total shards checked:      {len(shards_b)}")
print(f"   Corrupted/bad shards:      {bad_shards}")
print(f"   Shards with annotations:   {non_empty_shards}")
print(f"   Shards without modal hits: {empty_shards}")
print(f"   Total valid intervals:     {total_annotations}")
print(f"   Unexpected labels/errors:  {unexpected_labels or 'NONE (100% pure non_breathy)'}")

print("\n5. Sample Annotations:")
for k, items in sample_entries:
    print(f"   • {k}: {items}")

# 4. Check GPU infer caches
print("\n6. GPU Inference Caches (wavfry_infer_cache.json):")
for tname in ["_creaky_cache", "_breathy_cache"]:
    c_path = cache_dir.parent / tname / "wavfry_infer_cache.json"
    if c_path.exists():
        c_data = json.loads(c_path.read_text(encoding="utf-8"))
        print(f"   • {tname}: task={c_data.get('task')}, model_hash={c_data.get('model_hash')}, total_files={len(c_data.get('files', {}))}")
    else:
        print(f"   • {tname}: MISSING")

print("\n" + "=" * 70)
print("AUDIT SUMMARY: " + ("ALL PASSED PERFECTLY" if bad_shards == 0 and not unexpected_labels else "ISSUES FOUND"))
print("=" * 70)
